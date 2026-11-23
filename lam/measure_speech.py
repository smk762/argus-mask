"""Measure a head's mouth in speech against a fixed take (#171).

    python lam/measure_speech.py --url http://127.0.0.1:8109 --face splat-01 \\
        --take src/argus_mask/editor/page/talk.json --model face_landmarker.task --out speech.json

Draws frames of the take in record mode, reads the lip gap with MediaPipe, and writes
``closed_share``, ``reversals``, ``opening_p95``. Needs playwright and mediapipe. The guided editor
runs this as a process (``editor/speech.py``); nothing here is imported into the package.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import numpy as np

from argus_mask.speech_eval import CLOSED_GAP, metrics_from_gaps

POSE_JS = "([w, h, n, t]) => window.__mask.pose(w, h, n, t)"
UPPER, LOWER = 13, 14
LEFT_EYE, RIGHT_EYE = 33, 263


def lip_gap(points: np.ndarray) -> float:
    """Vertical lip opening in eye-spacings."""
    span = float(np.linalg.norm(points[LEFT_EYE] - points[RIGHT_EYE]))
    if span < 1e-6:
        return 0.0
    return float(np.linalg.norm(points[UPPER] - points[LOWER]) / span)


class Detector:
    def __init__(self, model: str) -> None:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415

        self.mp = mp
        options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_faces=1)
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        found = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not found.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        return np.array([(p.x * w, p.y * h) for p in found.face_landmarks[0]])


async def run(url: str, face: str, take: Path, model: Path, out: Path, size: int, stride: int) -> None:
    from PIL import Image  # noqa: PLC0415
    from playwright.async_api import async_playwright  # noqa: PLC0415

    raw = json.loads(take.read_text(encoding="utf-8"))
    names = list(raw["names"])
    frames = list(raw["frames"])
    fps = int(raw.get("fps") or 30)
    detect = Detector(str(model))

    gaps: list[float] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
        try:
            page = await browser.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
            await page.goto(f"{url.rstrip('/')}/?face={face}&hud=0&record=1&life=0&turn=0&jaw=1")
            await page.wait_for_function("window.__mask && (__mask.ready || __mask.error)", timeout=120_000)
            if error := await page.evaluate("window.__mask.error"):
                raise SystemExit(f"{face}: {error}")

            for i, weights in enumerate(frames):
                if i % stride != 0:
                    continue
                t_ms = int(round(1000 * i / fps))
                await page.evaluate(
                    POSE_JS,
                    [list(weights), {"yaw": 0, "pitch": 0, "roll": 0}, names if i == 0 else None, t_ms],
                )
                png = await page.screenshot(type="png")
                rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
                points = detect(rgb)
                if points is None:
                    continue
                gaps.append(lip_gap(points))
        finally:
            await browser.close()

    if len(gaps) < 10:
        raise SystemExit(f"only {len(gaps)} frames had a face; need at least 10")
    metrics = metrics_from_gaps(gaps, closed_below=CLOSED_GAP)
    data = {
        "face": face,
        "take": take.name,
        "stride": stride,
        "threshold_closed": CLOSED_GAP,
        **metrics.as_data(),
        "gaps": [round(g, 5) for g in gaps],
    }
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics.as_data()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--take", type=Path, required=True, help="talk.json (names + frames)")
    ap.add_argument("--model", type=Path, required=True, help="MediaPipe face_landmarker.task")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame (default 2)")
    a = ap.parse_args()
    asyncio.run(run(a.url, a.face, a.take, a.model, a.out, a.size, max(1, a.stride)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
