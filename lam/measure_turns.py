"""Measure how far a LAM head turns before the hairline or temples give out (#170).

    python lam/measure_turns.py --url http://127.0.0.1:8109 --face splat-01 --out turns.json

Draws the head in the viewer (record mode, limits off), scores each pose for a hairline/temple gap, and writes
the measured ``limits`` plus per-pose scores. Needs playwright and pillow. The guided editor runs this as a
process (``editor/turns.py``); nothing here is imported into the package.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from argus_mask.turns import PITCH_UP_PROBE, POSES, YAW_PROBE, PoseScore, gap_score, limits_from_scores, open_rgb

POSE_JS = "([w, h, n]) => window.__mask.pose(w, h, n, 0)"


async def run(url: str, face: str, out: Path, size: int, threshold: float) -> None:
    from playwright.async_api import async_playwright  # noqa: PLC0415

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(args=["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
        try:
            page = await browser.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
            await page.goto(f"{url.rstrip('/')}/?face={face}&hud=0&record=1&life=0&turn=0&jaw=1")
            await page.wait_for_function("window.__mask && (__mask.ready || __mask.error)", timeout=120_000)
            if error := await page.evaluate("window.__mask.error"):
                raise SystemExit(f"{face}: {error}")

            async def shot(yaw: float, pitch: float) -> bytes:
                await page.evaluate(POSE_JS, [[0, 0, 0], {"yaw": yaw, "pitch": pitch, "roll": 0}, ["jawOpen"]])
                return await page.screenshot(type="png")

            front = open_rgb(await shot(0, 0))
            scores: list[PoseScore] = [PoseScore(label="front", yaw=0.0, pitch=0.0, gap=0.0, region="front")]
            sheet: dict[str, str] = {}  # label -> kept relative name for the editor

            async def add(label: str, yaw: float, pitch: float) -> None:
                image = open_rgb(await shot(yaw, pitch))
                found = gap_score(front, image, yaw=yaw, pitch=pitch)
                scores.append(
                    PoseScore(label=label, yaw=yaw, pitch=pitch, gap=found.gap, region=found.region)
                )
                sheet[label] = label

            for label, turn in POSES:
                if label == "front":
                    continue
                await add(label, float(turn.get("yaw", 0)), float(turn.get("pitch", 0)))
            for yaw in YAW_PROBE:
                for sign, tag in ((-1, "m"), (1, "p")):
                    label = f"probe_yaw_{tag}{yaw}"
                    if any(s.label == label for s in scores):
                        continue
                    await add(label, float(sign * yaw), 0.0)
            for pitch in PITCH_UP_PROBE:
                label = f"probe_back_{pitch}"
                if any(s.label == label for s in scores):
                    continue
                await add(label, 0.0, float(-pitch))

            limits = limits_from_scores(scores, threshold=threshold)
            data = {
                "threshold": threshold,
                "limits": limits.model_dump(mode="json", exclude_none=True),
                "poses": [
                    {
                        "label": s.label,
                        "yaw": s.yaw,
                        "pitch": s.pitch,
                        "gap": round(s.gap, 4),
                        "region": s.region,
                        "failed": s.gap >= threshold,
                    }
                    for s in scores
                ],
            }
            out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(data["limits"]))
        finally:
            await browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--face", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--threshold", type=float, default=0.18)
    a = ap.parse_args()
    asyncio.run(run(a.url, a.face, a.out, a.size, a.threshold))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
