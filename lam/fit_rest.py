"""Fit a splat face's rest set to its portrait: the ARKit weights at rest that make the head look like the photo (#151).

    python lam/fit_rest.py --url http://127.0.0.1:8109 --face lam-s622 --portrait portrait.png \
        --model face_landmarker.task --out rest.json

LAM gets the outline of a face right but hoods the lids and thins the lips; the viewer's rest set can take some of
that back. Each candidate set is drawn by the viewer (record mode, life off, rest ``neutral``, so the candidate is
the whole expression) and scored against the portrait with MediaPipe: the eyes, lips and brows placed against the
face's outline and nose (aligned for position, rotation and scale, so the camera need not match exactly), and
MediaPipe's own ARKit scores for the lids, the smile and the brows. The search is fit_shape.py's, from ``warm``.

Writes ``{"name", "weights", "life", "score": {...}}``: a rest set for expressions.json. Needs playwright,
mediapipe, numpy and pillow; the face must be served by the server at ``--url``.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_shape  # noqa: E402

# Searched in left/right pairs (a face at rest is symmetric), from the built-in "warm" set.
PARAMS = {
    "eyeWide": 0.0,
    "eyeBlink": 0.10,
    "eyeSquint": 0.05,
    "browInnerUp": 0.06,
    "browOuterUp": 0.0,
    "mouthSmile": 0.32,
    "cheekSquint": 0.12,
    "mouthUpperUp": 0.0,
    "mouthLowerDown": 0.0,
    "mouthDimple": 0.0,
}
SIDED = {"eyeWide", "eyeBlink", "eyeSquint", "browOuterUp", "mouthSmile", "cheekSquint", "mouthUpperUp",
         "mouthLowerDown", "mouthDimple"}  # fmt: skip
# MediaPipe's ARKit scores compared: the lids, the smile, the brows and the lips.
SCORES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
          "mouthSmileLeft", "mouthSmileRight", "browInnerUp", "browDownLeft", "browDownRight", "mouthRollLower",
          "mouthRollUpper", "mouthPressLeft", "mouthPressRight", "mouthFrownLeft", "mouthFrownRight"]  # fmt: skip
ANCHOR = fit_shape.OVAL + fit_shape.NOSE  # the face's frame: what the features are placed against
FEATURES = {"eyes": (fit_shape.EYES, 2.0), "lips": (fit_shape.LIPS, 2.0), "brows": (fit_shape.BROWS, 1.0)}
SCORE_WEIGHT = 0.1  # MediaPipe's scores (0 to 1) against the landmarks (in eye spacings)


def _gap(p: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


# Ratios a face is read by, from MediaPipe's points: the lids' opening over the iris's width (hooded lids read
# older), and the lips' thickness over the mouth's width.
RATIOS = {
    "aperture": lambda p: (_gap(p, 159, 145) / _gap(p, 469, 471) + _gap(p, 386, 374) / _gap(p, 474, 476)) / 2,
    "lip_ratio": lambda p: (_gap(p, 0, 13) + _gap(p, 14, 17)) / _gap(p, 61, 291),
}
RATIO_WEIGHT = 0.3


MIN_SMILE = 0.15  # a face rests with a slight smile: it reads warm, where none can read as a scowl
# Shapes that part the lips: the viewer drops them while the face speaks (motion.js PARTS_LIPS), so they shape
# the lips at rest only. Kept small all the same: they come and go as speech starts and stops (#151).
OPENERS = {"mouthUpperUp": 0.25, "mouthLowerDown": 0.25}


def weights(vector: np.ndarray, min_smile: float = MIN_SMILE) -> dict[str, float]:
    """The rest set for a searched vector: each pair to both sides, within 0..1, the smile at least *min_smile*,
    the lips parted at most OPENERS."""
    floor = np.array([min_smile if name == "mouthSmile" else 0.0 for name in PARAMS])
    ceiling = np.array([OPENERS.get(name, 1.0) for name in PARAMS])
    out = {}
    for (name, _), value in zip(PARAMS.items(), np.clip(vector, floor, ceiling), strict=True):
        for key in [f"{name}Left", f"{name}Right"] if name in SIDED else [name]:
            out[key] = round(float(value), 3)
    return out


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama: the scale, rotation and translation that best take ``src`` onto ``dst`` (2D)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    flip = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ flip @ vt
    scale = float((sig * np.diag(flip)).sum() / (s**2).sum(1).mean())
    return scale, rot, mu_d - scale * rot @ mu_s


def misfit(ref: dict, cand: dict | None) -> dict[str, float]:
    """``ref``/``cand``: {"points": (478, 2) pixels, "scores": {name: 0..1}}. Lower is better."""
    if cand is None:
        return {"total": float("inf")}
    a, b = ref["points"], cand["points"]
    scale, rot, shift = similarity(b[ANCHOR], a[ANCHOR])
    b = (scale * (rot @ b.T)).T + shift
    unit = float(np.linalg.norm(a[fit_shape.EYE_CORNERS[0]] - a[fit_shape.EYE_CORNERS[1]]))
    out, total, weight = {}, 0.0, 0.0
    for name, (idx, w) in FEATURES.items():
        out[name] = float(np.sqrt(((a[idx] - b[idx]) ** 2).sum(1).mean())) / unit
        total, weight = total + w * out[name], weight + w
    out["scores"] = float(np.mean([abs(ref["scores"][k] - cand["scores"][k]) for k in SCORES]))
    out["total"] = total / weight + SCORE_WEIGHT * out["scores"]
    # What the eye reads first: how much of the iris the lids show, and how full the lips are. Relative errors.
    for name, measure in RATIOS.items():
        want, got = measure(ref["points"]), measure(cand["points"])
        out[name] = abs(got - want) / want
        out["total"] += RATIO_WEIGHT * out[name]
    return out


class Detector:
    def __init__(self, model: str):
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415

        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model), num_faces=1, output_face_blendshapes=True
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def __call__(self, rgb: np.ndarray) -> dict | None:
        found = self.landmarker.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        if not found.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in found.face_landmarks[0]])
        return {"points": points, "scores": {c.category_name: c.score for c in found.face_blendshapes[0]}}


async def fit(a: argparse.Namespace) -> dict:
    from PIL import Image  # noqa: PLC0415
    from playwright.async_api import async_playwright  # noqa: PLC0415

    detect = Detector(a.model)
    ref = detect(np.asarray(Image.open(a.portrait).convert("RGB")))
    if ref is None:
        raise SystemExit("MediaPipe found no face in the portrait")
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = await browser.new_page(viewport={"width": a.size, "height": a.size}, device_scale_factor=2)
        await page.goto(f"{a.url}/?face={a.face}&hud=0&record=1&life=0&rest=neutral&scene=none")
        await page.wait_for_function("window.__mask && (__mask.ready || __mask.error)", timeout=180_000)
        if error := await page.evaluate("window.__mask.error"):
            raise SystemExit(f"{a.face}: {error}")
        head = {"yaw": 0, "pitch": 0, "roll": 0}
        shots: dict[str, Image.Image] = {}

        async def draw(vector: np.ndarray) -> np.ndarray:
            w = weights(vector, a.min_smile)
            await page.evaluate("([w, h, n]) => window.__mask.pose(w, h, n, 0)", [list(w.values()), head, list(w)])
            return np.asarray(Image.open(io.BytesIO(await page.screenshot())).convert("RGB"))

        async def score_all(vectors: list[np.ndarray]) -> list[float]:
            return [misfit(ref, detect(await draw(v)))["total"] for v in vectors]

        # fit_shape.search takes a plain function: run each batch on this loop from a worker thread.
        loop = asyncio.get_running_loop()

        def score(vectors: list[np.ndarray]) -> list[float]:
            return asyncio.run_coroutine_threadsafe(score_all(vectors), loop).result()

        start = np.array(list(PARAMS.values()))
        best, _, history = await asyncio.to_thread(
            fit_shape.search, start, score, dims=len(start), sigma=0.12, population=8,
            generations=a.generations, seed=0,
        )  # fmt: skip
        report = {}
        for label, vector in (("warm", start), ("fit", best)):
            rgb = await draw(vector)
            shots[label] = Image.fromarray(rgb)
            report[label] = misfit(ref, detect(rgb))
        await browser.close()
    for label, image in shots.items():
        image.save(Path(a.out).with_suffix(f".{label}.png"))
    return {"name": a.name or f"fit-{a.face}", "weights": weights(best, a.min_smile), "life": 1.0, "score": report,
            "history": history}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8109")
    ap.add_argument("--face", required=True)
    ap.add_argument("--portrait", required=True)
    ap.add_argument("--model", required=True, help="MediaPipe's face_landmarker.task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", help="the rest set's name [fit-<face>]")
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--min-smile", type=float, default=MIN_SMILE, help="the least mouthSmile the set may have")
    ap.add_argument("--size", type=int, default=720)
    a = ap.parse_args()
    result = asyncio.run(fit(a))
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("name", "weights", "score")}, indent=1))


if __name__ == "__main__":
    main()
