"""How like its portrait a head looks, as numbers (#169): colour, identity, and the lids and lips.

    python lam/likeness.py --portrait portrait.png --head render.png --out likeness.json [--dinov2 <dir>]

Two pictures on white: the portrait as LAM was given it (the tracker's crop, matted) and the head as the
portrait's own camera sees it (``make_head.py``'s ``likeness/``). MediaPipe's landmarks place the face in each
(``landmarks/detect.py``).

- **Colour**: the skin's, lips', irises' and hair's colour, compared as CIELAB distances.
- **Identity** (``--dinov2``): the cosine between the two faces' DINOv2 embeddings (Apache-2.0; what argus-proof
  scores likeness with). Face-recognition weights are not used: many are research-only.
- **Lids and lips**: how much of the iris the lids show and how full the lips are (``fit_rest.RATIOS``), as
  relative errors, plus the landmarks' misfit at the portrait's camera (``fit_shape.misfit``).

Calibrated on #151's heads (character vs photo). Colour and lip_ratio rank character above photo; identity does
when scored at the portrait's camera; lid aperture is weaker but kept as a measure. Detail (fine structure) did
not separate the styles and is not written to the record. The scores flag and rank; none refuses a head.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_rest  # noqa: E402
import fit_shape  # noqa: E402

# MediaPipe Face Mesh rings, each in order round its outline.
OVAL = fit_shape.OVAL
LIPS_OUTER = fit_shape.LIPS
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_RINGS = (fit_shape.EYES[:16], fit_shape.EYES[16:])
IRISES = ((468, [469, 470, 471, 472]), (473, [474, 475, 476, 477]))  # each: its centre, then its rim
LOWER_LIDS = [145, 153, 144, 374, 380, 373]
BROW_TOPS = [105, 334]

EYE_PX = 128  # the outer eye corners' spacing in the frame the two pictures are compared in
FRAME = 3 * EYE_PX  # the frame's side: the face and the hair round it
FINE = (1.0, 3.0)  # the band (two blurs' widths, in frame pixels) fine structure lives in: lashes, strands, pores
FORM = (3.0, 9.0)  # and the features' own edges and shading
WHITE = 0.94  # a pixel this bright in every channel is the backdrop, not hair
MIN_PIXELS = 40  # a region smaller than this has no colour to speak of


# ---- pictures -------------------------------------------------------------------------------------------


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of a (h, w) or (h, w, c) picture, its edges repeated."""
    if sigma <= 0:
        return image.astype(np.float64)
    radius = max(1, int(3 * sigma + 0.5))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    out = image.astype(np.float64)
    for axis in (0, 1):
        pad = [(radius, radius) if a == axis else (0, 0) for a in range(out.ndim)]
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(out, pad, mode="edge"), len(kernel), axis=axis)
        out = windows @ kernel
    return out


def placed(image: np.ndarray, scale: float, rot: np.ndarray, shift: np.ndarray, size: int = FRAME) -> np.ndarray:
    """*image* (h, w, 3 in 0..1) redrawn in a *size* frame, where a point p of it lands at ``scale * rot @ p +
    shift`` (x, y). Outside the picture is white. Made smaller, it is blurred first, so no detail aliases."""
    if scale < 1:
        image = blur(image, 0.5 * (1 / scale - 1))
    grid = np.stack(np.meshgrid(np.arange(size), np.arange(size)), axis=-1).reshape(-1, 2).astype(np.float64)
    source = (grid - shift) @ rot / scale  # rot is orthonormal: (R^-1 q) as a row is q @ R
    h, w = image.shape[:2]
    x, y = source[:, 0], source[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    outside = (x < 0) | (y < 0) | (x > w - 1) | (y > h - 1)
    x0, y0 = np.clip(x0, 0, w - 2), np.clip(y0, 0, h - 2)
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    out = top * (1 - fy) + bottom * fy
    out[outside] = 1.0
    return out.reshape(size, size, -1)


def framing(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The scale, rotation and shift that put a face (478 points in pixels) in the frame: its outer eye corners
    level and EYE_PX apart, the point between them a little above the middle."""
    left, right = points[fit_shape.EYE_CORNERS[0]], points[fit_shape.EYE_CORNERS[1]]
    across = right - left
    scale = EYE_PX / float(np.linalg.norm(across))
    angle = -np.arctan2(across[1], across[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([FRAME / 2, FRAME * 0.45]) - scale * rot @ ((left + right) / 2)
    return scale, rot, shift


def move(points: np.ndarray, scale: float, rot: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return scale * points[:, :2] @ rot.T + shift


def inside(ring: np.ndarray, size: int = FRAME) -> np.ndarray:
    """The pixels of a *size* frame inside a closed ring of (x, y) points: the even-odd rule."""
    ys, xs = np.mgrid[0:size, 0:size]
    xs, ys = xs + 0.5, ys + 0.5
    mask = np.zeros((size, size), dtype=bool)
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0), strict=True):
        if y1 == y2:
            continue
        crosses = ((y1 <= ys) & (ys < y2)) | ((y2 <= ys) & (ys < y1))
        mask ^= crosses & (xs < x1 + (ys - y1) * (x2 - x1) / (y2 - y1))
    return mask


def grown(ring: np.ndarray, by: float) -> np.ndarray:
    centre = ring.mean(axis=0)
    return centre + by * (ring - centre)


def regions(points: np.ndarray, image: np.ndarray) -> dict[str, np.ndarray]:
    """Where to read each colour in a framed picture, from its landmarks (frame pixels).

    ``skin``: the cheeks, nose and chin: the outline below the lower lids, less the lips. Above the lids a
    fringe covers the forehead in half the portraits. ``lips``: between their outer and inner outlines.
    ``iris``: each iris's disc where the lids leave it open, less the pupil. ``hair``: a band round the top of
    the outline, less the backdrop. ``face``: the whole outline.
    """
    size = image.shape[0]
    ys, xs = np.mgrid[0:size, 0:size]
    face = inside(points[OVAL], size)
    lips = inside(points[LIPS_OUTER], size)
    below_lids = ys > points[LOWER_LIDS, 1].max() + 0.04 * EYE_PX
    iris = np.zeros_like(face)
    for (centre, rim), ring in zip(IRISES, EYE_RINGS, strict=True):
        radius = float(np.linalg.norm(points[rim] - points[centre], axis=1).mean())
        away = np.hypot(xs + 0.5 - points[centre, 0], ys + 0.5 - points[centre, 1])
        iris |= (away < 0.9 * radius) & (away > 0.45 * radius) & inside(points[ring], size)
    backdrop = (image > WHITE).all(axis=-1)
    above_brows = ys < points[BROW_TOPS, 1].min()
    hair = inside(grown(points[OVAL], 1.3), size) & ~inside(grown(points[OVAL], 1.05), size) & above_brows & ~backdrop
    return {
        "skin": face & below_lids & ~inside(grown(points[LIPS_OUTER], 1.25), size),
        "lips": lips & ~inside(points[LIPS_INNER], size),
        "iris": iris,
        "hair": hair,
        "face": face,
    }


# ---- colour ---------------------------------------------------------------------------------------------


def lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in 0..1 to CIELAB (D65)."""
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = linear @ np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]]).T  # fmt: skip
    f = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(f > 216 / 24389, np.cbrt(f), (24389 / 27 * f + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def colour(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """A region's colour: the median of its pixels in CIELAB (a catch-light or a stray hair does not move it)."""
    if int(mask.sum()) < MIN_PIXELS:
        return None
    return np.median(lab(image[mask]), axis=0)


def colours(portrait: np.ndarray, portrait_at: dict, head: np.ndarray, head_at: dict) -> dict[str, float]:
    """Each region's colour distance (CIE76: about 2 is just seen, 10 is plainly another colour), and their mean.
    A region one picture does not show (a bald head, closed eyes) has no score."""
    out = {}
    for name in ("skin", "lips", "iris", "hair"):
        a, b = colour(portrait, portrait_at[name]), colour(head, head_at[name])
        if a is not None and b is not None:
            out[name] = float(np.linalg.norm(a - b))
    if out:
        out["mean"] = float(np.mean(list(out.values())))
    return out


# ---- detail ---------------------------------------------------------------------------------------------


def band(image: np.ndarray, widths: tuple[float, float]) -> np.ndarray:
    """What of a picture's lightness lies between two blurs: structure of that size, and nothing flatter."""
    light = lab(image)[..., 0] / 100
    return blur(light, widths[0]) - blur(light, widths[1])


def detail(portrait: np.ndarray, head: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Inside *mask* (the face in both): ``kept``, the head's fine structure over the portrait's, at most 1;
    ``same``, how far the two pictures' features coincide (their correlation: 1 the same, 0 unrelated)."""
    out = {}
    fine_p, fine_h = band(portrait, FINE)[mask], band(head, FINE)[mask]
    energy = float(np.abs(fine_p).mean())
    out["kept"] = min(1.0, float(np.abs(fine_h).mean()) / energy) if energy > 0 else 1.0
    form_p, form_h = band(portrait, FORM)[mask], band(head, FORM)[mask]
    form_p, form_h = form_p - form_p.mean(), form_h - form_h.mean()
    norm = float(np.sqrt((form_p**2).sum() * (form_h**2).sum()))
    out["same"] = float((form_p * form_h).sum()) / norm if norm > 0 else 0.0
    return out


# ---- the whole score ------------------------------------------------------------------------------------


def geometry(portrait_points: np.ndarray, head_points: np.ndarray) -> dict[str, float]:
    """The landmarks' misfit once the head is laid over the portrait (in eye spacings), and the lids' opening and
    the lips' fullness as relative errors. Points in pixels of any one frame each."""
    scale, rot, shift = fit_rest.similarity(head_points[fit_rest.ANCHOR], portrait_points[fit_rest.ANCHOR])
    out = fit_shape.misfit(portrait_points, move(head_points, scale, rot, shift))
    for name, measure in fit_rest.RATIOS.items():
        want, got = measure(portrait_points), measure(head_points)
        out[name] = (got - want) / want
    return out


def compare(portrait: np.ndarray, portrait_points: np.ndarray, head: np.ndarray, head_points: np.ndarray) -> dict:
    """Everything but identity. Pictures (h, w, 3) in 0..1; points (478, 2+) in each picture's own pixels.
    ``frames`` are the two pictures as they were compared, for whoever wants to look or embed them."""
    at = framing(portrait_points)
    portrait_framed = move(portrait_points, *at)
    # The head's picture goes where its outline and nose fall on the portrait's.
    scale, rot, shift = fit_rest.similarity(head_points[fit_rest.ANCHOR, :2], portrait_framed[fit_rest.ANCHOR])
    head_framed = move(head_points, scale, rot, shift)
    frames = {"portrait": placed(portrait, *at), "head": placed(head, scale, rot, shift)}
    portrait_at, head_at = regions(portrait_framed, frames["portrait"]), regions(head_framed, frames["head"])
    return {
        "geometry": geometry(portrait_framed, head_framed),
        "colour": colours(frames["portrait"], portrait_at, frames["head"], head_at),
        "detail": detail(frames["portrait"], frames["head"], portrait_at["face"] & head_at["face"]),
        "frames": frames,
    }



# Scores that go in the build record (#169): calibrated to rank character heads above photo heads. Detail is
# computed for anyone looking, and left out of the record.
RECORD_COLOUR = ("skin", "lips", "iris", "hair", "mean")
RECORD_GEOMETRY = ("aperture", "lip_ratio", "total")  # total: misfit at the portrait's camera


def scores_for_record(result: dict) -> dict[str, float]:
    """The flat scores a likeness step keeps: colour, identity, lids, lips, and the camera misfit."""
    out: dict[str, float] = {}
    colour = result.get("colour") or {}
    for name in RECORD_COLOUR:
        if name in colour and isinstance(colour[name], (int, float)):
            out[f"colour_{name}"] = float(colour[name])
    geom = result.get("shape") or result.get("geometry") or {}
    for name in RECORD_GEOMETRY:
        if name in geom and isinstance(geom[name], (int, float)):
            key = "shape_misfit" if name == "total" else name
            out[key] = float(geom[name])
    if isinstance(result.get("identity"), (int, float)):
        out["identity"] = float(result["identity"])
    return out


# ---- what needs more than numpy --------------------------------------------------------------------------


def detect(paths: list[str]) -> np.ndarray:
    """MediaPipe's 478 points for each picture, x and y in 0..1 (NaN where it found no face): detect.py, run in
    the image's MediaPipe environment."""
    python = os.environ.get("ARGUS_MEDIAPIPE_PYTHON", "/opt/mediapipe/bin/python")
    model = Path(os.environ.get("ARGUS_LAM_MODELS", "/opt/argus-models")) / "face_landmarker.task"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "points.npz"
        detector = Path(__file__).resolve().parent / "landmarks" / "detect.py"
        subprocess.run([python, str(detector), str(model), str(out), *paths], check=True, capture_output=True)  # noqa: S603
        return np.load(out)["points"]


def identity(frames: dict[str, np.ndarray], model_dir: str) -> float:
    """The cosine between the two framed faces' DINOv2 embeddings (the layer-normed CLS token, as argus-proof's
    likeness scorer takes it). On the CPU: two pictures."""
    import torch  # noqa: PLC0415
    from transformers import AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    side = 224
    shrink = side / FRAME
    batch = np.stack([placed(f, shrink, np.eye(2), np.zeros(2), side) for f in (frames["portrait"], frames["head"])])
    pixels = torch.as_tensor(((batch - mean) / std).transpose(0, 3, 1, 2), dtype=torch.float32)
    with torch.inference_mode():
        a, b = torch.nn.functional.normalize(model(pixel_values=pixels).pooler_output.float(), dim=1)
    return float(a @ b)


def main() -> None:
    from PIL import Image  # noqa: PLC0415

    ap = argparse.ArgumentParser()
    ap.add_argument("--portrait", required=True, help="the portrait as LAM was given it, on white")
    ap.add_argument("--head", required=True, help="the head's picture from the front, on white")
    ap.add_argument("--render", help="LAM's render of the head at the portrait's own camera (make_head.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dinov2", help="a DINOv2 checkpoint's folder, for the identity score")
    ap.add_argument("--frames", help="a folder to write the two pictures as they were compared")
    a = ap.parse_args()
    pictures = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255 for p in (a.portrait, a.head)]
    found = detect([a.portrait, a.head, *([a.render] if a.render else [])])
    for name, points in zip(("portrait", "head"), found, strict=False):
        if np.isnan(points[0, 0]):
            Path(a.out).write_text(json.dumps({"error": f"MediaPipe found no face in the {name}"}))
            sys.exit(f"MediaPipe found no face in the {name}")
    points = [p[:, :2] * (im.shape[1], im.shape[0]) for p, im in zip(found, pictures, strict=False)]
    result = compare(pictures[0], points[0], pictures[1], points[1])
    frames = result.pop("frames")
    # The head at the portrait's camera shares the portrait's frame: score geometry as the shape fit does.
    at_camera = bool(a.render) and len(found) > 2 and not np.isnan(found[2][0, 0])
    cam = found[2] if at_camera else found[1]
    cam_px = (
        cam[:, :2] * (pictures[0].shape[1], pictures[0].shape[0]) if at_camera else points[1]
    )
    shape = fit_shape.misfit(found[0], cam)
    for name, measure in fit_rest.RATIOS.items():
        want, got = measure(points[0]), measure(cam_px)
        shape[name] = (got - want) / want
    result["shape"] = shape
    if a.dinov2:
        result["identity"] = identity(frames, a.dinov2)
    result["scores"] = scores_for_record(result)
    if a.frames:
        Path(a.frames).mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            Image.fromarray((frame * 255).round().astype(np.uint8)).save(Path(a.frames) / f"{name}.png")
    # Detail did not rank character above photo on #151's heads: leave it out of what the record keeps.
    Path(a.out).write_text(json.dumps({k: v for k, v in result.items() if k != "detail"}, indent=1))
    print("LIKENESS", json.dumps(result["scores"]))


if __name__ == "__main__":
    main()
