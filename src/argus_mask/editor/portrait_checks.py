"""Mask's own portrait checks and the mapping from curator / OCR (#168).

Each function returns a :class:`~argus_mask.editor.guided.Check`. Sibling services own face_source and OCR;
mask adds framing, expression, lighting, temples and the photograph advisory.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from argus_mask.editor.curator import CURATOR_CHECK_NAMES, FACE_SOURCE_CHECKS, FaceSourceRow
from argus_mask.editor.guided import Check

# MediaPipe Face Landmarker blendshape names we care about.
_SMILE = ("mouthSmileLeft", "mouthSmileRight")
_MOUTH_OPEN = ("jawOpen", "mouthLowerDownLeft", "mouthLowerDownRight")


@dataclass(frozen=True)
class FaceGeometry:
    """What mask needs of a face in the portrait, from MediaPipe or a test double."""

    face_count: int
    # Normalised landmarks (x, y) in 0..1 when known; else empty.
    landmarks: tuple[tuple[float, float], ...] = ()
    blendshapes: dict[str, float] | None = None
    # Axis-aligned face box as fractions of the image (left, top, right, bottom).
    box: tuple[float, float, float, float] | None = None


def open_rgb(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def check_framing(image: Image.Image, *, min_side_px: int, geometry: FaceGeometry | None = None) -> Check:
    """Head-and-shoulders / resolution: refuse a portrait that is too small."""
    w, h = image.size
    short = min(w, h)
    measure = {"width": float(w), "height": float(h), "short_side": float(short)}
    if short < min_side_px:
        return Check(
            name="framing",
            verdict="refused",
            reason=f"portrait is {w}×{h}: need at least {min_side_px} px on the short side",
            measure=measure,
        )
    if geometry is not None and geometry.box is not None:
        left, top, right, bottom = geometry.box
        face_short = min(right - left, bottom - top) * short
        measure["face_short_px"] = float(face_short)
        if face_short < min_side_px * 0.35:
            return Check(
                name="framing",
                verdict="refused",
                reason=f"the face is about {face_short:.0f} px: too small in the frame",
                measure=measure,
            )
    return Check(name="framing", verdict="passed", measure=measure)


def check_expression(geometry: FaceGeometry | None) -> Check:
    """Neutral or a slight smile, mouth closed. Advisory when the mouth is open or the smile is strong."""
    if geometry is None or geometry.blendshapes is None:
        return Check(
            name="expression",
            verdict="not_checked",
            reason="no face landmarker: expression was not scored",
            required=False,
        )
    shapes = geometry.blendshapes
    smile = max((shapes.get(n, 0.0) for n in _SMILE), default=0.0)
    open_m = max((shapes.get(n, 0.0) for n in _MOUTH_OPEN), default=0.0)
    measure = {"smile": float(smile), "mouth_open": float(open_m)}
    if open_m >= 0.25:
        return Check(
            name="expression",
            verdict="advisory",
            reason=f"mouth looks open (score {open_m:.2f}): prefer lips closed",
            measure=measure,
        )
    if smile >= 0.45:
        return Check(
            name="expression",
            verdict="advisory",
            reason=f"smile looks strong (score {smile:.2f}): a slight smile is fine, a wide one is baked in",
            measure=measure,
        )
    return Check(name="expression", verdict="passed", measure=measure)


def check_lighting(image: Image.Image, geometry: FaceGeometry | None = None) -> Check:
    """Even light: refuse nothing; advise when left and right cheeks differ a lot."""
    w, h = image.size
    if geometry is not None and geometry.box is not None:
        left, top, right, bottom = geometry.box
        mid_x = (left + right) / 2
        mid_y = (top + bottom) / 2
        cheek_y0, cheek_y1 = mid_y, bottom
        left_box = (left, cheek_y0, mid_x, cheek_y1)
        right_box = (mid_x, cheek_y0, right, cheek_y1)
    else:
        # Upper-middle thirds without landmarks: a coarse stand-in.
        left_box = (0.2, 0.35, 0.45, 0.6)
        right_box = (0.55, 0.35, 0.8, 0.6)
    left_l = _mean_luma(image, left_box)
    right_l = _mean_luma(image, right_box)
    delta = abs(left_l - right_l)
    measure = {"left_luma": left_l, "right_luma": right_l, "delta": delta}
    if delta >= 35.0:
        return Check(
            name="lighting",
            verdict="advisory",
            reason=f"light differs side to side (ΔL {delta:.0f}): prefer flat front light",
            measure=measure,
        )
    return Check(name="lighting", verdict="passed", measure=measure)


def check_temples(image: Image.Image, geometry: FaceGeometry | None = None) -> Check:
    """Hair should cover the temples and hairline; a bare temple gaps when the head turns (#151)."""
    regions = _temple_regions(geometry)
    scores = []
    for box in regions:
        skin = _skin_frac(image, box)
        scores.append(skin)
    bare = max(scores) if scores else 0.0
    measure = {"temple_skin_frac": float(bare), **{f"temple_{i}": float(s) for i, s in enumerate(scores)}}
    if bare >= 0.55:
        return Check(
            name="temples",
            verdict="advisory",
            reason="temples look bare: hair should cover the hairline and temples for a LAM head",
            measure=measure,
        )
    return Check(name="temples", verdict="passed", measure=measure)


def check_photograph(image: Image.Image, source_data: dict[str, Any], *, photo_laplacian: float) -> Check:
    """A photograph comes out of LAM about ten years older; character art survives (#151). Advisory."""
    style = str(source_data.get("style") or "")
    way = str(source_data.get("way") or "")
    lap = _laplacian_var(image)
    measure = {"laplacian": float(lap)}
    if way == "generate" and style in ("character", "stylised", "semireal"):
        return Check(name="photograph", verdict="passed", measure={**measure, "character_style": 1.0})
    # Generate with photo style, or an upload/pick that measures like a photo.
    if (way == "generate" and style == "photo") or (way != "generate" and lap >= photo_laplacian):
        return Check(
            name="photograph",
            verdict="advisory",
            reason=(
                "this looks like a photograph: LAM softens it and the head reads older; "
                "restyle into character art (#167) or overrule"
            ),
            measure=measure,
        )
    return Check(name="photograph", verdict="passed", measure=measure)


def check_text(ocr_lines: list[str] | None, *, down: str | None = None) -> Check:
    """No text or signature in the picture (FLUX paints one sometimes). Required when OCR is configured."""
    if down is not None:
        return Check(name="text", verdict="not_checked", reason=down, required=True)
    if ocr_lines is None:
        return Check(name="text", verdict="not_checked", reason="OCR was not run", required=True)
    joined = " ".join(ocr_lines).strip()
    if joined:
        sample = joined[:120]
        return Check(
            name="text",
            verdict="advisory",
            reason=f"OCR read text in the picture ({sample!r}): reject a signature or regenerate",
            measure={"lines": float(len(ocr_lines))},
        )
    return Check(name="text", verdict="passed", measure={"lines": 0.0})


def checks_from_face_source(row: FaceSourceRow | None, *, down: str | None = None) -> list[Check]:
    """Curator's face_source outcome as portrait checks. *down* makes every curator check ``not_checked``."""
    if down is not None:
        return [Check(name=name, verdict="not_checked", reason=down, required=True) for name in CURATOR_CHECK_NAMES]
    if row is None:
        return [
            Check(name=name, verdict="not_checked", reason="curator returned no row", required=True)
            for name in CURATOR_CHECK_NAMES
        ]
    reason = row.keep_reason.strip()
    if not reason:
        return [Check(name=name, verdict="passed") for name in CURATOR_CHECK_NAMES]
    mapped = FACE_SOURCE_CHECKS.get(reason)
    if mapped is None:
        # Unknown exclusion: treat as a required refusal so a new curator reason cannot pass silently.
        failed = Check(name="identity", verdict="refused", reason=reason)
        others = [Check(name=n, verdict="passed") for n in CURATOR_CHECK_NAMES if n != "identity"]
        return [failed, *others]
    check_name, verdict = mapped
    out: list[Check] = []
    for name in CURATOR_CHECK_NAMES:
        if name == check_name:
            out.append(Check(name=name, verdict=verdict, reason=reason))  # type: ignore[arg-type]
        else:
            out.append(Check(name=name, verdict="passed"))
    return out


def rank_score(checks: list[Check]) -> float:
    """Higher is better: refusals sink, advisories cost, passes win. Used to order a batch."""
    score = 100.0
    for check in checks:
        if check.verdict == "refused":
            score -= 100.0
        elif check.verdict == "failed":
            score -= 80.0
        elif check.verdict == "not_checked" and check.required:
            score -= 60.0
        elif check.verdict == "advisory":
            score -= 5.0
        elif check.verdict == "not_checked":
            score -= 1.0
    return score


def measure_face(data: bytes, landmarker: Any | None) -> FaceGeometry | None:
    """Run MediaPipe Face Landmarker when *landmarker* is a path to the task file; else None."""
    if landmarker is None:
        return None
    try:
        import mediapipe as mp  # noqa: PLC0415
        from mediapipe.tasks.python import BaseOptions, vision  # noqa: PLC0415
    except ImportError:
        return None
    path = str(landmarker)
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            num_faces=3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        rgb = open_rgb(data)
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(rgb)
        found = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    except Exception:  # noqa: BLE001 - landmarker missing or broken: treat as unavailable
        return None
    faces = found.face_landmarks or []
    if not faces:
        return FaceGeometry(face_count=0)
    primary = faces[0]
    landmarks = tuple((float(p.x), float(p.y)) for p in primary)
    xs = [p.x for p in primary]
    ys = [p.y for p in primary]
    box = (min(xs), min(ys), max(xs), max(ys))
    shapes: dict[str, float] = {}
    if found.face_blendshapes:
        for cat in found.face_blendshapes[0]:
            shapes[cat.category_name] = float(cat.score)
    return FaceGeometry(face_count=len(faces), landmarks=landmarks, blendshapes=shapes or None, box=box)


# ---- image helpers --------------------------------------------------------------------------------


def _crop_frac(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, int(left * w)),
            max(0, int(top * h)),
            min(w, int(right * w)),
            min(h, int(bottom * h)),
        )
    )


def _mean_luma(image: Image.Image, box: tuple[float, float, float, float]) -> float:
    patch = _crop_frac(image, box).convert("L")
    if patch.width < 1 or patch.height < 1:
        return 0.0
    return float(ImageStat.Stat(patch).mean[0])


def _skin_frac(image: Image.Image, box: tuple[float, float, float, float]) -> float:
    """Share of pixels that look like bare skin (high L, moderate chroma) in *box*."""
    patch = _crop_frac(image, box)
    if patch.width * patch.height == 0:
        return 0.0
    ycbcr = patch.convert("YCbCr")
    pixels = list(ycbcr.getdata())
    skin = 0
    for y, cb, cr in pixels:
        # Rough skin cluster in YCbCr (works for synthetic fixtures too).
        if 80 <= y <= 240 and 85 <= cb <= 135 and 130 <= cr <= 180:
            skin += 1
    return skin / len(pixels)


def _temple_regions(geometry: FaceGeometry | None) -> list[tuple[float, float, float, float]]:
    if geometry is not None and geometry.box is not None:
        left, top, right, bottom = geometry.box
        mid_y = top + (bottom - top) * 0.35
        h = (bottom - top) * 0.2
        w = (right - left) * 0.18
        return [
            (left - w * 0.2, mid_y - h / 2, left + w, mid_y + h / 2),
            (right - w, mid_y - h / 2, right + w * 0.2, mid_y + h / 2),
        ]
    # Default: upper corners where temples sit on a head-and-shoulders crop.
    return [(0.08, 0.18, 0.22, 0.38), (0.78, 0.18, 0.92, 0.38)]


def _laplacian_var(image: Image.Image) -> float:
    gray = image.convert("L").filter(ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1))
    stat = ImageStat.Stat(gray)
    # Pillow's sum2/count gives variance via E[x²] - E[x]².
    if not stat.count or stat.count[0] == 0:
        return 0.0
    mean = stat.mean[0]
    sum2 = stat.sum2[0]
    n = stat.count[0]
    return max(0.0, sum2 / n - mean * mean)
