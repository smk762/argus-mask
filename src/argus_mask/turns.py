"""Turn checks for a LAM head (#170): hairline gap and temple smear as the head turns and tips.

Each pose is a still of the head (limits off). The gap score is how much more bare skin shows at the
outer temple (yaw) or the top hairline (pitch up) than in the front pose. The measured limits are the
largest angles that still pass — what the manifest's ``limits`` should hold.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageStat

from argus_mask.models import Limits

# Contact-sheet poses the guided turns step keeps (#170 / lam/sheet.py), without the jaw opens.
POSES: tuple[tuple[str, dict[str, float]], ...] = (
    ("front", {}),
    ("yaw_-12", {"yaw": -12}),
    ("yaw_-20", {"yaw": -20}),
    ("yaw_12", {"yaw": 12}),
    ("yaw_20", {"yaw": 20}),
    ("back_8", {"pitch": -8}),
    ("forward_15", {"pitch": 15}),
)

# Finer steps used when a renderer can draw arbitrary angles (measure_limits).
YAW_PROBE = (8, 10, 12, 14, 16, 18, 20)
PITCH_UP_PROBE = (5, 6, 7, 8, 10, 12)

# Skin-share rise over the front pose that counts as a gap / smear.
DEFAULT_GAP = 0.18


@dataclass(frozen=True)
class PoseScore:
    label: str
    yaw: float
    pitch: float
    gap: float  # skin rise vs front; higher = worse
    region: str  # temple_left | temple_right | hairline


@dataclass(frozen=True)
class TurnReport:
    """What the turns step found: per-pose scores and the limits to keep."""

    scores: tuple[PoseScore, ...]
    limits: Limits
    threshold: float

    def as_data(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "limits": self.limits.model_dump(mode="json", exclude_none=True),
            "poses": [
                {
                    "label": s.label,
                    "yaw": s.yaw,
                    "pitch": s.pitch,
                    "gap": round(s.gap, 4),
                    "region": s.region,
                    "failed": s.gap >= self.threshold,
                }
                for s in self.scores
            ],
        }


def open_rgb(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def skin_frac(image: Image.Image, box: tuple[float, float, float, float]) -> float:
    """Share of YCbCr skin-like pixels in a fractional box."""
    w, h = image.size
    left, top, right, bottom = box
    patch = image.crop(
        (
            max(0, int(left * w)),
            max(0, int(top * h)),
            min(w, max(1, int(right * w))),
            min(h, max(1, int(bottom * h))),
        )
    )
    if patch.width * patch.height == 0:
        return 0.0
    pixels = list(patch.convert("YCbCr").getdata())
    skin = sum(1 for y, cb, cr in pixels if 80 <= y <= 240 and 85 <= cb <= 135 and 130 <= cr <= 180)
    return skin / len(pixels)


def temple_box(yaw: float) -> tuple[tuple[float, float, float, float], str]:
    """Outer temple for this yaw: the side the turn exposes."""
    if yaw < 0:
        return (0.05, 0.15, 0.22, 0.42), "temple_left"
    return (0.78, 0.15, 0.95, 0.42), "temple_right"


def hairline_box() -> tuple[tuple[float, float, float, float], str]:
    """Top of the forehead / hairline, where tipping back shows a bare scalp gap."""
    return (0.35, 0.05, 0.65, 0.22), "hairline"


def gap_score(front: Image.Image, posed: Image.Image, *, yaw: float = 0.0, pitch: float = 0.0) -> PoseScore:
    """How much barer the critical region is than in *front*."""
    if pitch < 0:  # tip back
        box, region = hairline_box()
    else:
        box, region = temple_box(yaw if yaw != 0 else 12.0)  # unused when yaw=0 and pitch>=0
        if yaw == 0 and pitch >= 0:
            # Front or tip forward: average both temples (forward rarely gaps).
            left_box, _ = temple_box(-12)
            right_box, _ = temple_box(12)
            gap = max(
                0.0,
                (skin_frac(posed, left_box) + skin_frac(posed, right_box)) / 2
                - (skin_frac(front, left_box) + skin_frac(front, right_box)) / 2,
            )
            return PoseScore(label="", yaw=yaw, pitch=pitch, gap=gap, region="temples")
    gap = max(0.0, skin_frac(posed, box) - skin_frac(front, box))
    return PoseScore(label="", yaw=yaw, pitch=pitch, gap=gap, region=region)


def score_poses(
    poses: dict[str, bytes] | dict[str, Image.Image],
    *,
    threshold: float = DEFAULT_GAP,
) -> TurnReport:
    """Score every named pose against ``front``. *poses* must include ``front``."""
    images = {k: (v if isinstance(v, Image.Image) else open_rgb(v)) for k, v in poses.items()}
    if "front" not in images:
        raise ValueError("poses need a front still")
    front = images["front"]
    scores: list[PoseScore] = []
    for label, turn in POSES:
        if label == "front":
            scores.append(PoseScore(label=label, yaw=0.0, pitch=0.0, gap=0.0, region="front"))
            continue
        if label not in images:
            continue
        yaw = float(turn.get("yaw", 0))
        pitch = float(turn.get("pitch", 0))
        found = gap_score(front, images[label], yaw=yaw, pitch=pitch)
        scores.append(PoseScore(label=label, yaw=yaw, pitch=pitch, gap=found.gap, region=found.region))
    limits = limits_from_scores(scores, threshold=threshold)
    return TurnReport(scores=tuple(scores), limits=limits, threshold=threshold)


def _first_fail_limit(samples: list[tuple[float, float]], *, threshold: float, default: float, floor: float) -> float:
    """Largest angle that still passes. *samples* are ``(angle, gap)`` with angle > 0."""
    last_ok: float | None = None
    for angle, gap in sorted(samples, key=lambda item: item[0]):
        if gap >= threshold:
            if last_ok is not None:
                return float(last_ok)
            return max(floor, angle - 2.0)
        last_ok = angle
    return float(last_ok if last_ok is not None else default)


def limits_from_scores(scores: list[PoseScore] | tuple[PoseScore, ...], *, threshold: float = DEFAULT_GAP) -> Limits:
    """Largest |yaw| / pitch_up / pitch_down that still pass."""
    by_yaw: dict[float, float] = {}
    for s in scores:
        if s.yaw == 0:
            continue
        angle = abs(s.yaw)
        by_yaw[angle] = max(by_yaw.get(angle, 0.0), s.gap)
    yaw_ok = _first_fail_limit(list(by_yaw.items()), threshold=threshold, default=20.0, floor=6.0)

    backs = [(abs(s.pitch), s.gap) for s in scores if s.pitch < 0]
    pitch_up = _first_fail_limit(backs, threshold=threshold, default=8.0, floor=5.0)

    forwards = [(s.pitch, s.gap) for s in scores if s.pitch > 0]
    pitch_down = _first_fail_limit(forwards, threshold=threshold, default=20.0, floor=10.0)

    return Limits(yaw=yaw_ok, pitch_up=pitch_up, pitch_down=pitch_down)


def measure_limits(
    capture: Any,
    *,
    threshold: float = DEFAULT_GAP,
    yaw_steps: tuple[int, ...] = YAW_PROBE,
    pitch_up_steps: tuple[int, ...] = PITCH_UP_PROBE,
) -> TurnReport:
    """Probe angles via *capture(yaw, pitch) -> PNG bytes*, then score the sheet poses plus the probes."""
    front = open_rgb(capture(0, 0))
    poses: dict[str, Image.Image] = {"front": front}
    scores: list[PoseScore] = [PoseScore(label="front", yaw=0.0, pitch=0.0, gap=0.0, region="front")]
    for yaw in yaw_steps:
        for sign, tag in ((-1, "m"), (1, "p")):
            label = f"yaw_{tag}{yaw}"
            img = open_rgb(capture(sign * yaw, 0))
            poses[label] = img
            found = gap_score(front, img, yaw=float(sign * yaw), pitch=0.0)
            scores.append(PoseScore(label=label, yaw=float(sign * yaw), pitch=0.0, gap=found.gap, region=found.region))
    for pitch in pitch_up_steps:
        label = f"back_{pitch}"
        img = open_rgb(capture(0, -pitch))
        poses[label] = img
        found = gap_score(front, img, yaw=0.0, pitch=float(-pitch))
        scores.append(PoseScore(label=label, yaw=0.0, pitch=float(-pitch), gap=found.gap, region=found.region))
    # Keep the sheet's forward pose too.
    forward = open_rgb(capture(0, 15))
    poses["forward_15"] = forward
    found = gap_score(front, forward, yaw=0.0, pitch=15.0)
    scores.append(PoseScore(label="forward_15", yaw=0.0, pitch=15.0, gap=found.gap, region=found.region))
    limits = limits_from_scores(scores, threshold=threshold)
    return TurnReport(scores=tuple(scores), limits=limits, threshold=threshold)


# ---- fixtures for tests: synthetic heads that gap like splat-01 vs syn-w101 ---------------------------------


def synthetic_pose(width: int, height: int, *, yaw: float = 0.0, pitch: float = 0.0, weak: bool = False) -> bytes:
    """A head-and-shoulders still. *weak* gaps from ~12° yaw / ~7° back; strong holds 20° / 8°."""
    image = Image.new("RGB", (width, height), (16, 18, 22))
    draw = ImageDraw.Draw(image)
    # Bust / face
    draw.ellipse((width * 0.28, height * 0.18, width * 0.72, height * 0.72), fill=(210, 175, 155))
    # Hair: dark oval; for weak heads, peel it back past the fail angles.
    hair = (35, 25, 20)
    draw.ellipse((width * 0.22, height * 0.08, width * 0.78, height * 0.48), fill=hair)
    # Temples covered by default.
    draw.rectangle((int(width * 0.18), int(height * 0.2), int(width * 0.32), int(height * 0.42)), fill=hair)
    draw.rectangle((int(width * 0.68), int(height * 0.2), int(width * 0.82), int(height * 0.42)), fill=hair)

    fail_yaw = 12 if weak else 22
    fail_back = 7 if weak else 9
    skin = (220, 185, 165)
    if abs(yaw) >= fail_yaw:
        # Outer temple bare (the smear / gap).
        if yaw < 0:
            draw.rectangle((int(width * 0.06), int(height * 0.16), int(width * 0.22), int(height * 0.42)), fill=skin)
        else:
            draw.rectangle((int(width * 0.78), int(height * 0.16), int(width * 0.94), int(height * 0.42)), fill=skin)
    if pitch <= -fail_back:
        # Hairline gap on tip-back.
        draw.rectangle((int(width * 0.34), int(height * 0.04), int(width * 0.66), int(height * 0.2)), fill=skin)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def mean_luma(image: Image.Image) -> float:
    return float(ImageStat.Stat(image.convert("L")).mean[0])
