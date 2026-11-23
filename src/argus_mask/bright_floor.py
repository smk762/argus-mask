"""Bright floor behind a splat bust (#175 / #151): a pale path under the fade reads as a grey band.

Measured on the plate in the region a close shot leaves open under the head (lower-middle of the
picture). Procedural scenes and a plain background have no plate to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

# Relative luminance (sRGB bytes) above which a pixel counts as "pale" for the floor band.
BRIGHT = 160.0
# Share of the behind-bust band that may be pale before the step flags an advisory.
# Calibrated so garden's pale path (~0.13) warns and lake's dusk water (~0.01) does not.
BRIGHT_SHARE = 0.05
# Band in image fractions: top, bottom, left, right — behind the bust in a close shot.
BAND = (0.55, 0.85, 0.30, 0.70)


@dataclass(frozen=True)
class BrightFloor:
    """What the plate's floor band looked like."""

    mean: float
    p95: float
    bright_share: float
    bright: bool
    threshold: float = BRIGHT_SHARE

    def as_scores(self) -> dict[str, float]:
        return {
            "floor_mean": round(self.mean, 3),
            "floor_p95": round(self.p95, 3),
            "floor_bright_share": round(self.bright_share, 5),
        }


def measure_plate(picture: bytes | Path, *, threshold: float = BRIGHT_SHARE) -> BrightFloor:
    """Score the behind-bust band of a plate PNG. Raises ``ValueError`` if it is not a readable RGB image."""
    data = picture.read_bytes() if isinstance(picture, Path) else picture
    with Image.open(BytesIO(data)) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        if w < 8 or h < 8:
            raise ValueError("plate is too small to score")
        yt, yb, xl, xr = BAND
        y0, y1 = int(h * yt), max(int(h * yt) + 1, int(h * yb))
        x0, x1 = int(w * xl), max(int(w * xl) + 1, int(w * xr))
        # Downsample the band for a cheap scan: every n-th pixel is enough for a share.
        step = max(1, min((x1 - x0) // 64, (y1 - y0) // 32, 8))
        samples: list[float] = []
        px = rgb.load()
        for y in range(y0, y1, step):
            for x in range(x0, x1, step):
                r, g, b = px[x, y][:3]
                samples.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
    if not samples:
        raise ValueError("plate band is empty")
    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    p95 = samples[min(n - 1, int(n * 0.95))]
    bright_share = sum(1 for v in samples if v > BRIGHT) / n
    return BrightFloor(
        mean=mean,
        p95=p95,
        bright_share=bright_share,
        bright=bright_share >= threshold,
        threshold=threshold,
    )


def measure_path(path: Path, *, threshold: float = BRIGHT_SHARE) -> BrightFloor:
    return measure_plate(path, threshold=threshold)
