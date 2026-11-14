"""Hair colour from melanin: the colour a preset's ``hair_colour`` asks for, and the recolour of a texture.

Pure numpy, and no import from the package: ``build_head.py`` runs in Blender's own Python, which has numpy
but not argus_mask, so it loads this file by path. The editor's tests import it like any module, so the
recipe and the tests run the same code.

The colour follows the melanin model of physically based hair shaders (d'Eon et al. 2011, as in Blender's
Principled Hair BSDF): each pigment absorbs light per channel, and a strand's colour is what passes through.
The cuticle reflects a little white light first. Eumelanin makes hair brown to black, pheomelanin makes it red to ginger, and neither gives near-white. A dye,
if any, is mixed over the natural colour.

The recolour keeps the texture's own strand variation: each pixel keeps its luminance relative to the
texture's mean, and takes the new colour at that brightness. Alpha is never touched.
"""

from __future__ import annotations

import numpy as np

# Absorption per unit of pigment, for (R, G, B): the coefficients Blender's Principled Hair BSDF uses.
EUMELANIN_ABSORPTION = np.array([0.419, 0.697, 1.37])
PHEOMELANIN_ABSORPTION = np.array([0.187, 0.4, 1.05])
# Rec. 709 luminance, on linear values.
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
# How far light travels through pigment before it leaves the hair, in strand widths: a strand seen in a
# texture is light that went through and back several times. Tuned by eye so the sliders span platinum to
# near-black, and auburn to copper.
PATH = 4.0
# The densest a slider's 1 makes the pigment: short of fully opaque, so black hair keeps its strand detail.
_DENSEST = 0.95
# White light scattered back from the cuticle without reaching the pigment, as a share of the albedo. It is
# why black hair reads as black, not as the deep red eumelanin passes alone. Less than keratin's Fresnel
# reflectance (about 0.047): the viewer's material draws the specular highlight itself.
CUTICLE = 0.02
PARTS = ("hair", "eyebrows", "eyelashes")
# Pixels at or under this alpha are cut out on screen; they do not count toward the texture's mean.
_SEEN_ALPHA = 0.35


def _amount(x: float) -> float:
    """A slider's 0 to 1 as an amount of pigment: 1 is dense enough to read as black."""
    return -float(np.log(1.0 - _DENSEST * min(max(x, 0.0), 1.0)))


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex_to_linear(value: str) -> np.ndarray:
    """``#rrggbb`` to linear RGB."""
    text = value.removeprefix("#")
    if len(text) != 6:
        raise ValueError(f"not a #rrggbb colour: {value!r}")
    return srgb_to_linear(np.array([int(text[i : i + 2], 16) / 255 for i in (0, 2, 4)]))


def natural(eumelanin: float, pheomelanin: float) -> np.ndarray:
    """The linear RGB of undyed hair with these amounts of each pigment, each 0 to 1."""
    absorption = _amount(eumelanin) * EUMELANIN_ABSORPTION + _amount(pheomelanin) * PHEOMELANIN_ABSORPTION
    return CUTICLE + (1.0 - CUTICLE) * np.exp(-PATH * absorption)


def colour(spec: dict) -> np.ndarray:
    """The linear RGB a preset's ``hair_colour`` asks for."""
    base = natural(float(spec.get("eumelanin", 0.0)), float(spec.get("pheomelanin", 0.0)))
    dye, strength = spec.get("dye"), float(spec.get("dye_strength", 0.0))
    if dye and strength > 0:
        base = (1.0 - strength) * base + strength * hex_to_linear(dye)
    return base


def to_hex(rgb_linear: np.ndarray) -> str:
    """Linear RGB to ``#rrggbb``: what the editor shows as the swatch."""
    return "#" + "".join(f"{round(float(v) * 255):02x}" for v in linear_to_srgb(rgb_linear))


def recolour(pixels: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Recolour sRGB *pixels* (N x 3 or N x 4, 0 to 1) to linear *target*, keeping their variation and alpha.

    Returns a new array. Each pixel keeps its luminance relative to the mean of the visible pixels; a
    texture with nothing visible comes back unchanged.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    out = pixels.copy()
    linear = srgb_to_linear(pixels[:, :3])
    lum = linear @ LUMINANCE
    seen = pixels[:, 3] > _SEEN_ALPHA if pixels.shape[1] == 4 else np.ones(len(pixels), dtype=bool)
    if not seen.any():
        return out
    mean = float(lum[seen].mean())
    if mean <= 0.0:
        return out
    out[:, :3] = linear_to_srgb(np.outer(lum / mean, target))
    return out
