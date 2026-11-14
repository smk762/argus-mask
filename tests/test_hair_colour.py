from __future__ import annotations

import numpy as np
import pytest

from argus_mask.editor import hair_colour as hc


def lum(rgb_linear: np.ndarray) -> float:
    return float(np.asarray(rgb_linear) @ hc.LUMINANCE)


def test_no_pigment_is_near_white_and_more_eumelanin_is_darker() -> None:
    assert lum(hc.natural(0.0, 0.0)) > 0.99
    shades = [lum(hc.natural(e, 0.1)) for e in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert shades == sorted(shades, reverse=True)
    black = hc.natural(1.0, 0.0)
    assert lum(black) < 0.06  # a slider's 1 reads as black
    assert black.max() / black.min() < 1.6  # a near-neutral black, not the deep red eumelanin passes alone


def test_pheomelanin_makes_it_redder() -> None:
    brown = hc.natural(0.4, 0.0)
    auburn = hc.natural(0.4, 0.8)
    assert auburn[0] / auburn[2] > brown[0] / brown[2]


def test_dye_mixes_over_the_natural_colour() -> None:
    plain = hc.colour({"eumelanin": 0.3})
    assert np.allclose(hc.colour({"eumelanin": 0.3, "dye": "#2040ff", "dye_strength": 0.0}), plain)
    assert np.allclose(
        hc.colour({"eumelanin": 0.3, "dye": "#2040ff", "dye_strength": 1.0}), hc.hex_to_linear("#2040ff")
    )
    half = hc.colour({"eumelanin": 0.3, "dye": "#2040ff", "dye_strength": 0.5})
    assert np.allclose(half, (plain + hc.hex_to_linear("#2040ff")) / 2)


def test_hex_round_trips() -> None:
    for value in ("#000000", "#ffffff", "#8a4b2c"):
        assert hc.to_hex(hc.hex_to_linear(value)) == value
    with pytest.raises(ValueError):
        hc.hex_to_linear("#fff")


def test_recolour_keeps_strand_variation_and_alpha() -> None:
    rng = np.random.default_rng(7)
    grey = rng.uniform(0.2, 0.6, size=(500, 1))
    pixels = np.hstack([grey * [1.0, 0.8, 0.6], rng.uniform(0, 1, size=(500, 1))])  # a brown texture, RGBA
    target = hc.natural(0.8, 0.1)
    out = hc.recolour(pixels, target)
    assert out.shape == pixels.shape
    assert np.array_equal(out[:, 3], pixels[:, 3])  # alpha untouched
    before = hc.srgb_to_linear(pixels[:, :3]) @ hc.LUMINANCE
    after = hc.srgb_to_linear(out[:, :3]) @ hc.LUMINANCE
    assert np.corrcoef(before, after)[0, 1] > 0.99  # light strands stay light, dark stay dark
    # Each pixel is the target's hue: its channels in the target's proportions.
    linear = hc.srgb_to_linear(out[:, :3])
    assert np.allclose(linear / linear[:, :1], target / target[0], atol=1e-6)
    # The visible pixels average to the target.
    seen = pixels[:, 3] > 0.35
    assert np.allclose(linear[seen].mean(axis=0), target, rtol=1e-6)


def test_recolour_is_deterministic_and_ignores_cut_out_pixels() -> None:
    pixels = np.array([[0.5, 0.4, 0.3, 1.0], [0.9, 0.9, 0.9, 0.0]])
    a = hc.recolour(pixels, hc.natural(0.5, 0.2))
    assert np.array_equal(a, hc.recolour(pixels, hc.natural(0.5, 0.2)))
    # The invisible white pixel does not count toward the mean: the visible one lands on the target.
    assert np.allclose(hc.srgb_to_linear(a[0, :3]), hc.natural(0.5, 0.2))


def test_a_texture_with_nothing_visible_is_left_alone() -> None:
    pixels = np.array([[0.5, 0.4, 0.3, 0.0]])
    assert np.array_equal(hc.recolour(pixels, hc.natural(0.5, 0.2)), pixels)
