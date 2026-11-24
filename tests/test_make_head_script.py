"""make_head.py runs LAM as it is imported, so these hold the script's text rather than calling it."""

from __future__ import annotations

from pathlib import Path

HEAD = Path(__file__).resolve().parent.parent / "lam" / "make_head.py"


def test_the_fitted_shape_stays_on_the_input_device() -> None:
    """The fit is a drop-in for the shape it was given: same dtype and the same device, GPU included."""
    text = HEAD.read_text(encoding="utf-8")
    assert "return torch.as_tensor(best, dtype=shape_param.dtype, device=shape_param.device)" in text


def test_motion_enlarge_ratio_matches_the_portrait() -> None:
    """``[1.0, 1, 0]`` is a three-element list; the portrait is enlarged by ``[1.0, 1.0]``."""
    text = HEAD.read_text(encoding="utf-8")
    assert "enlarge_ratio=[1.0, 1, 0]" not in text
    assert text.count("enlarge_ratio=[1.0, 1.0]") == 3


def test_the_tracker_works_in_the_jobs_own_directory() -> None:
    """A shared one would hand a head another portrait's crop, matte and shape."""
    text = HEAD.read_text(encoding="utf-8")
    assert 'FlameTrackingSingleImage(output_dir=os.path.join(out_dir, "tracking"))' in text
    assert '"output/tracking"' not in text
