"""bake/'s offline tools: picking expressions, camera views and the check page's comparison. They are not part of the
package and need Blender, Pillow or playwright to run, so they are loaded by path with what's missing stubbed."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

BAKE = Path(__file__).parent.parent / "bake"


def _load(name: str, path: Path, stubs: tuple[str, ...] = ()) -> types.ModuleType:
    """Import *path* as *name*, with an empty module standing in for each of *stubs* not installed."""
    missing = {}
    for stub in stubs:
        try:
            importlib.import_module(stub)
        except ImportError:
            missing[stub] = types.ModuleType(stub)
    if "mathutils" in missing:
        missing["mathutils"].Matrix = missing["mathutils"].Vector = object
    if "PIL" in missing:
        missing["PIL.Image"] = missing["PIL"].Image = types.ModuleType("PIL.Image")
    if "playwright" in missing:
        missing["playwright.async_api"] = types.ModuleType("playwright.async_api")
        missing["playwright.async_api"].async_playwright = None
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, missing):
        spec.loader.exec_module(module)
    return module


expressions = _load("bake_expressions", BAKE / "expressions.py")
render_views = _load("bake_render_views", BAKE / "render_views.py", ("bpy", "mathutils"))
drive = _load("bake_drive", BAKE / "check" / "drive.py", ("PIL", "playwright"))


# ---- expressions.spread


def test_spread_never_picks_a_frame_twice() -> None:
    weights = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], np.float32)
    picked = expressions.spread(weights, 5)
    assert sorted(picked) == [0, 1, 2]


def test_spread_skips_frames_that_repeat_a_pick() -> None:
    weights = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]], np.float32)
    picked = expressions.spread(weights, 4)
    assert len(picked) == len({tuple(weights[i]) for i in picked}) == 2


def test_spread_of_no_frames_is_empty() -> None:
    assert expressions.spread(np.zeros((0, 3), np.float32), 3, start=np.zeros(3, np.float32)) == []


def test_spread_starts_with_the_frame_farthest_from_start() -> None:
    weights = np.array([[0.1, 0.0], [0.9, 0.9], [0.5, 0.0]], np.float32)
    assert expressions.spread(weights, 2, start=np.zeros(2, np.float32))[0] == 1


# ---- render_views


@pytest.mark.parametrize("eye", [[0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.3, -0.5]])
def test_look_at_gives_a_rotation_from_any_side(eye: list[float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = render_views.look_at(np.array(eye), np.zeros(3))
    rot = m[:3, :3]
    assert np.isfinite(m).all()
    assert np.allclose(rot.T @ rot, np.eye(3), atol=1e-9)
    assert np.allclose(-rot[:, 2], -np.array(eye) / np.linalg.norm(eye))  # looks down -Z at the target


def test_look_at_refuses_a_camera_at_its_target() -> None:
    with pytest.raises(ValueError, match="target"):
        render_views.look_at(np.ones(3), np.ones(3))


def test_the_camera_index_survives_a_write_cut_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = tmp_path / "cameras.json"
    render_views.save_index(index, [{"file": "train/00_00.png"}])
    real = Path.write_text

    def dies_halfway(self: Path, text: str, *args: object, **kwargs: object) -> int:
        real(self, text[: len(text) // 2], *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", dies_halfway)
    with pytest.raises(KeyboardInterrupt):
        render_views.save_index(index, [{"file": "train/00_00.png"}, {"file": "train/00_01.png"}])
    monkeypatch.undo()
    assert json.loads(index.read_text()) == [{"file": "train/00_00.png"}]


# ---- check/drive


def test_psnr_of_identical_frames_is_finite() -> None:
    a = np.full((4, 4, 3), 128, np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value = drive.psnr(a, a.copy())
    assert math.isfinite(value)
    assert value > drive.psnr(a, a + 1)


def test_a_view_without_ground_truth_is_drawn_not_compared(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drive, "MODE", "splat")
    (tmp_path / "view0003_trained.png").write_bytes(b"")
    line = drive.compare(3, np.zeros((2, 2, 3), np.uint8), tmp_path)
    assert line.startswith("view 3: drawn")
    assert "view0003_gt.png" in line
