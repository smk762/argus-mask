"""lam/'s measuring and Blender scripts and the photometric guard (#126), with LAM's environment stubbed.

`lam/` runs inside LAM's container (torch, pytorch3d) or Blender (bpy), none of which the repo's venv has. Each
module is loaded by path with small stand-ins for those imports: numpy arrays play torch's tensors, which is enough
for the code paths tested here, so these run in the main suite. `test_lam.py` runs the FLAME core on real torch.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

LAM = Path(__file__).resolve().parent.parent / "lam"


class Tensor(np.ndarray):
    """A numpy array with the few tensor methods the tested code calls."""

    def to(self, dtype=None, **_):
        return self.astype(dtype or self.dtype)

    def float(self):
        return self.astype(np.float64)


def _tensor(a, **_):
    return np.asarray(a).view(Tensor)


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float32, torch.float64, torch.long = np.float32, np.float64, np.int64
    torch.Tensor = Tensor
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.as_tensor = torch.tensor = _tensor
    torch.nonzero = lambda a: np.argwhere(np.asarray(a)).view(Tensor)
    torch.zeros = lambda *shape, dtype=None, **_: np.zeros(shape, dtype=dtype).view(Tensor)
    return torch


def _knn_points(a, b, K=1):
    d = ((np.asarray(a)[0][:, None] - np.asarray(b)[0][None]) ** 2).sum(-1)
    return types.SimpleNamespace(idx=np.argsort(d, axis=1)[:, :K][None])


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch):
    """Load a lam/ file by path with the stand-ins in place; whatever it imports or adds to sys.path is undone."""
    monkeypatch.setattr(sys, "path", [str(LAM), *sys.path])  # overrides.py imports flame_core from beside it
    before = set(sys.modules)
    pytorch3d = types.ModuleType("pytorch3d")
    pytorch3d.ops = types.SimpleNamespace(knn_points=_knn_points)
    stubs = {"torch": _fake_torch(), "pytorch3d": pytorch3d, "pytorch3d.ops": pytorch3d.ops}

    def _load(relative: str, **extra: types.ModuleType) -> types.ModuleType:
        for name, module in {**stubs, **extra}.items():
            monkeypatch.setitem(sys.modules, name, module)
        for name in ("flame_core", "ibug68", "overrides"):  # imported again, over the stand-ins
            monkeypatch.delitem(sys.modules, name, raising=False)
        spec = importlib.util.spec_from_file_location(f"lam_test_{Path(relative).stem}", LAM / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield _load
    for name in set(sys.modules) - before - set(stubs):
        sys.modules.pop(name, None)


# ---- landmarks/measure.py --------------------------------------------------------------------------------------


def test_casting_no_rays_gives_no_hits(load) -> None:
    measure = load("landmarks/measure.py")
    tris = np.zeros((2, 3, 3))
    hits = measure.ray_hits(np.empty((0, 3)), np.empty((0, 3)), tris)
    assert hits.shape == (0, 3)
    hits, which = measure.ray_hits(np.empty((0, 3)), np.empty((0, 3)), tris, with_triangle=True)
    assert hits.shape == (0, 3) and which.shape == (0,)


def test_a_head_no_view_detected_says_so(load, tmp_path: Path) -> None:
    measure = load("landmarks/measure.py")
    views = np.array(["a", "b", "c"])
    np.savez(tmp_path / "scene.npz", views=views, verts=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    np.savez(tmp_path / "mediapipe.npz", names=views, points=np.full((3, 478, 3), np.nan))
    with pytest.raises(SystemExit, match="only 0 views had a detection"):
        measure.head_landmarks(tmp_path)


def test_a_vertex_on_a_degenerate_face_still_lands_on_the_vertex(load) -> None:
    measure = load("landmarks/measure.py")
    faces = _tensor(np.array([[4, 4, 7], [4, 5, 6]]))
    fidx, w = measure.vertex_on_face(faces, 4)
    assert fidx == 0 and np.allclose(w, [0.5, 0.5, 0.0])
    _, w = measure.vertex_on_face(faces, 6)
    assert np.allclose(w, [0, 0, 1])


def test_surfaces_with_no_pair_in_reach_give_a_finite_loss(load) -> None:
    measure = load("landmarks/measure.py")
    a = _tensor(np.zeros((4, 3)))
    far = _tensor(np.full((4, 3), 1.0))  # a metre away: every pair beyond TRUNCATE
    with np.errstate(all="ignore"):
        loss, kept = measure.matched(a, far)
    assert np.isfinite(loss) and loss == pytest.approx(measure.TRUNCATE**2) and kept == 0
    near = _tensor(np.full((4, 3), 0.001))
    loss, kept = measure.matched(a, near)
    assert loss == pytest.approx(3e-6) and kept == 1
    with pytest.raises(SystemExit, match="no surface"):
        measure.matched(a, _tensor(np.empty((0, 3))))


# ---- landmarks/render_head.py ----------------------------------------------------------------------------------


class _Mesh:
    def __init__(self, verts, tris, fail: bool = False):
        self.vertices = [types.SimpleNamespace(co=v) for v in verts]
        self.loop_triangles = [types.SimpleNamespace(vertices=t) for t in tris]
        self._fail = fail

    def calc_loop_triangles(self):
        if self._fail:
            raise RuntimeError("bad mesh")


class _Object:
    def __init__(self, mesh: _Mesh):
        self.mesh, self.cleared = mesh, 0
        self.matrix_world = np.eye(4)

    def evaluated_get(self, _depsgraph):
        return self

    def to_mesh(self):
        return self.mesh

    def to_mesh_clear(self):
        self.cleared += 1


def _render_head(load) -> types.ModuleType:
    bpy = types.ModuleType("bpy")
    bpy.context = types.SimpleNamespace(evaluated_depsgraph_get=lambda: None)
    mathutils = types.ModuleType("mathutils")
    mathutils.Euler = mathutils.Vector = object
    return load("landmarks/render_head.py", bpy=bpy, mathutils=mathutils)


def test_an_empty_mesh_joins_the_surface(load) -> None:
    render_head = _render_head(load)
    head = _Object(_Mesh([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)]))
    lines = _Object(_Mesh([(0, 0, 1), (1, 0, 1)], []))  # no triangles
    empty = _Object(_Mesh([], []))
    verts, faces, counts = render_head.surface([head, lines, empty])
    assert verts.shape == (5, 3) and faces.tolist() == [[0, 1, 2]] and counts.tolist() == [3, 2, 0]
    assert [o.cleared for o in (head, lines, empty)] == [1, 1, 1]


def test_an_evaluated_mesh_is_cleared_when_reading_it_fails(load) -> None:
    render_head = _render_head(load)
    bad = _Object(_Mesh([(0, 0, 0)], [], fail=True))
    with pytest.raises(RuntimeError, match="bad mesh"):
        render_head.surface([bad])
    assert bad.cleared == 1


# ---- make_head.py ----------------------------------------------------------------------------------------------


def test_make_head_checks_nothing_with_assert() -> None:
    """Under python -O an assert is skipped, and with it any tracking step written inside one."""
    tree = ast.parse((LAM / "make_head.py").read_text(encoding="utf-8"))
    asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == [], f"make_head.py asserts on lines {asserts}"


# ---- overrides.py ----------------------------------------------------------------------------------------------


def _tyro_config(tmp_path: Path, photometric: str, extra: str = "") -> str:
    """A tracking config as tyro writes LAM's: a YAML string holding YAML with tyro's tags."""
    inner = (
        "# tyro YAML.\n!dataclass:BaseTrackingConfig\ndevice: cuda\nexp: !dataclass:ExperimentConfig\n"
        "  keyframes: !!python/tuple []\n  output_folder: !!python/object/apply:pathlib.PosixPath\n  - output\n"
        f"  photometric: {photometric}\n  reuse_landmarks: true\n{extra}"
    )
    path = tmp_path / "base_tracking_config.yaml"
    path.write_text(yaml.safe_dump(inner))
    return str(path)


@pytest.mark.parametrize("value", ["false", "False", "no", "off"])
def test_photometric_off_in_any_yaml_spelling_passes(load, tmp_path: Path, value: str) -> None:
    overrides = load("overrides.py")
    overrides.check_photometric_off(_tyro_config(tmp_path, value))


@pytest.mark.parametrize(
    ("value", "extra"),
    [
        ("True", "# photometric: false\n"),  # on, with the off spelling only in a comment
        ("yes", ""),
        ("true", "use_photometric: false\n"),  # on, with the off spelling in another key
    ],
)
def test_photometric_on_is_refused_however_it_is_written(load, tmp_path: Path, value: str, extra: str) -> None:
    overrides = load("overrides.py")
    with pytest.raises(RuntimeError, match="photometric tracking must be off"):
        overrides.check_photometric_off(_tyro_config(tmp_path, value, extra))


@pytest.mark.parametrize("text", ["exp:\n  reuse_landmarks: true\n", "photometric: false\nexp:\n  photometric: yes\n"])
def test_a_config_without_photometric_off_everywhere_is_refused(load, tmp_path: Path, text: str) -> None:
    overrides = load("overrides.py")
    path = tmp_path / "tracking.yaml"
    path.write_text(text)
    with pytest.raises(RuntimeError, match="photometric tracking must be off"):
        overrides.check_photometric_off(str(path))
