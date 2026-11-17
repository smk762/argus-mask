"""lam/'s checked pipeline, without LAM or a GPU (#136): the preflight's refusals, our FLAME core against a small
synthetic FLAME, and `overrides.verify()` against stand-ins for LAM's and VHAP's modules.

The preflight needs only numpy. The rest needs torch, which mask does not depend on: CI's `lam` job installs a CPU
build and sets ``ARGUS_MASK_LAM_TESTS=1``, so there a missing torch fails instead of skipping.
"""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

LAM = Path(__file__).resolve().parent.parent / "lam"
# lam/'s modules, imported at top level as make_head.py does from /argus, and the LAM and VHAP modules the tests
# register: taken out again after each test. Nothing else is: torch, for one, can be imported only once.
LAM_NAMES = {"flame_core", "overrides", "preflight", "landmarks", "lam", "vhap"}


@pytest.fixture
def lam(monkeypatch):
    """lam/ on the path, as make_head.py has it; every module imported or registered here is gone afterwards."""
    monkeypatch.syspath_prepend(str(LAM))
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.split(".")[0] in LAM_NAMES:
            del sys.modules[name]
    for name in [n for n in sys.modules if n.split(".")[0] in {"lam", "vhap"}]:
        del sys.modules[name]  # registered by a test, under a name that was there before


def torch_or_skip():
    if os.environ.get("ARGUS_MASK_LAM_TESTS") == "1":
        import torch  # noqa: PLC0415 - CI's lam job installs it; a missing torch must fail there

        return torch
    return pytest.importorskip("torch", reason="lam/'s FLAME code needs torch (CI's lam job installs it)")


# -- the preflight: FLAME 2023 Open by sha256, our landmark embedding ---------------------------------------------


def _checkout(root: Path, flame: bytes, embedding_of) -> None:
    """A LAM checkout's model files: *flame* at each FLAME path, *embedding_of(path)* writing each embedding."""
    import preflight  # noqa: PLC0415 - from lam/, on the path by the fixture

    for rel in preflight.FLAME_PATHS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(flame)
    for rel in preflight.EMBEDDING_PATHS:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        embedding_of(root / rel)


def _ours(path: Path) -> None:
    from landmarks import embedding  # noqa: PLC0415

    np.save(path, embedding.load(), allow_pickle=True)


def test_the_preflight_passes_flame_open_and_our_embedding(lam, tmp_path) -> None:
    import preflight  # noqa: PLC0415

    model = b"stands in for flame2023_Open.pkl"
    _checkout(tmp_path, model, _ours)
    assert preflight.problems(tmp_path, flame_sha256=hashlib.sha256(model).hexdigest()) == []


def test_the_preflight_refuses_a_flame_model_that_is_not_flame_open(lam, tmp_path) -> None:
    import preflight  # noqa: PLC0415

    _checkout(tmp_path, b"the academic FLAME 2023, or anything else", _ours)
    refused = preflight.problems(tmp_path)
    assert len(refused) == 2 and all("is not FLAME 2023 Open" in r for r in refused), refused
    (tmp_path / preflight.FLAME_PATHS[0]).unlink()
    assert any("is missing" in r for r in preflight.problems(tmp_path))


def _shifted(path: Path) -> None:
    from landmarks import embedding  # noqa: PLC0415

    theirs = embedding.load()
    theirs["full_lmk_bary_coords"] = theirs["full_lmk_bary_coords"] + 1e-3
    np.save(path, theirs, allow_pickle=True)


@pytest.mark.parametrize(
    "embedding_of",
    [
        _shifted,  # another embedding: MPI's, or ours re-measured and not committed
        lambda path: path.write_bytes(b"not an array"),
        lambda path: np.save(path, np.zeros(3)),
        lambda path: None,  # not mounted
    ],
    ids=["another", "garbage", "not-a-dict", "missing"],
)
def test_the_preflight_refuses_an_embedding_that_is_not_ours(lam, tmp_path, embedding_of) -> None:
    import preflight  # noqa: PLC0415

    model = b"stands in for flame2023_Open.pkl"
    _checkout(tmp_path, model, embedding_of)
    refused = preflight.problems(tmp_path, flame_sha256=hashlib.sha256(model).hexdigest())
    assert len(refused) == 2 and all("not argus-mask's landmark embedding" in r for r in refused), refused


def test_make_head_refuses_before_any_lam_code_runs() -> None:
    """The preflight sits between overrides.install() and the first import from LAM: nothing of LAM's runs first."""
    lines = (LAM / "make_head.py").read_text(encoding="utf-8").splitlines()
    check = next(i for i, line in enumerate(lines) if line.startswith("if refused := preflight.problems()"))
    first_lam = next(i for i, line in enumerate(lines) if line.startswith(("from app_lam", "from lam.", "from tools.")))
    assert check < first_lam


# -- flame_core against a small synthetic FLAME ---------------------------------------------------------------------

PARENTS = [-1, 0, 1, 1, 1]  # FLAME's chain: global, neck, jaw, two eyes


def _quaternion_rotation(v: np.ndarray) -> np.ndarray:
    """Axis-angle to a rotation matrix through the unit quaternion: a route independent of Rodrigues' formula."""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    w, (x, y, z) = np.cos(theta / 2), np.sin(theta / 2) * v / theta
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _synthetic_flame(seed: int = 0, vertices: int = 24, betas: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    joints = len(PARENTS)
    weights = rng.random((vertices, joints)) ** 3
    regressor = rng.random((joints, vertices))
    return {
        "template": rng.normal(size=(vertices, 3)),
        "shapedirs": rng.normal(scale=0.1, size=(vertices, 3, betas)),
        "posedirs": rng.normal(scale=0.01, size=((joints - 1) * 9, vertices * 3)),
        "J_regressor": regressor / regressor.sum(axis=1, keepdims=True),
        "weights": weights / weights.sum(axis=1, keepdims=True),
        "betas": rng.normal(size=(2, betas)),
        "pose": rng.normal(scale=0.4, size=(2, joints * 3)),
    }


def _reference_lbs(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """FLAME posed by the textbook, one vertex and one joint at a time: (vertices (B, V, 3), joints (B, J, 3))."""
    out_v, out_j = [], []
    for betas, pose in zip(m["betas"], m["pose"], strict=True):
        shaped = m["template"] + np.einsum("vck,k->vc", m["shapedirs"], betas)
        rest = m["J_regressor"] @ shaped
        rots = [_quaternion_rotation(pose[3 * j : 3 * j + 3]) for j in range(len(PARENTS))]
        feature = np.concatenate([(r - np.eye(3)).ravel() for r in rots[1:]])
        posed = shaped + (feature @ m["posedirs"]).reshape(-1, 3)
        world = []
        for j, parent in enumerate(PARENTS):
            local = np.eye(4)
            local[:3, :3] = rots[j]
            local[:3, 3] = rest[j] - (rest[parent] if parent >= 0 else 0)
            world.append(local if parent < 0 else world[parent] @ local)
        skin = []
        for j, w in enumerate(world):
            unbind = np.eye(4)
            unbind[:3, 3] = -rest[j]
            skin.append(w @ unbind)
        verts = []
        for v, weights in zip(posed, m["weights"], strict=True):
            t = sum(wt * s for wt, s in zip(weights, skin, strict=True))
            verts.append((t @ np.append(v, 1.0))[:3])
        out_v.append(verts)
        out_j.append([w[:3, 3] for w in world])
    return np.array(out_v), np.array(out_j)


def _tensors(torch, m: dict) -> dict:
    return {k: torch.as_tensor(v, dtype=torch.float64) for k, v in m.items()}


def test_rodrigues_agrees_with_the_quaternion_route(lam) -> None:
    torch = torch_or_skip()
    import flame_core  # noqa: PLC0415

    vecs = np.random.default_rng(1).normal(size=(16, 3))
    vecs[0] = 0  # the identity, whatever the axis
    got = flame_core.batch_rodrigues(torch.as_tensor(vecs), dtype=torch.float64).numpy()
    want = np.stack([_quaternion_rotation(v) for v in vecs])
    assert np.allclose(got, want, atol=1e-7)


def test_lbs_poses_a_synthetic_flame_as_the_textbook_does(lam) -> None:
    torch = torch_or_skip()
    import flame_core  # noqa: PLC0415

    m = _synthetic_flame()
    t = _tensors(torch, m)
    shaped = t["template"] + flame_core.blend_shapes(t["betas"], t["shapedirs"])
    parents = torch.tensor(PARENTS)
    verts, joints, neck = flame_core.lbs(
        t["pose"], shaped, t["posedirs"], t["J_regressor"], parents, t["weights"], dtype=torch.float64
    )
    want_v, want_j = _reference_lbs(m)
    assert np.allclose(verts.numpy(), want_v, atol=1e-6)
    assert np.allclose(joints.numpy(), want_j, atol=1e-6)
    assert neck.shape == (2, 4, 4)


def test_skinning_is_undone_by_inverse_skinning(lam) -> None:
    torch = torch_or_skip()
    import flame_core  # noqa: PLC0415

    t = _tensors(torch, _synthetic_flame(seed=2))
    shaped = t["template"].expand(2, -1, -1)
    transforms, _ = flame_core.transforms_for_pose(
        t["pose"], shaped, torch.tensor(PARENTS), t["J_regressor"], dtype=torch.float64
    )
    posed = flame_core.skinning(shaped, transforms, t["weights"])
    assert not torch.allclose(posed, shaped)
    assert torch.allclose(flame_core.inverse_skinning(posed, transforms, t["weights"]), shaped, atol=1e-9)


def test_landmarks_are_barycentric_points_on_their_faces(lam) -> None:
    torch = torch_or_skip()
    import flame_core  # noqa: PLC0415

    vertices = torch.as_tensor(np.random.default_rng(3).normal(size=(2, 6, 3)))
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    idx, bary = torch.tensor([1, 0]), torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]], dtype=torch.float64)
    got = flame_core.vertices2landmarks(vertices, faces, idx, bary)
    want = torch.stack([0.2 * vertices[:, 3] + 0.3 * vertices[:, 4] + 0.5 * vertices[:, 5], vertices[:, 0]], dim=1)
    assert torch.allclose(got, want)


# -- overrides: every Max-Planck-noticed name replaced, or verify() says which is left ------------------------------


def _stand_ins(torch) -> dict[str, types.ModuleType]:
    """LAM's and VHAP's FLAME modules as overrides.py sees them: their own helpers, skinning and texture."""

    def theirs(*args, **kwargs):  # any of LAM's own functions
        raise AssertionError("LAM's code ran")

    class FlameMask:
        def process_vertex_mask(self, flame_parts_path=None):
            theirs()

        def create_custom_mask(self):
            self.custom = True

    class BufferContainer(torch.nn.Module):
        pass

    class FlameHeadSubdivided:
        get_transformed_mat = skinning = inverse_skinning = theirs

    class FlameTexPainted:
        pass

    modules = {}
    for name in (
        "lam.models.rendering.flame_model.flame",
        "lam.models.rendering.flame_model.flame_arkit",
        "vhap.model.flame",
        "vhap.model.tracker",
    ):
        module = types.ModuleType(name)
        if name != "vhap.model.tracker":
            module.FlameMask, module.BufferContainer = FlameMask, BufferContainer
            module.FlameMask.__module__ = name
            for fn in ("to_tensor", "to_np", "Struct", "lbs", "vertices2landmarks", "batch_rodrigues"):
                setattr(module, fn, theirs)
        if name.endswith("flame_model.flame"):
            module.FlameHeadSubdivided = FlameHeadSubdivided
        if name.startswith("vhap."):
            module.FlameTexPainted = FlameTexPainted
        modules[name] = module
    return modules


def test_verify_names_what_is_still_lams_until_patch_replaces_it(lam, tmp_path, monkeypatch) -> None:
    torch = torch_or_skip()
    import flame_core  # noqa: PLC0415
    import overrides  # noqa: PLC0415

    overrides.install()
    for name, module in _stand_ins(torch).items():
        sys.modules[name] = module
    before = overrides.verify()
    assert "vhap.model.flame.lbs" in before and "vhap.model.tracker.FlameTexPainted" in before
    assert "lam.models.rendering.flame_model.flame.FlameHeadSubdivided.skinning" in before

    config = tmp_path / overrides.TRACKING_CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("photometric: false\n")
    monkeypatch.chdir(tmp_path)
    overrides.patch()
    assert overrides.verify() == []
    mask = sys.modules["vhap.model.flame"].FlameMask()
    mask.process_vertex_mask()  # placeholders, and no file read
    assert mask.custom and all(getattr(mask.v, r).numel() == 1 for r in overrides.MASK_REGIONS)

    sys.modules["vhap.model.flame"].to_np = lambda *a: None  # one name back to LAM's
    assert overrides.verify() == ["vhap.model.flame.to_np"]
    assert sys.modules["vhap.model.lbs"] is flame_core


def test_install_refuses_once_lams_skinning_is_imported(lam) -> None:
    torch_or_skip()
    import overrides  # noqa: PLC0415

    sys.modules["vhap.model.lbs"] = types.ModuleType("vhap.model.lbs")
    with pytest.raises(RuntimeError, match="before overrides.install"):
        overrides.install()


def test_photometric_tracking_is_refused(lam, tmp_path) -> None:
    torch_or_skip()
    import overrides  # noqa: PLC0415

    config = tmp_path / "tracking.yaml"
    config.write_text("photometric: true\n")
    with pytest.raises(RuntimeError, match="photometric tracking must be off"):
        overrides.check_photometric_off(str(config))
