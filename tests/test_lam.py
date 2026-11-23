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
LAM_NAMES = {"flame_core", "overrides", "preflight", "landmarks", "fit_shape", "fit_rest", "likeness", "portraits",
             "refine_gaussians", "lam", "vhap"}  # fmt: skip


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


def _pickled(path: Path, pickle: bytes) -> None:
    """An object-array `.npy` whose pickle is *pickle*, as np.save would frame it."""
    np.save(path, np.array(None, dtype=object), allow_pickle=True)
    data = path.read_bytes()
    end = data.index(b"\n") + 1  # the header ends the first line
    path.write_bytes(data[:end] + pickle)


def test_the_preflight_never_runs_what_it_checks(lam, tmp_path) -> None:
    """A pickle that would run code on load is refused unrun: the check that vets the embedding must not execute it."""
    import preflight  # noqa: PLC0415

    ran = tmp_path / "ran"
    model = b"stands in for flame2023_Open.pkl"
    payload = f"cos\nsystem\n(Vtouch {ran}\ntR.".encode()
    _checkout(tmp_path, model, lambda path: _pickled(path, payload))
    refused = preflight.problems(tmp_path, flame_sha256=hashlib.sha256(model).hexdigest())
    assert len(refused) == 2 and all("not argus-mask's landmark embedding" in r for r in refused), refused
    assert not ran.exists()


@pytest.mark.parametrize(
    "embedding_of",
    [
        _shifted,  # another embedding: MPI's, or ours re-measured and not committed
        lambda path: path.write_bytes(b"not an array"),
        lambda path: np.save(path, np.zeros(3)),
        lambda path: None,  # not mounted
        lambda path: _pickled(path, b"cchumpy.ch\nCh\n)\x81."),  # a class not importable here (chumpy's)
    ],
    ids=["another", "garbage", "not-a-dict", "missing", "foreign-class"],
)
def test_the_preflight_refuses_an_embedding_that_is_not_ours(lam, tmp_path, embedding_of) -> None:
    import preflight  # noqa: PLC0415

    model = b"stands in for flame2023_Open.pkl"
    _checkout(tmp_path, model, embedding_of)
    refused = preflight.problems(tmp_path, flame_sha256=hashlib.sha256(model).hexdigest())
    assert len(refused) == 2 and all("not argus-mask's landmark embedding" in r for r in refused), refused


def test_a_format_3_npy_keeps_unicode_field_names(lam, tmp_path) -> None:
    """Format 3.0 encodes the header as UTF-8. The 2.0 reader would latin1-decode a non-ASCII field name."""
    import preflight  # noqa: PLC0415

    values = np.zeros(1, dtype=np.dtype([("café", "<f8")]))
    values["café"] = 1.25
    path = tmp_path / "embedding.npy"
    with path.open("wb") as handle:
        np.lib.format.write_array(handle, values, version=(3, 0))
    loaded = preflight._load_embedding(path)
    assert loaded.dtype.names == ("café",) and loaded["café"][0] == 1.25


def test_a_fortran_ordered_npy_loads_unscrambled(lam, tmp_path) -> None:
    import preflight  # noqa: PLC0415

    values = np.asfortranarray(np.arange(6).reshape(2, 3))
    path = tmp_path / "embedding.npy"
    np.save(path, values)
    loaded = preflight._load_embedding(path)
    assert loaded.shape == values.shape and np.array_equal(loaded, values)


def test_an_unknown_npy_version_is_refused(lam, tmp_path) -> None:
    import preflight  # noqa: PLC0415

    path = tmp_path / "embedding.npy"
    path.write_bytes(b"\x93NUMPY\x04\x00" + b"\x00" * 8)
    with pytest.raises(ValueError, match="not supported"):
        preflight._load_embedding(path)


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
            # Each its own class, so each stand-in's FlameMask says it comes from its own module.
            module.FlameMask = type("FlameMask", (FlameMask,), {"__module__": name})
            module.BufferContainer = BufferContainer
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


def test_install_refuses_once_lams_skinning_is_imported(lam, monkeypatch) -> None:
    torch_or_skip()
    import overrides  # noqa: PLC0415

    monkeypatch.setitem(sys.modules, "vhap.model.lbs", types.ModuleType("vhap.model.lbs"))
    with pytest.raises(RuntimeError, match="before overrides.install"):
        overrides.install()


def test_photometric_tracking_is_refused(lam, tmp_path) -> None:
    torch_or_skip()
    import overrides  # noqa: PLC0415

    config = tmp_path / "tracking.yaml"
    config.write_text("photometric: true\n")
    with pytest.raises(RuntimeError, match="photometric tracking must be off"):
        overrides.check_photometric_off(str(config))


def test_a_portrait_darkroom_did_not_pass_or_made_with_a_lora_is_not_a_cleared_source(lam) -> None:
    """lam/portraits.py (#151) keeps a portrait only with a record that shows it synthetic: passed, and no LoRA."""
    import portraits  # noqa: PLC0415 - lam/ is on the path only inside the fixture

    try:
        assert portraits.seeds("501-503,507") == [501, 502, 503, 507]
        assert portraits.refusal({"outcome": "passed", "request": {"loras": []}}) is None
        assert "refused" in portraits.refusal({"outcome": "refused", "outcome_reason": "unchecked"})
        assert "no record" in portraits.refusal({})
        assert "LoRA" in portraits.refusal({"outcome": "passed", "request": {"loras": [{"name": "a-character"}]}})
        stream = [b"event: progress\n", b'data: {"step": 1}\n', b"\n", b'data: {"record": {"id": "r1"}}\n']
        assert portraits.last_event(stream) == {"record": {"id": "r1"}}
        assert portraits.last_event([]) == {}
        assert "{subject}" in portraits.TEMPLATE and "not smiling" in portraits.TEMPLATE
    finally:
        sys.modules.pop("portraits", None)


def test_every_portrait_style_frames_the_subject_square_on_and_neutral(lam) -> None:
    """Each --style (#151) is a text prompt around the subject, asking for what LAM needs: square on, neutral, lit flat."""
    import portraits  # noqa: PLC0415 - lam/ is on the path only inside the fixture

    assert set(portraits.TEMPLATES) == {"photo", "stylised", "semireal", "character"}
    for template in portraits.TEMPLATES.values():
        assert template.count("{subject}") == 1
        assert "facing the camera squarely" in template and "neutral expression" in template
        assert "lighting" in template and "background" in template


# -- fitting a head to its portrait (#151): pure numpy, the renders handed in --------------------------------------


def _face(rng: np.random.Generator) -> np.ndarray:
    """478 MediaPipe-like points in pixels, the iris points placed so the ratios are defined."""
    points = rng.uniform(100, 400, (478, 2))
    points[469], points[471], points[474], points[476] = (200, 250), (220, 250), (300, 250), (320, 250)
    points[159], points[145], points[386], points[374] = (210, 242), (210, 258), (310, 242), (310, 258)
    return points


def test_a_render_that_lands_on_the_portrait_scores_zero_and_a_moved_outline_scores_most(lam) -> None:
    import fit_shape  # noqa: PLC0415

    rng = np.random.default_rng(0)
    ref = np.c_[rng.random((478, 2)), np.zeros(478)]
    assert fit_shape.misfit(ref, ref.copy())["total"] == 0.0
    assert fit_shape.misfit(ref, None)["total"] == float("inf")
    assert fit_shape.misfit(ref, np.full_like(ref, np.nan))["total"] == float("inf")
    oval, lips = ref.copy(), ref.copy()
    oval[fit_shape.OVAL, 0] += 0.01
    lips[fit_shape.LIPS, 0] += 0.01
    assert fit_shape.misfit(ref, oval)["total"] > fit_shape.misfit(ref, lips)["total"] > 0


def test_the_search_is_seeded_finds_a_better_shape_and_never_returns_a_worse_one(lam) -> None:
    import fit_shape  # noqa: PLC0415

    target = np.r_[np.linspace(1, -1, 20), np.zeros(80)]

    def score(shapes):
        return [float(((s[:20] - target[:20]) ** 2).sum()) for s in shapes]

    start = np.zeros(100)
    first = fit_shape.search(start, score, log=lambda _: None)
    again = fit_shape.search(start, score, log=lambda _: None)
    assert np.array_equal(first[0], again[0]) and first[1] == again[1]
    assert first[1] < 0.5 * score([start])[0]
    assert np.array_equal(first[0][20:], start[20:])  # only the first `dims` components move
    assert first[1] <= min(h["best"] for h in first[2])

    def worse_everywhere(shapes):
        return [0.0 if np.array_equal(s, start) else 1.0 for s in shapes]

    assert np.array_equal(fit_shape.search(start, worse_everywhere, log=lambda _: None)[0], start)


def test_a_rest_fit_is_scored_after_aligning_the_face_and_reads_the_lids_and_lips(lam) -> None:
    import fit_rest  # noqa: PLC0415

    rng = np.random.default_rng(1)
    points = _face(rng)
    scores = {k: 0.2 for k in fit_rest.SCORES}
    ref = {"points": points, "scores": scores}
    # The same face moved, turned and scaled on the screen: the alignment takes all of it out.
    angle = np.radians(7)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = {"points": 1.3 * points @ rot.T + (40, -25), "scores": dict(scores)}
    assert fit_rest.misfit(ref, moved)["total"] < 1e-9
    assert fit_rest.misfit(ref, None)["total"] == float("inf")
    # Lids lower over the iris: the aperture reads it, and the score rises.
    hooded = {"points": points.copy(), "scores": dict(scores)}
    hooded["points"][[159, 386], 1] += 4
    result = fit_rest.misfit(ref, hooded)
    assert result["aperture"] == pytest.approx(0.25)
    assert result["total"] > 0.25 * fit_rest.RATIO_WEIGHT


def test_a_rest_fit_gives_a_symmetric_rest_set_inside_the_wire_limits(lam) -> None:
    import fit_rest  # noqa: PLC0415

    from argus_mask.arkit import ARKIT_52
    from argus_mask.expressions import BUILTIN
    from argus_mask.models import RestSet

    weights = fit_rest.weights(np.array([-0.5, 2.0, *[0.25] * (len(fit_rest.PARAMS) - 2)]), min_smile=0.0)
    assert weights["eyeWideLeft"] == weights["eyeWideRight"] == 0.0
    assert weights["eyeBlinkLeft"] == weights["eyeBlinkRight"] == 1.0
    assert weights["browInnerUp"] == 0.25  # one shape, no sides
    assert weights["mouthUpperUpLeft"] == 0.25 <= fit_rest.OPENERS["mouthUpperUp"]
    assert fit_rest.weights(np.ones(len(fit_rest.PARAMS)))["mouthUpperUpLeft"] == fit_rest.OPENERS["mouthUpperUp"]
    # A face rests with at least a slight smile, whatever the search found.
    none = fit_rest.weights(np.zeros(len(fit_rest.PARAMS)))
    assert none["mouthSmileLeft"] == none["mouthSmileRight"] == fit_rest.MIN_SMILE
    assert fit_rest.weights(np.zeros(len(fit_rest.PARAMS)), min_smile=0.0)["mouthSmileLeft"] == 0.0
    assert all(name in ARKIT_52 for name in weights)
    RestSet(name="fit-a-face", weights=weights)  # valid on the wire
    # It starts from the built-in warm set.
    warm = BUILTIN["warm"].weights
    start = fit_rest.weights(np.array(list(fit_rest.PARAMS.values())))
    assert {k: v for k, v in start.items() if v} == warm


# -- how like its portrait a head looks (#169): pure numpy, the pictures and landmarks handed in --------------------

SKIN, LIPS, IRIS, HAIR = (0.86, 0.66, 0.56), (0.70, 0.27, 0.27), (0.30, 0.45, 0.32), (0.27, 0.16, 0.10)


def _ring(centre, radii, count: int, start: float, turn: int) -> np.ndarray:
    """*count* points round an ellipse, from *start* degrees (y down: 90 is the bottom), clockwise for turn 1."""
    angles = np.radians(start + turn * 360 * np.arange(count) / count)
    return np.asarray(centre) + np.asarray(radii) * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _drawn_face(
    lk,  # noqa: ANN001 - lam/likeness.py
    size: int = 512,
    eye: float = 120.0,
    at: tuple[float, float] = (256, 250),
    turned: float = 0.0,
    lips: tuple[float, float, float] = LIPS,
    lids: float = 1.0,
    grain: float = 0.0,
    soft: float = 0.0,
    seed: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """A face as a child draws one, on white, with MediaPipe's landmarks where it was drawn: (picture, points).

    *eye* is the outer eye corners' spacing; *lids* how far the eyes are open (1 open, 0 shut); *grain* fine
    texture on the skin, the same for the same *seed*; *soft* a blur over the whole picture.
    """
    points = np.tile(np.array([0.0, 0.1 * eye]), (478, 1))
    points[lk.OVAL] = _ring((0, 0.15 * eye), (0.78 * eye, 1.05 * eye), 36, -90, 1)
    points[lk.EYE_RINGS[0]] = _ring((-0.28 * eye, -0.2 * eye), (0.22 * eye, 0.09 * eye * lids), 16, 180, -1)
    points[lk.EYE_RINGS[1]] = _ring((0.28 * eye, -0.2 * eye), (0.22 * eye, 0.09 * eye * lids), 16, 0, 1)
    for (centre, rim), x in zip(lk.IRISES, (-0.28 * eye, 0.28 * eye), strict=True):
        points[centre] = (x, -0.2 * eye)
        points[rim] = _ring((x, -0.2 * eye), (0.1 * eye, 0.1 * eye), 4, 0, -1)
    points[lk.LIPS_OUTER] = _ring((0, 0.62 * eye), (0.3 * eye, 0.1 * eye), 20, 180, -1)
    points[lk.LIPS_INNER] = _ring((0, 0.62 * eye), (0.24 * eye, 0.012 * eye), 20, 180, -1)
    points[lk.fit_shape.BROWS] = _ring((0, -0.42 * eye), (0.5 * eye, 0.03 * eye), 20, 180, 1)
    points[lk.fit_shape.NOSE] = _ring((0, 0.2 * eye), (0.14 * eye, 0.2 * eye), 14, -90, 1)
    angle = np.radians(turned)
    points = points @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]).T + at

    picture = np.ones((size, size, 3))
    picture[lk.inside(lk.grown(points[lk.OVAL], 1.25), size)] = HAIR
    face = lk.inside(points[lk.OVAL], size)
    picture[face] = SKIN
    if grain:
        noise = np.random.default_rng(seed).normal(0, grain, (size, size, 1))
        picture[face] = np.clip(picture + noise, 0, 1)[face]
    ys, xs = np.mgrid[0:size, 0:size]
    for (centre, rim), ring in zip(lk.IRISES, lk.EYE_RINGS, strict=True):
        opening = lk.inside(points[ring], size)
        picture[opening] = (0.95, 0.95, 0.95)
        away = np.hypot(xs + 0.5 - points[centre, 0], ys + 0.5 - points[centre, 1])
        radius = np.linalg.norm(points[rim[0]] - points[centre])
        picture[opening & (away < radius)] = IRIS
        picture[opening & (away < 0.4 * radius)] = (0.05, 0.05, 0.05)
    picture[lk.inside(points[lk.LIPS_OUTER], size)] = lips
    return (lk.blur(picture, soft) if soft else picture), points


def test_a_head_that_is_its_portrait_scores_no_difference(lam) -> None:
    import likeness as lk  # noqa: PLC0415

    picture, points = _drawn_face(lk, grain=0.05)
    found = lk.compare(picture, points, picture, points)
    assert found["geometry"]["total"] == pytest.approx(0, abs=1e-9)
    assert found["geometry"]["aperture"] == pytest.approx(0, abs=1e-12)
    assert found["geometry"]["lip_ratio"] == pytest.approx(0, abs=1e-12)
    assert set(found["colour"]) == {"skin", "lips", "iris", "hair", "mean"}
    assert all(value == pytest.approx(0, abs=1e-9) for value in found["colour"].values())
    assert found["detail"]["kept"] == pytest.approx(1.0) and found["detail"]["same"] == pytest.approx(1.0)
    assert found["frames"]["portrait"].shape == (lk.FRAME, lk.FRAME, 3)


def test_the_two_pictures_need_not_share_a_camera(lam) -> None:
    """The head's picture is the viewer's and the portrait's is LAM's crop: another size, another place in the
    frame, a head a little tipped. Laid over by its outline, the same face scores as the same face."""
    import likeness as lk  # noqa: PLC0415

    portrait, at = _drawn_face(lk, size=512, eye=120)
    head, seen = _drawn_face(lk, size=720, eye=170, at=(380, 330), turned=6)
    found = lk.compare(portrait, at, head, seen)
    assert found["geometry"]["total"] < 1e-6 and abs(found["geometry"]["aperture"]) < 1e-6
    assert found["colour"]["mean"] < 0.5  # CIELAB: far under what an eye sees
    assert found["detail"]["same"] > 0.95
    # In the frame they are compared in, both faces' eyes are level and the same way apart.
    for frame in found["frames"].values():
        skin = np.abs(frame - SKIN).sum(axis=-1) < 0.05
        assert skin[int(lk.FRAME * 0.62), lk.FRAME // 2] and not skin[5, 5]


def test_each_colour_is_read_where_it_is(lam) -> None:
    import likeness as lk  # noqa: PLC0415

    portrait, points = _drawn_face(lk)
    paler, _ = _drawn_face(lk, lips=(0.80, 0.55, 0.52))
    found = lk.compare(portrait, points, paler, points)["colour"]
    assert found["lips"] > 20 and found["skin"] == found["iris"] == found["hair"] == 0
    assert found["mean"] == pytest.approx(found["lips"] / 4)
    # A region a picture does not show has no colour and no score: shut eyes show no iris.
    shut, shut_points = _drawn_face(lk, lids=0.0)
    assert "iris" not in lk.compare(portrait, points, shut, shut_points)["colour"]
    # The backdrop is never read as hair.
    bald = portrait.copy()
    bald[lk.inside(lk.grown(points[lk.OVAL], 1.25), 512) & ~lk.inside(points[lk.OVAL], 512)] = 1.0
    assert "hair" not in lk.compare(portrait, points, bald, points)["colour"]


def test_detail_is_what_the_head_keeps_of_the_portraits_fine_structure(lam) -> None:
    import likeness as lk  # noqa: PLC0415

    portrait, points = _drawn_face(lk, grain=0.06)
    kept = {soft: lk.compare(portrait, points, _drawn_face(lk, grain=0.06, soft=soft)[0], points)["detail"]["kept"]
            for soft in (0.0, 0.8, 2.0)}  # fmt: skip
    assert kept[0.0] == pytest.approx(1.0)
    assert kept[0.0] > kept[0.8] > kept[2.0] > 0
    # A head sharper than its portrait keeps all of it: never more than 1.
    smooth, _ = _drawn_face(lk)
    assert lk.compare(smooth, points, portrait, points)["detail"]["kept"] == 1.0


def test_the_same_features_in_the_same_places_read_as_the_same(lam) -> None:
    import likeness as lk  # noqa: PLC0415

    portrait, points = _drawn_face(lk)
    softened, _ = _drawn_face(lk, soft=1.5)
    # Softening keeps the features where they were; shifting them on the face does not.
    shifted = np.roll(portrait, 40, axis=0)
    same = lk.compare(portrait, points, softened, points)["detail"]["same"]
    other = lk.compare(portrait, points, shifted, points)["detail"]["same"]
    assert same > 0.95 > other
    # The lids show less of the iris: the aperture reads it.
    hooded, hooded_points = _drawn_face(lk, lids=0.45)
    assert lk.compare(portrait, points, hooded, hooded_points)["geometry"]["aperture"] == pytest.approx(-0.55, abs=0.01)


def test_the_rasteriser_and_cielab_are_right(lam) -> None:
    import likeness as lk  # noqa: PLC0415

    square = lk.inside(np.array([(2, 2), (6, 2), (6, 5), (2, 5)], dtype=float), 8)
    assert square.sum() == 12 and square[2:5, 2:6].all()
    assert lk.lab(np.array([1.0, 1.0, 1.0])) == pytest.approx([100, 0, 0], abs=0.01)
    assert lk.lab(np.array([1.0, 0.0, 0.0])) == pytest.approx([53.24, 80.09, 67.20], abs=0.02)  # sRGB red
    flat = np.full((16, 16, 3), 0.4)
    assert lk.blur(flat, 2.0) == pytest.approx(flat)


def test_the_calibrated_scores_are_the_ones_the_record_keeps(lam) -> None:
    """Colour, identity, lids, lips and the camera misfit; detail is left out (#169)."""
    import likeness as lk  # noqa: PLC0415

    found = lk.scores_for_record(
        {
            "colour": {"skin": 1.0, "lips": 2.0, "iris": 3.0, "hair": 1.5, "mean": 1.875},
            "shape": {"total": 0.03, "aperture": -0.1, "lip_ratio": 0.2, "oval": 0.02},
            "identity": 0.94,
            "detail": {"kept": 0.8, "same": 0.7},
        }
    )
    assert found == {
        "colour_skin": 1.0,
        "colour_lips": 2.0,
        "colour_iris": 3.0,
        "colour_hair": 1.5,
        "colour_mean": 1.875,
        "aperture": -0.1,
        "lip_ratio": 0.2,
        "shape_misfit": 0.03,
        "identity": 0.94,
    }
    assert "kept" not in found and "same" not in found and "oval" not in found


def test_refine_ply_round_trip_keeps_count_order_and_floats(lam) -> None:
    """#173: offset.ply layout is fixed; read→write must not reorder or requantise."""
    import refine_gaussians as rg  # noqa: PLC0415

    from tests.helpers import make_ply  # noqa: PLC0415

    names, attrs = rg.read_ply(make_ply(5))
    out = rg.write_ply(names, attrs)
    names2, attrs2 = rg.read_ply(out)
    assert names2 == names and len(attrs2[names2[0]]) == 5
    assert all(np.array_equal(attrs[n], attrs2[n]) for n in names)
    again = rg.ply_round_trip(out)
    assert all(np.array_equal(rg.read_ply(again)[1][n], attrs2[n]) for n in names2)


def test_refine_pull_and_zero_steps_leave_attrs_alone(lam) -> None:
    """#173: infinite pull / zero motion keeps LAM's attributes (numpy helpers, no CUDA)."""
    import refine_gaussians as rg  # noqa: PLC0415

    start = {k: np.arange(4, dtype=np.float32) * (i + 1) for i, k in enumerate(("f_dc_0", "f_dc_1", "opacity"))}
    same = {k: v.copy() for k, v in start.items()}
    assert rg.pull_loss(same, start, ("f_dc_0", "f_dc_1", "opacity")) == 0.0
    moved = {k: v + 1.0 for k, v in start.items()}
    assert rg.pull_loss(moved, start, ("f_dc_0",)) > 0.0
