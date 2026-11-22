"""bake/package.py puts a trained run's splats beside a face's mesh. It is not part of the package, so it is loaded by
path, with bake/ on the path for its `import spz`. A run it refuses, for any reason, leaves the face as it was."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from argus_mask.faces import FaceError, FaceStore, glb_parts, read_glb
from tests.helpers import make_face

BAKE = Path(__file__).resolve().parent.parent / "bake"


@pytest.fixture
def package(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(BAKE))
    monkeypatch.delitem(sys.modules, "spz", raising=False)
    spec = importlib.util.spec_from_file_location("bake_package", BAKE / "package.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(root: Path, face: Path, tris: list[float]) -> Path:
    """A trained run for *face*'s mesh: one splat per entry of *tris*, the triangle it rides."""
    run = root / "run"
    run.mkdir(exist_ok=True)
    parts = glb_parts(read_glb(face / "head.glb")[0])
    head = {"parts": [{"name": n, "count": v} for n, v, _ in parts], "triangles": sum(t for *_, t in parts)}
    (run / "head.json").write_text(json.dumps({**head, "splats": len(tris)}))
    np.array([[t, 0.25, 0.25] for t in tris], np.float32).tofile(run / "binding.bin")
    names = ["x", "y", "z", "opacity", *(f"f_dc_{c}" for c in range(3)), *(f"scale_{c}" for c in range(3))]
    names += [f"rot_{c}" for c in range(4)]
    table = np.zeros((len(tris), len(names)), "<f4")
    table[:, names.index("rot_0")] = 1.0
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(tris)}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    (run / "splats.ply").write_bytes(header.encode() + table.tobytes())
    return run


def _bake(package, monkeypatch: pytest.MonkeyPatch, run: Path, face: Path) -> None:
    argv = ["package.py", "--run", str(run), "--face", str(face), "--license", "CC0-1.0", "--sh", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    package.main()


def _state(face: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(face.iterdir())}


def test_a_run_is_packaged_into_a_face_the_server_serves(package, monkeypatch, tmp_path) -> None:
    face = make_face(tmp_path / "faces")
    _bake(package, monkeypatch, _run(tmp_path, face, [0, 0, 0]), face)
    manifest, _ = FaceStore(face.parent).load(face.name)
    assert manifest.model.splats is not None and manifest.model.splats.count == 3
    assert not [p for p in face.iterdir() if p.name.endswith(".part")]


@pytest.mark.parametrize("tri", [-1, 10**6], ids=["negative", "past-the-last"])
def test_a_binding_naming_a_triangle_the_mesh_lacks_is_refused(package, monkeypatch, tmp_path, tri) -> None:
    face = make_face(tmp_path / "faces")
    before = _state(face)
    with pytest.raises(SystemExit, match="outside"):
        _bake(package, monkeypatch, _run(tmp_path, face, [0, tri]), face)
    assert _state(face) == before


def test_a_rebake_the_server_refuses_leaves_the_face_as_it_was(package, monkeypatch, tmp_path) -> None:
    face = make_face(tmp_path / "faces")
    _bake(package, monkeypatch, _run(tmp_path, face, [0, 0, 0]), face)
    before = _state(face)

    def refuse(self, face_id, **kwargs):
        raise FaceError(f"{face_id}: refused")

    monkeypatch.setattr(package.FaceStore, "load", refuse)
    with pytest.raises(FaceError, match="refused"):
        _bake(package, monkeypatch, _run(tmp_path, face, [0, 0, 0, 0]), face)
    assert _state(face) == before
