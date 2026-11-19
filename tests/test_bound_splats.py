"""Splats bound to a mesh model's triangles (``ModelFile.splats``, #126 route A): the contract, the checks, serving."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import struct

import numpy as np
import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from argus_mask import faces
from argus_mask.config import Settings
from argus_mask.faces import (
    MANIFEST_NAME,
    SPZ_MAGIC,
    FaceError,
    FaceStore,
    inspect_spz,
    spz_point_bytes,
)
from argus_mask.models import MAX_BOUND_SPLATS, BoundSplats, Credit, FaceManifest, ModelFile, SplatPart
from argus_mask.server.app import create_app
from tests.helpers import make_face

SHA = "0" * 64
BIND = np.dtype([("tri", "<u4"), ("u", "<f4"), ("v", "<f4")])


def spz(count: int = 4, *, sh: int = 0, version: int = 3, magic: int = SPZ_MAGIC, extra: int = 0) -> bytes:
    """An SPZ whose body is zeros: the checks read the header and the length, never a splat."""
    header = struct.pack("<IIIBBBB", magic, version, count, sh, 12, 0, 0)
    return gzip.compress(header + bytes(count * spz_point_bytes(version, sh) + extra), mtime=0)


def binding(count: int = 4, *, tri: int = 0, u: float = 0.25, v: float = 0.25) -> bytes:
    rows = np.zeros(count, BIND)
    rows["tri"], rows["u"], rows["v"] = tri, u, v
    return rows.tobytes()


def splats_for(model: ModelFile, splat_bytes: bytes, binding_bytes: bytes, count: int = 4, **over: object) -> dict:
    record = {
        "file": "head.splats.spz",
        "sha256": hashlib.sha256(splat_bytes).hexdigest(),
        "binding": "head.binding.bin",
        "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
        "count": count,
        "mesh_sha256": model.sha256,
        "parts": [{"name": "head", "vertices": 3, "triangles": 1}],
        "license": "CC0-1.0",
        "inputs": [{"name": "the test head", "license": "CC0-1.0"}],
    }
    return {**record, **over}


def bound_face(tmp_path, *, splat_bytes: bytes | None = None, binding_bytes: bytes | None = None, **over: object):
    directory = make_face(tmp_path)
    splat_bytes = spz() if splat_bytes is None else splat_bytes
    binding_bytes = binding() if binding_bytes is None else binding_bytes
    (directory / "head.splats.spz").write_bytes(splat_bytes)
    (directory / "head.binding.bin").write_bytes(binding_bytes)
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    model = ModelFile.model_validate(manifest["model"])
    manifest["model"]["splats"] = splats_for(model, splat_bytes, binding_bytes, **over)
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def test_a_mesh_with_splats_is_served_and_so_are_its_files(tmp_path) -> None:
    directory = bound_face(tmp_path)
    manifest, _ = FaceStore(tmp_path).load("test-head")
    assert manifest.model.splats is not None and manifest.model.splats.count == 4
    assert manifest.model.files == ["head.glb", "head.splats.spz", "head.binding.bin"]
    client = TestClient(create_app(Settings(assets_dir=tmp_path)))
    got = client.get("/face/test-head/model/head.splats.spz")
    assert got.status_code == 200 and got.content == (directory / "head.splats.spz").read_bytes()
    assert got.headers["etag"] == f'"{manifest.model.splats.sha256}"'
    assert client.get("/face/test-head/model/head.binding.bin").content == binding()
    assert client.get("/face/test-head/model/other.spz").status_code == 404
    assert client.get("/face/test-head").json()["model"]["splats"]["inputs"][0]["license"] == "CC0-1.0"


def test_splats_ride_only_a_mesh_and_only_the_build_they_were_bound_to() -> None:
    record = splats_for(ModelFile(file="head.glb", sha256=SHA), b"", b"")
    assert ModelFile(file="head.glb", sha256=SHA, splats=record).splats is not None
    with pytest.raises(ValidationError, match="another build of the mesh"):
        ModelFile(file="head.glb", sha256="1" * 64, splats=record)
    with pytest.raises(ValidationError, match="only a mesh"):
        ModelFile(file="head.zip", sha256=SHA, format="lam-splat", splats=record)
    with pytest.raises(ValidationError):
        ModelFile(file="head.glb", sha256=SHA, splats={**record, "file": "../x.spz"})
    with pytest.raises(ValidationError):
        ModelFile(file="head.glb", sha256=SHA, splats={**record, "inputs": []})  # every input's licence, or none
    with pytest.raises(ValidationError):
        ModelFile(file="head.glb", sha256=SHA, splats={**record, "count": MAX_BOUND_SPLATS + 1})


def test_a_manifest_names_each_file_once() -> None:
    model = {"file": "head.glb", "sha256": SHA, "splats": splats_for(ModelFile(file="head.glb", sha256=SHA), b"", b"")}
    base = {"id": "x", "name": "x", "model": model, "provenance": {"kind": "stock", "source": "t", "license": "CC0"}}
    assert FaceManifest.model_validate(base)
    waist = {"model": {"file": "waist.glb", "sha256": SHA}, "framing": {}}
    assert FaceManifest.model_validate({**base, "framings": {"waist": waist}})
    clash = {**model, "splats": {**model["splats"], "binding": "waist.bin"}}
    waist_clash = {"model": {"file": "waist.glb", "sha256": SHA, "splats": clash["splats"]}, "framing": {}}
    with pytest.raises(ValidationError, match="files of their own"):  # two models' splats in one pair of files
        FaceManifest.model_validate({**base, "model": clash, "framings": {"waist": waist_clash}})


@pytest.mark.parametrize(
    ("splat_bytes", "binding_bytes", "over", "why"),
    [
        (None, binding(tri=1), {}, "rides triangle 1"),
        (None, binding(u=0.7, v=0.7), {}, "off its triangle"),
        (None, binding(u=-0.1), {}, "off its triangle"),
        (None, binding(u=float("nan")), {}, "not a number"),
        (None, binding(3), {}, "4 splats take 48"),
        (spz(5), binding(), {}, "5 splats; the manifest says 4"),
        (spz(extra=1), None, {}, "more than 4 splats"),
        (spz(extra=-1), None, {}, "less than 4 splats"),
        (spz(400)[:40], binding(400), {"count": 400}, "too short|unpack|less than"),  # the gzip stream cut off
        (b"not gzip at all", None, {}, "not gzip"),
        (spz(magic=0x12345678), None, {}, "not an SPZ"),
        (spz(version=1), None, {}, "version 1"),
        (None, None, {"parts": [{"name": "hair", "vertices": 3, "triangles": 1}]}, "parts"),
        (None, None, {"parts": [{"name": "head", "vertices": 3, "triangles": 2}]}, "parts"),
        (None, None, {"sha256": "1" * 64}, "does not match its sha256"),
    ],
)
def test_splats_that_break_a_rule_refuse_the_face(tmp_path, splat_bytes, binding_bytes, over, why) -> None:
    bound_face(tmp_path, splat_bytes=splat_bytes, binding_bytes=binding_bytes, **over)
    store = FaceStore(tmp_path)
    with pytest.raises(FaceError, match=why):
        store.load("test-head")
    faces, refused = store.scan()
    assert not faces and "splats" in refused["test-head"]
    client = TestClient(create_app(Settings(assets_dir=tmp_path)))
    assert client.get("/face/test-head/model/head.splats.spz").status_code == 404


def test_a_missing_splat_file_refuses_the_face(tmp_path) -> None:
    directory = bound_face(tmp_path)
    (directory / "head.binding.bin").unlink()
    with pytest.raises(FaceError, match="missing a file"):
        FaceStore(tmp_path).load("test-head")


def test_a_splat_file_linking_out_is_refused(tmp_path) -> None:
    directory = bound_face(tmp_path)
    outside = tmp_path / "elsewhere.spz"
    outside.write_bytes((directory / "head.splats.spz").read_bytes())
    (directory / "head.splats.spz").unlink()
    (directory / "head.splats.spz").symlink_to(outside)
    with pytest.raises(FaceError, match="not in the face's directory"):
        FaceStore(tmp_path).load("test-head")


def test_a_bomb_stops_one_byte_past_the_count(tmp_path) -> None:
    bomb = gzip.compress(struct.pack("<IIIBBBB", SPZ_MAGIC, 3, 1, 0, 12, 0, 0) + bytes(200 << 20), mtime=0)
    assert len(bomb) < 1 << 20
    with pytest.raises(FaceError, match="more than 1 splats"):
        inspect_spz(bomb)


def test_a_store_checks_splats_once_per_version(tmp_path, monkeypatch) -> None:
    directory = bound_face(tmp_path)
    store = FaceStore(tmp_path)
    calls = []

    real = faces.inspect_bound_splats
    monkeypatch.setattr(faces, "inspect_bound_splats", lambda *a: calls.append(1) or real(*a))
    store.load("test-head")
    store.load("test-head")
    assert len(calls) == 1
    (directory / "head.binding.bin").write_bytes(binding(tri=1))  # same size, new content: a new version
    st = (directory / "head.binding.bin").stat()
    os.utime(directory / "head.binding.bin", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    with pytest.raises(FaceError):
        store.load("test-head")
    assert len(calls) == 2


def test_a_same_size_copy_with_its_mtime_kept_is_checked_again(tmp_path) -> None:
    directory = bound_face(tmp_path)
    store = FaceStore(tmp_path)
    store.load("test-head")
    path = directory / "head.binding.bin"
    st = path.stat()
    replacement = tmp_path / "other.bin"
    replacement.write_bytes(binding(tri=1))  # the same size, another triangle: not what the sha256 names
    os.utime(replacement, ns=(st.st_atime_ns, st.st_mtime_ns))  # cp -p, rsync -t
    os.replace(replacement, path)
    assert (path.stat().st_mtime_ns, path.stat().st_size) == (st.st_mtime_ns, st.st_size)
    with pytest.raises(FaceError, match="sha256"):
        store.load("test-head")


@pytest.mark.parametrize(
    "doc",
    [
        {
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "accessors": [{"count": "3"}],
        },
        {"nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": [0]}]}], "accessors": []},
        {"nodes": [{"mesh": 0}], "meshes": [{"primitives": {"attributes": {}}}], "accessors": []},
        {"nodes": "x", "meshes": [], "accessors": []},
    ],
)
def test_a_malformed_gltf_is_a_face_error_when_its_parts_are_counted(doc) -> None:
    with pytest.raises(FaceError, match="well-formed"):
        faces.glb_parts(doc)


def test_a_part_and_a_credit_refuse_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SplatPart(name="head", vertices=3, triangles=1, colour="red")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Credit(name="x", license="")
    assert BoundSplats.model_validate(splats_for(ModelFile(file="h.glb", sha256=SHA), b"", b"")).triangles == 1
