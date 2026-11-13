"""LAM Gaussian-splat heads: the manifest's format, the archive's checks, and serving one like any face."""

from __future__ import annotations

import json
import stat
import struct
import warnings
import zipfile

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient
from typer.testing import CliRunner

from argus_mask import faces
from argus_mask.cli import app
from argus_mask.config import Settings
from argus_mask.faces import MANIFEST_NAME, FaceError, FaceStore, inspect_model, inspect_splat
from argus_mask.models import FaceManifest, ModelFile
from argus_mask.server.app import create_app
from tests.helpers import SPLAT_SHAPES, make_face, make_glb, make_splat_face, make_splat_zip, splat_members

SHA = "0" * 64
runner = CliRunner()


# ---- the contract ------------------------------------------------------------------------------


def test_a_model_file_says_its_format_and_agrees_with_its_name() -> None:
    assert ModelFile(file="head.glb", sha256=SHA).format == "gltf"  # a manifest from before 1.2 reads as it did
    assert ModelFile(file="c101.zip", sha256=SHA, format="lam-splat").format == "lam-splat"
    with pytest.raises(ValidationError, match="a gltf model is a .glb file"):
        ModelFile(file="c101.zip", sha256=SHA)
    with pytest.raises(ValidationError, match="a lam-splat model is a .zip file"):
        ModelFile(file="head.glb", sha256=SHA, format="lam-splat")
    with pytest.raises(ValidationError):
        ModelFile(file="head.ply", sha256=SHA)
    with pytest.raises(ValidationError):
        ModelFile(file="head.zip", sha256=SHA, format="usdz")  # type: ignore[arg-type]


def test_a_1_1_manifest_still_reads() -> None:
    old = {
        "mask_version": "1.1",
        "id": "old-head",
        "name": "Old head",
        "model": {"file": "head.glb", "sha256": SHA},
        "provenance": {"kind": "personal", "source": "made before splat heads"},
    }
    manifest = FaceManifest.model_validate(old)
    assert manifest.model.format == "gltf"


# ---- the archive -------------------------------------------------------------------------------


def test_a_splat_archive_is_read_in_memory(tmp_path) -> None:
    report = inspect_splat(make_splat_zip(tmp_path / "head.zip"))
    assert report.folder == "avatar" and report.splats == 3
    assert set(report.members) == {"skin.glb", "offset.ply", "animation.glb", "vertex_order.json"}
    assert report.present == [n for n in faces.ARKIT_52 if n in SPLAT_SHAPES]
    assert report.extra == [] and len(report.missing) == 52 - len(SPLAT_SHAPES)
    assert report.warnings == [f"{52 - len(SPLAT_SHAPES)} of the 52 ARKit shapes are missing; they will be ignored"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["head.zip"]  # nothing unpacked beside it
    assert inspect_model(tmp_path / "head.zip").folder == "avatar"


def test_the_optional_iris_file_is_taken(tmp_path) -> None:
    members = splat_members() | {"iris_occlusion.json": json.dumps({"left_iris": [[1, 2]]}).encode()}
    assert "iris_occlusion.json" in inspect_splat(make_splat_zip(tmp_path / "head.zip", members)).members


def link(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


GOOD = splat_members()
WITHOUT_PLY = {k: v for k, v in GOOD.items() if k != "offset.ply"}


@pytest.mark.parametrize(
    ("members", "entries", "why"),
    [
        (None, [("avatar/../evil.glb", b"x")], "climbs out of its folder"),
        (None, [("../evil.glb", b"x")], "climbs out of its folder"),
        (None, [("/etc/evil", b"x")], "absolute path"),
        (None, [("C:/evil", b"x")], "absolute path"),
        (None, [("avatar\\evil.json", b"x")], "backslash"),
        (None, [("avatar//skin2.glb", b"x")], "empty part"),
        (None, [(link("avatar/iris_occlusion.json"), b"/etc/passwd")], "a link or a special file"),
        (None, [("avatar/skin.glb", make_glb())], "appears twice"),
        (WITHOUT_PLY, [], "missing offset.ply"),
        (None, [("avatar/notes.txt", b"hello")], "is not part of a LAM head"),
        (None, [("avatar/.DS_Store", b"x")], "is not part of a LAM head"),
        (None, [("other/skin.glb", make_glb())], "a LAM head is one folder"),
        (None, [("readme.txt", b"x")], "not a file directly in the head's folder"),
        (None, [("avatar/sub/", b"")], "a folder inside the head's folder"),
        (None, [("avatar/sub/skin.glb", b"x")], "not a file directly in the head's folder"),
        (GOOD | {"offset.ply": b"not a ply at all"}, [], "offset.ply is not a PLY"),
        (GOOD | {"vertex_order.json": b"[1, 2"}, [], "vertex_order.json is not JSON"),
        (GOOD | {"skin.glb": b"glTF but not really"}, [], "skin.glb: too short to be a .glb"),
        (GOOD | {"skin.glb": make_glb(["jawOpen", "myOwnShape"])}, [], "not ARKit shapes: myOwnShape"),
        (GOOD | {"skin.glb": make_glb(())}, [], "no ARKit morph targets"),
        # A zip bomb's tell: far more out than in (a megabyte of zeros packs to about a kilobyte).
        (GOOD | {"offset.ply": b"ply\nend_header\n" + bytes(1 << 20)}, [], "over 100 times its packed size"),
    ],
)
def test_an_archive_that_breaks_a_rule_is_refused(tmp_path, members, entries, why) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zipfile warns of the duplicate name it is asked to write
        path = make_splat_zip(tmp_path / "head.zip", members, entries=entries)
    with pytest.raises(FaceError, match=why.replace("(", r"\(").replace(".", r"\.")):
        inspect_splat(path)


def test_the_folder_must_be_a_plain_name(tmp_path) -> None:
    with pytest.raises(FaceError, match="not a plain name"):
        inspect_splat(make_splat_zip(tmp_path / "head.zip", folder="-avatar"))


def test_size_caps_hold_for_each_member_all_of_them_and_the_count(tmp_path, monkeypatch) -> None:
    path = make_splat_zip(tmp_path / "head.zip")
    monkeypatch.setattr(faces, "SPLAT_MAX_MEMBER", 1000)
    with pytest.raises(FaceError, match="skin.glb unpacks to over"):
        inspect_splat(path)
    monkeypatch.setattr(faces, "SPLAT_MAX_MEMBER", 64 << 20)
    monkeypatch.setattr(faces, "SPLAT_MAX_TOTAL", 2000)
    with pytest.raises(FaceError, match="unpack to over 0 MiB together"):
        inspect_splat(path)
    monkeypatch.setattr(faces, "SPLAT_MAX_TOTAL", 4000)
    big = make_splat_zip(tmp_path / "big.zip", GOOD | {"vertex_order.json": json.dumps(list(range(2000))).encode()})
    with pytest.raises(FaceError, match="the archive is over"):
        inspect_splat(big)
    monkeypatch.setattr(faces, "SPLAT_MAX_TOTAL", 160 << 20)
    monkeypatch.setattr(faces, "SPLAT_MAX_MEMBERS", 4)
    with pytest.raises(FaceError, match="5 members; a LAM head has at most 4"):
        inspect_splat(path)


def test_an_encrypted_member_is_refused(tmp_path) -> None:
    # zipfile writes no encrypted members: set the flag in the central directory, which is what a reader trusts.
    path = make_splat_zip(tmp_path / "head.zip", GOOD | {"iris_occlusion.json": b"{}"})
    data = bytearray(path.read_bytes())
    at = data.find(b"PK\x01\x02")
    while at >= 0:
        size = int.from_bytes(data[at + 28 : at + 30], "little")
        if data[at + 46 : at + 46 + size] == b"avatar/iris_occlusion.json":
            data[at + 8] |= 0x1
        at = data.find(b"PK\x01\x02", at + 46)
    path.write_bytes(bytes(data))
    with pytest.raises(FaceError, match="iris_occlusion.json is encrypted"):
        inspect_splat(path)


def test_not_a_zip_is_refused(tmp_path) -> None:
    (tmp_path / "head.zip").write_bytes(b"PK but not really a zip")
    with pytest.raises(FaceError, match="not a zip archive"):
        inspect_splat(tmp_path / "head.zip")


def glb_of(doc: object) -> bytes:
    """A .glb whose JSON chunk is *doc*, whatever its shape."""
    body = json.dumps(doc).encode()
    body += b" " * (-len(body) % 4)
    return struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(body)) + struct.pack("<II", len(body), 0x4E4F534A) + body


@pytest.mark.parametrize(
    "doc",
    [{"meshes": [1, 2]}, {"meshes": [{"primitives": [{"indices": 0}]}], "accessors": [7]}, {"images": ["x"]}],
)
def test_a_skin_glb_not_shaped_like_gltf_is_refused_not_raised(tmp_path, doc) -> None:
    # The archive is an upload: a crash here would come out of FaceStore.scan() and break every listing.
    with pytest.raises(FaceError, match="not a well-formed glTF"):
        inspect_splat(make_splat_zip(tmp_path / "head.zip", splat_members() | {"skin.glb": glb_of(doc)}))


def test_json_nested_past_the_stack_is_refused(tmp_path) -> None:
    deep = b"[" * 200_000 + b"]" * 200_000
    members = splat_members()
    del members["vertex_order.json"]
    stored = zipfile.ZipInfo("avatar/vertex_order.json")  # stored: deflated, it would trip the ratio cap first
    with pytest.raises(FaceError, match="vertex_order.json is not JSON"):
        inspect_splat(make_splat_zip(tmp_path / "head.zip", members, entries=[(stored, deep)]))


# ---- a splat face, served -------------------------------------------------------------------------


def test_the_store_lists_a_splat_face_and_refuses_a_bad_archive(tmp_path) -> None:
    make_face(tmp_path, "mesh-head")
    make_splat_face(tmp_path, "splat-head")
    bad = make_splat_face(tmp_path, "bad-splat")
    model = bad / "head.zip"
    make_splat_zip(model, entries=[("avatar/notes.txt", b"hello")])
    manifest = json.loads((bad / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["model"]["sha256"] = faces.sha256_file(model)
    (bad / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    accepted, refused = FaceStore(tmp_path).scan()
    assert [(m.id, m.model.format) for m in accepted] == [("mesh-head", "gltf"), ("splat-head", "lam-splat")]
    assert refused == {"bad-splat": "avatar/notes.txt is not part of a LAM head"}
    splat = accepted[1]
    assert splat.model.file == "head.zip" and splat.blendshapes == [n for n in faces.ARKIT_52 if n in SPLAT_SHAPES]
    assert splat.framing.height == 1.65  # LAM's heads stand at about 1.65 m


def test_a_manifest_naming_shapes_the_skin_lacks_is_refused(tmp_path) -> None:
    directory = make_splat_face(tmp_path, "splat-head")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["blendshapes"].append("tongueOut")
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FaceError, match="the head's skin.glb does not have: tongueOut"):
        FaceStore(tmp_path).load("splat-head")


def test_the_server_serves_the_archive_by_its_own_name_too(tmp_path) -> None:
    make_splat_face(tmp_path, "splat-head")
    make_face(tmp_path, "mesh-head")
    data = (tmp_path / "splat-head" / "head.zip").read_bytes()
    with TestClient(create_app(Settings(assets_dir=tmp_path))) as client:
        listed = {f["id"]: f["model"]["format"] for f in client.get("/faces").json()}
        assert listed == {"mesh-head": "gltf", "splat-head": "lam-splat"}
        for path in ("/face/splat-head/model", "/face/splat-head/model/head.zip"):
            reply = client.get(path)
            assert reply.status_code == 200 and reply.content == data
            assert reply.headers["content-type"] == "application/zip"
        mesh = client.get("/face/mesh-head/model/head.glb")
        assert mesh.status_code == 200 and mesh.headers["content-type"] == "model/gltf-binary"
        assert client.get("/face/splat-head/model/other.zip").status_code == 404
        assert client.get("/face/splat-head/model/manifest.json").status_code == 404
        assert client.get("/face/splat-head/model/..%2Fmesh-head%2Fhead.glb").status_code == 404
        assert client.get("/splat.js").status_code == 200
        assert client.get("/vendor/jszip.js").status_code == 200


def test_the_editor_serves_a_splat_face_for_its_previews(tmp_path) -> None:
    from argus_mask.config import EditorSettings
    from argus_mask.editor.app import create_editor_app

    make_splat_face(tmp_path / "assets", "splat-head")
    editor = EditorSettings(blender="blender", blender_user=tmp_path / "user", mpfb_data=tmp_path / "data")
    with TestClient(create_editor_app(Settings(assets_dir=tmp_path / "assets"), editor)) as client:
        reply = client.get("/face/splat-head/model/head.zip?from=export")
        assert reply.status_code == 200 and reply.headers["content-type"] == "application/zip"
        assert client.get("/face/splat-head/model/x.zip?from=export").status_code == 404
        assert client.get("/splat.js").status_code == 200


def test_inspect_writes_a_manifest_for_a_splat_archive(tmp_path) -> None:
    directory = tmp_path / "my-splat"
    directory.mkdir()
    make_splat_zip(directory / "avatar.zip")
    written = runner.invoke(
        app, ["inspect", str(directory / "avatar.zip"), "--write-manifest", "--source", "a LAM head, made in a test"]
    )
    assert written.exit_code == 0, written.output
    assert "splat head avatar/" in written.output and "3 Gaussians" in written.output
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["model"] == {
        "file": "avatar.zip",
        "sha256": faces.sha256_file(directory / "avatar.zip"),
        "format": "lam-splat",
    }
    assert manifest["framing"] == {"distance": 0.7, "height": 1.65, "fov": 50.0}
    assert manifest["provenance"]["kind"] == "personal" and manifest["mask_version"] == "1.2"
    report = json.loads(runner.invoke(app, ["inspect", str(directory), "--json"]).output)
    assert report["format"] == "lam-splat" and report["refused"] is None and report["splats"] == 3
    assert report["manifest"]["id"] == "my-splat"
    # A directory whose manifest no longer matches still names the one archive it holds.
    make_splat_zip(directory / "avatar.zip", GOOD | {"vertex_order.json": b"[2, 1, 0]"})
    changed = runner.invoke(app, ["inspect", str(directory)])
    assert changed.exit_code == 1 and "sha256 does not match" in changed.output and "avatar.zip" in changed.output


def test_a_member_that_lies_about_its_size_is_refused(tmp_path) -> None:
    # The sizes in a zip's headers are its own claim: what unpacks is counted, and checked against the CRC.
    path = make_splat_zip(tmp_path / "head.zip", GOOD | {"vertex_order.json": json.dumps(list(range(500))).encode()})
    data = bytearray(path.read_bytes())
    at = data.find(b"PK\x01\x02")
    while at >= 0:
        size = int.from_bytes(data[at + 28 : at + 30], "little")
        if data[at + 46 : at + 46 + size] == b"avatar/vertex_order.json":
            data[at + 24 : at + 28] = (10).to_bytes(4, "little")  # uncompressed size: 10 bytes, it says
        at = data.find(b"PK\x01\x02", at + 46)
    path.write_bytes(bytes(data))
    with pytest.raises(FaceError, match="vertex_order.json (does not unpack|unpacks to more than it says)"):
        inspect_splat(path)
