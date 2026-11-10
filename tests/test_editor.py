from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.builder import SCRIPT, Builder, BuildError, Busy
from argus_mask.editor.library import AssetLibrary
from argus_mask.editor.preset import ADULT_AGE, MODIFIER_GROUPS, MODIFIERS, Preset, one_way
from argus_mask.faces import FaceStore
from argus_mask.server.app import create_app
from tests.helpers import make_face, make_glb

REPO = Path(__file__).resolve().parent.parent
STOCK_PRESETS = sorted((REPO / "scripts" / "stock_heads").glob("stock-*[0-9].json"))

# Stands in for Blender: reads the argv build_head.py would get, and does what FAKE_BLENDER says.
FAKE_BLENDER = """#!{python}
import json, os, pathlib, sys, time
argv = sys.argv[1:]
log = os.environ.get("FAKE_BLENDER_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(json.dumps({{"argv": argv, "user": os.environ.get("BLENDER_USER_RESOURCES")}}) + "\\n")
mode = os.environ.get("FAKE_BLENDER", "ok")
rest = argv[argv.index("--") + 1:]
out = pathlib.Path(rest[rest.index("--out") + 1])
if mode == "refuse":
    print("build_head: phenotype.age is too young", file=sys.stderr); sys.exit(2)
if mode == "crash":
    print("Traceback: something broke", file=sys.stderr); sys.exit(3)
if mode == "sleep":
    time.sleep(30)
if mode == "silent":
    sys.exit(0)
out.write_bytes(pathlib.Path(os.environ["FAKE_BLENDER_GLB"]).read_bytes())
recipe = {{"blender": "4.5.15 LTS", "mpfb": "2.0.18", "triangles_total": 1, "bytes": out.stat().st_size}}
out.with_suffix(".recipe.json").write_text(json.dumps(recipe))
"""

PACK = {
    "s1": {"type": "skins", "license": "CC0", "author": "makehuman_system"},
    "e1": {"type": "eyes", "license": "CC0", "author": "makehuman_system"},
    "b1": {"type": "eyebrows", "license": "CC0", "author": "makehuman_system"},
    "l1": {"type": "eyelashes", "license": "CC0", "author": "makehuman_system"},
    "t1": {"type": "teeth", "license": "CC0", "author": "makehuman_system"},
    "g1": {"type": "tongue", "license": "CC0", "author": "makehuman_system"},
    "h1": {"type": "hair", "license": "CC0", "author": "makehuman_system"},
}
BY_PACK = {"h2": {"type": "hair", "license": "CC-BY 4.0", "author": "Some Artist"}}
CLOTHES_PACK = {
    "top1": {"type": "clothes", "license": "CC0", "author": "makehuman_system"},
    "hat1": {"type": "clothes", "license": "CC0", "author": "makehuman_system"},
    "shoes1": {"type": "clothes", "license": "CC0", "author": "makehuman_system"},
}
CLOTHES_TAGS = {"top1": ["Female", "Casual"], "hat1": ["Hats"], "shoes1": ["Shoes"]}


def make_library(root: Path) -> Path:
    """An MPFB data directory: licensed assets, one CC-BY hair, one unlicensed hair, and the face units."""
    data = root / "mpfb" / "data"
    dirs = {"skins": ["s1"], "eyes": ["e1"], "eyebrows": ["b1"], "eyelashes": ["l1"], "teeth": ["t1"]}
    dirs |= {"tongue": ["g1"], "hair": ["h1", "h2", "rogue"]}
    for directory, names in dirs.items():
        for name in names:
            sub = data / directory / name
            sub.mkdir(parents=True)
            (sub / f"{name}{'.mhmat' if directory == 'skins' else '.mhclo'}").write_text("# asset\n")
            (sub / f"{name}.thumb").write_bytes(b"\x89PNG\r\n\x1a\nthumb")
    for name, tags in CLOTHES_TAGS.items():
        sub = data / "clothes" / name
        sub.mkdir(parents=True)
        (sub / f"{name}.mhclo").write_text("".join(f"tag {t}\n" for t in tags) + f"name {name}\n")
        (sub / f"{name}.thumb").write_bytes(b"\x89PNG\r\n\x1a\nthumb")
    (data / "eyes" / "materials").mkdir()  # eye colours: not an asset of the eyes part
    (data / "packs").mkdir()
    (data / "packs" / "system.json").write_text(json.dumps(PACK))
    (data / "packs" / "by.json").write_text(json.dumps(BY_PACK))
    (data / "packs" / "clothes.json").write_text(json.dumps(CLOTHES_PACK))
    (data / "packs" / "faceunits01.json").write_text(json.dumps({"jawOpen": {"type": "target", "license": "CC0"}}))
    return data


def preset(**changes) -> dict:
    body = {
        "id": "my-face",
        "name": "My face",
        "phenotype": {
            "gender": 0.0,
            "age": 0.5,
            "muscle": 0.5,
            "weight": 0.5,
            "proportions": 0.5,
            "height": 0.5,
            "cupsize": 0.5,
            "firmness": 0.5,
            "race": {"asian": 0.2, "caucasian": 0.6, "african": 0.2},
        },
        "assets": {
            "skin": "skins/s1/s1.mhmat",
            "eyes": "eyes/e1/e1.mhclo",
            "eyebrows": "eyebrows/b1/b1.mhclo",
            "eyelashes": "eyelashes/l1/l1.mhclo",
            "teeth": "teeth/t1/t1.mhclo",
            "tongue": "tongue/g1/g1.mhclo",
            "hair": "hair/h1/h1.mhclo",
            "clothes": ["clothes/top1/top1.mhclo"],
        },
        "mouth": {"tongue_flatten": 0.6, "upper_teeth_drop_mm": 2.5, "cavity_shade": 0.02},
    }
    body.update(changes)
    return body


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A fake Blender, an asset library and an empty assets root."""
    blender = tmp_path / "blender"
    blender.write_text(FAKE_BLENDER.format(python=sys.executable))
    blender.chmod(blender.stat().st_mode | stat.S_IXUSR)
    glb = tmp_path / "fixture.glb"
    glb.write_bytes(make_glb())
    log = tmp_path / "blender.log"
    monkeypatch.setenv("FAKE_BLENDER_GLB", str(glb))
    monkeypatch.setenv("FAKE_BLENDER_LOG", str(log))
    monkeypatch.setenv("FAKE_BLENDER", "ok")
    user = tmp_path / "user"
    data = make_library(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    editor = EditorSettings(blender=str(blender), blender_user=user, mpfb_data=data, build_timeout_s=20)
    return {"editor": editor, "assets": assets, "log": log, "data": data}


@pytest.fixture
def client(rig):
    from argus_mask.editor.app import create_editor_app

    with TestClient(create_editor_app(Settings(assets_dir=rig["assets"]), rig["editor"])) as c:
        yield c


def calls(rig) -> list[dict]:
    log: Path = rig["log"]
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


def leftovers(assets: Path) -> list[str]:
    """Half-made builds or exports: dot-directories other than the preview area, in either place."""
    found = [p.name for p in assets.iterdir() if p.name.startswith(".") and p.name != ".preview"]
    preview = assets / ".preview"
    return found + ([p.name for p in preview.iterdir() if p.name.startswith(".")] if preview.is_dir() else [])


def nothing_built(assets: Path) -> bool:
    preview = assets / ".preview"
    return [p.name for p in assets.iterdir() if p.name != ".preview"] == [] and (
        not preview.is_dir() or list(preview.iterdir()) == []
    )


def served(assets: Path) -> list[str]:
    with TestClient(create_app(Settings(assets_dir=assets))) as server:
        return [f["id"] for f in server.get("/faces").json()]


# ---- the preset ----------------------------------------------------------------------------


@pytest.mark.parametrize("path", STOCK_PRESETS, ids=lambda p: p.stem)
def test_the_stock_presets_are_valid_presets(path) -> None:
    Preset.model_validate_json(path.read_text())


def test_the_editor_and_the_recipe_agree_on_the_adult_line() -> None:
    tree = ast.parse(SCRIPT.read_text())
    values = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ADULT_AGE" for t in node.targets)
    ]
    assert values == [ADULT_AGE]


def test_the_target_catalogue_is_well_formed() -> None:
    names = list(MODIFIERS)
    assert len(names) == len(set(names)) > 100
    for m in MODIFIERS.values():
        assert m["positive"] and m["group"] in MODIFIER_GROUPS and m["label"]
        for fragment in m["negative"] + m["positive"]:
            group, stem = fragment.split("/")
            assert group == m["group"] and re.fullmatch(r"[a-z0-9][a-z0-9-]*", stem)
    assert one_way("head-square") and not one_way("chin-width")


def test_the_recipe_reads_the_same_catalogue() -> None:
    assert '"modifiers.json"' in SCRIPT.read_text()
    assert (SCRIPT.parent / "modifiers.json").is_file()


def test_targets_are_named_from_the_list_and_bounded() -> None:
    assert Preset.model_validate(preset(targets={"mouth-upperlip-volume": 0.4})).targets == {
        "mouth-upperlip-volume": 0.4
    }
    assert (
        Preset.model_validate(preset(targets={"head-square": 0.7, "nose-point-down-up": -0.3})).targets["head-square"]
        == 0.7
    )
    for bad in [
        {"mouth-upperlip-volume": 1.5},
        {"../../etc/passwd": 0.5},
        {"mouth-upperlip-volume-incr": 0.5},
        {"head-square": -0.5},  # one-way: 0 to 1
    ]:
        with pytest.raises(ValidationError):
            Preset.model_validate(preset(targets=bad))


def test_a_preset_under_18_is_refused() -> None:
    body = preset()
    body["phenotype"]["age"] = 0.35  # about 17.8 years on MakeHuman's scale
    with pytest.raises(ValidationError, match="age"):
        Preset.model_validate(body)


def test_the_age_floor_is_18_years() -> None:
    body = preset()
    body["phenotype"]["age"] = ADULT_AGE
    assert Preset.model_validate(body).age_years() == 18
    body["phenotype"]["age"] = 0.5
    assert Preset.model_validate(body).age_years() == 25
    assert 1 + 48 * ADULT_AGE >= 18 > 1 + 48 * (ADULT_AGE - 0.0001)


@pytest.mark.parametrize("clothes", [[], None])
def test_a_preset_with_no_clothes_is_refused(clothes) -> None:
    body = preset()
    if clothes is None:
        del body["assets"]["clothes"]
    else:
        body["assets"]["clothes"] = clothes
    with pytest.raises(ValidationError, match="clothes"):
        Preset.model_validate(body)


def test_a_hat_alone_never_reaches_blender(client, rig) -> None:
    body = preset()
    body["assets"]["clothes"] = ["clothes/hat1/hat1.mhclo"]
    reply = client.post("/build", json=body)
    assert reply.status_code == 422 and "outfit" in reply.json()["detail"]
    assert calls(rig) == []


@pytest.mark.parametrize(
    "ref", ["skins/../../etc/passwd.mhmat", "/etc/s1.mhmat", "skins/s1/s1.blend", "skins/.hidden/x.mhmat"]
)
def test_an_asset_reference_is_a_plain_name_under_a_part(ref) -> None:
    body = preset()
    body["assets"]["skin"] = ref
    with pytest.raises(ValidationError):
        Preset.model_validate(body)


def test_a_preset_has_no_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Preset.model_validate(preset(script="import os"))


# ---- the library ---------------------------------------------------------------------------


def test_the_library_offers_only_assets_with_a_stated_licence(tmp_path) -> None:
    library = AssetLibrary(make_library(tmp_path))
    found = library.scan()
    assert [a.name for a in found["hair"]] == ["h1", "h2"]  # "rogue" is in no pack
    assert [a.name for a in found["eyes"]] == ["e1"]  # eyes/materials is not an eye mesh
    assert found["skin"][0].ref == "skins/s1/s1.mhmat" and found["skin"][0].license == "CC0-1.0"
    assert found["hair"][1].license == "CC-BY 4.0" and found["hair"][1].author == "Some Artist"
    assert library.has_faceunits()


def test_clothes_are_outfits_and_hats_and_never_shoes(tmp_path) -> None:
    found = AssetLibrary(make_library(tmp_path)).scan()["clothes"]
    assert [(a.name, a.slot) for a in found] == [("hat1", "hat"), ("top1", "outfit")]
    assert found[1].ref == "clothes/top1/top1.mhclo" and found[1].public()["slot"] == "outfit"


def test_clothes_are_each_listed_once_and_few() -> None:
    body = preset()
    body["assets"]["clothes"] = ["clothes/top1/top1.mhclo"] * 2
    with pytest.raises(ValidationError, match="twice"):
        Preset.model_validate(body)
    body["assets"]["clothes"] = [f"clothes/c{i}/c{i}.mhclo" for i in range(4)]
    with pytest.raises(ValidationError):
        Preset.model_validate(body)


def test_the_library_scan_is_reused_until_an_asset_is_installed(tmp_path) -> None:
    data = make_library(tmp_path)
    library = AssetLibrary(data)
    first = library.scan()
    assert library.scan() is first  # one scan serves every thumbnail
    sub = data / "hair" / "h3"
    sub.mkdir()
    (sub / "h3.mhclo").write_text("# asset\n")
    (data / "packs" / "more.json").write_text(json.dumps({"h3": {"type": "hair", "license": "CC0"}}))
    assert [a.name for a in library.scan()["hair"]] == ["h1", "h2", "h3"]


def test_an_unconfigured_library_is_empty() -> None:
    library = AssetLibrary(None)
    assert all(v == [] for v in library.scan().values()) and not library.has_faceunits()


# ---- building ------------------------------------------------------------------------------


def test_a_build_is_a_preview_and_export_makes_it_a_face(client, rig) -> None:
    reply = client.post("/build", json=preset())
    assert reply.status_code == 200, reply.text
    manifest = reply.json()["manifest"]
    assert manifest["id"] == "my-face" and manifest["provenance"]["kind"] == "synthetic"
    assert manifest["provenance"]["license"] == "CC0-1.0"
    assert "hair h1" in manifest["provenance"]["notes"] and "not a real person" in manifest["provenance"]["notes"]
    assert len(manifest["blendshapes"]) == 52

    files = ["head.glb", "head.recipe.json", "manifest.json", "preset.json"]
    assert sorted(p.name for p in (rig["assets"] / ".preview" / "my-face").iterdir()) == files
    assert leftovers(rig["assets"]) == []

    # Blender got the recipe, the user directory and a non-zero exit on a Python error.
    [call] = calls(rig)
    assert call["argv"][:5] == ["-b", "--python-exit-code", "3", "-P", str(SCRIPT)]
    assert call["user"] == str(rig["editor"].blender_user)

    # A preview is the editor's, not the server's.
    assert served(rig["assets"]) == []
    assert client.get("/face/my-face/model").content[:4] == b"glTF"
    assert client.get("/presets").json() == [
        {"preset": client.get("/presets").json()[0]["preset"], "exported": False, "previewed": True}
    ]

    reply = client.post("/export/my-face")
    assert reply.status_code == 200, reply.text
    assert reply.json()["manifest"]["model"]["sha256"] == manifest["model"]["sha256"]
    assert sorted(p.name for p in (rig["assets"] / "my-face").iterdir()) == files
    assert served(rig["assets"]) == ["my-face"]
    [entry] = client.get("/presets").json()
    assert entry["exported"] and entry["previewed"] and entry["preset"]["id"] == "my-face"
    assert leftovers(rig["assets"]) == []


def test_export_again_replaces_the_editors_own_export(client, rig) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/export/my-face").status_code == 200
    assert client.post("/build", json=preset(name="Renamed")).status_code == 200
    assert FaceStore(rig["assets"]).load("my-face")[0].name == "My face"  # a rebuild leaves the export alone
    assert client.post("/export/my-face").status_code == 200
    assert FaceStore(rig["assets"]).load("my-face")[0].name == "Renamed"
    assert leftovers(rig["assets"]) == []


@pytest.mark.parametrize("face_id", ["nothing-here", "..", "Bad Id"])
def test_export_needs_a_finished_preview(client, rig, face_id) -> None:
    # ".." never reaches the route: the client resolves /export/.. to /, which only takes GET.
    assert client.post(f"/export/{face_id}").status_code in (404, 405)
    assert nothing_built(rig["assets"])


def test_export_never_replaces_a_face_the_editor_did_not_make(client, rig) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    make_face(rig["assets"], "my-face")  # put there by hand after the build
    before = (rig["assets"] / "my-face" / "manifest.json").read_text()
    reply = client.post("/export/my-face")
    assert reply.status_code == 409 and "did not make" in reply.json()["detail"]
    assert (rig["assets"] / "my-face" / "manifest.json").read_text() == before
    assert leftovers(rig["assets"]) == []


def test_export_is_a_guarded_write(client, rig) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/export/my-face", headers={"Origin": "https://elsewhere.example"}).status_code == 403
    assert served(rig["assets"]) == []


def test_a_cc_by_asset_is_in_the_licence(client) -> None:
    body = preset()
    body["assets"]["hair"] = "hair/h2/h2.mhclo"
    manifest = client.post("/build", json=body).json()["manifest"]
    assert manifest["provenance"]["license"] == "CC-BY 4.0 AND CC0-1.0"
    assert "Some Artist" in manifest["provenance"]["notes"]


def test_clothes_go_into_the_build_and_the_credits(client, rig) -> None:
    body = preset()
    body["assets"]["clothes"] = ["clothes/top1/top1.mhclo", "clothes/hat1/hat1.mhclo"]
    reply = client.post("/build", json=body)
    assert reply.status_code == 200, reply.text
    notes = reply.json()["manifest"]["provenance"]["notes"]
    assert "clothes top1 (clothes, makehuman_system, CC0-1.0)" in notes and "clothes hat1" in notes
    sent = json.loads((rig["assets"] / ".preview" / "my-face" / "preset.json").read_text())
    assert sent["assets"]["clothes"] == body["assets"]["clothes"]


@pytest.mark.parametrize("ref", ["clothes/shoes1/shoes1.mhclo", "clothes/nothere/nothere.mhclo", "hair/h1/h1.mhclo"])
def test_clothes_not_in_the_library_never_reach_blender(client, rig, ref) -> None:
    body = preset()
    body["assets"]["clothes"] = [ref]
    reply = client.post("/build", json=body)
    assert reply.status_code == 422 and "clothes" in reply.json()["detail"]
    assert calls(rig) == []


def test_targets_reach_the_recipe_and_the_credits(client, rig) -> None:
    body = preset(targets={"mouth-upperlip-volume": 0.4, "mouth-angles-down-up": -0.25})
    manifest = client.post("/build", json=body).json()["manifest"]
    assert "mouth-angles-down-up -0.25, mouth-upperlip-volume +0.40" in manifest["provenance"]["notes"]
    sent = json.loads((rig["assets"] / ".preview" / "my-face" / "preset.json").read_text())
    assert sent["targets"] == body["targets"]


def test_hair_is_optional(client) -> None:
    body = preset()
    del body["assets"]["hair"]
    assert client.post("/build", json=body).status_code == 200


@pytest.mark.parametrize("ref", ["hair/rogue/rogue.mhclo", "hair/nothere/nothere.mhclo", "skins/s1/s1.mhmat"])
def test_an_asset_not_in_the_library_never_reaches_blender(client, rig, ref) -> None:
    body = preset()
    body["assets"]["hair"] = ref
    reply = client.post("/build", json=body)
    assert reply.status_code == 422 and "hair" in reply.json()["detail"]
    assert calls(rig) == []


def test_the_recipe_refusing_is_reported_and_leaves_nothing(client, rig, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_BLENDER", "refuse")
    reply = client.post("/build", json=preset())
    assert reply.status_code == 422 and reply.json()["detail"] == "phenotype.age is too young"
    assert nothing_built(rig["assets"])


@pytest.mark.parametrize("mode", ["crash", "silent"])
def test_a_failed_build_leaves_nothing(client, rig, monkeypatch, mode) -> None:
    monkeypatch.setenv("FAKE_BLENDER", mode)
    reply = client.post("/build", json=preset())
    assert reply.status_code == 500
    assert nothing_built(rig["assets"])


def test_a_failed_rebuild_keeps_the_face_that_was_there(client, rig, monkeypatch) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    monkeypatch.setenv("FAKE_BLENDER", "crash")
    assert client.post("/build", json=preset()).status_code == 500
    assert FaceStore(rig["assets"] / ".preview").load("my-face")[0].id == "my-face"
    assert leftovers(rig["assets"]) == []


def test_a_rebuild_replaces_the_preview(client, rig) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    reply = client.post("/build", json=preset(name="Renamed"))
    assert reply.status_code == 200 and reply.json()["manifest"]["name"] == "Renamed"
    assert FaceStore(rig["assets"] / ".preview").load("my-face")[0].name == "Renamed"
    assert leftovers(rig["assets"]) == []


def test_the_editor_never_replaces_a_face_it_did_not_make(client, rig) -> None:
    make_face(rig["assets"], "stock-01")
    before = (rig["assets"] / "stock-01" / "manifest.json").read_text()
    reply = client.post("/build", json=preset(id="stock-01"))
    assert reply.status_code == 409 and "did not make" in reply.json()["detail"]
    assert (rig["assets"] / "stock-01" / "manifest.json").read_text() == before
    assert calls(rig) == []
    # It may show one, read-only, for review: the same face the server serves.
    assert client.get("/face/stock-01").json()["id"] == "stock-01"


# ---- saved faces --------------------------------------------------------------------------


def test_saved_lists_previews_exports_and_other_faces(client, rig) -> None:
    make_face(rig["assets"], "stock-01")
    make_face(rig["assets"], "anonymous", provenance=None)
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/build", json=preset(id="other")).status_code == 200
    assert client.post("/export/other").status_code == 200
    rows = {r["id"]: r for r in client.get("/saved").json()}
    assert set(rows) == {"my-face", "other", "stock-01", "anonymous"}
    assert rows["my-face"]["preview"]["editable"] and rows["my-face"]["export"] is None
    assert rows["other"]["preview"] and rows["other"]["export"]["editable"]
    assert rows["stock-01"]["export"]["kind"] == "stock" and not rows["stock-01"]["export"]["editable"]
    assert "provenance" in rows["anonymous"]["refused"]
    assert client.get("/saved/other/preset?from=export").json()["id"] == "other"
    assert client.get("/saved/stock-01/preset?from=export").status_code == 404


def test_rename_moves_the_export_and_the_preview_together(client, rig) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/export/my-face").status_code == 200
    reply = client.post("/saved/my-face/rename", json={"id": "renamed", "name": "Renamed"})
    assert reply.status_code == 200, reply.text
    for root in (rig["assets"], rig["assets"] / ".preview"):
        assert not (root / "my-face").exists()
        manifest = FaceStore(root).load("renamed")[0]
        assert manifest.name == "Renamed"
        assert json.loads((root / "renamed" / "preset.json").read_text())["id"] == "renamed"
    assert served(rig["assets"]) == ["renamed"]
    assert leftovers(rig["assets"]) == []
    # A name alone keeps the id.
    assert client.post("/saved/renamed/rename", json={"name": "Again"}).json() == {"id": "renamed"}
    assert FaceStore(rig["assets"]).load("renamed")[0].name == "Again"


def test_a_failed_rename_leaves_both_copies_as_they_were(client, rig, monkeypatch) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/export/my-face").status_code == 200
    builder: Builder = client.app.state.builder
    real = builder._rewrite

    def fail_on_preview(root, *args):
        if root == builder.preview_dir:
            raise OSError("disk full")
        return real(root, *args)

    monkeypatch.setattr(builder, "_rewrite", fail_on_preview)
    assert client.post("/saved/my-face/rename", json={"id": "renamed"}).status_code == 500
    for root in (rig["assets"], rig["assets"] / ".preview"):
        assert FaceStore(root).load("my-face")[0].id == "my-face"
        assert not (root / "renamed").exists()
    assert leftovers(rig["assets"]) == []


def test_rename_refuses_a_taken_id_another_faces_or_a_bad_one(client, rig) -> None:
    make_face(rig["assets"], "stock-01")
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/saved/my-face/rename", json={"id": "stock-01"}).status_code == 409
    assert client.post("/saved/stock-01/rename", json={"name": "Mine now"}).status_code == 409
    assert client.post("/saved/my-face/rename", json={"id": "../escape"}).status_code == 422
    assert client.post("/saved/my-face/rename", json={}).status_code == 422
    assert client.post("/saved/nothing/rename", json={"name": "x"}).status_code == 404
    assert FaceStore(rig["assets"]).load("stock-01")[0].name != "Mine now"


def test_delete_moves_to_the_trash(client, rig) -> None:
    make_face(rig["assets"], "stock-01")
    assert client.post("/build", json=preset()).status_code == 200
    assert client.post("/export/my-face").status_code == 200
    # Just the preview first, then the rest.
    assert client.delete("/saved/my-face?from=preview").status_code == 200
    assert not (rig["assets"] / ".preview" / "my-face").exists() and served(rig["assets"]) == ["my-face", "stock-01"]
    trash = client.delete("/saved/my-face").json()["trash"]
    assert served(rig["assets"]) == ["stock-01"]
    assert (Path(trash) / "export" / "head.glb").is_file()
    assert client.delete("/saved/stock-01").status_code == 409  # never another's face
    assert client.delete("/saved/my-face").status_code == 404


def test_rename_and_delete_are_guarded_writes(client, rig) -> None:
    assert client.post("/build", json=preset()).status_code == 200
    cross = {"Origin": "https://elsewhere.example"}
    assert client.post("/saved/my-face/rename", json={"name": "x"}, headers=cross).status_code == 403
    assert client.delete("/saved/my-face", headers=cross).status_code == 403
    assert (rig["assets"] / ".preview" / "my-face").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", sorted((SCRIPT.parent / "page").glob("*.js")), ids=lambda p: p.name)
def test_the_page_scripts_parse(script) -> None:
    # The pages have no build step and no browser test in CI: at least they must parse.
    done = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_the_faces_page_is_served(client) -> None:
    assert "Saved faces" in client.get("/faces.html").text
    assert client.get("/faces.js").status_code == 200


def test_a_cross_site_build_is_refused(client, rig) -> None:
    reply = client.post("/build", json=preset(), headers={"Origin": "https://elsewhere.example"})
    assert reply.status_code == 403
    assert calls(rig) == []


def test_a_read_only_editor_does_not_build(rig) -> None:
    from argus_mask.editor.app import create_editor_app

    app = create_editor_app(Settings(assets_dir=rig["assets"], read_only=True), rig["editor"])
    with TestClient(app) as c:
        assert c.post("/build", json=preset()).status_code == 403
        assert c.post("/export/my-face").status_code == 403
        assert c.get("/health").json()["read_only"] is True
    assert calls(rig) == []


def test_health_says_why_it_cannot_build(rig, tmp_path) -> None:
    from argus_mask.editor.app import create_editor_app

    for editor, why in [
        (EditorSettings(blender=str(tmp_path / "no-blender"), mpfb_data=rig["data"]), "Blender not found"),
        (EditorSettings(blender=rig["editor"].blender), "not set"),
        (EditorSettings(blender=rig["editor"].blender, mpfb_data=tmp_path / "nowhere"), "does not exist"),
    ]:
        with TestClient(create_editor_app(Settings(assets_dir=rig["assets"]), editor)) as c:
            health = c.get("/health").json()
            assert health["ready"] is False and why in health["reason"]
            assert c.post("/build", json=preset()).status_code in (422, 500)


def test_library_routes(client) -> None:
    library = client.get("/library").json()
    assert [a["name"] for a in library["hair"]] == ["h1", "h2"]
    assert client.get("/library/hair/h1/thumb").content.startswith(b"\x89PNG")
    assert client.get("/library/clothes/top1/thumb").status_code == 200
    assert client.get("/library/clothes/shoes1/thumb").status_code == 404
    assert [a["slot"] for a in library["clothes"]] == ["hat", "outfit"]
    for bad in ["/library/hair/rogue/thumb", "/library/nope/h1/thumb", "/library/hair/..%2F..%2Fpacks/thumb"]:
        assert client.get(bad).status_code == 404


def test_the_modifiers_route_lists_the_catalogue(client) -> None:
    body = client.get("/modifiers").json()
    assert list(body["groups"]) == list(MODIFIER_GROUPS)
    by_name = {m["name"]: m for m in body["modifiers"]}
    assert len(by_name) == len(MODIFIERS)
    assert by_name["head-square"]["min"] == 0.0 and by_name["chin-width"]["min"] == -1.0
    assert "negative" not in by_name["chin-width"]  # file paths stay on the server
    # Each two-way slider says what its ends mean, so the page can read "45% wider", not "+0.45".
    assert by_name["chin-width"]["ends"] == ["narrower", "wider"]
    assert by_name["head-age"]["ends"] == ["younger", "older"] and by_name["head-age"]["label"] == "Apparent age"
    assert all(len(m["ends"]) == 2 for m in body["modifiers"] if m["min"] < 0)
    assert all(m["ends"] == [] for m in body["modifiers"] if m["min"] == 0)


def test_the_page_and_three_are_served(client) -> None:
    assert "Face editor" in client.get("/").text
    assert client.get("/editor.js").status_code == 200
    assert client.get("/vendor/three.module.js").status_code == 200


# ---- the builder on its own ----------------------------------------------------------------


def builder_for(rig, **kw) -> Builder:
    ed = rig["editor"]
    return Builder(
        blender=ed.blender,
        blender_user=ed.blender_user,
        library=AssetLibrary(ed.data_dir),
        assets_dir=rig["assets"],
        **kw,
    )


async def test_one_build_at_a_time(rig, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_BLENDER", "sleep")
    builder = builder_for(rig, timeout_s=5)
    first = asyncio.create_task(builder.build(Preset.model_validate(preset())))
    await asyncio.sleep(0.2)
    with pytest.raises(Busy):
        await builder.build(Preset.model_validate(preset(id="other")))
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert leftovers(rig["assets"]) == []


async def test_a_build_that_hangs_is_stopped(rig, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_BLENDER", "sleep")
    builder = builder_for(rig, timeout_s=0.5)
    with pytest.raises(BuildError, match="was stopped"):
        await builder.build(Preset.model_validate(preset()))
    assert not builder.busy and nothing_built(rig["assets"])


def test_editor_settings_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BLENDER_USER_RESOURCES", raising=False)
    monkeypatch.setenv("ARGUS_MASK_BLENDER_USER", str(tmp_path))
    ed = EditorSettings.from_env()
    assert ed.data_dir == tmp_path / "extensions/.user/user_default/mpfb/data"
    monkeypatch.setenv("ARGUS_MASK_BUILD_TIMEOUT_S", "1")
    with pytest.raises(ValueError, match="between"):
        EditorSettings.from_env()
    assert os.environ["ARGUS_MASK_BLENDER_USER"] == str(tmp_path)
