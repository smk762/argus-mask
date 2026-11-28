"""Guided save: write the talking face from the build record (#176)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedStore, StepResult, Tool
from argus_mask.editor.save import editor_owned, owned_guided
from argus_mask.faces import FaceStore, sha256_file
from argus_mask.models import LookAlike, Provenance
from tests.helpers import make_face, make_splat_zip
from tests.test_guided import ticking


@pytest.fixture
def editor(tmp_path: Path):
    assets = tmp_path / "assets"
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    store = GuidedStore(assets, clock=ticking())
    with TestClient(create_editor_app(Settings(assets_dir=assets, driver="energy"), ed)) as client:
        yield client, store, assets


def _ready_with_head(store: GuidedStore, assets: Path, build_id: str = "guide") -> None:
    store.create(build_id, "A guide")
    portrait = store.keep(build_id, "portrait", "candidates/c1/portrait.png", b"\x89PNG\r\n\x1a\nfake")
    store.put(
        build_id,
        StepResult(
            step="source",
            candidate="c1",
            outputs=[portrait],
            data={"kind": "synthetic", "source": "darkroom record test", "license": "Apache-2.0"},
        ),
    )
    store.put(
        build_id,
        StepResult(step="portrait", candidate="c1", checks=[Check(name="frontal", verdict="passed")]),
    )
    zip_path = assets / ".guided" / build_id / "candidates" / "c1" / "head.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    make_splat_zip(zip_path, folder="head")
    head = store.keep(build_id, "head", "candidates/c1/head.zip", zip_path.read_bytes())
    assert head.sha256 == sha256_file(zip_path)
    store.put(
        build_id,
        StepResult(
            step="build",
            candidate="c1",
            checks=[Check(name="pipeline", verdict="passed")],
            outputs=[head],
            tools=[Tool(name="LAM", license="Apache-2.0")],
        ),
    )
    store.put(
        build_id,
        StepResult(
            step="rest",
            candidate="c1",
            data={"weights": {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.2}, "life": 1.0, "name": f"fit-{build_id}"},
            scores={"score": 0.5},
        ),
    )
    store.put(
        build_id,
        StepResult(
            step="turns",
            candidate="c1",
            data={"limits": {"yaw": 18.0, "pitch_up": 8.0, "pitch_down": 12.0}},
            scores={"score": 0.5},
        ),
    )
    for step in ("likeness", "speech", "age"):
        store.put(build_id, StepResult(step=step, candidate="c1", scores={"score": 0.5}))
    store.put(
        build_id,
        StepResult(step="pick", candidate="c1", checks=[Check(name="pick", verdict="passed")]),
    )
    store.put(
        build_id,
        StepResult(
            step="scene",
            checks=[Check(name="choice", verdict="passed")],
            data={"scene": "ftl", "shot": "close"},
        ),
    )


def test_lookalike_sets_demo_ready() -> None:
    bare = LookAlike(who="me", at="2026-11-16", how="search", found="none")
    p = Provenance(kind="synthetic", source="flux", license="Apache-2.0", lookalike=bare)
    assert p.demo_ready is True
    assert Provenance(kind="synthetic", source="flux", license="Apache-2.0").demo_ready is False


def test_save_needs_pick_and_scene(editor) -> None:
    client, store, _assets = editor
    store.create("guide")
    assert client.post("/guided/guide/save", json={}).status_code == 409


def test_save_writes_the_face_with_build_json(editor) -> None:
    client, store, assets = editor
    _ready_with_head(store, assets)
    reply = client.post(
        "/guided/guide/save",
        json={
            "name": "Guide head",
            "lookalike": {
                "who": "the maintainer",
                "at": "2026-11-23",
                "how": "Bing visual search of the portrait",
                "found": "no match",
            },
        },
    )
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["step"] == "save" and body["data"]["face"] == "guide"
    assert body["data"]["demo_ready"] is True
    face_dir = assets / "guide"
    assert (face_dir / "build.json").is_file()
    assert (face_dir / "guide.zip").is_file()
    assert (face_dir / "manifest.json").is_file()
    assert owned_guided(assets, "guide") and editor_owned(assets, "guide")
    manifest, _model = FaceStore(assets).load("guide")
    assert manifest.provenance.lookalike is not None
    assert manifest.provenance.demo_ready is True
    assert manifest.model.format == "lam-splat"
    assert manifest.limits is not None and manifest.limits.yaw == 18.0
    assert client.get("/guided/guide").json()["face"]["save"]["status"] == "passed"


def test_save_without_lookalike_is_not_demo_ready(editor) -> None:
    client, store, assets = editor
    _ready_with_head(store, assets, "plain")
    reply = client.post("/guided/plain/save", json={"name": "Plain"})
    assert reply.status_code == 200, reply.text
    assert reply.json()["data"]["demo_ready"] is False
    by_name = {c["name"]: c for c in reply.json()["checks"]}
    assert by_name["lookalike"]["verdict"] == "advisory"
    manifest, _model = FaceStore(assets).load("plain")
    assert manifest.provenance.cleared is True
    assert manifest.provenance.demo_ready is False


def test_a_guided_face_is_never_owned_without_its_build_json(editor) -> None:
    """Must never: a guided face is saved without its build record (#178 / #176)."""
    client, store, assets = editor
    _ready_with_head(store, assets)
    assert client.post("/guided/guide/save", json={"name": "Guide"}).status_code == 200
    assert owned_guided(assets, "guide")
    (assets / "guide" / "build.json").unlink()
    assert not owned_guided(assets, "guide") and not editor_owned(assets, "guide")


def test_save_never_replaces_a_face_without_build_or_preset(editor) -> None:
    """Must never: save overwrites a stock/hand-added face that has no ownership mark (#178)."""
    client, store, assets = editor
    make_face(assets, "guide")
    assert not editor_owned(assets, "guide")
    _ready_with_head(store, assets)
    reply = client.post("/guided/guide/save", json={"name": "Guide"})
    assert reply.status_code == 409 and "did not make" in reply.json()["detail"]
    assert not (assets / "guide" / "build.json").is_file()


def test_save_cannot_skip_a_refused_portrait(editor) -> None:
    """Must never: a later step's route runs after a refusal (#178 / #166)."""
    client, store, assets = editor
    _ready_with_head(store, assets)
    store.put(
        "guide",
        StepResult(
            step="portrait",
            candidate="c1",
            checks=[Check(name="adult", verdict="refused", reason="reads as under 18")],
        ),
    )
    reply = client.post("/guided/guide/save", json={})
    assert reply.status_code == 409
    assert client.get("/guided/guide").json()["face"]["save"]["status"] != "passed"


def _face_bytes(face: Path) -> dict[str, bytes]:
    return {p.relative_to(face).as_posix(): p.read_bytes() for p in face.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fails", ["replace", "load"])
def test_a_save_that_cannot_finish_leaves_the_face_as_it_was(
    editor, monkeypatch: pytest.MonkeyPatch, fails: str
) -> None:
    import os

    from argus_mask.editor import save as save_mod
    from argus_mask.faces import FaceError

    client, store, assets = editor
    _ready_with_head(store, assets)
    assert client.post("/guided/guide/save", json={"name": "First"}).status_code == 200
    before = _face_bytes(assets / "guide")
    if fails == "replace":
        replace = os.replace

        def refuse(src: object, dst: object) -> None:
            if Path(str(dst)) == assets / "guide":
                raise PermissionError("no")
            replace(src, dst)

        monkeypatch.setattr(save_mod.os, "replace", refuse)
        error: type[Exception] = PermissionError
    else:

        def load(self: FaceStore, face_id: str, **_: object) -> None:
            raise FaceError("not served")

        monkeypatch.setattr(save_mod.FaceStore, "load", load)
        error = save_mod.SaveError
    with pytest.raises(error):
        _save(store, assets, name="Second")
    assert _face_bytes(assets / "guide") == before
    assert not [p for p in assets.iterdir() if p.name.startswith((".install-", ".save-"))]


def _save(store: GuidedStore, assets: Path, **over: object) -> StepResult:
    from argus_mask.backgrounds import PlateStore, SceneChoices
    from argus_mask.editor.save import save
    from argus_mask.expressions import Expressions

    scenes = SceneChoices(assets, PlateStore(assets))
    return save(store, assets_dir=assets, expressions=Expressions(assets), scenes=scenes, build_id="guide", **over)


def test_the_first_save_makes_the_assets_root(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path / "work", clock=ticking())
    _ready_with_head(store, tmp_path / "work")
    _save(store, tmp_path / "assets")
    assert (tmp_path / "assets" / "guide" / "manifest.json").is_file()


def test_a_face_saved_over_leaves_no_copy_behind(editor) -> None:
    """The face before is held only to put back if the new one cannot be served."""
    client, store, assets = editor
    _ready_with_head(store, assets)
    for name in ("First", "Second", "Third"):
        assert client.post("/guided/guide/save", json={"name": name}).status_code == 200
    assert json.loads((assets / "guide" / "manifest.json").read_text())["name"] == "Third"
    trash = assets / ".trash"
    assert not trash.exists() or not list(trash.iterdir())
