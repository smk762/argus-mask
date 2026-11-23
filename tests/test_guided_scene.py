"""Guided scene step: background, shot, and bright-floor advisory (#175)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.bright_floor import BRIGHT_SHARE, measure_path
from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import GuidedStore
from argus_mask.editor.scene import bright_floor_check, choose
from tests.test_guided import evaluated, result, ticking

GARDEN = Path(__file__).resolve().parent.parent / "src" / "argus_mask" / "viewer" / "scenes" / "garden" / "plate.png"
LAKE = Path(__file__).resolve().parent.parent / "src" / "argus_mask" / "viewer" / "scenes" / "lake" / "plate.png"


@pytest.fixture
def editor(tmp_path: Path):
    assets = tmp_path / "assets"
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    store = GuidedStore(assets, clock=ticking())
    with TestClient(create_editor_app(Settings(assets_dir=assets, driver="energy"), ed)) as client:
        yield client, store


def _picked(store: GuidedStore, build_id: str = "guide") -> None:
    store.create(build_id)
    store.change(build_id, lambda record, _now: evaluated(record, "c1"))
    store.put(build_id, result("pick", "c1"))


def test_garden_floor_is_bright_and_lake_is_not() -> None:
    """#151 / #175: garden's pale path warns; lake's dusk water does not."""
    garden = measure_path(GARDEN)
    lake = measure_path(LAKE)
    assert garden.bright is True and garden.bright_share >= BRIGHT_SHARE
    assert lake.bright is False
    assert bright_floor_check(garden).verdict == "advisory"
    assert bright_floor_check(lake).verdict == "passed"
    assert bright_floor_check(None).verdict == "passed"


def test_scene_needs_a_settled_pick(editor) -> None:
    client, store = editor
    store.create("guide")
    refused = client.post("/guided/guide/scene", json={"scene": "garden", "shot": "close"})
    assert refused.status_code == 409
    assert "cannot run" in refused.json()["detail"]


def test_scene_records_garden_with_bright_floor_advisory(editor) -> None:
    client, store = editor
    _picked(store)
    reply = client.post("/guided/guide/scene", json={"scene": "garden", "shot": "close"})
    assert reply.status_code == 200
    body = reply.json()
    assert body["step"] == "scene"
    assert body["data"] == {"scene": "garden", "shot": "close"}
    by_name = {c["name"]: c for c in body["checks"]}
    assert by_name["choice"]["verdict"] == "passed"
    assert by_name["bright-floor"]["verdict"] == "advisory"
    assert "grey band" in by_name["bright-floor"]["reason"]
    assert body["scores"]["floor_bright_share"] >= BRIGHT_SHARE

    view = client.get("/guided/guide").json()
    assert view["face"]["scene"]["status"] == "advisory"


def test_scene_plain_and_ftl_pass_without_a_plate(editor) -> None:
    client, store = editor
    _picked(store, "plain")
    plain = client.post("/guided/plain/scene", json={"scene": None, "shot": "wide"})
    assert plain.status_code == 200
    assert plain.json()["data"] == {"scene": None, "shot": "wide"}
    assert all(c["verdict"] == "passed" for c in plain.json()["checks"])

    _picked(store, "ftl")
    ftl = client.post("/guided/ftl/scene", json={"scene": "ftl", "shot": None})
    assert ftl.status_code == 200
    assert ftl.json()["data"]["scene"] == "ftl"
    bright = next(c for c in ftl.json()["checks"] if c["name"] == "bright-floor")
    assert bright["verdict"] == "passed" and "no plate" in bright["reason"]


def test_scene_unknown_name_is_refused(editor) -> None:
    client, store = editor
    _picked(store)
    bad = client.post("/guided/guide/scene", json={"scene": "not-a-scene", "shot": "close"})
    assert bad.status_code == 422


def test_face_overrule_clears_the_bright_floor_advisory(editor) -> None:
    client, store = editor
    _picked(store)
    client.post("/guided/guide/scene", json={"scene": "garden", "shot": "close"})
    ok = client.post(
        "/guided/guide/overrule",
        json={"step": "scene", "check": "bright-floor", "reason": "the shadow covers it on screen"},
    )
    assert ok.status_code == 200
    assert any(o["check"] == "bright-floor" for o in ok.json()["overruled"])
    assert client.get("/guided/guide").json()["face"]["scene"]["status"] == "overruled"


def test_choose_helper_writes_lake_as_passed(tmp_path: Path) -> None:
    from argus_mask.backgrounds import PlateStore

    assets = tmp_path / "assets"
    store = GuidedStore(assets, clock=ticking())
    _picked(store)
    kept = choose(store, PlateStore(assets), "guide", scene="lake", shot="close")
    assert kept.standing == "passed"
    assert kept.data == {"scene": "lake", "shot": "close"}
