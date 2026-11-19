"""Rest sets: the named baseline expression a face holds at rest and speaks from."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.expressions import BUILTIN, DEFAULT_SET, EXPRESSIONS_NAME, ExpressionError, Expressions, check_set
from argus_mask.models import RestSet
from argus_mask.server.app import create_app
from tests.helpers import make_face


@pytest.fixture
def assets(tmp_path):
    make_face(tmp_path, "test-head")
    make_face(tmp_path, "other-head")
    return tmp_path


def write(assets, body: dict) -> None:
    (assets / EXPRESSIONS_NAME).write_text(json.dumps(body))


# ---- the store ------------------------------------------------------------------------------------


def test_a_face_rests_warm_until_it_chooses(assets) -> None:
    ex = Expressions(assets)
    assert DEFAULT_SET == "warm" and ex.for_face("test-head").name == "warm"
    assert set(ex.sets()) >= {"neutral", "calm", "warm", "bright", "attentive"}
    ex.choose("test-head", "calm")
    assert ex.for_face("test-head").name == "calm" and ex.for_face("other-head").name == "warm"
    assert ex.for_face("test-head", override="bright").name == "bright"  # a screen's ?rest=
    with pytest.raises(ExpressionError, match="no rest set"):
        ex.for_face("test-head", override="smirk")
    ex.choose("test-head", None)  # back to the default, and not written down
    assert ex.choices() == {} and json.loads((assets / EXPRESSIONS_NAME).read_text())["faces"] == {}


@pytest.mark.parametrize(
    ("name", "weights", "life", "why"),
    [
        ("Bad Name", {}, 1.0, "not a usable name"),
        ("ok", {"mouthSmirk": 0.2}, 1.0, "not an ARKit blendshape"),
        ("ok", {"mouthSmileLeft": 1.5}, 1.0, "from 0 to 1"),
        ("ok", {"mouthSmileLeft": True}, 1.0, "from 0 to 1"),
        ("ok", {}, 3.0, "life"),
        pytest.param("ok", {"mouthSmileLeft": 10**400}, 1.0, "from 0 to 1", id="huge-weight"),  # too big for a float
        pytest.param("ok", {"mouthSmileLeft": -(10**400)}, 1.0, "from 0 to 1", id="huge-negative-weight"),
        pytest.param("ok", {"mouthSmileLeft": float("nan")}, 1.0, "from 0 to 1", id="nan-weight"),
        pytest.param("ok", {}, 10**400, "life", id="huge-life"),
        ("ok", [], 1.0, "must be an object"),
    ],
)
def test_an_unsound_set_is_refused(name, weights, life, why) -> None:
    with pytest.raises(ExpressionError, match=why):
        check_set(name, weights, life)


@pytest.mark.parametrize("weight", [5.0, -3.0, float("inf"), float("nan")])
def test_a_rest_set_holds_its_weights_from_0_to_1(weight: float) -> None:
    with pytest.raises(ValidationError, match="jawOpen"):
        RestSet(name="odd", weights={"jawOpen": weight})


def test_a_huge_integer_from_json_is_refused_as_a_bad_set() -> None:
    entry = json.loads('{"weights": {"jawOpen": 1' + "0" * 400 + '}, "life": 1}')
    with pytest.raises(ExpressionError):
        check_set("ok", entry["weights"], entry["life"])


def test_a_change_builds_on_the_file_even_when_its_stat_looks_unchanged(assets) -> None:
    ex = Expressions(assets)
    ex.choose("face-a", "calm")
    assert ex.choices() == {"face-a": "calm"}
    path = assets / EXPRESSIONS_NAME
    stat = path.stat()
    # Another writer's change with the same inode, size and mtime: what a reused inode within one clock tick looks like.
    with path.open("r+", encoding="utf-8") as f:
        text = f.read().replace("face-a", "face-b")
        f.seek(0)
        f.write(text)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_ino == stat.st_ino and path.stat().st_size == stat.st_size
    ex.choose("face-c", "bright")
    assert json.loads(path.read_text())["faces"] == {"face-b": "calm", "face-c": "bright"}
    assert ex.choices() == {"face-b": "calm", "face-c": "bright"}, "the instance reads what it wrote"


def test_a_hand_edited_file_is_read_again_and_its_bad_entries_ignored(assets) -> None:
    ex = Expressions(assets)
    write(assets, {"sets": {"cheery": {"weights": {"mouthSmileLeft": 0.5}, "life": 1.2},
                            "broken": {"weights": {"nope": 1}}},
                   "faces": {"test-head": "cheery", "other-head": "missing"}})  # fmt: skip
    assert ex.for_face("test-head").weights == {"mouthSmileLeft": 0.5}
    assert ex.for_face("other-head").name == "warm"  # chose a set that does not exist: the default
    assert set(ex.problems()) == {"set broken", "face other-head"}
    # a file's set may replace a built-in one, and removing it brings the built-in back
    write(assets, {"sets": {"warm": {"weights": {"browInnerUp": 0.3}}}, "faces": {"test-head": "warm"}})
    assert ex.for_face("test-head").weights == {"browInnerUp": 0.3}
    assert ex.remove("warm") and ex.for_face("test-head") == BUILTIN["warm"]
    assert not ex.remove("calm")  # built in: nothing of the file's to remove


@pytest.mark.parametrize("body", ['{"sets": {"cheery": ', "[]"])
def test_a_file_that_cannot_be_read_is_never_overwritten(assets, body) -> None:
    ex = Expressions(assets)
    (assets / EXPRESSIONS_NAME).write_text(body)  # a hand edit with a typo, or not an object
    assert ex.for_face("test-head").name == "warm"  # read as no sets and no choices
    assert set(ex.problems()) == {EXPRESSIONS_NAME}
    with pytest.raises(ExpressionError, match="not changed"):
        ex.choose("test-head", "calm")
    with pytest.raises(ExpressionError, match="not changed"):
        ex.define(check_set("cheery", {"mouthSmileLeft": 0.4}))
    assert (assets / EXPRESSIONS_NAME).read_text() == body
    write(assets, {"sets": {}, "faces": {}})  # mended: writes go through again
    ex.choose("test-head", "calm")
    assert ex.choices() == {"test-head": "calm"} and ex.problems() == {}


def test_a_choice_follows_a_renamed_face_and_goes_with_a_deleted_one(assets) -> None:
    ex = Expressions(assets)
    ex.choose("test-head", "calm")
    ex.repoint("test-head", "new-head")
    assert ex.choices() == {"new-head": "calm"}
    ex.repoint("new-head", None)
    assert ex.choices() == {}


def test_removing_a_set_sends_its_faces_back_to_the_default(assets) -> None:
    ex = Expressions(assets)
    ex.define(check_set("cheery", {"mouthSmileLeft": 0.4, "mouthSmileRight": 0.0}))
    assert ex.sets()["cheery"].weights == {"mouthSmileLeft": 0.4}  # zero weights are dropped
    ex.choose("test-head", "cheery")
    assert ex.remove("cheery") and ex.for_face("test-head").name == "warm" and ex.choices() == {}


# ---- the server ------------------------------------------------------------------------------------


def test_the_server_serves_each_face_its_rest_set(assets) -> None:
    write(assets, {"sets": {}, "faces": {"other-head": "calm"}})
    (assets / "aliases.json").write_text(json.dumps({"aliases": {"holly": "other-head"}}))
    with TestClient(create_app(Settings(assets_dir=assets))) as c:
        assert c.get("/face/test-head/rest").json()["name"] == "warm"
        assert c.get("/face/holly/rest").json() == BUILTIN["calm"].model_dump()  # by its alias, the face's own set
        assert c.get("/face/test-head/rest", params={"set": "bright"}).json()["name"] == "bright"
        assert c.get("/face/test-head/rest", params={"set": "smirk"}).status_code == 404
        assert c.get("/face/nobody/rest").status_code == 404
        listed = c.get("/expressions").json()
        assert listed["default"] == "warm" and listed["faces"] == {"other-head": "calm"} and "neutral" in listed["sets"]


# ---- the editor -----------------------------------------------------------------------------------


@pytest.fixture
def editor(assets, tmp_path_factory):
    from argus_mask.editor.app import create_editor_app  # noqa: PLC0415

    work = tmp_path_factory.mktemp("editor")
    ed = EditorSettings(voices_dir=work / "voices", recordings_dir=work / "rec")
    with TestClient(create_editor_app(Settings(assets_dir=assets), ed)) as c:
        yield c


def test_the_editor_chooses_defines_and_removes_rest_sets(editor, assets) -> None:
    listed = editor.get("/expressions").json()
    assert listed["default"] == "warm" and "warm" in listed["builtin"] and listed["problems"] == {}
    assert editor.put("/expressions/faces/test-head", json={"set": "calm"}).json() == {
        "face": "test-head",
        "set": "calm",
    }
    assert Expressions(assets).choices() == {"test-head": "calm"}
    assert editor.put("/expressions/faces/test-head", json={"set": "smirk"}).status_code == 422
    assert editor.put("/expressions/faces/Not.A.Face", json={"set": "calm"}).status_code == 404
    made = editor.put("/expressions/sets/cheery", json={"weights": {"mouthSmileLeft": 0.5}, "life": 1.4})
    assert made.status_code == 200 and made.json()["life"] == 1.4
    assert editor.put("/expressions/sets/cheery", json={"weights": {"nope": 0.5}}).status_code == 422
    assert editor.put("/expressions/sets/Bad", json={"weights": {}}).status_code == 422
    assert editor.delete("/expressions/sets/cheery").status_code == 200
    assert editor.delete("/expressions/sets/calm").status_code == 404  # built in
    assert editor.get("/motion.js").headers["content-type"].startswith("text/javascript")
    assert editor.get("/talk.json").json()["fps"] == 30


def test_another_site_may_not_change_rest_sets(editor) -> None:
    evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    assert editor.put("/expressions/faces/test-head", json={"set": "calm"}, headers=evil).status_code == 403
    assert editor.put("/expressions/sets/x", json={"weights": {}}, headers=evil).status_code == 403
