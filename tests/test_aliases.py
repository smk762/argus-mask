"""Face aliases: a stable name (a persona's "holly") pointing at whichever face the editor chose."""

from __future__ import annotations

import json
import os

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.faces import ALIASES_NAME, AliasError, Aliases, FaceStore
from argus_mask.server.app import create_app
from tests.helpers import make_face, tone
from tests.test_push import wav
from tests.test_server import HELLO, _screens, drain


@pytest.fixture
def assets(tmp_path):
    make_face(tmp_path, "test-head")
    make_face(tmp_path, "other-head")
    make_face(tmp_path, "anonymous", provenance=None)  # not served: no provenance
    return tmp_path


def write(assets, entries: dict) -> None:
    (assets / ALIASES_NAME).write_text(json.dumps({"aliases": entries}))


# ---- the store ------------------------------------------------------------------------------------


def test_an_alias_resolves_to_its_face(assets) -> None:
    aliases = Aliases(FaceStore(assets))
    assert aliases.all() == {} and aliases.resolve("holly") == "holly"
    aliases.set("holly", "test-head")
    assert json.loads((assets / ALIASES_NAME).read_text()) == {"aliases": {"holly": "test-head"}}
    assert aliases.resolve("holly") == "test-head" and aliases.resolve("test-head") == "test-head"
    aliases.set("holly", "other-head")  # repointed
    assert aliases.all() == {"holly": "other-head"}
    assert aliases.remove("holly") and not aliases.remove("holly") and aliases.all() == {}


@pytest.mark.parametrize(
    ("name", "target", "why"),
    [
        ("test-head", "other-head", "is a face"),  # never shadows a face
        ("holly", "nobody", "not a face this server serves"),
        ("holly", "anonymous", "not a face this server serves"),  # a refused face is not a target
        ("Holly!", "test-head", "not a face id"),
        ("../x", "test-head", "not a face id"),
    ],
)
def test_an_unsound_alias_is_refused(assets, name, target, why) -> None:
    with pytest.raises(AliasError, match=why):
        Aliases(FaceStore(assets)).set(name, target)


def test_no_chains(assets) -> None:
    aliases = Aliases(FaceStore(assets))
    aliases.set("holly", "test-head")
    with pytest.raises(AliasError, match="itself an alias"):
        aliases.set("kiosk", "holly")
    aliases.set("kiosk", "test-head")
    # holly may not become a chain's target by being pointed at: nothing points at an alias
    write(assets, {"holly": "test-head", "kiosk": "holly"})
    assert aliases.all() == {"holly": "test-head"} and "kiosk" in aliases.problems()


def test_a_hand_edited_file_is_read_again_and_its_bad_entries_ignored(assets) -> None:
    aliases = Aliases(FaceStore(assets))
    write(assets, {"holly": "test-head"})
    assert aliases.resolve("holly") == "test-head"
    write(assets, {"holly": "other-head", "test-head": "other-head", "ghost": "nobody"})
    stat = (assets / ALIASES_NAME).stat()
    os.utime(assets / ALIASES_NAME, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert aliases.all() == {"holly": "other-head"}
    assert set(aliases.problems()) == {"test-head", "ghost"}
    (assets / ALIASES_NAME).write_text("not json")
    assert aliases.all() == {}


def test_aliases_follow_a_renamed_face_and_hold_a_deleted_one(assets) -> None:
    aliases = Aliases(FaceStore(assets))
    aliases.set("holly", "test-head")
    assert aliases.pointing_at("test-head") == ["holly"]
    aliases.repoint("test-head", "new-head")  # the editor's rename does this after moving the face
    assert aliases.raw() == {"holly": "new-head"}
    aliases.repoint("new-head", None)
    assert aliases.raw() == {}


# ---- the server -----------------------------------------------------------------------------------


@pytest.fixture
def client(assets):
    write(assets, {"holly": "test-head"})
    with TestClient(create_app(Settings(assets_dir=assets))) as c:
        yield c


def test_the_server_serves_a_face_by_its_alias(client) -> None:
    assert client.get("/aliases").json() == {"holly": "test-head"}
    assert client.get("/health").json()["aliases"] == {"holly": "test-head"}
    assert client.get("/face/holly").json()["id"] == "test-head"
    assert client.get("/face/holly/model").content == client.get("/face/test-head/model").content
    assert [f["id"] for f in client.get("/faces").json()] == ["other-head", "test-head"]  # /faces is unchanged
    assert client.get("/face/kiosk").status_code == 404


def test_a_viewer_and_a_source_meet_on_the_alias(client) -> None:
    with client.websocket_connect("/session/holly/view") as viewer:
        viewer.send_text(json.dumps({"type": "status", "audio": True, "display": "hld", "kiosk": True}))
        assert len(_screens(client, "holly", want=None, tries=1) or _screens(client, "holly", tries=100)) == 1
        done = client.post("/push", params={"face": "holly", "require_screen": True}, content=wav(tone(0.5)))
        assert done.status_code == 200, done.text
        notes, _ = drain(viewer)
        # the header names the face itself, so a page that loaded another (the alias was repointed) reloads
        assert notes[0]["type"] == "header" and notes[0]["face_id"] == "test-head"
        with client.websocket_connect("/session/holly/source") as source:
            source.send_text(HELLO)
            assert json.loads(source.receive_text())["type"] == "ready"
            source.send_bytes(tone(0.2).astype("<f4").tobytes())
            source.send_text(json.dumps({"type": "end"}))
            assert json.loads(source.receive_text())["type"] == "done"
        assert drain(viewer)[0][0]["face_id"] == "test-head"


def test_a_repointed_alias_streams_the_new_face(client, assets) -> None:
    with client.websocket_connect("/session/holly/view") as viewer:
        write(assets, {"holly": "other-head"})
        stat = (assets / ALIASES_NAME).stat()
        os.utime(assets / ALIASES_NAME, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert client.post("/push", params={"face": "holly"}, content=wav(tone(0.3))).status_code == 200
        assert drain(viewer)[0][0]["face_id"] == "other-head"  # the same socket: no reconnect needed
    assert client.get("/face/holly").json()["id"] == "other-head"


def test_a_broken_alias_is_no_face(client, assets) -> None:
    write(assets, {"holly": "anonymous"})
    assert client.get("/face/holly").status_code == 404
    assert client.post("/push", params={"face": "holly"}, content=wav(tone(0.3))).status_code == 404
    with client.websocket_connect("/session/holly/view") as viewer:
        assert json.loads(viewer.receive_text())["type"] == "error"


# ---- the editor -----------------------------------------------------------------------------------


@pytest.fixture
def editor(assets, tmp_path_factory):
    from argus_mask.editor.app import create_editor_app  # noqa: PLC0415

    work = tmp_path_factory.mktemp("editor")
    ed = EditorSettings(voices_dir=work / "voices", recordings_dir=work / "rec")
    with TestClient(create_editor_app(Settings(assets_dir=assets), ed)) as c:
        yield c


def test_the_editor_sets_repoints_and_removes_aliases(editor, assets) -> None:
    assert editor.get("/aliases").json() == {"aliases": {}, "problems": {}}
    assert editor.put("/aliases/holly", json={"face": "test-head"}).json() == {"alias": "holly", "face": "test-head"}
    assert editor.put("/aliases/holly", json={"face": "other-head"}).status_code == 200
    assert editor.get("/aliases").json()["aliases"] == {"holly": "other-head"}
    # no face may take an alias's name: it would hide the alias
    taken = editor.post("/saved/test-head/rename", json={"id": "holly"})
    assert taken.status_code == 409 and "is an alias" in taken.json()["detail"]
    assert editor.post("/export/holly").status_code == 409
    shadow = editor.put("/aliases/test-head", json={"face": "other-head"})
    assert shadow.status_code == 422 and "is a face" in shadow.json()["detail"]
    assert editor.put("/aliases/kiosk", json={"face": "nobody"}).status_code == 422
    assert editor.put("/aliases/kiosk", json={"face": "../etc"}).status_code == 422
    # a face an alias points at is not deleted from under it
    held = editor.delete("/saved/other-head")
    assert held.status_code == 409 and "holly points at other-head" in held.json()["detail"]
    # another site may not repoint a kiosk's face
    evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    assert editor.put("/aliases/holly", json={"face": "test-head"}, headers=evil).status_code == 403
    assert editor.delete("/aliases/holly", headers=evil).status_code == 403
    assert editor.delete("/aliases/holly").json() == {"removed": "holly"}
    assert editor.delete("/aliases/holly").status_code == 404
    assert json.loads((assets / ALIASES_NAME).read_text()) == {"aliases": {}}


def test_the_screen_dialog_takes_an_alias(editor, assets) -> None:
    write(assets, {"holly": "test-head"})
    out = editor.get("/screen/holly").json()
    assert out["exported"] and out["alias_of"] == "test-head" and out["face"] == "holly"
