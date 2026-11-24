"""The guided page's pick, overrule and file routes (#174)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedStore, StepResult
from tests.test_guided import evaluated, result, ticking


@pytest.fixture
def editor(tmp_path: Path):
    assets = tmp_path / "assets"
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    store = GuidedStore(assets, clock=ticking())
    with TestClient(create_editor_app(Settings(assets_dir=assets, driver="energy"), ed)) as client:
        yield client, store


def _ready(store: GuidedStore, candidate: str = "c1") -> None:
    store.create("guide", "A guide")
    store.change("guide", lambda record, _now: evaluated(record, candidate))


def test_the_guided_page_is_served(editor) -> None:
    client, _store = editor
    page = client.get("/guided.html")
    assert page.status_code == 200
    assert b"Guided heads" in page.content
    assert b'role="status"' in page.content
    assert b"guided.js" in page.content
    js = client.get("/guided.js")
    assert js.status_code == 200 and b"Guided head builder" in js.content


def test_pick_needs_every_eval_settled(editor) -> None:
    client, store = editor
    store.create("blocked")
    store.put("blocked", result("source", "c1"))
    refused = client.post("/guided/blocked/pick", json={"candidate": "c1"})
    assert refused.status_code == 409
    assert "cannot run" in refused.json()["detail"]

    store.create("ready")
    store.change("ready", lambda record, _now: evaluated(record, "c1"))
    picked = client.post("/guided/ready/pick", json={"candidate": "c1"})
    assert picked.status_code == 200
    assert picked.json()["step"] == "pick" and picked.json()["candidate"] == "c1"
    view = client.get("/guided/ready").json()
    assert view["picked"] == "c1"
    assert view["face"]["pick"]["status"] == "passed"


def test_pick_of_a_missing_build_is_404(editor) -> None:
    client, _store = editor
    assert client.post("/guided/nobody/pick", json={"candidate": "c1"}).status_code == 404


def test_overrule_an_advisory_and_refuse_anything_else(editor) -> None:
    client, store = editor
    store.create("guide")
    store.put("guide", result("source", "c1"))
    store.put(
        "guide",
        result("portrait", "c1", Check(name="temples", verdict="advisory", reason="bare at the temple")),
    )
    ok = client.post(
        "/guided/guide/candidates/c1/overrule",
        json={"step": "portrait", "check": "temples", "reason": "a fringe will cover it"},
    )
    assert ok.status_code == 200
    assert any(o["check"] == "temples" for o in ok.json()["overruled"])

    store.put(
        "guide",
        result("portrait", "c1", Check(name="adult", verdict="refused", reason="under 18")),
    )
    bad = client.post(
        "/guided/guide/candidates/c1/overrule",
        json={"step": "portrait", "check": "adult", "reason": "I say so"},
    )
    assert bad.status_code == 409
    assert "only an advisory" in bad.json()["detail"]


def test_a_guided_file_is_served_only_when_the_record_names_it(editor) -> None:
    client, store = editor
    store.create("guide")
    portrait = store.keep("guide", "portrait", "candidates/c1/portrait.png", b"PNG-bytes")
    store.put("guide", StepResult(step="source", candidate="c1", outputs=[portrait]))
    # A file left on disk but never in the record is not served.
    (store.directory("guide") / "candidates" / "c1").mkdir(parents=True, exist_ok=True)
    (store.directory("guide") / "candidates" / "c1" / "secret.png").write_bytes(b"nope")

    ok = client.get("/guided/guide/file/candidates/c1/portrait.png")
    assert ok.status_code == 200 and ok.content == b"PNG-bytes"
    assert client.get("/guided/guide/file/candidates/c1/secret.png").status_code == 404
    # Encoded, so the client does not resolve the dots before sending: the handler itself must refuse them.
    for path in ("candidates%2F..%2Fbuild.json", "%2e%2e%2Fguide%2Fbuild.json", "build.json"):
        refused = client.get(f"/guided/guide/file/{path}")
        assert refused.status_code == 404 and refused.json()["detail"] == "no such guided file"


def test_head_zip_serves_the_built_head(editor) -> None:
    client, store = editor
    store.create("guide")
    store.put("guide", result("source", "c1"))
    store.put("guide", result("portrait", "c1", Check(name="frontal", verdict="passed")))
    head = store.keep("guide", "head", "candidates/c1/head.zip", b"PK-fake-zip")
    store.put(
        "guide",
        StepResult(
            step="build",
            candidate="c1",
            checks=[Check(name="pipeline", verdict="passed")],
            outputs=[head],
        ),
    )
    reply = client.get("/guided/guide/candidates/c1/head.zip")
    assert reply.status_code == 200
    assert reply.content == b"PK-fake-zip"
    assert "zip" in reply.headers.get("content-type", "")
    assert client.get("/guided/guide/candidates/c2/head.zip").status_code == 404


def test_pick_rejects_a_candidate_id_that_is_not_a_face_id(editor) -> None:
    client, store = editor
    _ready(store)
    assert client.post("/guided/guide/pick", json={"candidate": "Not Valid"}).status_code == 422
