from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from argus_mask.config import Settings
from argus_mask.models import Health, StreamHeader
from argus_mask.server.app import create_app
from tests.helpers import fake_a2f, make_face, tone

HELLO = json.dumps({"sample_rate": 24000, "channels": 1, "format": "f32le"})


@pytest.fixture
def assets(tmp_path):
    make_face(tmp_path, "test-head")
    make_face(tmp_path, "anonymous", provenance=None)
    return tmp_path


@pytest.fixture
def client(assets):
    with TestClient(create_app(Settings(assets_dir=assets))) as c:
        yield c


def pcm(seconds: float = 0.5) -> bytes:
    return tone(seconds).astype("<f4").tobytes()


def drain(viewer) -> tuple[list[dict], list[bytes]]:
    """Read a viewer socket up to the end message: its JSON messages, and its audio chunks."""
    notes, audio = [], []
    while True:
        message = viewer.receive()
        if message.get("bytes") is not None:
            audio.append(message["bytes"])
            continue
        notes.append(json.loads(message["text"]))
        if notes[-1]["type"] == "end":
            return notes, audio


def test_health_and_faces(client) -> None:
    health = Health.model_validate(client.get("/health").json())
    assert health.driver.name == "energy" and health.driver.available
    assert health.faces == 1 and health.live == [] and health.read_only is False
    assert [f["id"] for f in client.get("/faces").json()] == ["test-head"]
    assert client.get("/face/test-head").json()["provenance"]["kind"] == "stock"
    model = client.get("/face/test-head/model")
    assert model.status_code == 200 and model.content[:4] == b"glTF"
    assert model.headers["content-type"] == "model/gltf-binary"


def test_a_face_without_provenance_is_not_served(client) -> None:
    assert client.get("/face/anonymous").status_code == 404
    assert client.get("/face/anonymous/model").status_code == 404
    assert client.get("/face/..%2F..%2Fetc").status_code == 404


def test_the_viewer_page_is_served(client) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]


def test_a_stream_is_a_header_then_audio_and_frames_then_an_end(client) -> None:
    with (
        client.websocket_connect("/session/test-head/view") as viewer,
        client.websocket_connect("/session/test-head/source") as source,
    ):
        source.send_text(HELLO)
        ready = source.receive_json()
        assert ready["type"] == "ready"
        assert client.get("/health").json()["live"] == ["test-head"]
        for _ in range(4):
            source.send_bytes(pcm(0.25))
        source.send_text(json.dumps({"type": "end"}))
        done = source.receive_json()
        notes, audio = drain(viewer)
    header = StreamHeader.model_validate(notes[0])
    assert header.session_id == ready["session_id"] and header.face_id == "test-head"
    assert header.audio.model_dump() == {"sample_rate": 24000, "channels": 1, "format": "f32le"}
    frames = [n for n in notes if n["type"] == "frame"]
    assert done == {"type": "done", "frames": 30, "duration_ms": 1000}
    assert [f["seq"] for f in frames] == list(range(30))
    assert all(b["t_ms"] > a["t_ms"] for a, b in zip(frames, frames[1:], strict=False))
    assert notes[-1] == {"type": "end", "frames": 30, "duration_ms": 1000}
    # Each audio chunk says where it starts on the timeline, and together they are the audio that was sent.
    starts = [struct.unpack_from("<d", chunk)[0] for chunk in audio]
    assert starts == [0.0, 250.0, 500.0, 750.0]
    heard = np.concatenate([np.frombuffer(chunk[8:], dtype="<f4") for chunk in audio])
    assert np.array_equal(heard, np.tile(tone(0.25), 4))
    assert client.get("/health").json()["live"] == []


def test_a_second_source_for_a_face_is_refused(client) -> None:
    with client.websocket_connect("/session/test-head/source") as first:
        first.send_text(HELLO)
        assert first.receive_json()["type"] == "ready"
        with client.websocket_connect("/session/test-head/source") as second:
            assert second.receive_json() == {"type": "error", "detail": "face 'test-head' already has a source"}
            with pytest.raises(WebSocketDisconnect) as closed:
                second.receive_json()
            assert closed.value.code == 1013
        first.send_bytes(pcm(0.1))  # the first is undisturbed
        first.send_text(json.dumps({"type": "end"}))
        assert first.receive_json()["type"] == "done"
    with client.websocket_connect("/session/test-head/source") as third:  # and the face is free again
        third.send_text(HELLO)
        assert third.receive_json()["type"] == "ready"


def test_a_viewer_that_joins_mid_stream_gets_the_header_then_live_frames(client) -> None:
    with client.websocket_connect("/session/test-head/source") as source:
        source.send_text(HELLO)
        session = source.receive_json()["session_id"]
        source.send_bytes(pcm(0.5))
        with client.websocket_connect("/session/test-head/view") as late:
            assert client.get("/health").json()["live"] == ["test-head"]  # the join has been handled
            source.send_bytes(pcm(0.5))
            source.send_text(json.dumps({"type": "end"}))
            assert source.receive_json()["type"] == "done"
            notes, _ = drain(late)
    assert notes[0]["type"] == "header" and notes[0]["session_id"] == session
    frames = [n for n in notes if n["type"] == "frame"]
    # Nothing from before it joined. The resampler holds a few samples back, so the frame that
    # straddles the half-second mark is finished by the audio sent after the join.
    assert frames and frames[0]["seq"] >= 14


def test_two_viewers_see_the_same_motion(client) -> None:
    with (
        client.websocket_connect("/session/test-head/view") as one,
        client.websocket_connect("/session/test-head/view") as two,
        client.websocket_connect("/session/test-head/source") as source,
    ):
        source.send_text(HELLO)
        source.receive_json()
        source.send_bytes(pcm(0.5))
        source.send_text(json.dumps({"type": "end"}))
        source.receive_json()
        assert drain(one) == drain(two)


def test_a_source_that_just_disconnects_still_ends_the_stream(client) -> None:
    with client.websocket_connect("/session/test-head/view") as viewer:
        with client.websocket_connect("/session/test-head/source") as source:
            source.send_text(HELLO)
            source.receive_json()
            source.send_bytes(pcm(0.2))
        notes, _ = drain(viewer)
    assert notes[-1]["type"] == "end" and notes[-1]["frames"] == 6


def test_a_bad_hello_and_an_unknown_face_are_refused_with_the_reason(client) -> None:
    with client.websocket_connect("/session/test-head/source") as source:
        source.send_text(json.dumps({"sample_rate": 3, "format": "mp3"}))
        assert "audio format" in source.receive_json()["detail"]
    for path in ("/session/nobody/source", "/session/nobody/view", "/session/anonymous/view"):
        with client.websocket_connect(path) as ws:
            assert ws.receive_json()["detail"].startswith("no face")


def test_a_cross_site_page_cannot_open_either_socket(assets) -> None:
    app = create_app(Settings(assets_dir=assets, cors_origins=("https://mask.example",)))
    with TestClient(app) as client:
        for path in ("/session/test-head/source", "/session/test-head/view"):
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect(path, headers={"origin": "https://evil.example"}),
            ):
                pass
            # The configured origin, and the server's own host behind a TLS proxy, are let in.
            for origin in ("https://mask.example", "https://testserver"):
                with client.websocket_connect(path, headers={"origin": origin}) as ws:
                    if path.endswith("source"):
                        ws.send_text(HELLO)
                        assert ws.receive_json()["type"] == "ready"


def test_read_only_refuses_a_source_and_still_serves_viewers(assets) -> None:
    with TestClient(create_app(Settings(assets_dir=assets, read_only=True))) as client:
        assert client.get("/health").json()["read_only"] is True
        with client.websocket_connect("/session/test-head/source") as source:
            assert "read-only" in source.receive_json()["detail"]
        with client.websocket_connect("/session/test-head/view"):
            pass


def test_a2f_down_refuses_the_source_and_sends_no_energy_frames(assets) -> None:
    app = create_app(Settings(assets_dir=assets, driver="a2f", a2f_url="http://127.0.0.1:9"))
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["driver"]["name"] == "a2f" and health["driver"]["available"] is False
        assert "does not answer" in health["driver"]["reason"]
        with client.websocket_connect("/session/test-head/source") as source:
            refusal = source.receive_json()
            assert refusal["type"] == "error" and "the a2f driver is down" in refusal["detail"]
        stage = app.state.stages["test-head"]
        assert stage.header is None and stage.claimed is False


def test_a2f_frames_reach_the_viewer_in_arkit_order(assets) -> None:
    with fake_a2f() as url, TestClient(create_app(Settings(assets_dir=assets, driver="a2f", a2f_url=url))) as client:
        assert client.get("/health").json()["driver"] == {
            "name": "a2f",
            "model": "fake-a2f",
            "available": True,
            "reason": None,
        }
        with (
            client.websocket_connect("/session/test-head/view") as viewer,
            client.websocket_connect("/session/test-head/source") as source,
        ):
            source.send_text(HELLO)
            assert source.receive_json()["type"] == "ready"
            source.send_bytes(pcm(1.0))
            source.send_text(json.dumps({"type": "end"}))
            assert source.receive_json() == {"type": "done", "frames": 30, "duration_ms": 1000}
            notes, _ = drain(viewer)
    assert notes[0]["driver"] == {"name": "a2f", "model": "fake-a2f"}
    jaw = notes[0]["blendshapes"].index("jawOpen")
    frames = [n for n in notes if n["type"] == "frame"]
    # The fake announces its shapes in reverse; the weights still land on jawOpen.
    assert [f["weights"][jaw] for f in frames[:11]] == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.0]


def test_a2f_failing_mid_stream_ends_the_stream_and_tells_the_source(assets) -> None:
    with (
        fake_a2f(fail_after=5) as url,
        TestClient(create_app(Settings(assets_dir=assets, driver="a2f", a2f_url=url))) as client,
    ):
        with (
            client.websocket_connect("/session/test-head/view") as viewer,
            client.websocket_connect("/session/test-head/source") as source,
        ):
            source.send_text(HELLO)
            assert source.receive_json()["type"] == "ready"
            source.send_bytes(pcm(1.0))
            source.send_text(json.dumps({"type": "end"}))
            failure = source.receive_json()
            notes, _ = drain(viewer)
        assert failure["type"] == "error" and "the model fell over" in failure["detail"]
        assert notes[-1]["type"] == "end" and "the model fell over" in notes[-1]["error"]
        assert client.get("/health").json()["live"] == []
