from __future__ import annotations

import asyncio
import json
import struct

import numpy as np
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from argus_mask.config import Settings
from argus_mask.models import Health, StreamHeader
from argus_mask.server.app import _DriverProbe, create_app
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


def test_a_late_cleanup_never_frees_a_source_that_claimed_the_face_since(assets) -> None:
    import asyncio
    import threading

    from argus_mask.drivers.energy import EnergyDriver, EnergySession

    closing, let_close = threading.Event(), threading.Event()

    class SlowToClose(EnergySession):
        first = True

        async def close(self) -> None:
            if SlowToClose.first:  # the first source's performer is still closing when the next one claims the face
                SlowToClose.first = False
                closing.set()
                while not let_close.is_set():
                    await asyncio.sleep(0.01)
            await super().close()

    class Driver(EnergyDriver):
        async def open(self, fps: int) -> EnergySession:
            return SlowToClose(fps)

    app = create_app(Settings(assets_dir=assets), driver=Driver())
    with TestClient(app) as client, client.websocket_connect("/session/test-head/source") as first:
        first.send_text(HELLO)
        assert first.receive_json()["type"] == "ready"
        first.send_bytes(pcm(0.1))
        first.send_text(json.dumps({"type": "end"}))
        assert closing.wait(5)  # the face is free, the first performer not yet closed
        with client.websocket_connect("/session/test-head/source") as second:
            second.send_text(HELLO)
            session = second.receive_json()["session_id"]
            let_close.set()
            assert first.receive_json()["type"] == "done"
            stage = app.state.stages["test-head"]
            assert stage.claimed and stage.header is not None and stage.header.session_id == session
            assert client.get("/health").json()["live"] == ["test-head"]
            with client.websocket_connect("/session/test-head/source") as third:
                assert third.receive_json() == {"type": "error", "detail": "face 'test-head' already has a source"}
            second.send_text(json.dumps({"type": "end"}))
            assert second.receive_json()["type"] == "done"
        assert client.get("/health").json()["live"] == []


def until_frame(viewer, seq: int) -> None:
    """Read a viewer socket until it has had frame *seq*: the server has made every frame up to it."""
    while True:
        message = viewer.receive()
        if message.get("text") is not None and json.loads(message["text"]).get("seq", -1) >= seq:
            return


def test_a_viewer_that_joins_mid_stream_gets_the_header_then_live_frames(client) -> None:
    with (
        client.websocket_connect("/session/test-head/view") as early,
        client.websocket_connect("/session/test-head/source") as source,
    ):
        source.send_text(HELLO)
        session = source.receive_json()["session_id"]
        source.send_bytes(pcm(0.5))
        until_frame(early, 13)  # the first half-second is made before the late viewer joins, not merely sent
        with client.websocket_connect("/session/test-head/view") as late:
            # The header comes only once the join is handled: anything sent after it reaches this viewer.
            header = late.receive_json()
            source.send_bytes(pcm(0.5))
            source.send_text(json.dumps({"type": "end"}))
            assert source.receive_json()["type"] == "done"
            notes, _ = drain(late)
    assert header["type"] == "header" and header["session_id"] == session
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
            # The configured origin is let in; the server's own host by a name is not, unless the name is
            # trusted: under DNS rebinding an attacker's page sends its own name as both Origin and Host.
            with client.websocket_connect(path, headers={"origin": "https://mask.example"}) as ws:
                if path.endswith("source"):
                    ws.send_text(HELLO)
                    assert ws.receive_json()["type"] == "ready"
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect(path, headers={"origin": "https://testserver"}),
            ):
                pass


@pytest.mark.parametrize(
    ("trusted", "host", "allowed"),
    [
        ((), "testserver", False),  # a screen at http://morpheus:8109 with nothing set
        (("testserver",), "testserver", True),  # ARGUS_MASK_TRUSTED_HOSTS=morpheus
        ((), "203.0.113.39:8109", True),  # an IP address cannot be rebound
        ((), "localhost:8109", True),
        (("other",), "testserver", False),
    ],
)
def test_a_screen_served_by_name_needs_the_name_trusted(assets, trusted, host, allowed) -> None:
    app = create_app(Settings(assets_dir=assets, trusted_hosts=trusted))
    with TestClient(app) as client:
        headers = {"origin": f"http://{host}", "host": host}
        if allowed:
            with client.websocket_connect("/session/test-head/view", headers=headers):
                pass
            assert client.post("/push?face=test-head", content=b"", headers=headers).status_code != 403
        else:
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect("/session/test-head/view", headers=headers),
            ):
                pass
            assert client.post("/push?face=test-head", content=b"", headers=headers).status_code == 403


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


def test_audio_is_held_until_the_frames_for_it_exist(assets) -> None:
    """A driver that answers late must not leave a viewer with sound and no mouth for it."""
    with (
        fake_a2f(lag_frames=9) as url,
        TestClient(create_app(Settings(assets_dir=assets, driver="a2f", a2f_url=url))) as client,
        client.websocket_connect("/session/test-head/view") as viewer,
        client.websocket_connect("/session/test-head/source") as source,
    ):
        source.send_text(HELLO)
        assert source.receive_json()["type"] == "ready"
        for _ in range(10):
            source.send_bytes(pcm(0.1))
        source.send_text(json.dumps({"type": "end"}))
        assert source.receive_json() == {"type": "done", "frames": 30, "duration_ms": 1000}
        covered_ms = 0.0
        chunks = 0
        while True:
            message = viewer.receive()
            if message.get("bytes") is not None:
                chunks += 1
                start_ms = struct.unpack_from("<d", message["bytes"])[0]
                end_ms = start_ms + (len(message["bytes"]) - 8) / 4 / 24000 * 1000
                assert end_ms <= covered_ms + 2 * 1000 / 30 + 1e-6, (
                    f"audio to {end_ms} ms sent with frames only to {covered_ms} ms"
                )
                continue
            note = json.loads(message["text"])
            if note["type"] == "frame":
                covered_ms = note["t_ms"] + 1000 / 30
            if note["type"] == "end":
                break
    assert chunks == 10  # every chunk still arrives, the last ones once the driver has finished


def test_a_source_faster_than_the_driver_is_slowed_not_cut_off(assets) -> None:
    """A burst of audio (#2): the driver holds the source back until it catches up."""
    with (
        fake_a2f(delay_s=0.03) as url,
        TestClient(create_app(Settings(assets_dir=assets, driver="a2f", a2f_url=url))) as client,
        client.websocket_connect("/session/test-head/view") as viewer,
        client.websocket_connect("/session/test-head/source") as source,
    ):
        source.send_text(HELLO)
        assert source.receive_json()["type"] == "ready"
        for _ in range(15):  # 15 s at once; the model takes longer than that would allow
            source.send_bytes(pcm(1.0))
        source.send_text(json.dumps({"type": "end"}))
        assert source.receive_json() == {"type": "done", "frames": 450, "duration_ms": 15000}
        notes, audio = drain(viewer)
        assert notes[-1] == {"type": "end", "frames": 450, "duration_ms": 15000}
        assert len(audio) == 15


def test_a_driver_that_stops_answering_ends_the_stream(assets, monkeypatch) -> None:
    monkeypatch.setattr("argus_mask.drivers.a2f.STALL_TIMEOUT_S", 0.3)
    with (
        fake_a2f(stall=True) as url,
        TestClient(create_app(Settings(assets_dir=assets, driver="a2f", a2f_url=url))) as client,
    ):
        with (
            client.websocket_connect("/session/test-head/view") as viewer,
            client.websocket_connect("/session/test-head/source") as source,
        ):
            source.send_text(HELLO)
            assert source.receive_json()["type"] == "ready"
            for _ in range(30):  # 3 s: more than the driver lets wait on frames
                source.send_bytes(pcm(0.1))
            failure = source.receive_json()
            notes, audio = drain(viewer)
        assert failure["type"] == "error" and "behind the audio" in failure["detail"]
        assert notes[-1]["type"] == "end" and "behind the audio" in notes[-1]["error"]
        assert client.get("/health").json()["live"] == []


def _screens(client, face="test-head", want=None, tries=100):
    """The screens /health lists for *face*, once they reach *want* (the socket is read in the background)."""
    import time  # noqa: PLC0415

    for _ in range(tries):
        screens = client.get("/health").json()["screens"].get(face, [])
        if want is None or screens == want:
            return screens
        time.sleep(0.01)
    return screens


def test_a_viewer_that_reports_is_a_screen(client) -> None:
    with client.websocket_connect("/session/test-head/view") as viewer:
        assert _screens(client, want=[]) == []  # connected, not yet said what it is: not a screen
        viewer.send_text(json.dumps({"type": "status", "audio": False, "display": "hld", "kiosk": True}))
        want = [{"display": "hld", "audio_ready": False, "kiosk": True, "sound": "local"}]
        assert _screens(client, want=want) == want
        viewer.send_text(json.dumps({"type": "status", "audio": True, "display": "hld", "kiosk": True}))
        want = [{"display": "hld", "audio_ready": True, "kiosk": True, "sound": "local"}]
        assert _screens(client, want=want) == want  # the second status replaces the first
    assert _screens(client, want=[]) == []  # gone with its socket


def test_a_screen_with_external_sound_is_ready_without_sound_of_its_own(client) -> None:
    """viewer ?audio=external: the house speakers play the sound; the screen only moves, and is ready to."""
    with client.websocket_connect("/session/test-head/view") as viewer:
        viewer.send_text(
            json.dumps({"type": "status", "audio": False, "display": "flat", "kiosk": True, "sound": "external"})
        )
        want = [{"display": "flat", "audio_ready": True, "kiosk": True, "sound": "external"}]
        assert _screens(client, want=want) == want
        viewer.send_text(json.dumps({"type": "status", "sound": "elsewhere"}))  # not a sound mode: ignored
        assert _screens(client, want=want) == want


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        json.dumps({"type": "status", "display": "cinema"}),
        json.dumps({"type": "status", "x": "y" * 2000}),  # over 1 KB
        json.dumps({"type": "say", "text": "hello"}),  # a command: a viewer only reports its state
        b"\x00" * 512,  # media: never taken from a viewer
    ],
)
def test_a_viewer_saying_anything_else_is_ignored(client, message) -> None:
    with client.websocket_connect("/session/test-head/view") as viewer:
        if isinstance(message, bytes):
            viewer.send_bytes(message)
        else:
            viewer.send_text(message)
        viewer.send_text(json.dumps({"type": "status", "audio": True}))  # still heard after the bad one
        want = [{"display": "flat", "audio_ready": True, "kiosk": False, "sound": "local"}]
        assert _screens(client, want=want) == want


async def test_a_failure_while_the_driver_is_asked_is_not_overwritten_by_the_answer() -> None:
    class Slow:
        calls = 0
        answer = asyncio.Event()

        async def check(self) -> str | None:
            self.calls += 1
            await self.answer.wait()
            return None  # asked before the failure: it says the driver is fine

    driver = Slow()
    probe = _DriverProbe(driver)
    asking = asyncio.create_task(probe.reason())
    await asyncio.sleep(0)
    probe.forget()  # a source saw the driver fail meanwhile
    driver.answer.set()
    assert await asking is None
    await probe.reason()
    assert driver.calls == 2  # asked again, not the stale answer kept


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b"\x00" * 5, {"X-Sample-Rate": "16000", "X-Audio-Format": "pcm_s16le", "X-Channels": "2"}),
        (b"\x00" * 4001, {"X-Sample-Rate": "24000", "X-Audio-Format": "pcm_f32le"}),
    ],
    ids=["s16-stereo-odd-byte", "f32-stray-byte"],
)
def test_raw_pcm_that_is_not_whole_samples_is_refused(client, body, headers) -> None:
    out = client.post("/push", params={"face": "test-head"}, content=body, headers=headers)
    assert out.status_code == 422 and "not whole samples" in out.json()["detail"]
