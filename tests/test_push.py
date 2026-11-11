"""POST /push and POST /speak: a face driven over plain HTTP, for agents that would rather not speak the socket."""

from __future__ import annotations

import io
import json
import struct
import wave

import numpy as np
import pytest
from starlette.testclient import TestClient

from argus_mask.config import Settings
from argus_mask.server.app import create_app
from tests.helpers import fake_tts, make_face, tone
from tests.test_server import HELLO, _screens, drain


@pytest.fixture
def assets(tmp_path):
    make_face(tmp_path, "test-head")
    make_face(tmp_path, "anonymous", provenance=None)
    return tmp_path


@pytest.fixture
def client(assets):
    with TestClient(create_app(Settings(assets_dir=assets))) as c:
        yield c


def wav(samples: np.ndarray, rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_a_pushed_wav_plays_into_the_face(client) -> None:
    with client.websocket_connect("/session/test-head/view") as viewer:
        done = client.post("/push", params={"face": "test-head", "seed": 3}, content=wav(tone(1.5)))
        assert done.status_code == 200, done.text
        body = done.json()
        assert set(body) == {"session_id", "frames", "duration_ms"} and body["duration_ms"] == 1500
        assert body["frames"] == 45  # 1.5 s at 30 fps
        notes, audio = drain(viewer)
    header = notes[0]
    assert header["type"] == "header" and header["session_id"] == body["session_id"] and header["seed"] == 3
    assert sum(1 for n in notes if n["type"] == "frame") == 45 and notes[-1]["frames"] == 45
    # every sample arrives, as float32 behind its start time
    assert sum((len(chunk) - 8) // 4 for chunk in audio) == 36000
    assert struct.unpack_from("<d", audio[0])[0] == 0.0
    # the face is free again at once
    assert client.get("/health").json()["live"] == []


def test_raw_pcm_with_the_tts_headers_plays(client) -> None:
    pcm = (tone(1.0, 16000) * 32767).astype("<i2")
    stereo = np.stack([pcm, pcm], axis=1).tobytes()
    headers = {"X-Sample-Rate": "16000", "X-Audio-Format": "pcm_s16le", "X-Channels": "2"}
    done = client.post("/push", params={"face": "test-head"}, content=stereo, headers=headers)
    assert done.status_code == 200, done.text
    assert done.json()["duration_ms"] == 1000 and done.json()["frames"] == 30
    f32 = tone(0.5).astype("<f4").tobytes()
    plain = {"X-Sample-Rate": "24000", "X-Audio-Format": "pcm_f32le"}
    assert client.post("/push", params={"face": "test-head"}, content=f32, headers=plain).json()["duration_ms"] == 500


@pytest.mark.parametrize(
    ("body", "headers", "why"),
    [
        (b"RIFF\x00\x00\x00\x00WAVEjunk", {}, "no fmt or data"),
        (b"\x00" * 400, {}, "X-Sample-Rate"),
        (b"\x00" * 400, {"X-Sample-Rate": "24000", "X-Audio-Format": "mp3"}, "X-Sample-Rate"),
        (b"\x00" * 400, {"X-Sample-Rate": "4000", "X-Audio-Format": "pcm_f32le"}, "do not describe"),
        (b"\x00\x00", {"X-Sample-Rate": "24000", "X-Audio-Format": "pcm_f32le"}, "less than one sample"),
        (b"", {"Content-Type": "audio/wav"}, "empty"),
    ],
)
def test_audio_it_cannot_play_is_refused(client, body, headers, why) -> None:
    refused = client.post("/push", params={"face": "test-head"}, content=body, headers=headers)
    assert refused.status_code == 422 and why in refused.json()["detail"], refused.text
    assert client.get("/health").json()["live"] == []


def test_a_face_that_is_not_served_is_not_found(client) -> None:
    assert client.post("/push", params={"face": "nobody"}, content=wav(tone(0.2))).status_code == 404
    assert client.post("/push", params={"face": "anonymous"}, content=wav(tone(0.2))).status_code == 404
    assert client.post("/push", params={"face": "../etc"}, content=wav(tone(0.2))).status_code == 422


def test_a_busy_face_is_refused(client) -> None:
    with client.websocket_connect("/session/test-head/source") as source:
        source.send_text(HELLO)
        assert json.loads(source.receive_text())["type"] == "ready"
        busy = client.post("/push", params={"face": "test-head"}, content=wav(tone(0.2)))
        assert busy.status_code == 409 and "already has a source" in busy.json()["detail"]
        source.send_text(json.dumps({"type": "end"}))
        assert json.loads(source.receive_text())["type"] == "done"
    assert client.post("/push", params={"face": "test-head"}, content=wav(tone(0.2))).status_code == 200


def test_a_screen_can_be_required(client) -> None:
    refused = client.post("/push", params={"face": "test-head", "require_screen": True}, content=wav(tone(0.2)))
    assert refused.status_code == 409 and "no screen" in refused.json()["detail"]
    with client.websocket_connect("/session/test-head/view") as viewer:
        viewer.send_text(json.dumps({"type": "status", "audio": False}))
        assert _screens(client, want=[{"display": "flat", "audio_ready": False, "kiosk": False}])
        muted = client.post("/push", params={"face": "test-head", "require_screen": True}, content=wav(tone(0.2)))
        assert muted.status_code == 409  # watching, but it cannot be heard
        viewer.send_text(json.dumps({"type": "status", "audio": True}))
        assert _screens(client, want=[{"display": "flat", "audio_ready": True, "kiosk": False}])
        heard = client.post("/push", params={"face": "test-head", "require_screen": True}, content=wav(tone(0.2)))
        assert heard.status_code == 200
        drain(viewer)


def test_a_read_only_server_and_another_site_are_refused(assets) -> None:
    with TestClient(create_app(Settings(assets_dir=assets, read_only=True))) as c:
        refused = c.post("/push", params={"face": "test-head"}, content=wav(tone(0.2)))
        assert refused.status_code == 403 and "read-only" in refused.text
        assert c.post("/speak", json={"face": "test-head", "text": "Hello there."}).status_code == 403
    with TestClient(create_app(Settings(assets_dir=assets))) as c:
        evil = {"Origin": "https://evil.example"}
        assert c.post("/push", params={"face": "test-head"}, content=wav(tone(0.2)), headers=evil).status_code == 403


def test_a_down_driver_refuses_with_its_reason_and_never_falls_back(assets) -> None:
    with TestClient(create_app(Settings(assets_dir=assets, driver="a2f", a2f_url="http://127.0.0.1:9"))) as c:
        with c.websocket_connect("/session/test-head/view") as viewer:
            refused = c.post("/push", params={"face": "test-head"}, content=wav(tone(0.2)))
            assert refused.status_code == 503 and "the a2f driver is down" in refused.json()["detail"]
            viewer.close()
        assert c.get("/health").json()["live"] == []


def test_too_much_audio_is_refused(client, monkeypatch) -> None:
    from argus_mask.server import app as server  # noqa: PLC0415

    monkeypatch.setattr(server, "MAX_PUSH_BYTES", 1000)
    refused = client.post("/push", params={"face": "test-head"}, content=wav(tone(0.2)))
    assert refused.status_code == 413


def test_speak_says_text_through_the_configured_tts(assets, capsys) -> None:
    with (
        fake_tts(tone(0.5), 24000) as (tts, seen),
        TestClient(create_app(Settings(assets_dir=assets, tts_url=tts))) as c,
        c.websocket_connect("/session/test-head/view") as viewer,
    ):
        text = "Hi. The secret word is 7 zebras."
        done = c.post("/speak", json={"face": "test-head", "text": text, "voice": "1", "temperature": 0.7})
        assert done.status_code == 200, done.text
        notes, _ = drain(viewer)
    # tidied as `say` tidies: the short greeting goes with the next sentence, the number as a word
    assert seen == [{"text": "Hi. The secret word is seven zebras.", "temperature": 0.7, "speaker": "1"}]
    # 0.5 s of speech, with the lead and tail silence `say` puts around it
    assert done.json()["duration_ms"] == 950 and notes[-1]["type"] == "end"
    out = capsys.readouterr()
    assert "chars=" in out.out + out.err and "zebras" not in out.out + out.err  # logged, but never the text


def test_speak_passes_a_named_speaker_and_an_instruction(assets, capsys) -> None:
    with (
        fake_tts(tone(0.5), 24000) as (tts, seen),
        TestClient(create_app(Settings(assets_dir=assets, tts_url=tts, tts_voice="speaker:Vivian"))) as c,
        c.websocket_connect("/session/test-head/view") as viewer,
    ):
        how = "Calm, low and measured."
        done = c.post("/speak", json={"face": "test-head", "text": "Good morning.", "instruction": how})
        assert done.status_code == 200, done.text
        drain(viewer)
    assert seen == [{"text": "Good morning.", "instruction": how, "speaker": "Vivian"}]
    out = capsys.readouterr()
    assert "measured" not in out.out + out.err  # like the text, never logged


def test_speak_needs_a_tts_service(client) -> None:
    refused = client.post("/speak", json={"face": "test-head", "text": "Hello there."})
    assert refused.status_code == 422 and "ARGUS_MASK_TTS_URL" in refused.json()["detail"]
    assert client.post("/speak", json={"face": "test-head", "text": ""}).status_code == 422


def test_speak_reports_the_tts_failing(assets) -> None:
    with (
        fake_tts(tone(0.2), status=500) as (tts, _),
        TestClient(create_app(Settings(assets_dir=assets, tts_url=tts))) as c,
    ):
        failed = c.post("/speak", json={"face": "test-head", "text": "Hello there."})
        assert failed.status_code == 502 and "500" in failed.json()["detail"]
        assert c.get("/health").json()["live"] == []
