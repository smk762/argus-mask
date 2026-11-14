"""The editor's Voices page: check clips, generate a candidate, make a voice with its provenance, try it, delete it."""

from __future__ import annotations

import io
import json
import wave
from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from tests.helpers import fake_tts, write_consent
from tests.test_voice import TEXT, speechlike

RELEASE = "ab" * 32


def wav(samples: np.ndarray, rate: int = 48000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.fixture
def voices_page(tmp_path):
    @contextmanager
    def run(tts_url: str = "", **tts_kwargs):
        ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
        with TestClient(create_editor_app(Settings(assets_dir=tmp_path / "assets", tts_url=tts_url), ed)) as c:
            yield c

    return run


def check(c, samples, transcript=TEXT, name="me.wav") -> dict:
    reply = c.post("/voice/check", params={"name": name, "transcript": transcript}, content=wav(samples))
    assert reply.status_code == 200, reply.text
    return reply.json()


def make_body(sha: str, **over) -> dict:
    return {
        "id": "ginger-warm",
        "name": "Ginger",
        "delivery": "warm",
        "clips": [{"sha": sha, "name": "me.wav", "transcript": TEXT}],
        "source": {"kind": "own", "confirmed": True},
    } | over


def test_a_clip_is_checked_and_kept_by_its_hash(voices_page, tmp_path) -> None:
    with voices_page() as c:
        good = check(c, speechlike())
        assert good["report"]["verdict"] == "good" and len(good["sha"]) == 64
        assert (tmp_path / "voices" / ".staged" / f"{good['sha']}.wav").is_file()
        served = c.get(f"/voice/clip/{good['sha']}")
        assert served.status_code == 200 and served.headers["content-type"] == "audio/wav"
        assert c.get("/voice/clip/../../etc/passwd").status_code == 404
        assert c.get(f"/voice/clip/{'0' * 64}").status_code == 404
        noisy = check(c, speechlike(noise_db=-38.0), transcript="")
        assert noisy["report"]["verdict"] == "not suitable"
        both = c.post("/voice/assess", json=[good["report"], noisy["report"]]).json()
        assert both["metrics"]["usable"] == 1
        assert c.post("/voice/check", content=b"not a wav").status_code == 422
        assert c.post("/voice/assess", json=[{"nope": 1}]).status_code == 422
    assert "Voices" in c.get("/voices.html").text


def test_making_a_voice_needs_its_provenance(voices_page) -> None:
    with (
        fake_tts(speechlike(1.0, 24000), 24000, voices={"clones": {"clones": []}}) as (tts, seen),
        voices_page(tts) as c,
    ):
        sha = check(c, speechlike())["sha"]
        for source, why in [
            ({"kind": "own"}, "confirm"),
            ({"kind": "film", "confirmed": True}, "one of"),
            ({"kind": "generated", "confirmed": True}, "names what made it"),
            ({"kind": "consented", "confirmed": True, "speaker": "A. Person"}, "sha256"),
            ({"kind": "consented", "confirmed": True, "release_sha256": RELEASE}, "name the consenting speaker"),
            ({"kind": "personal"}, "where the clips come from"),
        ]:
            refused = c.post("/voice/create", json=make_body(sha, source=source))
            assert refused.status_code == 422 and why in refused.json()["detail"], (source, refused.text)
        assert not [s for s in seen if "clone" in s]  # nothing reached the TTS service


def test_a_personal_voice_records_where_it_came_from_and_is_not_cleared(voices_page) -> None:
    clones = {"clones": []}
    with fake_tts(speechlike(1.0, 24000), 24000, voices={"clones": clones}) as (tts, _), voices_page(tts) as c:
        sha = check(c, speechlike())["sha"]
        made = c.post("/voice/create", json=make_body(sha, source={"kind": "personal", "origin": "a game character"}))
        assert made.status_code == 200, made.text
        assert made.json()["source"] == {"kind": "personal", "origin": "a game character", "cleared": False}


def test_a_voice_is_made_recorded_tried_and_deleted(voices_page, tmp_path) -> None:
    clones = {"clones": [{"companion_id": "inara-nobility", "name": "Inara"}]}
    write_consent(tmp_path / "assets")
    with fake_tts(speechlike(1.0, 24000), 24000, voices={"clones": clones}) as (tts, seen), voices_page(tts) as c:
        sha = check(c, speechlike())["sha"]
        source = {"kind": "consented", "confirmed": True, "speaker": "A. Person", "release_sha256": RELEASE}
        made = c.post("/voice/create", json=make_body(sha, source=source))
        assert made.status_code == 200, made.text
        record = made.json()
        assert record["source"] == {
            "kind": "consented",
            "release_sha256": RELEASE,
            "speaker": "A. Person",
            "confirmed": True,
            "cleared": True,
        }
        assert (
            record["clips"][0]["sha256"] == sha
            and record["delivery"] == "warm"
            and record["check"]["verdict"] == "good"
        )
        sent = next(s for s in seen if "clone" in s)
        assert sent["clone"] == {"reference_text": TEXT, "companion_id": "ginger-warm", "voice_name": "Ginger"}
        assert sent["files"] == [("me.wav", len(wav(speechlike())))]
        folder = tmp_path / "voices" / "ginger-warm"
        assert (
            json.loads((folder / "voice.json").read_text())["id"] == "ginger-warm" and (folder / "clip-1.wav").is_file()
        )
        assert [v["id"] for v in c.get("/voice/state").json()["voices"]] == ["ginger-warm"]
        # the same id again, or one the TTS service already has, is refused
        assert c.post("/voice/create", json=make_body(sha, source=source)).status_code == 409
        taken = c.post("/voice/create", json=make_body(sha, id="inara-nobility", source=source))
        assert taken.status_code == 409 and "already has" in taken.json()["detail"]

        tried = c.post(
            "/voice/try", json={"voice": "ginger-warm", "text": "It is 9:30. Shall we start?", "temperature": 0.7}
        )
        assert tried.status_code == 200 and tried.content[:4] == b"RIFF"
        asked = [s for s in seen if "text" in s]
        assert asked[-1] == {
            "text": "It is nine thirty. Shall we start?",
            "voice_clone_id": "ginger-warm",
            "temperature": 0.7,
        }

        gone = c.delete("/voice/ginger-warm")
        assert gone.status_code == 200 and gone.json()["removed_from_tts"] and {"delete": "ginger-warm"} in seen
        assert not folder.exists() and any((tmp_path / "voices" / ".trash").iterdir())
        assert c.delete("/voice/inara-nobility").status_code == 404  # not made here: not deleted here


def test_an_unsuitable_clip_is_not_made_into_a_voice(voices_page) -> None:
    with (
        fake_tts(speechlike(1.0, 24000), 24000, voices={"clones": {"clones": []}}) as (tts, seen),
        voices_page(tts) as c,
    ):
        sha = check(c, speechlike(noise_db=-38.0))["sha"]
        refused = c.post("/voice/create", json=make_body(sha))
        assert refused.status_code == 422 and "Background noise" in refused.json()["detail"]
        assert not [s for s in seen if "clone" in s]


def test_a_synthetic_candidate_is_generated_and_checked(voices_page) -> None:
    import time  # noqa: PLC0415

    script = "Good morning. I keep an eye on the systems, and I read out what matters. Shall we start?"
    with fake_tts(speechlike(6.0, 24000, lead=0.0, tail=0.0), 24000) as (tts, seen), voices_page(tts) as c:
        job = c.post("/voice/generate", json={"script": script, "speaker": "1", "temperature": 0.8}).json()
        assert c.post("/voice/generate", json={"script": script}).status_code == 409  # one at a time
        end = time.monotonic() + 10
        while job["state"] == "running" and time.monotonic() < end:
            time.sleep(0.05)
            job = c.get(f"/voice/generate/{job['id']}").json()
        assert job["state"] == "done", job
        assert seen[0] == {"text": script, "speaker": "1", "temperature": 0.8}
        result = job["result"]
        assert result["transcript"] == script and len(result["sha"]) == 64
        # padded with quiet, so its edges pass
        assert next(ch for ch in result["report"]["checks"] if ch["id"] == "edges")["verdict"] == "good"
        assert c.get(f"/voice/clip/{result['sha']}").status_code == 200
        described = c.post("/voice/generate", json={"script": script, "description": "warm, calm, alto"}).json()
        while described["state"] == "running":
            time.sleep(0.05)
            described = c.get(f"/voice/generate/{described['id']}").json()
        assert seen[-1] == {"text": script, "voice_description": "warm, calm, alto"}


def test_a_wyoming_service_cannot_make_voices(voices_page) -> None:
    with voices_page("wyoming://127.0.0.1:1") as c:
        state = c.get("/voice/state").json()
        assert not state["can_make"] and "HTTP TTS" in state["reason"]
        sha = check(c, speechlike())["sha"]
        assert c.post("/voice/create", json=make_body(sha)).status_code == 422


def test_another_site_may_not_point_the_voices_page_at_a_host(voices_page) -> None:
    with voices_page("http://127.0.0.1:9/synthesise") as c:
        evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
        assert (
            c.post(
                "/voice/try", json={"voice": "x", "text": "hi", "tts_url": "http://10.0.0.1/synthesise"}, headers=evil
            ).status_code
            == 403
        )  # noqa: E501
        assert c.post("/voice/check", content=wav(speechlike(2.0)), headers=evil).status_code == 403
