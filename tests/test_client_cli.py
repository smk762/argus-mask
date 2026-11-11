from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from typer.testing import CliRunner

from argus_mask import client
from argus_mask.cli import app
from argus_mask.config import ConfigError, Settings
from argus_mask.drivers import DriverError, build_driver
from argus_mask.performance import blocks
from argus_mask.server.app import create_app
from tests.helpers import fake_a2f, fake_tts, fake_wyoming, live_server, make_face, make_glb, tone, write_wav

runner = CliRunner()


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "LEAD_S", 60.0)  # tests need not wait for real time
    make_face(tmp_path / "assets", "test-head")
    with live_server(create_app(Settings(assets_dir=tmp_path / "assets"))) as url:
        yield url


def test_schema_is_committed_and_current() -> None:
    result = runner.invoke(app, ["schema", "--check"])
    assert result.exit_code == 0, result.output


def test_drive_writes_the_same_take_twice(tmp_path) -> None:
    wav = write_wav(tmp_path / "hello.wav", tone(1.5))
    digests = []
    for name in ("a.ndjson", "b.ndjson"):
        result = runner.invoke(app, ["drive", str(wav), "--out", str(tmp_path / name)])
        assert result.exit_code == 0, result.output
        digests.append(hashlib.sha256((tmp_path / name).read_bytes()).hexdigest())
    assert digests[0] == digests[1]
    assert "45 frames from 1.50s" in result.output
    assert (
        runner.invoke(app, ["drive", str(tmp_path / "missing.wav"), "--out", str(tmp_path / "c.ndjson")]).exit_code == 1
    )


def test_inspect_writes_a_manifest_and_reports_a_refused_face(tmp_path) -> None:
    directory = tmp_path / "my-head"
    directory.mkdir()
    (directory / "head.glb").write_bytes(make_glb(["jawOpen"]))
    no_provenance = runner.invoke(app, ["inspect", str(directory / "head.glb"), "--write-manifest"])
    assert no_provenance.exit_code == 1 and not (directory / "manifest.json").exists()
    written = runner.invoke(
        app,
        [
            "inspect",
            str(directory / "head.glb"),
            "--write-manifest",
            "--kind",
            "stock",
            "--source",
            "made in a test",
            "--license",
            "CC0-1.0",
            "--distance",
            "0.9",
            "--height",
            "-0.08",
        ],
    )
    assert written.exit_code == 0, written.output
    assert "1 of 52 ARKit shapes" in written.output
    report = json.loads(runner.invoke(app, ["inspect", str(directory), "--json"]).output)
    assert report["manifest"]["id"] == "my-head" and report["present"] == ["jawOpen"] and report["refused"] is None
    assert report["manifest"]["framing"] == {"distance": 0.9, "height": -0.08, "fov": 30.0}
    (directory / "head.glb").write_bytes(make_glb(["jawOpen", "jawLeft"]))
    changed = runner.invoke(app, ["inspect", str(directory)])
    assert changed.exit_code == 1 and "sha256 does not match" in changed.output


def test_push_plays_a_file_into_a_running_server(server, tmp_path) -> None:
    wav = write_wav(tmp_path / "hello.wav", tone(1.0))
    result = runner.invoke(app, ["push", str(wav), "--face", "test-head", "--url", server])
    assert result.exit_code == 0, result.output
    assert "done: 30 frames, 1.00s" in result.output
    refused = runner.invoke(app, ["push", str(wav), "--face", "nobody", "--url", server])
    assert refused.exit_code == 1 and "no face 'nobody'" in refused.output


async def test_a_source_paces_from_its_first_chunk_not_from_connecting(tmp_path, monkeypatch) -> None:
    """A TTS model is slow to produce its first audio; the rest must still go out at the speed of speech."""
    import time

    from argus_mask.models import AudioFormat

    monkeypatch.setattr(client, "LEAD_S", 0.2)
    make_face(tmp_path / "assets", "test-head")
    with live_server(create_app(Settings(assets_dir=tmp_path / "assets"))) as url:
        sent_at: list[float] = []

        async def slow_start():
            await asyncio.sleep(0.5)  # the "model" thinking
            for block in blocks(tone(2.0, 16000), 16000):
                sent_at.append(time.monotonic())
                yield block.astype("<f4").tobytes()

        done = await client.send(url, "test-head", AudioFormat(sample_rate=16000), slow_start())
    assert done["frames"] == 60
    # 2.0 s of audio with a 0.2 s lead takes about 1.8 s to send once the first chunk is out. Before the
    # fix, the 0.5 s wait counted as time already spent and everything went out at once. The bound leaves
    # 0.6 s for a slow runner's jitter, and is still far above a burst (#5).
    assert sent_at[-1] - sent_at[0] > 1.2


def test_say_streams_tts_audio_into_the_server(server) -> None:
    with fake_tts(tone(1.0, 24000), 24000) as (tts_url, seen):
        result = runner.invoke(
            app,
            [
                "say",
                "Hello there.",
                "--face",
                "test-head",
                "--url",
                server,
                "--tts-url",
                tts_url,
                "-p",
                "speed=1.1",
                "-p",
                "emotion=calm",
            ],
        )
    assert result.exit_code == 0, result.output
    # The utterance is 1.00 s; the take is longer by the silence put either side of it.
    assert "speech: 1.00s, playing" in result.output and "done: 44 frames, 1.45s" in result.output
    assert seen == [{"speed": 1.1, "emotion": "calm", "text": "Hello there."}]
    with fake_tts(tone(1.0, 24000), 24000) as (tts_url, _):
        streamed = runner.invoke(
            app, ["say", "Hello again.", "--face", "test-head", "--url", server, "--tts-url", tts_url, "--stream"]
        )
    assert streamed.exit_code == 0 and "done: 44 frames, 1.45s" in streamed.output


def test_sentences_split_where_a_pause_belongs() -> None:
    assert client.sentences("Hello there. How are you?  Fine!") == [
        ("Hello there.", 1),
        ("How are you?", 1),
        ("Fine!", 0),
    ]
    assert client.sentences('He said "Go." Then left.') == [('He said "Go."', 1), ("Then left.", 0)]
    assert client.sentences("One.\n\nTwo. Three.\n") == [("One.", 2), ("Two.", 1), ("Three.", 0)]
    assert client.sentences("No full stop at the end") == [("No full stop at the end", 0)]
    assert client.sentences("Version 2.5 is out. Yes.") == [("Version 2.5 is out.", 1), ("Yes.", 0)]


def test_say_sends_one_sentence_at_a_time_with_pauses_between(server) -> None:
    with fake_tts(tone(1.0, 24000), 24000) as (tts_url, seen):
        result = runner.invoke(
            app,
            [
                "say",
                "One. Two!\n\nThree?",
                "--face",
                "test-head",
                "--url",
                server,
                "--tts-url",
                tts_url,
                "-p",
                "x=1",
                "--as-is",
            ],
        )
    assert result.exit_code == 0, result.output
    assert [b["text"] for b in seen] == ["One.", "Two!", "Three?"] and all(b["x"] == 1 for b in seen)
    assert "sentence 1/3" in result.output and "sentence 3/3" in result.output
    # 3 x 1.0 s of speech, 0.3 s after the first sentence, 0.6 s after the paragraph, 0.15 + 0.3 s either side.
    assert "speech: 3.00s, playing" in result.output and "4.35s" in result.output
    with fake_tts(tone(1.0, 24000), 24000) as (tts_url, seen):
        whole = runner.invoke(
            app,
            [
                "say",
                "One. Two!",
                "--face",
                "test-head",
                "--url",
                server,
                "--tts-url",
                tts_url,
                "--pause",
                "0",
                "--as-is",
            ],
        )
    assert whole.exit_code == 0 and [b["text"] for b in seen] == ["One. Two!"] and "1.45s" in whole.output
    with fake_tts(tone(1.0, 24000), 24000) as (tts_url, seen):
        streamed = runner.invoke(
            app,
            ["say", "One. Two!", "--face", "test-head", "--url", server, "--tts-url", tts_url, "--stream", "--as-is"],
        )
    assert streamed.exit_code == 0 and [b["text"] for b in seen] == ["One.", "Two!"] and "2.75s" in streamed.output


def test_say_tidies_the_text_unless_told_not_to(server) -> None:
    text = "Hi. The meeting is at 9:30 a.m. tomorrow. It has 3 items."
    with fake_tts(tone(0.5, 24000), 24000) as (tts_url, seen):
        result = runner.invoke(app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert result.exit_code == 0, result.output
    # The one-word greeting goes with the next sentence; the numbers go as words.
    assert [b["text"] for b in seen] == ["Hi. The meeting is at nine thirty a m tomorrow.", "It has three items."]
    with fake_tts(tone(0.5, 24000), 24000) as (tts_url, seen):
        result = runner.invoke(
            app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url, "--stream"]
        )
    assert result.exit_code == 0, result.output
    # Streamed, the audio is faded but not levelled, and keeps its length: 2 x 0.5 s, a pause, the lead and tail.
    assert [b["text"] for b in seen] == ["Hi. The meeting is at nine thirty a m tomorrow.", "It has three items."]
    assert "1.75s" in result.output


def test_say_picks_the_voice(server, monkeypatch) -> None:
    def spoken(*args: str) -> dict:
        with fake_tts(tone(0.2, 24000), 24000) as (tts_url, seen):
            result = runner.invoke(
                app, ["say", "Hi.", "--face", "test-head", "--url", server, "--tts-url", tts_url, *args]
            )
        assert result.exit_code == 0, result.output
        return seen[0]

    monkeypatch.delenv("ARGUS_MASK_TTS_VOICE", raising=False)
    assert spoken() == {"text": "Hi."}
    # A name is a stored clone, a number one of the service's own speakers; agent-composer's prefix is dropped.
    assert spoken("--voice", "inara-nobility") == {"text": "Hi.", "voice_clone_id": "inara-nobility"}
    assert spoken("--voice", "clone:inara-nobility") == {"text": "Hi.", "voice_clone_id": "inara-nobility"}
    assert spoken("--voice", "1") == {"text": "Hi.", "speaker": "1"}
    # An explicit field wins over the default voice.
    monkeypatch.setenv("ARGUS_MASK_TTS_VOICE", "inara-nobility")
    assert spoken() == {"text": "Hi.", "voice_clone_id": "inara-nobility"}
    assert spoken("-p", "speaker=0") == {"text": "Hi.", "speaker": 0}
    assert spoken("--voice", "2") == {"text": "Hi.", "speaker": "2"}
    # A Wyoming server takes the name as it is.
    with fake_wyoming(tone(0.2, 22050), 22050) as (tts_url, seen):
        result = runner.invoke(app, ["say", "Hi.", "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert result.exit_code == 0, result.output
    assert seen == [{"text": "Hi.", "voice": {"name": "inara-nobility"}}]


def test_say_reports_a_tts_error_and_a_missing_url(server, monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_MASK_TTS_URL", raising=False)
    missing = runner.invoke(app, ["say", "Hello.", "--face", "test-head", "--url", server])
    assert missing.exit_code == 1 and "ARGUS_MASK_TTS_URL is not set" in missing.output
    with fake_tts(tone(0.1), status=422) as (tts_url, _):
        refused = runner.invoke(app, ["say", "Hello.", "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert refused.exit_code == 1 and "answered 422" in refused.output
    empty = runner.invoke(app, ["say", " ", "--face", "test-head", "--url", server, "--tts-url", "http://127.0.0.1:9"])
    assert empty.exit_code == 1 and "nothing to say" in empty.output


def test_a_tts_stream_that_breaks_ends_the_session_cleanly(server) -> None:
    with fake_tts(tone(2.0), break_after=24000) as (tts_url, _):
        result = runner.invoke(app, ["say", "Hello.", "--face", "test-head", "--url", server, "--tts-url", tts_url])
    # Whether the client sees a short read or an error, the face must be free for the next source.
    assert result.exit_code in (0, 1)
    with fake_tts(tone(0.5)) as (tts_url, _):
        again = runner.invoke(app, ["say", "Again.", "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert again.exit_code == 0, again.output


async def test_the_a2f_driver_maps_the_servers_order_and_reports_a_dead_server() -> None:
    with fake_a2f(names=["tongueOut", "jawOpen", "notAnArkitShape"]) as url:
        driver = build_driver("a2f", a2f_url=url)
        assert await driver.check() is None and driver.model == "fake-a2f"
        session = await driver.open(30)
        frames = await session.feed(tone(1.0, 16000))
        frames += await session.finish()
        await session.close()
    assert len(frames) == 30 and frames[3][17] == pytest.approx(0.3) and frames[3].sum() == pytest.approx(0.3)
    dead = build_driver("a2f", a2f_url="http://127.0.0.1:9")
    assert "does not answer" in await dead.check()
    with pytest.raises(DriverError):
        await dead.open(30)
    assert await build_driver("a2f").check() == "ARGUS_MASK_A2F_URL is not set"


def test_settings_reject_an_unknown_driver_and_read_the_environment(monkeypatch) -> None:
    with pytest.raises(ConfigError):
        Settings(driver="telepathy")
    monkeypatch.setenv("ARGUS_MASK_DRIVER", "a2f")
    monkeypatch.setenv("ARGUS_MASK_CORS_ORIGINS", "https://mask.example/, http://localhost:3000")
    monkeypatch.setenv("ARGUS_MASK_READ_ONLY", "1")
    monkeypatch.setenv("ARGUS_MASK_FPS", "60")
    cfg = Settings.from_env()
    assert (cfg.driver, cfg.fps, cfg.read_only) == ("a2f", 60, True)
    assert cfg.cors_origins == ("https://mask.example", "http://localhost:3000")


async def test_cancelling_say_while_the_tts_thinks_returns_at_once(server) -> None:
    """Ctrl-C must not leave the process waiting on the TTS socket until the service gives up."""
    import time

    with fake_tts(tone(1.0), stall_s=5.0) as (tts_url, _):
        task = asyncio.ensure_future(client.say(server, "test-head", tts_url, {"text": "slow"}))
        await asyncio.sleep(0.4)
        started = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The cut socket ends the reader thread at once; `_cancellable` would otherwise wait up to 2 s for it.
        assert time.monotonic() - started < 1.0


def test_say_speaks_through_a_wyoming_server(server) -> None:
    with fake_wyoming(tone(1.0, 22050), 22050) as (tts_url, seen):
        result = runner.invoke(
            app, ["say", "Hello there.", "--face", "test-head", "--url", server, "--tts-url", tts_url, "-p", "voice=0"]
        )
    assert result.exit_code == 0, result.output
    assert "speech: 1.00s, playing" in result.output and "done: 44 frames, 1.45s" in result.output
    assert seen == [{"text": "Hello there.", "voice": {"name": "0"}}]
    with fake_wyoming(tone(0.5), error="no such voice") as (tts_url, _):
        failed = runner.invoke(app, ["say", "Hello.", "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert failed.exit_code == 1 and "no such voice" in failed.output
    bad = runner.invoke(app, ["say", "Hello.", "--face", "test-head", "--url", server, "--tts-url", "ftp://x"])
    assert bad.exit_code == 1 and "wyoming://" in bad.output


def test_say_sends_the_previous_sentence_as_context(server) -> None:
    import base64  # noqa: PLC0415

    from argus_mask.models import AudioFormat  # noqa: PLC0415
    from argus_mask.speech import tidy_audio  # noqa: PLC0415

    F32_24K = AudioFormat(sample_rate=24000, format="f32le", channels=1)  # noqa: N806

    pcm = tone(0.5, 24000)
    text = "This is the first sentence here. And this is the second one. Then a third to finish."
    for extra in (["--context"], ["--context", "--stream"]):
        with fake_tts(pcm, 24000) as (tts_url, seen):
            result = runner.invoke(
                app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url, *extra]
            )
        assert result.exit_code == 0, result.output
        assert "context" not in seen[0]
        for before, now in zip(seen, seen[1:], strict=False):
            (ctx,) = now["context"]
            assert ctx["text"] == before["text"]
            # the audio the service sent, levelled (sent as it came, a briefing sank sentence by sentence)
            assert base64.b64decode(ctx["audio"]) == tidy_audio(pcm.astype("<f4").tobytes(), F32_24K)
    with fake_tts(pcm, 24000) as (tts_url, seen):
        off = runner.invoke(app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert off.exit_code == 0 and all("context" not in s for s in seen)


def test_a_service_that_refuses_context_is_asked_again_without_it(server) -> None:
    text = "This is the first sentence here. And this is the second one. Then a third to finish."
    with fake_tts(tone(0.5, 24000), 24000, refuse_context=True) as (tts_url, seen):
        result = runner.invoke(
            app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url, "--context"]
        )
    assert result.exit_code == 0, result.output
    # the second sentence is refused with context, asked again without; the third goes without
    assert ["context" in s for s in seen] == [False, True, False, False]
    assert [s["text"] for s in seen if "context" not in s] == [
        "This is the first sentence here.",
        "And this is the second one.",
        "Then a third to finish.",
    ]


def test_a_wyoming_service_gets_no_context(server) -> None:
    text = "This is the first sentence here. And this is the second one."
    with fake_wyoming(tone(0.3, 22050), 22050) as (tts_url, seen):
        result = runner.invoke(
            app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url, "--context"]
        )
    assert result.exit_code == 0, result.output
    assert len(seen) == 2 and all("context" not in s for s in seen)


def test_say_refuses_more_speech_than_it_will_buffer(server, monkeypatch) -> None:
    monkeypatch.setattr(client, "MAX_BUFFERED_S", 1.5)
    text = "This is the first sentence here. And this is the second one."
    with fake_tts(tone(1.0, 24000), 24000) as (tts_url, _):
        result = runner.invoke(app, ["say", text, "--face", "test-head", "--url", server, "--tts-url", tts_url])
    # 1.0 s fits, the second 1.0 s passes the 1.5 s cap: refused before anything plays (#3)
    assert result.exit_code == 1 and "more than 0.025 minutes of speech" in result.output
