from __future__ import annotations

import hashlib
import json

import pytest
from typer.testing import CliRunner

from argus_mask import client
from argus_mask.cli import app
from argus_mask.config import ConfigError, Settings
from argus_mask.drivers import DriverError, build_driver
from argus_mask.server.app import create_app
from tests.helpers import fake_a2f, fake_tts, live_server, make_face, make_glb, tone, write_wav

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
    assert "done: 30 frames, 1.00s" in result.output  # the audio arrived whole, at the rate the service stated
    assert seen == [{"speed": 1.1, "emotion": "calm", "text": "Hello there."}]


def test_say_reports_a_tts_error_and_a_missing_url(server, monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_MASK_TTS_URL", raising=False)
    missing = runner.invoke(app, ["say", "Hello.", "--face", "test-head", "--url", server])
    assert missing.exit_code == 1 and "ARGUS_MASK_TTS_URL is not set" in missing.output
    with fake_tts(tone(0.1), status=422) as (tts_url, _):
        refused = runner.invoke(app, ["say", "", "--face", "test-head", "--url", server, "--tts-url", tts_url])
    assert refused.exit_code == 1 and "answered 422" in refused.output


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
