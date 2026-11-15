"""`record`: a session captured from the viewer socket, laid out as a take and a WAV, and rendered to a clip."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from argus_mask import client, record
from argus_mask.config import Settings
from argus_mask.models import AudioFormat
from argus_mask.server.app import create_app
from argus_mask.take import read_take
from tests.helpers import live_server, make_face, tone


@pytest.fixture
def server(tmp_path: Path):
    make_face(tmp_path / "assets", "test-head")
    with live_server(create_app(Settings(assets_dir=tmp_path / "assets"))) as url:
        yield url


async def _push_after(url: str, delay: float, seconds: float, face: str = "test-head") -> dict:
    await asyncio.sleep(delay)
    pcm = tone(seconds, 16000)

    async def chunks():
        for start in range(0, pcm.size, 1600):
            yield pcm[start : start + 1600].astype("<f4").tobytes()

    return await client.send(url, face, AudioFormat(sample_rate=16000), chunks())


def test_capture_keeps_the_next_whole_session(server) -> None:
    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.3, 1.0))
        captured = await record.capture(server, "test-head")
        await pushed
        return captured

    rec = asyncio.run(run())
    assert len(rec.frames) == 30 and rec.end is not None and rec.end.frames == 30
    assert [f.seq for f in rec.frames] == list(range(30))
    assert rec.audio and rec.audio[0][0] == 0.0 and sum(x.size for _, x in rec.audio) == 16000
    assert rec.duration_ms == pytest.approx(1000.0)


def test_wav_and_take_are_laid_out_on_the_timeline(server, tmp_path: Path) -> None:
    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.3, 0.5))
        captured = await record.capture(server, "test-head")
        await pushed
        return captured

    rec = asyncio.run(run())
    # A chunk the server never sent (a gap) is silence in the WAV, and the audio ends where the frames do.
    rec.audio = [rec.audio[0], (400.0, rec.audio[-1][1])]
    record.write_wav(tmp_path / "a.wav", rec)
    with wave.open(str(tmp_path / "a.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert pcm.size == 8000
    assert np.abs(pcm[1700:6300]).max() == 0 and np.abs(pcm[:1600]).max() > 1000 and np.abs(pcm[6400:]).max() > 1000
    record.write_take(tmp_path / "a.ndjson", rec)
    header, frames, end = read_take(tmp_path / "a.ndjson")
    assert header.face_id == "test-head" and len(frames) == 15 and end is not None and end.frames == 15


def _can_render() -> bool:
    try:
        import playwright  # noqa: F401, PLC0415
    except ImportError:
        return False
    try:
        record.find_ffmpeg()
    except record.RecordError:
        return False
    return True


@pytest.mark.skipif(not _can_render(), reason="needs playwright with a browser, and ffmpeg")
def test_record_renders_a_clip(server, tmp_path: Path) -> None:
    out = tmp_path / "clips" / "test.mp4"

    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.5, 0.2))
        try:
            rec = await record.record(server, "test-head", out, width=160, height=120, keep=True)
        except record.RecordError as exc:
            if "no browser" in str(exc):
                pytest.skip(str(exc))
            raise
        await pushed
        return rec

    rec = asyncio.run(run())
    assert len(rec.frames) == 6
    assert out.exists() and out.stat().st_size > 0
    assert sorted(p.name for p in (tmp_path / "clips" / "test.frames").iterdir()) == [f"{i:06d}.png" for i in range(6)]
    assert (tmp_path / "clips" / "test.wav").exists() and (tmp_path / "clips" / "test.take.ndjson").exists()
    probe = shutil.which("ffprobe")
    if probe:
        import subprocess  # noqa: PLC0415

        streams = subprocess.run(
            [probe, "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        assert sorted(streams) == ["audio", "video"]


def test_an_unknown_quality_is_refused_before_waiting(tmp_path: Path) -> None:
    # No server is listening: the refusal must come before record() reaches for one.
    with pytest.raises(record.RecordError, match="quality 'ultra'"):
        asyncio.run(record.record("http://127.0.0.1:9", "x", tmp_path / "c.mp4", quality="ultra"))


@pytest.mark.parametrize(("given", "sent"), [({}, "high"), ({"quality": "low"}, "low")])
def test_the_viewer_is_asked_for_the_look(server, tmp_path: Path, monkeypatch, given, sent) -> None:
    urls = []

    async def render(page_url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003 - stands in for the browser
        urls.append(page_url)

    monkeypatch.setattr(record, "render_frames", render)
    monkeypatch.setattr(record, "mux", lambda *a, **k: None)

    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.5, 0.2))
        await record.record(server, "test-head", tmp_path / "c.mp4", width=160, height=120, **given)
        await pushed

    asyncio.run(run())
    assert f"quality={sent}" in urls[0] and "record=1" in urls[0]


def test_an_unknown_scene_is_refused_before_waiting(tmp_path: Path) -> None:
    with pytest.raises(record.RecordError, match="scene 'warp'"):
        asyncio.run(record.record("http://127.0.0.1:9", "x", tmp_path / "c.mp4", scene="warp"))


def test_record_knows_the_viewers_scenes() -> None:
    backdrop = (Path(record.__file__).parent / "viewer" / "backdrop.js").read_text()
    table = backdrop.split("export const SCENES = {", 1)[1].split("};", 1)[0]
    assert tuple(re.findall(r"^\s*([a-z][a-z0-9-]*):", table, re.M)) == record.SCENES
    scenes = Path(record.__file__).parent / "viewer" / "scenes"
    for name in record.SCENES:  # a procedural scene's module, or a plate's directory
        assert (scenes / f"{name}.js").is_file() or (scenes / name / "background.json").is_file()


@pytest.mark.parametrize("scene", ["ftl", "none"])
def test_the_viewer_is_asked_for_the_scene(server, tmp_path: Path, monkeypatch, scene: str) -> None:
    urls = []

    async def render(page_url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003 - stands in for the browser
        urls.append(page_url)

    monkeypatch.setattr(record, "render_frames", render)
    monkeypatch.setattr(record, "mux", lambda *a, **k: None)

    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.5, 0.2))
        await record.record(server, "test-head", tmp_path / "c.mp4", width=160, height=120, scene=scene)
        await pushed

    asyncio.run(run())
    assert f"scene={scene}" in urls[0]


def test_the_video_check_refuses_when_the_server_cannot_check(monkeypatch) -> None:
    import io
    import urllib.error

    def fail(code: int):  # noqa: ANN202 - stands in for urlopen
        def urlopen(url, timeout):  # noqa: ANN001, ANN202
            raise urllib.error.HTTPError(url, code, "x", {}, io.BytesIO(b""))

        return urlopen

    monkeypatch.setattr(record.urllib.request, "urlopen", fail(500))
    assert "could not check" in record.video_refusal("http://127.0.0.1:9", "someone")
    monkeypatch.setattr(record.urllib.request, "urlopen", fail(404))
    assert record.video_refusal("http://127.0.0.1:9", "someone") is None


@pytest.mark.skipif(not _can_render(), reason="needs playwright with a browser, and ffmpeg")
@pytest.mark.parametrize("scene", ["", *record.SCENES])
def test_two_recordings_of_one_take_match_at_high(server, tmp_path: Path, scene: str) -> None:
    # high adds film grain, whose noise once followed the composer's wall clock: how long the browser took (#81).
    # A backdrop runs on the posed frame's time too, a plate's images loaded before the first pose.
    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.5, 0.2))
        rec = await record.capture(server, "test-head")
        await pushed
        page = f"{server}/?face=test-head&record=1&hud=0&display=flat&quality=high" + (
            f"&scene={scene}" if scene else ""
        )
        try:
            for n in ("a", "b"):
                await record.render_frames(page, rec, tmp_path / n, width=160, height=120)
        except record.RecordError as exc:
            if "no browser" in str(exc):
                pytest.skip(str(exc))
            raise
        return rec

    rec = asyncio.run(run())
    names = [f"{f.seq:06d}.png" for f in rec.frames]
    assert len(names) == 6
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_the_viewer_is_asked_for_the_framing(server, tmp_path: Path, monkeypatch) -> None:
    urls = []

    async def render(page_url, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003 - stands in for the browser
        urls.append(page_url)

    monkeypatch.setattr(record, "render_frames", render)
    monkeypatch.setattr(record, "mux", lambda *a, **k: None)

    async def run():
        pushed = asyncio.create_task(_push_after(server, 0.5, 0.2))
        await record.record(server, "test-head", tmp_path / "c.mp4", width=160, height=120, framing="waist")
        await pushed

    asyncio.run(run())
    assert "framing=waist" in urls[0]
    with pytest.raises(record.RecordError, match="framing 'feet'"):
        asyncio.run(record.record("http://127.0.0.1:9", "x", tmp_path / "c.mp4", framing="feet"))


REAL_HEAD = Path(__file__).resolve().parent.parent / "assets" / "stock-01"


@pytest.mark.skipif(not _can_render(), reason="needs playwright with a browser, and ffmpeg")
@pytest.mark.skipif(not (REAL_HEAD / "manifest.json").is_file(), reason="needs a real head: make head (stock-01)")
@pytest.mark.parametrize("framing", ["head", "waist"])
def test_a_real_head_records_the_same_twice(tmp_path: Path, framing: str) -> None:
    # The one-triangle fixture cannot show what a real head does: textures, alpha-tested and blended parts, a rig,
    # a body (#108). The GPU still blends a few pixels of the blended layers (hair fringe, lashes, brows) one level
    # apart between runs; anything more is a bug (a stale frame was hundreds of pixels, by up to 90).
    assets = tmp_path / "assets"
    shutil.copytree(REAL_HEAD, assets / "stock-01")
    manifest = json.loads((assets / "stock-01" / "manifest.json").read_text())
    if framing != "head" and framing not in manifest.get("framings", {}):
        pytest.skip(f"stock-01 here has no {framing} framing: fetch it again (scripts/fetch_stock_head.py)")
    with live_server(create_app(Settings(assets_dir=assets))) as url:

        async def run():
            pushed = asyncio.create_task(_push_after(url, 0.5, 1.2, face="stock-01"))
            rec = await record.capture(url, "stock-01")
            await pushed
            page = f"{url}/?face=stock-01&record=1&hud=0&display=flat&quality=high&framing={framing}"
            try:
                for n in ("a", "b"):
                    await record.render_frames(page, rec, tmp_path / n, width=640, height=480)
            except record.RecordError as exc:
                if "no browser" in str(exc):
                    pytest.skip(str(exc))
                raise
            return rec

        rec = asyncio.run(run())

    def pixels(path: Path) -> np.ndarray:
        raw = subprocess.run(
            [record.find_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True,
            check=True,
        ).stdout
        return np.frombuffer(raw, np.uint8).reshape(480, 640, 3).astype(int)

    for frame in rec.frames:
        name = f"{frame.seq:06d}.png"
        diff = np.abs(pixels(tmp_path / "a" / name) - pixels(tmp_path / "b" / name)).max(axis=2)
        assert diff.max() <= 2 and (diff > 0).sum() <= 50, (
            f"{name}: {(diff > 0).sum()} pixels differ, by up to {diff.max()}"
        )
