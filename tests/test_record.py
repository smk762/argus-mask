"""`record`: a session captured from the viewer socket, laid out as a take and a WAV, and rendered to a clip."""

from __future__ import annotations

import asyncio
import shutil
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


async def _push_after(url: str, delay: float, seconds: float) -> dict:
    await asyncio.sleep(delay)
    pcm = tone(seconds, 16000)

    async def chunks():
        for start in range(0, pcm.size, 1600):
            yield pcm[start : start + 1600].astype("<f4").tobytes()

    return await client.send(url, "test-head", AudioFormat(sample_rate=16000), chunks())


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
