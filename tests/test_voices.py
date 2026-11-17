"""The Voices page's store: where a made voice's record and clips are written."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from argus_mask.editor import voices
from argus_mask.editor.voices import Generations, VoiceError, VoiceStore


def clips(tmp_path: Path, n: int) -> list[Path]:
    out = []
    for i in range(n):
        path = tmp_path / f"in-{i}.wav"
        path.write_bytes(b"RIFF" + bytes([i]))
        out.append(path)
    return out


@pytest.mark.parametrize("bad", ["../escaped", "/tmp/escaped", "Upper", "", None])
def test_a_voice_is_written_only_under_a_voice_id(tmp_path: Path, bad: object) -> None:
    store = VoiceStore(tmp_path / "voices")
    with pytest.raises(VoiceError):
        store.write({"id": bad}, clips(tmp_path, 1))
    assert not (tmp_path / "escaped").exists()


def test_rewriting_a_voice_with_fewer_clips_keeps_none_of_the_old(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    store.write({"id": "holly"}, clips(tmp_path, 3))
    store.write({"id": "holly"}, clips(tmp_path, 1))
    folder = tmp_path / "voices" / "holly"
    assert sorted(p.name for p in folder.glob("clip-*.wav")) == ["clip-1.wav"]
    assert store.record("holly") == {"id": "holly"}


def test_staging_skips_a_clip_another_request_evicted_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VoiceStore(tmp_path / "voices")
    store.stage(b"RIFF first")
    ghost = store.staged / f"{'0' * 64}.wav"  # listed, then unlinked by another request before it is stat'ed
    listed = type(store.staged).glob
    monkeypatch.setattr(type(store.staged), "glob", lambda self, pattern: [*listed(self, pattern), ghost])
    sha = store.stage(b"RIFF second")
    assert store.clip(sha).read_bytes() == b"RIFF second"


def test_staging_writes_through_a_temp_name_of_its_own(tmp_path: Path) -> None:
    import hashlib  # noqa: PLC0415

    store = VoiceStore(tmp_path / "voices")
    data = b"RIFF same clip"
    sha = hashlib.sha256(data).hexdigest()
    store.staged.mkdir(parents=True)
    (store.staged / f"{sha}.part").mkdir()  # another writer's temp, mid-write
    assert store.stage(data) == sha and store.clip(sha).read_bytes() == data


def test_a_voice_deleted_twice_in_one_second_is_kept_both_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = VoiceStore(tmp_path / "voices")
    monkeypatch.setattr(voices.time, "strftime", lambda fmt: "20260101-000000")
    for n in (3, 1):
        store.write({"id": "holly"}, clips(tmp_path, n))
        assert store.trash("holly")
    assert len(list((tmp_path / "voices" / ".trash").iterdir())) == 2


async def test_a_stopped_generation_does_not_block_the_next(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    reached = threading.Event()

    def blocked(url: str, body: dict) -> None:
        reached.set()
        release.wait(5)
        raise voices.client.ClientError("released")

    monkeypatch.setattr(voices.client, "open_tts", blocked)
    gens = Generations(VoiceStore(tmp_path / "voices"))
    try:
        job = gens.start("http://tts", {"text": "hi"})
        await asyncio.to_thread(reached.wait, 5)  # cancelled mid-run
        await gens.stop_all()
        assert job.state == "failed" and job.error == "cancelled" and job.ended is not None
        early = gens.start("http://tts", {"text": "hi"})  # cancelled before it began
        await gens.stop_all()
        assert early.state == "failed"
        gens.start("http://tts", {"text": "hi"})  # not refused as "already being generated"
        await gens.stop_all()
    finally:
        release.set()


async def test_a_reply_opened_after_its_generation_was_stopped_is_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, reached, closed = threading.Event(), threading.Event(), threading.Event()

    class Reply:
        def close(self) -> None:
            closed.set()

    def slow_open(url: str, body: dict) -> Reply:
        reached.set()
        release.wait(5)  # still connecting when the generation is stopped
        return Reply()

    monkeypatch.setattr(voices.client, "open_tts", slow_open)
    gens = Generations(VoiceStore(tmp_path / "voices"))
    try:
        gens.start("http://tts", {"text": "hi"})
        await asyncio.to_thread(reached.wait, 5)
        await gens.stop_all()
    finally:
        release.set()  # the open completes after the stop
    assert await asyncio.to_thread(closed.wait, 5), "the reply opened after the stop was never closed"
