"""The Voices page's store: where a made voice's record and clips are written."""

from __future__ import annotations

import asyncio
import json
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


def test_a_copy_that_fails_leaves_the_voice_as_it_was(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    store.write({"id": "holly", "v": 1}, clips(tmp_path, 2))
    with pytest.raises(OSError):  # the second clip was evicted from .staged after it was checked
        store.write({"id": "holly", "v": 2}, [clips(tmp_path, 1)[0], tmp_path / "evicted.wav"])
    folder = tmp_path / "voices" / "holly"
    assert sorted(p.name for p in folder.iterdir()) == ["clip-1.wav", "clip-2.wav", "voice.json"]
    assert [(folder / f"clip-{i}.wav").read_bytes() for i in (1, 2)] == [b"RIFF\x00", b"RIFF\x01"]
    assert store.record("holly") == {"id": "holly", "v": 1}


def test_a_voice_resaved_from_its_own_clips_keeps_them(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    store.write({"id": "holly"}, clips(tmp_path, 2))
    folder = tmp_path / "voices" / "holly"
    store.write({"id": "holly"}, [folder / "clip-2.wav", folder / "clip-1.wav"])  # reordered
    assert [(folder / f"clip-{i}.wav").read_bytes() for i in (1, 2)] == [b"RIFF\x01", b"RIFF\x00"]


def test_saves_of_one_voice_at_once_each_land_whole(tmp_path: Path) -> None:
    store = VoiceStore(tmp_path / "voices")
    sets: dict[int, list[Path]] = {}
    for n in (1, 2, 3, 4):
        (tmp_path / str(n)).mkdir()
        sets[n] = clips(tmp_path / str(n), n)
    errors: list[BaseException] = []

    def save(n: int) -> None:
        try:
            for _ in range(20):
                store.write({"id": "holly", "clips": n}, sets[n])
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(n,)) for n in sets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    folder = tmp_path / "voices" / "holly"
    n = store.record("holly")["clips"]
    assert sorted(p.name for p in folder.iterdir()) == [*(f"clip-{i + 1}.wav" for i in range(n)), "voice.json"]


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


def test_trashing_a_voice_waits_for_its_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """trash() takes the same per-voice lock as write(), so it cannot move the folder mid-save."""
    store = VoiceStore(tmp_path / "voices")
    store.write({"id": "holly"}, clips(tmp_path, 1))
    writing = threading.Event()
    release = threading.Event()
    original = VoiceStore._write

    def slow(folder: Path, record: dict, saved: list[Path]) -> None:
        writing.set()
        assert release.wait(5)
        original(folder, record, saved)

    monkeypatch.setattr(VoiceStore, "_write", staticmethod(slow))
    saver = threading.Thread(target=lambda: store.write({"id": "holly", "n": 2}, clips(tmp_path, 2)))
    saver.start()
    assert writing.wait(2)
    done: list[bool] = []
    trasher = threading.Thread(target=lambda: done.append(store.trash("holly")))
    trasher.start()
    try:
        trasher.join(0.2)
        assert trasher.is_alive()
    finally:
        release.set()
        trasher.join(2)
        saver.join(2)
    assert done == [True]
    assert not (tmp_path / "voices" / "holly").exists()
    kept = list((tmp_path / "voices" / ".trash").iterdir())
    assert len(kept) == 1 and json.loads((kept[0] / "voice.json").read_text())["n"] == 2


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
