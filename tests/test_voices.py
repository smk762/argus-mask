"""The Voices page's store: where a made voice's record and clips are written."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus_mask.editor.voices import VoiceError, VoiceStore


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
