from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from argus_mask.arkit import ARKIT_52, INDEX
from argus_mask.audio import AudioError, PcmDecoder, Resampler, read_wav
from argus_mask.drivers import build_driver
from argus_mask.idle import IdleLayer
from argus_mask.models import AudioFormat, StreamHeader
from argus_mask.performance import Performer, perform
from argus_mask.take import TakeError, read_take, render_take
from tests.helpers import tone, write_wav


def test_arkit_names_are_the_52() -> None:
    assert len(ARKIT_52) == len(set(ARKIT_52)) == 52


def test_decoder_carries_a_sample_split_across_chunks() -> None:
    samples = np.linspace(-0.5, 0.5, 10, dtype="<f4")
    raw = samples.tobytes()
    decoder = PcmDecoder(AudioFormat(sample_rate=16000))
    out = np.concatenate([decoder.feed(raw[:7]), decoder.feed(raw[7:22]), decoder.feed(raw[22:])])
    assert np.array_equal(out, samples)


def test_decoder_mixes_stereo_s16_to_mono_and_cleans_bad_floats() -> None:
    stereo = np.array([[16384, -16384], [32767, 32767]], dtype="<i2")
    out = PcmDecoder(AudioFormat(sample_rate=16000, channels=2, format="s16le")).feed(stereo.tobytes())
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(1.0, abs=1e-4)
    bad = np.array([np.nan, np.inf, -np.inf, 5.0], dtype="<f4")
    assert PcmDecoder(AudioFormat(sample_rate=16000)).feed(bad.tobytes()).tolist() == [0.0, 1.0, -1.0, 1.0]


def test_resampler_gives_the_driver_rate(tmp_path) -> None:
    resampler = Resampler(24000, 16000)
    out = np.concatenate(
        [resampler.feed(tone(0.5)), resampler.feed(tone(0.5)), resampler.feed(np.zeros(0, dtype=np.float32), last=True)]
    )
    assert abs(out.size - 16000) <= 2


def test_read_wav_reads_int16_and_float32(tmp_path) -> None:
    mono = tone(0.25, 16000)
    got, rate = read_wav(write_wav(tmp_path / "a.wav", mono, 16000))
    assert rate == 16000 and np.allclose(got, mono, atol=1e-3)
    body = mono.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, 16000, 16000 * 4, 4, 32)
    riff = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(body)) + body
    (tmp_path / "f.wav").write_bytes(b"RIFF" + struct.pack("<I", len(riff)) + riff)
    got, rate = read_wav(tmp_path / "f.wav")
    assert rate == 16000 and np.array_equal(got, mono)
    (tmp_path / "bad.wav").write_bytes(b"not a wav file at all")
    with pytest.raises(AudioError):
        read_wav(tmp_path / "bad.wav")


async def test_energy_is_silent_on_silence_and_opens_on_sound() -> None:
    driver = build_driver("energy")
    _, quiet, _ = await perform(np.zeros(16000, dtype=np.float32), 16000, driver, fps=30, seed=0)
    _, loud, _ = await perform(tone(1.0, 16000), 16000, driver, fps=30, seed=0)
    jaw = INDEX["jawOpen"]
    assert max(f.weights[jaw] for f in quiet) == 0.0
    assert max(f.weights[jaw] for f in loud) > 0.2
    assert all(0.0 <= w <= 1.0 for f in loud for w in f.weights)


async def test_a_clip_gives_the_same_take_every_time() -> None:
    audio = tone(2.0)
    takes = [render_take(*await perform(audio, 24000, build_driver("energy"), fps=30, seed=7)) for _ in range(2)]
    assert hashlib.sha256(takes[0].encode()).hexdigest() == hashlib.sha256(takes[1].encode()).hexdigest()


async def test_frames_cover_the_clip_and_time_never_goes_backwards() -> None:
    header, frames, end = await perform(tone(2.0), 24000, build_driver("energy"), fps=30, seed=0)
    assert isinstance(header, StreamHeader) and header.blendshapes == list(ARKIT_52)
    assert header.audio.sample_rate == 24000 and header.audio.format == "f32le"
    assert len(frames) == end.frames == 60 and end.duration_ms == 2000
    assert [f.seq for f in frames] == list(range(60))
    assert all(b.t_ms > a.t_ms for a, b in zip(frames, frames[1:], strict=False))
    assert all(len(f.weights) == 52 for f in frames)


async def test_frame_count_holds_at_a_rate_that_does_not_divide_the_sample_rate() -> None:
    _, frames, end = await perform(tone(3.0, 16000), 16000, build_driver("energy"), fps=24, seed=0)
    assert len(frames) == end.frames == 72


async def test_feeding_in_other_block_sizes_gives_the_same_number_of_frames() -> None:
    audio = tone(1.0, 16000)
    performer = Performer(await build_driver("energy").open(30), in_rate=16000, fps=30, seed=0)
    frames = []
    for start in range(0, audio.size, 777):
        frames += await performer.feed(audio[start : start + 777])
    tail, end = await performer.finish()
    assert len(frames + tail) == end.frames == 30


def test_idle_is_a_function_of_the_seed() -> None:
    zeros = np.zeros(52, dtype=np.float32)

    def run(seed: int) -> list:
        layer = IdleLayer(seed, 30)
        return [(w.tolist(), h.model_dump()) for w, h in (layer.apply(i, zeros) for i in range(600))]

    assert run(3) == run(3)
    assert run(3) != run(4)
    blink = INDEX["eyeBlinkLeft"]
    weights = [w[blink] for w, _ in run(3)]
    assert max(weights) > 0.9 and min(weights) == 0.0  # it blinks, and the eyes reopen
    assert all(0.0 <= v <= 1.0 for w, _ in run(3) for v in w)
    assert max(abs(h["yaw"]) for _, h in run(3)) < 5.0


def test_idle_keeps_a_drivers_own_blink() -> None:
    weights = np.zeros(52, dtype=np.float32)
    weights[INDEX["eyeBlinkLeft"]] = 1.0
    out, _ = IdleLayer(0, 30).apply(0, weights)
    assert out[INDEX["eyeBlinkLeft"]] == 1.0


async def test_take_round_trips_and_a_broken_one_is_refused(tmp_path) -> None:
    header, frames, end = await perform(tone(0.5), 24000, build_driver("energy"), fps=30, seed=1)
    path = tmp_path / "take.ndjson"
    path.write_text(render_take(header, frames, end), encoding="utf-8")
    assert read_take(path) == (header, frames, end)
    (tmp_path / "no-header.ndjson").write_text(frames[0].model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(TakeError):
        read_take(tmp_path / "no-header.ndjson")
    (tmp_path / "v9.ndjson").write_text(header.model_dump_json().replace('"1.0"', '"9.0"') + "\n", encoding="utf-8")
    with pytest.raises(TakeError):
        read_take(tmp_path / "v9.ndjson")
