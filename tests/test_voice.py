"""Checking a reference clip for a cloned voice: each fault found, with what to do; a good clip passes."""

from __future__ import annotations

import numpy as np
import pytest

from argus_mask.voice import analyse, assess


def speechlike(
    seconds: float = 15.0,
    rate: int = 48000,
    *,
    f0: float = 200.0,
    noise_db: float = -75.0,
    gain: float = 0.3,
    lead: float = 0.3,
    tail: float = 0.3,
    pause_at: float | None = None,
    pause_s: float = 0.0,
    cutoff: float | None = None,
    swing: float = 2.0,
) -> np.ndarray:
    """Voiced syllables (harmonics of a drifting pitch) in bursts, with quiet either side and a noise floor."""
    rng = np.random.default_rng(7)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    pitch = f0 * 2 ** (swing * np.sin(2 * np.pi * 0.31 * t) / 12)
    phase = 2 * np.pi * np.cumsum(pitch) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 60) if k * f0 < rate * 0.45)
    syllable = (np.sin(2 * np.pi * 3.2 * t) > -0.3).astype(float)  # ~3 syllables a second, short gaps
    phrase = (np.sin(2 * np.pi * 0.25 * t) > -0.85).astype(float)  # a breath every few seconds
    env = np.convolve(syllable * phrase, np.hanning(int(0.03 * rate)) / (0.015 * rate), mode="same")
    env[: int(lead * rate)] = 0
    env[n - int(tail * rate) :] = 0
    if pause_at is not None:
        env[int(pause_at * rate) : int((pause_at + pause_s) * rate)] = 0
    signal = voice * env
    signal = gain * signal / np.abs(signal).max()
    if cutoff is not None:
        spectrum = np.fft.rfft(signal)
        spectrum[int(cutoff / rate * n) :] = 0
        signal = np.fft.irfft(spectrum, n)
    signal += rng.normal(0, 10 ** (noise_db / 20), n)
    return signal.astype(np.float32)


TEXT = " ".join(["Good morning, this is a calm and even sentence."] * 5)  # 45 words: about 3 a second of 15 s


def verdicts(report) -> dict[str, str]:
    return {c.id: c.verdict for c in report.checks}


def test_a_good_clip_passes() -> None:
    report = analyse(speechlike(), 48000, TEXT)
    assert report.verdict == "good", [(c.label, c.value) for c in report.checks if c.verdict != "good"]
    assert report.score == 100
    m = report.metrics
    assert 10 < m["speech_seconds"] < 15 and m["snr_db"] > 40 and m["bandwidth_hz"] > 10000
    assert 180 < m["pitch_hz"] < 220 and 1.2 < m["pitch_spread_semitones"] < 7


def test_the_same_clip_gives_the_same_report() -> None:
    clip = speechlike(8.0)
    assert analyse(clip, 48000, TEXT).public() == analyse(clip.copy(), 48000, TEXT).public()


@pytest.mark.parametrize(
    ("kwargs", "check", "verdict"),
    [
        ({"noise_db": -38.0}, "noise", "bad"),
        ({"noise_db": -45.0}, "noise", "warn"),
        ({"cutoff": 3400.0}, "bandwidth", "bad"),
        ({"cutoff": 6000.0}, "bandwidth", "warn"),
        ({"gain": 0.03}, "level", "warn"),
        ({"lead": 0.0}, "edges", "warn"),
        ({"tail": 2.5, "seconds": 17.5}, "edges", "warn"),
        ({"pause_at": 6.0, "pause_s": 2.5}, "pauses", "warn"),
        ({"seconds": 4.0}, "length", "warn"),
        ({"seconds": 1.6}, "length", "bad"),
        ({"seconds": 26.0}, "length", "warn"),
        ({"swing": 0.2}, "delivery", "warn"),
        ({"swing": 12.0}, "delivery", "warn"),
    ],
)
def test_each_fault_is_found(kwargs, check, verdict) -> None:
    report = analyse(speechlike(**kwargs), 48000, TEXT)
    found = next((c for c in report.checks if c.id == check), None)
    assert found is not None and found.verdict == verdict, verdicts(report)
    assert verdict == "good" or found.advice


def test_clipping_is_found() -> None:
    loud = np.clip(speechlike() * 6, -1, 1)
    assert verdicts(analyse(loud, 48000, TEXT))["clipping"] == "bad"


def test_a_low_sample_rate_is_found() -> None:
    assert verdicts(analyse(speechlike(rate=8000), 8000, TEXT))["rate"] == "bad"
    assert verdicts(analyse(speechlike(rate=16000), 16000, TEXT))["rate"] == "warn"


def test_silence_is_not_a_voice() -> None:
    report = analyse(np.zeros(48000 * 5, np.float32), 48000, TEXT)
    assert report.verdict == "not suitable" and verdicts(report) == {"speech": "bad"}


def test_the_transcript_is_checked() -> None:
    clip = speechlike()
    assert verdicts(analyse(clip, 48000, ""))["transcript"] == "bad"
    sloppy = analyse(clip, 48000, "good morning i have 3 things" * 5)
    assert verdicts(sloppy)["transcript"] == "warn"
    advice = next(c.value for c in sloppy.checks if c.id == "transcript")
    assert "lower case" in advice and "digits" in advice and "full stop" in advice
    # far too few words for the speech: not the transcript of this clip
    assert verdicts(analyse(clip, 48000, "Hello there."))["pace"] == "warn"


def test_a_generated_candidate_that_wandered_is_found() -> None:
    # 27 words over ~13 s of speech is natural for a person, slow for a TTS reading its script
    clip, text = speechlike(), " ".join(["Good morning, this is a calm and even sentence."] * 3)
    assert verdicts(analyse(clip, 48000, text))["pace"] == "good"
    assert verdicts(analyse(clip, 48000, text, generated=True))["pace"] == "warn"


def test_a_set_needs_enough_speech_and_one_voice() -> None:
    short = analyse(speechlike(4.0), 48000, TEXT).public()
    assert assess([short]).verdict == "not suitable"
    one = analyse(speechlike(9.0), 48000, TEXT).public()
    assert verdicts(assess([one]))["total"] == "warn"  # usable, short of 12 s
    other = analyse(speechlike(9.0, f0=320.0), 48000, TEXT).public()
    both = assess([one, other])
    assert verdicts(both)["total"] == "good" and verdicts(both)["alike"] == "warn"
    quiet = analyse(speechlike(9.0, gain=0.04), 48000, TEXT).public()
    assert verdicts(assess([one, quiet]))["levels"] == "warn"
    assert assess([analyse(np.zeros(48000), 48000, "").public()]).verdict == "not suitable"
