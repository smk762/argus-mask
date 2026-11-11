"""Tidying speech: numbers as words, short sentences joined, soft edges, even loudness."""

from __future__ import annotations

import numpy as np
import pytest

from argus_mask.models import AudioFormat
from argus_mask.speech import (
    FADE_IN_MS,
    FADE_OUT_MS,
    PEAK,
    TARGET_DBFS,
    Edges,
    join_short,
    number_words,
    ordinal_words,
    speakable,
    speech_level,
    tidy_audio,
)

F32 = AudioFormat(sample_rate=24000, format="f32le", channels=1)
S16 = AudioFormat(sample_rate=16000, format="s16le", channels=2)


@pytest.mark.parametrize(
    ("n", "words"),
    [
        (0, "zero"),
        (13, "thirteen"),
        (21, "twenty-one"),
        (100, "one hundred"),
        (115, "one hundred and fifteen"),
        (1005, "one thousand and five"),
        (1234, "one thousand two hundred and thirty-four"),
        (2_000_000, "two million"),
        (-7, "minus seven"),
    ],
)
def test_number_words(n, words) -> None:
    assert number_words(n) == words


def test_ordinal_words() -> None:
    assert [ordinal_words(n) for n in (1, 2, 3, 5, 12, 20, 21, 42, 100)] == [
        "first", "second", "third", "fifth", "twelfth", "twentieth", "twenty-first", "forty-second", "one hundredth",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("text", "said"),
    [
        # what the recordings got wrong
        ("The time is currently 9:30am. The night agent", "The time is currently nine thirty a m. The night agent"),
        (
            "The temperature is 0.9 on seed 623.",
            "The temperature is zero point nine on seed six hundred and twenty-three.",
        ),
        ("the meeting at 1400.", "the meeting at fourteen hundred."),
        ("seven pull requests and 82 issues", "seven pull requests and eighty-two issues"),
        # times
        ("at 2 p.m. today", "at two p m today"),
        ("at 14:05", "at two oh five p m"),
        ("at 9:00", "at nine o'clock"),
        ("Ready at 10 PM.", "Ready at ten p m."),
        ("at 9 a.m. Then", "at nine a m. Then"),
        # quantities
        ("1,234 stars", "one thousand two hundred and thirty-four stars"),
        ("45% done, 2.5% left", "forty-five percent done, two point five percent left"),
        ("21°C and 70 °F", "twenty-one degrees Celsius and seventy degrees Fahrenheit"),
        ("$5.50 and £1", "five dollars and fifty cents and one pound"),
        ("-0.5 and -7", "minus zero point five and minus seven"),
        ("pages 5-10", "pages five to ten"),
        ("the 21st and the 3rd", "the twenty-first and the third"),
        ("issue #3", "issue number three"),
        ("Q & A", "Q and A"),
        # years, codes, dates, addresses
        ("in 1999, 2005 and 2026", "in nineteen ninety-nine, two thousand and five and twenty twenty-six"),
        ("agent 007", "agent zero zero seven"),
        ("on 2026-11-11", "on the eleventh of November twenty twenty-six"),
        ("host 192.168.1.39", "host one nine two dot one six eight dot one dot thirty-nine"),
        # names keep their digits
        ("A2F renders 3D to mp4, v1.2.3", "A2F renders 3D to mp4, v1.2.3"),
        ("No numbers here.", "No numbers here."),
    ],
)
def test_speakable(text, said) -> None:
    assert speakable(text) == said


def test_short_sentences_go_with_a_neighbour_within_a_paragraph() -> None:
    assert join_short([("Greetings.", 1), ("I am Ginger, your assistant.", 1), ("Shall we?", 0)]) == [
        ("Greetings. I am Ginger, your assistant. Shall we?", 0)
    ]
    assert join_short([("Hi.", 1), ("Yes.", 1), ("Here we are again.", 0)]) == [("Hi. Yes. Here we are again.", 0)]
    # never across a paragraph break
    assert join_short([("Hello there.", 2), ("This one is long enough.", 0)]) == [
        ("Hello there.", 2),
        ("This one is long enough.", 0),
    ]
    long = [("This sentence is long enough.", 1), ("So is this one here.", 0)]
    assert join_short(long) == long


def _tone(seconds: float, rate: int, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_tidy_audio_fades_the_edges_and_levels_the_speech() -> None:
    quiet = _tone(1.0, 24000, 0.02)
    out = np.frombuffer(tidy_audio(quiet.tobytes(), F32), dtype="<f4")
    assert len(out) == len(quiet)
    assert abs(out[0]) < 1e-3 and abs(out[-1]) < 1e-3  # no step at either end
    # brought up, but by no more than MAX_GAIN_DB: a quiet sentence stays a little quiet
    assert 20 * np.log10(speech_level(out, 24000)) == pytest.approx(20 * np.log10(0.02 / np.sqrt(2)) + 9, abs=0.2)
    middle = _tone(1.0, 24000, 0.2)
    out = np.frombuffer(tidy_audio(middle.tobytes(), F32), dtype="<f4")
    assert 20 * np.log10(speech_level(out, 24000)) == pytest.approx(TARGET_DBFS, abs=0.2)
    # a loud, peaky one is never pushed past PEAK
    spiky = _tone(1.0, 24000, 0.05)
    spiky[12000] = 0.95
    out = np.frombuffer(tidy_audio(spiky.tobytes(), F32), dtype="<f4")
    assert np.abs(out).max() <= PEAK + 1e-6


def test_tidy_audio_keeps_the_format() -> None:
    stereo = (np.stack([_tone(0.5, 16000, 0.3)] * 2, axis=1) * 32767).astype("<i2")
    data = stereo.tobytes() + b"\x01"  # a stray byte stays where it was
    out = tidy_audio(data, S16)
    assert len(out) == len(data) and out[-1:] == b"\x01"
    samples = np.frombuffer(out[:-1], dtype="<i2").reshape(-1, 2)
    assert (samples[:, 0] == samples[:, 1]).all() and samples[0, 0] == 0
    assert tidy_audio(b"", F32) == b""
    silent = bytes(4 * 2400)
    assert tidy_audio(silent, F32) == silent


@pytest.mark.parametrize("piece", [1, 7, 333, 4096, 10**6])
def test_edges_in_pieces_match_edges_on_the_whole(piece) -> None:
    sound = _tone(0.4, 24000, 0.3).tobytes()
    edges = Edges(F32)
    out = b"".join(edges.feed(sound[i : i + piece]) for i in range(0, len(sound), piece)) + edges.end()
    assert out == tidy_audio(sound, F32, level=False)
    head = np.frombuffer(out, dtype="<f4")
    assert abs(head[0]) < 1e-3 and abs(head[-1]) < 1e-3
    assert FADE_IN_MS < FADE_OUT_MS


def test_edges_on_a_sound_shorter_than_its_fade() -> None:
    blip = _tone(0.01, 24000, 0.3).tobytes()
    edges = Edges(F32)
    assert edges.feed(blip) == b""
    out = np.frombuffer(edges.end(), dtype="<f4")
    assert len(out) == 240 and abs(out[-1]) < 1e-2
