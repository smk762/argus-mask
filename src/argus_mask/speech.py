"""Tidying speech for a TTS service: the text before it goes, the audio after it comes back.

What ``say`` does by default, and ``--as-is`` turns off. Each step answers something heard
in recordings from a one-shot cloning TTS (Miso):

- **Numbers as words** (`speakable`). A model left to read ``9:30``, ``0.9`` or ``1400``
  guesses, and guesses differently each time ("maybe month", "ornate on seed or so").
  Times, decimals, integers, ordinals, percentages, degrees and currency are written
  out the way a person would say them.
- **No very short sentences** (`join_short`). A sentence goes to the service alone, and
  a one-word one ("Greetings.") has the least to go on: its pitch and loudness land
  furthest from the rest. It is sent with the sentence after it.
- **Soft edges** (`Edges`, `tidy_audio`). The model stops on its own end-of-speech
  token, often while the last sound is still at speaking level, and starts on a step;
  with silence either side that is a click, or a word that loses its tail. A short
  fade at both ends.
- **Even loudness** (`tidy_audio`). Each sentence is generated alone, and comes back
  anything up to 9 dB louder or quieter than its neighbours. Each is brought to the same
  speech level, within limits, and never above a safe peak.

All of it is a pure function of its input: no clock, no randomness.
"""

from __future__ import annotations

import re

import numpy as np

from argus_mask.audio import _SAMPLE_BYTES
from argus_mask.models import AudioFormat

# ── Numbers as words ──────────────────────────────────────────────────────────

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
    "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]  # fmt: skip
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]
_ORDINAL = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}


def number_words(n: int) -> str:
    """*n* in words: 21 is "twenty-one", 1205 "one thousand two hundred and five"."""
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{_ONES[hundreds]} hundred" + (f" and {number_words(rest)}" if rest else "")
    if n >= 10**15:
        return " ".join(_ONES[int(d)] for d in str(n))
    for size, name in _SCALES:
        if n >= size:
            high, rest = divmod(n, size)
            tail = "" if not rest else (f" and {number_words(rest)}" if rest < 100 else f" {number_words(rest)}")
            return f"{number_words(high)} {name}{tail}"
    raise AssertionError(n)  # pragma: no cover


def ordinal_words(n: int) -> str:
    """21 is "twenty-first", 12 "twelfth"."""
    words = number_words(n)
    head, sep, last = words.rpartition("-" if "-" in words.split(" ")[-1] else " ")
    if last in _ORDINAL:
        last = _ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def year_words(n: int) -> str:
    """A four-digit number read as a year or a clock time: 1999 "nineteen ninety-nine", 1400 "fourteen hundred"."""
    high, low = divmod(n, 100)
    if 2000 <= n <= 2009:
        return number_words(n)
    if low == 0:
        return f"{number_words(high)} hundred"
    return f"{number_words(high)} {'oh ' + _ONES[low] if low < 10 else number_words(low)}"


def _digits(s: str) -> str:
    return " ".join(_ONES[int(d)] for d in s)


def _int(s: str) -> int:
    return int(s.replace(",", ""))


def _sentence_dot(match: re.Match) -> str:
    """The full stop of an "a.m." that also ends its sentence: kept, so the sentence still ends."""
    if not match["dot"]:
        return ""
    rest = match.string[match.end() :]
    return "." if not rest.strip() or rest[:1] == "\n" or rest.lstrip()[:1].isupper() else ""


def _clock(hour: int, minute: int, suffix: str | None) -> str:
    if suffix is None and hour > 12:  # a 24-hour time: said as a 12-hour one
        hour, suffix = hour - 12, "pm"
    elif suffix is None and hour == 0:
        hour, suffix = 12, "am"
    words = number_words(hour)
    if minute:
        words += " " + ("oh " + _ONES[minute] if minute < 10 else number_words(minute))
    elif suffix is None:
        words += " o'clock"
    if suffix:
        words += " a m" if suffix.startswith("a") else " p m"
    return words


_AMPM = r"(?:\s*(?P<ampm>[aApP])\.?\s?[mM](?P<dot>\.)?(?![A-Za-z]))"
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}  # fmt: skip


def _money(match: re.Match) -> str:
    one, many, small_one, small_many = _CURRENCY[match["sym"]]
    if match["cents"] and len(match["cents"]) > 2:  # $1.234: not a sum in cents
        return f"{_decimal_text(match['whole'] + '.' + match['cents'])} {many}"
    whole = _int(match["whole"])
    words = f"{number_words(whole)} {one if whole == 1 else many}"
    if match["cents"]:
        cents = int(match["cents"].ljust(2, "0"))
        if cents:
            words += f" and {number_words(cents)} {small_one if cents == 1 else small_many}"
    return words


def _decimal_text(s: str) -> str:
    sign, s = ("minus ", s[1:]) if s.startswith("-") else ("", s)
    whole, _, frac = s.partition(".")
    return sign + (f"{number_words(_int(whole))} point {_digits(frac)}" if frac else number_words(_int(whole)))


def _decimal(match: re.Match) -> str:
    return _decimal_text(match[0])


def _degrees(match: re.Match) -> str:
    value = match["value"]
    words = _decimal_text(value)
    unit = {"c": " Celsius", "f": " Fahrenheit"}.get((match["unit"] or "").lower(), "")
    return f"{words} degree{'' if value in ('1', '-1') else 's'}{unit}"


def _plain(match: re.Match) -> str:
    return _plain_text(match[0])


_MONTHS = "January February March April May June July August September October November December".split()  # noqa: SIM905, E501


def _date(match: re.Match) -> str:
    year, month, day = int(match["y"]), int(match["m"]), int(match["d"])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match[0]
    return f"the {ordinal_words(day)} of {_MONTHS[month - 1]} {year_words(year) if 1100 <= year <= 2099 else number_words(year)}"


def _plain_text(text: str) -> str:
    if "," not in text and len(text) == 4 and 1100 <= int(text) <= 2099:
        return year_words(int(text))
    if text.startswith("0") and len(text) > 1:  # a code, not a quantity: 007 is "zero zero seven"
        return _digits(text)
    return number_words(_int(text))


# Order matters: the most specific shapes first, so 9:30 is a time before 9 is a number.
_RULES: list[tuple[re.Pattern, object]] = [
    (re.compile(r"(?<!\w)#(?=\d)"), lambda m: "number "),
    # 192.168.1.39, 1.2.3: dotted groups, each group a number
    (
        re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?!\w|\.\d)"),
        lambda m: " dot ".join(_digits(g) if len(g) > 2 else number_words(int(g)) for g in m[0].split(".")),
    ),
    (
        re.compile(r"(?<![\w:])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)" + _AMPM + r"?(?![\w:])"),
        lambda m: _clock(int(m["h"]), int(m["m"]), (m["ampm"] or "").lower() or None) + _sentence_dot(m),
    ),
    (
        re.compile(r"(?<![\w.:])(?P<h>1[0-2]|0?[1-9])" + _AMPM),
        lambda m: _clock(int(m["h"]), 0, m["ampm"].lower()) + _sentence_dot(m),
    ),
    (re.compile(r"(?<![\w.-])(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?![\w-])"), _date),
    (re.compile(r"(?P<sym>[$£€])(?P<whole>\d[\d,]*)(?:\.(?P<cents>\d+))?(?!\d)"), _money),
    (re.compile(r"(?<![\w.])(?P<value>-?\d[\d,]*(?:\.\d+)?)\s?°\s?(?P<unit>[CcFf](?![A-Za-z]))?"), _degrees),
    (
        re.compile(r"(?<![\w.])(?P<value>-?\d[\d,]*(?:\.\d+)?)\s?%"),
        lambda m: _decimal_text(m["value"]) + " percent",
    ),
    (re.compile(r"(?<![\w.])(?P<n>\d+)(?:st|nd|rd|th)\b"), lambda m: ordinal_words(int(m["n"]))),
    (
        re.compile(r"(?<![\w.-])(?P<a>\d+)\s?[-–]\s?(?P<b>\d+)(?![\w.]|[-–]\d)"),
        lambda m: f"{_plain_text(m['a'])} to {_plain_text(m['b'])}",
    ),
    (re.compile(r"(?<![\w.])(?P<whole>-?\d[\d,]*)\.(?P<frac>\d+)(?![\w]|\.\d)"), _decimal),
    (re.compile(r"(?<![\w.-])-(?=\d)"), lambda m: "minus "),
    (re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\d\w])|(?<![\w.])\d+(?![\w])"), _plain),
    (re.compile(r"\s&\s"), lambda m: " and "),
]


def speakable(text: str) -> str:
    """*text* with its numbers, times and symbols written as words, the way a person reads them aloud.

    Digits inside words (``A2F``, ``mp4``, ``3D``) are left alone: they are names, and a model
    reads them as well as a rule would.
    """
    for pattern, replace in _RULES:
        text = pattern.sub(replace, text)  # type: ignore[arg-type]
    return text


# ── Sentences ─────────────────────────────────────────────────────────────────

MIN_WORDS = 4  # a sentence with fewer words goes with the next one


def join_short(parts: list[tuple[str, int]], min_words: int = MIN_WORDS) -> list[tuple[str, int]]:
    """*parts* (sentence, pauses after it) with each sentence under *min_words* words sent with its neighbour.

    A short sentence joins the one after it, never across a paragraph break (more than one pause);
    the last of a paragraph joins the one before it instead.
    """
    out: list[tuple[str, int]] = []
    carry = ""
    for text, pauses in parts:
        text = f"{carry} {text}" if carry else text
        carry = ""
        if len(text.split()) < min_words and pauses == 1:
            carry = text
            continue
        if len(text.split()) < min_words and out and out[-1][1] == 1:
            prev, _ = out.pop()
            text = f"{prev} {text}"
        out.append((text, pauses))
    if carry:  # pragma: no cover - a carry only happens with pauses == 1, so a later part always follows
        out.append((carry, 0))
    return out


# ── Audio ─────────────────────────────────────────────────────────────────────

FADE_IN_MS = 5
FADE_OUT_MS = 30
TARGET_DBFS = -20.0  # speech level (the RMS of its voiced frames) each sentence is brought to
MAX_GAIN_DB = 9.0  # in either direction: a whisper stays quieter than a shout
PEAK = 0.89  # about -1 dBFS


def _frame_bytes(fmt: AudioFormat) -> int:
    return _SAMPLE_BYTES[fmt.format] * fmt.channels


def _decode(data: bytes, fmt: AudioFormat) -> np.ndarray:
    """A new, writable (samples, channels) array; a NaN or infinite sample becomes 0 or +-1, as `PcmDecoder` does,
    so that one bad sample cannot turn the level, and with it the whole sentence, into NaN or silence."""
    if fmt.format == "f32le":
        samples = np.frombuffer(data, dtype="<f4").astype(np.float32)
        samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0, copy=False)
    else:
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    return samples.reshape(-1, fmt.channels)


def _encode(samples: np.ndarray, fmt: AudioFormat) -> bytes:
    if fmt.format == "f32le":
        return samples.astype("<f4").tobytes()
    return (np.clip(samples, -1.0, 32767 / 32768) * 32768.0).round().astype("<i2").tobytes()


def _ramp(n: int) -> np.ndarray:
    """A raised-cosine ramp from 0 to 1 over *n* samples."""
    return (0.5 - 0.5 * np.cos(np.pi * (np.arange(n, dtype=np.float32) + 0.5) / n)).astype(np.float32)


def speech_level(samples: np.ndarray, rate: int) -> float | None:
    """The RMS of the voiced 20 ms frames of *samples* (frames within 30 dB of the loudest), or None for silence."""
    mono = samples.mean(axis=1) if samples.ndim == 2 else samples
    n = max(1, rate // 50)
    frames = len(mono) // n
    if frames == 0:
        return None
    rms = np.sqrt((mono[: frames * n].reshape(frames, n).astype(np.float64) ** 2).mean(axis=1))
    loud = rms.max()
    if loud < 1e-4:
        return None
    voiced = rms[rms >= loud * 10 ** (-30 / 20)]
    return float(np.sqrt((voiced**2).mean()))


def tidy_audio(data: bytes, fmt: AudioFormat, *, level: bool = True) -> bytes:
    """One sentence's PCM with soft edges and, if *level*, brought to `TARGET_DBFS`, in the same format."""
    usable = len(data) - len(data) % _frame_bytes(fmt)
    samples = _decode(data[:usable], fmt)
    if not len(samples):
        return data
    if level and (current := speech_level(samples, fmt.sample_rate)) is not None:
        gain_db = float(np.clip(TARGET_DBFS - 20 * np.log10(current), -MAX_GAIN_DB, MAX_GAIN_DB))
        gain = 10 ** (gain_db / 20)
        peak = float(np.abs(samples).max())
        if peak * gain > PEAK:
            gain = PEAK / peak
        samples *= gain
    fade_in = min(len(samples) // 2, fmt.sample_rate * FADE_IN_MS // 1000)
    fade_out = min(len(samples) // 2, fmt.sample_rate * FADE_OUT_MS // 1000)
    if fade_in:
        samples[:fade_in] *= _ramp(fade_in)[:, None]
    if fade_out:
        samples[-fade_out:] *= _ramp(fade_out)[::-1, None]
    return _encode(samples, fmt) + data[usable:]


class Edges:
    """Soft edges for one sentence that arrives in pieces: for ``say --stream``, where it cannot be levelled.

    `feed` returns what may be sent now, holding back the last `FADE_OUT_MS` (the end is not
    known until it comes) and any part of a sample split across pieces; `end` returns the rest, faded out.
    """

    def __init__(self, fmt: AudioFormat) -> None:
        self.fmt = fmt
        self.frame = _frame_bytes(fmt)
        self.hold = fmt.sample_rate * FADE_OUT_MS // 1000 * self.frame
        self.fade_in = fmt.sample_rate * FADE_IN_MS // 1000
        self.done_in = 0  # samples of the fade-in applied so far
        self.pending = b""

    def _in(self, data: bytes) -> bytes:
        if self.done_in >= self.fade_in or not data:
            return data
        samples = _decode(data, self.fmt)
        ramp = _ramp(self.fade_in)[self.done_in : self.done_in + len(samples)]
        samples[: len(ramp)] *= ramp[:, None]
        self.done_in += len(ramp)
        return _encode(samples, self.fmt)

    def feed(self, data: bytes) -> bytes:
        self.pending += data
        ready = len(self.pending) - self.hold
        ready -= ready % self.frame
        if ready <= 0:
            return b""
        out, self.pending = self.pending[:ready], self.pending[ready:]
        return self._in(out)

    def end(self) -> bytes:
        usable = len(self.pending) - len(self.pending) % self.frame
        if not usable:
            return b""
        samples = _decode(self._in(self.pending[:usable]), self.fmt)
        self.pending = b""
        samples *= _ramp(len(samples))[::-1, None]
        return _encode(samples, self.fmt)
