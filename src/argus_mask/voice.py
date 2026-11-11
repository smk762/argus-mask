"""Is this recording a good reference for a cloned voice? Measured, with the reason and what to do.

A one-shot cloning TTS (Miso, XTTS) copies everything in its reference clip: the voice, but
also the room, the noise, the microphone's bandwidth, the level, the pace and the mood of the
delivery. And it reads the clip against its transcript, so a transcript that does not match
the audio teaches it to say words that are not there. `analyse` measures one clip and
`assess` a set of them, each as a list of checks with a verdict (good, warn, bad) and, for
anything short of good, what to change. See docs/voices.md for the recording guide.

Pure numpy, a function of the samples alone: the same clip gives the same report.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np

FRAME_S = 0.02
TARGET_TOTAL_S = (12.0, 20.0)  # speech across the set; Miso keeps the first 20 s
MIN_TOTAL_S = 6.0
WPS = (1.8, 3.8)  # words a second of speech a matching transcript gives
WPS_GENERATED = (2.7, 4.2)  # a TTS reading its script: slower usually means it repeated or added words
PENALTY = {"bad": 30, "warn": 8, "good": 0, "info": 0}


@dataclass
class Check:
    id: str
    label: str
    value: str
    verdict: str  # good, warn, bad, info
    advice: str = ""


@dataclass
class Report:
    metrics: dict
    checks: list[Check] = field(default_factory=list)

    @property
    def score(self) -> int:
        return max(0, 100 - sum(PENALTY[c.verdict] for c in self.checks))

    @property
    def verdict(self) -> str:
        verdicts = {c.verdict for c in self.checks}
        return "not suitable" if "bad" in verdicts else "usable" if "warn" in verdicts else "good"

    def public(self) -> dict:
        return {
            "score": self.score,
            "verdict": self.verdict,
            "metrics": self.metrics,
            "checks": [asdict(c) for c in self.checks],
        }


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-9)))


def _frames_db(mono: np.ndarray, rate: int) -> np.ndarray:
    n = max(1, int(rate * FRAME_S))
    count = len(mono) // n
    if count == 0:
        return np.zeros(0)
    rms = np.sqrt((mono[: count * n].reshape(count, n).astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-9))


def _speech_mask(db: np.ndarray) -> tuple[np.ndarray, float]:
    """Which frames are speech, and the noise floor: frames 12 dB over the quiet ones and within 40 dB of the loudest."""
    if not len(db):
        return np.zeros(0, bool), -120.0
    floor = float(np.percentile(db, 10))
    mask = (db > floor + 12) & (db > db.max() - 40)
    # close gaps shorter than 150 ms (stops, short breaths) and drop blips shorter than 60 ms
    gap = int(0.15 / FRAME_S)
    idx = np.flatnonzero(mask)
    for a, b in zip(idx[:-1], idx[1:], strict=False):
        if 1 < b - a <= gap:
            mask[a:b] = True
    runs = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    for a, b in zip(runs[::2], runs[1::2], strict=False):
        if b - a < int(0.06 / FRAME_S):
            mask[a:b] = False
    return mask, floor


def _pitch(mono: np.ndarray, rate: int, mask: np.ndarray) -> np.ndarray:
    """F0 estimates (Hz) of the voiced 40 ms windows inside speech, by autocorrelation, 70-500 Hz."""
    n = int(0.04 * rate)
    hop = int(FRAME_S * rate)
    lo, hi = rate // 500, rate // 70
    window = np.hanning(n)
    out = []
    for i in np.flatnonzero(mask)[::2]:  # every other speech frame is plenty
        start = i * hop
        w = mono[start : start + n]
        if len(w) < n:
            break
        w = (w - w.mean()) * window
        spectrum = np.fft.rfft(w, 2 * n)
        ac = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
        if ac[0] <= 0:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.45 * ac[0]:
            out.append(rate / k)
    return np.array(out)


def _bandwidth(mono: np.ndarray, rate: int, mask: np.ndarray) -> float:
    """The highest frequency the speech reaches: where its long-term spectrum stands 10 dB over the clip's own noise
    (the frames between words), or, with too little quiet to measure, within 50 dB of its peak."""
    hop = int(FRAME_S * rate)
    n = 2048

    def spectrum(frames: np.ndarray) -> np.ndarray | None:
        audio = np.concatenate([mono[i * hop : (i + 1) * hop] for i in frames]) if len(frames) else mono[:0]
        segs = len(audio) // n
        if segs == 0:
            return None
        power = (np.abs(np.fft.rfft(audio[: segs * n].reshape(segs, n) * np.hanning(n), axis=1)) ** 2).mean(axis=0)
        return np.convolve(10 * np.log10(power + 1e-20), np.ones(9) / 9, mode="same")

    speech = spectrum(np.flatnonzero(mask))
    if speech is None:
        return rate / 2
    noise = spectrum(np.flatnonzero(~mask))
    floor = speech.max() - 60 if noise is None else np.maximum(noise + 10, speech.max() - 60)
    above = np.flatnonzero(speech > floor)
    return float(above[-1] * rate / n) if len(above) else 0.0


def _words(transcript: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", transcript))


def analyse(samples: np.ndarray, rate: int, transcript: str = "", *, generated: bool = False) -> Report:
    """One clip: *samples* (frames x channels, or mono) at *rate*, and what is said in it.

    *generated*: the clip is a TTS reading *transcript* (a synthetic candidate), held to a tighter pace,
    since a model that wanders off its script (repeats a sentence, adds one) shows up as too few words a second.
    """
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    mono = (samples if samples.ndim == 1 else samples.mean(axis=1)).astype(np.float32)
    seconds = len(mono) / rate if rate else 0.0
    db = _frames_db(mono, rate)
    mask, floor = _speech_mask(db)
    speech_s = float(mask.sum() * FRAME_S)
    m: dict = {"rate": rate, "channels": channels, "seconds": round(seconds, 2), "speech_seconds": round(speech_s, 2)}
    report = Report(metrics=m)
    add = report.checks.append

    if speech_s < 0.5:
        add(
            Check(
                "speech",
                "Speech",
                "none found",
                "bad",
                "No speech found: is this the right file, and is it not silent?",
            )
        )
        return report

    idx = np.flatnonzero(mask)
    lead, tail = idx[0] * FRAME_S, seconds - (idx[-1] + 1) * FRAME_S
    runs = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    gaps = [(a - b) * FRAME_S for a, b in zip(runs[2::2], runs[1:-1:2], strict=False)]
    longest_pause = max(gaps, default=0.0)
    peak = float(np.abs(mono).max())
    clipped = int((np.abs(mono) >= 0.999).sum())
    flat = int(((np.abs(mono[1:]) > 0.95) & (np.diff(mono) == 0)).sum())
    speech_db = 10 * np.log10(np.mean(10 ** (db[mask] / 10)))
    quiet = db[~mask]
    noise_db = float(np.median(quiet)) if len(quiet) >= 10 else floor
    snr = speech_db - noise_db
    dc = float(mono.mean())
    bandwidth = _bandwidth(mono, rate, mask)
    f0 = _pitch(mono, rate, mask)
    words = _words(transcript)
    m |= {
        "lead_seconds": round(lead, 2),
        "tail_seconds": round(tail, 2),
        "longest_pause_seconds": round(longest_pause, 2),
        "peak_dbfs": round(_db(peak), 1),
        "clipped_samples": clipped + flat,
        "speech_dbfs": round(float(speech_db), 1),
        "noise_dbfs": round(noise_db, 1),
        "snr_db": round(float(snr), 1),
        "dc_offset": round(dc, 4),
        "bandwidth_hz": round(bandwidth),
        "pitch_hz": round(float(np.median(f0))) if len(f0) >= 5 else None,
        "pitch_spread_semitones": round(float(12 * np.log2(np.percentile(f0, 75) / np.percentile(f0, 25))), 1)
        if len(f0) >= 5
        else None,
        "words": words,
        "words_per_second": round(words / speech_s, 2) if words else None,
    }

    # length
    if speech_s < 2:
        add(
            Check(
                "length",
                "Speech length",
                f"{speech_s:.1f} s",
                "bad",
                "Under 2 s: too little to clone from. Aim for 12-20 s in all, in 1-3 clips.",
            )
        )  # noqa: E501
    elif speech_s < 6:
        add(
            Check(
                "length",
                "Speech length",
                f"{speech_s:.1f} s",
                "warn",
                "Short: the voice will wander. Add clips, or record a longer passage: 12-20 s in all.",
            )
        )  # noqa: E501
    elif speech_s > 20:
        add(
            Check(
                "length",
                "Speech length",
                f"{speech_s:.1f} s",
                "warn",
                "Over 20 s: Miso keeps the first 20 s. Cut it to the best 12-20 s, ending on a full sentence.",
            )
        )  # noqa: E501
    else:
        add(Check("length", "Speech length", f"{speech_s:.1f} s", "good"))

    # format
    if rate < 16000:
        add(
            Check("rate", "Sample rate", f"{rate} Hz", "bad", "Below 16 kHz (phone quality). Record at 44.1 or 48 kHz.")
        )
    elif rate < 22050:
        add(
            Check(
                "rate",
                "Sample rate",
                f"{rate} Hz",
                "warn",
                "16 kHz loses the top of the voice. Record at 44.1 or 48 kHz; 24 kHz is the least Miso uses.",
            )
        )  # noqa: E501
    else:
        add(Check("rate", "Sample rate", f"{rate} Hz{', stereo (mixed to mono)' if channels > 1 else ''}", "good"))

    # bandwidth: measured against the clip's own noise, so in a noisy clip it is the noise that cuts it short
    if bandwidth < 7000 and snr < 30:
        add(
            Check(
                "bandwidth",
                "Bandwidth",
                f"{bandwidth / 1000:.1f} kHz",
                "warn",
                f"Above {bandwidth / 1000:.1f} kHz the voice is buried in the background noise: the clone will sound dull as well as noisy. A quieter recording fixes both.",
            )
        )  # noqa: E501
    elif bandwidth < 4000:
        add(
            Check(
                "bandwidth",
                "Bandwidth",
                f"{bandwidth / 1000:.1f} kHz",
                "bad",
                "Sounds like a phone line or a heavily compressed file. Use the original recording, uncompressed (WAV/FLAC).",
            )
        )  # noqa: E501
    elif bandwidth < 7000:
        add(
            Check(
                "bandwidth",
                "Bandwidth",
                f"{bandwidth / 1000:.1f} kHz",
                "warn",
                "The top end is missing: a low-bitrate MP3, a video soundtrack, or strong noise reduction. The clone will sound muffled.",
            )
        )  # noqa: E501
    else:
        add(Check("bandwidth", "Bandwidth", f"{bandwidth / 1000:.1f} kHz", "good"))

    # level and clipping
    if clipped + flat > max(3, len(mono) * 1e-4):
        add(
            Check(
                "clipping",
                "Clipping",
                f"{clipped + flat} samples at full scale",
                "bad",
                "The recording overloaded: the clone will crackle. Record again with the gain lower, peaks around -6 dBFS.",
            )
        )  # noqa: E501
    elif clipped + flat:
        add(
            Check(
                "clipping",
                "Clipping",
                f"{clipped + flat} samples at full scale",
                "warn",
                "A few samples touch full scale. Fine if they are rare; lower the gain next time.",
            )
        )  # noqa: E501
    else:
        add(Check("clipping", "Clipping", "none", "good"))
    if peak < 10 ** (-24 / 20):
        add(
            Check(
                "level",
                "Level",
                f"peak {_db(peak):.0f} dBFS",
                "warn",
                "Very quiet: bringing it up raises the noise too. Record closer, or with more gain (peaks -6 to -3 dBFS).",
            )
        )  # noqa: E501
    else:
        add(Check("level", "Level", f"peak {_db(peak):.0f} dBFS, speech {speech_db:.0f} dBFS", "good"))

    # noise
    if snr < 20:
        add(
            Check(
                "noise",
                "Background noise",
                f"{snr:.0f} dB below the speech",
                "bad",
                "Noisy: the clone copies the hiss, hum or room tone. Record somewhere quieter, away from fans and traffic, closer to the mic.",
            )
        )  # noqa: E501
    elif snr < 30:
        add(
            Check(
                "noise",
                "Background noise",
                f"{snr:.0f} dB below the speech",
                "warn",
                "Some noise between words. A quieter room or a closer mic helps; avoid noise reduction, which smears the voice.",
            )
        )  # noqa: E501
    else:
        add(Check("noise", "Background noise", f"{snr:.0f} dB below the speech", "good"))
    if abs(dc) > 0.01:
        add(
            Check(
                "dc",
                "DC offset",
                f"{dc:+.3f}",
                "warn",
                "The waveform sits off centre (a cheap interface or mic). Most editors have a 'remove DC offset' step.",
            )
        )  # noqa: E501

    # edges and pauses
    if lead < 0.05 or tail < 0.05:
        add(
            Check(
                "edges",
                "Start and end",
                f"{lead:.2f} s before, {tail:.2f} s after",
                "warn",
                "Speech touches the edge of the file: was a word cut? Leave about 0.2 s of quiet at each end.",
            )
        )  # noqa: E501
    elif lead > 1.5 or tail > 1.5:
        add(
            Check(
                "edges",
                "Start and end",
                f"{lead:.1f} s before, {tail:.1f} s after",
                "warn",
                "Long silence at an end. It is trimmed, but trim it to about 0.2 s yourself so you hear what is kept.",
            )
        )  # noqa: E501
    else:
        add(Check("edges", "Start and end", f"{lead:.2f} s before, {tail:.2f} s after", "good"))
    if longest_pause > 1.5:
        add(
            Check(
                "pauses",
                "Pauses",
                f"longest {longest_pause:.1f} s",
                "warn",
                "A long pause inside the clip. The clone learns to stall: cut it to under a second. (A generated candidate that pauses long and then speaks has often wandered off the script.)",
            )
        )  # noqa: E501
    else:
        add(Check("pauses", "Pauses", f"longest {longest_pause:.1f} s", "good"))

    # delivery
    spread = m["pitch_spread_semitones"]
    if spread is None:
        add(Check("delivery", "Delivery", "pitch not measurable", "info"))
    elif spread < 1.2:
        add(
            Check(
                "delivery",
                "Delivery",
                f"{m['pitch_hz']} Hz, ±{spread} st",
                "warn",
                "Very flat: the clone will be monotone. Read as if speaking to someone, not reading a list.",
            )
        )  # noqa: E501
    elif spread > 7:
        add(
            Check(
                "delivery",
                "Delivery",
                f"{m['pitch_hz']} Hz, ±{spread} st",
                "warn",
                "Very animated: shouting, laughing or acting. The clone copies the mood into every line; use a calmer take for an everyday voice.",
            )
        )  # noqa: E501
    else:
        add(Check("delivery", "Delivery", f"{m['pitch_hz']} Hz, ±{spread} st", "good"))

    # transcript
    text = transcript.strip()
    if not text:
        add(
            Check(
                "transcript",
                "Transcript",
                "missing",
                "bad",
                "Type exactly what is said. Without it, or with a wrong one, the clone invents words.",
            )
        )  # noqa: E501
    else:
        problems = []
        if text[:1].islower():
            problems.append("starts in lower case")
        if text[-1:] not in ".!?\"')":
            problems.append("no full stop at the end")
        if re.search(r"\d", text):
            problems.append("has digits: write numbers as they are said")
        wps = words / speech_s
        band = WPS_GENERATED if generated else WPS
        if generated and wps < band[0]:
            add(
                Check(
                    "pace",
                    "Script vs audio",
                    f"{wps:.1f} words/s",
                    "warn",
                    "More speech than the script needs: the model probably repeated or added words. Listen to the end, or generate another.",
                )
            )  # noqa: E501
        elif not band[0] <= wps <= band[1]:
            add(
                Check(
                    "pace",
                    "Transcript vs audio",
                    f"{wps:.1f} words/s",
                    "warn",
                    f"{'Too few' if wps < band[0] else 'Too many'} words for the speech: does the transcript match the clip word for word? (Natural speech is about 2-3.5 words a second.)",
                )
            )  # noqa: E501
        else:
            add(Check("pace", "Transcript vs audio", f"{wps:.1f} words/s", "good"))
        if problems:
            add(
                Check(
                    "transcript",
                    "Transcript",
                    "; ".join(problems),
                    "warn",
                    "Write it as a sentence, with capitals and punctuation, word for word with the audio (ums and repeats included).",
                )
            )  # noqa: E501
        else:
            add(Check("transcript", "Transcript", f"{words} words", "good"))
    return report


def assess(clips: list[dict]) -> Report:
    """A set of clips (each a `Report.public()`): enough speech in all, and alike enough to be one voice."""
    usable = [c for c in clips if c.get("verdict") != "not suitable"]
    total = sum(float(c["metrics"].get("speech_seconds") or 0) for c in usable)
    m = {"clips": len(clips), "usable": len(usable), "speech_seconds": round(total, 1)}
    report = Report(metrics=m)
    add = report.checks.append
    if not usable:
        add(Check("set", "Usable clips", "none", "bad", "Fix the clips marked not suitable first."))
        return report
    lo, hi = TARGET_TOTAL_S
    if total < MIN_TOTAL_S:
        add(
            Check(
                "total", "Speech in all", f"{total:.1f} s", "bad", f"Under {MIN_TOTAL_S:g} s. Aim for {lo:g}-{hi:g} s."
            )
        )
    elif total < lo:
        add(
            Check(
                "total",
                "Speech in all",
                f"{total:.1f} s",
                "warn",
                f"Usable, but {lo:g}-{hi:g} s gives a steadier voice.",
            )
        )
    elif total > hi:
        add(
            Check(
                "total",
                "Speech in all",
                f"{total:.1f} s",
                "warn",
                f"Miso keeps the first {hi:g} s: put the best clip first.",
            )
        )  # noqa: E501
    else:
        add(Check("total", "Speech in all", f"{total:.1f} s", "good"))
    if len(usable) > 1:
        levels = [c["metrics"]["speech_dbfs"] for c in usable if c["metrics"].get("speech_dbfs") is not None]
        if levels and max(levels) - min(levels) > 6:
            add(
                Check(
                    "levels",
                    "Levels match",
                    f"{max(levels) - min(levels):.0f} dB apart",
                    "warn",
                    "The clips are at very different levels: recorded at different distances or settings? They are evened out, but the room and mic sound will differ too.",
                )
            )  # noqa: E501
        pitches = [c["metrics"]["pitch_hz"] for c in usable if c["metrics"].get("pitch_hz")]
        if len(pitches) > 1 and 12 * np.log2(max(pitches) / min(pitches)) > 3:
            add(
                Check(
                    "alike",
                    "One voice",
                    f"{min(pitches)}-{max(pitches)} Hz",
                    "warn",
                    "The clips sit at quite different pitches: different deliveries, or different speakers? Keep one delivery per voice; make a variant for another (see the guide).",
                )
            )  # noqa: E501
        rates = {c["metrics"]["rate"] for c in usable}
        if len(rates) > 1:
            add(Check("rates", "Formats", ", ".join(f"{r} Hz" for r in sorted(rates)), "info"))
    return report
