"""Speech cues (prosody.py) and the idle layer that reads them: nods, brow raises, blinks in pauses, saccades."""

from __future__ import annotations

import numpy as np

from argus_mask.arkit import INDEX
from argus_mask.audio import DRIVER_RATE
from argus_mask.drivers import build_driver
from argus_mask.idle import BLINK_DOUBLE_GAP_S, BLINK_HOLD_S, BLINK_MAX_GAP_S, FIXATION_S, IdleLayer
from argus_mask.performance import perform
from argus_mask.prosody import Cue, Prosody

FPS = 30


def speech(phrases: list[tuple[float, float]], rate: int = DRIVER_RATE, *, f0: float = 160.0) -> np.ndarray:
    """Phrases of syllables (each (start s, length s)) in silence, over the length of the last plus a second.

    Syllables are 4 a second, a harmonic voice whose every other syllable is louder and higher: stressed.
    """
    total = max(start + length for start, length in phrases) + 1.0
    out = np.zeros(int(total * rate), dtype=np.float32)
    for start, length in phrases:
        n = int(length * rate)
        t = np.arange(n) / rate
        syllable = np.floor(t * 4).astype(int)
        stressed = syllable % 2 == 0
        pitch = np.where(stressed, f0 * 1.5, f0)
        phase = 2 * np.pi * np.cumsum(pitch) / rate
        voice = sum(np.sin(k * phase) / k for k in (1, 2, 3))
        envelope = np.sin(np.pi * ((t * 4) % 1.0)) ** 2 * np.where(stressed, 0.35, 0.08)
        out[int(start * rate) : int(start * rate) + n] = (voice * envelope).astype(np.float32)
    return out


def cues(pcm: np.ndarray, chunk: int | None = None) -> list[Cue]:
    prosody = Prosody(FPS)
    frames = -(-pcm.size * FPS // DRIVER_RATE)
    out: list[Cue] = []
    step = chunk or pcm.size
    for start in range(0, pcm.size, step):
        prosody.push(pcm[start : start + step])
        ready = (start + step) * FPS // DRIVER_RATE  # frames whose audio has all arrived
        while len(out) < min(ready, frames):
            out.append(prosody.cue(len(out)))
    while len(out) < frames:
        out.append(prosody.cue(len(out)))
    return out


def at(cue_list: list[Cue], flag: str) -> list[float]:
    return [i / FPS for i, c in enumerate(cue_list) if getattr(c, flag)]


def test_silence_has_no_cues() -> None:
    quiet = cues(np.zeros(DRIVER_RATE * 2, dtype=np.float32))
    assert not any(c.sounding or c.speaking or c.stress or c.phrase_end or c.phrase_start for c in quiet)
    assert all(c.pitch == c.roll == c.brow == 0.0 for c in quiet)


def test_phrases_start_and_end_where_the_voice_does() -> None:
    found = cues(speech([(0.5, 2.0), (3.5, 1.5)]))
    starts, ends = at(found, "phrase_start"), at(found, "phrase_end")
    assert len(starts) == 2 and abs(starts[0] - 0.5) < 0.1 and abs(starts[1] - 3.5) < 0.1
    # An end is noticed a little into the pause (PAUSE_S): causal, it cannot know sooner.
    assert len(ends) == 2 and 2.5 < ends[0] < 2.9 and 5.0 < ends[1] < 5.4
    # Between syllables the voice dips, but the phrase goes on: "speaking" holds through it.
    assert all(c.speaking for c in found[int(0.6 * FPS) : int(2.4 * FPS)])


def test_stressed_syllables_nod_the_head_and_emphasis_raises_the_brows() -> None:
    found = cues(speech([(0.5, 3.0)]))
    stresses = at(found, "stress")
    # Stressed syllables come twice a second; each nod is a stressed one, not every syllable.
    assert 4 <= len(stresses) <= 7
    pitch = np.array([c.pitch for c in found])
    assert pitch.max() > 1.0  # a nod tips the head down
    peak = int(np.argmax(pitch))
    assert any(0 <= peak - s * FPS <= 0.3 * FPS for s in stresses)  # just after a stress
    brow = np.array([c.brow for c in found])
    assert 0.05 < brow.max() <= 0.25
    assert (brow > 0.05).mean() < 0.4  # emphasis now and then, not raised brows throughout


def test_a_phrase_end_settles_the_head_and_tilts_alternate() -> None:
    found = cues(speech([(0.3, 1.5), (2.8, 1.5), (5.3, 1.5)]))
    ends = [int(s * FPS) for s in at(found, "phrase_end")]
    assert len(ends) == 3
    rolls = [max((found[i].roll for i in range(e, min(e + FPS, len(found)))), key=abs) for e in ends]
    assert rolls[0] * rolls[1] < 0 and rolls[1] * rolls[2] < 0  # left, right, left
    assert all(0.5 < abs(r) < 2.0 for r in rolls)


def test_a_cue_depends_on_the_audio_only_not_on_how_it_was_split() -> None:
    pcm = speech([(0.4, 1.8), (2.9, 1.0)])
    whole = cues(pcm)
    assert cues(pcm, chunk=777) == whole
    assert cues(pcm, chunk=1600) == whole


def test_blinks_wait_for_pauses() -> None:
    """While a word sounds, no blink starts (unless one is long overdue); a phrase end often brings one."""
    phrases = [(0.3 + 2.6 * k, 2.0) for k in range(12)]  # 31 s of speech in 2 s phrases
    found = cues(speech(phrases))
    zeros = np.zeros(52, dtype=np.float32)
    layer = IdleLayer(5, FPS)
    blinks = [layer.apply(i, zeros, c)[0][INDEX["eyeBlinkLeft"]] for i, c in enumerate(found)]
    starts = [i for i in range(1, len(blinks)) if blinks[i] > 0.0 and blinks[i - 1] == 0.0]
    # The first frame of a blink is the one before its weight rises.
    started = [i - 1 for i in starts]
    assert len(started) >= 8
    assert not any(found[i].sounding for i in started)
    gaps = np.diff(started) / FPS
    assert gaps.min() >= BLINK_DOUBLE_GAP_S - 1 / FPS  # the closest pair is a double blink
    assert gaps.max() <= BLINK_MAX_GAP_S + BLINK_HOLD_S + 0.1
    ends = [int(s * FPS) for s in at(found, "phrase_end")]
    after_end = sum(any(0 <= s - e <= 0.15 * FPS for e in ends) for s in started)
    assert after_end >= len(ends) // 3


def test_gaze_moves_in_saccades_with_fixations_between() -> None:
    zeros = np.zeros(52, dtype=np.float32)
    layer = IdleLayer(2, FPS)
    look = INDEX["eyeLookOutLeft"], INDEX["eyeLookInLeft"], INDEX["eyeLookUpLeft"], INDEX["eyeLookDownLeft"]
    xs = []
    for i in range(20 * FPS):
        w, _ = layer.apply(i, zeros, Cue())
        xs.append([w[j] for j in look])
    speed = np.abs(np.diff(np.array(xs), axis=0)).max(axis=1)
    jumps = np.flatnonzero(speed > 0.004)
    # Jumps come in short bursts (the saccade) with fixations of FIXATION_S between them.
    bursts = [j for k, j in enumerate(jumps) if k == 0 or j - jumps[k - 1] > 2]
    holds = np.diff(bursts) / FPS
    assert len(bursts) > 20
    assert holds.min() >= FIXATION_S[0] - 2 / FPS and np.median(holds) <= FIXATION_S[1] + 0.1


async def test_a_spoken_clip_nods_on_its_stresses_and_still_gives_the_same_take() -> None:
    audio = speech([(0.3, 2.0), (2.8, 1.5)], 24000)
    takes = [await perform(audio, 24000, build_driver("energy"), fps=FPS, seed=9) for _ in range(2)]
    assert takes[0] == takes[1]
    _, frames, _ = takes[0]
    # Same seed, no voice: the head turn differs by what the speech added (nods, settles, the glances it prompted).
    _, quiet, _ = await perform(np.zeros_like(audio), 24000, build_driver("energy"), fps=FPS, seed=9)
    nod = [a.head.pitch - b.head.pitch for a, b in zip(frames, quiet, strict=True)]
    assert max(nod) > 1.0
