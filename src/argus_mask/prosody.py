"""Speech cues: what the voice is doing at each frame, for the head, the brows and the idle layer.

People nod on stressed syllables, raise their brows on emphasis, settle or tilt at the end of a
phrase, and blink and glance in the pauses. :class:`Prosody` reads the 16 kHz stream a performer
feeds its driver and gives each frame a :class:`Cue`: whether speech is sounding, the events that
began on that frame (a stress, a phrase end, a phrase start), and the head turn and brow raise
those events add.

Causal and deterministic: a frame's cue depends only on the audio up to the end of that frame
(``(n + 1) * 16000 // fps``, integer arithmetic) and on the frames before it, never on the clock
or on how the audio was split into chunks. No random numbers at all: which way a phrase-end tilt
leans alternates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from argus_mask.audio import DRIVER_RATE

# Analysis windows, in samples at 16 kHz.
LEVEL_WINDOW = 512  # 32 ms of loudness
PITCH_WINDOW = 640  # 40 ms: two periods of the lowest voice
PITCH_LAGS = (DRIVER_RATE // 400, DRIVER_RATE // 70)  # 400 Hz down to 70 Hz
VOICED_CORRELATION = 0.5  # normalised autocorrelation at the period: above this the sound is voiced
OCTAVE_TOLERANCE = 0.9  # the shortest lag that correlates this well, against the best, is the period

ACTIVE_DB = -42.0  # louder than this is speech sounding (dBFS)
HANG_S = 0.15  # speech stays "speaking" this long after it stops sounding (between words)
PAUSE_S = 0.22  # this long without sound after a phrase is a pause: the phrase has ended
MIN_PHRASE_S = 0.5  # a phrase shorter than this is a word, not a phrase: its end is no event

# Stress: a voiced syllable whose loudness rises this much in RISE_S, near or above the running
# speech level, at least REFRACTORY_S after the last one.
RISE_DB = 6.0
RISE_S = 0.12
STRESS_UNDER_DB = 2.0  # and within this much of the running level of the loud syllables
REFRACTORY_S = 0.32
LEVEL_S = 1.5  # time constant of the speech level falling
LEVEL_RISE_S = 0.08  # the level rises this fast: it follows the loud syllables, not the average
# A stress raises the brows when its pitch is ACCENT_FROM semitones above the floor of the speaker's
# voice, fully at ACCENT_FULL: emphasis, not every stressed syllable. The floor follows the voice
# down quickly and up slowly, so it sits near the bottom of the speaker's range.
ACCENT_FROM = 2.0
ACCENT_FULL = 6.0
FLOOR_FALL_S = 0.3
FLOOR_RISE_S = 3.0

# What the events do. Degrees for the head, weights for the brows; each event's effect rises to its
# peak after TAU seconds and falls away over about five times that (a "bump": (t/tau) e^(1 - t/tau)).
NOD_DEG = 2.2  # positive pitch tips the head down
NOD_TAU = 0.12
BROW = 0.2  # browInnerUp at full accent; the outer brows take 0.7 of it
BROW_MAX = 0.25  # raises that overlap add up to no more than this
BROW_GAP_S = 1.5  # emphasis, not every stress: at most one raise in this long
BROW_TAU = 0.16
SETTLE_DEG = (-0.9, 1.4)  # (pitch: a slight lift, roll: a tilt) at a phrase end
SETTLE_TAU = 0.45


@dataclass(frozen=True)
class Cue:
    """What the voice is doing at one frame."""

    sounding: bool = False  # speech is sounding now (above ACTIVE_DB)
    speaking: bool = False  # inside a phrase: sounding, or only between words
    stress: float = 0.0  # a stress began on this frame (0..1, how strong)
    phrase_end: bool = False  # a pause began on this frame, after a phrase
    phrase_start: bool = False  # speech began on this frame, after a pause or silence
    pitch: float = 0.0  # head, degrees, added to the idle layer's
    roll: float = 0.0
    brow: float = 0.0  # browInnerUp to add (outer brows take 0.7 of it)


@dataclass
class _Bump:
    start: int  # frame
    tau: float  # seconds
    pitch: float = 0.0
    roll: float = 0.0
    brow: float = 0.0


def _bump(t: float, tau: float) -> float:
    return (t / tau) * math.exp(1.0 - t / tau) if t > 0.0 else 0.0


def _db(window: np.ndarray) -> float:
    if not window.size:
        return -120.0
    rms = math.sqrt(float(np.mean(np.square(window, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


def _pitch(window: np.ndarray) -> float | None:
    """The fundamental in Hz, or ``None`` when the window is not voiced."""
    x = window.astype(np.float64) - float(np.mean(window))
    lo, hi = PITCH_LAGS
    if x.size <= hi * 2:
        return None
    energy = float(np.dot(x, x))
    if energy <= 1e-9:
        return None
    # Normalised autocorrelation at every lag at once: x[:-lag] . x[lag:] over the energies of both parts.
    n = x.size
    lags = np.arange(lo, hi + 1)
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    squares = np.concatenate(([0.0], np.cumsum(x * x)))
    r = ac[lags] / np.sqrt(squares[n - lags] * (squares[n] - squares[lags]) + 1e-12)
    best = float(r.max())
    if best < VOICED_CORRELATION:
        return None
    # Multiples of the period correlate nearly as well as the period itself: take the shortest lag
    # that peaks close to the best, or a voice is heard an octave (or a twelfth) too low.
    for i in range(r.size):
        peak = 0 < i < r.size - 1 and r[i - 1] <= r[i] >= r[i + 1]  # a peak, not a slope at the edge
        if peak and r[i] >= OCTAVE_TOLERANCE * best:
            return DRIVER_RATE / (lo + i)
    return DRIVER_RATE / (lo + int(np.argmax(r)))


@dataclass
class Prosody:
    """Feed it the 16 kHz stream (:meth:`push`), then ask for each frame's cue in order (:meth:`cue`)."""

    fps: int
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _start: int = 0  # the sample index of _buf[0]
    _level: float | None = None  # running speech level, dB
    _floor: float | None = None  # the floor of the speaker's pitch, semitones re 1 Hz
    _recent: list[float] = field(default_factory=list)  # loudness over the last RISE_S
    _since_sound: int = 10**9  # frames since speech last sounded
    _phrase: int = 0  # frames of the current phrase so far
    _last_stress: int = -(10**9)
    _last_brow: int = -(10**9)
    _bumps: list[_Bump] = field(default_factory=list)
    _tilt: float = 1.0  # which way the next phrase-end tilt leans
    _next: int = 0  # the next frame to cue

    def push(self, pcm: np.ndarray) -> None:
        if pcm.size:
            self._buf = np.concatenate([self._buf, pcm.astype(np.float32, copy=False)])

    def _window(self, end: int, size: int) -> np.ndarray:
        lo = max(end - size, self._start)
        hi = min(end, self._start + self._buf.size)
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        return self._buf[lo - self._start : hi - self._start]

    def cue(self, frame: int) -> Cue:
        """The cue for *frame*; frames are asked for once each, in order. Past the audio is silence."""
        if frame != self._next:
            raise ValueError(f"cue for frame {frame}, expected {self._next}")
        self._next += 1
        dt = 1.0 / self.fps
        end = (frame + 1) * DRIVER_RATE // self.fps
        level_db = _db(self._window(end, LEVEL_WINDOW))
        sounding = level_db > ACTIVE_DB
        f0 = _pitch(self._window(end, PITCH_WINDOW)) if sounding else None

        # Forget audio no window will reach again.
        keep_from = end - PITCH_WINDOW
        if keep_from > self._start + DRIVER_RATE:  # trim in steps of a second, not every frame
            cut = keep_from - self._start
            self._buf = self._buf[cut:]
            self._start = keep_from

        was_speaking = self._since_sound * dt <= HANG_S
        self._since_sound = 0 if sounding else self._since_sound + 1
        speaking = self._since_sound * dt <= HANG_S
        phrase_start = sounding and self._since_sound == 0 and not was_speaking and self._phrase == 0
        phrase_end = False
        if sounding:
            self._phrase += 1
        elif self._phrase and self._since_sound * dt >= PAUSE_S:
            phrase_end = self._phrase * dt >= MIN_PHRASE_S
            self._phrase = 0

        # Running speech level (of the loud syllables: quick up, slow down), only while sounding.
        k = 1.0 - math.exp(-dt / LEVEL_S)
        if sounding:
            if self._level is None:
                self._level = level_db
            rise_k = 1.0 - math.exp(-dt / LEVEL_RISE_S)
            self._level += (level_db - self._level) * (rise_k if level_db > self._level else k)
        semitones = 12.0 * math.log2(f0) if f0 else None
        accent = 0.0
        if semitones is not None:
            if self._floor is None:
                self._floor = semitones
            accent = min(1.0, max(0.0, (semitones - self._floor - ACCENT_FROM) / (ACCENT_FULL - ACCENT_FROM)))
            tau = FLOOR_FALL_S if semitones < self._floor else FLOOR_RISE_S
            self._floor += (semitones - self._floor) * (1.0 - math.exp(-dt / tau))

        self._recent.append(level_db)
        span = max(1, round(RISE_S * self.fps))
        del self._recent[: -span - 1]
        rise = level_db - min(self._recent)
        stress = 0.0
        if (
            f0 is not None
            and self._level is not None
            and rise >= RISE_DB
            and level_db >= self._level - STRESS_UNDER_DB
            and (frame - self._last_stress) * dt >= REFRACTORY_S
        ):
            stress = min(1.0, max(0.3, (level_db - self._level + STRESS_UNDER_DB * 2) / (STRESS_UNDER_DB * 3)))
            self._last_stress = frame
            self._bumps.append(_Bump(frame, NOD_TAU, pitch=NOD_DEG * stress * (0.7 + 0.3 * accent)))
            if accent > 0.0 and (frame - self._last_brow) * dt >= BROW_GAP_S:
                self._last_brow = frame
                self._bumps.append(_Bump(frame, BROW_TAU, brow=BROW * accent * stress))
        if phrase_end:
            self._bumps.append(_Bump(frame, SETTLE_TAU, pitch=SETTLE_DEG[0], roll=SETTLE_DEG[1] * self._tilt))
            self._tilt = -self._tilt

        pitch = roll = brow = 0.0
        live = []
        for bump in self._bumps:
            t = (frame - bump.start) * dt
            if t > bump.tau * 8.0:
                continue
            live.append(bump)
            b = _bump(t, bump.tau)
            pitch += bump.pitch * b
            roll += bump.roll * b
            brow += bump.brow * b
        self._bumps = live
        return Cue(
            sounding=sounding,
            speaking=speaking,
            stress=round(stress, 4),
            phrase_end=phrase_end,
            phrase_start=phrase_start,
            pitch=pitch,
            roll=roll,
            brow=min(brow, BROW_MAX),
        )
