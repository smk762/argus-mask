"""The idle layer: blinks, gaze, small movements of the face, and the head.

A talking face that never blinks looks dead, and one that blinks on a timer looks mechanical. This
layer is added on top of the driver's weights, and reads each frame's :class:`~argus_mask.prosody.Cue`
(what the voice is doing) to time itself the way people do:

* **Blinks** come in pauses: likely just after a phrase ends, at about a resting rate between
  words and in silence, sometimes doubled. One that falls due while a word sounds waits for the
  gap after it, unless it is well overdue. Never closer than BLINK_MIN_GAP_S, and a blink is due
  once BLINK_MAX_GAP_S have passed without one.
* **Gaze** moves in saccades: a fast jump, then a fixation of 0.2-0.6 s with a slight drift. Most
  fixations stay on the viewer; a phrase start or a pause sometimes takes the eyes away (up or to
  the side, as people look away to think), and the head turns a little with them.
* **Micro-motion**: slow, small changes of brow and mouth tension; the mouth's only while not
  speaking, so they never fight the driver's shapes.
* **The head**: a slow sway, the cue's nods and phrase-end settles, and a share of each glance.
  The cue's brow raises are added to the driver's own brows.

Everything is a function of the seed, the frame number and the audio (through the cue), so a take
replays identically and two servers given the same seed and audio agree.
"""

from __future__ import annotations

import math
import random

import numpy as np

from argus_mask.arkit import INDEX
from argus_mask.models import Head
from argus_mask.prosody import Cue

BLINK_CLOSE_S = 0.07
BLINK_OPEN_S = 0.12
BLINK_MIN_GAP_S = 1.0  # from the start of one blink to the next (a double blink is the exception)
BLINK_MAX_GAP_S = 6.0
BLINK_RATE = 0.30  # blinks per second while no word sounds (about 18 a minute at rest)
BLINK_AT_PHRASE_END = 0.7  # the chance of a blink as a phrase ends
BLINK_DELAY_S = (0.0, 0.08)  # how soon after the phrase end (it is noticed 0.22 s into the pause)
BLINK_HOLD_S = 2.0  # how long past BLINK_MAX_GAP_S a blink waits for a word to end
BLINK_DOUBLE = 0.12  # the chance a blink is followed by another
BLINK_DOUBLE_GAP_S = 0.32

FIXATION_S = (0.2, 0.6)
SACCADE_S = 0.05  # the jump itself
MICRO_SACCADE = 0.02  # as a blendshape weight: the eyes' small jumps while they rest on the viewer
GAZE_MAX = 0.18  # as a blendshape weight: a glance, not a stare off-screen
GLANCE_AT_PHRASE_START = 0.3
GLANCE_IN_PAUSE = 0.35  # at a phrase end
GLANCE_AT_REST = 0.12  # at each fixation while silent
GLANCE_FIXATIONS = (1, 3)  # how many fixations a glance away lasts
DRIFT = 0.006  # the eyes' slow drift within a fixation
HEAD_FOLLOW_DEG = (14.0, 8.0)  # head (yaw, pitch) degrees per unit of gaze: the head turns with a glance
HEAD_FOLLOW_S = 0.35  # the head follows the eyes over about this long

# Micro-motion: (weights, amplitude). Each channel is two slow sines with seeded periods and phases.
MICRO_BROW = ((("browInnerUp",), 0.035), (("browDownLeft", "browDownRight"), 0.025))
MICRO_MOUTH = (
    (("mouthPressLeft", "mouthPressRight"), 0.04),
    (("mouthSmileLeft", "mouthSmileRight"), 0.025),
    (("mouthRollLower",), 0.02),
)
MICRO_PERIODS_S = ((3.0, 7.0), (1.3, 2.9))

_BLINK = (INDEX["eyeBlinkLeft"], INDEX["eyeBlinkRight"])
# (weight raised when the eyes turn this way) for x > 0 = the face's own left, y > 0 = up.
_LOOK_LEFT = (INDEX["eyeLookOutLeft"], INDEX["eyeLookInRight"])
_LOOK_RIGHT = (INDEX["eyeLookInLeft"], INDEX["eyeLookOutRight"])
_LOOK_UP = (INDEX["eyeLookUpLeft"], INDEX["eyeLookUpRight"])
_LOOK_DOWN = (INDEX["eyeLookDownLeft"], INDEX["eyeLookDownRight"])
_BROW_INNER = INDEX["browInnerUp"]
_BROW_OUTER = (INDEX["browOuterUpLeft"], INDEX["browOuterUpRight"])


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class IdleLayer:
    """Call :meth:`apply` once per frame, in order."""

    def __init__(self, seed: int, fps: int) -> None:
        self.fps = fps
        rng = random.Random(seed)
        self._rng = rng
        # Head sway: a few slow sines per axis. (amplitude in degrees, period in seconds, phase)
        self._sway = {
            axis: [
                (amp * rng.uniform(0.7, 1.3), period * rng.uniform(0.8, 1.25), rng.uniform(0, math.tau))
                for amp, period in parts
            ]
            for axis, parts in {
                "yaw": [(1.6, 7.3), (0.7, 3.1)],
                "pitch": [(1.0, 5.9), (0.5, 2.3)],
                "roll": [(0.6, 9.7), (0.3, 4.1)],
            }.items()
        }

        def channels(spec: tuple) -> list[tuple[list[int], float, list[tuple[float, float]]]]:
            return [
                (
                    [INDEX[n] for n in names],
                    amp,
                    [(rng.uniform(lo, hi), rng.uniform(0, math.tau)) for lo, hi in MICRO_PERIODS_S],
                )
                for names, amp in spec
            ]

        self._micro_brow = channels(MICRO_BROW)
        self._micro_mouth = channels(MICRO_MOUTH)
        self._drift = [(rng.uniform(1.5, 3.5), rng.uniform(0, math.tau)) for _ in range(2)]

        self._due: float | None = rng.uniform(0.6, 2.5)  # when the next blink starts
        self._blinking: float | None = None  # when the blink in progress started
        self._last_blink = -BLINK_MIN_GAP_S
        self._gaze_from = (0.0, 0.0)
        self._gaze_to = (0.0, 0.0)
        self._gaze_at = 0.0  # when the jump to _gaze_to began
        self._next_fixation = rng.uniform(*FIXATION_S)
        self._away = 0  # fixations left in a glance away
        self._follow = (0.0, 0.0)  # the head's share of the gaze, smoothed
        self._mouth_free = 1.0  # 1 while not speaking, easing to 0 in speech: the mouth's micro-motion

    # ---- blinks -------------------------------------------------------------------------------

    def _blink(self, t: float, cue: Cue) -> float:
        dt = 1.0 / self.fps
        # The same draws on every frame, so the random sequence depends on nothing but the frame count.
        chance, delay, double = self._rng.random(), self._rng.random(), self._rng.random()
        if self._blinking is not None and t >= self._blinking + BLINK_CLOSE_S + BLINK_OPEN_S:
            self._blinking = None
        earliest = self._last_blink + BLINK_MIN_GAP_S
        if cue.phrase_end and chance < BLINK_AT_PHRASE_END:
            lo, hi = BLINK_DELAY_S
            at = max(t + lo + (hi - lo) * delay, earliest)
            if self._due is None or at < self._due:
                self._due = at
        elif self._due is None:
            gap = t - self._last_blink
            if gap >= BLINK_MAX_GAP_S or (not cue.sounding and chance < BLINK_RATE * dt and t >= earliest):
                self._due = t
        # A blink waits out a sounding word, unless it is well overdue.
        held = cue.sounding and t - self._last_blink < BLINK_MAX_GAP_S + BLINK_HOLD_S
        if self._due is not None and t >= self._due and self._blinking is None and not held:
            # A blink that waited out a word starts now, not when it fell due: started in the past, it
            # would be over (or nearly) before its first frame, and never seen.
            # (A due within rounding of the previous frame was due on that frame, and held there.)
            start = self._due if self._due > t - dt + 1e-9 else t
            self._blinking = self._last_blink = start
            self._due = start + BLINK_DOUBLE_GAP_S if double < BLINK_DOUBLE else None
        if self._blinking is None:
            return 0.0
        since = t - self._blinking
        if since < BLINK_CLOSE_S:
            return _smooth(since / BLINK_CLOSE_S)
        return 1.0 - _smooth((since - BLINK_CLOSE_S) / BLINK_OPEN_S)

    # ---- gaze ---------------------------------------------------------------------------------

    def _jump(self, t: float, to: tuple[float, float]) -> None:
        self._gaze_from = self._gaze_now(t)
        self._gaze_to = to
        self._gaze_at = t

    def _gaze_now(self, t: float) -> tuple[float, float]:
        k = _smooth((t - self._gaze_at) / SACCADE_S)
        return (
            self._gaze_from[0] + (self._gaze_to[0] - self._gaze_from[0]) * k,
            self._gaze_from[1] + (self._gaze_to[1] - self._gaze_from[1]) * k,
        )

    def _gaze(self, t: float, cue: Cue) -> tuple[float, float]:
        rng = self._rng
        # Five draws per frame, always; how many fixations a glance lasts (e) is drawn apart from how long each is (b).
        a, b, c, d, e = rng.random(), rng.random(), rng.random(), rng.random(), rng.random()
        glance = (cue.phrase_start and a < GLANCE_AT_PHRASE_START) or (cue.phrase_end and a < GLANCE_IN_PAUSE)
        if glance and self._away == 0:
            self._away = GLANCE_FIXATIONS[0] + int(e * (GLANCE_FIXATIONS[1] - GLANCE_FIXATIONS[0] + 1))
            self._jump(t, self._away_target(c, d))
            self._next_fixation = t + FIXATION_S[0] + (FIXATION_S[1] - FIXATION_S[0]) * b
        elif t >= self._next_fixation:
            if self._away > 1:
                self._away -= 1
                x, y = self._gaze_to  # a small correction within the glance
                self._jump(t, (x + (c - 0.5) * 2 * MICRO_SACCADE, y + (d - 0.5) * 2 * MICRO_SACCADE))
            elif not cue.speaking and a < GLANCE_AT_REST and self._away == 0:
                self._away = GLANCE_FIXATIONS[0]
                self._jump(t, self._away_target(c, d))
            else:
                self._away = 0  # back to the viewer, with a micro-saccade's scatter
                self._jump(t, ((c - 0.5) * 2 * MICRO_SACCADE, (d - 0.5) * 2 * MICRO_SACCADE * 0.6))
            self._next_fixation = t + FIXATION_S[0] + (FIXATION_S[1] - FIXATION_S[0]) * b
        x, y = self._gaze_now(t)
        drift = [DRIFT * math.sin(math.tau * t / period + phase) for period, phase in self._drift]
        return x + drift[0], y + drift[1]

    @staticmethod
    def _away_target(c: float, d: float) -> tuple[float, float]:
        """A glance away: to one side, and more often up than down."""
        side = (c - 0.5) * 2.0
        return GAZE_MAX * side, GAZE_MAX * 0.6 * (d * 1.4 - 0.4)

    # ---- the frame ----------------------------------------------------------------------------

    def apply(self, frame: int, weights: np.ndarray, cue: Cue | None = None) -> tuple[np.ndarray, Head]:
        """The frame's weights with the idle layer added, and its head rotation."""
        cue = cue or Cue()
        t = frame / self.fps
        dt = 1.0 / self.fps
        out = weights.astype(np.float32, copy=True)

        blink = self._blink(t, cue)
        for i in _BLINK:
            out[i] = max(out[i], blink)  # a driver that blinks keeps its own blinks

        x, y = self._gaze(t, cue)
        for indices, amount in ((_LOOK_LEFT, x), (_LOOK_RIGHT, -x), (_LOOK_UP, y), (_LOOK_DOWN, -y)):
            if amount > 0:
                for i in indices:
                    out[i] += amount

        self._mouth_free += ((0.0 if cue.speaking else 1.0) - self._mouth_free) * (1.0 - math.exp(-dt / 0.15))
        for chans, scale in ((self._micro_brow, 1.0), (self._micro_mouth, self._mouth_free)):
            for indices, amp, sines in chans:
                v = amp * scale * 0.5 * sum(1.0 + math.sin(math.tau * t / p + ph) for p, ph in sines) / len(sines)
                for i in indices:
                    out[i] += v

        if cue.brow:  # a stress's brow raise, over the driver's own brows
            out[_BROW_INNER] += cue.brow
            for i in _BROW_OUTER:
                out[i] += cue.brow * 0.7

        k = 1.0 - math.exp(-dt / HEAD_FOLLOW_S)
        self._follow = (self._follow[0] + (x - self._follow[0]) * k, self._follow[1] + (y - self._follow[1]) * k)
        angles = {
            axis: sum(amp * math.sin(math.tau * t / period + phase) for amp, period, phase in parts)
            for axis, parts in self._sway.items()
        }
        yaw = angles["yaw"] + HEAD_FOLLOW_DEG[0] * self._follow[0]
        pitch = angles["pitch"] - HEAD_FOLLOW_DEG[1] * self._follow[1] + cue.pitch  # positive pitch is down
        roll = angles["roll"] + cue.roll
        head = Head(yaw=round(yaw, 3), pitch=round(pitch, 3), roll=round(roll, 3))
        return np.clip(out, 0.0, 1.0), head
