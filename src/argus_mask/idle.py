"""The idle layer: blinks, small eye movement and a slow head sway.

A talking face that never blinks looks dead. This layer is added on top of the
driver's weights. Everything here is a function of the seed and the frame
number, so a take replays identically and two servers given the same seed and
audio agree.
"""

from __future__ import annotations

import math
import random

import numpy as np

from argus_mask.arkit import INDEX
from argus_mask.models import Head

BLINK_CLOSE_S = 0.07
BLINK_OPEN_S = 0.12
BLINK_GAP_S = (2.0, 6.0)
GAZE_HOLD_S = (0.8, 3.0)
GAZE_MOVE_S = 0.08
GAZE_MAX = 0.18  # as a blendshape weight: a glance, not a stare off-screen

_BLINK = (INDEX["eyeBlinkLeft"], INDEX["eyeBlinkRight"])
# (weight raised when the eyes turn this way) for x > 0 = the face's own left, y > 0 = up.
_LOOK_LEFT = (INDEX["eyeLookOutLeft"], INDEX["eyeLookInRight"])
_LOOK_RIGHT = (INDEX["eyeLookInLeft"], INDEX["eyeLookOutRight"])
_LOOK_UP = (INDEX["eyeLookUpLeft"], INDEX["eyeLookUpRight"])
_LOOK_DOWN = (INDEX["eyeLookDownLeft"], INDEX["eyeLookDownRight"])


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
        self._next_blink = rng.uniform(0.6, 2.5)
        self._gaze_from = (0.0, 0.0)
        self._gaze_to = (0.0, 0.0)
        self._gaze_at = 0.0  # when the move to _gaze_to began
        self._next_gaze = rng.uniform(*GAZE_HOLD_S)

    def _blink(self, t: float) -> float:
        while t >= self._next_blink + BLINK_CLOSE_S + BLINK_OPEN_S:
            self._next_blink += BLINK_CLOSE_S + BLINK_OPEN_S + self._rng.uniform(*BLINK_GAP_S)
        since = t - self._next_blink
        if since < 0:
            return 0.0
        if since < BLINK_CLOSE_S:
            return _smooth(since / BLINK_CLOSE_S)
        return 1.0 - _smooth((since - BLINK_CLOSE_S) / BLINK_OPEN_S)

    def _gaze(self, t: float) -> tuple[float, float]:
        while t >= self._next_gaze:
            self._gaze_from = self._gaze_to
            # Mostly back toward the viewer, sometimes a glance away.
            away = self._rng.random() < 0.35
            reach = GAZE_MAX if away else GAZE_MAX * 0.25
            self._gaze_to = (self._rng.uniform(-reach, reach), self._rng.uniform(-reach, reach) * 0.6)
            self._gaze_at = self._next_gaze
            self._next_gaze += self._rng.uniform(*GAZE_HOLD_S)
        k = _smooth((t - self._gaze_at) / GAZE_MOVE_S)
        return (
            self._gaze_from[0] + (self._gaze_to[0] - self._gaze_from[0]) * k,
            self._gaze_from[1] + (self._gaze_to[1] - self._gaze_from[1]) * k,
        )

    def apply(self, frame: int, weights: np.ndarray) -> tuple[np.ndarray, Head]:
        """The frame's weights with the idle layer added, and its head rotation."""
        t = frame / self.fps
        out = weights.astype(np.float32, copy=True)
        blink = self._blink(t)
        for i in _BLINK:
            out[i] = max(out[i], blink)  # a driver that blinks keeps its own blinks
        x, y = self._gaze(t)
        for indices, amount in ((_LOOK_LEFT, x), (_LOOK_RIGHT, -x), (_LOOK_UP, y), (_LOOK_DOWN, -y)):
            if amount > 0:
                for i in indices:
                    out[i] += amount
        angles = {
            axis: sum(amp * math.sin(math.tau * t / period + phase) for amp, period, phase in parts)
            for axis, parts in self._sway.items()
        }
        head = Head(yaw=round(angles["yaw"], 3), pitch=round(angles["pitch"], 3), roll=round(angles["roll"], 3))
        return np.clip(out, 0.0, 1.0), head
