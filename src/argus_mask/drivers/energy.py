"""The ``energy`` driver: mouth shapes from loudness. CPU only, deterministic.

It is crude on purpose. It opens the jaw with the level of the signal and
widens the mouth on hissy sounds, which reads as "talking" and nothing more:
no visemes, no closed lips on a "b" or an "m". It exists so the whole pipeline
runs and is tested without a GPU, and as a fallback on a host with no NVIDIA
card. The real driver is ``a2f``.
"""

from __future__ import annotations

import numpy as np

from argus_mask.arkit import ARKIT_52, INDEX
from argus_mask.audio import DRIVER_RATE

# Level mapping: RMS in dBFS, from "silence" to "loud speech", onto 0..1.
FLOOR_DB = -48.0
CEIL_DB = -14.0
# One-pole smoothing per frame at 30 fps, rescaled for other rates. A mouth opens faster than it closes.
ATTACK = 0.65
RELEASE = 0.30
# Zero crossings per sample: above this a frame is hiss ("s", "sh", "f") more than voice.
SIBILANT_ZCR = 0.18


class EnergySession:
    def __init__(self, fps: int) -> None:
        self.fps = fps
        self._buf = np.zeros(0, dtype=np.float32)
        self._consumed = 0  # samples before the start of _buf
        self._frame = 0
        self._level = 0.0
        self._hiss = 0.0
        scale = 30.0 / fps
        self._attack = 1.0 - (1.0 - ATTACK) ** scale
        self._release = 1.0 - (1.0 - RELEASE) ** scale

    def _bounds(self, frame: int) -> tuple[int, int]:
        # Integer arithmetic, so frame boundaries never drift at rates that do not divide 16000.
        return frame * DRIVER_RATE // self.fps, (frame + 1) * DRIVER_RATE // self.fps

    def _weights(self, window: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64)))) if window.size else 0.0
        db = 20.0 * np.log10(max(rms, 1e-9))
        target = min(1.0, max(0.0, (db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)))
        self._level += (target - self._level) * (self._attack if target > self._level else self._release)
        zcr = float(np.count_nonzero(np.diff(np.signbit(window)))) / window.size if window.size > 1 else 0.0
        hiss_target = target * min(1.0, zcr / SIBILANT_ZCR) if zcr > SIBILANT_ZCR / 2 else 0.0
        self._hiss += (hiss_target - self._hiss) * self._attack

        level, hiss = self._level, self._hiss
        voiced = level * (1.0 - 0.7 * hiss)
        w = np.zeros(len(ARKIT_52), dtype=np.float32)
        w[INDEX["jawOpen"]] = 0.55 * voiced
        w[INDEX["mouthFunnel"]] = 0.25 * voiced
        w[INDEX["mouthLowerDownLeft"]] = w[INDEX["mouthLowerDownRight"]] = 0.30 * voiced
        w[INDEX["mouthUpperUpLeft"]] = w[INDEX["mouthUpperUpRight"]] = 0.12 * voiced
        w[INDEX["mouthStretchLeft"]] = w[INDEX["mouthStretchRight"]] = 0.35 * hiss
        w[INDEX["mouthSmileLeft"]] = w[INDEX["mouthSmileRight"]] = 0.10 * hiss
        return w

    def _drain(self, *, flush: bool) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        while True:
            start, end = self._bounds(self._frame)
            have = self._consumed + self._buf.size
            if have < end and not (flush and have > start):
                break
            out.append(self._weights(self._buf[start - self._consumed : end - self._consumed]))
            self._frame += 1
        start, _ = self._bounds(self._frame)
        drop = min(self._buf.size, max(0, start - self._consumed))
        self._buf = self._buf[drop:]
        self._consumed += drop
        return out

    async def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, pcm.astype(np.float32, copy=False)])
        return self._drain(flush=False)

    async def finish(self) -> list[np.ndarray]:
        return self._drain(flush=True)

    async def close(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)


class EnergyDriver:
    name = "energy"
    model: str | None = None

    async def check(self) -> str | None:
        return None

    async def open(self, fps: int) -> EnergySession:
        return EnergySession(fps)
