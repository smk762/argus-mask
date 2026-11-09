"""Drivers turn 16 kHz mono audio into ARKit blendshape weights.

A driver is the seam of the package: everything downstream of it (the idle
layer, the stream, the viewer) is tested with ``energy``, which needs no GPU.

A :class:`DriverSession` is fed audio as it arrives and hands back whatever
whole frames it has finished, in order. Frame *n* covers the audio from
``n / fps`` seconds. Weights come back in :data:`argus_mask.arkit.ARKIT_52`
order, whatever order the model behind the driver uses.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

DRIVERS = ("energy", "a2f")


class DriverError(RuntimeError):
    """The driver cannot run or failed mid-stream: its service is down, or answered nonsense."""


@runtime_checkable
class DriverSession(Protocol):
    async def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Take 16 kHz mono float32; return the frames finished so far, each 52 weights."""
        ...

    async def finish(self) -> list[np.ndarray]:
        """No more audio: return every remaining frame."""
        ...

    async def close(self) -> None: ...


@runtime_checkable
class Driver(Protocol):
    name: str
    model: str | None

    async def check(self) -> str | None:
        """``None`` when the driver can run, otherwise the reason it cannot."""
        ...

    async def open(self, fps: int) -> DriverSession: ...


def build_driver(name: str, *, a2f_url: str = "") -> Driver:
    """A driver by name, one of :data:`DRIVERS`."""
    key = (name or "energy").strip().lower()
    if key == "energy":
        from argus_mask.drivers.energy import EnergyDriver  # noqa: PLC0415

        return EnergyDriver()
    if key == "a2f":
        from argus_mask.drivers.a2f import A2FDriver  # noqa: PLC0415

        return A2FDriver(a2f_url)
    raise ValueError(f"unknown driver {name!r}; choose from: {', '.join(DRIVERS)}")
