"""One performance: audio in, finished frames out.

A :class:`Performer` joins the pieces in order: resample to the driver's rate,
drive, read the voice's cues (stresses, pauses) from the same audio, add the
idle layer with them, stamp each frame with its time on the audio timeline. The server runs one per source; ``argus-mask drive`` runs one over a
file. Both feed it the same block size, so a file driven offline and the same
file pushed live give the same frames.
"""

from __future__ import annotations

import numpy as np

from argus_mask.arkit import ARKIT_52
from argus_mask.audio import DRIVER_RATE, Resampler
from argus_mask.drivers import Driver, DriverSession
from argus_mask.idle import IdleLayer
from argus_mask.models import AudioFormat, DriverInfo, Frame, StreamEnd, StreamHeader
from argus_mask.prosody import Prosody

BLOCK_MS = 100  # how much audio the offline path feeds at a time, and what `push` sends
WEIGHT_DECIMALS = 4


class Performer:
    def __init__(self, session: DriverSession, *, in_rate: int, fps: int, seed: int) -> None:
        self.fps = fps
        self._session = session
        self._resampler = Resampler(in_rate, DRIVER_RATE)
        self._idle = IdleLayer(seed, fps)
        self._prosody = Prosody(fps)
        self._seq = 0
        self._samples_in = 0
        self._in_rate = in_rate

    def _frames(self, weights: list[np.ndarray]) -> list[Frame]:
        out = []
        for w in weights:
            layered, head = self._idle.apply(self._seq, w, self._prosody.cue(self._seq))
            out.append(
                Frame(
                    seq=self._seq,
                    t_ms=round(self._seq * 1000 / self.fps),
                    weights=[round(float(v), WEIGHT_DECIMALS) for v in layered],
                    head=head,
                )
            )
            self._seq += 1
        return out

    async def feed(self, mono: np.ndarray) -> list[Frame]:
        """Take mono float32 at the source's rate; return the frames it completed."""
        self._samples_in += mono.size
        pcm = self._resampler.feed(mono)
        self._prosody.push(pcm)  # before the driver: a frame's audio is always there when its frame is
        return self._frames(await self._session.feed(pcm))

    async def finish(self) -> tuple[list[Frame], StreamEnd]:
        """No more audio: the remaining frames, and the stream's closing message."""
        tail = self._resampler.feed(np.zeros(0, dtype=np.float32), last=True)
        self._prosody.push(tail)
        weights = await self._session.feed(tail) if tail.size else []
        frames = self._frames(weights + await self._session.finish())
        return frames, StreamEnd(frames=self._seq, duration_ms=round(self._samples_in * 1000 / self._in_rate))

    async def close(self) -> None:
        await self._session.close()


def make_header(
    *, face_id: str, session_id: str, fps: int, seed: int, sample_rate: int, driver: Driver
) -> StreamHeader:
    return StreamHeader(
        face_id=face_id,
        session_id=session_id,
        fps=fps,
        seed=seed,
        blendshapes=list(ARKIT_52),
        # What a viewer is sent, whatever the source sent: mono float32 at the source's rate.
        audio=AudioFormat(sample_rate=sample_rate, channels=1, format="f32le"),
        driver=DriverInfo(name=driver.name, model=driver.model),
    )


def blocks(mono: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    """*mono* cut into :data:`BLOCK_MS` pieces."""
    size = max(1, sample_rate * BLOCK_MS // 1000)
    return [mono[i : i + size] for i in range(0, mono.size, size)]


async def perform(
    mono: np.ndarray,
    sample_rate: int,
    driver: Driver,
    *,
    fps: int,
    seed: int,
    face_id: str = "",
    session_id: str = "take",
) -> tuple[StreamHeader, list[Frame], StreamEnd]:
    """A whole clip, offline."""
    header = make_header(
        face_id=face_id, session_id=session_id, fps=fps, seed=seed, sample_rate=sample_rate, driver=driver
    )
    performer = Performer(await driver.open(fps), in_rate=sample_rate, fps=fps, seed=seed)
    try:
        frames: list[Frame] = []
        for block in blocks(mono, sample_rate):
            frames.extend(await performer.feed(block))
        tail, end = await performer.finish()
        return header, frames + tail, end
    finally:
        await performer.close()
