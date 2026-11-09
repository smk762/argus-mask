"""Clients of a running server's source socket: ``push`` a file, ``say`` some text.

Both send audio no faster than real time plus a small lead. A viewer plays on
its own audio clock and buffers what arrives early, but a source that sends a
whole clip at once would bury a slow viewer.

``say`` gets its audio from a TTS service that streams raw PCM and states the
format in ``X-Sample-Rate``, ``X-Audio-Format`` and ``X-Channels`` headers
(Miso TTS does). The text goes to that service and nowhere else.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Callable

import numpy as np

from argus_mask.models import AudioFormat, SourceHello
from argus_mask.performance import blocks

LEAD_S = 1.0  # how far ahead of real time a source may run
TTS_TIMEOUT_S = 180.0  # the first byte waits for a cold model to load
_FORMATS = {"pcm_f32le": "f32le", "f32le": "f32le", "pcm_s16le": "s16le", "s16le": "s16le"}


class ClientError(RuntimeError):
    """The server or the TTS service refused, or went away."""


def source_url(base: str, face_id: str) -> str:
    scheme, sep, rest = base.rstrip("/").partition("://")
    if not sep:
        raise ClientError(f"{base!r} is not a URL")
    return f"{'wss' if scheme in ('https', 'wss') else 'ws'}://{rest}/session/{face_id}/source"


async def _reply(ws, expect: str) -> dict:  # noqa: ANN001 - websockets is optional
    import websockets  # noqa: PLC0415

    try:
        while True:
            raw = await ws.recv()
            if isinstance(raw, bytes):
                continue
            message = json.loads(raw)
            if message.get("type") == "error":
                raise ClientError(message.get("detail") or "the server refused")
            if message.get("type") == expect:
                return message
    except websockets.ConnectionClosed as exc:
        raise ClientError(f"the server closed the socket: {exc.reason or exc.code}") from exc


async def send(
    base: str,
    face_id: str,
    fmt: AudioFormat,
    chunks: AsyncIterator[bytes],
    *,
    seed: int = 0,
    on_ready: Callable[[dict], None] | None = None,
) -> dict:
    """Stream *chunks* of PCM in *fmt* to a face. Returns the server's closing ``done`` message."""
    try:
        import websockets  # noqa: PLC0415
    except ImportError as exc:
        raise ClientError("push and say need websockets: install argus-mask[cli]") from exc
    frame_bytes = (4 if fmt.format == "f32le" else 2) * fmt.channels
    try:
        async with websockets.connect(source_url(base, face_id), max_size=2**22, proxy=None) as ws:
            await ws.send(SourceHello(**fmt.model_dump(), seed=seed).model_dump_json())
            ready = await _reply(ws, "ready")
            if on_ready:
                on_ready(ready)
            started = time.monotonic()
            sent_s = 0.0
            async for chunk in chunks:
                await ws.send(chunk)
                sent_s += len(chunk) / frame_bytes / fmt.sample_rate
                ahead = sent_s - (time.monotonic() - started) - LEAD_S
                if ahead > 0:
                    await asyncio.sleep(ahead)
            await ws.send(json.dumps({"type": "end"}))
            return await _reply(ws, "done")
    except (OSError, websockets.InvalidURI, websockets.InvalidHandshake) as exc:
        raise ClientError(f"cannot reach {base}: {exc}") from exc


async def push(
    base: str,
    face_id: str,
    mono: np.ndarray,
    sample_rate: int,
    *,
    seed: int = 0,
    on_ready: Callable[[dict], None] | None = None,
) -> dict:
    """Play mono float32 audio into a face, in real time."""

    async def chunks() -> AsyncIterator[bytes]:
        for block in blocks(mono, sample_rate):
            yield block.astype("<f4", copy=False).tobytes()

    return await send(base, face_id, AudioFormat(sample_rate=sample_rate), chunks(), seed=seed, on_ready=on_ready)


def open_tts(tts_url: str, body: dict):  # noqa: ANN201 - an http.client.HTTPResponse
    """POST *body* to the TTS service; the streaming response and the audio format it declared."""
    if not tts_url.startswith(("http://", "https://")):
        raise ClientError(
            "ARGUS_MASK_TTS_URL is not set" if not tts_url else f"ARGUS_MASK_TTS_URL {tts_url!r} is not http(s)"
        )
    request = urllib.request.Request(  # noqa: S310 - scheme checked above
        tts_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        reply = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=TTS_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        raise ClientError(f"the TTS service answered {exc.code}: {exc.read(300).decode(errors='replace')}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ClientError(f"the TTS service at {tts_url} does not answer: {exc}") from exc
    kind = (reply.headers.get("X-Audio-Format") or "").lower()
    try:
        fmt = AudioFormat(
            sample_rate=int(reply.headers.get("X-Sample-Rate") or 0),
            channels=int(reply.headers.get("X-Channels") or 1),
            format=_FORMATS[kind],  # type: ignore[arg-type]
        )
    except (KeyError, ValueError) as exc:
        reply.close()
        raise ClientError(
            "the TTS service did not state a PCM format this understands "
            f"(X-Sample-Rate={reply.headers.get('X-Sample-Rate')!r}, X-Audio-Format={kind!r})"
        ) from exc
    return reply, fmt


async def say(
    base: str, face_id: str, tts_url: str, body: dict, *, seed: int = 0, on_ready: Callable[[dict], None] | None = None
) -> dict:
    """Ask the TTS service for *body*'s speech and stream it into a face as it is generated."""
    reply, fmt = await asyncio.to_thread(open_tts, tts_url, body)
    block = fmt.sample_rate // 10 * (4 if fmt.format == "f32le" else 2) * fmt.channels  # ~100 ms

    async def chunks() -> AsyncIterator[bytes]:
        try:
            while data := await asyncio.to_thread(reply.read1, block):
                yield data
        except OSError as exc:
            raise ClientError(f"the TTS stream broke: {exc}") from exc

    try:
        return await send(base, face_id, fmt, chunks(), seed=seed, on_ready=on_ready)
    finally:
        reply.close()
