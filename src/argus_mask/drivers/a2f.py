"""The ``a2f`` driver: a client of the Audio2Face server (``a2f/`` in this repo).

The server is reached over one WebSocket per stream, ``<url>/drive``:

* client -> ``{"fps": 30, "sample_rate": 16000}`` (text)
* server -> ``{"blendshapes": [name, ...]}`` (text): the order of its weights
* client -> 16 kHz mono ``f32le`` PCM (binary), as it arrives
* server -> ``{"i": n, "weights": [...]}`` (text), one per finished frame, in order
* client -> ``{"end": true}``; the server sends its remaining frames, then ``{"end": true}``

Names the server sends that are not ARKit's are ignored; ARKit names it does
not send stay at zero. ``GET <url>/health`` answering 200 means it can run.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import numpy as np

from argus_mask.arkit import ARKIT_52, INDEX
from argus_mask.audio import DRIVER_RATE
from argus_mask.drivers import DriverError

HEALTH_TIMEOUT_S = 3.0
OPEN_TIMEOUT_S = 10.0
FINISH_TIMEOUT_S = 30.0


def _ws_url(base: str) -> str:
    scheme, sep, rest = base.rstrip("/").partition("://")
    if not sep:
        raise DriverError(f"ARGUS_MASK_A2F_URL {base!r} is not a URL")
    return f"{'wss' if scheme in ('https', 'wss') else 'ws'}://{rest}/drive"


class A2FSession:
    def __init__(self, ws, order: list[int | None]) -> None:  # noqa: ANN001 - websockets is optional
        self._ws = ws
        self._order = order  # position in the server's weights -> index in ARKIT_52, or None
        self._next = 0
        self._ended = False

    def _frame(self, message: dict) -> np.ndarray:
        if message.get("i") != self._next:
            raise DriverError(f"a2f sent frame {message.get('i')!r}, expected {self._next}")
        raw = message.get("weights")
        if not isinstance(raw, list) or len(raw) != len(self._order):
            raise DriverError("a2f sent a frame whose weights do not match its blendshape list")
        self._next += 1
        out = np.zeros(len(ARKIT_52), dtype=np.float32)
        for value, index in zip(raw, self._order, strict=True):
            if index is not None:
                out[index] = value
        return np.clip(np.nan_to_num(out), 0.0, 1.0)

    def _take(self, raw: str | bytes) -> np.ndarray | None:
        try:
            message = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise DriverError("a2f sent a message that is not JSON") from exc
        if not isinstance(message, dict):
            raise DriverError("a2f sent a message that is not an object")
        if message.get("error"):
            raise DriverError(f"a2f failed: {message['error']}")
        if message.get("end"):
            self._ended = True
            return None
        return self._frame(message)

    async def _recv(self, timeout: float | None) -> np.ndarray | None:
        import websockets  # noqa: PLC0415

        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout)
        except websockets.ConnectionClosed as exc:
            raise DriverError("a2f closed the stream") from exc
        return self._take(raw)

    async def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        import websockets  # noqa: PLC0415

        try:
            await self._ws.send(pcm.astype("<f4", copy=False).tobytes())
        except websockets.ConnectionClosed as exc:
            raise DriverError("a2f closed the stream") from exc
        out: list[np.ndarray] = []
        while not self._ended:
            try:
                frame = await self._recv(0.001)  # only what has already arrived
            except TimeoutError:
                break
            if frame is not None:
                out.append(frame)
        return out

    async def finish(self) -> list[np.ndarray]:
        import websockets  # noqa: PLC0415

        try:
            await self._ws.send(json.dumps({"end": True}))
        except websockets.ConnectionClosed as exc:
            raise DriverError("a2f closed the stream") from exc
        out: list[np.ndarray] = []
        while not self._ended:
            try:
                frame = await self._recv(FINISH_TIMEOUT_S)
            except TimeoutError as exc:
                raise DriverError(f"a2f did not finish within {FINISH_TIMEOUT_S:g}s") from exc
            if frame is not None:
                out.append(frame)
        return out

    async def close(self) -> None:
        await self._ws.close()


class A2FDriver:
    name = "a2f"

    def __init__(self, url: str) -> None:
        self.url = url.strip()
        self.model: str | None = None

    def _health(self) -> str | None:
        if not self.url:
            return "ARGUS_MASK_A2F_URL is not set"
        try:
            # The operator's own configured service: no proxy, no redirects to follow blindly.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(f"{self.url.rstrip('/')}/health", timeout=HEALTH_TIMEOUT_S) as reply:
                body = json.loads(reply.read(65536) or b"{}")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return f"the Audio2Face server at {self.url} does not answer: {exc}"
        if isinstance(body, dict):
            if isinstance(body.get("model"), str):
                self.model = body["model"]
            if body.get("ready") is False:
                return f"the Audio2Face server at {self.url} is not ready: {body.get('reason') or 'loading'}"
        return None

    async def check(self) -> str | None:
        if not self.url.startswith(("http://", "https://")):
            return (
                "ARGUS_MASK_A2F_URL is not set" if not self.url else f"ARGUS_MASK_A2F_URL {self.url!r} is not http(s)"
            )
        return await asyncio.to_thread(self._health)

    async def open(self, fps: int) -> A2FSession:
        try:
            import websockets  # noqa: PLC0415
        except ImportError as exc:
            raise DriverError("the a2f driver needs websockets: install argus-mask[a2f]") from exc
        try:
            ws = await websockets.connect(_ws_url(self.url), open_timeout=OPEN_TIMEOUT_S, max_size=2**22, proxy=None)
        except (OSError, TimeoutError, websockets.WebSocketException) as exc:
            raise DriverError(f"the Audio2Face server at {self.url} does not answer: {exc}") from exc
        try:
            await ws.send(json.dumps({"fps": fps, "sample_rate": DRIVER_RATE}))
            hello = json.loads(await asyncio.wait_for(ws.recv(), OPEN_TIMEOUT_S))
            if not isinstance(hello, dict) or hello.get("error"):
                raise DriverError(
                    f"a2f refused the stream: {hello.get('error') if isinstance(hello, dict) else hello!r}"
                )
            names = hello.get("blendshapes")
            if not isinstance(names, list) or not names:
                raise DriverError("a2f did not say which blendshapes it sends")
        except (TimeoutError, ValueError, websockets.WebSocketException) as exc:
            await ws.close()
            raise DriverError(f"a2f did not open the stream: {exc}") from exc
        except DriverError:
            await ws.close()
            raise
        return A2FSession(ws, [INDEX.get(name) if isinstance(name, str) else None for name in names])
