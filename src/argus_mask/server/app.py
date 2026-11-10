"""The argus-mask server.

``GET /health``                       version, the driver and whether it answers, free VRAM, ``read_only``, live faces
``GET /faces``                        the faces that may be served
``GET /face/{id}``                    one face's manifest
``GET /face/{id}/model``              its ``.glb``
``WS  /session/{face_id}/source``     audio in: one JSON hello, then binary PCM. One source per face.
``WS  /session/{face_id}/view``       the stream out: a header, then audio chunks and frames
``GET /``                             the viewer (static files in the package)

A source is a write and a viewer hears the audio, so both sockets apply the
same origin rule as cortex's write guard: no ``Origin`` (a CLI), the request's
own host, or a configured origin. CORS does not cover WebSockets, so without
this any page the operator visits could open either one. The server binds to
localhost by default.
"""

from __future__ import annotations

import asyncio
import json
import struct
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio
import structlog
from argus_cortex.gpu import free_vram_mb
from argus_cortex.server import WriteGuard, constant_refuse, cross_site_refuse
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from argus_mask import __version__
from argus_mask.audio import PcmDecoder
from argus_mask.config import Settings
from argus_mask.drivers import Driver, DriverError, build_driver
from argus_mask.faces import FaceError, FaceStore
from argus_mask.models import DriverHealth, FaceManifest, Health, SourceHello, StreamHeader
from argus_mask.performance import Performer, make_header
from argus_mask.take import line

log = structlog.get_logger(__name__)

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"
READ_ONLY_REFUSAL = "this server is read-only (ARGUS_MASK_READ_ONLY)"
HELLO_TIMEOUT_S = 10.0
DRIVER_CHECK_TTL_S = 2.0
MAX_CHUNK_BYTES = 4 * 1024 * 1024  # one source message: over 5 s of 48 kHz stereo float
# A viewer this many messages behind is dropped. Sources send in real time plus a small lead, so a
# healthy viewer's queue stays near empty.
VIEWER_QUEUE = 2048
# Audio is released once frames reach its end, give or take this many frames (the resampler holds
# a few samples back, so the frame under a chunk's last milliseconds comes with the next chunk).
AUDIO_SLACK_FRAMES = 2
# A driver this far behind the audio is not going to catch up: the stream is ended.
MAX_DRIVER_LAG_MS = 10_000.0

# WebSocket close codes.
POLICY = 1008
TRY_LATER = 1013
INTERNAL = 1011


class Stage:
    """One face's live state: the stream in progress, if any, and who is watching."""

    def __init__(self, face_id: str) -> None:
        self.face_id = face_id
        self.header: StreamHeader | None = None
        self.claimed = False  # a source holds the face, from its connect to its last frame
        self.viewers: set[asyncio.Queue[str | bytes | None]] = set()

    def send(self, message: str | bytes) -> None:
        for queue in list(self.viewers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Too far behind to catch up. Its own loop sees the None and closes the socket.
                self.viewers.discard(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)


class _DriverProbe:
    """Whether the driver can run, asked at most every :data:`DRIVER_CHECK_TTL_S`: for ``a2f`` it is an HTTP call."""

    def __init__(self, driver: Driver) -> None:
        self._driver = driver
        self._at = float("-inf")
        self._reason: str | None = None

    async def reason(self) -> str | None:
        if time.monotonic() - self._at > DRIVER_CHECK_TTL_S:
            self._reason = await self._driver.check()
            self._at = time.monotonic()
        return self._reason

    def forget(self) -> None:
        """The driver just failed: ask it again next time."""
        self._at = float("-inf")


def create_app(settings: Settings | None = None, *, driver: Driver | None = None) -> FastAPI:
    """The app. With no arguments it reads ``ARGUS_MASK_*`` from the environment."""
    cfg = settings or Settings.from_env()
    driver = driver or build_driver(cfg.driver, a2f_url=cfg.a2f_url)
    store = FaceStore(cfg.assets_dir)
    stages: dict[str, Stage] = {}
    app = FastAPI(
        title="Argus Mask", description="A talking face: audio in, animation frames out.", version=__version__
    )
    app.state.settings = cfg
    app.state.driver = driver
    app.state.stages = stages

    wildcard = "*" in cfg.cors_origins
    named = [o for o in cfg.cors_origins if o != "*"]
    origin_refusal = cross_site_refuse(named)
    # A wildcard grants anonymous reads, never a cross-site write: naming an origin is the way to get those.
    app.add_middleware(WriteGuard, refuse=origin_refusal)
    if cfg.read_only:
        app.add_middleware(WriteGuard, refuse=constant_refuse(READ_ONLY_REFUSAL))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else named,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def stage_for(face_id: str) -> Stage:
        return stages.setdefault(face_id, Stage(face_id))

    def face_or_404(face_id: str) -> tuple[FaceManifest, Path]:
        try:
            return store.load(face_id)
        except FaceError:
            # One answer for "no such face" and "refused": a refused face is not served, and says so only to `inspect`.
            raise HTTPException(status_code=404, detail=f"no face {face_id!r}") from None

    probe = _DriverProbe(driver)

    @app.get("/health", response_model=Health)
    async def health() -> Health:
        reason = await probe.reason()
        faces, _ = store.scan()
        return Health(
            version=__version__,
            driver=DriverHealth(name=driver.name, model=driver.model, available=reason is None, reason=reason),
            fps=cfg.fps,
            free_vram_mb=free_vram_mb(),
            read_only=cfg.read_only,
            faces=len(faces),
            live=sorted(fid for fid, stage in stages.items() if stage.claimed),
        )

    @app.get("/faces", response_model=list[FaceManifest])
    def faces() -> list[FaceManifest]:
        return store.scan()[0]

    @app.get("/face/{face_id}", response_model=FaceManifest)
    def face(face_id: str) -> FaceManifest:
        return face_or_404(face_id)[0]

    @app.get("/face/{face_id}/model")
    def face_model(face_id: str) -> FileResponse:
        manifest, path = face_or_404(face_id)
        # The sha256 in the manifest changes when the model does, so a viewer may cache by it.
        return FileResponse(path, media_type="model/gltf-binary", headers={"ETag": f'"{manifest.model.sha256}"'})

    def cross_site(ws: WebSocket) -> str | None:
        return origin_refusal({"type": "websocket", "method": "WebSocket"}, ws.headers)  # type: ignore[arg-type]

    async def refuse(ws: WebSocket, detail: str, code: int = POLICY) -> None:
        """Accept, say why, close: a client that is only told "403" cannot report the reason."""
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "detail": detail}))
        await ws.close(code=code, reason=detail[:120])

    @app.websocket("/session/{face_id}/source")
    async def source(ws: WebSocket, face_id: str) -> None:
        if cross_site(ws) is not None:
            await ws.close(code=POLICY)  # before accept: a cross-site page gets a plain 403
            return
        if cfg.read_only:
            await refuse(ws, READ_ONLY_REFUSAL)
            return
        try:
            store.load(face_id)
        except FaceError:
            await refuse(ws, f"no face {face_id!r}")
            return
        stage = stage_for(face_id)
        if stage.claimed:
            await refuse(ws, f"face {face_id!r} already has a source", TRY_LATER)
            return
        stage.claimed = True
        performer: Performer | None = None
        try:
            if (reason := await probe.reason()) is not None:
                await refuse(ws, f"the {driver.name} driver is down: {reason}", TRY_LATER)
                return
            await ws.accept()
            try:
                hello = SourceHello.model_validate_json(await asyncio.wait_for(ws.receive_text(), HELLO_TIMEOUT_S))
            except (ValidationError, TimeoutError, KeyError) as exc:
                await ws.send_text(
                    json.dumps({"type": "error", "detail": f"the first message must be the audio format: {exc}"})
                )
                await ws.close(code=POLICY)
                return
            try:
                performer = Performer(
                    await driver.open(cfg.fps), in_rate=hello.sample_rate, fps=cfg.fps, seed=hello.seed
                )
            except DriverError as exc:
                probe.forget()  # ask the driver again next time
                await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
                await ws.close(code=TRY_LATER)
                return
            header = make_header(
                face_id=face_id,
                session_id=uuid.uuid4().hex,
                fps=cfg.fps,
                seed=hello.seed,
                sample_rate=hello.sample_rate,
                driver=driver,
            )
            stage.header = header
            stage.send(line(header))
            await ws.send_text(json.dumps({"type": "ready", "session_id": header.session_id, "fps": cfg.fps}))
            log.info("source_connected", face=face_id, session=header.session_id, sample_rate=hello.sample_rate)

            decoder = PcmDecoder(hello)
            samples = 0
            failure: str | None = None
            # A driver answers late: Audio2Face needs a quarter second of audio that has not been
            # spoken yet. So audio is held until frames reach its end, and a viewer always has the
            # mouth for a sound before it has the sound.
            held: deque[tuple[float, bytes]] = deque()  # (where the chunk ends, in ms; the message)
            covered_ms = 0.0
            slack_ms = AUDIO_SLACK_FRAMES * 1000.0 / cfg.fps

            def release(*, everything: bool = False) -> None:
                while held and (everything or held[0][0] <= covered_ms + slack_ms):
                    stage.send(held.popleft()[1])

            try:
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("text") is not None:
                        try:
                            note = json.loads(message["text"])
                        except ValueError:
                            continue
                        if isinstance(note, dict) and note.get("type") == "end":
                            break
                        continue
                    data = message.get("bytes") or b""
                    if len(data) > MAX_CHUNK_BYTES:
                        failure = f"an audio message over {MAX_CHUNK_BYTES} bytes"
                        break
                    mono = decoder.feed(data)
                    if not mono.size:
                        continue
                    start_ms = samples * 1000.0 / hello.sample_rate
                    samples += mono.size
                    end_ms = samples * 1000.0 / hello.sample_rate
                    held.append((end_ms, struct.pack("<d", start_ms) + mono.astype("<f4", copy=False).tobytes()))
                    for frame in await performer.feed(mono):
                        stage.send(line(frame))
                        covered_ms = frame.t_ms + 1000.0 / cfg.fps
                    release()
                    if end_ms - covered_ms > MAX_DRIVER_LAG_MS:
                        failure = (
                            f"the {driver.name} driver fell more than {MAX_DRIVER_LAG_MS / 1000:g}s behind the audio"
                        )
                        break
            except DriverError as exc:
                failure = str(exc)
                probe.forget()
            except WebSocketDisconnect:
                pass

            frames = 0
            duration_ms = round(samples * 1000 / hello.sample_rate)
            if failure is None:
                try:
                    tail, end = await performer.finish()
                    for frame in tail:
                        stage.send(line(frame))
                    release(everything=True)
                    stage.send(line(end))
                    frames, duration_ms = end.frames, end.duration_ms
                except DriverError as exc:
                    failure = str(exc)
                    probe.forget()
            if failure is not None:
                # Viewers are told the stream is over; what they already have is still valid. Audio no
                # frame reached is dropped: playing it would be sound with no mouth for it.
                held.clear()
                stage.send(json.dumps({"type": "end", "frames": 0, "duration_ms": duration_ms, "error": failure}))
                log.warning("source_failed", face=face_id, session=header.session_id, error=failure)
            else:
                log.info(
                    "source_finished", face=face_id, session=header.session_id, frames=frames, duration_ms=duration_ms
                )
            # Free the face before telling the source it is over: a source that reconnects the moment
            # it reads "done" must not be told the face is taken.
            stage.header = None
            stage.claimed = False
            try:
                if failure is not None:
                    await ws.send_text(json.dumps({"type": "error", "detail": failure}))
                    await ws.close(code=INTERNAL)
                else:
                    await ws.send_text(json.dumps({"type": "done", "frames": frames, "duration_ms": duration_ms}))
                    await ws.close()
            except (RuntimeError, WebSocketDisconnect):
                pass  # the source already went away
        finally:
            if performer is not None:
                await performer.close()
            stage.header = None
            stage.claimed = False

    @app.websocket("/session/{face_id}/view")
    async def view(ws: WebSocket, face_id: str) -> None:
        if cross_site(ws) is not None:
            await ws.close(code=POLICY)
            return
        try:
            store.load(face_id)
        except FaceError:
            await refuse(ws, f"no face {face_id!r}")
            return
        await ws.accept()
        stage = stage_for(face_id)
        queue: asyncio.Queue[str | bytes | None] = asyncio.Queue(maxsize=VIEWER_QUEUE)
        if stage.header is not None:
            queue.put_nowait(line(stage.header))  # joined mid-stream: the header, then whatever comes next
        stage.viewers.add(queue)

        async def pump() -> None:
            while True:
                message = await queue.get()
                if message is None:
                    await ws.close(code=TRY_LATER, reason="too far behind")
                    return
                if isinstance(message, bytes):
                    await ws.send_bytes(message)
                else:
                    await ws.send_text(message)

        async def watch() -> None:
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass  # a viewer has nothing to say; only its leaving matters

        async def first(job: Callable[[], Awaitable[None]], scope: anyio.CancelScope) -> None:
            try:
                await job()
            except (WebSocketDisconnect, RuntimeError):
                pass  # the socket went away under a send
            finally:
                scope.cancel()

        try:
            # A task group, not bare asyncio tasks: the server and the test client both cancel this
            # handler through anyio, and only anyio's own tasks unwind cleanly under that.
            async with anyio.create_task_group() as group:
                group.start_soon(first, pump, group.cancel_scope)
                group.start_soon(first, watch, group.cancel_scope)
        finally:
            stage.viewers.discard(queue)

    if VIEWER_DIR.is_dir():
        # Last, so it never shadows a route above.
        app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")
    return app
