"""The argus-mask server.

``GET /health``                       version, the driver and whether it answers, free VRAM, ``read_only``, live faces
``GET /faces``                        the faces that may be served
``GET /aliases``                      stable names for faces (alias -> face id); every route taking a face takes one
``GET /face/{id}``                    one face's manifest
``GET /face/{id}/rest``               the rest set it shows (``?set=`` another); ``GET /expressions`` lists them
``GET /face/{id}/model``              its model: a ``.glb``, or a LAM splat head's ``.zip``
``GET /face/{id}/model/{file}``       the same, by its file name (the splat renderer wants a URL ending ``.zip``)
``WS  /session/{face_id}/source``     audio in: one JSON hello, then binary PCM. One source per face.
``WS  /session/{face_id}/view``       the stream out: a header, then audio chunks and frames
``POST /push?face=…``                 audio in over HTTP (a WAV, or PCM with format headers), played as a source
``POST /speak``                       text through the configured TTS service, then as ``/push``
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
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import anyio
import structlog
from argus_cortex.gpu import free_vram_mb
from argus_cortex.server import WriteGuard, constant_refuse, cross_site_refuse
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from argus_mask import __version__, client
from argus_mask.audio import AudioError, PcmDecoder, decode_wav
from argus_mask.config import Settings
from argus_mask.consent import CONSENT_USES, ConsentVerdict, voice_disclosed, voice_refusal
from argus_mask.drivers import Driver, DriverError, build_driver
from argus_mask.expressions import DEFAULT_SET, ExpressionError, Expressions
from argus_mask.faces import MEDIA_TYPES, Aliases, FaceError, FaceStore
from argus_mask.models import (
    FACE_ID_PATTERN,
    VOICE_DISCLOSURE,
    DriverHealth,
    FaceManifest,
    Health,
    RestSet,
    Screen,
    SourceHello,
    StreamHeader,
    ViewerStatus,
)
from argus_mask.performance import BLOCK_MS, Performer, make_header
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
# A viewer's status message is a few fields; anything longer is not one.
MAX_STATUS_BYTES = 1024
# POST /push: the most audio one request may carry (minutes of 48 kHz 16-bit), and how far ahead of real time
# it is fed to the face, as a client source runs ahead.
MAX_PUSH_BYTES = 25 * 1024 * 1024
PUSH_LEAD_S = 1.0

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
        # The viewers that have said what they are (by their queue): the screens. A socket client
        # that never reports (record, an older page) watches, but is not counted as a screen.
        self.screens: dict[asyncio.Queue[str | bytes | None], Screen] = {}

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


class SourceRefused(RuntimeError):
    """The driver would not open a session for a source."""


@dataclass
class Played:
    """How a source's stream ended: the closing numbers, or why it failed."""

    session_id: str
    frames: int
    duration_ms: int
    failure: str | None


class Speak(BaseModel):
    """``POST /speak``: what ``argus-mask say`` takes, for an agent that hands the server text."""

    model_config = ConfigDict(extra="forbid")
    face: str = Field(..., pattern=FACE_ID_PATTERN)
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = Field(
        "",
        max_length=200,
        description="default: ARGUS_MASK_TTS_VOICE; a clone's name, a speaker number, or speaker:<name>",
    )
    instruction: str = Field(
        "", max_length=500, description="how to say it, for a service that takes one (Qwen3-TTS CustomVoice)"
    )
    seed: int = Field(0, ge=0, le=2**31 - 1)
    temperature: float | None = Field(None, ge=0.1, le=1.5)
    tidy: bool = Field(True, description="numbers as words, short sentences joined, soft edges, even loudness")
    pause: int = Field(300, ge=0, le=10_000, description="ms between sentences; 0 sends the text whole")
    require_screen: bool = False


async def read_body(request: Request) -> bytes:
    """The request's body, refused (413) over :data:`MAX_PUSH_BYTES` before and while it is read."""
    too_big = HTTPException(413, f"send at most {MAX_PUSH_BYTES // 2**20} MB of audio")
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > MAX_PUSH_BYTES):
        raise too_big
    data = bytearray()
    async for piece in request.stream():
        data += piece
        if len(data) > MAX_PUSH_BYTES:
            raise too_big
    if not data:
        raise HTTPException(422, "no audio: the body is empty")
    return bytes(data)


def audio_of(request: Request, data: bytes, seed: int) -> tuple[SourceHello, bytes]:
    """A pushed body as (its format, its PCM): a WAV file, or raw PCM described by the TTS contract's headers."""
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if data[:4] == b"RIFF" or kind in ("audio/wav", "audio/x-wav", "audio/wave"):
        try:
            mono, rate = decode_wav(data, "the pushed WAV")
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not mono.size:
            raise HTTPException(422, "the pushed WAV has no audio")
        try:
            hello = SourceHello(sample_rate=rate, channels=1, format="f32le", seed=seed)
        except ValidationError as exc:
            raise HTTPException(
                422, f"the pushed WAV's format is not one this plays: {exc.errors()[0]['msg']}"
            ) from exc
        return hello, mono.astype("<f4", copy=False).tobytes()
    rate = request.headers.get("x-sample-rate")
    declared = (request.headers.get("x-audio-format") or "").lower()
    if rate is None or declared not in client._FORMATS:
        raise HTTPException(
            422,
            "send a WAV file, or raw PCM with X-Sample-Rate, X-Audio-Format (pcm_f32le or pcm_s16le) and X-Channels",
        )
    try:
        hello = SourceHello(
            sample_rate=int(rate),
            channels=int(request.headers.get("x-channels") or 1),
            format=client._FORMATS[declared],
            seed=seed,
        )
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, f"the PCM headers do not describe a format this plays: {exc}") from exc
    if len(data) < client._frame_bytes(hello):
        raise HTTPException(422, "no audio: less than one sample")
    return hello, data


async def paced(pcm: bytes, fmt: SourceHello) -> AsyncIterator[bytes]:
    """*pcm* in :data:`BLOCK_MS` pieces, no faster than real time plus :data:`PUSH_LEAD_S`, as a client source sends."""
    frame = client._frame_bytes(fmt)
    size = fmt.sample_rate * BLOCK_MS // 1000 * frame
    started = time.monotonic()
    sent_s = 0.0
    for start in range(0, len(pcm), size):
        piece = pcm[start : start + size]
        yield piece
        sent_s += len(piece) / frame / fmt.sample_rate
        ahead = sent_s - (time.monotonic() - started) - PUSH_LEAD_S
        if ahead > 0:
            await asyncio.sleep(ahead)


def install_guards(app: FastAPI, cfg: Settings) -> Callable:
    """The write guards and CORS every mask app uses. Returns the origin predicate, for the sockets to call."""
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
    return origin_refusal


def create_app(settings: Settings | None = None, *, driver: Driver | None = None) -> FastAPI:
    """The app. With no arguments it reads ``ARGUS_MASK_*`` from the environment."""
    cfg = settings or Settings.from_env()
    driver = driver or build_driver(cfg.driver, a2f_url=cfg.a2f_url)
    store = FaceStore(cfg.assets_dir)
    aliases = Aliases(store)
    expressions = Expressions(cfg.assets_dir)
    stages: dict[str, Stage] = {}  # by the name sources and viewers use: a face id, or an alias
    app = FastAPI(
        title="Argus Mask", description="A talking face: audio in, animation frames out.", version=__version__
    )
    app.state.settings = cfg
    app.state.driver = driver
    app.state.stages = stages

    origin_refusal = install_guards(app, cfg)

    def stage_for(face_id: str) -> Stage:
        return stages.setdefault(face_id, Stage(face_id))

    def face_or_404(face_id: str) -> tuple[FaceManifest, Path]:
        try:
            return store.load(aliases.resolve(face_id))
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
            screens={fid: list(stage.screens.values()) for fid, stage in sorted(stages.items()) if stage.screens},
            aliases=aliases.all(),
        )

    @app.get("/aliases")
    def alias_list() -> dict[str, str]:
        """Stable names for faces (alias -> face id), as the editor set them in the assets' aliases.json."""
        return aliases.all()

    @app.get("/expressions")
    def expression_list() -> dict:
        """The rest sets (built in, and the assets' expressions.json), and which one each face chose."""
        return {
            "sets": {n: s.model_dump() for n, s in expressions.sets().items()},
            "faces": expressions.choices(),
            "default": DEFAULT_SET,
        }

    @app.get("/face/{face_id}/rest", response_model=RestSet)
    def face_rest(face_id: str, set: str | None = Query(None, max_length=64)) -> RestSet:  # noqa: A002
        """The rest set a face shows: its own choice, or ``?set=`` (a screen's ``?rest=``)."""
        manifest, _ = face_or_404(face_id)
        try:
            return expressions.for_face(manifest.id, set)
        except ExpressionError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get("/faces", response_model=list[FaceManifest])
    def faces() -> list[FaceManifest]:
        return store.scan()[0]

    @app.get("/face/{face_id}", response_model=FaceManifest)
    def face(face_id: str) -> FaceManifest:
        return face_or_404(face_id)[0]

    def model_response(manifest: FaceManifest, path: Path) -> FileResponse:
        # The sha256 in the manifest changes when the model does, so a viewer may cache by it.
        return FileResponse(
            path, media_type=MEDIA_TYPES[manifest.model.format], headers={"ETag": f'"{manifest.model.sha256}"'}
        )

    @app.get("/face/{face_id}/consent", response_model=ConsentVerdict)
    def face_consent(
        face_id: str, use: str = Query(..., pattern="^(" + "|".join(CONSENT_USES) + ")$")
    ) -> ConsentVerdict:
        """Whether a face may be used for *use* (``record`` asks for ``video``): a verdict, never the record.

        A face that is not a real person's needs no consent: ok. One that is not served at all is a 404.
        """
        manifest, _ = face_or_404(face_id)
        if manifest.provenance.kind != "consented":
            return ConsentVerdict(ok=True)
        found = store.consents.check_face(manifest.provenance, use=use, at=store.clock())  # type: ignore[arg-type]
        return found if isinstance(found, ConsentVerdict) else ConsentVerdict(ok=False, detail=found)

    @app.get("/face/{face_id}/model")
    def face_model(face_id: str) -> FileResponse:
        return model_response(*face_or_404(face_id))

    @app.get("/face/{face_id}/model/{file}")
    def face_model_named(face_id: str, file: str) -> FileResponse:
        """The model under its own name: only the manifest's file, never a path the request makes up."""
        manifest, path = face_or_404(face_id)
        if file != manifest.model.file:
            raise HTTPException(status_code=404, detail=f"face {face_id!r} has no file {file!r}")
        return model_response(manifest, path)

    def cross_site(ws: WebSocket) -> str | None:
        return origin_refusal({"type": "websocket", "method": "WebSocket"}, ws.headers)  # type: ignore[arg-type]

    async def refuse(ws: WebSocket, detail: str, code: int = POLICY) -> None:
        """Accept, say why, close: a client that is only told "403" cannot report the reason."""
        await ws.accept()
        await ws.send_text(json.dumps({"type": "error", "detail": detail}))
        await ws.close(code=code, reason=detail[:120])

    async def perform(
        face_id: str,
        stage: Stage,
        hello: SourceHello,
        chunks: AsyncIterator[bytes],
        on_ready: Callable[[StreamHeader], Awaitable[None]] | None = None,
    ) -> Played:
        """Play a source's audio into a claimed face: the one path for the source socket, ``/push`` and ``/speak``.

        Opens the driver (`SourceRefused` if it will not), announces the stream to the viewers, feeds
        *chunks* (PCM in *hello*'s format) to the performer, holds the audio until frames cover it, and
        ends the stream. The face is freed before this returns, so a caller can say "done" at once.
        """
        performer: Performer | None = None
        try:
            try:
                performer = Performer(
                    await driver.open(cfg.fps), in_rate=hello.sample_rate, fps=cfg.fps, seed=hello.seed
                )
            except DriverError as exc:
                probe.forget()  # ask the driver again next time
                raise SourceRefused(str(exc)) from exc
            header = make_header(
                face_id=aliases.resolve(face_id),  # the face itself: a viewer that loaded another reloads
                session_id=uuid.uuid4().hex,
                fps=cfg.fps,
                seed=hello.seed,
                sample_rate=hello.sample_rate,
                driver=driver,
                disclosure=VOICE_DISCLOSURE if hello.disclose else None,
            )
            stage.header = header
            stage.send(line(header))
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
                if on_ready is not None:  # inside: a source gone before "ready" still ends the stream for viewers
                    await on_ready(header)
                async for data in chunks:
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
                    # A driver that falls behind holds feed() until it catches up (the source is not
                    # read meanwhile), and raises DriverError if it stops: no lag check here.
                    release()
            except DriverError as exc:
                failure = str(exc)
                probe.forget()
            except BaseException:  # cancelled (the caller went away), or anything unforeseen: end it for viewers
                held.clear()
                gone = round(samples * 1000 / hello.sample_rate)
                stage.send(
                    json.dumps({"type": "end", "frames": 0, "duration_ms": gone, "error": "the source went away"})
                )
                raise

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
            return Played(header.session_id, frames, duration_ms, failure)
        finally:
            if performer is not None:
                await performer.close()
            stage.header = None
            stage.claimed = False

    @app.websocket("/session/{face_id}/source")
    async def source(ws: WebSocket, face_id: str) -> None:
        if cross_site(ws) is not None:
            await ws.close(code=POLICY)  # before accept: a cross-site page gets a plain 403
            return
        if cfg.read_only:
            await refuse(ws, READ_ONLY_REFUSAL)
            return
        try:
            store.load(aliases.resolve(face_id))
        except FaceError:
            await refuse(ws, f"no face {face_id!r}")
            return
        stage = stage_for(face_id)
        if stage.claimed:
            await refuse(ws, f"face {face_id!r} already has a source", TRY_LATER)
            return
        stage.claimed = True
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

            async def received() -> AsyncIterator[bytes]:
                try:
                    while True:
                        message = await ws.receive()
                        if message["type"] == "websocket.disconnect":
                            return
                        if message.get("text") is not None:
                            try:
                                note = json.loads(message["text"])
                            except ValueError:
                                continue
                            if isinstance(note, dict) and note.get("type") == "end":
                                return
                            continue
                        yield message.get("bytes") or b""
                except WebSocketDisconnect:
                    return

            async def ready(header: StreamHeader) -> None:
                await ws.send_text(json.dumps({"type": "ready", "session_id": header.session_id, "fps": cfg.fps}))

            try:
                played = await perform(face_id, stage, hello, received(), ready)
            except SourceRefused as exc:
                await ws.send_text(json.dumps({"type": "error", "detail": str(exc)}))
                await ws.close(code=TRY_LATER)
                return
            try:
                if played.failure is not None:
                    await ws.send_text(json.dumps({"type": "error", "detail": played.failure}))
                    await ws.close(code=INTERNAL)
                else:
                    done = {"type": "done", "frames": played.frames, "duration_ms": played.duration_ms}
                    await ws.send_text(json.dumps(done))
                    await ws.close()
            except (RuntimeError, WebSocketDisconnect):
                pass  # the source already went away
        finally:
            stage.header = None
            stage.claimed = False

    # -- HTTP sources: for agents that would rather not speak the socket protocol -------------------------

    def http_claim(face_id: str, *, require_screen: bool) -> Stage:
        """The face's stage, if a source may start on it now; otherwise the HTTP refusal, with the reason."""
        face_or_404(face_id)
        stage = stage_for(face_id)
        if stage.claimed:
            raise HTTPException(409, f"face {face_id!r} already has a source")
        if require_screen and not any(screen.audio_ready for screen in stage.screens.values()):
            raise HTTPException(409, f"no screen watching {face_id!r} can be heard (see /health screens)")
        return stage

    async def play_http(face_id: str, stage: Stage, hello: SourceHello, pcm: bytes) -> dict:
        """Claim the face and play *pcm* into it at the pace of speech; the closing numbers, or the HTTP error."""
        if stage.claimed:  # someone started while this request was being read or synthesised
            raise HTTPException(409, f"face {face_id!r} already has a source")
        stage.claimed = True
        try:
            if (reason := await probe.reason()) is not None:
                raise HTTPException(503, f"the {driver.name} driver is down: {reason}")
            try:
                played = await perform(face_id, stage, hello, paced(pcm, hello))
            except SourceRefused as exc:
                raise HTTPException(503, str(exc)) from exc
        finally:
            stage.header = None
            stage.claimed = False
        if played.failure is not None:
            raise HTTPException(502, played.failure)
        return {"session_id": played.session_id, "frames": played.frames, "duration_ms": played.duration_ms}

    @app.post("/push")
    async def push(
        request: Request,
        face: str = Query(..., pattern=FACE_ID_PATTERN, description="the face to animate"),
        seed: int = Query(0, ge=0, le=2**31 - 1, description="the idle layer's seed"),
        require_screen: bool = Query(False, description="refuse (409) unless a screen watching the face can be heard"),
    ) -> dict:
        """Audio into a face, as a source on its socket would send it; answers when the speech ends.

        The body is a WAV file, or raw PCM described by ``X-Sample-Rate``, ``X-Audio-Format``
        (``pcm_f32le`` or ``pcm_s16le``) and ``X-Channels``, the headers a TTS service's stream carries.
        It is played at the pace of speech: the request lasts as long as the audio.
        """
        stage = http_claim(face, require_screen=require_screen)
        data = await read_body(request)
        hello, pcm = audio_of(request, data, seed)
        log.info("push", face=face, bytes=len(pcm), sample_rate=hello.sample_rate)
        return await play_http(face, stage, hello, pcm)

    @app.post("/speak")
    async def speak(req: Speak) -> dict:
        """Text through the configured TTS service (``ARGUS_MASK_TTS_URL``) into a face, as ``argus-mask say`` does.

        The text goes to the TTS service and is never logged. An agent that synthesises its own
        audio should use ``/push``, so the server never sees the text at all.
        """
        tts_url = cfg.tts_url
        if not tts_url.startswith(("http://", "https://", "wyoming://")):
            raise HTTPException(422, "no TTS service is set on this server (ARGUS_MASK_TTS_URL)")
        voice = req.voice or cfg.tts_voice
        if why := voice_refusal(cfg.voices_dir, store.consents, voice, use="avatar", at=store.clock()):
            raise HTTPException(403, why)
        stage = http_claim(req.face, require_screen=req.require_screen)
        body: dict = {"text": req.text}
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.instruction:
            body["instruction"] = req.instruction
        body = client.voiced(body, voice, tts_url)
        log.info("speak", face=req.face, chars=len(req.text))  # never the text itself
        try:
            pcm, fmt, _ = await client.synthesise(tts_url, body, pause_ms=req.pause, tidy=req.tidy)
        except client.ClientError as exc:
            raise HTTPException(502, str(exc)) from exc
        hello = SourceHello(**fmt.model_dump(), seed=req.seed, disclose=voice_disclosed(cfg.voices_dir, voice))
        pcm = client._silence(fmt, client.SAY_LEAD_MS) + pcm + client._silence(fmt, client.SAY_TAIL_MS)
        return await play_http(req.face, stage, hello, pcm)

    @app.websocket("/session/{face_id}/view")
    async def view(ws: WebSocket, face_id: str) -> None:
        if cross_site(ws) is not None:
            await ws.close(code=POLICY)
            return
        try:
            store.load(aliases.resolve(face_id))
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
            # A viewer says only what state it is in (ViewerStatus); anything else, or too big, is ignored.
            while (message := await ws.receive())["type"] != "websocket.disconnect":
                text = message.get("text")
                if not text or len(text) > MAX_STATUS_BYTES or len(text.encode()) > MAX_STATUS_BYTES:
                    continue
                try:
                    status = ViewerStatus.model_validate_json(text)
                except ValidationError:
                    continue
                stage.screens[queue] = Screen(
                    display=status.display,
                    audio_ready=status.audio or status.sound == "external",
                    kiosk=status.kiosk,
                    sound=status.sound,
                )

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
            stage.screens.pop(queue, None)

    if VIEWER_DIR.is_dir():
        # Last, so it never shadows a route above.
        app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")
    return app
