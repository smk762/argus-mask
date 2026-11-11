"""Clients of a running server's source socket: ``push`` a file, ``say`` some text.

Both send audio no faster than real time plus a small lead. A viewer plays on
its own audio clock and buffers what arrives early, but a source that sends a
whole clip at once would bury a slow viewer.

``say`` gets its audio from a TTS service: an ``http(s)://`` endpoint that streams
raw PCM and states the format in ``X-Sample-Rate``, ``X-Audio-Format`` and
``X-Channels`` headers (Miso TTS does), or a ``wyoming://host:port`` server (the
Wyoming protocol Home Assistant uses; agent-composer's XTTS and Miso bridges
speak it). The text goes to that service and nowhere else.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import http.client
import json
import re
import secrets
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import AsyncIterator, Callable

import numpy as np

from argus_mask.models import AudioFormat, SourceHello
from argus_mask.performance import blocks
from argus_mask.speech import Edges, join_short, speakable, tidy_audio

LEAD_S = 1.0  # how far ahead of real time a source may run
TTS_TIMEOUT_S = 180.0  # the first byte waits for a cold model to load
SAY_LEAD_MS = 150  # silence before a TTS utterance: the output device is awake before the first sound
SAY_TAIL_MS = 300  # silence after: the mouth closes while the sound is still live, and nothing clips the last word
SAY_PAUSE_MS = 300  # silence between sentences; a blank line in the text gets twice this
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"'\u201d\u2019)\]]))\s+(?=\S)")
_PARAGRAPH = re.compile(r"\n\s*\n")


def sentences(text: str) -> list[tuple[str, int]]:
    """*text* as sentences, each with the number of pauses to leave after it: one between sentences, two
    between paragraphs, none after the last."""
    out: list[tuple[str, int]] = []
    paragraphs = [p.strip() for p in _PARAGRAPH.split(text) if p.strip()]
    for i, paragraph in enumerate(paragraphs):
        parts = [part.strip() for part in _SENTENCE_END.split(paragraph) if part.strip()]
        for j, part in enumerate(parts):
            last_in_paragraph = j == len(parts) - 1
            out.append((part, 0 if last_in_paragraph and i == len(paragraphs) - 1 else 2 if last_in_paragraph else 1))
    return out


_FORMATS = {"pcm_f32le": "f32le", "f32le": "f32le", "pcm_s16le": "s16le", "s16le": "s16le"}


class ClientError(RuntimeError):
    """The server or the TTS service refused, or went away."""


class TtsRefused(ClientError):
    """The TTS service answered a request with an HTTP error status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ServiceRefused(ClientError):
    """An HTTP TTS service answered with an error *status*."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def _session_url(base: str, face_id: str, end: str) -> str:
    scheme, sep, rest = base.rstrip("/").partition("://")
    if not sep:
        raise ClientError(f"{base!r} is not a URL")
    return f"{'wss' if scheme in ('https', 'wss') else 'ws'}://{rest}/session/{face_id}/{end}"


def source_url(base: str, face_id: str) -> str:
    return _session_url(base, face_id, "source")


def view_url(base: str, face_id: str) -> str:
    return _session_url(base, face_id, "view")


def _frame_bytes(fmt: AudioFormat) -> int:
    """Bytes in one sample of every channel."""
    return (4 if fmt.format == "f32le" else 2) * fmt.channels


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
    frame_bytes = _frame_bytes(fmt)
    try:
        async with websockets.connect(source_url(base, face_id), max_size=2**22, proxy=None) as ws:
            await ws.send(SourceHello(**fmt.model_dump(), seed=seed).model_dump_json())
            ready = await _reply(ws, "ready")
            if on_ready:
                on_ready(ready)
            # The clock starts at the first chunk, not at "ready": a source whose first chunk is slow
            # to come (a TTS model) must not then dump the rest at once to "catch up".
            started: float | None = None
            sent_s = 0.0
            async for chunk in chunks:
                if started is None:
                    started = time.monotonic()
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


class TtsReply:
    """An open TTS response: the HTTP response to read PCM from, and the socket under it.

    Kept together so a cancelled ``say`` can cut the socket from another thread:
    closing a response does not wake a thread blocked in a read, a shutdown does.
    """

    def __init__(
        self, connection: http.client.HTTPConnection, response: http.client.HTTPResponse, fmt: AudioFormat
    ) -> None:
        self.connection = connection
        self.response = response
        self.fmt = fmt

    def read(self, n: int = -1) -> bytes:
        return self.response.read() if n < 0 else self.response.read1(n)

    def cut(self) -> None:
        """Stop a read in progress, from any thread."""
        sock = self.connection.sock
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        self.close()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.response.close()
        with contextlib.suppress(OSError):
            self.connection.close()


class WyomingReply:
    """A Wyoming TTS server's answer to one ``synthesize``: PCM from its ``audio-chunk`` events.

    The protocol is one JSON line per event, each optionally followed by that many
    bytes of payload. ``audio-start`` carries the format, ``audio-chunk`` the
    samples, ``audio-stop`` the end. Same reading interface as :class:`TtsReply`.
    """

    def __init__(self, sock: socket.socket, fmt: AudioFormat, buffer: bytes) -> None:
        self.sock = sock
        self.fmt = fmt
        self._buffer = buffer  # bytes received past the last event parsed
        self._pcm = b""  # samples received and not yet read
        self._stopped = False

    @staticmethod
    def send(sock: socket.socket, kind: str, data: dict | None = None) -> None:
        head: dict = {"type": kind}
        if data is not None:
            head["data"] = data
        sock.sendall(json.dumps(head).encode() + b"\n")

    @staticmethod
    def receive(sock: socket.socket, buffer: bytes) -> tuple[tuple[str, dict, bytes] | None, bytes]:
        """The next ``(type, data, payload)`` event, or ``None`` at end of stream; and the bytes left over."""
        while b"\n" not in buffer:
            more = sock.recv(65536)
            if not more:
                return None, buffer
            buffer += more
        line, _, buffer = buffer.partition(b"\n")
        try:
            head = json.loads(line)
            data_length = int(head.get("data_length") or 0)
            payload_length = int(head.get("payload_length") or 0)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ClientError("the Wyoming server sent a line that is not an event") from exc
        while len(buffer) < data_length + payload_length:
            more = sock.recv(65536)
            if not more:
                raise ClientError("the Wyoming server closed mid-event")
            buffer += more
        data = json.loads(buffer[:data_length]) if data_length else head.get("data") or {}
        payload = buffer[data_length : data_length + payload_length]
        return (str(head.get("type")), data if isinstance(data, dict) else {}, payload), buffer[
            data_length + payload_length :
        ]

    def _fill(self) -> bool:
        """Take the next event into the sample buffer. False once the audio has stopped."""
        if self._stopped:
            return False
        event, self._buffer = self.receive(self.sock, self._buffer)
        if event is None:
            self._stopped = True
            return False
        kind, data, payload = event
        if kind == "audio-chunk":
            self._pcm += payload
        elif kind == "audio-stop":
            self._stopped = True
            return False
        elif kind == "error":
            raise ClientError(f"the Wyoming server failed: {data.get('text') or data.get('code') or data}")
        return True

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            while self._fill():
                pass
            out, self._pcm = self._pcm, b""
            return out
        while not self._pcm and self._fill():
            pass
        out, self._pcm = self._pcm[:n], self._pcm[n:]
        return out

    def cut(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.shutdown(socket.SHUT_RDWR)
        self.close()

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self.sock.close()


def open_wyoming(tts_url: str, body: dict) -> WyomingReply:
    """Ask a Wyoming TTS server for *body*'s speech; the reply once its ``audio-start`` has arrived."""
    parts = urllib.parse.urlsplit(tts_url)
    if not parts.hostname or not parts.port:
        raise ClientError(f"{tts_url!r}: a Wyoming URL is wyoming://host:port")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ClientError("nothing to say")
    # Wyoming's synthesize event: text, and a voice {name, language, speaker}. Any other -p field (a
    # temperature, say) goes beside the text in the event's data: a server that does not know it ignores
    # it, and a bridge to an engine that takes it can pass it on.
    data: dict = {"text": text}
    voice = {k: v for k, v in body.items() if k in ("name", "language", "speaker") or k == "voice"}
    if "voice" in voice:
        voice["name"] = str(voice.pop("voice"))
    if voice:
        data["voice"] = voice
    data |= {k: v for k, v in body.items() if k not in ("text", "voice", "name", "language", "speaker")}
    try:
        sock = socket.create_connection((parts.hostname, parts.port), timeout=TTS_TIMEOUT_S)
    except OSError as exc:
        raise ClientError(f"the Wyoming server at {tts_url} does not answer: {exc}") from exc
    try:
        WyomingReply.send(sock, "synthesize", data)
        buffer = b""
        while True:
            event, buffer = WyomingReply.receive(sock, buffer)
            if event is None:
                raise ClientError(f"the Wyoming server at {tts_url} closed without sending audio")
            kind, info, _ = event
            if kind == "error":
                raise ClientError(f"the Wyoming server failed: {info.get('text') or info.get('code') or info}")
            if kind == "audio-start":
                break
        width, channels, rate = int(info.get("width") or 0), int(info.get("channels") or 1), int(info.get("rate") or 0)
        if width != 2:
            raise ClientError(f"the Wyoming server sends {width * 8}-bit samples; only 16-bit is understood")
        try:
            fmt = AudioFormat(sample_rate=rate, channels=channels, format="s16le")
        except ValueError as exc:
            raise ClientError(f"the Wyoming server declared an audio format this cannot use: {info}") from exc
    except TimeoutError as exc:
        sock.close()
        raise ClientError(
            f"the Wyoming server at {tts_url} sent nothing for {TTS_TIMEOUT_S:g}s. A cold model takes a minute or two to load; "
            "longer than that, the service is stuck."
        ) from exc
    except (OSError, ClientError):
        sock.close()
        raise
    return WyomingReply(sock, fmt, buffer)


VOICES_TIMEOUT_S = 5.0


def http_base(tts_url: str) -> str | None:
    """The root of an HTTP service with the ``/synthesise`` contract, from its synthesis URL; None for anything else."""
    if not tts_url.startswith(("http://", "https://")):
        return None
    parts = urllib.parse.urlsplit(tts_url)
    base_path = re.sub(r"/synthesi[sz]e(/stream)?/?$", "", parts.path)
    if base_path == parts.path.rstrip("/"):
        return None
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, base_path, "", ""))


CLONE_TIMEOUT_S = 300.0  # a cold model loads before it clones


def _http_call(method: str, url: str, body: bytes | None, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(url, data=body, method=method, headers=headers)  # noqa: S310 - http(s) only
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310
            data = reply.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:500].decode("utf-8", "replace")
        try:
            parsed = json.loads(detail)
            detail = str(parsed.get("error") or parsed.get("detail") or detail) if isinstance(parsed, dict) else detail
        except ValueError:
            pass
        raise ServiceRefused(f"the TTS service answered {exc.code}: {detail}", exc.code) from exc
    except OSError as exc:
        raise ClientError(f"the TTS service at {url} does not answer: {exc}") from exc
    try:
        parsed = json.loads(data or b"{}")
    except ValueError as exc:
        raise ClientError("the TTS service sent something that is not JSON") from exc
    return parsed if isinstance(parsed, dict) else {}


def clone_voice(tts_url: str, voice_id: str, name: str, clips: list[tuple[str, bytes]], transcript: str) -> dict:
    """Store a voice on an HTTP TTS service with the ``/synthesise`` contract: ``POST …/clone-voice``.

    *clips* are (file name, WAV bytes), in order; *transcript* is what they say, in the same order.
    Miso standardises each clip (24 kHz mono, trimmed, loudness), joins them and keeps the first 20 s.
    """
    base = http_base(tts_url)
    if base is None:
        raise ClientError("making a voice needs an HTTP TTS service with the /synthesise contract (Miso)")
    # Random, so no field (a transcript, a name) can know it and close its part early to forge another.
    boundary = "argus-mask-" + secrets.token_hex(12)
    parts: list[bytes] = []

    def field(name_: str, value: str) -> None:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name_}"\r\n\r\n{value}\r\n'.encode())

    for filename, data in clips:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "clip.wav"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{safe}"\r\n'
            "Content-Type: audio/wav\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    field("reference_text", transcript)
    field("companion_id", voice_id)
    field("voice_name", name)
    parts.append(f"--{boundary}--\r\n".encode())
    return _http_call(
        "POST",
        base + "/clone-voice",
        b"".join(parts),
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        CLONE_TIMEOUT_S,
    )


def delete_voice(tts_url: str, voice_id: str) -> dict:
    """Remove a stored voice from an HTTP TTS service: ``DELETE …/clones/<id>``."""
    base = http_base(tts_url)
    if base is None:
        raise ClientError("an HTTP TTS service with the /synthesise contract is needed")
    return _http_call("DELETE", f"{base}/clones/{urllib.parse.quote(voice_id, safe='')}", None, {}, VOICES_TIMEOUT_S)


def _http_voices(tts_url: str) -> list[dict] | None:
    """Voices from an HTTP service with the ``/synthesise`` contract (Miso, agent-composer's TTS servers).

    Beside ``POST …/synthesise[/stream]`` they answer ``GET …/speakers`` (built-in speakers, by id) and
    ``GET …/clones`` (stored clones). Named as the Wyoming bridge names them, and as ``voiced`` maps
    them back: a speaker by its id, a clone as ``clone:<id>``. None if the service answers neither.
    """
    base = http_base(tts_url)
    if base is None:
        return None  # not the /synthesise contract: no known place to ask

    def get(path: str) -> dict | None:
        try:
            with urllib.request.urlopen(base + path, timeout=VOICES_TIMEOUT_S) as reply:  # noqa: S310 - http(s) only
                body = json.loads(reply.read())
                return body if isinstance(body, dict) else None
        except (OSError, ValueError):
            return None

    speakers, clones = get("/speakers"), get("/clones")
    if speakers is None and clones is None:
        return None
    voices = []
    for s in (speakers or {}).get("speakers") or []:
        if isinstance(s, dict) and s.get("id") is not None:
            voices.append(
                {
                    "name": str(s["id"]),
                    "description": str(s.get("name") or ""),
                    "languages": [str(x) for x in s.get("languages") or []],
                    "speakers": [],
                }
            )
    for c in (clones or {}).get("clones") or []:
        if isinstance(c, dict) and c.get("companion_id"):
            voices.append(
                {
                    "name": f"clone:{c['companion_id']}",
                    "description": str(c.get("name") or ""),
                    "languages": [],
                    "speakers": [],
                }
            )
    return voices


def tts_voices(tts_url: str) -> list[dict] | None:
    """The voices a TTS service offers, or None if it does not say.

    A Wyoming server answers ``describe`` with ``info``: each TTS program's voices, with their
    languages and, for multi-speaker models, their speakers. An HTTP service with the
    ``/synthesise`` contract lists its speakers and clones (:func:`_http_voices`). Flattened to
    ``[{"name", "description", "languages", "speakers": [names]}]``.
    """
    if tts_url.startswith(("http://", "https://")):
        return _http_voices(tts_url)
    if not tts_url.startswith("wyoming://"):
        return None
    try:
        parts = urllib.parse.urlsplit(tts_url)
        port = parts.port
    except ValueError as exc:
        raise ClientError(f"{tts_url!r}: a Wyoming URL is wyoming://host:port") from exc
    if not parts.hostname or not port:
        raise ClientError(f"{tts_url!r}: a Wyoming URL is wyoming://host:port")
    try:
        with socket.create_connection((parts.hostname, port), timeout=VOICES_TIMEOUT_S) as sock:
            WyomingReply.send(sock, "describe")
            buffer = b""
            while True:
                event, buffer = WyomingReply.receive(sock, buffer)
                if event is None:
                    raise ClientError(f"the Wyoming server at {tts_url} closed without describing itself")
                if event[0] == "info":
                    info = event[1]
                    break
    except OSError as exc:
        raise ClientError(f"the Wyoming server at {tts_url} does not answer: {exc}") from exc
    except ValueError as exc:  # an event's data that is not JSON
        raise ClientError(f"the Wyoming server at {tts_url} sent an event that is not JSON") from exc
    voices = []
    programs = info.get("tts")
    for program in programs if isinstance(programs, list) else []:
        for v in (program.get("voices") if isinstance(program, dict) else None) or []:
            if not isinstance(v, dict) or not v.get("name"):
                continue
            voices.append(
                {
                    "name": str(v["name"]),
                    "description": str(v.get("description") or ""),
                    "languages": [str(x) for x in v.get("languages") or []],
                    "speakers": [
                        str(s.get("name")) for s in v.get("speakers") or [] if isinstance(s, dict) and s.get("name")
                    ],
                }
            )
    return voices


def voiced(body: dict, voice: str, tts_url: str) -> dict:
    """*body* with *voice* as its voice, unless it already names one.

    A Wyoming server takes a voice name as it is. The HTTP contract has two fields: ``speaker``
    for one of the service's own speakers (Miso numbers them) and ``voice_clone_id`` for a
    stored clone. A number goes to the first, any other name to the second, with the
    ``clone:`` prefix agent-composer's bridges use dropped, so the same name works on both.
    """
    if not voice:
        return body
    wyoming = tts_url.startswith("wyoming://")
    # A Wyoming voice is {name, speaker}: a speaker picks one of the named voice's speakers, so only a
    # name in the request stands instead of --voice. Over HTTP a speaker is itself the voice.
    named = ("voice", "name") if wyoming else ("voice", "name", "speaker", "voice_clone_id")
    if any(k in body for k in named):
        return body
    body = dict(body)
    if wyoming:
        body["voice"] = voice
    elif voice.isdigit():
        body["speaker"] = voice
    else:
        body["voice_clone_id"] = voice.removeprefix("clone:")
    return body


def open_tts(tts_url: str, body: dict) -> TtsReply | WyomingReply:
    """Ask the TTS service at *tts_url* for *body*'s speech; the reply, with the audio format it declared."""
    if tts_url.startswith("wyoming://"):
        return open_wyoming(tts_url, body)
    if not tts_url.startswith(("http://", "https://")):
        raise ClientError(
            "ARGUS_MASK_TTS_URL is not set"
            if not tts_url
            else f"ARGUS_MASK_TTS_URL {tts_url!r} is not http(s):// or wyoming://"
        )
    parts = urllib.parse.urlsplit(tts_url)
    kind = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = kind(parts.hostname or "", parts.port, timeout=TTS_TIMEOUT_S)
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    try:
        # One stream per connection. Without "Connection: close" a keep-alive server waits for the next
        # request on the same socket while the client waits for more body, and both sit there.
        connection.request(
            "POST",
            path,
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Connection": "close"},
        )
        response = connection.getresponse()
    except TimeoutError as exc:
        connection.close()
        raise ClientError(
            f"the TTS service at {tts_url} sent nothing for {TTS_TIMEOUT_S:g}s. A cold model takes a minute or two to load; "
            "longer than that, the service is stuck."
        ) from exc
    except (OSError, http.client.HTTPException) as exc:
        connection.close()
        raise ClientError(f"the TTS service at {tts_url} does not answer: {exc}") from exc
    if response.status != 200:
        detail = response.read(300).decode(errors="replace")
        connection.close()
        raise TtsRefused(response.status, f"the TTS service answered {response.status}: {detail}")
    declared = (response.getheader("X-Audio-Format") or "").lower()
    try:
        fmt = AudioFormat(
            sample_rate=int(response.getheader("X-Sample-Rate") or 0),
            channels=int(response.getheader("X-Channels") or 1),
            format=_FORMATS[declared],  # type: ignore[arg-type]
        )
    except (KeyError, ValueError) as exc:
        connection.close()
        raise ClientError(
            "the TTS service did not state a PCM format this understands "
            f"(X-Sample-Rate={response.getheader('X-Sample-Rate')!r}, X-Audio-Format={declared!r})"
        ) from exc
    return TtsReply(connection, response, fmt)


async def say(
    base: str,
    face_id: str,
    tts_url: str,
    body: dict,
    *,
    seed: int = 0,
    stream: bool = False,
    pause_ms: int = SAY_PAUSE_MS,
    tidy: bool = True,
    context: bool = False,
    on_ready: Callable[[dict], None] | None = None,
    on_audio: Callable[[float], None] | None = None,
    on_wait: Callable[[], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Ask the TTS service for *body*'s speech and play it into a face.

    The text is sent one sentence at a time, with *pause_ms* of silence between sentences
    (twice that between paragraphs): TTS models run sentences into each other, and a short
    request wanders less than a long one. *pause_ms* 0 sends the text whole.

    *context* sends each sentence with the one before it, text and audio as the service returned it, for a
    CSM-style model (Miso) to carry on from; see `_Carry`. Off by default: with Miso and a cloned voice,
    chaining generated audio as context made pitch drift up and loudness sink sentence after sentence.

    *tidy* (the default) is what `argus_mask.speech` does: numbers and times go as words, a
    sentence of a few words goes with the next, and each sentence comes back with soft
    edges and, unless streamed, at the same speech level as the others. Off, the text goes
    as written and the audio as it came.

    By default the whole utterance is fetched first and then pushed in real time: a
    TTS model slower than real time (Miso TTS 8B runs at about a quarter of real
    time on an RTX 3090) would otherwise stall the face mid-sentence. *stream*
    forwards the audio as it is generated, for a service that is faster than real
    time. *on_wait* is called once the first request is out and the wait for audio
    begins; *on_progress* before each sentence with its index and the count;
    *on_audio* is told the utterance's length in seconds once it is known.

    Cancelling this (Ctrl-C in the CLI) cuts the TTS socket, so the thread reading
    it ends at once instead of holding the process open until the service gives up.
    """
    if not stream:
        # Fetch everything before the face is claimed: the service answers its headers at once and
        # then generates for as long as it takes, and the face stays free for others meanwhile.
        whole, fmt, seconds = await synthesise(
            tts_url, body, pause_ms=pause_ms, tidy=tidy, context=context, on_wait=on_wait, on_progress=on_progress
        )
        if on_audio:
            on_audio(seconds)
        block = fmt.sample_rate // 10 * _frame_bytes(fmt)  # ~100 ms
        lead, tail = _silence(fmt, SAY_LEAD_MS), _silence(fmt, SAY_TAIL_MS)

        async def buffered() -> AsyncIterator[bytes]:
            yield lead
            for start in range(0, len(whole), block):
                yield whole[start : start + block]
            yield tail

        return await send(base, face_id, fmt, buffered(), seed=seed, on_ready=on_ready)

    parts = _parts(body, pause_ms, tidy)
    if on_wait:
        on_wait()
    carry = _Carry(tts_url, body, context)
    reply = await _open_part(tts_url, body, parts, 0, None, on_progress, carry)
    fmt = reply.fmt
    block = fmt.sample_rate // 10 * _frame_bytes(fmt)  # ~100 ms

    async def streamed() -> AsyncIterator[bytes]:
        nonlocal reply
        # TTS models start speaking on their first sample. A little silence either side gives the output
        # device time to wake before the first sound, and the mouth time to close before the stream ends.
        yield _silence(fmt, SAY_LEAD_MS)
        for i, (text, pauses) in enumerate(parts):
            if i:
                reply = await _open_part(tts_url, body, parts, i, fmt, on_progress, carry)
            edges = Edges(fmt) if tidy else None
            heard: list[bytes] = []
            try:
                while data := await _cancellable(asyncio.to_thread(reply.read, block), cut=reply.cut):
                    if carry.on:
                        heard.append(data)
                    if edges is None:
                        yield data
                    elif out := edges.feed(data):
                        yield out
            except OSError as exc:
                raise ClientError(f"the TTS stream broke: {exc}") from exc
            finally:
                reply.close()
            if edges is not None and (out := edges.end()):
                yield out
            carry.keep(text, b"".join(heard), fmt)
            if pauses:
                yield _silence(fmt, pause_ms * pauses)
        yield _silence(fmt, SAY_TAIL_MS)

    return await send(base, face_id, fmt, streamed(), seed=seed, on_ready=on_ready)


def _silence(fmt: AudioFormat, ms: int) -> bytes:
    return bytes(fmt.sample_rate * ms // 1000 * _frame_bytes(fmt))


def _parts(body: dict, pause_ms: int, tidy: bool) -> list[tuple[str, int]]:
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ClientError("nothing to say")
    if tidy:
        text = speakable(text)
    parts = sentences(text) if pause_ms > 0 else [(text, 0)]
    return join_short(parts) if tidy else parts


MAX_BUFFERED_S = 600.0  # speech fetched whole before playing (not --stream): 10 minutes, then a clear refusal
CONTEXT_MAX_S = 15.0  # the longest previous sentence sent as context; Miso takes up to 20 s


class _Carry:
    """The previous sentence, sent with the next as context, so the voice keeps its tone across sentences.

    Miso's ``/synthesise`` contract (voluptas#4) takes ``context: [{text, audio}]``, the audio being
    base64 ``pcm_f32le`` at the service's rate: a CSM-style model then carries on from how the last
    sentence sounded instead of starting afresh from the clone's reference. The audio sent is levelled
    (`speech.tidy_audio`) first: sent as it came, each sentence copied the last one's level a little
    lower, and a briefing sank by 15 dB from its first sentence to its last. Sent only to
    an HTTP service, only for mono f32 audio, only if the body does not already have a ``context``.
    A service that refuses the field (400 or 422) is asked again without it, and not sent it again;
    one that ignores it sounds as before. Stateless: nothing is kept between utterances.
    """

    def __init__(self, tts_url: str, body: dict, enabled: bool) -> None:
        self.on = enabled and tts_url.startswith(("http://", "https://")) and "context" not in body
        self.previous: tuple[str, bytes] | None = None

    def body(self, body: dict, text: str, fmt: AudioFormat | None) -> dict:
        out = {**body, "text": text}
        if self.on and self.previous is not None and fmt is not None and fmt.format == "f32le" and fmt.channels == 1:
            said, audio = self.previous
            if 0 < len(audio) / 4 / fmt.sample_rate <= CONTEXT_MAX_S:
                out["context"] = [{"text": said, "audio": base64.b64encode(audio).decode("ascii")}]
        return out

    def keep(self, text: str, audio: bytes, fmt: AudioFormat) -> None:
        if self.on and audio:
            self.previous = (text, tidy_audio(audio, fmt))


async def _open_part(
    tts_url: str,
    body: dict,
    parts: list[tuple[str, int]],
    i: int,
    fmt: AudioFormat | None,
    on_progress: Callable[[int, int], None] | None,
    carry: _Carry | None = None,
) -> TtsReply | WyomingReply:
    if on_progress and len(parts) > 1:
        on_progress(i + 1, len(parts))
    request = carry.body(body, parts[i][0], fmt) if carry is not None else {**body, "text": parts[i][0]}
    try:
        reply = await _cancellable(asyncio.to_thread(open_tts, tts_url, request), cut=None)
    except TtsRefused as exc:
        if "context" not in request or carry is None or exc.status not in (400, 422):
            raise
        carry.on = False  # this service does not take context: ask again without it, and stop sending it
        reply = await _cancellable(asyncio.to_thread(open_tts, tts_url, {**body, "text": parts[i][0]}), cut=None)
    if fmt is not None and reply.fmt != fmt:
        reply.close()
        raise ClientError(f"the TTS service changed its audio format mid-utterance: {reply.fmt} after {fmt}")
    return reply


async def synthesise(
    tts_url: str,
    body: dict,
    *,
    pause_ms: int = SAY_PAUSE_MS,
    tidy: bool = True,
    context: bool = False,
    on_wait: Callable[[], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[bytes, AudioFormat, float]:
    """*body*'s speech, whole, as `say` would play it: (PCM in the service's format, that format, seconds of speech).

    Sentence by sentence with pauses between, tidied unless *tidy* is off, each sentence sent with the one
    before it as context unless *context* is off (see `say`). No face is involved:
    `say` plays this into one, the editor's Voices page plays it in the browser.
    """
    parts = _parts(body, pause_ms, tidy)
    if on_wait:
        on_wait()
    pieces: list[bytes] = []
    spoken = 0
    fmt: AudioFormat | None = None
    limit = 0  # bytes of speech allowed in all, once the format is known
    carry = _Carry(tts_url, body, context)
    for i, (text, pauses) in enumerate(parts):
        reply = await _open_part(tts_url, body, parts, i, fmt, on_progress, carry)
        fmt = reply.fmt
        limit = limit or int(MAX_BUFFERED_S * fmt.sample_rate) * _frame_bytes(fmt)
        try:
            whole = await _cancellable(asyncio.to_thread(_read_capped, reply, limit - spoken), cut=reply.cut)
        except OSError as exc:
            raise ClientError(f"the TTS stream broke: {exc}") from exc
        finally:
            reply.close()
        if not whole:
            raise ClientError("the TTS service sent no audio" + (f" for sentence {i + 1}" if len(parts) > 1 else ""))
        spoken += len(whole)
        carry.keep(text, whole, fmt)
        pieces.append(tidy_audio(whole, fmt) if tidy else whole)
        if pauses:
            pieces.append(_silence(fmt, pause_ms * pauses))
    assert fmt is not None  # parts is never empty
    return b"".join(pieces), fmt, spoken / _frame_bytes(fmt) / fmt.sample_rate


def _read_capped(reply: TtsReply | WyomingReply, allowed: int) -> bytes:
    """All of *reply*, refusing once it passes *allowed* bytes: a service that never stops must not fill memory."""
    chunks: list[bytes] = []
    got = 0
    while data := reply.read(1 << 20):
        got += len(data)
        if got > allowed:
            raise ClientError(
                f"the TTS service sent more than {MAX_BUFFERED_S / 60:g} minutes of speech: send less text at a time, "
                "or use --stream"
            )
        chunks.append(data)
    return b"".join(chunks)


async def _cancellable(awaitable, *, cut: Callable[[], None] | None):  # noqa: ANN001, ANN202
    """Await a thread's result; if cancelled meanwhile, *cut* what the thread is blocked on, then let the cancel through."""
    task = asyncio.ensure_future(awaitable)
    try:
        return await task
    except asyncio.CancelledError:
        if cut is not None:
            cut()
        task.cancel()
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(task, 2)
        raise
