"""Fixtures built in code: a tiny ``.glb``, WAV files, and fake Audio2Face and TTS services.

No real person's face or voice is in the repo. The model is one triangle with
named morph targets; the audio is a tone.
"""

from __future__ import annotations

import asyncio
import json
import socket
import struct
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from argus_mask.arkit import ARKIT_52
from argus_mask.faces import MANIFEST_NAME, build_manifest
from argus_mask.models import Provenance

STOCK = Provenance(kind="stock", source="generated in tests/helpers.py", license="CC0-1.0")


def make_glb(targets: tuple[str, ...] | list[str] = ARKIT_52, *, extra_static_mesh: bool = False) -> bytes:
    """A valid glTF 2.0 binary: one triangle, one morph target per name."""
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    blobs = [base.tobytes()]
    for i in range(len(targets)):
        delta = np.zeros((3, 3), dtype="<f4")
        delta[i % 3, 2] = 0.1
        blobs.append(delta.tobytes())
    views = [{"buffer": 0, "byteOffset": 36 * i, "byteLength": 36} for i in range(len(blobs))]
    accessors = [
        {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0]}
    ]
    for i in range(len(targets)):
        accessors.append(
            {
                "bufferView": i + 1,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [0.1, 0.1, 0.1],
            }
        )
    primitive: dict = {"attributes": {"POSITION": 0}, "mode": 4}
    mesh: dict = {"name": "head", "primitives": [primitive]}
    if targets:
        primitive["targets"] = [{"POSITION": i + 1} for i in range(len(targets))]
        mesh["extras"] = {"targetNames": list(targets)}
        mesh["weights"] = [0.0] * len(targets)
    meshes = [mesh]
    nodes = [{"mesh": 0, "name": "head"}]
    if extra_static_mesh:
        meshes.append({"name": "hair", "primitives": [{"attributes": {"POSITION": 0}, "mode": 4}]})
        nodes.append({"mesh": 1, "name": "hair"})
    doc = {
        "asset": {"version": "2.0", "generator": "argus-mask tests"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "buffers": [{"byteLength": 36 * len(blobs)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary = b"".join(blobs)
    total = 12 + 8 + len(text) + 8 + len(binary)
    return b"".join(
        [
            struct.pack("<III", 0x46546C67, 2, total),
            struct.pack("<II", len(text), 0x4E4F534A),
            text,
            struct.pack("<II", len(binary), 0x004E4942),
            binary,
        ]
    )


def make_face(
    root: Path,
    face_id: str = "test-head",
    *,
    targets: tuple[str, ...] | list[str] = ARKIT_52,
    provenance: Provenance | None = STOCK,
) -> Path:
    """A face directory under *root*. ``provenance=None`` writes a manifest without one."""
    directory = root / face_id
    directory.mkdir(parents=True)
    model = directory / "head.glb"
    model.write_bytes(make_glb(targets))
    manifest = build_manifest(model, face_id=face_id, name="Test head", provenance=provenance or STOCK).model_dump(
        mode="json"
    )
    if provenance is None:
        del manifest["provenance"]
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def tone(seconds: float = 1.0, rate: int = 24000, *, level: float = 0.3) -> np.ndarray:
    """A voiced-sounding test signal: a low tone, its amplitude wobbling like syllables."""
    t = np.arange(int(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def write_wav(path: Path, mono: np.ndarray, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def fake_a2f(
    *, names: list[str] | None = None, fail_after: int | None = None, lag_frames: int = 0, stall: bool = False
) -> Iterator[str]:
    """An Audio2Face server speaking the ``/drive`` protocol. jawOpen follows the frame number.

    ``names`` is the blendshape order it announces (default: ARKit's, reversed, so a
    client that ignores the order is caught). ``fail_after`` makes it send an error
    after that many frames. ``lag_frames`` holds its answers that many frames behind
    the audio, as the real model does; ``stall`` makes it answer nothing until the end.
    """
    import websockets
    from websockets.http11 import Response

    order = names if names is not None else list(reversed(ARKIT_52))
    port = free_port()
    ready = threading.Event()
    stop: dict = {}

    def process_request(connection, request):  # noqa: ANN001, ANN202
        if request.path == "/health":
            body = json.dumps({"ready": True, "model": "fake-a2f"}).encode()
            return Response(
                200,
                "OK",
                websockets.Headers({"Content-Type": "application/json", "Content-Length": str(len(body))}),
                body,
            )
        return None

    async def handler(ws) -> None:  # noqa: ANN001
        hello = json.loads(await ws.recv())
        rate, rate_fps = hello["sample_rate"], hello["fps"]
        await ws.send(json.dumps({"blendshapes": order}))
        have = 0
        sent = 0

        async def emit(upto: int) -> bool:
            nonlocal sent
            while sent < upto:
                if fail_after is not None and sent >= fail_after:
                    await ws.send(json.dumps({"error": "the model fell over"}))
                    return False
                weights = [0.0] * len(order)
                if "jawOpen" in order:
                    weights[order.index("jawOpen")] = round((sent % 10) / 10, 3)
                await ws.send(json.dumps({"i": sent, "weights": weights}))
                sent += 1
            return True

        async for message in ws:
            if isinstance(message, bytes):
                have += len(message) // 4
                if not stall and not await emit(max(0, have * rate_fps // rate - lag_frames)):
                    return
            elif json.loads(message).get("end"):
                if await emit(-(-have * rate_fps // rate)):
                    await ws.send(json.dumps({"end": True}))
                return

    async def main() -> None:
        stop["event"] = asyncio.Event()
        stop["loop"] = asyncio.get_running_loop()
        async with websockets.serve(handler, "127.0.0.1", port, process_request=process_request):
            ready.set()
            await stop["event"].wait()

    thread = threading.Thread(target=lambda: asyncio.run(main()), daemon=True)
    thread.start()
    assert ready.wait(5), "fake a2f did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        stop["loop"].call_soon_threadsafe(stop["event"].set)
        thread.join(5)


@contextmanager
def fake_tts(
    pcm: np.ndarray, rate: int = 24000, *, status: int = 200, break_after: int | None = None
) -> Iterator[tuple[str, list[dict]]]:
    """A TTS service that streams *pcm* as chunked ``pcm_f32le`` with the format in headers.

    Yields its URL and the list of request bodies it received. ``break_after``
    drops the connection after that many bytes.
    """
    seen: list[dict] = []
    payload = pcm.astype("<f4").tobytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            seen.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"))
            if status != 200:
                body = b'{"error": "text is required"}'
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(rate))
            self.send_header("X-Audio-Format", "pcm_f32le")
            self.send_header("X-Channels", "1")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sent = 0
            step = rate // 4 * 4  # 250 ms
            for start in range(0, len(payload), step):
                chunk = payload[start : start + step]
                if break_after is not None and sent >= break_after:
                    self.connection.close()
                    return
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                sent += len(chunk)
            self.wfile.write(b"0\r\n\r\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/synthesise/stream", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@contextmanager
def live_server(app) -> Iterator[str]:  # noqa: ANN001
    """The app on a real port, for clients that open real sockets."""
    import uvicorn

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "the server did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(10)
