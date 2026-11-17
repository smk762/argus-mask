"""a2f/check.py, the live check of an Audio2Face server, against small fake servers. Loaded by path."""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from websockets.asyncio.server import Server, serve

from tests.helpers import free_port

_SPEC = importlib.util.spec_from_file_location("a2f_check", Path(__file__).parent.parent / "a2f" / "check.py")
assert _SPEC is not None and _SPEC.loader is not None
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)

PCM = b"\0" * (check.RATE * 4 // 2)  # half a second of silence, f32le


@contextmanager
def http(status: int, body: bytes) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_get_reports_a_server_that_is_down_as_status_0() -> None:
    status, body = check.get(f"http://127.0.0.1:{free_port()}/health")
    assert status == 0
    assert body["error"]


def test_get_reports_a_body_that_is_not_json_as_empty() -> None:
    with http(200, b"<html>not json</html>") as base:
        assert check.get(base + "/health") == (200, {})


def test_get_reads_a_json_body() -> None:
    with http(200, b'{"ready": true}') as base:
        assert check.get(base + "/health") == (200, {"ready": True})


async def _drive(handler) -> tuple[Server, str]:  # noqa: ANN001
    port = free_port()
    server = await serve(handler, "127.0.0.1", port)
    return server, f"ws://127.0.0.1:{port}/drive"


async def test_a_stream_closed_without_its_end_marker_fails() -> None:
    async def handler(ws) -> None:  # noqa: ANN001
        await ws.recv()
        await ws.send(json.dumps({"blendshapes": ["jawOpen"]}))
        async for message in ws:
            if isinstance(message, str) and json.loads(message).get("end"):
                await ws.send(json.dumps({"i": 0, "weights": [0.5]}))
                return  # closes cleanly, no {"end": true}

    server, url = await _drive(handler)
    async with server:
        with pytest.raises(RuntimeError, match="end marker"):
            await check.stream(url, PCM)


async def test_a_mid_stream_error_is_the_one_raised_when_the_server_closes() -> None:
    async def handler(ws) -> None:  # noqa: ANN001
        await ws.recv()
        await ws.send(json.dumps({"blendshapes": ["jawOpen"]}))
        await ws.recv()  # the first chunk of audio
        await ws.send(json.dumps({"error": "the model fell over"}))
        # and the socket closes, while the client is still sending

    server, url = await _drive(handler)
    async with server:
        with pytest.raises(RuntimeError, match="the model fell over"):
            await check.stream(url, PCM, realtime=True)


async def test_a_whole_stream_returns_its_frames() -> None:
    async def handler(ws) -> None:  # noqa: ANN001
        await ws.recv()
        await ws.send(json.dumps({"blendshapes": ["jawOpen"]}))
        async for message in ws:
            if isinstance(message, str) and json.loads(message).get("end"):
                await ws.send(json.dumps({"i": 0, "weights": [0.5]}))
                await ws.send(json.dumps({"end": True}))
                return

    server, url = await _drive(handler)
    async with server:
        result = await check.stream(url, PCM)
    assert result["names"] == ["jawOpen"]
    assert result["frames"] == [[0.5]]
