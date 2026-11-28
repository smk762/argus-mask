"""The TTS and session clients against malformed URLs and misbehaving servers: a ClientError, and no socket left open."""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest

from argus_mask import client
from argus_mask.client import ClientError


@contextmanager
def one_shot_server(serve: Callable[[socket.socket], None]) -> Iterator[tuple[int, threading.Event]]:
    """A listener that hands its first connection to *serve*; the event is set once *serve* returns."""
    listener = socket.create_server(("127.0.0.1", 0))
    done = threading.Event()

    def run() -> None:
        try:
            conn, _ = listener.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            with contextlib.suppress(OSError):
                serve(conn)
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield listener.getsockname()[1], done
    finally:
        listener.close()
        thread.join(5)


def wyoming_event(kind: str, data: dict | None = None) -> bytes:
    return json.dumps({"type": kind, **({"data": data} if data is not None else {})}).encode() + b"\n"


@pytest.mark.parametrize("base", ["HTTPS://host:8109", "Wss://host:8109", "wss://host:8109"])
def test_a_tls_base_in_any_case_opens_a_tls_socket(base: str) -> None:
    assert client.source_url(base, "f") == "wss://host:8109/session/f/source"


def test_an_unknown_scheme_is_refused_not_sent_plaintext() -> None:
    assert client.view_url("HTTP://host", "f") == "ws://host/session/f/view"
    with pytest.raises(ClientError):
        client.source_url("ftp://host", "f")


@pytest.mark.parametrize("url", ["http://host/", "http://host/foo/", "http://host/foo"])
def test_a_url_without_synthesise_has_no_http_base(url: str) -> None:
    assert client.http_base(url) is None


def test_a_synthesise_url_keeps_its_base() -> None:
    assert client.http_base("http://host/synthesise/") == "http://host"
    assert client.http_base("http://host/api/synthesize/stream") == "http://host/api"
    assert client.http_base("http://host/api/voice/synthesise/pcm") == "http://host/api/voice"  # rag-chat


def _received(raw: bytes) -> object:
    a, b = socket.socketpair()
    with a, b:
        a.sendall(raw)
        a.shutdown(socket.SHUT_WR)
        return client.WyomingReply.receive(b, b"")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"type": "info", "data_length": 5}\n{oops',  # data that is not JSON
        b'{"type": "info", "data_length": 2}\n\xff\xfe',  # nor UTF-8
        b'{"type": "audio-chunk", "data_length": -3}\n{}xyz',  # a negative length
        b'{"type": "audio-chunk", "payload_length": 1000000000000}\n',  # a length past any sane event
    ],
)
def test_a_malformed_wyoming_event_is_a_client_error(raw: bytes) -> None:
    with pytest.raises(ClientError):
        _received(raw)


@pytest.mark.parametrize(
    "start", [{"rate": "22k", "width": 2, "channels": 1}, {"rate": 22050, "width": [2], "channels": 1}]
)
def test_a_malformed_audio_start_is_refused_and_the_socket_closed(start: dict) -> None:
    closed = threading.Event()

    def serve(conn: socket.socket) -> None:
        conn.recv(65536)
        conn.sendall(wyoming_event("audio-start", start))
        if conn.recv(1) == b"":
            closed.set()

    with one_shot_server(serve) as (port, done):
        with pytest.raises(ClientError):
            client.open_wyoming(f"wyoming://127.0.0.1:{port}", {"text": "hello"})
        assert done.wait(5) and closed.is_set(), "the client left the socket open"


@pytest.mark.parametrize("url", ["wyoming://host:abc", "wyoming://host:99999"])
def test_a_bad_wyoming_port_is_a_client_error(url: str) -> None:
    with pytest.raises(ClientError):
        client.open_wyoming(url, {"text": "hello"})
    with pytest.raises(ClientError):
        client.tts_voices(url)


@pytest.mark.parametrize("url", ["http://host:abc/synthesise", "http://host:99999/synthesise"])
def test_a_bad_http_port_is_a_client_error(url: str) -> None:
    with pytest.raises(ClientError):
        client.open_tts(url, {"text": "hello"})


def test_describe_gives_up_on_a_server_that_never_describes_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "VOICES_TIMEOUT_S", 0.5)

    def serve(conn: socket.socket) -> None:
        conn.recv(65536)
        for _ in range(100):  # chatter, each event well inside the per-read timeout
            conn.sendall(wyoming_event("ping"))
            time.sleep(0.05)

    with one_shot_server(serve) as (port, _):
        started = time.monotonic()
        with pytest.raises(ClientError):
            client.tts_voices(f"wyoming://127.0.0.1:{port}")
        assert time.monotonic() - started < 2


def test_an_info_that_is_not_an_object_lists_no_voices() -> None:
    def serve(conn: socket.socket) -> None:
        conn.recv(65536)
        conn.sendall(b'{"type": "info", "data": [1, 2]}\n')

    with one_shot_server(serve) as (port, _):
        assert client.tts_voices(f"wyoming://127.0.0.1:{port}") == []


def test_an_error_status_whose_body_breaks_is_still_a_refusal() -> None:
    def serve(conn: socket.socket) -> None:
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 100\r\n\r\nshort")

    with one_shot_server(serve) as (port, _), pytest.raises(client.ServiceRefused) as refused:
        client.open_tts(f"http://127.0.0.1:{port}/synthesise", {"text": "hello"})
    assert refused.value.status == 500


def test_a_reply_in_a_format_it_does_not_take_is_refused_and_the_socket_closed() -> None:
    closed = threading.Event()

    def serve(conn: socket.socket) -> None:
        request = b""
        while not request.endswith(b"}"):  # the headers and the JSON body, which may come apart
            request += conn.recv(65536)
        conn.sendall(
            b"HTTP/1.1 200 OK\r\nConnection: close\r\nX-Sample-Rate: 24000\r\nX-Audio-Format: mp3\r\n"
            b"Content-Length: 100\r\n\r\nab"
        )
        try:
            if conn.recv(1) == b"":
                closed.set()
        except ConnectionResetError:  # closed with the body unread
            closed.set()

    with one_shot_server(serve) as (port, done):
        with pytest.raises(ClientError) as refused:  # the traceback, held here, must not hold the socket open
            client.open_tts(f"http://127.0.0.1:{port}/synthesise", {"text": "hello"})
        assert done.wait(5) and closed.is_set(), "the client left the socket open"
    assert "X-Audio-Format='mp3'" in str(refused.value)


def test_a_deadline_does_not_outlive_its_read() -> None:
    a, b = socket.socketpair()
    with a, b:
        b.settimeout(None)
        a.sendall(wyoming_event("info", {}))
        event, _ = client.WyomingReply.receive(b, b"", time.monotonic() + 5)
        assert event is not None and event[0] == "info"
        assert b.gettimeout() is None, "a later read without a deadline would time out"


def test_a_reply_opened_after_its_utterance_was_cancelled_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    release, closed = threading.Event(), threading.Event()

    class Reply:
        fmt = None

        def close(self) -> None:
            closed.set()

    def slow_open(tts_url: str, body: dict) -> Reply:
        release.wait(5)  # still connecting when the utterance is cancelled
        return Reply()

    monkeypatch.setattr(client, "open_tts", slow_open)

    async def main() -> None:
        carry = client._Carry("http://tts/synthesise", {"text": "Hello."}, False)
        task = asyncio.create_task(
            client._open_part("http://tts/synthesise", {"text": "Hello."}, [("Hello.", 0)], 0, None, None, carry)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()  # the open completes after the cancel

    asyncio.run(main())
    assert closed.wait(5), "the reply opened after the cancel was never closed"


def test_a_garbled_http_voices_reply_is_no_answer_not_a_crash() -> None:
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(5)

    def serve() -> None:
        for _ in range(2):  # /speakers, then /clones
            with contextlib.suppress(OSError):
                conn, _ = listener.accept()
                with conn:
                    conn.recv(65536)
                    conn.sendall(b"not http at all\r\n\r\n")

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        assert client._http_voices(f"http://127.0.0.1:{listener.getsockname()[1]}/synthesise") is None
    finally:
        listener.close()
        thread.join(5)


@contextmanager
def _tts_that_loads(busy_answers: int) -> Iterator[tuple[str, list[str]]]:
    """An HTTP TTS whose ``/speakers`` answers 503 *busy_answers* times before it lists one speaker."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    asked: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            asked.append(self.path)
            loading = self.path == "/speakers" and asked.count("/speakers") <= busy_answers
            body = {"error": "Model still loading"} if loading else {"speakers": [{"id": "Ana", "name": "Ana"}]}
            if self.path == "/clones":
                body = {"clones": []}
            raw = json.dumps(body).encode()
            self.send_response(503 if loading else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/synthesise", asked
    finally:
        server.shutdown()
        server.server_close()


def test_a_tts_still_loading_its_model_is_asked_again_not_listed_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "VOICES_LOADING_POLL_S", 0.01)
    with _tts_that_loads(busy_answers=2) as (url, asked):
        voices = client._http_voices(url)
    assert [v["name"] for v in voices or []] == ["speaker:Ana"]
    assert asked.count("/speakers") == 3


def test_a_tts_that_never_finishes_loading_says_so_instead_of_listing_no_voices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client, "VOICES_LOADING_POLL_S", 0.01)
    monkeypatch.setattr(client, "VOICES_LOADING_WAIT_S", 0.05)
    with _tts_that_loads(busy_answers=10**6) as (url, _), pytest.raises(ClientError, match="still loading"):
        client._http_voices(url)
