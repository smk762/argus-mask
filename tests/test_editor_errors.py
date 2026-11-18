"""The editor's answers when what it talks to fails: the mask server, the TTS service, a recording."""

from __future__ import annotations

import asyncio
import socket
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus_mask import client
from argus_mask.editor import broadcast
from argus_mask.editor.app import server_get
from argus_mask.editor.broadcast import Broadcasts, RecordOptions
from tests.test_voice import speechlike
from tests.test_voices_page import check, make_body, voices_page  # noqa: F401 - the fixture


@contextmanager
def answering(reply: bytes):
    """A server that reads a request and answers *reply*, raw, then hangs up."""
    sock = socket.create_server(("127.0.0.1", 0))

    def serve() -> None:
        while True:
            try:
                conn, _ = sock.accept()
            except OSError:
                return
            with conn:
                conn.recv(65536)
                conn.sendall(reply)

    threading.Thread(target=serve, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        sock.close()


@pytest.mark.parametrize(
    "reply",
    [
        b"NONSENSE\r\n\r\n",  # BadStatusLine
        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{}",  # IncompleteRead
        b"HTTP/1.1 200 OK\r\nX: " + b"a" * 70_000 + b"\r\n\r\n",  # LineTooLong
    ],
)
def test_server_get_never_raises_on_a_broken_reply(reply: bytes) -> None:
    with answering(reply) as base:
        status, body, why = server_get(base, "/faces")
    assert status is None and body is None and why and base in why


def test_making_a_voice_when_the_tts_cannot_list_its_voices_is_a_502(voices_page, monkeypatch) -> None:  # noqa: F811
    def refused(tts: str) -> None:
        raise client.ClientError(f"the TTS service at {tts} does not answer")

    monkeypatch.setattr(client, "tts_voices", refused)
    with voices_page("http://127.0.0.1:1/synthesise") as c:
        sha = check(c, speechlike())["sha"]
        reply = c.post("/voice/create", json=make_body(sha))
    assert reply.status_code == 502 and "does not answer" in reply.json()["detail"], reply.text


def test_two_recordings_started_in_the_same_second_keep_both_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(broadcast.time, "strftime", lambda fmt, *a: "20260101-000000")

    async def record(base, face, out: Path, *, on_wait, **kw):  # noqa: ANN001, ANN003, ANN202
        on_wait()
        await asyncio.sleep(0)
        out.write_bytes(out.name.encode())
        return SimpleNamespace(header=SimpleNamespace(session_id="s"))

    monkeypatch.setattr(broadcast.recorder, "record", record)

    async def run(job):  # noqa: ANN001, ANN202
        await asyncio.sleep(0.05)  # the recorder is listening by now
        job.ready({"session_id": "s"})
        return {"frames": 1, "duration_ms": 10}

    async def both() -> list[broadcast.Job]:
        b = Broadcasts("http://127.0.0.1:1", tmp_path)
        jobs = []
        for _ in range(2):
            job = b._start("stock-01", "push", "x", b._recorded(run, "stock-01", RecordOptions()))
            await job.task
            jobs.append(job)
        return jobs

    first, second = asyncio.run(both())
    assert first.state == second.state == "done", (first.events, second.events)
    [a], [b] = first.files, second.files
    assert a != b
    assert (tmp_path / a).read_bytes() == a.encode() and (tmp_path / b).read_bytes() == b.encode()
