from __future__ import annotations

import socket
import threading
import time

from argus_mask import client


def test_cut_wakes_a_read_on_a_reply_that_closes_the_connection() -> None:
    """A ``Connection: close`` reply leaves ``connection.sock`` None; cut must still reach the socket."""
    listener = socket.create_server(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    stop = threading.Event()

    def serve() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nConnection: close\r\nX-Sample-Rate: 24000\r\n"
                b"X-Audio-Format: pcm_f32le\r\nX-Channels: 1\r\n\r\n"
            )
            stop.wait(10)  # a model still generating: headers, then nothing

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        reply = client.open_tts(f"http://127.0.0.1:{port}/tts", {"text": "hello"})
        assert isinstance(reply, client.TtsReply)
        assert reply.connection.sock is None  # the response owns the socket
        done = threading.Event()

        def read() -> None:
            try:
                reply.read(4096)
            except (OSError, ValueError, AttributeError):
                pass  # woken by the cut: an empty read or the closed response, either way no longer blocked
            finally:
                done.set()

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        time.sleep(0.2)
        started = time.monotonic()
        reply.cut()
        assert done.wait(3), "cut did not wake the blocked read"
        assert time.monotonic() - started < 3
    finally:
        stop.set()
        server.join(5)
        listener.close()
