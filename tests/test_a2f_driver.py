"""The a2f driver's own edges: its health probe and a stream that fails while it opens."""

from __future__ import annotations

import asyncio

import pytest
import websockets
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from argus_mask.drivers import DriverError
from argus_mask.drivers.a2f import A2FDriver
from tests.helpers import fake_a2f, live_server


def test_the_health_probe_does_not_follow_a_redirect() -> None:
    with fake_a2f() as healthy:
        app = FastAPI()

        @app.get("/health")
        def health() -> RedirectResponse:
            return RedirectResponse(f"{healthy}/health", status_code=307)

        with live_server(app) as url:
            why = asyncio.run(A2FDriver(url).check())
    assert why is not None and "does not answer" in why


class FakeSocket:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.closed = False

    async def send(self, _message: object) -> None:
        raise self.error

    async def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize(
    ("error", "raised"),
    [
        (ConnectionResetError("reset by peer"), DriverError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
def test_a_handshake_that_fails_closes_its_socket(
    monkeypatch: pytest.MonkeyPatch, error: BaseException, raised: type[BaseException]
) -> None:
    socket = FakeSocket(error)

    async def connect(*_args: object, **_kwargs: object) -> FakeSocket:
        return socket

    monkeypatch.setattr(websockets, "connect", connect)
    with pytest.raises(raised):
        asyncio.run(A2FDriver("http://127.0.0.1:9").open(30))
    assert socket.closed
