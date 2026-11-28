"""a2f/server.py's GET /health body: argus-cortex's ServiceHealth, with the fields its readers already use.

The a2f image has no argus-cortex, so the server builds the body by hand; this checks it with cortex's own model,
so the two cannot drift. Loaded by path: a2f/ is not a package. Importing it needs only the standard library and
websockets (no CUDA, no worker).
"""

from __future__ import annotations

import asyncio
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pytest
from argus_cortex.peers import PEER_NAME_PATTERN
from argus_cortex.service import SERVICE_VERSION, ServiceHealth
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from argus_mask.drivers.a2f import A2FDriver
from tests.helpers import live_server

_SPEC = importlib.util.spec_from_file_location("a2f_server", Path(__file__).parent.parent / "a2f" / "server.py")
assert _SPEC is not None and _SPEC.loader is not None
server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(server)

STARTED = datetime(2026, 11, 28, 10, 0, 0, 123456, tzinfo=UTC)


def body(**overrides: object) -> dict:
    fields: dict = {
        "ready": True,
        "reason": "starting",
        "started_at": STARTED,
        "model": "Audio2Face-3D-v2.3.1-James",
        "fps": 30,
        "gpu_solver": False,
        "busy": False,
        "streams": 3,
    }
    return server.health_body(**{**fields, **overrides})


def test_the_body_is_a_service_health_whether_ready_or_not() -> None:
    ready = ServiceHealth.model_validate(body(version="0.9.0", build="abc1234", peer="a2f"))
    assert (ready.service, ready.version, ready.status, ready.peer, ready.build) == (
        "argus-mask",
        "0.9.0",
        "ok",
        "a2f",
        "abc1234",
    )
    assert ready.started_at == STARTED.replace(microsecond=0)
    starting = ServiceHealth.model_validate(body(ready=False, reason="building the TensorRT engine"))
    assert starting.status == "starting"
    assert not starting.ready  # cortex's own reading agrees with the legacy key
    assert (starting.version, starting.peer, starting.build) == (server.UNKNOWN_VERSION, None, None)


def test_it_speaks_cortexs_health_contract_version() -> None:
    assert server.SERVICE_VERSION == SERVICE_VERSION
    assert server.PEER_NAME.pattern == PEER_NAME_PATTERN


def test_the_legacy_fields_stay_for_the_driver_and_check() -> None:
    """mask's a2f driver reads ready, reason and model; a2f/check.py reads ready and busy."""
    ready = body(busy=True)
    assert ready["ready"] is True and "reason" not in ready
    assert (ready["model"], ready["fps"], ready["gpu_solver"], ready["busy"], ready["streams"]) == (
        "Audio2Face-3D-v2.3.1-James",
        30,
        False,
        True,
        3,
    )
    starting = body(ready=False, reason="downloading network.onnx")
    assert starting["ready"] is False and starting["reason"] == "downloading network.onnx"
    # Through the model too: the extras survive validation and a dump, as a reader that re-serialises sees them.
    dumped = ServiceHealth.model_validate(starting).model_dump(mode="json")
    assert {k: dumped[k] for k in ("ready", "reason", "model", "busy")} == {
        "ready": False,
        "reason": "downloading network.onnx",
        "model": "Audio2Face-3D-v2.3.1-James",
        "busy": False,
    }


@pytest.mark.parametrize(
    ("ready", "said"),
    [(True, None), (False, "the Audio2Face server at {url} is not ready: building the TensorRT engine")],
)
def test_the_driver_reads_the_new_body_as_before(ready: bool, said: str | None) -> None:
    """mask's driver probe, against a server answering exactly what a2f/server.py now answers."""
    app = FastAPI()

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(body(ready=ready, reason="building the TensorRT engine", peer="a2f"))

    with live_server(app) as url:
        driver = A2FDriver(url)
        why = asyncio.run(driver.check())
    assert why == (said.format(url=url) if said else None)
    assert driver.model == "Audio2Face-3D-v2.3.1-James"


@pytest.mark.parametrize("peer", ["A2F", "a2f_", "-a2f", "a" * 101])
def test_a_bad_peer_name_stops_the_server_at_startup(monkeypatch: pytest.MonkeyPatch, peer: str) -> None:
    monkeypatch.setenv("ARGUS_PEER", peer)
    with pytest.raises(SystemExit, match="ARGUS_PEER"):
        server.Config()


def test_config_reads_the_identity_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARGUS_PEER", "a2f")
    monkeypatch.setenv("ARGUS_BUILD", "abc1234")
    monkeypatch.setenv("A2F_VERSION", "0.9.0")
    config = server.Config()
    assert (config.peer, config.build, config.version) == ("a2f", "abc1234", "0.9.0")
    for name in ("ARGUS_PEER", "ARGUS_BUILD", "A2F_VERSION"):
        monkeypatch.delenv(name)
    config = server.Config()
    assert (config.peer, config.build, config.version) == (None, None, server.UNKNOWN_VERSION)
