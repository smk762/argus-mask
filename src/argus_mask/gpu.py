"""GPU leases for work mask asks of others (#125): it runs none itself.

Mask's own process uses no GPU. Two services it drives do, and argus-iris (the suite's GPU broker) is told about
them through cortex's lease client, under the kinds in ``gpu-tenant.yaml``:

- ``a2f-live``: the Audio2Face server while a source is live on a face. ``realtime``, never preempted. Held in
  the background (:func:`hold`): asked for when the stream starts, released when it ends, and never waited on, so
  a slow or absent broker can never delay a mouth. Only the broker is asked: when it can't be reached there is no
  fallback, because the local lease is a host-wide lock and a broadcast would hold it for hours.
- ``voice-make``: a TTS service (Miso) making a voice, a candidate or a clone. ``interactive``; waited for, with
  ``ARGUS_GPU_BROKER_FALLBACK`` when the broker can't be reached, like any other job.

Nothing here runs unless ``ARGUS_GPU_COORDINATOR`` says so (``none`` by default). The env is cortex's
(``ARGUS_GPU_BROKER_URL``, ``_TOKEN``, ``ARGUS_GPU_TENANT_MANIFEST``...): see ``.env.example``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import Any

import structlog

log = structlog.get_logger()

A2F_LIVE = "a2f-live"
VOICE_MAKE = "voice-make"
# What each kind asks for: gpu-tenant.yaml declares the same (tests/test_gpu_tenant.py checks they agree).
A2F_VRAM_MB = 500  # the Audio2Face worker, measured on a 3090: its model stays loaded
VOICE_VRAM_MB = 2500  # Miso with its model loaded (2.3 GB on a 3090), and room to synthesise
LEASES: dict[str, dict[str, Any]] = {
    A2F_LIVE: {
        "kind": A2F_LIVE,
        "target": "a2f",
        "min_vram_mb": A2F_VRAM_MB,
        "model_mb": A2F_VRAM_MB,
        "priority": "realtime",
        "preemptible": False,
    },
    VOICE_MAKE: {
        "kind": VOICE_MAKE,
        "target": "miso",
        "min_vram_mb": VOICE_VRAM_MB,
        "model_mb": 2300,
        "priority": "interactive",
        "preemptible": False,
    },
}
RELEASE_WAIT_S = 5.0  # a granted lease's release, at the end of a stream; past this it is cancelled


def coordinator() -> Any:  # noqa: ANN401 - a cortex GpuCoordinator; cortex.gpu needs no extra to import
    """The coordinator the environment names, as every suite service reads it."""
    from argus_cortex.gpu import coordinator_from_env  # noqa: PLC0415

    return coordinator_from_env()


def live_coordinator() -> Any:  # noqa: ANN401
    """The broker, with no fallback, when ``ARGUS_GPU_COORDINATOR=broker``; otherwise one that does nothing."""
    from argus_cortex.gpu import NullCoordinator, build_coordinator  # noqa: PLC0415

    if os.environ.get("ARGUS_GPU_COORDINATOR", "").strip().lower() != "broker":
        return NullCoordinator()
    env = os.environ.get
    return build_coordinator(
        "broker",
        broker_url=env("ARGUS_GPU_BROKER_URL", "").strip() or None,
        broker_token=env("ARGUS_GPU_BROKER_TOKEN", "").strip() or None,
        tenant=env("ARGUS_GPU_TENANT", "").strip() or None,
        manifest_path=env("ARGUS_GPU_TENANT_MANIFEST", "").strip() or None,
        fallback="none",
    )


@contextlib.asynccontextmanager
async def hold(gpu: Any, kind: str, caller: str) -> AsyncIterator[None]:  # noqa: ANN401
    """Ask *gpu* for a lease of *kind* while the block runs, without waiting for it.

    The block starts at once. If the lease is granted it is released when the block ends (waiting up to
    ``RELEASE_WAIT_S``); if it is still queued then, the request is given up. A broker error is logged, never
    raised: the lease tells the broker what is running, and the work does not depend on it.
    """
    done = asyncio.Event()
    granted = asyncio.Event()

    async def run() -> None:
        try:
            async with gpu.alease(caller=caller, **LEASES[kind]):
                granted.set()
                await done.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - BrokerError, a refusal, an unreachable broker with no fallback
            log.warning("gpu_lease_failed", kind=kind, caller=caller, error=str(exc))

    task = asyncio.create_task(run())
    try:
        yield
    finally:
        done.set()
        if not granted.is_set():
            task.cancel()
        try:
            await asyncio.wait_for(task, RELEASE_WAIT_S)
        except (asyncio.CancelledError, TimeoutError):
            if not task.done():
                task.cancel()
            if asyncio.current_task() is not None and asyncio.current_task().cancelling():  # type: ignore[union-attr]
                raise


@contextlib.asynccontextmanager
async def lease(gpu: Any, kind: str, caller: str) -> AsyncIterator[None]:  # noqa: ANN401
    """Wait for a lease of *kind* and hold it while the block runs (``GpuLeaseTimeout``/``Refused`` if not had)."""
    async with gpu.alease(caller=caller, **LEASES[kind]):
        yield
