"""The argus-iris broker wiring (#125): the committed tenant manifest, and the leases mask takes for others' GPUs."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

import pytest
from argus_cortex.gpu import GpuLeaseRefused, NullCoordinator
from argus_cortex.gpu_wire import load_tenant_manifest
from starlette.testclient import TestClient

from argus_mask import gpu
from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.server.app import create_app
from tests.helpers import fake_a2f, fake_tts, make_face, tone, write_wav

MANIFEST = Path(__file__).resolve().parent.parent / "gpu-tenant.yaml"
HELLO = json.dumps({"sample_rate": 16000, "format": "f32le"})


class Recording:
    """A coordinator that records each lease: its options when asked, and when it is released."""

    name = "rec"

    def __init__(self, *, block: bool = False, refuse: bool = False) -> None:
        self.events: list[tuple[str, dict]] = []
        self.block, self.refuse = block, refuse

    @contextlib.asynccontextmanager
    async def alease(self, *, caller: str, min_vram_mb: int, **options):
        self.events.append(("ask", {"caller": caller, "min_vram_mb": min_vram_mb, **options}))
        if self.refuse:
            raise GpuLeaseRefused("no room")
        try:
            if self.block:
                await asyncio.Event().wait()  # queued for ever
            self.events.append(("granted", {}))
            yield
        finally:
            self.events.append(("released" if ("granted", {}) in self.events else "gave up", {}))

    @property
    def kinds(self) -> list[str]:
        return [o["kind"] for e, o in self.events if e == "ask"]


def test_the_manifest_validates_and_agrees_with_the_leases_mask_takes() -> None:
    manifest = load_tenant_manifest(MANIFEST)
    assert manifest.tenant == "argus-mask"
    for kind, options in gpu.LEASES.items():
        declared = manifest.kind_for(kind)
        assert declared is not None, kind
        assert (declared.target, declared.vram_mb, declared.model_mb) == (
            options["target"],
            options["min_vram_mb"],
            options["model_mb"],
        )
        assert (declared.priority, declared.preemptible) == (options["priority"], options["preemptible"])
    assert manifest.kind_for(gpu.A2F_LIVE).priority == "realtime"
    assert [r.name for r in manifest.residents if r.pinned] == ["a2f"]


async def test_a_held_lease_never_delays_the_work_and_is_released_after_it() -> None:
    rec = Recording()
    async with gpu.hold(rec, gpu.A2F_LIVE, "t"):
        await asyncio.sleep(0)
        assert [e for e, _ in rec.events] == ["ask", "granted"]
    assert [e for e, _ in rec.events] == ["ask", "granted", "released"]

    queued = Recording(block=True)
    started = time.monotonic()
    async with gpu.hold(queued, gpu.A2F_LIVE, "t"):
        await asyncio.sleep(0.01)  # the work runs while the lease is still queued
    assert time.monotonic() - started < 1.0
    assert [e for e, _ in queued.events] == ["ask", "gave up"]  # the request is given up, not left queued

    refused = Recording(refuse=True)
    async with gpu.hold(refused, gpu.A2F_LIVE, "t"):  # a broker error is logged, never raised
        await asyncio.sleep(0.01)


async def test_cancelling_the_work_still_cancels_it() -> None:
    async def work() -> None:
        async with gpu.hold(Recording(), gpu.A2F_LIVE, "t"):
            await asyncio.sleep(10)

    task = asyncio.create_task(work())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_the_live_lease_goes_only_to_the_broker_with_no_fallback(monkeypatch) -> None:
    monkeypatch.delenv("ARGUS_GPU_COORDINATOR", raising=False)
    assert isinstance(gpu.live_coordinator(), NullCoordinator)
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "lease")  # the host-wide lock: never for a broadcast
    assert isinstance(gpu.live_coordinator(), NullCoordinator)
    monkeypatch.setenv("ARGUS_GPU_COORDINATOR", "broker")
    monkeypatch.setenv("ARGUS_GPU_BROKER_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("ARGUS_GPU_TENANT_MANIFEST", str(MANIFEST))
    monkeypatch.setenv("ARGUS_GPU_BROKER_FALLBACK", "lease")
    live = gpu.live_coordinator()
    assert live.name == "broker" and isinstance(live.fallback, NullCoordinator)
    assert live.tenant == "argus-mask"
    assert gpu.coordinator().fallback.name == "lease"  # voice making falls back as any job does


def test_a_live_a2f_stream_holds_a_realtime_lease_for_its_length(tmp_path) -> None:
    make_face(tmp_path)
    rec = Recording()
    with fake_a2f() as url:
        app = create_app(Settings(assets_dir=tmp_path, driver="a2f", a2f_url=url), gpu=rec)
        with TestClient(app) as client, client.websocket_connect("/session/test-head/source") as source:
            source.send_text(HELLO)
            assert source.receive_json()["type"] == "ready"
            source.send_bytes(tone(0.5, 16000).astype("<f4").tobytes())
            assert [e for e, _ in rec.events] == ["ask", "granted"]  # held while the stream runs
            source.send_text(json.dumps({"type": "end"}))
            assert source.receive_json()["type"] == "done"
    asked = rec.events[0][1]
    assert (asked["kind"], asked["priority"], asked["preemptible"]) == ("a2f-live", "realtime", False)
    assert asked["caller"] == "argus-mask:test-head"
    assert rec.events[-1][0] == "released"


def test_the_energy_driver_takes_no_lease(tmp_path) -> None:
    make_face(tmp_path)
    rec = Recording()
    wav = write_wav(tmp_path / "a.wav", tone(0.3))
    with TestClient(create_app(Settings(assets_dir=tmp_path), gpu=rec)) as client:
        assert client.post("/push?face=test-head", content=wav.read_bytes()).status_code == 200
    assert rec.events == []


def test_making_a_voice_waits_for_a_voice_make_lease(tmp_path) -> None:
    from tests.test_voices_page import speechlike  # noqa: PLC0415

    script = "Good morning. I keep an eye on the systems, and I read out what matters."
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    for coordinator, state in ((Recording(), "done"), (Recording(refuse=True), "failed")):
        with (
            fake_tts(speechlike(3.0, 24000), 24000) as (tts, seen),
            TestClient(
                create_editor_app(Settings(assets_dir=tmp_path / "assets", tts_url=tts), ed, gpu=coordinator)
            ) as c,
        ):
            job = c.post("/voice/generate", json={"script": script}).json()
            end = time.monotonic() + 10
            while job["state"] == "running" and time.monotonic() < end:
                time.sleep(0.05)
                job = c.get(f"/voice/generate/{job['id']}").json()
            assert job["state"] == state, job
            assert coordinator.kinds == ["voice-make"]
            if state == "failed":
                assert "no room" in job["error"] and not seen  # the service was never asked
