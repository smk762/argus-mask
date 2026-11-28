"""Turn checks and measured limits (#170)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings, TurnsSettings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedError, GuidedStore, StepResult
from argus_mask.editor.turns import TurnsJobs, analyze_synthetic
from argus_mask.turns import DEFAULT_GAP, measure_limits, synthetic_pose


def _report(weak: bool):
    def capture(yaw: float, pitch: float) -> bytes:
        return synthetic_pose(320, 400, yaw=yaw, pitch=pitch, weak=weak)

    return measure_limits(capture, threshold=DEFAULT_GAP)


def test_weak_head_limits_near_splat01() -> None:
    """splat-01 gives out around 12° yaw and 7° back — within a few degrees."""
    report = _report(weak=True)
    assert report.limits.yaw is not None and 8 <= report.limits.yaw <= 14
    assert report.limits.pitch_up is not None and 5 <= report.limits.pitch_up <= 8
    assert any(s.gap >= DEFAULT_GAP and abs(s.yaw) >= 12 for s in report.scores)


def test_strong_head_holds_like_syn_w101() -> None:
    """syn-w101 holds about 20° yaw and 8° back."""
    report = _report(weak=False)
    assert report.limits.yaw is not None and report.limits.yaw >= 18
    assert report.limits.pitch_up is not None and report.limits.pitch_up >= 8


def _seed_build(store: GuidedStore, candidate: str = "c1") -> None:
    store.create("guide")
    portrait = store.keep("guide", "portrait", f"candidates/{candidate}/portrait.png", b"p")
    store.put("guide", StepResult(step="source", candidate=candidate, outputs=[portrait]))
    store.put(
        "guide",
        StepResult(
            step="portrait",
            candidate=candidate,
            checks=[Check(name="framing", verdict="passed")],
        ),
    )
    head = store.keep("guide", "head", f"candidates/{candidate}/head.zip", b"fake-zip")
    store.put(
        "guide",
        StepResult(
            step="build",
            candidate=candidate,
            checks=[Check(name="pipeline", verdict="passed")],
            outputs=[head],
        ),
    )


@pytest.mark.asyncio
async def test_turns_job_writes_limits(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed_build(store)
    jobs = TurnsJobs(
        store,
        TurnsSettings(script=None),
        analyze=analyze_synthetic(weak=True),
    )
    assert jobs.check() is None
    job = jobs.start("guide", "c1")
    while job.live:
        await asyncio.sleep(0.02)
    assert job.state == "done"
    assert job.limits is not None
    assert 8 <= float(job.limits["yaw"]) <= 14
    result = store.load("guide").get("turns", "c1")
    assert result is not None and result.standing in ("passed", "advisory")
    assert "limits" in result.data


def test_editor_turns_route(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed_build(store)
    jobs = TurnsJobs(store, TurnsSettings(script=None), analyze=analyze_synthetic(weak=False))
    app = create_editor_app(
        Settings(assets_dir=assets, driver="energy"),
        EditorSettings(),
        turns=jobs,
    )
    with TestClient(app) as client:
        assert client.get("/guided").json()["turns"]["ready"] is True
        started = client.post("/guided/guide/candidates/c1/turns")
        assert started.status_code == 200
        job = started.json()
        for _ in range(100):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.02)
        assert job["state"] == "done"
        assert float(job["limits"]["yaw"]) >= 18


def test_a_head_with_no_limit_one_way_still_gets_its_advisory(tmp_path: Path) -> None:
    from argus_mask.models import Limits
    from argus_mask.turns import PoseScore, TurnReport

    store = GuidedStore(tmp_path / "assets")
    _seed_build(store)
    report = TurnReport(
        scores=(PoseScore(label="down", yaw=0, pitch=12, gap=0.9, region="crown"),),
        limits=Limits(pitch_down=8.0),
        threshold=DEFAULT_GAP,
    )

    async def run():  # noqa: ANN202
        jobs = TurnsJobs(store, TurnsSettings(script=None), analyze=lambda *_: report)
        job = jobs.start("guide", "c1")
        await asyncio.wait({job.task}, timeout=10)
        return job

    job = asyncio.run(run())
    assert (job.state, job.error) == ("done", None)
    (check,) = store.load("guide").get("turns", "c1").checks
    assert check.verdict == "advisory" and "yaw=none pitch_up=none" in check.reason


@pytest.mark.asyncio
async def test_asking_for_turns_hashes_nothing_on_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start() runs on the loop: the head is hashed by the job, in a thread, where a changed one still fails it."""
    import threading

    from argus_mask.editor import turns

    store = GuidedStore(tmp_path / "assets")
    _seed_build(store)
    jobs = TurnsJobs(store, TurnsSettings(script=None), analyze=analyze_synthetic(weak=True))
    loop, on_loop = threading.get_ident(), []
    sha256_file = turns.sha256_file
    monkeypatch.setattr(
        turns, "sha256_file", lambda path: on_loop.append(threading.get_ident() == loop) or sha256_file(path)
    )
    store.path("guide", "candidates/c1/head.zip").write_bytes(b"another head")
    job = jobs.start("guide", "c1")
    while job.live:
        await asyncio.sleep(0.02)
    assert on_loop and not any(on_loop)
    assert job.state == "failed" and "head changed" in job.error
    store.path("guide", "candidates/c1/head.zip").unlink()
    with pytest.raises(GuidedError, match="not where its build kept it"):
        jobs.start("guide", "c1")
