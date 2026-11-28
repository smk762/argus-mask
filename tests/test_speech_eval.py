"""Speech eval: lip-gap metrics and the guided step (#171)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings, SpeechSettings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedError, GuidedStore, StepResult
from argus_mask.editor.speech import SpeechJobs, analyze_synthetic
from argus_mask.speech_eval import (
    REF_CLOSED,
    REF_REVERSALS,
    metrics_from_gaps,
    reference_baseline,
    score,
    synthetic_gaps,
)


def test_stutter_series_matches_161_before() -> None:
    """Lip-parting under speech: low closed share, many reversals (#161 before)."""
    m = metrics_from_gaps(synthetic_gaps(stutter=True))
    assert m.closed_share < 0.50
    assert m.reversals > 45
    report = score(m, reference_baseline())
    assert report.stutter is True
    assert report.oversized is False


def test_clean_series_matches_161_after() -> None:
    """Without lip-parting under speech: closed ~62%, reversals near splat-01 (#161 after)."""
    m = metrics_from_gaps(synthetic_gaps(stutter=False))
    assert m.closed_share >= 0.55
    assert m.reversals <= 40
    report = score(m, reference_baseline())
    assert report.stutter is False
    assert abs(REF_CLOSED - 0.62) < 1e-9
    assert REF_REVERSALS == 38


def test_oversized_opening_vs_reference() -> None:
    gaps = [0.01] * 40 + [0.25] * 20  # mostly closed but huge openings
    m = metrics_from_gaps(gaps)
    report = score(m, reference_baseline())
    assert report.oversized is True


def _seed(store: GuidedStore, candidate: str = "c1") -> None:
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
    store.put(
        "guide",
        StepResult(
            step="rest",
            candidate=candidate,
            checks=[Check(name="rest", verdict="passed"), Check(name="smile", verdict="passed")],
            data={"name": "fit-c1", "weights": {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.2}, "life": 1.0},
        ),
    )


@pytest.mark.asyncio
async def test_speech_job_writes_scores(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed(store)
    jobs = SpeechJobs(
        store,
        SpeechSettings(script=None, take=None, landmarker=None),
        analyze=analyze_synthetic(stutter=False),
    )
    assert jobs.check() is None
    job = jobs.start("guide", "c1")
    while job.live:
        await asyncio.sleep(0.02)
    assert job.state == "done"
    assert job.scores is not None and float(job.scores["closed_share"]) >= 0.55
    result = store.load("guide").get("speech", "c1")
    assert result is not None and result.standing == "passed"
    assert result.data["stutter"] is False


@pytest.mark.asyncio
async def test_speech_job_flags_stutter(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed(store)
    jobs = SpeechJobs(
        store,
        SpeechSettings(script=None),
        analyze=analyze_synthetic(stutter=True),
    )
    job = jobs.start("guide", "c1")
    while job.live:
        await asyncio.sleep(0.02)
    assert job.state == "done"
    result = store.load("guide").get("speech", "c1")
    assert result is not None and result.standing == "advisory"
    assert result.data["stutter"] is True
    assert any(c.name == "stutter" and c.verdict == "advisory" for c in result.checks)


def test_speech_needs_rest(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    store.create("guide")
    portrait = store.keep("guide", "portrait", "candidates/c1/portrait.png", b"p")
    store.put("guide", StepResult(step="source", candidate="c1", outputs=[portrait]))
    store.put(
        "guide",
        StepResult(step="portrait", candidate="c1", checks=[Check(name="framing", verdict="passed")]),
    )
    head = store.keep("guide", "head", "candidates/c1/head.zip", b"z")
    store.put(
        "guide",
        StepResult(step="build", candidate="c1", checks=[Check(name="pipeline", verdict="passed")], outputs=[head]),
    )
    jobs = SpeechJobs(store, SpeechSettings(script=None), analyze=analyze_synthetic(stutter=False))
    from argus_mask.editor.guided import NotReady

    with pytest.raises(NotReady, match="rest"):
        jobs.start("guide", "c1")


def test_editor_speech_route(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed(store)
    jobs = SpeechJobs(store, SpeechSettings(script=None), analyze=analyze_synthetic(stutter=False))
    app = create_editor_app(
        Settings(assets_dir=assets, driver="energy"),
        EditorSettings(),
        speech=jobs,
    )
    with TestClient(app) as client:
        assert client.get("/guided").json()["speech"]["ready"] is True
        started = client.post("/guided/guide/candidates/c1/speech")
        assert started.status_code == 200
        job = started.json()
        for _ in range(100):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.02)
        assert job["state"] == "done"
        assert float(job["scores"]["closed_share"]) >= 0.55


@pytest.mark.asyncio
async def test_asking_for_speech_hashes_nothing_on_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start() runs on the loop: the head is hashed by the job, in a thread, where a changed one still fails it."""
    import threading

    from argus_mask.editor import speech

    store = GuidedStore(tmp_path / "assets")
    _seed(store)
    jobs = SpeechJobs(
        store, SpeechSettings(script=None, take=None, landmarker=None), analyze=analyze_synthetic(stutter=False)
    )
    loop, on_loop = threading.get_ident(), []
    sha256_file = speech.sha256_file
    monkeypatch.setattr(
        speech, "sha256_file", lambda path: on_loop.append(threading.get_ident() == loop) or sha256_file(path)
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
