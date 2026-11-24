"""Age read of a guided head against its portrait (#172)."""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient

from argus_mask.age_eval import OLDER_YEARS, score
from argus_mask.config import AgeSettings, EditorSettings, Settings
from argus_mask.editor.age import AgeJobs, read_age_synthetic
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.curator import AgeReading, CuratorClient, CuratorError
from argus_mask.editor.guided import Check, GuidedStore, StepResult


def _png(colour: tuple[int, int, int] = (200, 180, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
    return buf.getvalue()


def test_character_head_passes() -> None:
    """#151 character heads did not read older."""
    report = score(28.0, 29.0)
    assert report.older is False
    assert report.delta_years == 1.0


def test_photo_head_flags_ten_years() -> None:
    """#151 photograph heads read about ten years older."""
    report = score(30.0, 40.0, threshold=OLDER_YEARS)
    assert report.older is True
    assert report.delta_years == 10.0


def test_curator_age_picks_largest_face() -> None:
    body = {
        "source": "mivolo",
        "faces": [
            {"age_estimate_years": 50.0, "bbox": [0, 0, 10, 10]},
            {"age_estimate_years": 34.2, "bbox": [0, 0, 100, 100]},
        ],
    }

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return json.dumps(body).encode()

    client = CuratorClient("http://curator.test", opener=lambda *a, **k: Reply())
    reading = client.age(_png())
    assert reading == AgeReading(years=34.2, source="mivolo", faces=2)


def test_curator_age_no_face() -> None:
    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def read(self):
            return b'{"source":"mivolo","faces":[]}'

    client = CuratorClient("http://curator.test", opener=lambda *a, **k: Reply())
    with pytest.raises(CuratorError, match="no face"):
        client.age(_png())


def _seed(store: GuidedStore, candidate: str = "c1") -> None:
    store.create("guide")
    portrait = store.keep("guide", "portrait", f"candidates/{candidate}/portrait.png", _png())
    store.put("guide", StepResult(step="source", candidate=candidate, outputs=[portrait]))
    store.put(
        "guide",
        StepResult(step="portrait", candidate=candidate, checks=[Check(name="framing", verdict="passed")]),
    )
    head = store.keep("guide", "head", f"candidates/{candidate}/head.zip", b"fake-zip")
    crop = store.keep("guide", "portrait-crop", f"candidates/{candidate}/likeness/portrait.png", _png((210, 180, 160)))
    render = store.keep("guide", "render", f"candidates/{candidate}/likeness/render.png", _png((190, 170, 150)))
    store.put(
        "guide",
        StepResult(
            step="build",
            candidate=candidate,
            checks=[Check(name="pipeline", verdict="passed")],
            outputs=[head, crop, render],
        ),
    )


@pytest.mark.asyncio
async def test_age_job_flags_photo_head(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed(store)
    jobs = AgeJobs(
        store,
        AgeSettings(curator_url=""),
        read_age=read_age_synthetic(30.0, 40.0),
    )
    assert jobs.check() is None
    job = jobs.start("guide", "c1")
    while job.live:
        await asyncio.sleep(0.02)
    assert job.state == "done"
    result = store.load("guide").get("age", "c1")
    assert result is not None and result.standing == "advisory"
    assert result.data["older"] is True
    assert float(result.scores["delta_years"]) == 10.0


@pytest.mark.asyncio
async def test_age_job_passes_character_head(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed(store)
    jobs = AgeJobs(store, AgeSettings(), read_age=read_age_synthetic(28.0, 29.5))
    job = jobs.start("guide", "c1")
    while job.live:
        await asyncio.sleep(0.02)
    assert job.state == "done"
    result = store.load("guide").get("age", "c1")
    assert result is not None and result.standing == "passed"
    assert result.data["older"] is False


def test_editor_age_route(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    _seed(store)
    jobs = AgeJobs(store, AgeSettings(), read_age=read_age_synthetic(28.0, 28.5))
    app = create_editor_app(
        Settings(assets_dir=assets, driver="energy"),
        EditorSettings(),
        age=jobs,
    )
    with TestClient(app) as client:
        assert client.get("/guided").json()["age"]["ready"] is True
        started = client.post("/guided/guide/candidates/c1/age")
        assert started.status_code == 200
        job = started.json()
        for _ in range(100):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.02)
        assert job["state"] == "done"
        assert float(job["scores"]["delta_years"]) == 0.5


@pytest.mark.asyncio
async def test_asking_for_an_age_reads_no_picture(tmp_path: Path) -> None:
    """The request runs on the event loop: a picture that changed is found by the job, off it."""
    store = GuidedStore(tmp_path / "assets")
    _seed(store)
    store.path("guide", "candidates/c1/likeness/render.png").write_bytes(_png((1, 2, 3)))
    jobs = AgeJobs(store, AgeSettings(), read_age=read_age_synthetic(30.0, 30.0))
    job = jobs.start("guide", "c1")
    await asyncio.wait({job.task}, timeout=10)
    assert job.state == "failed" and "changed" in job.error
    assert store.load("guide").get("age", "c1") is None


@pytest.mark.asyncio
async def test_the_age_is_of_the_bytes_that_were_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = GuidedStore(tmp_path / "assets")
    _seed(store)
    seen: dict[str, bytes] = {}

    def read(label: str, data: bytes) -> AgeReading:
        seen[label] = data
        return AgeReading(years=30.0, source="mivolo", faces=1)

    jobs = AgeJobs(store, AgeSettings(), read_age=read)
    checked = jobs._pictures

    def then_rewritten(*args: object) -> object:
        found = checked(*args)
        store.path("guide", "candidates/c1/likeness/render.png").write_bytes(_png((1, 2, 3)))
        return found

    job = jobs.start("guide", "c1")
    monkeypatch.setattr(jobs, "_pictures", then_rewritten)
    await asyncio.wait({job.task}, timeout=10)
    assert job.state == "done"
    assert seen == {"portrait": _png((210, 180, 160)), "head": _png((190, 170, 150))}
