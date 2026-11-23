"""Guided source step (#167): generate, pick, upload, restyle — each way in and each refusal."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.testclient import TestClient

from argus_mask.backgrounds import Darkroom
from argus_mask.config import EditorSettings, Settings, SourceSettings
from argus_mask.consent import ConsentStore
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import GuidedStore
from argus_mask.editor.source import SourceError, SourceJobs
from argus_mask.portraits import CHECKPOINT, prompt_for, subject_line
from tests.helpers import live_server, write_consent
from tests.test_backgrounds import png


@contextmanager
def fake_portrait_darkroom(
    *,
    outcome: str = "passed",
    loras: list | None = None,
    image: bytes | None = None,
    records: list[dict] | None = None,
    policy: str = "default",
) -> Iterator[tuple[str, list[str], dict]]:
    """Darkroom routes the source step uses: generate, image, record, records. Yields (url, calls, state)."""
    picture = image if image is not None else png(64, 80)
    sha = hashlib.sha256(picture).hexdigest()
    calls: list[str] = []
    state: dict = {"n": 0, "by_id": {}}

    def make_record(body: dict, rid: str) -> dict:
        return {
            "id": rid,
            "darkroom_version": "0.4.0",
            "outcome": outcome,
            "outcome_reason": None if outcome == "passed" else "blocked",
            "request": {
                "final_prompt": body["user_prompt"],
                "expanded_prompt": None,
                "negative_prompt": body.get("negative_prompt", ""),
                "seed": body["seed"],
                "checkpoint": {"name": body.get("checkpoint") or CHECKPOINT},
                "sampling": body.get("sampling") or {},
                "loras": loras if loras is not None else body.get("loras") or [],
                "init_image": {"sha256": "ab" * 32} if body.get("init_image") else None,
            },
            "guard": {"enforcing_policy": body.get("enforcing_policy") or policy},
            "image": {"sha256": sha},
        }

    if records is not None:
        for rec in records:
            state["by_id"][rec["id"]] = rec

    app = FastAPI()

    @app.post("/generate/stream")
    async def generate(request: Request):
        calls.append("generate")
        body = await request.json()
        state["n"] += 1
        rid = f"gen{state['n']}"
        record = make_record(body, rid)
        state["by_id"][rid] = record
        state["last_body"] = body

        def events():
            yield f"event: complete\ndata: {json.dumps({'record': record})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/image/{rid}")
    def get_image(rid: str):
        calls.append(f"image {rid}")
        return Response(picture, media_type="image/png")

    @app.get("/record/{rid}")
    def get_record(rid: str):
        calls.append(f"record {rid}")
        found = state["by_id"].get(rid)
        if found is None:
            return JSONResponse({"detail": "no such record"}, status_code=404)
        return found

    @app.get("/records")
    def list_records(outcome: str = "passed", page: int = 1, page_size: int = 50):
        calls.append("records")
        items = [r for r in state["by_id"].values() if r.get("outcome") == outcome]
        return {
            "records": items[(page - 1) * page_size : page * page_size],
            "total": len(items),
            "page": page,
            "page_size": page_size,
        }

    with live_server(app) as url:
        yield url, calls, state


def _jobs(assets: Path, url: str) -> SourceJobs:
    store = GuidedStore(assets)
    store.create("guide")
    return SourceJobs(
        store,
        SourceSettings(darkroom_url=url, policy="default"),
        ConsentStore(assets),
        darkroom=Darkroom(url, sleep=lambda _s: None),
    )


def test_upload_personal(tmp_path: Path) -> None:
    picture = png(32, 40)
    with fake_portrait_darkroom(image=picture) as (url, _calls, _state):
        jobs = _jobs(tmp_path / "assets", url)
        result = jobs.upload("guide", "c1", picture, kind="personal", source="my photo")
    assert result.standing == "passed"
    assert result.data["kind"] == "personal"
    assert result.outputs[0].name == "portrait"
    assert (jobs.store.path("guide", "candidates/c1/portrait.png")).read_bytes() == picture


def test_upload_synthetic_checks_record_sha(tmp_path: Path) -> None:
    picture = png(32, 40)
    sha = hashlib.sha256(picture).hexdigest()
    prior = {
        "id": "rec-syn",
        "outcome": "passed",
        "guard": {"enforcing_policy": "default"},
        "image": {"sha256": sha},
        "request": {"loras": [], "checkpoint": {"name": CHECKPOINT}, "final_prompt": "x", "seed": 1},
    }
    with fake_portrait_darkroom(image=picture, records=[prior]) as (url, _c, _s):
        jobs = _jobs(tmp_path / "assets", url)
        result = jobs.upload("guide", "c1", picture, kind="synthetic", record_id="rec-syn")
        assert result.data["kind"] == "synthetic"
        with pytest.raises(SourceError, match="does not match"):
            jobs.upload("guide", "c2", png(8, 8), kind="synthetic", record_id="rec-syn")


def test_upload_synthetic_lora_refused(tmp_path: Path) -> None:
    picture = png(32, 40)
    sha = hashlib.sha256(picture).hexdigest()
    prior = {
        "id": "rec-lora",
        "outcome": "passed",
        "guard": {"enforcing_policy": "default"},
        "image": {"sha256": sha},
        "request": {
            "loras": [{"name": "char.safetensors"}],
            "checkpoint": {"name": CHECKPOINT},
            "final_prompt": "x",
            "seed": 1,
        },
    }
    with fake_portrait_darkroom(image=picture, records=[prior]) as (url, _c, _s):
        jobs = _jobs(tmp_path / "assets", url)
        with pytest.raises(SourceError, match="personal, not synthetic"):
            jobs.upload("guide", "c1", picture, kind="synthetic", record_id="rec-lora")


def test_upload_consented_needs_covering_record(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    picture = png(32, 40)
    with fake_portrait_darkroom(image=picture) as (url, _c, _s):
        jobs = _jobs(assets, url)
        with pytest.raises(SourceError, match="may not be used"):
            jobs.upload(
                "guide",
                "c1",
                picture,
                kind="consented",
                subject="A. Person",
                release_sha256="ab" * 32,
            )
        write_consent(assets)
        result = jobs.upload(
            "guide",
            "c1",
            picture,
            kind="consented",
            subject="A. Person",
            release_sha256="ab" * 32,
        )
    assert result.data["kind"] == "consented"


def test_upload_over_size_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_mask import portraits as portraits_mod
    from argus_mask.editor import source as source_mod

    monkeypatch.setattr(portraits_mod, "MAX_PORTRAIT_BYTES", 20)
    monkeypatch.setattr(source_mod, "MAX_PORTRAIT_BYTES", 20)
    with fake_portrait_darkroom() as (url, _c, _s):
        jobs = _jobs(tmp_path / "assets", url)
        with pytest.raises(SourceError, match="at most|MB|bytes"):
            jobs.upload("guide", "c1", png(64, 80), kind="personal")


def test_pick_synthetic_and_lora_personal(tmp_path: Path) -> None:
    picture = png(32, 40)
    sha = hashlib.sha256(picture).hexdigest()
    syn = {
        "id": "pick-syn",
        "outcome": "passed",
        "guard": {"enforcing_policy": "default"},
        "image": {"sha256": sha},
        "request": {"loras": [], "checkpoint": {"name": CHECKPOINT}, "final_prompt": "p", "seed": 2},
        "darkroom_version": "0.4.0",
    }
    lora = {
        **syn,
        "id": "pick-lora",
        "request": {**syn["request"], "loras": [{"name": "x.safetensors"}]},
    }
    with fake_portrait_darkroom(image=picture, records=[syn, lora]) as (url, calls, _s):
        jobs = _jobs(tmp_path / "assets", url)
        syn_result = jobs.pick("guide", "c1", "pick-syn")
        assert syn_result.data["kind"] == "synthetic"
        lora_result = jobs.pick("guide", "c2", "pick-lora")
        assert lora_result.data["kind"] == "personal"
        assert "record pick-syn" in " ".join(calls)


@pytest.mark.asyncio
async def test_generate_batch_writes_candidates(tmp_path: Path) -> None:
    picture = png(48, 60)
    with fake_portrait_darkroom(image=picture) as (url, calls, _s):
        jobs = _jobs(tmp_path / "assets", url)
        job = jobs.start_generate(
            "guide", age=30, hair="short dark", clothes="a crew-neck", style="character", seed_text="501-502"
        )
        while job.live:
            await asyncio_sleep()
        assert job.state == "done"
        assert job.candidates == ["s501", "s502"]
        for cand in job.candidates:
            source = jobs.store.load("guide").get("source", cand)
            assert source is not None and source.standing == "passed"
            assert source.data["kind"] == "synthetic"
            assert source.seeds["generator"] in (501, 502)
        assert calls.count("generate") == 2


@pytest.mark.asyncio
async def test_generate_lora_refuses_synthetic(tmp_path: Path) -> None:
    picture = png(48, 60)
    with fake_portrait_darkroom(image=picture, loras=[{"name": "char.safetensors"}]) as (url, _c, _s):
        jobs = _jobs(tmp_path / "assets", url)
        job = jobs.start_generate("guide", age=28, style="character", seed_text="7")
        while job.live:
            await asyncio_sleep()
        assert job.state == "failed"
        source = jobs.store.load("guide").get("source", "s7")
        assert source is not None and source.standing == "refused"
        assert "LoRA" in source.named("refused")


@pytest.mark.asyncio
async def test_restyle_never_raises_kind(tmp_path: Path) -> None:
    picture = png(48, 60)
    with fake_portrait_darkroom(image=picture) as (url, calls, state):
        jobs = _jobs(tmp_path / "assets", url)
        jobs.upload("guide", "c1", picture, kind="personal", source="a photo of me")
        job = jobs.start_restyle("guide", "c1", style="character", seed=99)
        while job.live:
            await asyncio_sleep()
        assert job.state == "done"
        source = jobs.store.load("guide").get("source", "c1")
        assert source is not None
        assert source.data["kind"] == "personal"  # never raised to synthetic
        assert source.data["way"] == "restyle"
        assert source.data["source_sha256"] == hashlib.sha256(picture).hexdigest()
        assert state["last_body"].get("init_image") is not None
        assert "generate" in calls


@pytest.mark.asyncio
async def test_restyle_synthetic_stays_when_cleared(tmp_path: Path) -> None:
    picture = png(48, 60)
    with fake_portrait_darkroom(image=picture) as (url, _c, _s):
        jobs = _jobs(tmp_path / "assets", url)
        # Seed a synthetic source the same way generate would.
        subject = subject_line(age=40, hair="grey")
        prompt = prompt_for(subject, "character")
        from argus_mask.editor.guided import Check, StepResult
        from argus_mask.editor.source import PORTRAIT

        kept = jobs.store.keep("guide", PORTRAIT, "candidates/c1/portrait.png", picture)
        jobs.store.put(
            "guide",
            StepResult(
                step="source",
                candidate="c1",
                checks=[Check(name="provenance", verdict="passed")],
                outputs=[kept],
                data={
                    "way": "generate",
                    "kind": "synthetic",
                    "source": "argus-darkroom record x",
                    "license": "Apache-2.0",
                    "subject": subject,
                    "prompt": prompt,
                },
            ),
        )
        job = jobs.start_restyle("guide", "c1", style="character", seed=3)
        while job.live:
            await asyncio_sleep()
        assert job.state == "done"
        source = jobs.store.load("guide").get("source", "c1")
        assert source is not None and source.data["kind"] == "synthetic"


def test_editor_routes(tmp_path: Path) -> None:
    picture = png(32, 40)
    sha = hashlib.sha256(picture).hexdigest()
    prior = {
        "id": "via-api",
        "outcome": "passed",
        "guard": {"enforcing_policy": "default"},
        "image": {"sha256": sha},
        "request": {"loras": [], "checkpoint": {"name": CHECKPOINT}, "final_prompt": "p", "seed": 1},
        "darkroom_version": "0.4.0",
    }
    with fake_portrait_darkroom(image=picture, records=[prior]) as (url, _c, _s):
        assets = tmp_path / "assets"
        store = GuidedStore(assets)
        consents = ConsentStore(assets)
        source = SourceJobs(
            store,
            SourceSettings(darkroom_url=url),
            consents,
            darkroom=Darkroom(url, sleep=lambda _s: None),
        )
        app = create_editor_app(
            Settings(assets_dir=assets, driver="energy"),
            EditorSettings(),
            source=source,
        )
        with TestClient(app) as client:
            created = client.post("/guided", json={"id": "guide", "name": "Test"})
            assert created.status_code == 200
            assert created.json()["id"] == "guide"
            listed = client.get("/guided").json()
            assert listed["source"]["ready"] is True
            assert "character" in listed["styles"]

            records = client.get("/guided/darkroom/records").json()
            assert any(r["id"] == "via-api" for r in records["records"])

            picked = client.post("/guided/guide/candidates/c1/source/pick", json={"record_id": "via-api"})
            assert picked.status_code == 200
            assert picked.json()["data"]["kind"] == "synthetic"

            upload = client.post(
                "/guided/guide/candidates/c2/source/upload",
                files={"file": ("face.png", picture, "image/png")},
                data={"kind": "personal", "source": "mine"},
            )
            assert upload.status_code == 200
            assert upload.json()["data"]["kind"] == "personal"

            gen = client.post(
                "/guided/guide/source/generate",
                json={"age": 32, "hair": "red", "seeds": "10", "style": "character"},
            )
            assert gen.status_code == 200
            job = gen.json()
            for _ in range(100):
                job = client.get(f"/guided/guide/jobs/{job['id']}").json()
                if job["state"] not in ("queued", "running"):
                    break
                time.sleep(0.02)
            assert job["state"] == "done"
            assert "s10" in job["candidates"]

            restyle = client.post("/guided/guide/candidates/c2/source/restyle", json={"style": "character", "seed": 5})
            assert restyle.status_code == 200
            job = restyle.json()
            for _ in range(100):
                job = client.get(f"/guided/guide/jobs/{job['id']}").json()
                if job["state"] not in ("queued", "running"):
                    break
                time.sleep(0.02)
            assert job["state"] == "done"
            view = client.get("/guided/guide").json()
            kinds = {r["candidate"]: r["data"]["kind"] for r in view["results"] if r["step"] == "source"}
            assert kinds["c2"] == "personal"  # restyle did not raise


def test_source_never_imports_a_gpu_lease() -> None:
    """Must never: a LAM lease is held around darkroom (#178 / #155). Darkroom leases for itself."""
    import ast
    from pathlib import Path

    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/argus_mask/editor/source.py").read_text())
    imported = {
        f"{node.module}.{name.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for name in node.names
    }
    assert not any("gpu" in m or m.endswith(".lease") or "LAM_HEAD" in m for m in imported), imported


async def asyncio_sleep() -> None:
    import asyncio

    await asyncio.sleep(0.02)
