"""Guided portrait checks before any GPU time (#168)."""

from __future__ import annotations

import asyncio
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image, ImageDraw
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, PortraitSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.curator import CuratorClient, FaceSourceRow
from argus_mask.editor.guided import Check, GuidedStore, StepResult
from argus_mask.editor.portrait import PortraitJobs
from argus_mask.editor.portrait_checks import (
    FaceGeometry,
    check_framing,
    check_photograph,
    check_temples,
    check_text,
    checks_from_face_source,
    open_rgb,
    rank_score,
)
from argus_mask.editor.source import PORTRAIT
from tests.helpers import live_server


def _png(width: int, height: int, *, paint=None) -> bytes:
    image = Image.new("RGB", (width, height), (200, 170, 150))
    if paint:
        paint(image)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _temple_bare(width: int = 640, height: int = 800) -> bytes:
    """Skin-coloured temples (upper corners): should advisory."""

    def paint(im: Image.Image) -> None:
        draw = ImageDraw.Draw(im)
        # Face oval darker hair on top centre, bare skin at temples.
        draw.ellipse((180, 120, 460, 520), fill=(210, 175, 155))
        draw.rectangle((40, 100, 140, 280), fill=(220, 180, 160))  # left temple
        draw.rectangle((500, 100, 600, 280), fill=(220, 180, 160))  # right temple
        draw.ellipse((220, 80, 420, 200), fill=(40, 30, 25))  # hair on crown only

    return _png(width, height, paint=paint)


def _temple_covered(width: int = 640, height: int = 800) -> bytes:
    def paint(im: Image.Image) -> None:
        draw = ImageDraw.Draw(im)
        draw.ellipse((180, 120, 460, 520), fill=(210, 175, 155))
        draw.rectangle((40, 80, 200, 320), fill=(35, 25, 20))  # hair over left temple
        draw.rectangle((440, 80, 600, 320), fill=(35, 25, 20))  # hair over right temple

    return _png(width, height, paint=paint)


def test_framing_refuses_a_tiny_portrait() -> None:
    image = open_rgb(_png(64, 80))
    check = check_framing(image, min_side_px=512)
    assert check.verdict == "refused"
    assert "512" in check.reason


def test_temples_catch_bare_and_pass_covered() -> None:
    bare = check_temples(open_rgb(_temple_bare()))
    covered = check_temples(open_rgb(_temple_covered()))
    assert bare.verdict == "advisory"
    assert "temple" in bare.reason.lower()
    assert covered.verdict == "passed"


def test_text_advisory_when_ocr_reads_a_signature() -> None:
    assert check_text([]).verdict == "passed"
    found = check_text(["FLUX", "signature"])
    assert found.verdict == "advisory"
    assert "FLUX" in found.reason
    down = check_text(None, down="OCR unreachable")
    assert down.verdict == "not_checked" and down.required


def test_face_source_reasons_map_to_checks() -> None:
    kept = checks_from_face_source(FaceSourceRow(name="c1", keep_reason=""))
    assert all(c.verdict == "passed" for c in kept)
    age = checks_from_face_source(FaceSourceRow(name="c1", keep_reason="face-source:age"))
    assert next(c for c in age if c.name == "adult").verdict == "refused"
    down = checks_from_face_source(None, down="curator is down")
    assert all(c.verdict == "not_checked" and c.required for c in down)


def test_photograph_advisory_for_photo_style_and_detailed_upload() -> None:
    # High-frequency noise → high Laplacian → photograph advisory for an upload.
    def speckled(im: Image.Image) -> None:
        px = im.load()
        for y in range(0, im.height, 2):
            for x in range(0, im.width, 2):
                px[x, y] = (30, 30, 30)

    photo = check_photograph(
        open_rgb(_png(256, 320, paint=speckled)),
        {"way": "upload"},
        photo_laplacian=10.0,
    )
    assert photo.verdict == "advisory"
    character = check_photograph(
        open_rgb(_png(256, 320)),
        {"way": "generate", "style": "character"},
        photo_laplacian=10.0,
    )
    assert character.verdict == "passed"


def test_rank_score_orders_refusals_last() -> None:
    good = [Check(name="a", verdict="passed")]
    bad = [Check(name="a", verdict="refused", reason="no")]
    warn = [Check(name="a", verdict="advisory", reason="maybe")]
    assert rank_score(good) > rank_score(warn) > rank_score(bad)


class FakeOcr:
    def __init__(self, lines: list[str] | None = None, fail: str | None = None) -> None:
        self.lines = lines or []
        self.fail = fail

    def read(self, data: bytes, filename: str = "image"):
        if self.fail:
            raise __import__("argus_cortex.moderation", fromlist=["OcrCheckFailed"]).OcrCheckFailed(self.fail)

        class Line:
            def __init__(self, text: str) -> None:
                self.text = text
                self.score = 0.9

        class Result:
            def __init__(self, lines: list[str]) -> None:
                self.lines = [Line(t) for t in lines]

        return Result(self.lines)


class FakeCurator:
    def __init__(self, reasons: dict[str, str] | None = None) -> None:
        self.reasons = reasons or {}

    def face_source(self, images: dict[str, bytes], *, folder: str | None = None) -> list[FaceSourceRow]:
        return [
            FaceSourceRow(name=name, keep_reason=self.reasons.get(name, ""), rank=i + 1)
            for i, name in enumerate(images)
        ]


def _seed_source(store: GuidedStore, candidate: str, data: bytes, *, style: str = "character") -> None:
    kept = store.keep("guide", PORTRAIT, f"candidates/{candidate}/portrait.png", data)
    store.put(
        "guide",
        StepResult(
            step="source",
            candidate=candidate,
            checks=[Check(name="provenance", verdict="passed")],
            outputs=[kept],
            data={"way": "generate", "kind": "synthetic", "style": style, "source": "test", "license": "Apache-2.0"},
        ),
    )


@pytest.mark.asyncio
async def test_portrait_job_ranks_and_writes_results(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    store.create("guide")
    _seed_source(store, "c1", _temple_covered(640, 800))
    _seed_source(store, "c2", _temple_bare(640, 800), style="photo")
    jobs = PortraitJobs(
        store,
        PortraitSettings(curator_url="http://curator.test", ocr_url="http://ocr.test", min_side_px=512),
        curator=FakeCurator({"c2": "face-source:yaw"}),
        ocr=FakeOcr(lines=["signed by FLUX"]),
        measure=lambda _data, _lm: FaceGeometry(face_count=1, box=(0.25, 0.2, 0.75, 0.7)),
    )
    job = jobs.start("guide")
    while job.live:
        await asyncio.sleep(0.02)
    assert job.state == "done"
    assert [r["candidate"] for r in job.ranking] == ["c1", "c2"] or job.ranking[0]["score"] >= job.ranking[1]["score"]
    c2 = store.load("guide").get("portrait", "c2")
    assert c2 is not None
    assert any(c.name == "temples" and c.verdict == "advisory" for c in c2.checks)
    assert any(c.name == "text" and c.verdict == "advisory" for c in c2.checks)
    assert any(c.name == "frontal" and c.verdict == "advisory" for c in c2.checks)
    # Required OCR/curator configured: no blocking not_checked.
    assert c2.standing in ("advisory", "refused", "passed")


@pytest.mark.asyncio
async def test_unset_curator_and_ocr_block_as_not_checked(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    store.create("guide")
    _seed_source(store, "c1", _temple_covered(640, 800))
    jobs = PortraitJobs(store, PortraitSettings(min_side_px=512), measure=lambda *_a: None)
    job = jobs.start("guide")
    while job.live:
        await asyncio.sleep(0.02)
    result = store.load("guide").get("portrait", "c1")
    assert result is not None
    assert result.standing == "not_checked"
    assert any(c.name == "identity" and c.verdict == "not_checked" for c in result.checks)
    assert any(c.name == "text" and c.verdict == "not_checked" for c in result.checks)


def test_editor_portrait_routes(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    store = GuidedStore(assets)
    store.create("guide")
    _seed_source(store, "c1", _temple_covered(640, 800))
    jobs = PortraitJobs(
        store,
        PortraitSettings(curator_url="http://c", ocr_url="http://o", min_side_px=512),
        curator=FakeCurator(),
        ocr=FakeOcr(lines=[]),
        measure=lambda *_a: FaceGeometry(face_count=1, box=(0.2, 0.2, 0.8, 0.8)),
    )
    app = create_editor_app(
        Settings(assets_dir=assets, driver="energy"),
        EditorSettings(),
        portrait=jobs,
    )
    with TestClient(app) as client:
        listed = client.get("/guided").json()
        assert listed["portrait"]["ready"] is True
        started = client.post("/guided/guide/portrait")
        assert started.status_code == 200
        job = started.json()
        for _ in range(100):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            import time

            time.sleep(0.02)
        assert job["state"] == "done"
        ranking = client.get("/guided/guide/portrait").json()
        assert ranking["ranking"][0]["candidate"] == "c1"


@contextmanager
def fake_curator_http(reasons: dict[str, str]) -> Iterator[str]:
    """Minimal curator upload/scan/select for CuratorClient."""
    state: dict = {"images": {}}
    app = FastAPI()

    @app.post("/upload")
    async def upload(files: list[UploadFile] = File(...), folder: str = Form(...)) -> dict:
        for item in files:
            name = (item.filename or "x.png").rsplit(".", 1)[0]
            state["images"][name] = await item.read()
        state["folder"] = folder
        return {"folder": folder, "count": len(files)}

    @app.post("/scan/folder")
    def scan(body: dict) -> dict:
        return {"scan_id": "scan1", "path": body.get("path")}

    @app.post("/scan/{scan_id}/select")
    def select(scan_id: str, body: dict) -> dict:
        rows = []
        for i, name in enumerate(state["images"]):
            rows.append(
                {
                    "rel_path": f"{state.get('folder', '')}/{name}.png",
                    "abs_path": f"/tmp/{name}.png",
                    "keep_reason": reasons.get(name, ""),
                    "rank": i + 1 if not reasons.get(name) else None,
                }
            )
        return {
            "scan_id": scan_id,
            "total": len(rows),
            "kept": sum(1 for r in rows if not r["keep_reason"]),
            "rows": rows,
        }

    with live_server(app) as url:
        yield url


def test_curator_client_face_source_roundtrip() -> None:
    with fake_curator_http({"c2": "face-source:no-face"}) as url:
        client = CuratorClient(url, timeout=5)
        rows = client.face_source({"c1": _png(32, 40), "c2": _png(32, 40)})
    by = {r.name: r for r in rows}
    assert by["c1"].keep_reason == ""
    assert by["c2"].keep_reason == "face-source:no-face"
