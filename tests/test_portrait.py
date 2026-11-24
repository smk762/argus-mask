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


async def _ended(job) -> None:  # noqa: ANN001
    done, _ = await asyncio.wait({job.task}, timeout=30)
    assert done, f"the job is still {job.state} ({job.stage}) after 30 s"
    await asyncio.sleep(0)


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
    await _ended(job)
    assert job.state == "done"
    assert [r["candidate"] for r in job.ranking] == ["c1", "c2"]
    assert job.ranking[0]["score"] > job.ranking[1]["score"]
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
    await _ended(job)
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


# ---- ranking, framing, text, sharpness ----------------------------------------------------------------


def _check(verdict: str, **over: object) -> Check:
    return Check(name="a", verdict=verdict, reason="why", **over)


def test_no_number_of_lesser_findings_outranks_a_worse_one() -> None:
    refused = rank_score([_check("refused")])
    failed = rank_score([_check("failed")] * 30)
    not_checked = rank_score([_check("not_checked")] * 30)
    advisory = rank_score([_check("advisory")] * 30)
    optional = rank_score([_check("not_checked", required=False)] * 30)
    assert 100.0 > optional > advisory > not_checked > failed > refused
    assert rank_score([_check("not_checked")] * 2) > refused and rank_score([_check("failed")] * 2) > refused


def _geometry(*_: object) -> FaceGeometry:
    return FaceGeometry(face_count=1, box=(0.25, 0.2, 0.75, 0.7))


def _portraits(tmp_path: Path, *names: str, **over: object) -> PortraitJobs:
    store = GuidedStore(tmp_path / "assets")
    store.create("guide")
    for i, name in enumerate(names):
        _seed_source(store, name, _temple_covered(640 + i, 800))
    return PortraitJobs(
        store,
        PortraitSettings(min_side_px=512),
        **{"curator": FakeCurator(), "ocr": FakeOcr(), "measure": _geometry, **over},
    )


@pytest.mark.asyncio
async def test_curators_refusal_of_the_framing_stands(tmp_path: Path) -> None:
    jobs = _portraits(tmp_path, "c1", curator=FakeCurator({"c1": "face-source:face-size"}))
    job = jobs.start("guide")
    await _ended(job)
    found = jobs.store.load("guide").get("portrait", "c1")
    assert found.standing == "refused" and job.status == "refused"
    (framing,) = [c for c in found.checks if c.name == "framing"]
    assert framing.verdict == "refused" and framing.reason == "face-source:face-size"


@pytest.mark.asyncio
async def test_one_unreadable_picture_says_nothing_of_the_others_text(tmp_path: Path) -> None:
    class OneFails(FakeOcr):
        def read(self, data: bytes, filename: str = "image"):  # noqa: ANN202
            self.fail = "boom" if filename == "c2.png" else None
            return super().read(data, filename)

    jobs = _portraits(tmp_path, "c1", "c2", ocr=OneFails())
    await _ended(jobs.start("guide"))
    record = jobs.store.load("guide")
    text = {c: next(k for k in record.get("portrait", c).checks if k.name == "text") for c in ("c1", "c2")}
    assert text["c1"].verdict == "passed"
    assert text["c2"].verdict == "not_checked" and "boom" in text["c2"].reason


@pytest.mark.asyncio
async def test_the_portrait_scored_is_the_one_that_was_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bytes] = []
    jobs = _portraits(tmp_path, "c1", measure=lambda data, _lm: (seen.append(data), _geometry())[1])
    checked = jobs._portrait
    job = jobs.start("guide")

    def then_rewritten(*args: object) -> object:
        found = checked(*args)
        jobs.store.path("guide", "candidates/c1/portrait.png").write_bytes(_png(700, 700))
        return found

    monkeypatch.setattr(jobs, "_portrait", then_rewritten)
    await _ended(job)
    assert seen == [_temple_covered(640, 800)]


@pytest.mark.asyncio
async def test_a_stopped_check_writes_nothing_more_and_ends_before_the_next(tmp_path: Path) -> None:
    import threading

    reached, go = threading.Event(), threading.Event()
    running: list[str] = []

    def slow(*_: object) -> FaceGeometry:
        running.append("in")
        assert len(running) == 1 or go.is_set(), "two batches at once"
        reached.set()
        go.wait(10)
        return _geometry()

    jobs = _portraits(tmp_path, "c1", "c2", measure=slow)
    job = jobs.start("guide")
    assert await asyncio.to_thread(reached.wait, 10)
    stopping = asyncio.create_task(jobs.stop(job.id))
    await asyncio.sleep(0.1)
    assert jobs._one.locked(), "the lock is held while the thread still runs"
    go.set()
    await asyncio.wait_for(stopping, 10)
    await asyncio.sleep(0.1)
    assert job.state == "stopped" and not jobs._one.locked()
    record = jobs.store.load("guide")
    assert record.get("portrait", "c1") is None and record.get("portrait", "c2") is None


def test_the_laplacian_is_signed_and_not_clipped() -> None:
    import numpy as np

    from argus_mask.editor.portrait_checks import _laplacian_var

    gray = np.random.default_rng(0).integers(0, 256, (40, 30), dtype=np.uint8)
    picture = Image.fromarray(gray).convert("RGB")
    negative = Image.fromarray(255 - gray).convert("RGB")
    a = gray.astype(float)
    inner = [9 * a[y, x] - a[y - 1 : y + 2, x - 1 : x + 2].sum() for y in range(1, 39) for x in range(1, 29)]
    assert _laplacian_var(picture) == pytest.approx(np.var(inner), rel=1e-4)
    assert _laplacian_var(negative) == pytest.approx(_laplacian_var(picture), rel=1e-4)
    assert _laplacian_var(Image.new("RGB", (2, 2))) == 0.0


@pytest.mark.parametrize("fails", [False, True])
def test_a_face_landmarker_is_closed_after_each_portrait(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    import sys
    import types

    from argus_mask.editor.portrait_checks import measure_face

    closed: list[bool] = []

    class Detector:
        def __enter__(self):  # noqa: ANN204
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

        def close(self) -> None:
            closed.append(True)

        def detect(self, _image: object) -> object:
            if fails:
                raise RuntimeError("boom")
            return types.SimpleNamespace(face_landmarks=[], face_blendshapes=[])

    vision = types.SimpleNamespace(
        FaceLandmarkerOptions=lambda **_: None,
        FaceLandmarker=types.SimpleNamespace(create_from_options=lambda _o: Detector()),
    )
    mp = types.ModuleType("mediapipe")
    mp.Image, mp.ImageFormat = lambda **_: None, types.SimpleNamespace(SRGB=1)
    tasks, python = types.ModuleType("mediapipe.tasks"), types.ModuleType("mediapipe.tasks.python")
    python.BaseOptions, python.vision = lambda **_: None, vision
    for name, module in (("mediapipe", mp), ("mediapipe.tasks", tasks), ("mediapipe.tasks.python", python)):
        monkeypatch.setitem(sys.modules, name, module)
    found = measure_face(_png(32, 40), "model.task")
    assert found is None if fails else found.face_count == 0
    assert closed == [True]


# ---- the curator client ---------------------------------------------------------------------------------


class Opener:
    """Curator's replies by path, and what it was asked."""

    def __init__(self, **replies: object) -> None:
        self.replies = {"upload": {}, "folder": {"scan_id": "s1"}, "select": {"rows": []}, "delete": {}, **replies}
        self.asked: list[tuple[str, str, bytes]] = []

    def __call__(self, req, timeout: float):  # noqa: ANN001, ANN204
        import json
        from urllib.error import HTTPError

        path = req.full_url.removeprefix("http://c")
        self.asked.append((req.method, path, req.data))
        key = path.rsplit("/", 1)[-1]
        reply = self.replies.get(key, {})
        if isinstance(reply, int):
            raise HTTPError(req.full_url, reply, "no", {}, io.BytesIO(b"refused"))
        return io.BytesIO(json.dumps(reply).encode())


def _ask(opener: Opener, *names: str, **over: object) -> list[FaceSourceRow]:
    return CuratorClient("http://c", opener=opener).face_source({n: b"png" for n in names}, **over)


@pytest.mark.parametrize("name", ['a"; x="', "a\r\nX: y", "../a", "a/b", "", "a.b"])
def test_a_name_that_could_change_the_upload_is_never_sent(name: str) -> None:
    from argus_mask.editor.curator import CuratorError

    opener = Opener()
    with pytest.raises(CuratorError, match="not a name"):
        _ask(opener, "ok", name)
    with pytest.raises(CuratorError, match="not a folder"):
        _ask(opener, "ok", folder="mask-guided/../x")
    assert opener.asked == []


def test_a_row_without_its_path_is_skipped() -> None:
    rows = [{"keep_reason": "x"}, {"rel_path": 3}, {"rel_path": "f/c1.png", "keep_reason": "face-source:yaw"}]
    (c1, c2) = _ask(Opener(select={"rows": rows}), "c1", "c2")
    assert c1.keep_reason == "face-source:yaw" and c2.keep_reason == "face-source:missing"


@pytest.mark.parametrize("reply", [[], None, "text", 3.5])
def test_a_reply_that_is_no_object_is_curators_error(reply: object) -> None:
    from argus_mask.editor.curator import CuratorError

    for where in ("folder", "select"):
        with pytest.raises(CuratorError, match="not an object"):
            _ask(Opener(**{where: reply}), "c1")
    with pytest.raises(CuratorError, match="not an object"):
        CuratorClient("http://c", opener=Opener(faces=reply)).age(b"png")


def test_the_uploaded_portraits_are_taken_off_curator_again() -> None:
    import json

    from argus_mask.editor.curator import CuratorError

    rows = [{"rel_path": "c1.png", "keep_reason": ""}]
    opener = Opener(select={"rows": rows})
    _ask(opener, "c1", folder="mask-guided/guide")
    assert [(m, p) for m, p, _ in opener.asked[-2:]] == [("POST", "/scan/s1/delete"), ("DELETE", "/scan/s1")]
    assert json.loads(opener.asked[-2][2]) == {"rel_paths": ["c1.png"]}
    failed = Opener(select=500)
    with pytest.raises(CuratorError, match="500"):
        _ask(failed, "c1")
    assert ("POST", "/scan/s1/delete") in [(m, p) for m, p, _ in failed.asked]
    # A curator that deletes nothing (the default) costs the clean-up, never the check.
    assert _ask(Opener(select={"rows": rows}, delete=403), "c1")[0].keep_reason == ""


def test_each_run_uploads_to_a_folder_of_its_own() -> None:
    """Curator never overwrites: a name uploaded before to the same folder would be scored as it was then."""
    import json

    opener = Opener()
    for _ in range(2):
        _ask(opener, "c1", folder="mask-guided/guide")
    folders = [json.loads(data)["path"] for _, path, data in opener.asked if path == "/scan/folder"]
    assert len(set(folders)) == 2 and all(f.startswith("mask-guided/guide/") for f in folders)
