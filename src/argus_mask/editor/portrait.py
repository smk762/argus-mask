"""The portrait step of a guided face (#168; step 2 of #163): checks before any GPU time.

Runs curator ``face_source``, retina OCR (via cortex ``OcrClient``), and mask's own checks. Writes one
``portrait`` step result per candidate. A batch can be ranked by how the checks went.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from argus_cortex.moderation import OcrCheckFailed, OcrClient

from argus_mask.config import PortraitSettings
from argus_mask.editor.curator import CuratorClient, CuratorError, FaceSourceRow
from argus_mask.editor.guided import Artifact, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.editor.portrait_checks import (
    check_expression,
    check_framing,
    check_lighting,
    check_photograph,
    check_temples,
    check_text,
    checks_from_face_source,
    measure_face,
    open_rgb,
    rank_score,
)
from argus_mask.editor.source import PORTRAIT
from argus_mask.models import is_face_id

log = structlog.get_logger(__name__)

JOBS_KEEP = 20
WORST_FIRST = ("refused", "failed", "not_checked", "advisory")
STEP = "portrait"


class PortraitError(GuidedError):
    """Portrait checks cannot run now, with the reason."""


class Stopped(Exception):
    """The job was stopped: its thread writes nothing more."""


class Busy(GuidedError):
    """This build's portraits are already being checked."""


def _utcnow() -> datetime:
    from argus_mask.editor.guided import utc_now

    return utc_now()


@dataclass
class PortraitJob:
    id: str
    build_id: str
    candidates: list[str]
    state: str = "queued"
    stage: str = "waiting"
    error: str | None = None
    status: str | None = None
    ranking: list[dict] = field(default_factory=list)
    asked: float = field(default_factory=time.monotonic)
    ended: float | None = None
    task: asyncio.Task | None = None
    halt: threading.Event = field(default_factory=threading.Event)  # its thread stops at the next check

    @property
    def live(self) -> bool:
        return self.state in ("queued", "running")

    def public(self) -> dict:
        return {
            "id": self.id,
            "build": self.build_id,
            "candidates": list(self.candidates),
            "step": STEP,
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "status": self.status,
            "ranking": list(self.ranking),
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


class PortraitJobs:
    """Portrait checks for the guided faces under one store."""

    def __init__(
        self,
        store: GuidedStore,
        settings: PortraitSettings,
        *,
        curator: CuratorClient | None = None,
        ocr: OcrClient | None = None,
        measure: Callable[[bytes, Any], Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.curator = curator
        if self.curator is None and settings.curator_url:
            self.curator = CuratorClient(settings.curator_url, timeout=settings.timeout_s)
        self.ocr = ocr
        if self.ocr is None and settings.ocr_url:
            self.ocr = OcrClient(settings.ocr_url, timeout=min(settings.timeout_s, 60.0))
        self.measure = measure or measure_face
        self.clock = clock or _utcnow
        self.jobs: dict[str, PortraitJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        """Why portrait checks cannot start at all, or None.

        Curator and OCR may be unset: those checks become required ``not_checked`` and block the build, which
        is intentional when the operator has not pointed at the services yet.
        """
        return None

    def state(self) -> dict:
        return {
            "ready": True,
            "reason": None,
            "curator": self.settings.curator_url or None,
            "ocr": self.settings.ocr_url or None,
            "landmarker": self.settings.landmarker is not None,
            "jobs": [j.public() for j in self.jobs.values() if j.live],
        }

    def start(self, build_id: str, candidates: list[str] | None = None) -> PortraitJob:
        record = self.store.load(build_id)
        if candidates is None:
            candidates = sorted(
                {
                    r.candidate
                    for r in record.results
                    if r.step == "source" and r.candidate and r.standing not in ("failed",)
                }
            )
        if not candidates:
            raise PortraitError(f"{build_id} has no source candidates to check")
        for candidate in candidates:
            if not is_face_id(candidate):
                raise GuidedError(f"{candidate!r} is not a candidate")
            if blocked := record.blockers(STEP, candidate):
                raise NotReady(f"portrait cannot run on {candidate}: " + "; ".join(blocked))
            self._portrait(record.get("source", candidate), build_id, candidate)
        if any(j.live and j.build_id == build_id for j in self.jobs.values()):
            raise Busy(f"{build_id}'s portraits are already being checked")
        job = PortraitJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidates=list(candidates))
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if not j.live][:-JOBS_KEEP]:
            self.jobs.pop(old.id, None)

        def settled(_: asyncio.Task) -> None:
            if job.live:
                job.state, job.stage = "stopped", "stopped"
            job.ended = job.ended or time.monotonic()

        job.task = asyncio.create_task(self._run(job))
        job.task.add_done_callback(settled)
        return job

    async def stop(self, job_id: str) -> PortraitJob | None:
        job = self.jobs.get(job_id)
        if job is not None and job.task is not None and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        return job

    async def stop_all(self) -> None:
        tasks = [j.task for j in self.jobs.values() if j.task is not None and not j.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def ranking_of(self, build_id: str) -> list[dict]:
        """Candidates with a portrait result, best first."""
        record = self.store.load(build_id)
        rows = []
        for result in record.results:
            if result.step != STEP or not result.candidate:
                continue
            rows.append(
                {
                    "candidate": result.candidate,
                    "status": result.standing,
                    "score": rank_score(result.checks),
                    "checks": [c.model_dump(mode="json") for c in result.checks],
                    "reasons": result.named("refused", "advisory", "failed", "not_checked"),
                }
            )
        return sorted(rows, key=lambda r: (-r["score"], r["candidate"]))

    async def _run(self, job: PortraitJob) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "checking"
                batch = asyncio.create_task(asyncio.to_thread(self._check_batch, job))
                try:
                    ranking = await asyncio.shield(batch)
                except asyncio.CancelledError:
                    # Cancelling does not stop a thread: tell it, and let no other batch start until it has ended.
                    job.halt.set()
                    await asyncio.gather(batch, return_exceptions=True)
                    raise
                job.ranking = ranking
                found = {r["status"] for r in ranking}
                job.status = next((s for s in WORST_FIRST if s in found), "passed")
                job.state, job.stage = "done", "done"
        except asyncio.CancelledError:
            job.state, job.stage = "stopped", "stopped"
            raise
        except GuidedError as exc:
            job.state, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.state, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            log.warning("portrait_checks_failed", build=job.build_id, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("portrait_checks", build=job.build_id, state=job.state, n=len(job.candidates))

    def _check_batch(self, job: PortraitJob) -> list[dict]:
        started = self.clock()
        images: dict[str, bytes] = {}
        sources: dict[str, StepResult] = {}
        record = self.store.load(job.build_id)
        for candidate in job.candidates:
            source = record.get("source", candidate)
            images[candidate] = self._portrait(source, job.build_id, candidate)
            sources[candidate] = source  # type: ignore[assignment]

        face_rows, curator_down = self._face_source(job.build_id, images)
        ocr_down: dict[str, str] = {}  # by candidate: one that could not be read says nothing of the others
        ocr_lines: dict[str, list[str] | None] = {}
        if self.ocr is None:
            ocr_down = dict.fromkeys(images, "ARGUS_MASK_OCR_URL is unset: point it at argus-retina-ocr")
            ocr_lines = dict.fromkeys(images, None)
        else:
            for name, data in images.items():
                if job.halt.is_set():
                    raise Stopped
                try:
                    result = self.ocr.read(data, filename=f"{name}.png")
                    ocr_lines[name] = [line.text for line in result.lines if line.score >= self.settings.ocr_min_score]
                except OcrCheckFailed as exc:
                    ocr_down[name] = f"OCR failed: {exc}"
                    ocr_lines[name] = None

        ranking: list[dict] = []
        for candidate, data in images.items():
            if job.halt.is_set():
                raise Stopped
            job.stage = f"checking {candidate}"
            geometry = self.measure(data, self.settings.landmarker)
            image = open_rgb(data)
            source = sources[candidate]
            source_data = dict(source.data) if source else {}
            checks = []
            checks.extend(checks_from_face_source(face_rows.get(candidate), down=curator_down))
            # Curator's refusal of the framing stands: mask's own check speaks only where curator's did not.
            if not any(c.name == "framing" and c.verdict != "passed" for c in checks):
                framing = check_framing(image, min_side_px=self.settings.min_side_px, geometry=geometry)
                checks = [c for c in checks if c.name != "framing"] + [framing]
            checks.append(check_text(ocr_lines.get(candidate), down=ocr_down.get(candidate)))
            checks.append(check_expression(geometry))
            checks.append(check_lighting(image, geometry))
            checks.append(check_temples(image, geometry))
            checks.append(check_photograph(image, source_data, photo_laplacian=self.settings.photo_laplacian))
            # Deduplicate by name: last write wins (framing already handled).
            by_name = {c.name: c for c in checks}
            checks = list(by_name.values())
            tools = []
            if self.curator is not None:
                tools.append(Tool(name="argus-curator face_source", license="MIT"))
            if self.ocr is not None:
                tools.append(Tool(name="argus-retina-ocr", license="AGPL-3.0"))
            if self.settings.landmarker is not None:
                tools.append(Tool(name="MediaPipe Face Landmarker", license="Apache-2.0"))
            portrait = next(a for a in source.outputs if a.name == PORTRAIT and a.file) if source else None
            inputs = [Artifact(name=PORTRAIT, sha256=portrait.sha256, file=portrait.file)] if portrait else []
            result = StepResult(
                step=STEP,
                candidate=candidate,
                checks=checks,
                inputs=inputs,
                tools=tools,
                scores={"rank": rank_score(checks)},
                data={
                    "face_source": face_rows[candidate].keep_reason if candidate in face_rows else None,
                    "restyle": any(c.name == "photograph" and c.verdict == "advisory" for c in checks),
                },
                started=started,
                finished=self.clock(),
            )
            if job.halt.is_set():
                raise Stopped
            kept = self.store.put(job.build_id, result)
            ranking.append(
                {
                    "candidate": candidate,
                    "status": kept.standing,
                    "score": rank_score(kept.checks),
                    "reasons": kept.named("refused", "advisory", "failed", "not_checked"),
                }
            )
        return sorted(ranking, key=lambda r: (-r["score"], r["candidate"]))

    def _face_source(self, build_id: str, images: dict[str, bytes]) -> tuple[dict[str, FaceSourceRow], str | None]:
        """Curator's row per candidate, and why curator could not be asked, if it could not."""
        if self.curator is None:
            return {}, "ARGUS_MASK_CURATOR_URL is unset: point it at argus-curator"
        try:
            rows = self.curator.face_source(images, folder=f"mask-guided/{build_id}")
        except CuratorError as exc:
            log.warning("curator_face_source_failed", build=build_id, reason=str(exc))
            return {}, str(exc)
        return {r.name: r for r in rows}, None

    def _portrait(self, source: StepResult | None, build_id: str, candidate: str) -> bytes:
        """The portrait the source kept, read once: the bytes checked are the bytes scored."""
        if source is None:
            raise GuidedError(f"{candidate} has no source: run the source step first")
        found = next((a for a in source.outputs if a.name == PORTRAIT and a.file), None)
        if found is None:
            raise GuidedError(f"{candidate} has no portrait: its source step keeps one")
        path = self.store.path(build_id, found.file)
        if not path.is_file():
            raise GuidedError(f"{candidate}'s portrait is not where its source kept it")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != found.sha256:
            raise GuidedError(f"{candidate}'s portrait changed since its source was recorded")
        return data
