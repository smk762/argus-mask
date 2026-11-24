"""The age step of a guided face (#172; step 4 of #163): the head's apparent age against its portrait's.

Asks argus-curator ``POST /faces`` (MiVOLO) for the age of the portrait-crop and of the head render the
build kept, writes both and their difference into the record, and flags when the head reads much older
(#151: photograph heads ≈ +10 years; character heads did not). Scores advise; they do not refuse. This is
not the 18-or-over check — that one is on the portrait before any build.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from argus_mask.age_eval import OLDER_YEARS, AgeReport, score
from argus_mask.config import AgeSettings
from argus_mask.editor.curator import AgeReading, CuratorClient, CuratorError
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.models import is_face_id

log = structlog.get_logger(__name__)

PORTRAIT = "portrait-crop"
RENDER = "render"
JOBS_KEEP = 20
STEP = "age"
PNG = b"\x89PNG\r\n\x1a\n"


class AgeError(GuidedError):
    """Age read cannot run now, with the reason."""


class Busy(GuidedError):
    """This candidate's age is already being read."""


def _utcnow() -> datetime:
    from argus_mask.editor.guided import utc_now

    return utc_now()


@dataclass
class AgeJob:
    id: str
    build_id: str
    candidate: str
    state: str = "queued"
    stage: str = "waiting"
    error: str | None = None
    status: str | None = None
    scores: dict | None = None
    asked: float = field(default_factory=time.monotonic)
    ended: float | None = None
    task: asyncio.Task | None = None

    @property
    def live(self) -> bool:
        return self.state in ("queued", "running")

    def public(self) -> dict:
        return {
            "id": self.id,
            "build": self.build_id,
            "candidate": self.candidate,
            "step": STEP,
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "status": self.status,
            "scores": self.scores,
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


class AgeJobs:
    """Age reads for guided faces. *read_age* injects a stand-in that returns an :class:`AgeReading`."""

    def __init__(
        self,
        store: GuidedStore,
        settings: AgeSettings,
        *,
        curator: CuratorClient | None = None,
        read_age: Callable[[str, bytes], AgeReading] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.read_age = read_age
        self.clock = clock or _utcnow
        self.curator = curator
        if self.curator is None and settings.curator_url and self.read_age is None:
            self.curator = CuratorClient(settings.curator_url, timeout=settings.timeout_s)
        self.jobs: dict[str, AgeJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        if self.read_age is not None:
            return None
        if self.curator is None:
            return "ARGUS_MASK_CURATOR_URL is unset: point it at argus-curator for POST /faces ages"
        return None

    def state(self) -> dict:
        reason = self.check()
        return {
            "ready": reason is None,
            "reason": reason,
            "older_years": self.settings.older_years,
            "jobs": [j.public() for j in self.jobs.values() if j.live],
        }

    def start(self, build_id: str, candidate: str) -> AgeJob:
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        if blocked := record.blockers(STEP, candidate):
            raise NotReady("age cannot run: " + "; ".join(blocked))
        self._kept(record.get("build", candidate), build_id, candidate)  # hashed when the job runs, off the event loop
        if any(j.live and (j.build_id, j.candidate) == (build_id, candidate) for j in self.jobs.values()):
            raise Busy(f"{candidate}'s age is already being read")
        if (why := self.check()) is not None:
            raise AgeError(why)
        job = AgeJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidate=candidate)
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

    async def stop(self, job_id: str) -> AgeJob | None:
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

    async def _run(self, job: AgeJob) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "reading"
                result = await self._measure(job)
                kept = await asyncio.to_thread(self.store.put, job.build_id, result)
            job.status = kept.standing
            job.scores = dict(kept.scores) if kept.scores else None
            if kept.standing in ("passed", "advisory"):
                job.state, job.stage = "done", "done"
            else:
                job.state, job.stage = "failed", kept.standing
                job.error = kept.named("failed", "refused", "not_checked")
        except asyncio.CancelledError:
            job.state, job.stage = "stopped", "stopped"
            raise
        except GuidedError as exc:
            job.state, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.state, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            log.warning("age_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("age", build=job.build_id, candidate=job.candidate, state=job.state)

    async def _measure(self, job: AgeJob) -> StepResult:
        started = self.clock()
        record = await asyncio.to_thread(self.store.load, job.build_id)
        if blocked := record.blockers(STEP, job.candidate):
            raise NotReady("age cannot run: " + "; ".join(blocked))
        build = record.get("build", job.candidate)
        pictures = await asyncio.to_thread(self._pictures, build, job.build_id, job.candidate)
        job.stage = "portrait age"
        portrait_bytes, head_bytes = pictures[PORTRAIT][1], pictures[RENDER][1]
        try:
            portrait = await asyncio.to_thread(self._read, "portrait", portrait_bytes)
            job.stage = "head age"
            head = await asyncio.to_thread(self._read, "head", head_bytes)
        except CuratorError as exc:
            return StepResult(
                step=STEP,
                candidate=job.candidate,
                checks=[Check(name="age", verdict="not_checked", reason=str(exc))],
                inputs=[pictures[PORTRAIT][0], pictures[RENDER][0]],
                tools=[Tool(name="argus-curator POST /faces (MiVOLO)", license="Apache-2.0")],
                started=started,
                finished=self.clock(),
            )
        report = score(
            portrait.years,
            head.years,
            threshold=self.settings.older_years,
            source=portrait.source or head.source,
        )
        return self._collect(job, pictures, report, started)

    def _read(self, label: str, data: bytes) -> AgeReading:
        if self.read_age is not None:
            return self.read_age(label, data)
        assert self.curator is not None  # noqa: S101
        return self.curator.age(data)

    def _collect(
        self,
        job: AgeJob,
        pictures: dict[str, tuple[Artifact, bytes]],
        report: AgeReport,
        started: datetime,
    ) -> StepResult:
        if report.older:
            checks = [
                Check(
                    name="older",
                    verdict="advisory",
                    reason=(
                        f"head reads {report.head_years:.0f}, portrait {report.portrait_years:.0f} "
                        f"(+{report.delta_years:.0f} years; threshold {report.threshold:g})"
                    ),
                    measure={
                        "portrait_years": report.portrait_years,
                        "head_years": report.head_years,
                        "delta_years": report.delta_years,
                    },
                )
            ]
        else:
            checks = [
                Check(
                    name="older",
                    verdict="passed",
                    measure={
                        "portrait_years": report.portrait_years,
                        "head_years": report.head_years,
                        "delta_years": report.delta_years,
                    },
                )
            ]
        return StepResult(
            step=STEP,
            candidate=job.candidate,
            checks=checks,
            inputs=[pictures[PORTRAIT][0], pictures[RENDER][0]],
            tools=[
                Tool(name="argus-curator POST /faces (MiVOLO)", license="Apache-2.0"),
                Tool(name="argus-mask age_eval", license="MIT"),
            ],
            scores={
                "portrait_years": report.portrait_years,
                "head_years": report.head_years,
                "delta_years": report.delta_years,
            },
            data=report.as_data(),
            started=started,
            finished=self.clock(),
        )

    def _kept(self, build: Any, build_id: str, candidate: str) -> dict[str, tuple[Artifact, Path]]:
        """The two pictures the build kept, still where it left them. Not read: cheap enough for a request."""
        if build is None:
            raise GuidedError(f"{candidate} has no build: age reads a head against its portrait")
        found: dict[str, tuple[Artifact, Path]] = {}
        for name in (PORTRAIT, RENDER):
            art = next((a for a in build.outputs if a.name == name and a.file), None)
            if art is None:
                raise GuidedError(f"{candidate}'s build left no {name} to age")
            path = self.store.path(build_id, art.file)
            if not path.is_file():
                raise GuidedError(f"{candidate}'s {name} is not where its build kept it")
            found[name] = (art, path)
        return found

    def _pictures(self, build: Any, build_id: str, candidate: str) -> dict[str, tuple[Artifact, bytes]]:
        """The two pictures' bytes, unchanged since the build: read once, so what is checked is what is aged."""
        found: dict[str, tuple[Artifact, bytes]] = {}
        for name, (art, path) in self._kept(build, build_id, candidate).items():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != art.sha256:
                raise GuidedError(f"{candidate}'s {name} changed since its build: run the build again")
            if data[:8] != PNG:
                raise GuidedError(f"{candidate}'s {name} is not a PNG")
            found[name] = (art, data)
        return found


def read_age_synthetic(portrait_years: float, head_years: float) -> Callable[[str, bytes], AgeReading]:
    """A test double: fixed ages for portrait vs head (character ≈ same; photo ≈ +10)."""

    def read(label: str, _data: bytes) -> AgeReading:
        years = portrait_years if label == "portrait" else head_years
        return AgeReading(years=years, source="mivolo", faces=1)

    return read


__all__ = ["OLDER_YEARS", "AgeError", "AgeJob", "AgeJobs", "Busy", "read_age_synthetic"]
