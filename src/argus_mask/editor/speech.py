"""The speech step of a guided face (#171; step 4 of #163): the mouth in speech, against a reference head.

Plays one fixed, cleared take (``talk.json``) through the candidate (and ``splat-01`` when present), reads the
lip gap per frame with MediaPipe, and writes closed-share / reversals / opening-p95 into the build record.
Flags a stutter or an oversized mouth from #161's before/after thresholds. Scores advise; they do not refuse.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from argus_mask.config import SpeechSettings
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.editor.scripts import HeadServer, browsers_env, run_script
from argus_mask.faces import MANIFEST_NAME, build_manifest, sha256_file
from argus_mask.models import Provenance, is_face_id
from argus_mask.speech_eval import (
    LipMetrics,
    SpeechReport,
    metrics_from_gaps,
    reference_baseline,
    score,
    synthetic_gaps,
)

log = structlog.get_logger(__name__)

HEAD = "head"
JOBS_KEEP = 20
STEP = "speech"
FACE = "speech-head"


class SpeechError(GuidedError):
    """Speech eval cannot run now, with the reason."""


class Busy(GuidedError):
    """This candidate's speech is already being measured."""


def _utcnow() -> datetime:
    from argus_mask.editor.guided import utc_now

    return utc_now()


@dataclass
class SpeechJob:
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


class SpeechJobs:
    """Speech evals for guided faces. *analyze* injects a stand-in that returns a :class:`SpeechReport`."""

    def __init__(
        self,
        store: GuidedStore,
        settings: SpeechSettings,
        *,
        assets: Path | None = None,
        analyze: Callable[[str, str, Path], SpeechReport] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.assets = Path(assets) if assets is not None else None
        self.analyze = analyze
        self.clock = clock or _utcnow
        self.jobs: dict[str, SpeechJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        if self.analyze is not None:
            return None
        if self.settings.script is None or not self.settings.script.is_file():
            return (
                "lam/measure_speech.py was not found: lam/ is not part of the package "
                "(set ARGUS_MASK_SPEECH_SCRIPT to a checkout's)"
            )
        if self.settings.take is None or not self.settings.take.is_file():
            return "speech take (talk.json) was not found (set ARGUS_MASK_SPEECH_TAKE)"
        if self.settings.landmarker is None or not self.settings.landmarker.is_file():
            return (
                "MediaPipe's face_landmarker.task was not found "
                "(set ARGUS_MASK_LANDMARKER, or copy face_landmarker.task into the argus-mask cache)"
            )
        return None

    def state(self) -> dict:
        reason = self.check()
        return {
            "ready": reason is None,
            "reason": reason,
            "reference": self.settings.reference,
            "jobs": [j.public() for j in self.jobs.values() if j.live],
        }

    def start(self, build_id: str, candidate: str) -> SpeechJob:
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        if blocked := record.blockers(STEP, candidate):
            raise NotReady("speech cannot run: " + "; ".join(blocked))
        # On the event loop: that the head is there, not its hash. The job hashes it, in a thread, before it runs.
        self._inputs(record, build_id, candidate, verify=False)
        if any(j.live and (j.build_id, j.candidate) == (build_id, candidate) for j in self.jobs.values()):
            raise Busy(f"{candidate}'s speech is already being measured")
        if (why := self.check()) is not None:
            raise SpeechError(why)
        job = SpeechJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidate=candidate)
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

    async def stop(self, job_id: str) -> SpeechJob | None:
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

    async def _run(self, job: SpeechJob) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "measuring"
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
            log.warning("speech_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("speech", build=job.build_id, candidate=job.candidate, state=job.state)

    async def _measure(self, job: SpeechJob) -> StepResult:
        started = self.clock()
        record = await asyncio.to_thread(self.store.load, job.build_id)
        if blocked := record.blockers(STEP, job.candidate):
            raise NotReady("speech cannot run: " + "; ".join(blocked))
        head_art, head_path, rest_data = await asyncio.to_thread(self._inputs, record, job.build_id, job.candidate)
        if self.analyze is not None:
            report = await asyncio.to_thread(self.analyze, job.build_id, job.candidate, head_path)
            return await asyncio.to_thread(self._collect, job, head_art, report, None, started)
        work = (self.store.root / ".work" / f"speech-{job.id}").resolve()
        serve = work / "serve"
        face_dir = serve / FACE
        out = work / "speech.json"
        server = None
        try:
            face_dir.mkdir(parents=True)
            model = face_dir / f"{FACE}.zip"
            shutil.copyfile(head_path, model)
            manifest = build_manifest(
                model,
                face_id=FACE,
                name=FACE,
                provenance=Provenance(kind="synthetic", source="guided speech", license="MIT"),
            )
            (face_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            self._stage_rest(serve, rest_data)
            job.stage = "serving the head"
            server = HeadServer(serve)
            if not await asyncio.to_thread(server.start):
                raise SpeechError("the speech server did not start")
            job.stage = "drawing the take"
            ran = await self._script(job, server.url, FACE, out)
            cand = self._metrics_from_file(out)
            ref = await self._reference_metrics(job)
            report = score(cand, ref)
            return await asyncio.to_thread(self._collect, job, head_art, report, ran, started)
        finally:
            if server is not None:
                await asyncio.to_thread(server.stop)
            await asyncio.to_thread(shutil.rmtree, work, True)

    def _stage_rest(self, serve: Path, rest_data: dict) -> None:
        """Put the fitted rest set on the short-lived face so speech runs with it."""
        weights = rest_data.get("weights")
        if not isinstance(weights, dict) or not weights:
            return
        name = str(rest_data.get("name") or "fit")
        life = float(rest_data["life"]) if isinstance(rest_data.get("life"), (int, float)) else 1.0
        body = {
            "sets": {name: {"weights": weights, "life": life}},
            "faces": {FACE: name},
        }
        (serve / "expressions.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    def _collect(
        self,
        job: SpeechJob,
        head: Artifact,
        report: SpeechReport,
        ran: dict | None,
        started: datetime,
    ) -> StepResult:
        checks: list[Check] = []
        if report.stutter:
            checks.append(
                Check(
                    name="stutter",
                    verdict="advisory",
                    reason="; ".join(r for r in report.reasons if "opening p95" not in r) or "stutter",
                    measure=report.candidate.as_data(),
                )
            )
        else:
            checks.append(Check(name="stutter", verdict="passed", measure=report.candidate.as_data()))
        if report.oversized:
            checks.append(
                Check(
                    name="oversized",
                    verdict="advisory",
                    reason=next((r for r in report.reasons if "opening p95" in r), "oversized mouth"),
                    measure={"opening_p95": report.candidate.opening_p95, "reference": report.reference.opening_p95},
                )
            )
        else:
            checks.append(
                Check(
                    name="oversized",
                    verdict="passed",
                    measure={"opening_p95": report.candidate.opening_p95},
                )
            )
        tools = [Tool(name="argus-mask speech_eval", license="MIT")]
        if self.analyze is None:
            tools.append(Tool(name="playwright viewer record", license="Apache-2.0"))
            tools.append(Tool(name="MediaPipe Face Landmarker", license="Apache-2.0"))
        return StepResult(
            step=STEP,
            candidate=job.candidate,
            checks=checks,
            inputs=[head],
            tools=tools,
            scores={
                "closed_share": report.candidate.closed_share,
                "reversals": float(report.candidate.reversals),
                "opening_p95": report.candidate.opening_p95,
                "ref_closed_share": report.reference.closed_share,
                "ref_reversals": float(report.reference.reversals),
                "ref_opening_p95": report.reference.opening_p95,
            },
            data=report.as_data(),
            run=ran or {},
            started=started,
            finished=self.clock(),
        )

    def _inputs(
        self, record: Any, build_id: str, candidate: str, *, verify: bool = True
    ) -> tuple[Artifact, Path, dict]:
        build = record.get("build", candidate)
        head = next((a for a in (build.outputs if build else []) if a.name == HEAD and a.file), None)
        if head is None:
            raise GuidedError(f"{candidate} has no head: its build step keeps one")
        path = self.store.path(build_id, head.file)
        if not path.is_file():
            raise GuidedError(f"{candidate}'s head is not where its build kept it")
        if verify and sha256_file(path) != head.sha256:
            raise GuidedError(f"{candidate}'s head changed since its build was recorded")
        rest = record.get("rest", candidate)
        rest_data = dict(rest.data) if rest is not None and isinstance(rest.data, dict) else {}
        return head, path, rest_data

    async def _reference_metrics(self, job: SpeechJob) -> LipMetrics:
        """Measure splat-01 when it is under assets; else #161's after baseline."""
        ref_id = self.settings.reference
        root = self.assets
        if root is None or not (root / ref_id).is_dir():
            return reference_baseline()
        work = (self.store.root / ".work" / f"speech-ref-{uuid.uuid4().hex[:8]}").resolve()
        out = work / "ref.json"
        server = None
        try:
            work.mkdir(parents=True)
            # Point the short-lived server at the real assets so splat-01 loads as itself.
            server = HeadServer(root)
            if not await asyncio.to_thread(server.start):
                raise SpeechError("the speech server did not start")
            await self._script(job, server.url, ref_id, out)
            return self._metrics_from_file(out)
        except Exception as exc:  # noqa: BLE001
            log.warning("speech_reference_failed", face=ref_id, reason=str(exc))
            return reference_baseline()
        finally:
            if server is not None:
                await asyncio.to_thread(server.stop)
            await asyncio.to_thread(shutil.rmtree, work, True)

    async def _script(self, job: SpeechJob, url: str, face: str, out: Path) -> dict[str, Any]:
        s = self.settings
        assert s.script is not None and s.take is not None and s.landmarker is not None  # noqa: S101
        argv = [
            s.uv,
            "run",
            "--with",
            "pillow",
            "--with",
            "playwright",
            "--with",
            "mediapipe==0.10.14",
            "--with",
            "numpy",
            "python",
            str(s.script),
            "--url",
            url,
            "--face",
            face,
            "--take",
            str(s.take),
            "--model",
            str(s.landmarker),
            "--out",
            str(out),
            "--size",
            str(s.size),
            "--stride",
            str(s.stride),
        ]
        began = time.monotonic()
        try:
            code, text = await run_script(argv, cwd=s.script.parent, timeout_s=s.timeout_s, env=browsers_env())
        except TimeoutError:
            raise SpeechError(f"speech took more than {s.timeout_s:g} s") from None
        if code != 0 or not out.is_file():
            raise SpeechError(text.strip()[-2000:] or f"measure_speech exited {code}")
        return {"code": code, "seconds": round(time.monotonic() - began, 1), "log": text[-2000:]}

    @staticmethod
    def _metrics_from_file(path: Path) -> LipMetrics:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw.get("gaps"), list) and raw["gaps"]:
            return metrics_from_gaps([float(g) for g in raw["gaps"]])
        return LipMetrics(
            closed_share=float(raw["closed_share"]),
            reversals=int(raw["reversals"]),
            opening_p95=float(raw["opening_p95"]),
            frames=int(raw.get("frames") or 0),
        )


def analyze_synthetic(stutter: bool = False) -> Callable[[str, str, Path], SpeechReport]:
    """A test double: gap series like #161 before (*stutter*) or after (clean)."""

    def analyze(_build: str, _candidate: str, _head: Path) -> SpeechReport:
        return score(metrics_from_gaps(synthetic_gaps(stutter=stutter)), reference_baseline())

    return analyze
