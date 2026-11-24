"""The rest set of a guided face's head, fitted to its portrait (#169; step 5 of #163).

Runs ``lam/fit_rest.py`` as a process: the viewer draws each candidate set against a short-lived server that
serves only this head, and MediaPipe scores it. Nothing of ``lam/`` is imported here.

- **It starts only on settled ground.** The build must have passed; the portrait is the one the source step kept.
- **Smile at least 0.15, lip-parting shapes capped** — ``fit_rest.py``'s own rules; the fitted set and its score
  go in the record. Scores flag and rank; they do not refuse.
- **One fit at a time**: the script holds a browser and MediaPipe for minutes.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from argus_mask.config import RestSettings
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.editor.scripts import HeadServer, browsers_env, run_script
from argus_mask.faces import MANIFEST_NAME, build_manifest, sha256_file
from argus_mask.models import Provenance, is_face_id

log = structlog.get_logger(__name__)

HEAD = "head"
PORTRAIT = "portrait"  # the source step's portrait
FACE = "rest-fit"  # the short-lived face id the viewer loads; never a name from a request
JOBS_KEEP = 20


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


class RestError(GuidedError):
    """A rest fit cannot run now, with the reason."""


class Busy(GuidedError):
    """This candidate's rest set is already being fitted."""


@dataclass
class RestJob:
    id: str
    build_id: str
    candidate: str
    state: str = "queued"
    stage: str = "waiting"
    error: str | None = None
    status: str | None = None
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
            "step": "rest",
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "status": self.status,
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


class RestFits:
    """Rest-set fits for the guided faces under one store."""

    def __init__(self, store: GuidedStore, settings: RestSettings) -> None:
        self.store = store
        self.settings = settings
        self.jobs: dict[str, RestJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        s = self.settings
        if s.script is None or not s.script.is_file():
            return (
                "lam/fit_rest.py was not found: lam/ is not part of the package "
                "(set ARGUS_MASK_REST_SCRIPT to a checkout's)"
            )
        if s.landmarker is None or not s.landmarker.is_file():
            return (
                "MediaPipe's face_landmarker.task was not found "
                "(set ARGUS_MASK_LANDMARKER, or copy face_landmarker.task from the LAM image into the argus-mask cache)"
            )
        if shutil.which(s.uv) is None:
            return f"uv not found: {s.uv!r} (rest fit needs pillow and mediapipe for the script)"
        return None

    def state(self) -> dict:
        reason = self.check()
        return {"ready": reason is None, "reason": reason, "jobs": [j.public() for j in self.jobs.values() if j.live]}

    def start(self, build_id: str, candidate: str) -> RestJob:
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        if blocked := record.blockers("rest", candidate):
            raise NotReady("rest cannot run: " + "; ".join(blocked))
        self._inputs(record, build_id, candidate)
        if any(j.live and (j.build_id, j.candidate) == (build_id, candidate) for j in self.jobs.values()):
            raise Busy(f"{candidate}'s rest set is already being fitted")
        if (why := self.check()) is not None:
            raise RestError(why)
        job = RestJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidate=candidate)
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

    async def stop(self, job_id: str) -> RestJob | None:
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

    async def _run(self, job: RestJob) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "fitting"
                result = await self._fit(job)
                kept = await asyncio.to_thread(self.store.put, job.build_id, result)
            job.status = kept.standing
            if kept.standing == "passed":
                job.state, job.stage = "done", "done"
            else:
                job.state, job.stage = "failed", kept.standing
                job.error = kept.named("failed", "refused", "not_checked")
        except GuidedError as exc:
            job.state, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.state, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            log.warning("rest_fit_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("rest_fit", build=job.build_id, candidate=job.candidate, state=job.state)

    def _inputs(
        self,
        record: Any,
        build_id: str,
        candidate: str,  # noqa: ANN401
    ) -> tuple[tuple[Artifact, Path], tuple[Artifact, Path]]:
        source = record.get("source", candidate)
        build = record.get("build", candidate)
        portrait = next((a for a in (source.outputs if source else []) if a.name == PORTRAIT and a.file), None)
        head = next((a for a in (build.outputs if build else []) if a.name == HEAD and a.file), None)
        if portrait is None:
            raise GuidedError(f"{candidate} has no portrait: its source step keeps one")
        if head is None:
            raise GuidedError(f"{candidate} has no head: its build step keeps one")
        paths = []
        for art, label in ((portrait, "portrait"), (head, "head")):
            path = self.store.path(build_id, art.file)
            if not path.is_file():
                raise GuidedError(f"{candidate}'s {label} is not where it was kept")
            if sha256_file(path) != art.sha256:
                raise GuidedError(f"{candidate}'s {label} changed since it was recorded: run that step again")
            paths.append((art, path))
        return paths[0], paths[1]

    async def _fit(self, job: RestJob) -> StepResult:
        started = self.store.clock()
        record = await asyncio.to_thread(self.store.load, job.build_id)
        if blocked := record.blockers("rest", job.candidate):
            raise NotReady("rest cannot run: " + "; ".join(blocked))
        (portrait_art, portrait_path), (head_art, head_path) = await asyncio.to_thread(
            self._inputs, record, job.build_id, job.candidate
        )
        work = (self.store.root / ".work" / f"rest-{job.id}").resolve()
        serve = work / "serve"
        face_dir = serve / FACE
        out = work / "rest.json"
        server = None
        try:
            face_dir.mkdir(parents=True)
            model = face_dir / f"{FACE}.zip"
            shutil.copyfile(head_path, model)
            manifest = build_manifest(
                model,
                face_id=FACE,
                name=FACE,
                provenance=Provenance(kind="synthetic", source="guided rest fit", license="MIT"),
            )
            (face_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            job.stage = "serving the head"
            server = HeadServer(serve)
            if not await asyncio.to_thread(server.start):
                raise RestError("the rest-fit server did not start")
            job.stage = "fitting the rest set"
            ran = await self._script(job, server.url, portrait_path, out)
            return await asyncio.to_thread(
                self._collect,
                job,
                portrait_art,
                head_art,
                ran,
                out,
                started,
                time.monotonic() - job.asked,
            )
        finally:
            if server is not None:
                await asyncio.to_thread(server.stop)
            await asyncio.to_thread(shutil.rmtree, work, True)

    async def _script(self, job: RestJob, url: str, portrait: Path, out: Path) -> dict[str, Any]:
        s = self.settings
        assert s.script is not None and s.landmarker is not None  # noqa: S101
        # pillow and mediapipe are not package deps: pull them for this run only.
        argv = [
            s.uv,
            "run",
            "--with",
            "pillow",
            "--with",
            "mediapipe==0.10.14",
            "--with",
            "numpy",
            "python",
            str(s.script),
            "--url",
            url,
            "--face",
            FACE,
            "--portrait",
            str(portrait),
            "--model",
            str(s.landmarker),
            "--out",
            str(out),
            "--name",
            f"fit-{job.candidate}",
            "--generations",
            str(s.generations),
        ]
        began = time.monotonic()
        try:
            code, said = await run_script(argv, cwd=s.script.parent, timeout_s=s.timeout_s, env=browsers_env())
        except TimeoutError:
            raise RestError(f"rest fit took more than {s.timeout_s:g} s") from None
        return {
            "code": code,
            "said": said,
            "seconds": round(time.monotonic() - began, 1),
            "out": out if out.is_file() else None,
        }

    def _collect(
        self,
        job: RestJob,
        portrait: Artifact,
        head: Artifact,
        ran: dict[str, Any],
        out: Path,
        started: Any,  # noqa: ANN401
        seconds: float,
    ) -> StepResult:
        s = self.settings
        assert s.script is not None and s.landmarker is not None  # noqa: S101
        tools = [
            Tool(name="argus-mask lam/fit_rest.py", sha256=sha256_file(s.script), license="MIT"),
            Tool(name="MediaPipe Face Landmarker", sha256=sha256_file(s.landmarker), license="Apache-2.0"),
        ]
        checks: list[Check] = []
        scores: dict[str, float] = {}
        data: dict[str, Any] = {}
        outputs: list[Artifact] = []
        run = {"job": job.id, "seconds": round(seconds, 1), "script_s": ran["seconds"]}
        if ran["code"] != 0 or ran["out"] is None:
            last = "; ".join(line.strip()[:200] for line in ran["said"].splitlines() if line.strip())[-1500:]
            why = last or f"rest fit ended with status {ran['code']} and wrote nothing"
            checks.append(Check(name="rest", verdict="failed", reason=why[:2000]))
            return StepResult(
                step="rest",
                candidate=job.candidate,
                checks=checks,
                inputs=[portrait, head],
                tools=tools,
                run=run,
                started=started,
                finished=self.store.clock(),
            )
        try:
            raw = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            checks.append(Check(name="rest", verdict="failed", reason=f"rest fit wrote no JSON: {exc}"))
            return StepResult(
                step="rest",
                candidate=job.candidate,
                checks=checks,
                inputs=[portrait, head],
                tools=tools,
                run=run,
                started=started,
                finished=self.store.clock(),
            )
        weights = raw.get("weights")
        # json reads NaN and Infinity as numbers, and NaN is under no floor.
        unsound = [str(k) for k, v in weights.items() if not _finite(v)] if isinstance(weights, dict) else []
        if not isinstance(weights, dict) or not weights:
            checks.append(Check(name="rest", verdict="failed", reason="rest fit wrote no weights"))
        elif unsound:
            why = f"rest fit wrote weights that are not finite numbers: {', '.join(sorted(unsound))}"
            checks.append(Check(name="rest", verdict="failed", reason=why[:2000]))
        else:
            smile = min(float(weights.get("mouthSmileLeft", 0)), float(weights.get("mouthSmileRight", 0)))
            if smile + 1e-9 < 0.15:
                checks.append(
                    Check(
                        name="smile",
                        verdict="failed",
                        reason=f"mouthSmile is {smile:.3f}, under the 0.15 floor",
                        measure={"mouthSmile": smile},
                    )
                )
            else:
                checks.append(Check(name="smile", verdict="passed", measure={"mouthSmile": smile}))
            data["weights"] = {str(k): float(v) for k, v in weights.items()}
            data["life"] = float(raw["life"]) if _finite(raw.get("life")) else 1.0
            data["name"] = str(raw.get("name") or f"fit-{job.candidate}")
            report = raw.get("score") or {}
            for label in ("warm", "fit"):
                total = (report.get(label) or {}).get("total")
                if isinstance(total, (int, float)) and math.isfinite(total):
                    scores[f"rest_misfit_{label}"] = float(total)
            for name in ("aperture", "lip_ratio"):
                value = (report.get("fit") or {}).get(name)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    scores[name] = float(value)
            kept = f"candidates/{job.candidate}/rest.json"
            outputs.append(self.store.keep(job.build_id, "rest", kept, out.read_bytes()))
            for label in ("warm", "fit"):
                shot = out.with_suffix(f".{label}.png")
                if shot.is_file():
                    outputs.append(
                        self.store.keep(
                            job.build_id,
                            f"rest-{label}",
                            f"candidates/{job.candidate}/rest.{label}.png",
                            shot.read_bytes(),
                        )
                    )
            checks.append(Check(name="rest", verdict="passed"))
        return StepResult(
            step="rest",
            candidate=job.candidate,
            checks=checks,
            inputs=[portrait, head],
            outputs=outputs,
            tools=tools,
            scores=scores,
            data=data,
            run=run,
            started=started,
            finished=self.store.clock(),
        )
