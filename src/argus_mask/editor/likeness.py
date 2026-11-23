"""The likeness of a guided face's head to its portrait (#169; step 4 of #163).

Scores the two pictures the LAM build kept beside the head (the portrait as LAM was given it, and the head as
the portrait's own camera sees it) with ``lam/likeness.py``, run as a process in the LAM image: MediaPipe and
DINOv2 live there, and nothing of ``lam/`` is imported here.

- **It starts only on settled ground.** The record's own rule: the build must have passed, and the two pictures
  it kept must still be where it left them, unchanged.
- **Scores flag and rank; they do not refuse.** A head MediaPipe cannot find a face in fails the step (nothing
  to score); a missing DINOv2 checkpoint skips identity and leaves the rest. Colour, lids, lips and identity are
  the ones that ranked character heads above photo heads on #151's set; detail did not, and is not kept.
- **One candidate at a time** is enough: the script is CPU-bound and shares the image with nothing that needs the
  GPU for this step.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from argus_mask.config import LikenessSettings
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.faces import sha256_file
from argus_mask.models import is_face_id

log = structlog.get_logger(__name__)

PORTRAIT = "portrait-crop"  # the build step's name for the portrait as LAM was given it
RENDER = "render"  # the head at the portrait's own camera
JOBS_KEEP = 20
PNG = b"\x89PNG\r\n\x1a\n"


class LikenessError(GuidedError):
    """Likeness cannot run now, with the reason."""


class Busy(GuidedError):
    """This candidate's likeness is already being scored."""


@dataclass
class LikenessJob:
    id: str
    build_id: str
    candidate: str
    state: str = "queued"  # queued, running, done, failed, stopped
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
            "step": "likeness",
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "status": self.status,
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


class LikenessJobs:
    """Likeness scoring for the guided faces under one store."""

    def __init__(self, store: GuidedStore, settings: LikenessSettings) -> None:
        self.store = store
        self.settings = settings
        self.jobs: dict[str, LikenessJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        """Why likeness cannot run here, or None."""
        s = self.settings
        if s.script is None or not s.script.is_file():
            return (
                "lam/likeness.py was not found: lam/ is not part of the package "
                "(set ARGUS_MASK_LIKENESS_SCRIPT to a checkout's)"
            )
        if shutil_which(s.docker) is None:
            return f"docker not found: {s.docker!r}"
        return None

    def state(self) -> dict:
        reason = self.check()
        return {
            "ready": reason is None,
            "reason": reason,
            "dinov2": self.settings.dinov2 is not None,
            "jobs": [j.public() for j in self.jobs.values() if j.live],
        }

    def start(self, build_id: str, candidate: str) -> LikenessJob:
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        if blocked := record.blockers("likeness", candidate):
            raise NotReady("likeness cannot run: " + "; ".join(blocked))
        self._pictures(record.get("build", candidate), build_id, candidate)
        if any(j.live and (j.build_id, j.candidate) == (build_id, candidate) for j in self.jobs.values()):
            raise Busy(f"{candidate}'s likeness is already being scored")
        if (why := self.check()) is not None:
            raise LikenessError(why)
        job = LikenessJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidate=candidate)
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

    async def stop(self, job_id: str) -> LikenessJob | None:
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

    async def _run(self, job: LikenessJob) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "scoring"
                result = await self._score(job)
                kept = await asyncio.to_thread(self.store.put, job.build_id, result)
            job.status = kept.standing
            if kept.standing == "passed":
                job.state, job.stage = "done", "done"
            else:
                job.state, job.stage = "failed", kept.standing
                job.error = kept.named("failed", "refused", "not_checked")
        except GuidedError as exc:
            job.state, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # noqa: BLE001 - the page must hear of it
            job.state, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            log.warning("likeness_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("likeness", build=job.build_id, candidate=job.candidate, state=job.state)

    def _pictures(self, build: StepResult | None, build_id: str, candidate: str) -> dict[str, tuple[Artifact, Path]]:
        """The two pictures the build kept, still where it left them."""
        if build is None:
            raise GuidedError(f"{candidate} has no build: likeness scores a head against its portrait")
        found = {}
        for name in (PORTRAIT, RENDER):
            art = next((a for a in build.outputs if a.name == name and a.file), None)
            if art is None:
                raise GuidedError(f"{candidate}'s build left no {name} to score against")
            path = self.store.path(build_id, art.file)
            if not path.is_file():
                raise GuidedError(f"{candidate}'s {name} is not where its build kept it")
            if sha256_file(path) != art.sha256:
                raise GuidedError(f"{candidate}'s {name} changed since its build: run the build again")
            if path.read_bytes()[:8] != PNG:
                raise GuidedError(f"{candidate}'s {name} is not a PNG")
            found[name] = (art, path)
        return found

    async def _score(self, job: LikenessJob) -> StepResult:
        started = self.store.clock()
        record = await asyncio.to_thread(self.store.load, job.build_id)
        if blocked := record.blockers("likeness", job.candidate):
            raise NotReady("likeness cannot run: " + "; ".join(blocked))
        build = record.get("build", job.candidate)
        pictures = await asyncio.to_thread(self._pictures, build, job.build_id, job.candidate)
        work = (self.store.root / ".work" / f"likeness-{job.id}").resolve()
        work.mkdir(parents=True)
        try:
            portrait = work / "portrait.png"
            render = work / "render.png"
            out = work / "likeness.json"
            portrait.write_bytes(pictures[PORTRAIT][1].read_bytes())
            render.write_bytes(pictures[RENDER][1].read_bytes())
            tools = await asyncio.to_thread(self._tools)
            ran = await self._script(job, work, out)
            return await asyncio.to_thread(
                self._collect, job, pictures, tools, ran, out, started, time.monotonic() - job.asked
            )
        finally:
            for path in work.glob("*"):
                path.unlink(missing_ok=True)
            work.rmdir()

    def _tools(self) -> list[Tool]:
        s = self.settings
        assert s.script is not None  # noqa: S101 - check()
        tools = [
            Tool(name="argus-mask lam/likeness.py", sha256=sha256_file(s.script), license="MIT"),
            Tool(name="MediaPipe Face Landmarker", license="Apache-2.0"),
            Tool(name=f"image {s.image}"),
        ]
        if s.dinov2 is not None and (s.dinov2 / "model.safetensors").is_file():
            tools.append(
                Tool(
                    name="DINOv2-base (facebook/dinov2-base)",
                    sha256=sha256_file(s.dinov2 / "model.safetensors"),
                    license="Apache-2.0",
                )
            )
        return tools

    async def _script(self, job: LikenessJob, work: Path, out: Path) -> dict[str, Any]:
        """Run likeness.py in the LAM image. Returns how the run went."""
        s = self.settings
        assert s.script is not None  # noqa: S101
        lam = s.script.parent
        argv = [
            s.docker,
            "run",
            "--rm",
            "--user",
            f"{_uid()}:{_gid()}",
            "-e",
            "HOME=/tmp",
            "-e",
            "MPLCONFIGDIR=/tmp",
            "-v",
            f"{lam}:/argus:ro",
            "-v",
            f"{work}:/work",
        ]
        cmd = [
            "python",
            "/argus/likeness.py",
            "--portrait",
            "/work/portrait.png",
            "--head",
            "/work/render.png",
            "--out",
            "/work/likeness.json",
        ]
        if s.dinov2 is not None and (s.dinov2 / "model.safetensors").is_file():
            argv.extend(["-v", f"{s.dinov2.resolve()}:/dinov2:ro"])
            cmd.extend(["--dinov2", "/dinov2"])
        argv.append(s.image)
        argv.extend(cmd)
        began = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        try:
            said, _ = await asyncio.wait_for(proc.communicate(), s.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise LikenessError(f"likeness took more than {s.timeout_s:g} s") from None
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        text = said.decode(errors="replace")
        return {
            "code": proc.returncode,
            "said": text,
            "seconds": round(time.monotonic() - began, 1),
            "out": out if out.is_file() else None,
        }

    def _collect(
        self,
        job: LikenessJob,
        pictures: dict[str, tuple[Artifact, Path]],
        tools: list[Tool],
        ran: dict[str, Any],
        out: Path,
        started: Any,  # noqa: ANN401 - AwareDatetime
        seconds: float,
    ) -> StepResult:
        checks: list[Check] = []
        scores: dict[str, float] = {}
        data: dict[str, Any] = {}
        outputs: list[Artifact] = []
        run = {"job": job.id, "seconds": round(seconds, 1), "script_s": ran["seconds"]}
        if ran["code"] != 0 or ran["out"] is None:
            last = "; ".join(line.strip()[:200] for line in ran["said"].splitlines() if line.strip())[-1500:]
            why = last or f"likeness ended with status {ran['code']} and wrote nothing"
            # A crash before any score: the step failed. Scores never refuse a head that was scored.
            checks.append(Check(name="likeness", verdict="failed", reason=why[:2000]))
            return StepResult(
                step="likeness",
                candidate=job.candidate,
                checks=checks,
                inputs=[pictures[PORTRAIT][0], pictures[RENDER][0]],
                tools=tools,
                run=run,
                started=started,
                finished=self.store.clock(),
            )
        try:
            raw = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            checks.append(Check(name="likeness", verdict="failed", reason=f"likeness wrote no JSON: {exc}"))
            return StepResult(
                step="likeness",
                candidate=job.candidate,
                checks=checks,
                inputs=[pictures[PORTRAIT][0], pictures[RENDER][0]],
                tools=tools,
                run=run,
                started=started,
                finished=self.store.clock(),
            )
        if err := raw.get("error"):
            checks.append(Check(name="faces", verdict="failed", reason=str(err)[:2000]))
        else:
            checks.append(Check(name="faces", verdict="passed"))
            for name, value in (raw.get("scores") or {}).items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    scores[str(name)] = float(value)
            if "identity" in scores:
                checks.append(Check(name="identity", verdict="passed", measure={"identity": scores["identity"]}))
            elif self.settings.dinov2 is None:
                # Optional: no checkpoint configured. Do not leave an advisory that blocks the pick.
                run["identity"] = "skipped: no DINOv2 checkpoint (set ARGUS_MASK_DINOV2)"
            else:
                checks.append(
                    Check(
                        name="identity",
                        verdict="not_checked",
                        reason="DINOv2 did not return an identity score",
                        required=False,
                    )
                )
            if colour := raw.get("colour"):
                data["colour"] = colour
            if shape := raw.get("shape"):
                data["shape"] = shape
            kept = f"candidates/{job.candidate}/likeness.json"
            outputs.append(self.store.keep(job.build_id, "likeness", kept, out.read_bytes()))
            checks.append(Check(name="likeness", verdict="passed"))
        return StepResult(
            step="likeness",
            candidate=job.candidate,
            checks=checks,
            inputs=[pictures[PORTRAIT][0], pictures[RENDER][0]],
            outputs=outputs,
            tools=tools,
            scores=scores,
            data=data,
            run=run,
            started=started,
            finished=self.store.clock(),
        )


def shutil_which(name: str) -> str | None:
    import shutil  # noqa: PLC0415

    return shutil.which(name)


def _uid() -> int:
    import os  # noqa: PLC0415

    return os.getuid()


def _gid() -> int:
    import os  # noqa: PLC0415

    return os.getgid()
