"""The turns step of a guided face (#170; step 4 of #163): how far the head turns before it gives out.

Draws the head at the contact-sheet poses (limits off), scores hairline/temple gaps, and writes measured
``limits`` into the build record. Save (#176) copies them onto the face manifest.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
import uvicorn

from argus_mask.config import Settings, TurnsSettings
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.faces import MANIFEST_NAME, build_manifest, sha256_file
from argus_mask.models import Limits, Provenance, is_face_id
from argus_mask.server.app import create_app
from argus_mask.turns import DEFAULT_GAP, PoseScore, TurnReport, measure_limits, synthetic_pose

log = structlog.get_logger(__name__)

HEAD = "head"
JOBS_KEEP = 20
STEP = "turns"
FACE = "turn-head"


class TurnsError(GuidedError):
    """Turn checks cannot run now, with the reason."""


class Busy(GuidedError):
    """This candidate's turns are already being measured."""


def _utcnow() -> datetime:
    from argus_mask.editor.guided import utc_now

    return utc_now()


@dataclass
class TurnsJob:
    id: str
    build_id: str
    candidate: str
    state: str = "queued"
    stage: str = "waiting"
    error: str | None = None
    status: str | None = None
    limits: dict | None = None
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
            "limits": self.limits,
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


class TurnsJobs:
    """Turn checks for guided faces. *analyze* injects a stand-in that returns a :class:`TurnReport`."""

    def __init__(
        self,
        store: GuidedStore,
        settings: TurnsSettings,
        *,
        analyze: Callable[[str, str, Path], TurnReport] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.analyze = analyze
        self.clock = clock or _utcnow
        self.jobs: dict[str, TurnsJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        if self.analyze is not None:
            return None
        if self.settings.script is None or not self.settings.script.is_file():
            return (
                "lam/measure_turns.py was not found: lam/ is not part of the package "
                "(set ARGUS_MASK_TURNS_SCRIPT to a checkout's)"
            )
        return None

    def state(self) -> dict:
        reason = self.check()
        return {
            "ready": reason is None,
            "reason": reason,
            "jobs": [j.public() for j in self.jobs.values() if j.live],
        }

    def start(self, build_id: str, candidate: str) -> TurnsJob:
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        if blocked := record.blockers(STEP, candidate):
            raise NotReady("turns cannot run: " + "; ".join(blocked))
        self._inputs(record, build_id, candidate)
        if any(j.live and (j.build_id, j.candidate) == (build_id, candidate) for j in self.jobs.values()):
            raise Busy(f"{candidate}'s turns are already being measured")
        if (why := self.check()) is not None:
            raise TurnsError(why)
        job = TurnsJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidate=candidate)
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

    async def stop(self, job_id: str) -> TurnsJob | None:
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

    async def _run(self, job: TurnsJob) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "measuring"
                result = await self._measure(job)
                kept = await asyncio.to_thread(self.store.put, job.build_id, result)
            job.status = kept.standing
            job.limits = kept.data.get("limits") if isinstance(kept.data, dict) else None
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
            log.warning("turns_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("turns", build=job.build_id, candidate=job.candidate, state=job.state)

    async def _measure(self, job: TurnsJob) -> StepResult:
        started = self.clock()
        record = await asyncio.to_thread(self.store.load, job.build_id)
        if blocked := record.blockers(STEP, job.candidate):
            raise NotReady("turns cannot run: " + "; ".join(blocked))
        head_art, head_path = await asyncio.to_thread(self._inputs, record, job.build_id, job.candidate)
        if self.analyze is not None:
            report = await asyncio.to_thread(self.analyze, job.build_id, job.candidate, head_path)
            return await asyncio.to_thread(self._collect, job, head_art, report, None, started)
        work = (self.store.root / ".work" / f"turns-{job.id}").resolve()
        serve = work / "serve"
        face_dir = serve / FACE
        out = work / "turns.json"
        server = None
        try:
            face_dir.mkdir(parents=True)
            model = face_dir / f"{FACE}.zip"
            shutil.copyfile(head_path, model)
            manifest = build_manifest(
                model,
                face_id=FACE,
                name=FACE,
                provenance=Provenance(kind="synthetic", source="guided turns", license="MIT"),
            )
            (face_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            job.stage = "serving the head"
            url, server = await asyncio.to_thread(self._serve, serve)
            job.stage = "drawing poses"
            ran = await self._script(job, url, out)
            raw = json.loads(out.read_text(encoding="utf-8"))
            report = TurnReport(
                scores=tuple(
                    PoseScore(
                        label=p["label"],
                        yaw=float(p["yaw"]),
                        pitch=float(p["pitch"]),
                        gap=float(p["gap"]),
                        region=str(p["region"]),
                    )
                    for p in raw.get("poses") or []
                ),
                limits=Limits(**raw["limits"]),
                threshold=float(raw.get("threshold", DEFAULT_GAP)),
            )
            return await asyncio.to_thread(self._collect, job, head_art, report, raw, started, ran)
        finally:
            if server is not None:
                server.should_exit = True
            await asyncio.to_thread(shutil.rmtree, work, True)

    def _collect(
        self,
        job: TurnsJob,
        head: Artifact,
        report: TurnReport,
        raw: dict | None,
        started: datetime,
        ran: dict | None = None,
    ) -> StepResult:
        checks = []
        failed = [s for s in report.scores if s.gap >= report.threshold and s.label != "front"]
        if failed:
            worst = max(failed, key=lambda s: s.gap)
            checks.append(
                Check(
                    name="hairline",
                    verdict="advisory",
                    reason=(
                        f"{worst.label} gaps (score {worst.gap:.2f} at {worst.region}): "
                        f"limits yaw={report.limits.yaw:g}° pitch_up={report.limits.pitch_up:g}°"
                    ),
                    measure={
                        "yaw": float(report.limits.yaw or 0),
                        "pitch_up": float(report.limits.pitch_up or 0),
                        "gap": float(worst.gap),
                    },
                )
            )
        else:
            checks.append(
                Check(
                    name="hairline",
                    verdict="passed",
                    measure={
                        "yaw": float(report.limits.yaw or 0),
                        "pitch_up": float(report.limits.pitch_up or 0),
                    },
                )
            )
        data = report.as_data()
        if raw is not None:
            data["raw"] = raw
        tools = [Tool(name="argus-mask turns", license="MIT")]
        if self.analyze is None:
            tools.append(Tool(name="playwright viewer record", license="Apache-2.0"))
        return StepResult(
            step=STEP,
            candidate=job.candidate,
            checks=checks,
            inputs=[head],
            tools=tools,
            scores={
                "yaw": float(report.limits.yaw or 0),
                "pitch_up": float(report.limits.pitch_up or 0),
                "pitch_down": float(report.limits.pitch_down or 0),
            },
            data=data,
            run=ran or {},
            started=started,
            finished=self.clock(),
        )

    def _inputs(self, record: Any, build_id: str, candidate: str) -> tuple[Artifact, Path]:
        build = record.get("build", candidate)
        head = next((a for a in (build.outputs if build else []) if a.name == HEAD and a.file), None)
        if head is None:
            raise GuidedError(f"{candidate} has no head: its build step keeps one")
        path = self.store.path(build_id, head.file)
        if not path.is_file():
            raise GuidedError(f"{candidate}'s head is not where its build kept it")
        if sha256_file(path) != head.sha256:
            raise GuidedError(f"{candidate}'s head changed since its build was recorded")
        return head, path

    def _serve(self, assets: Path) -> tuple[str, uvicorn.Server]:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        app = create_app(Settings(assets_dir=assets, driver="energy"))
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise TurnsError("the turns server did not start")
        return f"http://127.0.0.1:{port}", server

    async def _script(self, job: TurnsJob, url: str, out: Path) -> dict[str, Any]:
        s = self.settings
        assert s.script is not None  # noqa: S101
        argv = [
            s.uv,
            "run",
            "--with",
            "pillow",
            "--with",
            "playwright",
            "python",
            str(s.script),
            "--url",
            url,
            "--face",
            FACE,
            "--out",
            str(out),
            "--size",
            str(s.size),
            "--threshold",
            str(s.threshold),
        ]
        env = {
            **os.environ,
            "PLAYWRIGHT_BROWSERS_PATH": os.environ.get(
                "PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright")
            ),
        }
        began = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=str(s.script.parent),
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=s.timeout_s)
        except TimeoutError:
            proc.kill()
            raise TurnsError(f"turns took more than {s.timeout_s:g} s") from None
        text = (stdout or b"").decode(errors="replace")
        if proc.returncode != 0 or not out.is_file():
            raise TurnsError(text.strip()[-2000:] or f"measure_turns exited {proc.returncode}")
        return {"code": proc.returncode, "seconds": round(time.monotonic() - began, 1), "log": text[-2000:]}


def analyze_synthetic(weak: bool = False) -> Callable[[str, str, Path], TurnReport]:
    """A test double: builds synthetic poses that gap like splat-01 (*weak*) or syn-w101."""

    def analyze(_build: str, _candidate: str, _head: Path) -> TurnReport:
        def capture(yaw: float, pitch: float) -> bytes:
            return synthetic_pose(320, 400, yaw=yaw, pitch=pitch, weak=weak)

        return measure_limits(capture, threshold=DEFAULT_GAP)

    return analyze
