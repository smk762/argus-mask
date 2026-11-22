"""The LAM build of a guided face (#155; step 3 of #163): one portrait in, one splat head in the build record.

The head is made by ``lam/make_heads.sh``, the checked pipeline, run as a process: nothing of ``lam/`` or of LAM
is imported here, and no other way of making a head exists in the package. This module is the job around it:

- **It starts only on settled ground.** The record's own rule (``guided.py``): a portrait a check refused never
  reaches LAM. The portrait is the one the source step kept, read by the build's own path and checked against the
  sha256 it was recorded with. No path comes from a request.
- **Host memory is checked first, and watched.** A build starts only with ``min_free_gb`` free, and one that
  runs while free memory falls under ``stop_free_gb`` has its container killed: a build on a host short of
  memory has taken the whole machine down (``lam/README.md``).
- **The record holds what a build found, not that one could not start.** No memory, no GPU or no image fails the
  job with its reason and leaves the record as it was; a pipeline that ran and refused, failed or was stopped is
  written as the step's result.
- **One head at a time**, each its own process, in the order asked.
- **The ``lam-head`` lease is held only while the pipeline runs** (``gpu.py``): never while a job waits its
  turn, reads the cache or writes the record, and never around darkroom, which leases for itself.
- **The same portrait gives the same head**, so a head is kept by its portrait's sha256 and the pipeline's own
  key (``lam/``'s scripts, the image, FLAME 2023 Open, the LAM checkout's commit, Blender) and is not built twice.

What it finds goes in the record as the ``build`` step's result: the portrait in, the head out (checked as the
server checks a splat face), every part the head depends on with its licence, and how the run went.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from argus_cortex.gpu import GpuLeaseRefused, GpuLeaseTimeout, NullCoordinator

from argus_mask.config import LamSettings
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, NotReady, StepResult, Tool
from argus_mask.faces import FaceError, inspect_splat, sha256_file
from argus_mask.gpu import LAM_HEAD, lease
from argus_mask.models import is_face_id

log = structlog.get_logger(__name__)

HEAD = "head"  # the name the script builds under: the archive's folder, never a name from a request
PORTRAIT = "portrait"  # the source step's output a head is built from
WATCH_S = 2.0  # how often a running build's memory, time and log are looked at
HALT_WAIT_S = 30.0  # a stopped script gives LAM's files back to this user, then ends; past this it is killed
ASK_TIMEOUT_S = 30.0
CLEAR_TIMEOUT_S = 60.0
LOG_TAIL = 30
JOBS_KEEP = 20
CLEANER_IMAGE = "alpine:3.21"  # as make_heads.sh: the container writes as root, so a container clears after it

# make_head.py prints this once its own checks have passed (FLAME 2023 Open, our embedding, no LAM FLAME code, our
# front end), and one of these when the preflight refuses.
CHECKED = "FLAME core: argus-mask's; FLAME model: 2023 Open"
REFUSALS = re.compile(
    r"is not FLAME 2023 Open|is missing: mount|is not argus-mask's landmark embedding|LAM's own code is still in use"
)

# Every part a head depends on, with its licence (lam/README.md's table). The pinned ones are added per build.
PARTS: tuple[Tool, ...] = (
    Tool(name="LAM weights (lam-20k)", license="CC BY-NC 4.0"),
    Tool(name="VHAP tracker, vendored in LAM: the FLAME fit", license="CC BY-NC-SA 4.0"),
    Tool(name="VGGHeads head detector", license="MIT (code), CC BY-NC 4.0 (weights)"),
    Tool(name="MediaPipe Face Landmarker", license="Apache-2.0"),
    Tool(name="BiRefNet-portrait matting", license="MIT"),
)


class LamError(GuidedError):
    """A LAM build cannot run now, with the reason."""


class Busy(GuidedError):
    """The candidate's head is already being built."""


def host_free_gb(meminfo: Path = Path("/proc/meminfo")) -> float | None:
    """The host's available memory in GiB, or None where it cannot be read."""
    try:
        found = re.search(r"^MemAvailable:\s+(\d+) kB", meminfo.read_text(encoding="ascii", errors="replace"), re.M)
    except OSError:
        return None
    return int(found.group(1)) / (1 << 20) if found else None


def _tree_sha256(directory: Path) -> str:
    """The pipeline's own files, as one hash: every script and its data under ``lam/``."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".sh", ".json") and "__pycache__" not in path.parts:
            digest.update(path.relative_to(directory).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _lines(text: str, count: int) -> list[str]:
    """The last *count* lines that say something, without terminal colours, each cut to a length a record can
    carry. A progress bar's redraws (carriage returns) are lines of their own, so only its last state shows."""
    return [line.strip()[:300] for line in _ANSI.sub("", text).splitlines() if line.strip()][-count:]


@dataclass
class LamJob:
    id: str
    build_id: str
    candidate: str
    fit_shape: bool = False
    state: str = "queued"  # queued, running, done, failed, stopped
    stage: str = "waiting for its turn"
    error: str | None = None
    status: str | None = None  # the build step's status in the record, once it is written
    cached: bool = False
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
            "fit_shape": self.fit_shape,
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "status": self.status,
            "cached": self.cached,
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


@dataclass(frozen=True)
class Pipeline:
    key: str
    tools: list[Tool]


@dataclass(frozen=True)
class Ran:
    """How the script's run ended: *why* when this module stopped it."""

    code: int
    said: str
    seconds: float
    why: str | None = None


class LamBuilds:
    """The LAM builds of the guided faces under one store. *free_gb* reads the host's free memory (a test's own)."""

    def __init__(
        self,
        store: GuidedStore,
        settings: LamSettings,
        gpu: Any = None,  # noqa: ANN401 - a cortex GpuCoordinator
        *,
        free_gb: Callable[[], float | None] = host_free_gb,
    ) -> None:
        self.store = store
        self.settings = settings
        self.gpu = gpu if gpu is not None else NullCoordinator()
        self.free_gb = free_gb
        self.jobs: dict[str, LamJob] = {}
        self._one = asyncio.Lock()  # one head at a time: each takes the GPU and several GB of the host's memory
        self._flame: tuple[tuple[str, int, int], str] | None = None

    # ---- whether a build can run ------------------------------------------------------------------

    def check(self) -> str | None:
        """Why no build can run on this host, or None. The image is asked for when a build runs."""
        s = self.settings
        if s.script is None or not s.script.is_file():
            return (
                "lam/make_heads.sh was not found: lam/ is not part of the package "
                "(set ARGUS_MASK_LAM_SCRIPT to a checkout's)"
            )
        if s.lam_dir is None:
            return "no LAM checkout: set ARGUS_MASK_LAM_DIR (lam/README.md)"
        if not (s.lam_dir / "model_zoo").is_dir():
            return f"{s.lam_dir} is not a LAM checkout with its model_zoo (ARGUS_MASK_LAM_DIR)"
        if s.flame_open is None:
            return "no FLAME 2023 Open: set ARGUS_MASK_FLAME_OPEN to flame2023_Open.pkl (lam/README.md)"
        if not s.flame_open.is_file():
            return f"no FLAME 2023 Open at {s.flame_open} (ARGUS_MASK_FLAME_OPEN)"
        if shutil.which(s.docker) is None:
            return f"docker not found: {s.docker!r}"
        if shutil.which(s.blender) is None:
            return f"Blender not found: {s.blender!r} (set ARGUS_MASK_LAM_BLENDER)"
        return None

    def memory(self) -> tuple[Check, float]:
        """The host has the memory a build needs (:class:`LamError` if not, or if it cannot be read): the check
        that passed, and what was free."""
        need, free = self.settings.min_free_gb, self.free_gb()
        if free is None:
            raise LamError(f"this host's free memory could not be read, and a LAM build needs {need:g} GB free")
        if free < need:
            raise LamError(
                f"{free:.1f} GB of host memory is free and a LAM build needs {need:g} GB: "
                "a build on a host short of memory has taken the machine down"
            )
        # What was free is how the run went (run), not what the step found: the same head has the same digest.
        return Check(name="memory", verdict="passed", measure={"needed_gb": need}), free

    def state(self) -> dict:
        reason = self.check()
        return {"ready": reason is None, "reason": reason, "jobs": [j.public() for j in self.jobs.values() if j.live]}

    # ---- jobs -------------------------------------------------------------------------------------

    def start(self, build_id: str, candidate: str, *, fit_shape: bool = False) -> LamJob:
        """Queue *candidate*'s head. Refused, with the reason, before anything runs: no such build, a step it needs
        not settled (:class:`NotReady`), no portrait, already building (:class:`Busy`), or a host that cannot
        build now (:class:`LamError`)."""
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        if blocked := record.blockers("build", candidate):
            raise NotReady("build cannot run: " + "; ".join(blocked))
        self._portrait(record.get("source", candidate), build_id, candidate)
        if any(j.live and (j.build_id, j.candidate) == (build_id, candidate) for j in self.jobs.values()):
            raise Busy(f"{candidate}'s head is already being built")
        if (why := self.check()) is not None:
            raise LamError(why)
        self.memory()
        job = LamJob(id=uuid.uuid4().hex[:12], build_id=build_id, candidate=candidate, fit_shape=fit_shape)
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if not j.live][:-JOBS_KEEP]:
            self.jobs.pop(old.id, None)

        def settled(task: asyncio.Task) -> None:
            # A cancelled run catches nothing: it must not stay live, which would refuse the candidate for good.
            if job.live:
                job.state, job.stage = "stopped", "stopped"
            job.ended = job.ended or time.monotonic()

        job.task = asyncio.create_task(self._run(job))
        job.task.add_done_callback(settled)
        return job

    async def stop(self, job_id: str) -> LamJob | None:
        """Stop a build: its container is killed, and nothing is written to the record."""
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

    async def _run(self, job: LamJob) -> None:
        try:
            async with self._one:
                job.state = "running"
                result = await self._build(job)
                # put() holds the record's rule again: a step this ran on that changed meanwhile refuses it.
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
            log.warning("lam_build_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("lam_build", build=job.build_id, candidate=job.candidate, state=job.state, cached=job.cached)

    # ---- one build --------------------------------------------------------------------------------

    def _portrait(self, source: StepResult | None, build_id: str, candidate: str) -> tuple[Artifact, Path]:
        """The portrait the source step kept, where the build keeps it, unchanged since."""
        found = next((a for a in source.outputs if a.name == PORTRAIT and a.file), None) if source else None
        if found is None:
            raise GuidedError(f"{candidate} has no portrait: its source step keeps one")
        path = self.store.path(build_id, found.file)
        if not path.is_file():
            raise GuidedError(f"{candidate}'s portrait is not where its source kept it")
        if sha256_file(path) != found.sha256:
            raise GuidedError(f"{candidate}'s portrait changed since its source was recorded: run the source again")
        return found, path

    async def _build(self, job: LamJob) -> StepResult:
        started = self.store.clock()
        record = await asyncio.to_thread(self.store.load, job.build_id)
        if blocked := record.blockers("build", job.candidate):
            raise NotReady("build cannot run: " + "; ".join(blocked))
        source = record.get("source", job.candidate)
        portrait, path = await asyncio.to_thread(self._portrait, source, job.build_id, job.candidate)
        job.stage = "checking the pipeline"
        pipeline = await asyncio.to_thread(self._pipeline)
        key = _sha256({"portrait": portrait.sha256, "pipeline": pipeline.key, "fit_shape": job.fit_shape})

        def result(checks: list[Check], run: dict, **found: Any) -> StepResult:  # noqa: ANN401
            return StepResult(
                step="build",
                candidate=job.candidate,
                checks=checks,
                inputs=[portrait],
                tools=pipeline.tools,
                data={"key": key, "fit_shape": job.fit_shape},
                run={"job": job.id, **run},
                started=started,
                finished=self.store.clock(),
                **found,
            )

        hit = await asyncio.to_thread(self._from_cache, key, job)
        if hit is not None:
            job.cached = True
            return result(hit["checks"], {"cached": True}, outputs=hit["outputs"], scores=hit["scores"])
        memory, free = self.memory()  # again: it was asked for when the job was queued, and others ran since
        work = (self.store.root / ".work" / job.id).resolve()
        try:
            ran = await self._script(job, path, work)
            job.stage = "checking the head"
            found = await asyncio.to_thread(self._collect, job, ran, work, memory, key)
        finally:
            await self._clear(work)
        found["run"]["free_gb"] = round(free, 1)
        return result(found.pop("checks"), found.pop("run"), **found)

    def _ask(self, *argv: str) -> str | None:
        """What a short command printed, or None where it failed."""
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=ASK_TIMEOUT_S, check=False)  # noqa: S603
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def _pipeline(self) -> Pipeline:
        """What pins the pipeline, as the head's tools and as one key. Refused where there is no image."""
        s = self.settings
        if (why := self.check()) is not None:
            raise LamError(why)
        assert s.script is not None and s.lam_dir is not None and s.flame_open is not None  # noqa: S101 - check()
        image = self._ask(s.docker, "image", "inspect", "--format", "{{.Id}}", s.image)
        if not image:
            raise LamError(f"the image {s.image!r} is not built, or docker cannot be asked: see lam/README.md")
        stat = s.flame_open.stat()
        seen = (str(s.flame_open), stat.st_size, stat.st_mtime_ns)
        if self._flame is None or self._flame[0] != seen:
            self._flame = (seen, sha256_file(s.flame_open))
        flame = self._flame[1]
        commit = self._ask("git", "-C", str(s.lam_dir), "rev-parse", "HEAD") or ""
        blender = (self._ask(shutil.which(s.blender) or s.blender, "--version") or "").splitlines()[:1]
        scripts = _tree_sha256(s.script.parent)
        pinned = {"image": image, "flame": flame, "lam": commit, "blender": "".join(blender), "scripts": scripts}
        tools = [
            Tool(name="argus-mask lam/ pipeline", sha256=scripts, license="MIT"),
            Tool(name=f"image {s.image}", version=image),
            Tool(name="LAM code (aigc3d/LAM)", version=commit, license="Apache-2.0"),
            # make_head.py refuses any other FLAME: this is the file it was handed, by its hash.
            Tool(name="FLAME 2023 Open", sha256=flame, license="CC BY 4.0"),
            *PARTS,
            Tool(name="Blender", version="".join(blender)[:200], license="GPL-3.0-or-later"),
        ]
        return Pipeline(key=_sha256(pinned), tools=tools)

    async def _script(self, job: LamJob, portrait: Path, work: Path) -> Ran:
        """Run the pipeline on one portrait, under the lease, watched. *work* is its scratch."""
        s = self.settings
        work.mkdir(parents=True)
        container = f"argus-mask-lam-{job.id}"
        env = {
            **os.environ,
            "LAM_DIR": str(s.lam_dir),
            "FLAME_OPEN": str(s.flame_open),
            "LAM_IMAGE": s.image,
            "BLENDER": shutil.which(s.blender) or s.blender,
            "WORK": str(work),
            "HEADS": str(work / "heads"),
            "LAM_CONTAINER": container,
        }
        for name in ("FIT_SHAPE", "FIT_DIMS", "FIT_GENERATIONS"):  # the request's, never the editor's own environment
            env.pop(name, None)
        if job.fit_shape:
            env["FIT_SHAPE"] = "1"
        began = time.monotonic()
        job.stage = "waiting for the GPU"
        try:
            async with lease(self.gpu, LAM_HEAD, f"argus-mask:lam-head:{job.build_id}:{job.candidate}"):
                job.stage = "building the head"
                proc = await asyncio.create_subprocess_exec(
                    "bash",
                    str(s.script),
                    f"{HEAD}={portrait}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    stdin=asyncio.subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,  # its own process group: stopped whole
                )
                reading = asyncio.create_task(proc.communicate())
                try:
                    why = await self._watch(job, reading, work)
                    if why is not None:
                        await self._halt(container, proc)
                    said, _ = await reading
                except asyncio.CancelledError:
                    await self._halt(container, proc)
                    reading.cancel()
                    raise
        except (GpuLeaseRefused, GpuLeaseTimeout) as exc:
            raise LamError(f"no GPU for the head yet: {exc}") from exc
        return Ran(code=proc.returncode, said=said.decode(errors="replace"), seconds=time.monotonic() - began, why=why)

    async def _watch(self, job: LamJob, reading: asyncio.Task, work: Path) -> str | None:
        """Wait for the script, looking at the host's memory and the clock. Why it must be stopped, or None."""
        s = self.settings
        deadline = time.monotonic() + s.timeout_s
        while True:
            done, _ = await asyncio.wait({reading}, timeout=max(0.0, min(WATCH_S, deadline - time.monotonic())))
            if done:
                return None
            free = self.free_gb()
            if free is not None and free < s.stop_free_gb:
                return f"stopped: host memory fell to {free:.1f} GB free (under {s.stop_free_gb:g} GB)"
            if time.monotonic() >= deadline:
                return f"stopped: it took more than {s.timeout_s:g} s"
            with contextlib.suppress(OSError):
                last = _lines((work / f"{HEAD}.log").read_text(errors="replace")[-4000:], 1)
                job.stage = f"building the head: {last[0][:120]}" if last else job.stage

    async def _docker(self, *argv: str, timeout: float) -> None:
        with contextlib.suppress(OSError, TimeoutError):
            proc = await asyncio.create_subprocess_exec(
                self.settings.docker,
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout)
            except TimeoutError:
                proc.kill()
                raise

    async def _halt(self, container: str, proc: asyncio.subprocess.Process) -> None:
        """Kill LAM's container and let the script end by itself (it gives the files back); then kill it too."""
        await self._docker("kill", container, timeout=ASK_TIMEOUT_S)
        try:
            await asyncio.wait_for(proc.wait(), HALT_WAIT_S)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            # The script may have started the container after the first kill: its client is gone, it is not.
            await self._docker("kill", container, timeout=ASK_TIMEOUT_S)

    async def _clear(self, work: Path) -> None:
        """Remove the scratch. What a killed container left as root is removed by a container, as the script does."""
        await asyncio.to_thread(shutil.rmtree, work, True)
        if work.exists():
            wipe = "rm -rf /work/* /work/.[!.]*"
            await self._docker(
                "run", "--rm", "-v", f"{work}:/work", CLEANER_IMAGE, "sh", "-c", wipe, timeout=CLEAR_TIMEOUT_S
            )
            await asyncio.to_thread(shutil.rmtree, work, True)
        if work.exists():
            log.warning("lam_scratch_left", path=str(work))

    def _collect(self, job: LamJob, ran: Ran, work: Path, memory: Check, key: str) -> dict:
        """What the run made, as the build step's checks, outputs and scores; and how it went."""
        try:
            text = (work / f"{HEAD}.log").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        kept = f"candidates/{job.candidate}"
        run: dict = {"seconds": round(ran.seconds, 1), "log_tail": _lines(text, LOG_TAIL)}
        if text:
            self.store.keep(job.build_id, "log", f"{kept}/build.log", text.encode())
            run["log"] = f"{kept}/build.log"
        if peak := re.search(r"^DONE .*peak VRAM GB ([0-9.]+)", text, re.M):
            run["peak_vram_gb"] = round(float(peak.group(1)), 1)
        checks, outputs, scores = [memory], [], {}
        packed = work / "heads" / HEAD / f"{HEAD}.zip"
        refusals = [line.strip()[:300] for line in text.splitlines() if REFUSALS.search(line)]
        if refusals and ran.why is None:
            checks.append(Check(name="pipeline", verdict="refused", reason="; ".join(refusals[:4])))
            return {"checks": checks, "run": run}
        if ran.why is not None or ran.code != 0 or "EXPORTED" not in text or not packed.is_file():
            last = "; ".join(_lines(ran.said, 2) + _lines(text, 2)) or "it made no head, and said nothing"
            why = ran.why or f"the pipeline ended with status {ran.code}: {last}"
            checks.append(Check(name="lam", verdict="failed", reason=why[:2000]))
            return {"checks": checks, "run": run}
        if CHECKED in text:
            checks.append(Check(name="pipeline", verdict="passed"))
        else:
            why = "the build's log does not show make_head.py's own checks passing"
            checks.append(Check(name="pipeline", verdict="refused", reason=why))
        checks.append(Check(name="lam", verdict="passed"))
        try:
            report = inspect_splat(packed)
        except FaceError as exc:
            checks.append(Check(name="head", verdict="refused", reason=str(exc).replace(str(packed), f"{HEAD}.zip")))
            return {"checks": checks, "run": run}
        measure = {"splats": float(report.splats or 0), "shapes": float(len(report.present))}
        checks.append(Check(name="head", verdict="passed", measure=measure))
        files = {"head": (packed, f"{HEAD}.zip")}
        fit = work / f"out-{HEAD}" / "fit"
        for name, file in (("shape-fit", "fit.json"), ("shape-fit-start", "start.png"), ("shape-fit-best", "best.png")):
            if (fit / file).is_file():
                files[name] = (fit / file, f"fit/{file}")
        if fitted := re.search(r"^FIT (\{.*\})\s*$", text, re.M):
            with contextlib.suppress(ValueError, KeyError, TypeError):
                found = json.loads(fitted.group(1))
                totals = {"shape_misfit_start": found["start"]["total"], "shape_misfit": found["best"]["total"]}
                if all(isinstance(v, (int, float)) and math.isfinite(v) for v in totals.values()):
                    scores = {name: round(float(value), 5) for name, value in totals.items()}
        if job.fit_shape and not scores:
            why = "the shape fit gave no score: MediaPipe found no face in the head's render, or the fit did not run"
            checks.append(Check(name="shape-fit", verdict="advisory", reason=why))
        for name, (path, file) in files.items():
            outputs.append(self.store.keep(job.build_id, name, f"{kept}/{file}", path.read_bytes()))
        if all(c.verdict == "passed" for c in checks):
            self._to_cache(key, files, checks, scores)
        return {"checks": checks, "run": run, "outputs": outputs, "scores": scores}

    # ---- the heads kept: the same portrait through the same pipeline is the same head ---------------

    @property
    def cache_dir(self) -> Path:
        return self.store.root / ".cache" / "lam"

    def _to_cache(self, key: str, files: dict[str, tuple[Path, str]], checks: list[Check], scores: dict) -> None:
        if self.settings.cache <= 0:
            return
        target = self.cache_dir / key
        new = self.cache_dir / f".new-{uuid.uuid4().hex}"
        try:
            new.mkdir(parents=True)
            listed = {}
            for name, (path, file) in files.items():
                (new / file).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, new / file)
                listed[name] = {"file": file, "sha256": sha256_file(new / file)}
            kept = {"files": listed, "checks": [c.model_dump(mode="json") for c in checks], "scores": scores}
            (new / "result.json").write_text(json.dumps(kept, indent=1), encoding="utf-8")
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            new.rename(target)
        except OSError as exc:  # a cache that cannot be written costs a rebuild, never the head just made
            log.warning("lam_cache_not_written", reason=str(exc))
        finally:
            shutil.rmtree(new, ignore_errors=True)
        entries = sorted(
            (p for p in self.cache_dir.iterdir() if p.is_dir() and not p.name.startswith(".")),
            key=lambda p: p.stat().st_mtime,
        )
        for old in entries[: -self.settings.cache]:
            shutil.rmtree(old, ignore_errors=True)

    def _from_cache(self, key: str, job: LamJob) -> dict | None:
        """A head kept under *key*, copied into the build: its checks, outputs and scores. None if there is none,
        or what is kept no longer matches what was written (it is dropped, and the head is built again)."""
        entry = self.cache_dir / key
        if self.settings.cache <= 0 or not entry.is_dir():
            return None
        try:
            kept = json.loads((entry / "result.json").read_text(encoding="utf-8"))
            data = {}
            for name, listed in kept["files"].items():
                data[name] = (listed["file"], (entry / listed["file"]).read_bytes())
                if hashlib.sha256(data[name][1]).hexdigest() != listed["sha256"]:
                    raise ValueError(f"{listed['file']} is not what was kept")
            checks = [Check.model_validate(c) for c in kept["checks"]]
            scores = {str(k): float(v) for k, v in kept["scores"].items()}
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            log.warning("lam_cache_dropped", key=key, reason=str(exc))
            shutil.rmtree(entry, ignore_errors=True)
            return None
        outputs = [
            self.store.keep(job.build_id, name, f"candidates/{job.candidate}/{file}", blob)
            for name, (file, blob) in data.items()
        ]
        with contextlib.suppress(OSError):
            os.utime(entry)  # read lately: the last to go
        return {"checks": checks, "outputs": outputs, "scores": scores}
