"""Broadcasts from the editor's Broadcast page: ``say`` and ``push``, run here as the CLI runs them.

A job is one broadcast to one face. It runs the client's own ``say`` (text through a TTS
service) or ``push`` (an audio file) against the server's source socket, so the server's
read-only switch, its guards and its one-source-per-face rule all apply, exactly as from a
terminal. What happens is kept as a list of events for the page to show: the session, each
sentence, the speech's length, done or the reason it failed. Stopping a job cancels it, which
cuts the TTS socket as Ctrl-C does in the CLI.

The text goes to the TTS service and nowhere else.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import structlog

from argus_mask import client

log = structlog.get_logger(__name__)

KEEP = 20  # finished jobs kept for the page to read


@dataclass
class Job:
    id: str
    face: str
    kind: str  # "say" or "push"
    summary: str  # what is being said, or which file
    state: str = "running"  # running, done, failed, stopped
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    started: float = field(default_factory=time.monotonic)
    task: asyncio.Task | None = None

    def note(self, kind: str, text: str) -> None:
        self.events.append({"at": round(time.monotonic() - self.started, 2), "kind": kind, "text": text})

    def public(self) -> dict:
        return {
            "id": self.id,
            "face": self.face,
            "kind": self.kind,
            "summary": self.summary,
            "state": self.state,
            "events": self.events,
            "result": self.result,
            "seconds": round(time.monotonic() - self.started, 1),
        }


class Busy(RuntimeError):
    """This face is already broadcasting from here."""


class Broadcasts:
    def __init__(self, server_url: str) -> None:
        self.server_url = server_url
        self.jobs: dict[str, Job] = {}

    def running_for(self, face: str) -> Job | None:
        return next((j for j in self.jobs.values() if j.face == face and j.state == "running"), None)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _start(self, face: str, kind: str, summary: str, run) -> Job:  # noqa: ANN001 - a coroutine function
        if (other := self.running_for(face)) is not None:
            raise Busy(f"{face} is already broadcasting ({other.kind}: {other.summary})")
        job = Job(id=uuid.uuid4().hex[:12], face=face, kind=kind, summary=summary)
        self.jobs[job.id] = job
        finished = [j for j in self.jobs.values() if j.state != "running"]
        for old in finished[: max(0, len(finished) - KEEP)]:
            self.jobs.pop(old.id, None)

        async def body() -> None:
            try:
                job.result = await run(job)
                job.state = "done"
                done = job.result or {}
                job.note("done", f"done: {done.get('frames')} frames, {done.get('duration_ms', 0) / 1000:.2f}s")
            except asyncio.CancelledError:
                job.state = "stopped"
                job.note("stopped", "stopped")
            except client.ClientError as exc:  # the server's or the TTS service's own reason
                job.state = "failed"
                job.note("error", str(exc))
            except Exception as exc:  # noqa: BLE001 - whatever it was, the page must hear of it
                job.state = "failed"
                job.note("error", f"{type(exc).__name__}: {exc}")
                log.warning("broadcast_failed", face=face, kind=kind, reason=str(exc))

        job.task = asyncio.create_task(body())
        return job

    def say(self, face: str, tts_url: str, body: dict, *, seed: int, stream: bool, pause_ms: int) -> Job:
        text = str(body.get("text") or "")
        summary = text if len(text) <= 80 else text[:77] + "…"

        async def run(job: Job) -> dict:
            return await client.say(
                self.server_url,
                face,
                tts_url,
                body,
                seed=seed,
                stream=stream,
                pause_ms=pause_ms,
                on_ready=lambda r: job.note("ready", f"session {r.get('session_id')}"),
                on_progress=lambda i, n: job.note("progress", f"sentence {i}/{n}"),
                on_audio=lambda secs: job.note("audio", f"speech: {secs:.2f}s, playing"),
                on_wait=lambda: job.note("wait", "waiting for the TTS service (a cold model takes a minute or two)"),
            )

        return self._start(face, "say", summary, run)

    def push(self, face: str, mono: np.ndarray, rate: int, *, seed: int, name: str) -> Job:
        async def run(job: Job) -> dict:
            job.note("audio", f"{name}: {len(mono) / rate:.2f}s, playing")
            return await client.push(
                self.server_url,
                face,
                mono,
                rate,
                seed=seed,
                on_ready=lambda r: job.note("ready", f"session {r.get('session_id')}"),
            )

        return self._start(face, "push", name, run)

    def stop(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.state != "running" or job.task is None:
            return False
        job.task.cancel()
        return True
