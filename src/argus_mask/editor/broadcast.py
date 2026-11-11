"""Broadcasts from the editor's Broadcast page: ``say`` and ``push``, run here as the CLI runs them.

A job is one broadcast to one face. It runs the client's own ``say`` (text through a TTS
service) or ``push`` (an audio file) against the server's source socket, so the server's
read-only switch, its guards and its one-source-per-face rule all apply, exactly as from a
terminal. What happens is kept as a list of events for the page to show: the session, each
sentence, the speech's length, done or the reason it failed. Stopping a job cancels it, which
cuts the TTS socket as Ctrl-C does in the CLI.

The text goes to the TTS service, and is never logged. A job's summary (its first 80
characters) is kept for the page to show, so anyone who can read this editor can read it.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from argus_mask import client
from argus_mask import record as recorder

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
    ended: float | None = None
    task: asyncio.Task | None = None
    files: list[str] = field(default_factory=list)  # recordings written, by name in the recordings directory

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
            "files": self.files,
            "seconds": round((self.ended or time.monotonic()) - self.started, 1),
        }


@dataclass
class RecordOptions:
    """Record the broadcast to a file, as `argus-mask record` does: listen, then render and mux."""

    width: int = 1280
    height: int = 720
    quality: str = "high"

    @property
    def display(self) -> str:
        return "hld" if (self.width, self.height) == (1080, 1920) else "flat"


def recording_ready() -> str | None:
    """Why the editor cannot record, or None: rendering needs Playwright (and its browser), muxing ffmpeg."""
    try:
        import playwright  # noqa: F401, PLC0415
    except ImportError:
        return "recording needs argus-mask[record] (Playwright) and `playwright install chromium-headless-shell`"
    try:
        recorder.find_ffmpeg()
    except recorder.RecordError as exc:
        return str(exc)
    return None


LISTEN_TIMEOUT_S = 15.0  # how long the recorder may take to start listening before the broadcast goes without it


class Busy(RuntimeError):
    """This face is already broadcasting from here."""


class Broadcasts:
    def __init__(self, server_url: str, recordings_dir: Path | None = None) -> None:
        self.server_url = server_url
        self.recordings_dir = Path(recordings_dir or "recordings")
        self.jobs: dict[str, Job] = {}

    def running_for(self, face: str) -> Job | None:
        return next((j for j in self.jobs.values() if j.face == face and j.state == "running"), None)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def _recorded(self, run, face: str, options: RecordOptions):  # noqa: ANN001, ANN202 - wraps a job's run
        """*run*, recorded: the recorder listens on the face first, the broadcast goes, then the clip renders."""

        async def with_recording(job: Job) -> dict:
            listening = asyncio.Event()
            name = f"{face}-{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            last = [-1]

            def on_wait() -> None:
                job.note("record", "recording: listening on the face")
                listening.set()

            def on_progress(i: int, n: int) -> None:
                if i == n or i * 4 // n > last[0]:  # every quarter, not every frame
                    last[0] = i * 4 // n
                    job.note("record", f"rendering {i}/{n} frames")

            rec = asyncio.create_task(
                recorder.record(
                    self.server_url,
                    face,
                    self.recordings_dir / name,
                    width=options.width,
                    height=options.height,
                    display=options.display,
                    quality=options.quality,
                    on_wait=on_wait,
                    on_captured=lambda r: job.note("record", f"captured {len(r.frames)} frames; rendering the clip"),
                    on_progress=on_progress,
                    on_renderer=lambda r: job.note("record", f"rendering on {r}"),
                )
            )
            try:
                heard = asyncio.create_task(listening.wait())
                try:
                    done, _ = await asyncio.wait([rec, heard], timeout=LISTEN_TIMEOUT_S, return_when="FIRST_COMPLETED")
                finally:
                    heard.cancel()
                if rec in done:  # the recorder ended before it listened: say why, then do not broadcast blind
                    rec.result()
                if not listening.is_set():
                    raise recorder.RecordError(f"the recorder did not start listening within {LISTEN_TIMEOUT_S:g}s")
                result = await run(job)
            except BaseException:
                rec.cancel()
                raise
            try:
                await rec
            except recorder.RecordError as exc:  # the broadcast went out; only the file failed
                job.note("error", f"recording failed: {exc}")
                return result
            job.files.append(name)
            job.note("file", name)
            log.info("broadcast_recorded", face=face, file=name)
            return result

        return with_recording

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
            except (
                client.ClientError,
                recorder.RecordError,
            ) as exc:  # the server's, the TTS's or the recorder's reason
                job.state = "failed"
                job.note("error", str(exc))
            except Exception as exc:  # noqa: BLE001 - whatever it was, the page must hear of it
                job.state = "failed"
                job.note("error", f"{type(exc).__name__}: {exc}")
                log.warning("broadcast_failed", face=face, kind=kind, reason=str(exc))
            finally:
                job.ended = time.monotonic()

        job.task = asyncio.create_task(body())
        return job

    def say(
        self,
        face: str,
        tts_url: str,
        body: dict,
        *,
        seed: int,
        stream: bool,
        pause_ms: int,
        record: RecordOptions | None = None,
    ) -> Job:
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

        return self._start(face, "say", summary, self._recorded(run, face, record) if record else run)

    def push(
        self, face: str, mono: np.ndarray, rate: int, *, seed: int, name: str, record: RecordOptions | None = None
    ) -> Job:
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

        return self._start(face, "push", name, self._recorded(run, face, record) if record else run)

    def stop(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.state != "running" or job.task is None:
            return False
        job.task.cancel()
        return True

    async def stop_all(self) -> None:
        """Cancel every running job and wait for each to end: on shutdown, so no TTS socket outlives the app."""
        tasks = [j.task for j in self.jobs.values() if j.state == "running" and j.task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
