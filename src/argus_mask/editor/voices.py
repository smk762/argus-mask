"""The Voices page's state: checked clips, the voices made from them, and generated candidates.

A clip is checked (`argus_mask.voice.analyse`) as it is uploaded, and staged under
``<voices>/.staged/<sha256>.wav``, so making a voice names clips by hash and nothing is sent
twice. Making a voice sends the chosen clips and their transcript to the TTS service's
``/clone-voice`` (Miso's contract) and writes ``<voices>/<id>/voice.json``: where the voice
came from, the clips by hash, the delivery, and the check's verdict. The clips are kept beside
it. Like a face, a voice carries its provenance: generated (no real person), the user's own,
or a consenting speaker's under a release (by the release's hash, never the document).

A generated candidate is a TTS service's own speaker (or a described voice, on a service that
designs them) reading a script: a voice no real person owns, with a transcript known exactly.
Generating one takes a minute or so on an RTX 3090, so it runs as a job the page polls.

The texts sent (scripts, test lines) go to the TTS service and are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import shutil
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from argus_mask import client
from argus_mask.audio import PcmDecoder
from argus_mask.voice import analyse

log = structlog.get_logger(__name__)

VOICE_ID = r"^[a-z0-9][a-z0-9-]{0,62}$"
SOURCES = ("generated", "own", "consented")
DELIVERIES = ("neutral", "warm", "bright", "calm", "brisk", "serious", "playful", "other")
STAGED_KEEP = 60  # checked clips kept for the page to use; the oldest go first
JOBS_KEEP = 20
PAD_S = 0.2  # quiet put either side of a generated candidate, as a recorded clip should have


def wav_bytes(samples: np.ndarray, rate: int) -> bytes:
    """Mono float samples as a 16-bit WAV file."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((np.clip(samples, -1.0, 32767 / 32768) * 32768).round().astype("<i2").tobytes())
    return buffer.getvalue()


def pcm_to_float(data: bytes, fmt) -> np.ndarray:  # noqa: ANN001 - an AudioFormat
    """Whole PCM in the service's format as mono float32, as the server decodes a source's audio."""
    return PcmDecoder(fmt).feed(data)


class VoiceError(ValueError):
    """A request the Voices page cannot carry out, with the reason."""


class VoiceStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.staged = self.root / ".staged"

    # -- staged clips --

    def stage(self, data: bytes) -> str:
        """Keep an uploaded (or generated) WAV by its hash; returns the hash."""
        sha = hashlib.sha256(data).hexdigest()
        self.staged.mkdir(parents=True, exist_ok=True)
        path = self.staged / f"{sha}.wav"
        if not path.exists():
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        path.touch()
        old = sorted(self.staged.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[STAGED_KEEP:]
        for p in old:
            p.unlink(missing_ok=True)
        return sha

    def clip(self, sha: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise VoiceError("not a clip")
        path = self.staged / f"{sha}.wav"
        if not path.is_file():
            raise VoiceError("that clip is no longer here: check it again")
        return path

    # -- voices --

    def records(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        out = []
        for path in sorted(self.root.glob("*/voice.json")):
            try:
                out.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda r: r.get("created", 0), reverse=True)

    def record(self, voice_id: str) -> dict | None:
        if not re.fullmatch(VOICE_ID, voice_id):
            return None
        path = self.root / voice_id / "voice.json"
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def write(self, record: dict, clips: list[Path]) -> None:
        folder = self.root / record["id"]
        folder.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(clips):
            shutil.copyfile(path, folder / f"clip-{i + 1}.wav")
        tmp = folder / "voice.json.part"
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(folder / "voice.json")

    def trash(self, voice_id: str) -> bool:
        folder = self.root / voice_id
        if not re.fullmatch(VOICE_ID, voice_id) or not (folder / "voice.json").is_file():
            return False
        bin_ = self.root / ".trash"
        bin_.mkdir(parents=True, exist_ok=True)
        folder.replace(bin_ / f"{voice_id}-{time.strftime('%Y%m%d-%H%M%S')}")
        return True


def provenance(source: dict) -> dict:
    """The source of a voice, checked: generated (by what), the user's own, or a consenting speaker's (under what)."""
    kind = str(source.get("kind") or "")
    if kind not in SOURCES:
        raise VoiceError(f"say where the voice comes from: one of {', '.join(SOURCES)}")
    out: dict = {"kind": kind}
    if kind == "generated":
        out["generator"] = str(source.get("generator") or "").strip()[:300]
        if not out["generator"]:
            raise VoiceError("a generated voice names what made it (the TTS model and speaker, or the description)")
    elif kind == "consented":
        release = str(source.get("release_sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", release):
            raise VoiceError("a consenting speaker's voice needs the sha256 of the signed release (never the release)")
        out["release_sha256"] = release
        out["speaker"] = str(source.get("speaker") or "").strip()[:120]
        if not out["speaker"]:
            raise VoiceError("name the consenting speaker, as the release does")
    if not source.get("confirmed"):
        raise VoiceError("confirm the recording is not of anyone who has not agreed to it")
    out["confirmed"] = True
    return out


@dataclass
class Generation:
    id: str
    state: str = "running"  # running, done, failed
    error: str | None = None
    result: dict | None = None
    started: float = field(default_factory=time.monotonic)
    task: asyncio.Task | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "state": self.state,
            "error": self.error,
            "result": self.result,
            "seconds": round(time.monotonic() - self.started, 1),
        }


class Generations:
    """Synthetic candidates: a TTS service reading a script in a voice no one owns."""

    def __init__(self, store: VoiceStore) -> None:
        self.store = store
        self.jobs: dict[str, Generation] = {}

    def start(self, tts_url: str, body: dict) -> Generation:
        if any(j.state == "running" for j in self.jobs.values()):
            raise VoiceError("a candidate is already being generated: one at a time, the GPU is shared")
        job = Generation(id=uuid.uuid4().hex[:12])
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if j.state != "running"][:-JOBS_KEEP]:
            self.jobs.pop(old.id, None)

        async def run() -> None:
            try:
                reply = await asyncio.to_thread(client.open_tts, tts_url, body)
                try:
                    data = await asyncio.to_thread(reply.read)
                finally:
                    reply.close()
                samples = pcm_to_float(data, reply.fmt)
                if not len(samples):
                    raise client.ClientError("the TTS service sent no audio")
                rate = reply.fmt.sample_rate
                pad = np.zeros(int(PAD_S * rate), np.float32)
                padded = np.concatenate([pad, samples, pad])
                # off the event loop: the editor keeps answering while the clip is written and measured
                sha = await asyncio.to_thread(self.store.stage, wav_bytes(padded, rate))
                report = await asyncio.to_thread(analyse, padded, rate, body["text"], generated=True)
                job.result = {"sha": sha, "transcript": body["text"], "report": report.public()}
                job.state = "done"
            except client.ClientError as exc:
                job.state, job.error = "failed", str(exc)
            except Exception as exc:  # noqa: BLE001 - the page must hear of it
                job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
                log.warning("voice_generate_failed", reason=str(exc))

        job.task = asyncio.create_task(run())
        return job

    async def stop_all(self) -> None:
        tasks = [j.task for j in self.jobs.values() if j.task is not None and not j.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
