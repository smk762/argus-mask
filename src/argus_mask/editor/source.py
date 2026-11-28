"""The source step of a guided face (#167; step 1 of #163): a portrait with its provenance.

Four ways in. Each ends in ``candidates/<id>/portrait.png`` and a ``source`` step result whose kind is computed
from what was checked, never chosen:

- **Generate** a batch of seeds through darkroom (a job): ``synthetic`` only when the record passed and names no
  LoRA. No GPU lease here — darkroom leases for itself.
- **Pick** a record darkroom already made: listed and fetched from darkroom; LoRA → ``personal``.
- **Upload** with declared kind: ``personal``; ``consented`` (a consent record covering this use now); or
  ``synthetic`` with its darkroom record id, checked against that record's image sha256.
- **Restyle** (a job): image-to-image through darkroom into a style template. Never raises the kind.

Refusals are written into the step result (or returned before any write) with their reason.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import io
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import structlog
from argus_cortex.limits import BadInput, TooLarge, open_image, read_upload
from pydantic import ValidationError

from argus_mask.backgrounds import Darkroom, PlateError
from argus_mask.config import SourceSettings
from argus_mask.consent import ConsentStore, refusal
from argus_mask.editor.guided import Artifact, Check, GuidedError, GuidedStore, StepResult, Tool
from argus_mask.models import Generator, Provenance, is_face_id
from argus_mask.portraits import (
    CHECKPOINT,
    DEFAULT_STYLE,
    MAX_PORTRAIT_BYTES,
    MAX_PORTRAIT_PIXELS,
    PORTRAIT_FORMATS,
    TEMPLATES,
    PortraitError,
    candidate_for_seed,
    check_source_record,
    generate_body,
    kind_from_record,
    license_for_checkpoint,
    prompt_for,
    seeds,
    subject_line,
)

log = structlog.get_logger(__name__)

PORTRAIT = "portrait"
JOBS_KEEP = 20
UploadKind = Literal["personal", "synthetic", "consented"]


class SourceError(GuidedError):
    """A source portrait could not be accepted, with the reason."""


class Stopped(SourceError):
    """The job was stopped: what it had not written yet is not written."""


class Busy(GuidedError):
    """A generate or restyle job for this build (or candidate) is already running."""


def _utcnow() -> datetime:
    from argus_mask.editor.guided import utc_now

    return utc_now()


@dataclass
class SourceJob:
    id: str
    build_id: str
    kind: str  # generate | restyle
    candidate: str | None = None  # restyle: one; generate: None (many)
    state: str = "queued"
    stage: str = "waiting"
    error: str | None = None
    status: str | None = None
    candidates: list[str] = field(default_factory=list)
    asked: float = field(default_factory=time.monotonic)
    ended: float | None = None
    task: asyncio.Task | None = None
    halted: bool = False
    _gate: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def live(self) -> bool:
        return self.state in ("queued", "running")

    def halt(self) -> None:
        """Stop the job's thread writing. Cancelling the task does not stop the thread it waits on: that goes on to
        darkroom's reply. Once this returns, a write under way has finished and no other begins."""
        with self._gate:
            self.halted = True

    @contextlib.contextmanager
    def writing(self) -> Iterator[None]:
        """Around each write the job's thread makes to the build: refused once the job is stopped."""
        with self._gate:
            if self.halted:
                raise Stopped("the job was stopped")
            yield

    def public(self) -> dict:
        return {
            "id": self.id,
            "build": self.build_id,
            "candidate": self.candidate,
            "step": "source",
            "kind": self.kind,
            "state": self.state,
            "stage": self.stage,
            "error": self.error,
            "status": self.status,
            "candidates": list(self.candidates),
            "seconds": round((self.ended or time.monotonic()) - self.asked, 1),
        }


class SourceJobs:
    """Source portraits for the guided faces under one store. *darkroom* is injected so tests use a fake."""

    def __init__(
        self,
        store: GuidedStore,
        settings: SourceSettings,
        consents: ConsentStore,
        *,
        darkroom: Darkroom | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.consents = consents
        self.darkroom = darkroom or Darkroom(settings.darkroom_url, timeout=settings.timeout_s)
        self.clock = clock or _utcnow
        self.jobs: dict[str, SourceJob] = {}
        self._one = asyncio.Lock()

    def check(self) -> str | None:
        """Why generate/pick/restyle cannot talk to darkroom, or None. Upload never needs darkroom."""
        if not self.settings.darkroom_url.startswith(("http://", "https://")):
            return f"ARGUS_MASK_DARKROOM_URL {self.settings.darkroom_url!r}: must be an http(s) URL"
        return None

    def state(self) -> dict:
        reason = self.check()
        return {
            "ready": reason is None,
            "reason": reason,
            "darkroom": self.settings.darkroom_url,
            "styles": sorted(TEMPLATES),
            "jobs": [j.public() for j in self.jobs.values() if j.live],
        }

    # ---- listing / reading darkroom records -------------------------------------------------------

    def list_records(self, *, page: int = 1, page_size: int = 50) -> dict:
        if (why := self.check()) is not None:
            raise SourceError(why)
        try:
            return self.darkroom.records(outcome="passed", page=page, page_size=page_size)
        except PlateError as exc:
            raise SourceError(str(exc)) from exc

    def get_record(self, record_id: str) -> dict:
        if (why := self.check()) is not None:
            raise SourceError(why)
        try:
            return self.darkroom.record(record_id)
        except PlateError as exc:
            raise SourceError(str(exc)) from exc

    # ---- sync ways in -----------------------------------------------------------------------------

    def pick(self, build_id: str, candidate: str, record_id: str) -> StepResult:
        """Take a passed darkroom record as this candidate's portrait."""
        self._candidate(build_id, candidate)
        if (why := self.check()) is not None:
            raise SourceError(why)
        record_id = record_id.strip()
        if not record_id:
            raise SourceError("pick a darkroom record id")
        try:
            record = self.darkroom.record(record_id)
            image = self.darkroom.image(record_id)
        except PlateError as exc:
            raise SourceError(str(exc)) from exc
        try:
            kind = kind_from_record(record, image, policy=self.settings.policy)
        except PortraitError as exc:
            raise SourceError(str(exc)) from exc
        licence = "" if kind == "personal" else _licence(_checkpoint_name(record))
        return self._write_source(
            build_id,
            candidate,
            image,
            kind=kind,
            source=f"argus-darkroom record {record_id}",
            license=licence,
            record=record,
            data={"way": "pick", "record_id": record_id},
        )

    def upload(
        self,
        build_id: str,
        candidate: str,
        data: bytes,
        *,
        kind: UploadKind,
        record_id: str = "",
        subject: str = "",
        release_sha256: str | None = None,
        source: str = "",
        license: str = "",  # noqa: A002
    ) -> StepResult:
        """Accept an uploaded image as this candidate's portrait, with the declared provenance checked."""
        self._candidate(build_id, candidate)
        try:
            image = _decode_portrait(data)
        except PortraitError as exc:
            raise SourceError(str(exc)) from exc
        sha = hashlib.sha256(image).hexdigest()

        if kind == "personal":
            return self._write_source(
                build_id,
                candidate,
                image,
                kind="personal",
                source=(source.strip() or "uploaded"),
                license="",
                data={"way": "upload"},
            )

        if kind == "consented":
            if not release_sha256 or not subject.strip():
                raise SourceError("a consented upload names its subject and the sha256 of the signed release")
            prov = Provenance(
                kind="consented",
                source=source.strip() or "uploaded under release",
                license=license.strip() or "per the release",
                subject=subject.strip(),
                release_sha256=release_sha256,
            )
            found = self.consents.check_face(prov, use="avatar", at=self.clock())
            if (why := refusal(found)) is not None:
                raise SourceError(f"this portrait may not be used: {why}")
            return self._write_source(
                build_id,
                candidate,
                image,
                kind="consented",
                source=prov.source,
                license=prov.license,
                data={
                    "way": "upload",
                    "subject": prov.subject,
                    "release_sha256": prov.release_sha256,
                },
            )

        if kind == "synthetic":
            rid = record_id.strip()
            if not rid:
                raise SourceError("a synthetic upload names its darkroom record id")
            if (why := self.check()) is not None:
                raise SourceError(why)
            try:
                record = self.darkroom.record(rid)
            except PlateError as exc:
                raise SourceError(str(exc)) from exc
            try:
                found_kind = check_source_record(record, image, policy=self.settings.policy)
            except PortraitError as exc:
                raise SourceError(str(exc)) from exc
            if found_kind != "synthetic":
                raise SourceError(f"darkroom record {rid}: names a LoRA, so the face is personal, not synthetic")
            recorded = (record.get("image") or {}).get("sha256")
            if recorded != sha:
                raise SourceError(f"darkroom record {rid}: the upload does not match the record's image sha256")
            licence = _licence(_checkpoint_name(record), license)
            return self._write_source(
                build_id,
                candidate,
                image,
                kind="synthetic",
                source=source.strip() or f"argus-darkroom record {rid}",
                license=licence,
                record=record,
                data={"way": "upload", "record_id": rid},
            )

        raise SourceError(f"{kind!r} is not a provenance kind an upload may declare")

    # ---- jobs: generate / restyle -----------------------------------------------------------------

    def start_generate(
        self,
        build_id: str,
        *,
        age: int,
        hair: str = "",
        clothes: str = "",
        eyes: str = "",
        description: str = "",
        style: str = DEFAULT_STYLE,
        seed_text: str,
    ) -> SourceJob:
        self.store.load(build_id)  # NoSuchBuild
        if (why := self.check()) is not None:
            raise SourceError(why)
        try:
            subject = subject_line(age=age, hair=hair, clothes=clothes, eyes=eyes, description=description)
            prompt = prompt_for(subject, style)
            seed_list = seeds(seed_text)
            license_for_checkpoint(CHECKPOINT)
        except PortraitError as exc:
            raise SourceError(str(exc)) from exc
        for seed in seed_list:
            cand = candidate_for_seed(seed)
            if not is_face_id(cand):
                raise SourceError(f"seed {seed}: {cand!r} is not a candidate id")
        if any(j.live and j.build_id == build_id and j.kind == "generate" for j in self.jobs.values()):
            raise Busy(f"{build_id} already has a generate job running")
        job = SourceJob(id=uuid.uuid4().hex[:12], build_id=build_id, kind="generate")
        job.stage = "generating"
        self._track(job)
        job.task = asyncio.create_task(
            self._run_generate(job, prompt=prompt, style=style, subject=subject, seed_list=seed_list)
        )
        job.task.add_done_callback(self._settled(job))
        return job

    def start_restyle(
        self,
        build_id: str,
        candidate: str,
        *,
        style: str = DEFAULT_STYLE,
        seed: int | None = None,
        denoise: float | None = None,
    ) -> SourceJob:
        record = self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")
        source = record.get("source", candidate)
        if source is None or source.standing in ("failed", "refused", "not_checked"):
            raise SourceError(
                f"{candidate} has no settled source to restyle"
                + (f": {source.named('failed', 'refused', 'not_checked')}" if source else "")
            )
        portrait = next((a for a in source.outputs if a.name == PORTRAIT and a.file), None)
        if portrait is None:
            raise SourceError(f"{candidate} has no portrait to restyle")
        path = self.store.path(build_id, portrait.file)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != portrait.sha256:
            raise SourceError(f"{candidate}'s portrait changed since its source was recorded")
        if (why := self.check()) is not None:
            raise SourceError(why)
        try:
            prompt_for("an adult", style)  # style exists
        except PortraitError as exc:
            raise SourceError(str(exc)) from exc
        if any(
            j.live and j.build_id == build_id and j.candidate == candidate and j.kind == "restyle"
            for j in self.jobs.values()
        ):
            raise Busy(f"{candidate}'s restyle is already running")
        job = SourceJob(id=uuid.uuid4().hex[:12], build_id=build_id, kind="restyle", candidate=candidate)
        self._track(job)
        job.task = asyncio.create_task(
            self._run_restyle(
                job,
                source=source,
                portrait_path=path,
                portrait_sha=portrait.sha256,
                style=style,
                seed=seed if seed is not None else secrets.randbelow(2**31),
                denoise=denoise if denoise is not None else self.settings.restyle_denoise,
            )
        )
        job.task.add_done_callback(self._settled(job))
        return job

    async def stop(self, job_id: str) -> SourceJob | None:
        job = self.jobs.get(job_id)
        if job is not None and job.task is not None and not job.task.done():
            job.halt()  # before it reads as stopped: its thread writes nothing after
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        return job

    async def stop_all(self) -> None:
        jobs = [j for j in self.jobs.values() if j.task is not None and not j.task.done()]
        for job in jobs:
            job.halt()
            job.task.cancel()
        await asyncio.gather(*(j.task for j in jobs), return_exceptions=True)

    # ---- job bodies -------------------------------------------------------------------------------

    async def _run_generate(
        self, job: SourceJob, *, prompt: str, style: str, subject: str, seed_list: list[int]
    ) -> None:
        try:
            async with self._one:
                job.state, job.stage = "running", "generating"
                kept: list[str] = []
                last_standing = "passed"
                last_error: str | None = None
                for seed in seed_list:
                    candidate = candidate_for_seed(seed)
                    job.stage = f"seed {seed}"
                    result = await asyncio.to_thread(
                        self._generate_one,
                        job,
                        candidate,
                        seed=seed,
                        prompt=prompt,
                        style=style,
                        subject=subject,
                    )
                    kept.append(candidate)
                    job.candidates = list(kept)
                    if result.standing != "passed":
                        last_standing = result.standing
                        last_error = result.named("failed", "refused", "not_checked")
                job.status = last_standing
                if last_standing == "passed":
                    job.state, job.stage = "done", "done"
                else:
                    job.state, job.stage, job.error = "failed", last_standing, last_error
        except asyncio.CancelledError:
            job.halt()  # cancelled by something other than stop(): the thread still must not write
            job.state, job.stage = "stopped", "stopped"
            raise
        except GuidedError as exc:
            job.state, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.state, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            log.warning("source_generate_failed", build=job.build_id, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("source_generate", build=job.build_id, state=job.state, candidates=job.candidates)

    def _generate_one(
        self, job: SourceJob, candidate: str, *, seed: int, prompt: str, style: str, subject: str
    ) -> StepResult:
        build_id = job.build_id
        started = self.clock()
        body = generate_body(
            prompt,
            seed,
            width=self.settings.width,
            height=self.settings.height,
            policy=self.settings.policy,
        )
        try:
            record = self.darkroom.generate(body)
            image = self.darkroom.image(str(record["id"]))
            kind = check_source_record(
                record,
                image,
                policy=self.settings.policy,
                expect_prompt=prompt,
                expect_seed=seed,
                expect_checkpoint=CHECKPOINT,
            )
        except (PlateError, PortraitError, KeyError) as exc:
            why = str(exc)
            with job.writing():
                return self.store.put(
                    build_id,
                    StepResult(
                        step="source",
                        candidate=candidate,
                        checks=[Check(name="provenance", verdict="refused", reason=why[:2000])],
                        seeds={"generator": seed},
                        data={"way": "generate", "style": style, "subject": subject, "prompt": prompt},
                        started=started,
                        finished=self.clock(),
                    ),
                )
        if kind != "synthetic":
            why = "the record names a LoRA: the face would be personal, not synthetic"
            with job.writing():
                return self.store.put(
                    build_id,
                    StepResult(
                        step="source",
                        candidate=candidate,
                        checks=[Check(name="provenance", verdict="refused", reason=why)],
                        seeds={"generator": seed},
                        data={
                            "way": "generate",
                            "style": style,
                            "subject": subject,
                            "prompt": prompt,
                            "record_id": record.get("id"),
                        },
                        started=started,
                        finished=self.clock(),
                    ),
                )
        with job.writing():
            return self._write_source(
                build_id,
                candidate,
                image,
                kind="synthetic",
                source=f"argus-darkroom record {record['id']}",
                license=license_for_checkpoint(CHECKPOINT),
                record=record,
                seed=seed,
                data={
                    "way": "generate",
                    "style": style,
                    "subject": subject,
                    "prompt": prompt,
                    "record_id": record["id"],
                },
                started=started,
            )

    async def _run_restyle(
        self,
        job: SourceJob,
        *,
        source: StepResult,
        portrait_path: Any,
        portrait_sha: str,
        style: str,
        seed: int,
        denoise: float,
    ) -> None:
        assert job.candidate is not None
        try:
            async with self._one:
                job.state, job.stage = "running", "restyling"
                result = await asyncio.to_thread(
                    self._restyle_one,
                    job,
                    job.candidate,
                    source=source,
                    portrait_path=portrait_path,
                    portrait_sha=portrait_sha,
                    style=style,
                    seed=seed,
                    denoise=denoise,
                )
                job.candidates = [job.candidate]
                job.status = result.standing
                if result.standing == "passed":
                    job.state, job.stage = "done", "done"
                else:
                    job.state, job.stage = "failed", result.standing
                    job.error = result.named("failed", "refused", "not_checked")
        except asyncio.CancelledError:
            job.halt()  # cancelled by something other than stop(): the thread still must not write
            job.state, job.stage = "stopped", "stopped"
            raise
        except GuidedError as exc:
            job.state, job.stage, job.error = "failed", "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            job.state, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            log.warning("source_restyle_failed", build=job.build_id, candidate=job.candidate, reason=str(exc))
        finally:
            job.ended = time.monotonic()
        log.info("source_restyle", build=job.build_id, candidate=job.candidate, state=job.state)

    def _restyle_one(
        self,
        job: SourceJob,
        candidate: str,
        *,
        source: StepResult,
        portrait_path: Any,
        portrait_sha: str,
        style: str,
        seed: int,
        denoise: float,
    ) -> StepResult:
        build_id = job.build_id
        started = self.clock()
        # The job may have waited a long time: the portrait and its provenance must still be the ones it was asked for.
        self._same_source(build_id, candidate, source)
        picture = portrait_path.read_bytes()
        if hashlib.sha256(picture).hexdigest() != portrait_sha:
            raise SourceError(f"{candidate}'s portrait changed since the restyle was asked for: ask again")
        prior = source.data or {}
        prior_kind = str(prior.get("kind") or "personal")
        prior_license = str(prior.get("license") or "")
        prior_source = str(prior.get("source") or prior.get("way") or "source")
        # Restyle never raises the kind: keep personal/consented; synthetic stays synthetic only if the new
        # record clears.
        subject = str(prior.get("subject") or "an adult").strip() or "an adult"
        try:
            prompt = prompt_for(subject, style)
        except PortraitError:
            prompt = prompt_for("an adult", style)
        if prior_kind == "consented":
            self._consented(prior)  # before the person's picture goes anywhere
        body = generate_body(
            prompt,
            seed,
            width=self.settings.width,
            height=self.settings.height,
            policy=self.settings.policy,
            init_image={
                "data_base64": base64.b64encode(picture).decode("ascii"),
                "denoise": denoise,
            },
        )
        try:
            record = self.darkroom.generate(body)
            image = self.darkroom.image(str(record["id"]))
            new_kind = check_source_record(
                record,
                image,
                policy=self.settings.policy,
                expect_prompt=prompt,
                expect_seed=seed,
                expect_checkpoint=CHECKPOINT,
            )
        except (PlateError, PortraitError, KeyError) as exc:
            # The job's failure alone: the source it would have replaced is as good as it was.
            raise SourceError(f"restyle failed, and the source is as it was: {exc}") from exc
        # Never raise: personal/consented stay; synthetic only if the restyle also cleared without LoRA.
        if prior_kind in ("personal", "consented"):
            kind = prior_kind
        elif new_kind == "synthetic":
            kind = "synthetic"
        else:
            kind = "personal"
        licence = prior_license
        if kind == "synthetic":
            licence = license_for_checkpoint(CHECKPOINT, prior_license)
        elif kind == "consented":
            licence = prior_license or "per the release"
        # The check and the write as one: a job stopped between them writes nothing.
        with job.writing():
            self._same_source(build_id, candidate, source)
            return self._write_source(
                build_id,
                candidate,
                image,
                kind=kind,
                source=prior_source if kind != "synthetic" else f"argus-darkroom record {record['id']}",
                license=licence if kind != "personal" else "",
                record=record if kind == "synthetic" else None,
                seed=seed,
                data={
                    "way": "restyle",
                    "style": style,
                    "source_sha256": portrait_sha,
                    "prior_kind": prior_kind,
                    "record_id": record.get("id"),
                    "denoise": denoise,
                    **(
                        {
                            "subject": source.data.get("subject"),
                            "release_sha256": source.data.get("release_sha256"),
                        }
                        if kind == "consented"
                        else {}
                    ),
                },
                started=started,
                # Beside the portrait it was made from, never over it: the step's input stays where it says.
                file=f"candidates/{candidate}/restyle-{hashlib.sha256(image).hexdigest()[:16]}.png",
                inputs=[
                    Artifact(
                        name="source-portrait",
                        sha256=portrait_sha,
                        file=next((a.file for a in source.outputs if a.name == PORTRAIT and a.file), ""),
                    )
                ],
            )

    # ---- write the step ---------------------------------------------------------------------------

    def _write_source(
        self,
        build_id: str,
        candidate: str,
        image: bytes,
        *,
        kind: str,
        source: str,
        license: str,  # noqa: A002
        record: dict | None = None,
        seed: int | None = None,
        data: dict[str, Any] | None = None,
        started: datetime | None = None,
        inputs: list[Artifact] | None = None,
        file: str | None = None,
    ) -> StepResult:
        started = started or self.clock()
        kept_path = file or f"candidates/{candidate}/portrait.png"
        portrait = self.store.keep(build_id, PORTRAIT, kept_path, image)
        tools: list[Tool] = []
        generator: dict[str, Any] | None = None
        if record is not None:
            tools.append(
                Tool(
                    name="argus-darkroom",
                    version=str(record.get("darkroom_version") or ""),
                    license="MIT",
                )
            )
            asked = record.get("request") or {}
            checkpoint = _checkpoint_name(record) or CHECKPOINT
            generator = Generator(
                model=checkpoint,
                seed=asked.get("seed") if isinstance(asked.get("seed"), int) else seed,
                prompt=str(asked.get("final_prompt") or ""),
            ).model_dump(mode="json")
            # Prove the Provenance shape for synthetic/personal-with-generator.
            if kind in ("synthetic", "personal"):
                Provenance(
                    kind=kind,  # type: ignore[arg-type]
                    source=source,
                    license=license or ("personal use" if kind == "personal" else license_for_checkpoint(checkpoint)),
                    generator=Generator(
                        model=checkpoint,
                        seed=generator["seed"],
                        prompt=generator["prompt"],
                    ),
                )
        payload = {
            "way": (data or {}).get("way", "source"),
            "kind": kind,
            "source": source,
            "license": license,
            **(data or {}),
        }
        if generator is not None:
            payload["generator"] = generator
        checks = [Check(name="provenance", verdict="passed")]
        return self.store.put(
            build_id,
            StepResult(
                step="source",
                candidate=candidate,
                checks=checks,
                inputs=inputs or [],
                outputs=[portrait],
                tools=tools,
                seeds={"generator": seed} if seed is not None else {},
                data=payload,
                started=started,
                finished=self.clock(),
            ),
        )

    def _same_source(self, build_id: str, candidate: str, source: StepResult) -> None:
        now = self.store.load(build_id).get("source", candidate)
        if now is None or now.digest != source.digest:
            raise SourceError(f"{candidate}'s source changed since the restyle was asked for: ask again")

    def _consented(self, data: dict[str, Any]) -> None:
        """The consent the source was taken under covers it now, or :class:`SourceError` says why not."""
        try:
            prov = Provenance(
                kind="consented",
                source=str(data.get("source") or ""),
                license=str(data.get("license") or "per the release"),
                subject=str(data.get("subject") or ""),
                release_sha256=data.get("release_sha256"),
            )
        except ValidationError as exc:
            raise SourceError(f"this portrait may not be used: {exc.errors()[0]['msg']}") from exc
        if (why := refusal(self.consents.check_face(prov, use="avatar", at=self.clock()))) is not None:
            raise SourceError(f"this portrait may not be used: {why}")

    def _candidate(self, build_id: str, candidate: str) -> None:
        self.store.load(build_id)
        if not is_face_id(candidate):
            raise GuidedError(f"{candidate!r} is not a candidate")

    def _track(self, job: SourceJob) -> None:
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if not j.live][:-JOBS_KEEP]:
            self.jobs.pop(old.id, None)

    @staticmethod
    def _settled(job: SourceJob) -> Callable[[asyncio.Task], None]:
        def done(_: asyncio.Task) -> None:
            if job.live:
                job.state, job.stage = "stopped", "stopped"
            job.ended = job.ended or time.monotonic()

        return done


def _licence(checkpoint: str, stated: str = "") -> str:
    try:
        return license_for_checkpoint(checkpoint, stated)
    except PortraitError as exc:
        raise SourceError(str(exc)) from exc


def _checkpoint_name(record: dict) -> str:
    asked = record.get("request") or {}
    checkpoint = asked.get("checkpoint")
    if isinstance(checkpoint, dict):
        return str(checkpoint.get("name") or "")
    if isinstance(checkpoint, str):
        return checkpoint
    return ""


def _decode_portrait(data: bytes) -> bytes:
    """Open *data* under cortex's caps and re-encode as PNG (so the build keeps one format)."""
    if not data:
        raise PortraitError("the upload is empty")
    if len(data) > MAX_PORTRAIT_BYTES:
        raise PortraitError(f"send at most {MAX_PORTRAIT_BYTES // (1024 * 1024)} MB")
    try:
        image = open_image(data, max_pixels=MAX_PORTRAIT_PIXELS, max_bytes=MAX_PORTRAIT_BYTES, formats=PORTRAIT_FORMATS)
        image.load()
    except (TooLarge, BadInput, OSError, ValueError) as exc:
        raise PortraitError(str(exc)) from exc
    if getattr(image, "format", None) == "PNG" and data.startswith(b"\x89PNG"):
        return data
    buf = io.BytesIO()
    rgb = (
        image.convert("RGBA") if image.mode in ("P", "LA") else image.convert("RGB") if image.mode != "RGBA" else image
    )
    rgb.save(buf, format="PNG")
    out = buf.getvalue()
    if len(out) > MAX_PORTRAIT_BYTES:
        raise PortraitError(f"the portrait is over {MAX_PORTRAIT_BYTES // (1024 * 1024)} MB as PNG")
    return out


async def read_portrait_upload(upload: Any) -> bytes:
    """The body of a Starlette UploadFile, at most :data:`MAX_PORTRAIT_BYTES`."""
    try:
        return await read_upload(upload, MAX_PORTRAIT_BYTES, what="portrait")
    except TooLarge as exc:
        raise SourceError(str(exc)) from exc


__all__ = [
    "Busy",
    "PORTRAIT",
    "SourceError",
    "SourceJob",
    "SourceJobs",
    "UploadKind",
    "read_portrait_upload",
]
