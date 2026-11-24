"""The build record of a guided face (#163, #166): what each step was given, what it made and how it scored.

A guided face is made in steps: a source portrait with its provenance, the checks on it, the LAM build, the evals
of the head, the pick among the candidates, the background and shot, the save. Each step writes one result here:
its inputs and outputs by sha256, the models it used, its checks with their reasons, its scores. So a head can be
made again, its provenance is complete, and the page shows any step without running it again.

The record keeps three rules itself, so no route has to remember them:

- **A step runs only on settled ground.** :meth:`GuidedBuild.put` refuses a result while a step it needs is
  missing, stale, failed, refused, not checked, or carries an advisory nobody overruled.
- **A refusal is never passed.** A result's standing is derived from its checks and never stored. Only an advisory
  (or a check that could not run and is not required) can be overruled, and the record keeps what was overruled,
  why and when. A result cannot arrive already overruled.
- **A result goes stale by itself.** Each result carries the digest of every result it ran on. Run an earlier step
  again and find something different, and everything after it reads ``stale`` until it is run again; find the
  same, and nothing after it changes.

Kept as ``<assets>/.guided/<id>/build.json``, beside the build's work files. The server skips dot-directories, so
a half-made face is never listed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from argus_cortex.fs import PathRefused, safe_join
from argus_cortex.wire import make_versioned_base
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from argus_mask.models import FACE_ID_PATTERN, SHA256_PATTERN, is_face_id

RECORD_VERSION = "1.0"  # 1.0: the first record.
SUPPORTED_RECORD_MAJORS: tuple[str, ...] = ("1",)
RECORD_NAME = "build.json"
GUIDED_DIR = ".guided"

_NAME = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
# A work file's path inside its build's directory: plain segments, so never `..`, absolute or hidden.
_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
FILE_PATTERN = rf"^{_SEGMENT}(/{_SEGMENT}){{0,3}}$"

T = TypeVar("T")


class GuidedError(ValueError):
    """Something a guided build cannot do, with the reason."""


class NoSuchBuild(GuidedError):
    """There is no build under that id."""


class NotReady(GuidedError):
    """A step was asked to run on ground that is not settled."""


class CannotOverrule(GuidedError):
    """Only an advisory can be overruled."""


# ---- the steps --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One step of the flow. ``per`` says how many results it has: one for each ``candidate`` (a portrait and
    the head made from it), one for the build that names the candidate it chose (``pick``), or one for the
    ``face``. ``needs`` are the steps that must be settled before it runs: a candidate's own for a candidate step
    and for the pick, the face's for a face step."""

    name: str
    per: Literal["candidate", "pick", "face"]
    needs: tuple[str, ...]
    about: str


STEPS: dict[str, Step] = {
    step.name: step
    for step in (
        Step("source", "candidate", (), "the portrait and where it came from"),
        Step("portrait", "candidate", ("source",), "the checks on the portrait, before any GPU time"),
        Step("build", "candidate", ("source", "portrait"), "the LAM head made from the portrait"),
        Step("likeness", "candidate", ("build",), "how close the head is to its portrait"),
        Step("rest", "candidate", ("build",), "the rest set fitted to the portrait"),
        Step("turns", "candidate", ("build",), "how far the head turns before it gives out"),
        Step("speech", "candidate", ("build", "rest"), "the mouth in speech, against a reference head"),
        Step("age", "candidate", ("build",), "the head's apparent age against its portrait's"),
        Step("pick", "pick", ("likeness", "rest", "turns", "speech", "age"), "the candidate chosen"),
        Step("scene", "face", ("pick",), "the background and the shot"),
        Step("save", "face", ("pick", "scene"), "the face written where the server serves it"),
    )
}

Verdict = Literal["passed", "advisory", "refused", "failed", "not_checked"]
# What a later step may run on. Anything else blocks it.
SETTLED: frozenset[str] = frozenset({"passed", "overruled"})

Finite = Annotated[float, Field(allow_inf_nan=False)]  # a score is a number: no face found is a check, not inf


def step_of(name: str) -> Step:
    try:
        return STEPS[name]
    except KeyError:
        raise GuidedError(f"{name!r} is not a step: {', '.join(STEPS)}") from None


# ---- a step's result --------------------------------------------------------------------------------


class Check(BaseModel):
    """One thing a step checked, and what it found.

    ``advisory`` warns, and may be overruled. ``refused`` may not, nor may ``failed`` (the check could not
    finish: a crash, a timeout). ``not_checked`` is a check that could not run, a service being down: it is never
    a pass. Where the check is ``required`` it blocks as a refusal does; where it is not, it stands as an advisory.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=_NAME)
    verdict: Verdict
    reason: str = Field("", max_length=2000)
    measure: dict[str, Finite] = Field(default_factory=dict, description="the numbers behind the verdict")
    required: bool = Field(True, description="read for not_checked: whether the step can go on without it")

    @model_validator(mode="after")
    def _says_why(self) -> Check:
        if self.verdict != "passed" and not self.reason.strip():
            raise ValueError(f"{self.name}: a check that did not pass says why")
        return self

    @property
    def overrulable(self) -> bool:
        return self.verdict == "advisory" or (self.verdict == "not_checked" and not self.required)


class Overrule(BaseModel):
    """A person going on past an advisory: which check, why, and when."""

    model_config = ConfigDict(extra="forbid")

    check: str = Field(..., pattern=_NAME)
    reason: str = Field(..., min_length=1, max_length=2000)
    at: AwareDatetime


class Artifact(BaseModel):
    """Something a step was given or made, by its sha256. ``file`` is where the build keeps it, if it does."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=_NAME)
    sha256: str = Field(..., pattern=SHA256_PATTERN)
    file: str = Field("", pattern=rf"^$|{FILE_PATTERN}")
    bytes: int | None = Field(None, ge=0)


class Tool(BaseModel):
    """A model, a service or a program a step used, with what pins it and its licence."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=300)
    version: str = Field("", max_length=200)
    sha256: str | None = Field(None, pattern=SHA256_PATTERN)
    license: str = Field("", max_length=200)


class StepResult(BaseModel):
    """What one step found: for one candidate, or for the face.

    ``overruled`` and ``after`` are the record's to write (:meth:`GuidedBuild.overrule`, :meth:`GuidedBuild.put`),
    never the step's.
    """

    model_config = ConfigDict(extra="forbid")

    step: str
    candidate: str | None = Field(None, pattern=FACE_ID_PATTERN)
    checks: list[Check] = Field(default_factory=list)
    overruled: list[Overrule] = Field(default_factory=list)
    inputs: list[Artifact] = Field(default_factory=list)
    outputs: list[Artifact] = Field(default_factory=list)
    tools: list[Tool] = Field(default_factory=list)
    seeds: dict[str, int] = Field(default_factory=dict)
    scores: dict[str, Finite] = Field(default_factory=dict)
    data: dict[str, JsonValue] = Field(default_factory=dict, description="what else the step decided or made")
    run: dict[str, JsonValue] = Field(
        default_factory=dict, description="how the run went: its timings, a log's tail. Not part of what it found"
    )
    started: AwareDatetime | None = None
    finished: AwareDatetime | None = None
    after: dict[str, str] = Field(default_factory=dict, description="each step this ran on, by its result's digest")

    @model_validator(mode="after")
    def _sound(self) -> StepResult:
        if self.step not in STEPS:
            raise ValueError(f"{self.step!r} is not a step: {', '.join(STEPS)}")
        if STEPS[self.step].per == "face" and self.candidate is not None:
            raise ValueError(f"{self.step} is the face's step, not a candidate's")
        if STEPS[self.step].per != "face" and self.candidate is None:
            raise ValueError(f"{self.step} names its candidate")
        for what, names in (
            ("check", [c.name for c in self.checks]),
            ("input", [a.name for a in self.inputs]),
            ("output", [a.name for a in self.outputs]),
            ("overrule", [o.check for o in self.overruled]),
        ):
            if len(set(names)) != len(names):
                raise ValueError(f"{self.step}: each {what} is named once")
        checks = {c.name: c for c in self.checks}
        for over in self.overruled:
            found = checks.get(over.check)
            if found is None:
                raise ValueError(f"{self.step}: {over.check} is overruled, and is not one of its checks")
            if not found.overrulable:
                raise ValueError(f"{self.step}: {over.check} is {found.verdict}, and only an advisory is overruled")
        return self

    @property
    def standing(self) -> str:
        """Where the result stands by its own checks: the worst of them. Staleness is the record's to say."""
        if any(c.verdict == "failed" for c in self.checks):
            return "failed"
        if any(c.verdict == "refused" for c in self.checks):
            return "refused"
        if any(c.verdict == "not_checked" and c.required for c in self.checks):
            return "not_checked"
        over = {o.check for o in self.overruled}
        if any(c.overrulable and c.name not in over for c in self.checks):
            return "advisory"
        return "overruled" if over else "passed"

    @property
    def digest(self) -> str:
        """What the step found, as one hash: not when or how it ran, what it ran on, or what a person overruled."""
        found = self.model_dump(mode="json", exclude={"started", "finished", "run", "overruled", "after"})
        return hashlib.sha256(json.dumps(found, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def named(self, *verdicts: str) -> str:
        """The checks with one of *verdicts*, each with its reason."""
        return "; ".join(f"{c.name}: {c.reason}" for c in self.checks if c.verdict in verdicts)


# ---- the record -------------------------------------------------------------------------------------

_Record = make_versioned_base(
    "record_version", RECORD_VERSION, SUPPORTED_RECORD_MAJORS, label="record_version", class_name="_Record"
)


class GuidedBuild(_Record):
    """``build.json``: every step's result for one guided face."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=FACE_ID_PATTERN)
    name: str = Field("", max_length=200)
    created: AwareDatetime
    updated: AwareDatetime
    results: list[StepResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_result_each(self) -> GuidedBuild:
        keys = [(r.step, r.candidate if STEPS[r.step].per == "candidate" else None) for r in self.results]
        if len(set(keys)) != len(keys):
            raise ValueError("a step has one result: for each candidate, or for the face")
        return self

    # -- reading --

    def get(self, step: str, candidate: str | None = None) -> StepResult | None:
        """A step's result: *candidate*'s for a candidate step, the one there is for the pick and the face's."""
        each = step_of(step).per == "candidate"
        return next((r for r in self.results if r.step == step and (not each or r.candidate == candidate)), None)

    @property
    def picked(self) -> str | None:
        pick = self.get("pick")
        return pick.candidate if pick else None

    def candidates(self) -> list[str]:
        """Every candidate with a result, in the order they first came."""
        return list(dict.fromkeys(r.candidate for r in self.results if r.candidate and r.step != "pick"))

    def _upstream(self, step: str, candidate: str | None) -> list[tuple[str, str | None]]:
        """Each step *step* needs, with the candidate whose result it is (None for the pick and the face's)."""
        return [(need, candidate if STEPS[need].per == "candidate" else None) for need in step_of(step).needs]

    def _stale(self, result: StepResult) -> bool:
        """Whether a step this result ran on has since found something else, or is itself stale."""
        for need, candidate in self._upstream(result.step, result.candidate):
            found = self.get(need, candidate)
            if found is None or result.after.get(need) != found.digest or self._stale(found):
                return True
        return False

    def status(self, step: str, candidate: str | None = None) -> str:
        """``missing``, ``stale``, or the result's own standing."""
        found = self.get(step, candidate)
        if found is None:
            return "missing"
        return "stale" if self._stale(found) else found.standing

    def blockers(self, step: str, candidate: str | None = None) -> list[str]:
        """Why *step* cannot run now (for *candidate*): each step it needs that is not settled, and why."""
        if step_of(step).per != "face" and candidate is None:
            return [f"{step} names its candidate"]
        out = []
        for need, whose in self._upstream(step, candidate):
            status = self.status(need, whose)
            if status in SETTLED:
                continue
            found = self.get(need, whose)
            if found is None:
                out.append(f"{need} has not run")
            elif status == "stale":
                out.append(f"{need} is stale: run it again")
            elif status == "advisory":
                over = {o.check for o in found.overruled}
                open_ = "; ".join(f"{c.name}: {c.reason}" for c in found.checks if c.overrulable and c.name not in over)
                out.append(f"{need} has an advisory to overrule ({open_})")
            else:
                why = {"failed": "failed", "refused": "refused", "not_checked": "could not check"}[status]
                out.append(f"{need} {why} ({found.named(status)})")
        return out

    # -- writing --

    def put(self, result: StepResult, *, at: datetime) -> StepResult:
        """Keep a step's result, in place of an earlier one. Refused unless every step it needs is settled."""
        if result.overruled:
            raise GuidedError(f"{result.step}: a result does not arrive overruled")
        if blocked := self.blockers(result.step, result.candidate):
            raise NotReady(f"{result.step} cannot run: " + "; ".join(blocked))
        ran = {}
        for need, whose in self._upstream(result.step, result.candidate):
            found = self.get(need, whose)
            if found is not None:  # always: nothing blocked it
                ran[need] = found.digest
        kept = StepResult.model_validate({**result.model_dump(), "after": ran})
        old = self.get(result.step, result.candidate)
        if old is None:
            self.results.append(kept)
        else:
            self.results[self.results.index(old)] = kept
        self.updated = at
        return kept

    def overrule(self, step: str, check: str, reason: str, *, at: datetime, candidate: str | None = None) -> StepResult:
        """Go on past an advisory of a step's result, keeping why and when. Nothing else can be overruled."""
        found = self.get(step, candidate)
        if found is None:
            raise GuidedError(f"{step} has not run")
        if self._stale(found):
            raise CannotOverrule(f"{step} is stale: run it again")
        named = next((c for c in found.checks if c.name == check), None)
        if named is None:
            raise GuidedError(f"{step} has no check {check!r}")
        if not named.overrulable:
            raise CannotOverrule(f"{step}: {check} is {named.verdict}, and only an advisory is overruled")
        if any(o.check == check for o in found.overruled):
            raise CannotOverrule(f"{step}: {check} is overruled already")
        try:
            over = Overrule(check=check, reason=reason.strip(), at=at)
        except ValidationError as exc:
            raise GuidedError(f"{step}: an overrule says why") from exc
        kept = StepResult.model_validate({**found.model_dump(), "overruled": [*found.overruled, over]})
        self.results[self.results.index(found)] = kept
        self.updated = at
        return kept

    # -- for the page --

    def _state(self, step: str, candidate: str | None) -> dict:
        return {"status": self.status(step, candidate), "blockers": self.blockers(step, candidate)}

    def summary(self) -> dict:
        """The build as the list shows it."""
        return {
            "id": self.id,
            "name": self.name,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "candidates": self.candidates(),
            "picked": self.picked,
            "face": {s.name: self.status(s.name) for s in STEPS.values() if s.per != "candidate"},
        }

    def view(self) -> dict:
        """The record with what is derived from it: each result's status and digest, and for each step that could
        run next, what stands in its way."""
        out = self.model_dump(mode="json")
        for raw, result in zip(out["results"], self.results, strict=True):
            raw["status"] = self.status(result.step, result.candidate)
            raw["digest"] = result.digest
        out["candidates"] = {
            candidate: {
                "steps": {s.name: self._state(s.name, candidate) for s in STEPS.values() if s.per == "candidate"},
                "pick": self.blockers("pick", candidate),
            }
            for candidate in self.candidates()
        }
        out["picked"] = self.picked
        out["face"] = {
            "pick": {"status": self.status("pick"), "candidate": self.picked},
            **{s.name: self._state(s.name, None) for s in STEPS.values() if s.per == "face"},
        }
        return out


def steps() -> list[dict]:
    """The steps in order, for the page."""
    return [{"name": s.name, "per": s.per, "needs": list(s.needs), "about": s.about} for s in STEPS.values()]


# ---- the store --------------------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(UTC)


def _why(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return f"{RECORD_NAME} {where}: {first['msg']}" if where else f"{RECORD_NAME}: {first['msg']}"
    return f"{RECORD_NAME}: {exc}"


LOCKS = 64


class GuidedStore:
    """The guided builds under one assets root: each a directory holding its record and its work files.

    A record is written whole under another name and renamed into place, so a reader never sees half of one.
    Changes to one build are made one at a time. *clock* gives the time written into a record (a test passes
    its own).
    """

    def __init__(self, assets_dir: str | Path, *, clock: Callable[[], datetime] = utc_now) -> None:
        self.root = Path(assets_dir) / GUIDED_DIR
        self.clock = clock
        # A fixed few, shared by id: a lock kept per id asked for would grow with every made-up one.
        self._locks = tuple(threading.Lock() for _ in range(LOCKS))

    def _lock(self, build_id: str) -> threading.Lock:
        return self._locks[hash(build_id) % LOCKS]

    def directory(self, build_id: str) -> Path:
        if not is_face_id(build_id):
            raise NoSuchBuild(f"{build_id!r} is not a build id")
        return self.root / build_id

    def load(self, build_id: str) -> GuidedBuild:
        path = self.directory(build_id) / RECORD_NAME
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise NoSuchBuild(f"there is no build {build_id!r}") from None
        except OSError as exc:
            raise GuidedError(f"{RECORD_NAME} cannot be read: {exc.strerror or exc}") from exc
        try:
            record = GuidedBuild.model_validate_json(text)
        except (ValueError, RuntimeError) as exc:  # not a record, or a record version this build does not read
            raise GuidedError(_why(exc)) from exc
        if record.id != build_id:
            raise GuidedError(f"the record's id is {record.id!r}, the directory is {build_id!r}")
        return record

    def scan(self) -> tuple[list[GuidedBuild], dict[str, str]]:
        """Every build, the latest changed first; and for each directory that holds no readable record, why."""
        found: list[GuidedBuild] = []
        refused: dict[str, str] = {}
        if not self.root.is_dir():
            return found, refused
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            try:
                found.append(self.load(directory.name))
            except GuidedError as exc:
                refused[directory.name] = str(exc)
        return sorted(found, key=lambda r: r.updated, reverse=True), refused

    def create(self, build_id: str, name: str = "") -> GuidedBuild:
        """A new, empty build. Never in place of another."""
        target = self.directory(build_id)
        now = self.clock()
        try:
            record = GuidedBuild(id=build_id, name=name, created=now, updated=now)
        except ValidationError as exc:
            raise GuidedError(_why(exc)) from exc
        self.root.mkdir(parents=True, exist_ok=True)
        work = self.root / f".new-{uuid.uuid4().hex}"
        work.mkdir()
        try:
            self._write(work, record)
            with self._lock(build_id):
                if target.exists():
                    raise GuidedError(f"there is already a build {build_id!r}")
                work.rename(target)
        finally:
            if work.exists():
                for left in work.iterdir():
                    left.unlink()
                work.rmdir()
        return record

    def change(self, build_id: str, how: Callable[[GuidedBuild, datetime], T]) -> T:
        """Read a build, change it with *how* (handed the record and the time), and write it back: all or nothing."""
        with self._lock(build_id):
            record = self.load(build_id)
            out = how(record, self.clock())
            self._write(self.directory(build_id), record)
            return out

    def put(self, build_id: str, result: StepResult, files: dict[str, bytes] | None = None) -> StepResult:
        """Keep a step's result, and with it the work *files* (by path) it made: none replaces what the build keeps
        until the record has taken the result."""
        staged: list[tuple[Path, Path]] = []
        try:
            for file, data in (files or {}).items():
                path = self.path(build_id, file)
                if not self.directory(build_id).is_dir():
                    raise NoSuchBuild(f"there is no build {build_id!r}")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
                staged.append((tmp, path))
                tmp.write_bytes(data)

            def how(record: GuidedBuild, now: datetime) -> StepResult:
                kept = record.put(result, at=now)
                for tmp, path in staged:
                    os.replace(tmp, path)
                return kept

            return self.change(build_id, how)
        finally:
            for tmp, _ in staged:
                tmp.unlink(missing_ok=True)

    @staticmethod
    def artifact(name: str, file: str, data: bytes) -> Artifact:
        """What a work file will be once :meth:`put` keeps it."""
        try:
            return Artifact(name=name, sha256=hashlib.sha256(data).hexdigest(), file=file, bytes=len(data))
        except ValidationError as exc:
            raise GuidedError(f"{name!r} is not a name for what a step keeps") from exc

    def overrule(
        self, build_id: str, step: str, check: str, reason: str, *, candidate: str | None = None
    ) -> StepResult:
        return self.change(
            build_id, lambda record, now: record.overrule(step, check, reason, at=now, candidate=candidate)
        )

    def path(self, build_id: str, file: str) -> Path:
        """Where the build keeps *file*: under its own directory, or refused."""
        directory = self.directory(build_id)
        if file == RECORD_NAME:
            raise GuidedError(f"{RECORD_NAME} is the record: written by its steps, never as a file")
        if not isinstance(file, str) or not re.fullmatch(FILE_PATTERN, file):
            raise GuidedError(f"{file!r} is not a file a build keeps")
        try:
            return safe_join(directory, file, label="file")  # resolved: a symlink in the way cannot lead out
        except PathRefused as exc:
            raise GuidedError(str(exc)) from exc

    def keep(self, build_id: str, name: str, file: str, data: bytes) -> Artifact:
        """Write a work file into the build's directory and say what it is: for a step's inputs or outputs."""
        path = self.path(build_id, file)
        if not self.directory(build_id).is_dir():
            raise NoSuchBuild(f"there is no build {build_id!r}")
        kept = self.artifact(name, file, data)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return kept

    @staticmethod
    def _write(directory: Path, record: GuidedBuild) -> None:
        tmp = directory / f".{RECORD_NAME}.{uuid.uuid4().hex}.part"
        try:
            tmp.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, directory / RECORD_NAME)
        finally:
            tmp.unlink(missing_ok=True)
