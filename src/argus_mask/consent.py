"""Consent records: the record that decides whether a real person's face or voice may be used, and for what.

A ``consented`` face (its :class:`~argus_mask.models.Provenance`) or voice (its ``voice.json``) names a signed
release by its sha256. The :class:`ConsentRecord` for that release says what the person agreed to: their face,
their voice (and which deliveries), which uses, until when, and whether it has been revoked. It holds the
release's hash and the hashes of the reference images and recordings taken at the signing, never the documents
or the media themselves, and never any copy of the ID the person's age was checked against.

The types and :func:`consent_valid` are the definition argus-cortex#31 lifts into cortex (argus-mask#26 sets it
out), so curator and the rest refuse for the same reasons with the same words. Until then they live here.

:func:`consent_valid` is pure: the caller passes the time, so a check never reads the clock. :class:`ConsentStore`
keeps the records as ``<assets>/.consent/<release_sha256>.json``. The folder is hidden, so the face store never
takes it for a face. A record is written once by ``consent new`` and changed only by ``consent revoke``.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from argus_mask.models import FACE_DISCLOSURE as FACE_DISCLOSURE  # the labels, beside Provenance which carries one
from argus_mask.models import SHA256_PATTERN, Provenance
from argus_mask.models import VOICE_DISCLOSURE as VOICE_DISCLOSURE

ConsentUse = Literal["avatar", "video", "speech", "training"]
"""``avatar``: a live face on a screen. ``video``: ``record`` output. ``speech``: audio with no face (a voice heard
on its own). ``training``: fitting or fine-tuning a model on the captures. A use not listed is not covered."""
CONSENT_USES: tuple[str, ...] = ("avatar", "video", "speech", "training")
Likeness = Literal["face", "voice"]
ConsentRefusal = Literal[
    "revoked", "not-yet-signed", "expired", "likeness-not-covered", "use-not-covered", "delivery-not-covered"
]
DELIVERY_PATTERN = r"^[a-z][a-z0-9-]{0,31}$"

CONSENT_DIR = ".consent"


class AgeCheck(BaseModel):
    """That the person's age was checked against ID at the signing, and by whom. Never the ID or a copy of it."""

    model_config = ConfigDict(extra="forbid")

    checked_by: str = Field(..., min_length=1, max_length=120)
    checked_on: date


class Revocation(BaseModel):
    """A revoked release: when it was asked for, from when it holds, and who recorded it."""

    model_config = ConfigDict(extra="forbid")

    requested_on: date
    effective_on: date
    by: str = Field(..., min_length=1, max_length=120)

    @model_validator(mode="after")
    def _order(self) -> Revocation:
        if self.effective_on < self.requested_on:
            raise ValueError("a revocation cannot take effect before it was asked for")
        return self


class ConsentRecord(BaseModel):
    """What a real person agreed to, under one signed release (argus-mask#26).

    Free-text fields (``exclusions``, ``media``, ``territories``, ``notes``) are for people and are not checked by
    code. Payment and retention terms stay in the signed document. The disclosure label is not a field: every
    output of a consented likeness carries it, and nothing turns it off.
    """

    model_config = ConfigDict(extra="forbid")

    release_sha256: str = Field(..., pattern=SHA256_PATTERN, description="the signed release; never the document")
    subject: str = Field(..., min_length=1, max_length=120, description="the person, as the release names them")
    collected_by: str = Field(..., min_length=1, max_length=120, description="who ran the signing session")
    signed_on: date
    expires_on: date | None = Field(None, description="the first day the release no longer holds; null: no expiry")
    age_check: AgeCheck
    face: bool = False
    voice: bool = False
    deliveries: list[str] = Field(default_factory=list, description="the voice deliveries covered (neutral, warm…)")
    uses: list[ConsentUse] = Field(..., min_length=1)
    reference_images: list[str] = Field(default_factory=list, description="sha256 of each, from the signing session")
    reference_recordings: list[str] = Field(
        default_factory=list, description="sha256 of each, from the signing session"
    )
    exclusions: list[str] = Field(default_factory=list, description="beyond the release template's standing list")
    media: list[str] = Field(default_factory=list)
    territories: list[str] = Field(default_factory=list)
    revocable: bool
    notice_days: int = Field(0, ge=0, le=3650)
    on_revoke: Literal["delete", "keep-labelled"] = Field(
        ..., description="what happens to recordings already made when the release is revoked"
    )
    revocation: Revocation | None = None
    notes: str = Field("", max_length=2000)

    @model_validator(mode="after")
    def _complete(self) -> ConsentRecord:
        if not self.subject.strip() or not self.collected_by.strip():
            raise ValueError("a consent record names its subject and who collected it")
        if not (self.face or self.voice):
            raise ValueError("a release covers the face, the voice, or both")
        if self.voice and not self.deliveries:
            raise ValueError("a release for the voice names the deliveries it covers")
        if not self.voice and self.deliveries:
            raise ValueError("deliveries are for a voice: this release does not cover the voice")
        if bad := [d for d in self.deliveries if not re.fullmatch(DELIVERY_PATTERN, d)]:
            raise ValueError(f"not a delivery name: {', '.join(bad)}")
        if self.face and not self.reference_images:
            raise ValueError("a release for the face has reference images from the signing session")
        if self.voice and not self.reference_recordings:
            raise ValueError("a release for the voice has a reference recording from the signing session")
        for name in ("reference_images", "reference_recordings"):
            if bad := [h for h in getattr(self, name) if not re.fullmatch(SHA256_PATTERN, h)]:
                raise ValueError(f"{name}: not a sha256: {', '.join(bad)}")
        if self.expires_on is not None and self.expires_on <= self.signed_on:
            raise ValueError("a release expires after it is signed")
        if self.age_check.checked_on > self.signed_on:
            raise ValueError("the age check comes before the release is signed, at the latest on the day")
        # A set, kept sorted so the same record writes the same file.
        self.uses = sorted(set(self.uses))  # type: ignore[assignment]
        self.deliveries = sorted(set(self.deliveries))
        return self


class ConsentVerdict(BaseModel):
    """Whether a record covers a use, and if not, why: a closed code, and words for a person."""

    ok: bool
    reason: ConsentRefusal | None = None
    detail: str = ""


def _day(at: datetime) -> date:
    if at.tzinfo is None:
        raise ValueError("pass an aware datetime: a consent check never guesses the time zone")
    return at.date()


def consent_valid(
    record: ConsentRecord,
    *,
    use: ConsentUse | None,
    likeness: Likeness,
    at: datetime,
    delivery: str | None = None,
) -> ConsentVerdict:
    """Whether *record* covers *likeness* used for *use* at *at* (and, for a voice, *delivery*).

    The checks run in a fixed order and the first failure is the answer: revoked, not yet signed, expired, the
    likeness not covered, the use not covered, the delivery not covered. *use* None asks only whether the record
    holds for this likeness at all (making a voice, trying a line on the Voices page). For a voice, *delivery* is
    required. Pure: *at* is the caller's.
    """
    day = _day(at)
    if record.revocation is not None and day >= record.revocation.effective_on:
        return ConsentVerdict(
            ok=False, reason="revoked", detail=f"the release was revoked from {record.revocation.effective_on}"
        )
    if day < record.signed_on:
        return ConsentVerdict(ok=False, reason="not-yet-signed", detail=f"the release is signed on {record.signed_on}")
    if record.expires_on is not None and day >= record.expires_on:
        return ConsentVerdict(ok=False, reason="expired", detail=f"the release expired on {record.expires_on}")
    if not getattr(record, likeness):
        return ConsentVerdict(
            ok=False, reason="likeness-not-covered", detail=f"the release does not cover {record.subject}'s {likeness}"
        )
    if use is not None and use not in record.uses:
        return ConsentVerdict(
            ok=False,
            reason="use-not-covered",
            detail=f"the release does not cover {use} (it covers {', '.join(record.uses)})",
        )
    if likeness == "voice":
        if not delivery:
            raise ValueError("a voice's consent check names the delivery")
        if delivery not in record.deliveries:
            return ConsentVerdict(
                ok=False,
                reason="delivery-not-covered",
                detail=f"the release does not cover the {delivery} delivery (it covers {', '.join(record.deliveries)})",
            )
    return ConsentVerdict(ok=True)


def same_subject(a: str, b: str) -> bool:
    """Two spellings of a name are the same person's: case, spacing and Unicode form aside."""

    def norm(s: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", s).casefold().split())

    return bool(norm(a)) and norm(a) == norm(b)


def consent_matches(record: ConsentRecord, provenance: Provenance) -> bool:
    """Whether *provenance* is under *record*: the same release, the same subject."""
    return provenance.release_sha256 == record.release_sha256 and same_subject(provenance.subject, record.subject)


# ── Mask's store and checks ───────────────────────────────────────────────────


class ConsentError(ValueError):
    """A record that cannot be read or written, with the reason."""


class ConsentStore:
    """The consent records under one assets root: ``<assets>/.consent/<release_sha256>.json``."""

    def __init__(self, assets_dir: str | Path) -> None:
        self.dir = Path(assets_dir) / CONSENT_DIR

    def path(self, release_sha256: str) -> Path:
        if not re.fullmatch(SHA256_PATTERN, release_sha256):
            raise ConsentError(f"{release_sha256!r} is not a release's sha256")
        return self.dir / f"{release_sha256}.json"

    def get(self, release_sha256: str) -> ConsentRecord | None:
        """The record for a release, None if there is none. A record that does not read is :class:`ConsentError`."""
        path = self.path(release_sha256)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ConsentError(f"the consent record cannot be read: {exc}") from exc
        try:
            record = ConsentRecord.model_validate_json(raw)
        except ValidationError as exc:
            raise ConsentError(f"the consent record is not valid: {exc.errors()[0]['msg']}") from exc
        if record.release_sha256 != release_sha256:
            raise ConsentError("the consent record is filed under another release's hash")
        return record

    def all(self) -> tuple[list[ConsentRecord], dict[str, str]]:
        """Every record that reads, and for each file that does not, why."""
        found: list[ConsentRecord] = []
        broken: dict[str, str] = {}
        if not self.dir.is_dir():
            return found, broken
        for path in sorted(self.dir.glob("*.json")):
            try:
                record = self.get(path.stem)
            except ConsentError as exc:
                broken[path.name] = str(exc)
                continue
            if record is not None:
                found.append(record)
        return found, broken

    def find(self, prefix: str) -> ConsentRecord:
        """The one record whose release hash starts with *prefix* (at least 8 hex characters)."""
        prefix = prefix.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{8,64}", prefix):
            raise ConsentError("name a release by its sha256, or at least its first 8 characters")
        matches = [r for r in self.all()[0] if r.release_sha256.startswith(prefix)]
        if not matches:
            raise ConsentError(f"no consent record for a release starting {prefix}")
        if len(matches) > 1:
            raise ConsentError(f"{len(matches)} releases start {prefix}: give more of the hash")
        return matches[0]

    def _write(self, record: ConsentRecord, *, new: bool) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path(record.release_sha256)
        text = record.model_dump_json(indent=2) + "\n"
        if new:
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError as exc:
                raise ConsentError(
                    "there is already a record for this release: a re-signed release has a new hash"
                ) from exc
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(text)
            return
        tmp = path.with_suffix(".part")
        tmp.write_text(text, encoding="utf-8")
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    def create(self, record: ConsentRecord) -> None:
        """Write a new record. A release has one record: writing a second for the same hash is refused."""
        if record.revocation is not None:
            raise ConsentError("a new record is not revoked: write it, then revoke it")
        self._write(record, new=True)

    def revoke(self, release_sha256: str, revocation: Revocation) -> ConsentRecord:
        """Mark a record revoked. Revoking a revoked record keeps the first revocation, and returns it unchanged."""
        record = self.get(release_sha256)
        if record is None:
            raise ConsentError(f"no consent record for release {release_sha256}")
        if record.revocation is not None:
            return record
        revoked = record.model_copy(update={"revocation": revocation})
        self._write(revoked, new=False)
        return revoked

    # -- checks with mask's own reasons on top of consent_valid's --

    def check_face(self, provenance: Provenance, *, use: ConsentUse | None, at: datetime) -> ConsentVerdict | str:
        """Whether a consented face may be used: the verdict, or (a string) why there is nothing to check."""
        if provenance.release_sha256 is None:
            return "a consented face names its release"
        try:
            record = self.get(provenance.release_sha256)
        except ConsentError as exc:
            return str(exc)
        if record is None:
            return "no consent record for its release (argus-mask consent new)"
        if not consent_matches(record, provenance):
            return "its subject is not the person its release's consent record names"
        return consent_valid(record, use=use, likeness="face", at=at)

    def check_voice(self, source: dict, *, delivery: str, use: ConsentUse | None, at: datetime) -> ConsentVerdict | str:
        """The same for a consented voice, from its ``voice.json`` source (``speaker``, ``release_sha256``)."""
        release = str(source.get("release_sha256") or "")
        if not re.fullmatch(SHA256_PATTERN, release):
            return "a consented voice names its release"
        try:
            record = self.get(release)
        except ConsentError as exc:
            return str(exc)
        if record is None:
            return "no consent record for its release (argus-mask consent new)"
        if not same_subject(str(source.get("speaker") or ""), record.subject):
            return "its speaker is not the person its release's consent record names"
        if not delivery:
            return "its record names no delivery"
        return consent_valid(record, use=use, likeness="voice", at=at, delivery=delivery)


def refusal(found: ConsentVerdict | str) -> str | None:
    """Why a check refuses, in words, or None if it allows."""
    if isinstance(found, str):
        return found
    return None if found.ok else found.detail


# ── Voices made on the Voices page ────────────────────────────────────────────

VOICE_ID = r"^[a-z0-9][a-z0-9-]{0,62}$"


def voice_record(voices_dir: str | Path, voice: str) -> dict | None:
    """The ``voice.json`` behind a voice name (``holly`` or ``clone:holly``), if it was made on the Voices page here.

    A service's own speaker (a number, ``speaker:<name>``) or a clone made elsewhere has none: None.
    """
    name = voice.strip().removeprefix("clone:")
    if not re.fullmatch(VOICE_ID, name):
        return None
    try:
        record = json.loads((Path(voices_dir) / name / "voice.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def voice_of(body: dict, default: str) -> str:
    """The voice a TTS request will speak in: one its fields name (``-p voice_clone_id=…``), else *default*."""
    for key in ("voice_clone_id", "voice", "name"):
        if isinstance(body.get(key), str) and body[key].strip():
            return body[key]
    return default


def voice_refusal(
    voices_dir: str | Path, consents: ConsentStore, voice: str, *, use: ConsentUse | None, at: datetime
) -> str | None:
    """Why *voice* may not be spoken for *use*, or None if it may (or is not a consented voice made here)."""
    record = voice_record(voices_dir, voice)
    if record is None:
        return None
    source = record.get("source")
    if not isinstance(source, dict) or source.get("kind") != "consented":
        return None
    why = refusal(consents.check_voice(source, delivery=str(record.get("delivery") or ""), use=use, at=at))
    return None if why is None else f"the voice {record.get('id', voice)} may not be used: {why}"


def voice_disclosed(voices_dir: str | Path, voice: str) -> bool:
    """Whether *voice* is a real, consenting person's (made here), so what it says carries the disclosure label."""
    record = voice_record(voices_dir, voice)
    source = record.get("source") if record else None
    return isinstance(source, dict) and source.get("kind") == "consented"


# ── Revocation: what ``consent revoke`` does ──────────────────────────────────


@dataclass
class Withdrawal:
    """What revoking a release did: the record, and what was removed, kept or still to do."""

    record: ConsentRecord
    pending: bool = False  # the revocation takes effect later: nothing removed yet
    voices_removed: list[str] = field(default_factory=list)
    voices_failed: dict[str, str] = field(default_factory=dict)  # voice id -> why it is still here
    faces_trashed: list[str] = field(default_factory=list)


def _trash(folder: Path, bin_: Path, stamp: str) -> None:
    bin_.mkdir(parents=True, exist_ok=True)
    folder.replace(bin_ / f"{folder.name}-{stamp}")


def withdraw(
    assets_dir: str | Path,
    voices_dir: str | Path,
    release_sha256: str,
    revocation: Revocation,
    *,
    today: date,
    stamp: str,
    delete_clone: Callable[[str, str], object],
) -> Withdrawal:
    """Revoke a release and remove what was made under it (argus-mask#26 § 3).

    The record is marked revoked first, so from its effective day nothing under it is served or spoken, whatever
    else fails. Then, once it is in effect: each voice made under it is deleted from the TTS service it was made
    on (*delete_clone(tts_url, voice_id)*, the Voices page's Delete) and its record and clips go to the voices'
    ``.trash/``; each face goes to the assets' ``.trash/``. A voice whose service cannot be reached stays, revoked
    and refused, and is reported: run it again to finish. Running it again is safe: the first revocation stands.
    """
    store = ConsentStore(assets_dir)
    record = store.revoke(release_sha256, revocation)
    assert record.revocation is not None
    done = Withdrawal(record=record)
    if today < record.revocation.effective_on:
        done.pending = True
        return done
    voices = Path(voices_dir)
    for path in sorted(voices.glob("*/voice.json")) if voices.is_dir() else []:
        try:
            voice = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        source = voice.get("source") if isinstance(voice, dict) else None
        if not isinstance(source, dict) or source.get("release_sha256") != record.release_sha256:
            continue
        voice_id = path.parent.name
        made_on = voice.get("tts")
        if isinstance(made_on, str) and made_on:
            try:
                delete_clone(f"{made_on}/synthesise", voice_id)
            except Exception as exc:  # noqa: BLE001 - every failure is reported, and the voice kept for a re-run
                if getattr(exc, "status", None) != 404:  # gone already: the record still goes
                    done.voices_failed[voice_id] = f"not deleted from {made_on}: {exc}"
                    continue
        _trash(path.parent, voices / ".trash", stamp)
        done.voices_removed.append(voice_id)
    assets = Path(assets_dir)
    for path in sorted(assets.glob("*/manifest.json")):
        if path.parent.name.startswith("."):
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        provenance = manifest.get("provenance") if isinstance(manifest, dict) else None
        if isinstance(provenance, dict) and provenance.get("release_sha256") == record.release_sha256:
            _trash(path.parent, assets / ".trash", stamp)
            done.faces_trashed.append(path.parent.name)
    return done
