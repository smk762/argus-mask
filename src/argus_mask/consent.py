"""Consent records: the record that decides whether a real person's face or voice may be used, and for what.

A ``consented`` face (its :class:`~argus_mask.models.Provenance`) or voice (its ``voice.json``) names a signed
release by its sha256. The :class:`~argus_cortex.consent.ConsentRecord` for that release says what the person agreed
to: their face, their voice (and which deliveries), which uses, until when, and whether it has been revoked. It holds
hashes of the release and of the reference images and recordings taken at the signing, never the documents or the
media, and never any copy of the ID the person's age was checked against.

The record, :func:`~argus_cortex.consent.consent_valid` and :func:`~argus_cortex.consent.consent_matches` are
argus-cortex's (``argus_cortex.consent``, from the definition on argus-mask#26), so mask, curator and darkroom
refuse for the same reasons in the same words. Mask owns what the fields mean; change them there together with
argus-mask#26. This module keeps what is mask's own:

- :class:`ConsentStore`: the records as ``<assets>/.consent/<release_sha256>.json`` (hidden, so the face store never
  takes the folder for a face), written once by ``consent new`` and changed only by ``consent revoke``, and mask's
  write-side rules on top of cortex's (:func:`check_new`: delivery names the Voices page can use, bounded text);
- :func:`consent_holds`: whether a record holds for a likeness at all, whatever the use (making a voice, Try);
- the face and voice checks, with mask's own reasons (no record, another person) beside cortex's verdicts;
- :func:`withdraw`, which is ``consent revoke``.

The checks are pure: the caller passes the time. Cortex reads a record's dates as UTC days, so mask's own "today"
(a revocation's default day, whether one is in effect yet) is the UTC day too: :func:`utc_today`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

from argus_cortex.consent import CONSENT_USES as CONSENT_USES
from argus_cortex.consent import AgeCheck as AgeCheck
from argus_cortex.consent import ConsentRecord, ConsentUse, ConsentVerdict, Likeness, Revocation, consent_valid
from argus_cortex.consent import consent_matches as consent_matches
from argus_cortex.provenance import Provenance as CortexProvenance
from pydantic import ValidationError

from argus_mask.models import FACE_DISCLOSURE as FACE_DISCLOSURE  # the labels, beside Provenance which carries one
from argus_mask.models import SHA256_PATTERN, Provenance
from argus_mask.models import VOICE_DISCLOSURE as VOICE_DISCLOSURE

# Mask's own, for records it writes: a delivery is named as the Voices page names it, so a record and a voice.json
# say it the same way (a "Warm" in the record would never match a voice's "warm").
DELIVERY_PATTERN = r"^[a-z][a-z0-9-]{0,31}$"
MAX_NAME = 120  # subject, collector, age checker, revoker
MAX_NOTES = 2000

CONSENT_DIR = ".consent"


def utc_today(at: datetime) -> date:
    """*at*'s calendar day in UTC: the day cortex compares a record's dates with."""
    return at.astimezone(UTC).date()


def uses_of(record: ConsentRecord) -> list[str]:
    """The record's uses, in the stable order cortex writes them."""
    return [u for u in CONSENT_USES if u in record.uses]


def check_new(record: ConsentRecord) -> None:
    """Mask's rules for a record it writes, beyond cortex's: :class:`ConsentError` if one fails."""
    if bad := [d for d in record.deliveries if not re.fullmatch(DELIVERY_PATTERN, d)]:
        raise ConsentError(f"not a delivery name (lower case, as the Voices page names them): {', '.join(bad)}")
    names = {"subject": record.subject, "collected_by": record.collected_by, "age_check": record.age_check.checked_by}
    if long := [k for k, v in names.items() if len(v) > MAX_NAME]:
        raise ConsentError(f"at most {MAX_NAME} characters: {', '.join(long)}")
    if len(record.notes) > MAX_NOTES:
        raise ConsentError(f"notes: at most {MAX_NOTES} characters")


def consent_holds(
    record: ConsentRecord, *, likeness: Likeness, at: datetime, delivery: str | None = None
) -> ConsentVerdict:
    """Whether *record* holds for *likeness* at all, whatever the use: for making a voice, or trying a line.

    Cortex's :func:`consent_valid` always names a use. Every check but the use's is the same for every use, and a
    use the record lists passes the use's check, so asking for any one of the record's own uses answers this.
    """
    return consent_valid(record, use=uses_of(record)[0], likeness=likeness, at=at, delivery=delivery)


def _verdict(
    record: ConsentRecord, *, use: ConsentUse | None, likeness: Likeness, at: datetime, delivery: str | None = None
) -> ConsentVerdict:
    if use is None:
        return consent_holds(record, likeness=likeness, at=at, delivery=delivery)
    return consent_valid(record, use=use, likeness=likeness, at=at, delivery=delivery)


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
        check_new(record)
        self._write(record, new=True)

    def revoke(self, release_sha256: str, revocation: Revocation) -> ConsentRecord:
        """Mark a record revoked. Revoking a revoked record keeps the first revocation, and returns it unchanged."""
        record = self.get(release_sha256)
        if record is None:
            raise ConsentError(f"no consent record for release {release_sha256}")
        if record.revocation is not None:
            return record
        # Validated again with the revocation, so cortex's checks run (model_copy would skip them).
        try:
            revoked = ConsentRecord.model_validate(record.model_dump() | {"revocation": revocation.model_dump()})
        except ValidationError as exc:
            raise ConsentError(f"cannot revoke: {exc.errors()[0]['msg'].removeprefix('Value error, ')}") from exc
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
        if not consent_matches(record, provenance):  # type: ignore[arg-type] - mask's Provenance has the same fields
            return "its subject is not the person its release's consent record names"
        return _verdict(record, use=use, likeness="face", at=at)

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
        if not _speaker_matches(record, str(source.get("speaker") or "")):
            return "its speaker is not the person its release's consent record names"
        if not delivery:
            return "its record names no delivery"
        return _verdict(record, use=use, likeness="voice", at=at, delivery=delivery)


def _speaker_matches(record: ConsentRecord, speaker: str) -> bool:
    """A voice's speaker is the record's subject by cortex's own rule: its consent_matches, on the voice's source."""
    try:
        source = CortexProvenance(
            kind="consented",
            source="a voice made on the Voices page",
            license="per the release",
            subject=speaker,
            release_sha256=record.release_sha256,
        )
    except ValidationError:  # no speaker named
        return False
    return consent_matches(record, source)


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
