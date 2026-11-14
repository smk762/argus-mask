"""Consent records (argus-mask#26): the record, its check, the store, revocation, and where mask enforces them.

Synthetic fixtures only: the release is a hash of nothing, the subject a placeholder name, the voice a tone.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest
from argus_cortex import consent as cortex_consent
from pydantic import ValidationError
from starlette.testclient import TestClient
from typer.testing import CliRunner

from argus_mask.cli import app as cli
from argus_mask.config import Settings
from argus_mask.consent import (
    ConsentError,
    ConsentStore,
    Revocation,
    consent_holds,
    consent_matches,
    consent_valid,
    uses_of,
    utc_today,
    voice_disclosed,
    voice_of,
    voice_refusal,
    withdraw,
)
from argus_mask.faces import FaceError, FaceStore
from argus_mask.models import FACE_DISCLOSURE, VOICE_DISCLOSURE, Generator, Provenance, StreamHeader
from argus_mask.server.app import create_app
from argus_mask.take import line
from tests.helpers import RELEASE, consent_record, make_face, write_consent

AT = datetime(2026, 6, 1, 12, tzinfo=UTC)
CONSENTED = Provenance(
    kind="consented", source="a capture session", license="per the release", subject="A. Person", release_sha256=RELEASE
)


def at(day: str) -> datetime:
    return datetime.fromisoformat(f"{day}T12:00:00+00:00")


# ── the record ────────────────────────────────────────────────────────────────


# The record is cortex's (argus_cortex.consent); cortex tests its own checks. These check that mask uses it, the
# refusals the 0.5.0 review added, and mask's own rules for the records it writes.


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"face": False, "voice": False, "deliveries": []}, "the face, the voice or both"),
        ({"expires_on": "2026-01-10"}, "expires after the day it was signed"),
        ({"age_check": {"checked_by": "   ", "checked_on": "2026-01-10"}}, "names who did it"),
        ({"id_scan": "…"}, "Extra inputs"),  # never the ID, nor any field nobody defined
    ],
)
def test_the_record_is_cortexs(over, why) -> None:
    with pytest.raises(ValidationError, match=why):
        consent_record(**over)
    assert type(consent_record()) is cortex_consent.ConsentRecord


def test_a_revocation_before_the_signing_is_refused(tmp_path) -> None:
    write_consent(tmp_path)
    early = Revocation(requested_on="2026-01-09", effective_on="2026-01-09", by="B")
    with pytest.raises(ConsentError, match="requested on or after the day the release was signed"):
        ConsentStore(tmp_path).revoke(RELEASE, early)
    assert ConsentStore(tmp_path).get(RELEASE).revocation is None


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"deliveries": ["Warm"]}, "not a delivery name"),
        ({"deliveries": ["warm voice"]}, "not a delivery name"),
        ({"subject": "A" * 121}, "subject"),
        ({"notes": "x" * 2001}, "notes"),
    ],
)
def test_mask_writes_only_records_its_voices_can_use(tmp_path, over, why) -> None:
    with pytest.raises(ConsentError, match=why):
        ConsentStore(tmp_path).create(consent_record(**over))
    assert not (tmp_path / ".consent" / f"{RELEASE}.json").exists()


def test_a_record_writes_its_uses_in_one_order() -> None:
    record = consent_record(uses=["video", "avatar", "video"])
    assert json.loads(record.model_dump_json())["uses"] == ["avatar", "video"] and uses_of(record) == [
        "avatar",
        "video",
    ]


# ── consent_valid: the order of the checks, and their words ────────────────────


@pytest.mark.parametrize(
    ("over", "kwargs", "reason"),
    [
        ({"revocation": Revocation(requested_on="2026-05-01", effective_on="2026-05-15", by="B")}, {}, "revoked"),
        (
            {"signed_on": "2026-07-01", "age_check": {"checked_by": "B", "checked_on": "2026-07-01"}},
            {},
            "not-yet-signed",
        ),
        ({"expires_on": "2026-06-01"}, {}, "expired"),  # the first day it no longer holds
        ({"face": False, "reference_images": []}, {}, "likeness-not-covered"),
        ({}, {"use": "training"}, "use-not-covered"),
        ({}, {"likeness": "voice", "delivery": "serious"}, "delivery-not-covered"),
        # Revoked and out of scope at once: revoked wins, every consumer says the same.
        (
            {"revocation": Revocation(requested_on="2026-05-01", effective_on="2026-05-01", by="B")},
            {"use": "training"},
            "revoked",
        ),
    ],
)
def test_consent_valid_refuses_in_a_fixed_order(over, kwargs, reason) -> None:
    verdict = consent_valid(consent_record(**over), **({"use": "avatar", "likeness": "face", "at": AT} | kwargs))
    assert not verdict.ok and verdict.reason == reason and verdict.detail


def test_consent_valid_allows_what_the_record_covers_and_reads_no_clock() -> None:
    record = consent_record()
    assert consent_valid(record, use="avatar", likeness="face", at=AT).ok
    assert consent_valid(record, use="video", likeness="voice", at=AT, delivery="warm").ok
    # Whether it holds at all, whatever the use (making a voice, Try): cortex always names a use, so mask asks for one
    # the record lists.
    assert consent_holds(record, likeness="voice", at=AT, delivery="neutral").ok
    assert consent_holds(consent_record(uses=["training"]), likeness="face", at=AT).ok
    assert consent_holds(record, likeness="voice", at=AT, delivery="serious").reason == "delivery-not-covered"
    # A revocation that takes effect later: the record holds until that day.
    later = consent_record(revocation=Revocation(requested_on="2026-05-01", effective_on="2026-06-02", by="B"))
    assert consent_valid(later, use="avatar", likeness="face", at=AT).ok
    assert consent_valid(later, use="avatar", likeness="face", at=at("2026-06-02")).reason == "revoked"
    with pytest.raises(ValueError, match="aware"):
        consent_valid(record, use="avatar", likeness="face", at=datetime(2026, 6, 1))  # noqa: DTZ001
    with pytest.raises(ValueError, match="delivery"):
        consent_valid(record, use="avatar", likeness="voice", at=AT)


def test_a_record_matches_a_face_by_release_and_subject() -> None:
    record = consent_record()
    assert consent_matches(record, CONSENTED)
    # The subject as the release names them: spaces at the ends aside, exactly (cortex's rule, the one curator uses).
    assert consent_matches(record, CONSENTED.model_copy(update={"subject": "  A. Person "}))
    assert not consent_matches(record, CONSENTED.model_copy(update={"subject": "a. person"}))
    assert not consent_matches(record, CONSENTED.model_copy(update={"subject": "Someone Else"}))
    assert not consent_matches(record, CONSENTED.model_copy(update={"release_sha256": "cd" * 32}))


# ── the store ─────────────────────────────────────────────────────────────────


def test_a_release_has_one_record_and_a_revocation_stands(tmp_path) -> None:
    store = ConsentStore(tmp_path)
    assert store.get(RELEASE) is None
    write_consent(tmp_path)
    assert store.get(RELEASE) == consent_record()
    assert (tmp_path / ".consent" / f"{RELEASE}.json").stat().st_mode & 0o077 == 0  # it names a person
    with pytest.raises(ConsentError, match="already a record"):
        store.create(consent_record())
    first = Revocation(requested_on="2026-05-01", effective_on="2026-05-01", by="B")
    assert store.revoke(RELEASE, first).revocation == first
    again = store.revoke(RELEASE, Revocation(requested_on="2026-09-01", effective_on="2026-09-01", by="C"))
    assert again.revocation == first and store.get(RELEASE).revocation == first
    assert store.find(RELEASE[:8]).release_sha256 == RELEASE
    with pytest.raises(ConsentError, match="at least its first 8"):
        store.find("ab")


def test_a_record_filed_under_another_hash_or_broken_is_refused(tmp_path) -> None:
    store = ConsentStore(tmp_path)
    store.dir.mkdir()
    (store.dir / f"{'cd' * 32}.json").write_text(consent_record().model_dump_json())
    (store.dir / f"{'ef' * 32}.json").write_text("{")
    records, broken = store.all()
    assert records == [] and set(broken) == {f"{'cd' * 32}.json", f"{'ef' * 32}.json"}
    assert "another release" in broken[f"{'cd' * 32}.json"]


# ── faces ─────────────────────────────────────────────────────────────────────


def test_a_consented_face_is_served_only_while_its_record_covers_it(tmp_path) -> None:
    make_face(tmp_path, "agreed", provenance=CONSENTED)
    store = FaceStore(tmp_path, clock=lambda: AT)
    with pytest.raises(FaceError, match="no consent record"):
        store.load("agreed")
    write_consent(tmp_path, uses=["video"])
    with pytest.raises(FaceError, match="does not cover the use 'avatar'"):
        store.load("agreed")
    assert store.load("agreed", use="video")[0].provenance.disclosure == FACE_DISCLOSURE
    ConsentStore(tmp_path).revoke(RELEASE, Revocation(requested_on="2026-05-01", effective_on="2026-05-01", by="B"))
    with pytest.raises(FaceError, match="revoked"):
        store.load("agreed", use="video")
    accepted, refused = store.scan()
    assert accepted == [] and "revoked" in refused["agreed"] and ".consent" not in refused


def test_a_face_under_someone_elses_release_is_refused(tmp_path) -> None:
    make_face(tmp_path, "agreed", provenance=CONSENTED.model_copy(update={"subject": "Someone Else"}))
    write_consent(tmp_path)
    with pytest.raises(FaceError, match="not the person"):
        FaceStore(tmp_path, clock=lambda: AT).load("agreed")


def test_the_server_labels_a_consented_face_and_answers_for_video(tmp_path) -> None:
    make_face(tmp_path, "agreed", provenance=CONSENTED)
    make_face(tmp_path, "test-head")
    write_consent(tmp_path, uses=["avatar"])
    with TestClient(create_app(Settings(assets_dir=tmp_path))) as c:
        face = c.get("/face/agreed").json()
        assert face["provenance"]["disclosure"] == FACE_DISCLOSURE and face["provenance"]["cleared"] is True
        assert c.get("/face/test-head").json()["provenance"]["disclosure"] is None
        assert c.get("/face/test-head/consent", params={"use": "video"}).json() == {
            "ok": True,
            "reason": None,
            "detail": "",
        }
        video = c.get("/face/agreed/consent", params={"use": "video"}).json()
        assert video["ok"] is False and video["reason"] == "use-not-covered"
        assert c.get("/face/agreed/consent", params={"use": "avatar"}).json()["ok"] is True
        assert c.get("/face/agreed/consent", params={"use": "parody"}).status_code == 422
        assert c.get("/face/nobody/consent", params={"use": "video"}).status_code == 404


# ── provenance: generated faces, and the label ────────────────────────────────


def test_a_generated_face_says_how_and_only_a_generated_one_does() -> None:
    made = Provenance(
        kind="synthetic",
        source="argus-darkroom",
        license="CC0-1.0",
        generator=Generator(model="sdxl-base-1.0", model_sha256="cd" * 32, seed=7, prompt="a portrait"),
    )
    assert Provenance.model_validate_json(made.model_dump_json()) == made
    with pytest.raises(ValidationError, match="was not generated"):
        Provenance(kind="stock", source="a pack", license="CC0", generator=Generator(model="x"))
    # The label is derived like ``cleared``: a dumped one is recomputed, never believed.
    dumped = CONSENTED.model_dump() | {"disclosure": None}
    assert Provenance.model_validate(dumped).disclosure == FACE_DISCLOSURE


def test_a_header_without_a_label_writes_the_take_it_wrote_before() -> None:
    header = StreamHeader(
        face_id="f",
        session_id="s",
        fps=30,
        seed=0,
        blendshapes=[],
        audio={"sample_rate": 16000},
        driver={"name": "energy"},
    )
    assert "disclosure" not in json.loads(line(header))
    labelled = header.model_copy(update={"disclosure": VOICE_DISCLOSURE})
    assert json.loads(line(labelled))["disclosure"] == VOICE_DISCLOSURE


def test_a_consented_voice_on_the_source_socket_is_labelled_for_its_viewers(tmp_path) -> None:
    make_face(tmp_path, "test-head")
    hello = {"sample_rate": 24000, "channels": 1, "format": "f32le", "disclose": True}
    with (
        TestClient(create_app(Settings(assets_dir=tmp_path))) as c,
        c.websocket_connect("/session/test-head/view") as viewer,
        c.websocket_connect("/session/test-head/source") as source,
    ):
        source.send_text(json.dumps(hello))
        assert source.receive_json()["type"] == "ready"
        header = viewer.receive_json()
        source.send_text(json.dumps({"type": "end"}))
        source.receive_json()
    assert header["type"] == "header" and header["disclosure"] == VOICE_DISCLOSURE


# ── voices ────────────────────────────────────────────────────────────────────


def make_voice(voices, voice_id="a-person-warm", *, kind="consented", delivery="warm", tts="http://127.0.0.1:9"):
    folder = voices / voice_id
    folder.mkdir(parents=True)
    source = {"kind": kind, "speaker": "A. Person", "release_sha256": RELEASE, "confirmed": True}
    record = {"id": voice_id, "delivery": delivery, "tts": tts, "source": source}
    (folder / "voice.json").write_text(json.dumps(record))
    (folder / "clip-1.wav").write_bytes(b"RIFF")
    return folder


def test_a_consented_voice_is_spoken_only_while_its_record_covers_it(tmp_path) -> None:
    voices, consents = tmp_path / "voices", ConsentStore(tmp_path)
    make_voice(voices)
    make_voice(voices, "a-person-serious", delivery="serious")
    make_voice(voices, "made-up", kind="generated")
    assert "no consent record" in voice_refusal(voices, consents, "clone:a-person-warm", use="avatar", at=AT)
    write_consent(tmp_path)
    assert voice_refusal(voices, consents, "clone:a-person-warm", use="avatar", at=AT) is None
    assert "does not cover the use 'training'" in voice_refusal(
        voices, consents, "a-person-warm", use="training", at=AT
    )
    assert "delivery 'serious'" in voice_refusal(voices, consents, "a-person-serious", use="avatar", at=AT)
    # Not a consented voice made here: nothing to check.
    for other in ("made-up", "speaker:Rosemary", "0", "elsewhere", "../a-person-warm"):
        assert voice_refusal(voices, consents, other, use="training", at=AT) is None
    assert voice_disclosed(voices, "a-person-warm") and not voice_disclosed(voices, "made-up")
    assert voice_of({"voice_clone_id": "a-person-warm"}, "made-up") == "a-person-warm"
    assert voice_of({"text": "hi"}, "made-up") == "made-up"


def test_speak_refuses_a_revoked_voice_before_any_tts(tmp_path) -> None:
    make_face(tmp_path, "test-head")
    make_voice(tmp_path / "voices")
    write_consent(tmp_path)
    ConsentStore(tmp_path).revoke(RELEASE, Revocation(requested_on="2026-02-01", effective_on="2026-02-01", by="B"))
    settings = Settings(assets_dir=tmp_path, tts_url="http://127.0.0.1:9/synthesise", voices_dir=tmp_path / "voices")
    with TestClient(create_app(settings)) as c:
        reply = c.post("/speak", json={"face": "test-head", "text": "Hello.", "voice": "clone:a-person-warm"})
    assert reply.status_code == 403 and "revoked" in reply.json()["detail"]


def test_say_refuses_a_voice_out_of_scope(tmp_path, monkeypatch) -> None:
    make_voice(tmp_path / "voices", delivery="serious")
    write_consent(tmp_path)
    monkeypatch.setenv("ARGUS_MASK_ASSETS_DIR", str(tmp_path))
    monkeypatch.setenv("ARGUS_MASK_VOICES_DIR", str(tmp_path / "voices"))
    result = CliRunner().invoke(
        cli, ["say", "Hello.", "--face", "f", "--tts-url", "http://127.0.0.1:9/synthesise", "--voice", "a-person-warm"]
    )
    assert result.exit_code == 1 and "delivery 'serious'" in result.output


# ── revocation ────────────────────────────────────────────────────────────────


def test_revoking_deletes_the_voices_and_trashes_the_faces(tmp_path) -> None:
    voices = tmp_path / "voices"
    make_voice(voices)
    make_voice(voices, "a-person-neutral", delivery="neutral", tts="")  # made nowhere reachable: just trashed
    make_voice(voices, "made-up", kind="generated").joinpath("voice.json").write_text(
        json.dumps({"source": {"kind": "generated", "generator": "x"}})
    )
    make_face(tmp_path, "agreed", provenance=CONSENTED)
    make_face(tmp_path, "test-head")
    write_consent(tmp_path)
    deleted = []
    done = withdraw(
        tmp_path,
        voices,
        RELEASE,
        Revocation(requested_on="2026-06-01", effective_on="2026-06-01", by="B"),
        today=date(2026, 6, 1),
        stamp="t",
        delete_clone=lambda url, voice_id: deleted.append((url, voice_id)),
    )
    assert deleted == [("http://127.0.0.1:9/synthesise", "a-person-warm")]
    assert sorted(done.voices_removed) == ["a-person-neutral", "a-person-warm"] and done.voices_failed == {}
    assert done.faces_trashed == ["agreed"] and not done.pending
    assert sorted(p.name for p in (voices / ".trash").iterdir()) == ["a-person-neutral-t", "a-person-warm-t"]
    assert (tmp_path / ".trash" / "agreed-t" / "manifest.json").is_file()
    assert (voices / "made-up").is_dir() and (tmp_path / "test-head").is_dir()
    assert ConsentStore(tmp_path).get(RELEASE).revocation.by == "B"  # the record stays, revoked


def test_a_voice_whose_service_is_down_stays_refused_until_a_rerun(tmp_path) -> None:
    voices = tmp_path / "voices"
    make_voice(voices)
    write_consent(tmp_path)
    revocation = Revocation(requested_on="2026-06-01", effective_on="2026-06-01", by="B")

    def down(url, voice_id):
        raise OSError("connection refused")

    done = withdraw(tmp_path, voices, RELEASE, revocation, today=date(2026, 6, 1), stamp="t", delete_clone=down)
    assert "connection refused" in done.voices_failed["a-person-warm"] and (voices / "a-person-warm").is_dir()
    assert "revoked" in voice_refusal(
        voices, ConsentStore(tmp_path), "a-person-warm", use="avatar", at=at("2026-06-01")
    )

    class Gone(Exception):
        status = 404

    def gone(url, voice_id):
        raise Gone

    again = withdraw(tmp_path, voices, RELEASE, revocation, today=date(2026, 6, 2), stamp="u", delete_clone=gone)
    assert again.voices_removed == ["a-person-warm"] and not (voices / "a-person-warm").exists()


def test_a_revocation_for_later_removes_nothing_yet(tmp_path) -> None:
    make_voice(tmp_path / "voices")
    write_consent(tmp_path)
    done = withdraw(
        tmp_path,
        tmp_path / "voices",
        RELEASE,
        Revocation(requested_on="2026-06-01", effective_on="2026-07-01", by="B"),
        today=date(2026, 6, 1),
        stamp="t",
        delete_clone=lambda *_: pytest.fail("deleted before the revocation took effect"),
    )
    assert done.pending and (tmp_path / "voices" / "a-person-warm").is_dir()


# ── the CLI ───────────────────────────────────────────────────────────────────


def test_consent_new_list_and_revoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ARGUS_MASK_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("ARGUS_MASK_VOICES_DIR", str(tmp_path / "voices"))
    release, image, clip = tmp_path / "release.pdf", tmp_path / "front.jpg", tmp_path / "read.wav"
    release.write_bytes(b"a signed release, as bytes")
    image.write_bytes(b"a frontal image")
    clip.write_bytes(b"a reference clip")
    runner = CliRunner()
    base = ["consent", "new", "--release", str(release), "--subject", "A. Person", "--collected-by", "B. Collector"]
    base += ["--signed-on", "2026-01-10", "--age-checked-by", "B. Collector", "--revocable", "--on-revoke", "delete"]
    missing = runner.invoke(cli, [*base, "--use", "avatar", "--voice"])
    assert missing.exit_code == 1 and "deliveries" in missing.output
    made = runner.invoke(
        cli,
        [*base, "--use", "avatar", "--use", "speech", "--face", "--voice", "--delivery", "warm"]
        + ["--reference-image", str(image), "--reference-recording", str(clip)],
    )
    assert made.exit_code == 0, made.output
    store = ConsentStore(tmp_path / "assets")
    (record,) = store.all()[0]
    assert record.subject == "A. Person" and uses_of(record) == ["avatar", "speech"] and record.age_check.checked_on
    written = (store.dir / f"{record.release_sha256}.json").read_text()
    assert "a signed release" not in written and "frontal" not in written  # hashes only
    assert runner.invoke(cli, [*base, "--use", "avatar", "--face", "--reference-image", str(image)]).exit_code == 1

    listed = runner.invoke(cli, ["consent", "list"])
    assert listed.exit_code == 0 and "A. Person" in listed.output and "holds" in listed.output

    revoked = runner.invoke(cli, ["consent", "revoke", record.release_sha256[:10], "--by", "B. Collector"])
    assert revoked.exit_code == 0, revoked.output
    assert "revoked from" in revoked.output and "recordings already made: delete" in revoked.output
    assert "revoked" in runner.invoke(cli, ["consent", "list"]).output


def test_mask_reads_today_as_cortex_does_in_utc() -> None:
    # 01:00 on 2 June in UTC+8 is still 1 June in UTC: a revocation dated "today" must not be a day ahead of
    # cortex's check, or the release would stay in force for the rest of the UTC day.
    east = datetime.fromisoformat("2026-06-02T01:00:00+08:00")
    assert utc_today(east) == date(2026, 6, 1)
    revoked = consent_record(revocation=Revocation(requested_on=utc_today(east), effective_on=utc_today(east), by="B"))
    assert consent_valid(revoked, use="avatar", likeness="face", at=east).reason == "revoked"
