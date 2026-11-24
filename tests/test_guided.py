"""The build record of a guided face (#166): what it keeps, and the three rules it holds by itself."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import (
    GUIDED_DIR,
    RECORD_NAME,
    STEPS,
    Artifact,
    CannotOverrule,
    Check,
    GuidedBuild,
    GuidedError,
    GuidedStore,
    NoSuchBuild,
    NotReady,
    StepResult,
    Tool,
)
from argus_mask.faces import FaceStore
from tests.helpers import make_face

T0 = datetime(2026, 11, 23, 9, 0, tzinfo=UTC)
SHA = "ab" * 32
EVALS = ("likeness", "rest", "turns", "speech", "age")


def ticking():
    """A clock a minute later each time it is read."""
    now = [T0]

    def clock() -> datetime:
        now[0] += timedelta(minutes=1)
        return now[0]

    return clock


def result(step: str, candidate: str | None = "c1", *checks: Check, **over: object) -> StepResult:
    return StepResult(step=step, candidate=candidate, checks=list(checks), **over)


def check(name: str, verdict: str = "passed", reason: str = "", **over: object) -> Check:
    return Check(
        name=name, verdict=verdict, reason=reason or ("" if verdict == "passed" else f"{name} said so"), **over
    )


def new_record() -> GuidedBuild:
    return GuidedBuild(id="guide", name="A guide", created=T0, updated=T0)


def built(record: GuidedBuild, candidate: str = "c1", *, portrait: str = SHA) -> GuidedBuild:
    """A candidate with its source, its checks and its head, all passed."""
    record.put(result("source", candidate, outputs=[Artifact(name="portrait", sha256=portrait)]), at=T0)
    record.put(result("portrait", candidate, check("frontal")), at=T0)
    record.put(result("build", candidate, outputs=[Artifact(name="head", sha256="cd" * 32)]), at=T0)
    return record


def evaluated(record: GuidedBuild, candidate: str = "c1") -> GuidedBuild:
    built(record, candidate)
    for step in EVALS:
        record.put(result(step, candidate, scores={"score": 0.5}), at=T0)
    return record


# ── What it keeps ─────────────────────────────────────────────────────────────────────────────────────────────


def test_a_record_survives_a_round_trip(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path, clock=ticking())
    store.create("guide", "A guide")
    portrait = store.keep("guide", "portrait", "c1/portrait.png", b"a portrait")
    store.put(
        "guide",
        result(
            "source",
            "c1",
            check("darkroom", measure={"records": 1.0}),
            outputs=[portrait],
            tools=[Tool(name="FLUX.1 schnell", sha256=SHA, license="Apache-2.0")],
            seeds={"portrait": 501},
            scores={"sharpness": 0.25},
            data={"kind": "synthetic", "record": {"id": "r1", "loras": []}},
            started=T0,
            finished=T0 + timedelta(seconds=9),
        ),
    )
    store.put("guide", result("portrait", "c1", check("temples", "advisory", "bare at the left temple")))
    store.overrule("guide", "portrait", "temples", "the hair covers it from the front", candidate="c1")

    first, again = store.load("guide"), GuidedStore(tmp_path).load("guide")
    assert again == first
    assert again.get("source", "c1").outputs == [portrait]
    assert (store.directory("guide") / "c1" / "portrait.png").read_bytes() == b"a portrait"
    assert portrait.bytes == 10 and len(portrait.sha256) == 64
    # A digest is what the step found, so it reads the same from the file as it did in memory.
    assert [r.digest for r in again.results] == [r.digest for r in first.results]
    assert again.created == T0 + timedelta(minutes=1) and again.updated > again.created


def test_a_digest_is_what_the_step_found_not_when_or_how_or_what_was_overruled() -> None:
    found = result("portrait", "c1", check("temples", "advisory", "bare"), scores={"light": 0.1})
    later = found.model_copy(
        update={"started": T0, "finished": T0, "after": {"source": SHA}, "run": {"seconds": 131.5, "log_tail": ["x"]}}
    )
    assert later.digest == found.digest
    assert result("portrait", "c1", check("temples", "advisory", "bare"), scores={"light": 0.2}).digest != found.digest
    assert result("portrait", "c2", check("temples", "advisory", "bare"), scores={"light": 0.1}).digest != found.digest


@pytest.mark.parametrize("score", [float("inf"), float("nan")])
def test_a_score_is_a_finite_number(score: float) -> None:
    """No face found is a check that failed, never a score of infinity: JSON has no way to write one."""
    with pytest.raises(ValidationError):
        result("likeness", "c1", scores={"misfit": score})
    with pytest.raises(ValidationError):
        Check(name="misfit", verdict="passed", measure={"misfit": score})


def test_a_check_that_did_not_pass_says_why() -> None:
    for verdict in ("advisory", "refused", "failed", "not_checked"):
        with pytest.raises(ValidationError, match="says why"):
            Check(name="light", verdict=verdict, reason="  ")


def test_a_result_is_for_a_known_step_and_names_its_candidate_or_none() -> None:
    with pytest.raises(ValidationError, match="is not a step"):
        result("polish")
    with pytest.raises(ValidationError, match="names its candidate"):
        result("build", None)
    with pytest.raises(ValidationError, match="names its candidate"):
        result("pick", None)
    with pytest.raises(ValidationError, match="the face's step"):
        result("scene", "c1")
    with pytest.raises(ValidationError, match="named once"):
        result("portrait", "c1", check("light"), check("light"))


def test_every_step_needs_only_steps_before_it() -> None:
    """The steps are in the order they run, and the face's steps never reach past the pick for a candidate's."""
    seen: list[str] = []
    for step in STEPS.values():
        assert set(step.needs) <= set(seen), f"{step.name} needs a step that comes after it"
        if step.per == "face":
            assert all(STEPS[n].per != "candidate" for n in step.needs), f"{step.name} needs a candidate's step"
        seen.append(step.name)
    assert STEPS["pick"].needs == EVALS


# ── A step runs only on settled ground ────────────────────────────────────────────────────────────────────────


def test_a_step_cannot_run_before_what_it_needs() -> None:
    record = new_record()
    with pytest.raises(NotReady, match="source has not run"):
        record.put(result("portrait"), at=T0)
    record.put(result("source"), at=T0)
    with pytest.raises(NotReady, match="portrait has not run"):
        record.put(result("build"), at=T0)
    assert record.results == [record.get("source", "c1")]  # nothing was kept of what was refused


@pytest.mark.parametrize(
    ("found", "why"),
    [
        (check("adult", "refused", "reads as under 18"), "portrait refused .adult: reads as under 18."),
        (check("faces", "failed", "the detector crashed"), "portrait failed .faces: the detector crashed."),
        (check("text", "not_checked", "the OCR server is down"), "portrait could not check .text: the OCR server"),
        (check("temples", "advisory", "bare at the temple"), "an advisory to overrule .temples: bare at the temple."),
    ],
)
def test_a_step_cannot_run_on_one_that_did_not_settle(found: Check, why: str) -> None:
    record = new_record()
    record.put(result("source"), at=T0)
    record.put(result("portrait", "c1", check("frontal"), found), at=T0)
    assert record.status("portrait", "c1") == found.verdict
    with pytest.raises(NotReady, match=why):
        record.put(result("build"), at=T0)
    assert record.get("build", "c1") is None


def test_one_candidate_never_stands_in_for_another() -> None:
    record = built(new_record(), "c1")
    record.put(result("source", "c2"), at=T0)
    with pytest.raises(NotReady, match="portrait has not run"):
        record.put(result("build", "c2"), at=T0)
    assert record.candidates() == ["c1", "c2"]


def test_the_pick_needs_every_eval_of_the_candidate_it_names() -> None:
    record = evaluated(new_record(), "c1")
    built(record, "c2")
    record.put(result("likeness", "c2"), at=T0)
    with pytest.raises(NotReady, match="rest has not run; turns has not run; speech has not run; age has not run"):
        record.put(result("pick", "c2"), at=T0)
    assert record.blockers("pick") == ["pick names its candidate"]
    with pytest.raises(NotReady, match="pick has not run"):
        record.put(result("scene", None), at=T0)

    record.put(result("pick", "c1"), at=T0)
    assert record.picked == "c1" and record.blockers("scene") == []
    record.put(result("scene", None, data={"scene": "garden", "shot": "close"}), at=T0)
    record.put(result("save", None), at=T0)
    assert record.summary()["face"] == {"pick": "passed", "scene": "passed", "save": "passed"}


# ── A refusal is never passed ─────────────────────────────────────────────────────────────────────────────────


def test_an_overruled_advisory_is_kept_with_its_reason_and_time() -> None:
    record = new_record()
    record.put(result("source"), at=T0)
    record.put(
        result("portrait", "c1", check("temples", "advisory", "bare"), check("light", "advisory", "uneven")), at=T0
    )
    later = T0 + timedelta(hours=1)

    record.overrule("portrait", "temples", "  a fringe will cover it  ", at=later, candidate="c1")
    assert record.status("portrait", "c1") == "advisory"  # one of two: the other still stands
    assert "light: uneven" in record.blockers("build", "c1")[0] and "temples" not in record.blockers("build", "c1")[0]
    record.overrule("portrait", "light", "the shot is a studio one", at=later, candidate="c1")

    assert record.status("portrait", "c1") == "overruled" and record.updated == later
    kept = record.get("portrait", "c1").overruled
    assert [(o.check, o.reason, o.at) for o in kept] == [
        ("temples", "a fringe will cover it", later),
        ("light", "the shot is a studio one", later),
    ]
    assert [c.verdict for c in record.get("portrait", "c1").checks] == ["advisory", "advisory"]  # still what it found
    record.put(result("build"), at=later)  # and the build may now run


@pytest.mark.parametrize(
    "found",
    [
        check("adult", "refused", "reads as under 18"),
        check("faces", "failed", "the detector crashed"),
        check("text", "not_checked", "the OCR server is down"),
        check("frontal"),
    ],
)
def test_nothing_but_an_advisory_can_be_overruled(found: Check) -> None:
    record = new_record()
    record.put(result("source"), at=T0)
    record.put(result("portrait", "c1", found), at=T0)
    before = record.model_dump()
    with pytest.raises(CannotOverrule, match="only an advisory is overruled"):
        record.overrule("portrait", found.name, "I know better", at=T0, candidate="c1")
    assert record.model_dump() == before
    assert record.status("portrait", "c1") == found.verdict


def test_an_overrule_names_a_real_check_once_and_says_why() -> None:
    record = new_record()
    record.put(result("source"), at=T0)
    record.put(result("portrait", "c1", check("temples", "advisory", "bare")), at=T0)
    with pytest.raises(GuidedError, match="has no check 'hairline'"):
        record.overrule("portrait", "hairline", "fine", at=T0, candidate="c1")
    with pytest.raises(GuidedError, match="says why"):
        record.overrule("portrait", "temples", "   ", at=T0, candidate="c1")
    with pytest.raises(GuidedError, match="build has not run"):
        record.overrule("build", "temples", "fine", at=T0, candidate="c1")
    record.overrule("portrait", "temples", "fine", at=T0, candidate="c1")
    with pytest.raises(CannotOverrule, match="overruled already"):
        record.overrule("portrait", "temples", "fine again", at=T0, candidate="c1")


def test_a_result_cannot_arrive_already_overruled() -> None:
    """A step that could overrule itself would pass its own advisory: only :meth:`GuidedBuild.overrule` does."""
    record = new_record()
    record.put(result("source"), at=T0)
    own = result("portrait", "c1", check("temples", "advisory", "bare")).model_copy(
        update={"overruled": [{"check": "temples", "reason": "I say so", "at": T0}]}
    )
    with pytest.raises(GuidedError, match="does not arrive overruled"):
        record.put(own, at=T0)
    # Nor does a step say what it ran on: the record does, from what is there.
    kept = record.put(result("portrait", "c1", check("frontal"), after={"source": "0" * 64}), at=T0)
    assert kept.after == {"source": record.get("source", "c1").digest}


def test_a_record_on_disk_that_overrules_a_refusal_is_refused(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide")
    store.put("guide", result("source"))
    store.put("guide", result("portrait", "c1", check("adult", "refused", "reads as under 18")))
    path = store.directory("guide") / RECORD_NAME
    raw = json.loads(path.read_text())
    raw["results"][1]["overruled"] = [{"check": "adult", "reason": "edited by hand", "at": T0.isoformat()}]
    path.write_text(json.dumps(raw))
    with pytest.raises(GuidedError, match="adult is refused, and only an advisory is overruled"):
        store.load("guide")
    assert "only an advisory" in store.scan()[1]["guide"]


def test_an_optional_check_that_could_not_run_stands_as_an_advisory_never_a_pass() -> None:
    record = new_record()
    record.put(result("source"), at=T0)
    down = check("age-read", "not_checked", "curator is down", required=False)
    record.put(result("portrait", "c1", down), at=T0)
    assert record.status("portrait", "c1") == "advisory"
    with pytest.raises(NotReady, match="age-read: curator is down"):
        record.put(result("build"), at=T0)
    record.overrule("portrait", "age-read", "going on without it", at=T0, candidate="c1")
    record.put(result("build"), at=T0)


def test_a_required_check_that_could_not_run_never_counts_as_passed() -> None:
    """Must never: a check that could not run counts as passed (#178 / #168)."""
    record = new_record()
    record.put(result("source"), at=T0)
    down = check("text", "not_checked", "the OCR server is down")
    record.put(result("portrait", "c1", down), at=T0)
    kept = record.get("portrait", "c1")
    assert kept is not None and kept.standing == "not_checked"
    assert record.status("portrait", "c1") == "not_checked"
    with pytest.raises(NotReady, match="could not check"):
        record.put(result("build"), at=T0)
    with pytest.raises(CannotOverrule, match="only an advisory"):
        record.overrule("portrait", "text", "skip it", at=T0, candidate="c1")


def test_running_a_step_again_drops_what_was_overruled() -> None:
    record = new_record()
    record.put(result("source"), at=T0)
    record.put(result("portrait", "c1", check("temples", "advisory", "bare")), at=T0)
    record.overrule("portrait", "temples", "fine", at=T0, candidate="c1")
    record.put(result("portrait", "c1", check("temples", "advisory", "bare")), at=T0)
    assert record.status("portrait", "c1") == "advisory" and record.get("portrait", "c1").overruled == []
    assert len(record.results) == 2  # in place of the earlier result, not beside it


# ── A result goes stale by itself ─────────────────────────────────────────────────────────────────────────────


def test_running_a_step_again_marks_what_followed_stale_only_if_it_found_something_else() -> None:
    record = evaluated(new_record(), "c1")
    evaluated(record, "c2")
    record.put(result("pick", "c1"), at=T0)
    record.put(result("scene", None), at=T0)
    after = ("build", *EVALS)

    # The same portrait checks, found again: nothing that followed changes.
    record.put(result("portrait", "c1", check("frontal")), at=T0)
    assert {record.status(s, "c1") for s in after} == {"passed"} and record.status("scene") == "passed"

    # A different finding: everything after it is stale, all the way to the face's steps. The other candidate is not.
    record.put(result("portrait", "c1", check("frontal"), check("light", "advisory", "uneven")), at=T0)
    assert {record.status(s, "c1") for s in after} == {"stale"}
    assert record.status("pick") == "stale" and record.status("scene") == "stale"
    assert {record.status(s, "c2") for s in after} == {"passed"}
    assert record.blockers("scene") == ["pick is stale: run it again"]
    with pytest.raises(NotReady, match="build is stale: run it again"):
        record.put(result("likeness", "c1"), at=T0)
    with pytest.raises(CannotOverrule, match="build is stale"):
        record.overrule("build", "anything", "fine", at=T0, candidate="c1")

    # Run again in order, each finding what it found before: each is fresh as it lands, and so is what ran on it.
    record.overrule("portrait", "light", "fine", at=T0, candidate="c1")
    assert record.status("build", "c1") == "stale"  # overruling settles the checks; the build still ran on others
    record.put(result("build", "c1", outputs=[Artifact(name="head", sha256="cd" * 32)]), at=T0)
    assert {record.status(s, "c1") for s in after} == {"passed"}
    assert record.status("pick") == "passed" and record.status("scene") == "passed"


def test_a_new_head_makes_its_evals_stale() -> None:
    record = evaluated(new_record(), "c1")
    record.put(result("build", "c1", outputs=[Artifact(name="head", sha256="ef" * 32)]), at=T0)
    assert {record.status(s, "c1") for s in EVALS} == {"stale"}
    record.put(result("rest", "c1", scores={"score": 0.5}), at=T0)
    assert record.status("rest", "c1") == "passed"
    assert record.status("speech", "c1") == "stale"  # it ran on the old head, though the rest set is as it was


def test_picking_another_candidate_makes_the_scene_stale() -> None:
    record = evaluated(new_record(), "c1")
    evaluated(record, "c2")
    record.put(result("pick", "c1"), at=T0)
    record.put(result("scene", None), at=T0)
    record.put(result("pick", "c2"), at=T0)
    assert record.picked == "c2" and record.status("scene") == "stale"
    assert [r.step for r in record.results].count("pick") == 1


# ── The store ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_a_build_is_never_made_in_place_of_another(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide", "First")
    store.put("guide", result("source"))
    with pytest.raises(GuidedError, match="already a build 'guide'"):
        store.create("guide", "Second")
    assert store.load("guide").name == "First" and len(store.load("guide").results) == 1
    assert sorted(p.name for p in (tmp_path / GUIDED_DIR).iterdir()) == ["guide"]  # nothing left of the attempt


@pytest.mark.parametrize("build_id", ["../guide", "Guide", ".guided", "a/b", "", "x" * 65])
def test_a_build_id_is_a_face_id(tmp_path: Path, build_id: str) -> None:
    store = GuidedStore(tmp_path)
    for call in (store.create, store.load, lambda i: store.put(i, result("source"))):
        with pytest.raises(NoSuchBuild, match="not a build id"):
            call(build_id)
    assert not (tmp_path / GUIDED_DIR).exists()


def test_the_list_is_latest_first_and_says_why_a_directory_is_not_a_build(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path, clock=ticking())
    for build_id in ("first", "second", "third"):
        store.create(build_id)
    store.put("first", result("source"))
    (tmp_path / GUIDED_DIR / "empty").mkdir()
    (tmp_path / GUIDED_DIR / "broken").mkdir()
    (tmp_path / GUIDED_DIR / "broken" / RECORD_NAME).write_text("{")
    moved = json.loads((tmp_path / GUIDED_DIR / "second" / RECORD_NAME).read_text())
    (tmp_path / GUIDED_DIR / "renamed").mkdir()
    (tmp_path / GUIDED_DIR / "renamed" / RECORD_NAME).write_text(json.dumps(moved))
    newer = {**moved, "id": "newer", "record_version": "2.0"}
    (tmp_path / GUIDED_DIR / "newer").mkdir()
    (tmp_path / GUIDED_DIR / "newer" / RECORD_NAME).write_text(json.dumps(newer))

    builds, refused = store.scan()
    assert [b.id for b in builds] == ["first", "third", "second"]
    assert refused["empty"] == "there is no build 'empty'"
    assert refused["broken"].startswith(f"{RECORD_NAME}:")
    assert refused["renamed"] == "the record's id is 'second', the directory is 'renamed'"
    assert "record_version" in refused["newer"] and "2.0" in refused["newer"]
    assert GuidedStore(tmp_path / "nowhere").scan() == ([], {})


def test_the_server_never_lists_a_guided_build_as_a_face(tmp_path: Path) -> None:
    make_face(tmp_path, "test-head")
    store = GuidedStore(tmp_path)
    store.create("guide")
    store.keep("guide", "portrait", "c1/portrait.png", b"a portrait")
    accepted, refused = FaceStore(tmp_path).scan()
    assert [m.id for m in accepted] == ["test-head"] and refused == {}


def test_a_failed_change_writes_nothing(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide")
    path = store.directory("guide") / RECORD_NAME
    before = path.read_bytes()
    with pytest.raises(NotReady):
        store.put("guide", result("build"))
    assert path.read_bytes() == before
    assert [p.name for p in store.directory("guide").iterdir()] == [RECORD_NAME]  # no part file left


@pytest.mark.parametrize(
    "file",
    ["../other/portrait.png", "/etc/passwd", "c1/../../x.png", ".hidden", "c1/.part", "a/b/c/d/e.png", "", RECORD_NAME],
)
def test_a_work_file_stays_inside_its_build(tmp_path: Path, file: str) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide")
    with pytest.raises(GuidedError):
        store.keep("guide", "portrait", file, b"x")
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [RECORD_NAME]


def test_a_work_file_cannot_follow_a_link_out_of_its_build(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.directory("guide") / "c1").symlink_to(outside, target_is_directory=True)
    with pytest.raises(GuidedError):
        store.keep("guide", "portrait", "c1/portrait.png", b"x")
    assert list(outside.iterdir()) == []


def test_a_work_file_needs_its_build_and_a_name(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    with pytest.raises(NoSuchBuild, match="there is no build 'guide'"):
        store.keep("guide", "portrait", "portrait.png", b"x")
    store.create("guide")
    with pytest.raises(GuidedError, match="not a name"):
        store.keep("guide", "The Portrait", "portrait.png", b"x")
    assert not (store.directory("guide") / "portrait.png").exists()
    store.keep("guide", "portrait", "portrait.png", b"one")
    assert store.keep("guide", "portrait", "portrait.png", b"two").bytes == 3  # a step run again writes its file again
    assert (store.directory("guide") / "portrait.png").read_bytes() == b"two"


def test_changes_to_one_build_from_many_threads_all_land(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide")
    names = [f"c{i}" for i in range(24)]
    threads = [threading.Thread(target=store.put, args=("guide", result("source", name))) for name in names]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(store.load("guide").candidates()) == sorted(names)


# ── The editor's routes ───────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def editor(tmp_path: Path):
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    with TestClient(create_editor_app(Settings(assets_dir=tmp_path / "assets"), ed)) as client:
        yield client, GuidedStore(tmp_path / "assets", clock=ticking())


def test_the_editor_lists_the_builds_and_the_steps(editor) -> None:
    client, store = editor
    assert client.get("/guided").json()["builds"] == []
    store.create("older", "Older")
    store.create("guide", "A guide")
    store.change("guide", lambda record, _now: evaluated(record, "c1"))
    store.put("guide", result("pick", "c1"))
    (store.root / "broken").mkdir()

    listed = client.get("/guided").json()
    assert [b["id"] for b in listed["builds"]] == ["guide", "older"]
    assert listed["builds"][0] | {"created": 0, "updated": 0} == {
        "id": "guide",
        "name": "A guide",
        "created": 0,
        "updated": 0,
        "candidates": ["c1"],
        "picked": "c1",
        "face": {"pick": "passed", "scene": "missing", "save": "missing"},
    }
    assert listed["refused"] == {"broken": "there is no build 'broken'"}
    assert [s["name"] for s in listed["steps"]] == list(STEPS)
    assert listed["steps"][2] == {
        "name": "build",
        "per": "candidate",
        "needs": ["source", "portrait"],
        "about": "the LAM head made from the portrait",
    }


def test_the_editor_serves_a_record_with_each_status_and_what_blocks_the_next_step(editor) -> None:
    client, store = editor
    store.create("guide", "A guide")
    store.put("guide", result("source", "c1"))
    store.put("guide", result("portrait", "c1", check("temples", "advisory", "bare at the temple")))
    store.put("guide", result("source", "c2"))

    view = client.get("/guided/guide").json()
    assert view["id"] == "guide" and view["record_version"] == "1.0" and view["picked"] is None
    by_step = {(r["step"], r["candidate"]): r for r in view["results"]}
    assert by_step[("portrait", "c1")]["status"] == "advisory"
    assert by_step[("portrait", "c1")]["digest"] == store.load("guide").get("portrait", "c1").digest
    assert by_step[("portrait", "c1")]["after"] == {"source": by_step[("source", "c1")]["digest"]}
    c1, c2 = view["candidates"]["c1"], view["candidates"]["c2"]
    assert c1["steps"]["portrait"] == {"status": "advisory", "blockers": []}
    assert c1["steps"]["build"] == {
        "status": "missing",
        "blockers": ["portrait has an advisory to overrule (temples: bare at the temple)"],
    }
    assert c2["steps"]["portrait"] == {"status": "missing", "blockers": []}
    assert c2["steps"]["build"]["blockers"] == ["portrait has not run"]
    assert c1["pick"] == [f"{step} has not run" for step in EVALS]
    assert view["face"] == {
        "pick": {"status": "missing", "candidate": None},
        "scene": {"status": "missing", "blockers": ["pick has not run"]},
        "save": {"status": "missing", "blockers": ["pick has not run", "scene has not run"]},
    }


def test_the_editor_says_why_there_is_no_record(editor) -> None:
    client, store = editor
    assert client.get("/guided/nobody").status_code == 404
    assert client.get("/guided/Not-An-Id").status_code == 404
    (store.root / "broken").mkdir(parents=True)
    (store.root / "broken" / RECORD_NAME).write_text("{")
    reply = client.get("/guided/broken")
    assert reply.status_code == 422 and RECORD_NAME in reply.json()["detail"]


def test_made_up_build_ids_leave_nothing_behind(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    before = len(store._locks)
    for build_id in [f"nobody-{i}" for i in range(50)] + ["../x", "Not An Id"]:
        with pytest.raises(NoSuchBuild):
            store.change(build_id, lambda record, now: None)
    assert len(store._locks) == before


def test_files_land_with_the_result_that_made_them(tmp_path: Path) -> None:
    store = GuidedStore(tmp_path)
    store.create("guide")
    store.put("guide", StepResult(step="source", candidate="c1"), {"c1/a.png": b"one"})
    assert store.path("guide", "c1/a.png").read_bytes() == b"one"
    with pytest.raises(NotReady):
        store.put("guide", StepResult(step="build", candidate="c1"), {"c1/a.png": b"two"})
    assert store.path("guide", "c1/a.png").read_bytes() == b"one"
    assert not list(store.directory("guide").rglob("*.part"))
