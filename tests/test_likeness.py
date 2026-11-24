"""The likeness of a guided face's head to its portrait (#169): the job around ``lam/likeness.py``.

No MediaPipe, DINOv2 or docker image here: docker is a stand-in that writes what the real script writes
(``likeness.json`` with calibrated scores). The real score is run by hand against the pictures a LAM build keeps
(``lam/likeness.py``, calibrated on #151's heads).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, LikenessSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedError, GuidedStore, NotReady, StepResult
from argus_mask.editor.likeness import PORTRAIT, RENDER, Busy, LikenessJobs
from tests.helpers import make_splat_zip

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

PNG = b"\x89PNG\r\n\x1a\n" + b"portrait"
RENDER_PNG = b"\x89PNG\r\n\x1a\n" + b"render"

SCORES = {
    "colour_skin": 1.2,
    "colour_lips": 2.0,
    "colour_iris": 4.5,
    "colour_hair": 2.5,
    "colour_mean": 2.55,
    "aperture": -0.05,
    "lip_ratio": -0.12,
    "shape_misfit": 0.028,
    "identity": 0.94,
}

# A docker that finds the work mount and copies FAKE/scores.json into it, as the LAM image's script would write.
DOCKER = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE/docker"
work=; prev=
for arg in "$@"; do
  if [ "$prev" = "-v" ]; then
    case "$arg" in
      *:\/work) work=${arg%%:*};;
      *:\/dinov2:ro) touch "$FAKE/dinov2-mounted";;
    esac
  fi
  prev=$arg
done
[ -n "$work" ] || exit 1
[ -e "$FAKE/no-image" ] && exit 125
cp "$FAKE/scores.json" "$work/likeness.json"
echo LIKENESS
"""


class Rig:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fake = tmp_path / "fake"
        self.lam = tmp_path / "lam"
        for directory in (self.fake, self.lam, tmp_path / "bin"):
            directory.mkdir(parents=True)
        monkeypatch.setenv("FAKE", str(self.fake))
        (self.fake / "scores.json").write_text(
            json.dumps(
                {
                    "colour": {"skin": 1.2, "lips": 2.0, "iris": 4.5, "hair": 2.5, "mean": 2.55},
                    "shape": {"total": 0.028, "aperture": -0.05, "lip_ratio": -0.12},
                    "identity": 0.94,
                    "scores": SCORES,
                }
            )
        )
        (self.lam / "likeness.py").write_text("# stand-in; docker writes the scores\n")
        docker = tmp_path / "bin" / "docker"
        docker.write_text(DOCKER)
        docker.chmod(0o755)
        self.dinov2 = tmp_path / "dinov2"
        self.dinov2.mkdir()
        (self.dinov2 / "model.safetensors").write_bytes(b"dinov2 weights")
        self.settings = LikenessSettings(
            script=self.lam / "likeness.py",
            docker=str(docker),
            image="argus-mask-lam",
            dinov2=self.dinov2,
            timeout_s=20,
        )
        self.store = GuidedStore(tmp_path / "assets")
        self.store.create("guide")

    def jobs(self, **over: object) -> LikenessJobs:
        return LikenessJobs(self.store, LikenessSettings(**{**self.settings.__dict__, **over}))

    def built(self, name: str = "c1", *, with_pictures: bool = True) -> str:
        """A candidate whose build has passed, with the two pictures likeness scores (unless asked not to)."""
        portrait = self.store.keep("guide", "portrait", f"candidates/{name}/portrait.png", b"a portrait")
        self.store.put("guide", StepResult(step="source", candidate=name, outputs=[portrait]))
        self.store.put("guide", StepResult(step="portrait", candidate=name, checks=[]))
        head = self.store.root / "guide" / f"candidates/{name}/head.zip"
        head.parent.mkdir(parents=True, exist_ok=True)
        make_splat_zip(head, folder="head")
        outputs = [self.store.keep("guide", "head", f"candidates/{name}/head.zip", head.read_bytes())]
        if with_pictures:
            outputs += [
                self.store.keep("guide", PORTRAIT, f"candidates/{name}/likeness/portrait.png", PNG),
                self.store.keep("guide", RENDER, f"candidates/{name}/likeness/render.png", RENDER_PNG),
            ]
        self.store.put(
            "guide",
            StepResult(
                step="build",
                candidate=name,
                checks=[Check(name="head", verdict="passed")],
                outputs=outputs,
                inputs=[portrait],
            ),
        )
        return name


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


async def finished(job):  # noqa: ANN001
    await asyncio.wait_for(asyncio.gather(job.task, return_exceptions=True), 30)
    await asyncio.sleep(0)
    return job


async def score(rig: Rig, candidate: str = "c1", **options: object):
    job = await finished(rig.jobs(**options).start("guide", candidate))
    return job, rig.store.load("guide").get("likeness", candidate)


async def test_a_built_head_gets_its_likeness_scores(rig: Rig) -> None:
    rig.built()
    job, found = await score(rig)
    assert (job.state, job.status, job.error) == ("done", "passed", None)
    assert found.scores == SCORES
    assert [(c.name, c.verdict) for c in found.checks] == [
        ("faces", "passed"),
        ("identity", "passed"),
        ("likeness", "passed"),
    ]
    assert [a.name for a in found.inputs] == [PORTRAIT, RENDER]
    assert found.outputs[0].name == "likeness"
    assert any("DINOv2" in t.name for t in found.tools)
    assert (rig.fake / "dinov2-mounted").exists()


async def test_without_dinov2_identity_is_not_checked_and_the_rest_still_score(rig: Rig) -> None:
    """Identity needs a checkpoint; colour and geometry do not. Scores never refuse a head."""
    (rig.fake / "scores.json").write_text(
        json.dumps(
            {
                "colour": {"skin": 1.0, "lips": 2.0, "iris": 3.0, "hair": 1.5, "mean": 1.875},
                "shape": {"total": 0.03, "aperture": 0.0, "lip_ratio": 0.1},
                "scores": {
                    "colour_skin": 1.0,
                    "colour_lips": 2.0,
                    "colour_iris": 3.0,
                    "colour_hair": 1.5,
                    "colour_mean": 1.875,
                    "aperture": 0.0,
                    "lip_ratio": 0.1,
                    "shape_misfit": 0.03,
                },
            }
        )
    )
    rig.built()
    job, found = await score(rig, dinov2=None)
    assert job.state == "done" and found.standing == "passed"
    assert "identity" not in found.scores
    assert "ARGUS_MASK_DINOV2" in found.run["identity"]
    assert found.scores["colour_mean"] == 1.875
    assert not (rig.fake / "dinov2-mounted").exists()


async def test_likeness_never_starts_before_the_build(rig: Rig) -> None:
    portrait = rig.store.keep("guide", "portrait", "candidates/c1/portrait.png", b"p")
    rig.store.put("guide", StepResult(step="source", candidate="c1", outputs=[portrait]))
    rig.store.put("guide", StepResult(step="portrait", candidate="c1", checks=[]))
    with pytest.raises(NotReady, match="build"):
        rig.jobs().start("guide", "c1")


async def test_a_build_without_its_pictures_is_refused(rig: Rig) -> None:
    rig.built(with_pictures=False)
    with pytest.raises(GuidedError, match="portrait-crop|render"):
        rig.jobs().start("guide", "c1")


async def test_a_picture_that_changed_is_refused(rig: Rig) -> None:
    rig.built()
    rig.store.path("guide", "candidates/c1/likeness/render.png").write_bytes(RENDER_PNG + b"changed")
    job, found = await score(rig)
    assert job.state == "failed" and "changed" in job.error and found is None


async def test_scoring_the_same_candidate_twice_at_once_is_busy(rig: Rig) -> None:
    rig.built()
    jobs = rig.jobs()
    first = jobs.start("guide", "c1")
    with pytest.raises(Busy):
        jobs.start("guide", "c1")
    await finished(first)


async def test_a_script_failure_fails_the_step_and_keeps_no_scores(rig: Rig) -> None:
    Path(rig.settings.docker).write_text("#!/usr/bin/env bash\necho boom >&2; exit 1\n")
    Path(rig.settings.docker).chmod(0o755)
    rig.built()
    job, found = await score(rig)
    assert job.state == "failed" and found.standing == "failed"
    assert "boom" in found.named("failed")
    assert found.scores == {} and found.outputs == []


def test_the_editor_scores_a_candidates_likeness(rig: Rig, tmp_path: Path) -> None:
    rig.built()
    app = create_editor_app(
        Settings(assets_dir=tmp_path / "assets"),
        EditorSettings(),
        likeness=rig.jobs(),
    )
    with TestClient(app) as client:
        listed = client.get("/guided").json()["likeness"]
        assert listed["ready"] is True and listed["dinov2"] is True
        job = client.post("/guided/guide/candidates/c1/likeness").json()
        assert job["step"] == "likeness"
        for _ in range(200):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.02)
        assert job["state"] == "done" and job["status"] == "passed"
        view = client.get("/guided/guide").json()
        assert view["candidates"]["c1"]["steps"]["likeness"]["status"] == "passed"
        likeness = next(r for r in view["results"] if r["step"] == "likeness")
        assert likeness["scores"]["identity"] == 0.94


def test_the_settings_find_dinov2_in_the_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from argus_mask import config

    cache = tmp_path / "dinov2-base"
    cache.mkdir()
    (cache / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(config, "DINOV2_IN_CACHE", cache)
    monkeypatch.setenv("ARGUS_MASK_LIKENESS_SCRIPT", str(tmp_path / "likeness.py"))
    (tmp_path / "likeness.py").write_text("#")
    monkeypatch.delenv("ARGUS_MASK_DINOV2", raising=False)
    assert LikenessSettings.from_env().dinov2 == cache


async def test_asking_for_likeness_reads_no_picture(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """The request runs on the event loop: the hashing is the job's, off it."""
    rig.built()
    read = Path.read_bytes
    during: list[Path] = []
    monkeypatch.setattr(Path, "read_bytes", lambda self: (during.append(self), read(self))[1])
    opened = Path.open
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: (during.append(self), opened(self, *a, **k))[1])
    job = rig.jobs().start("guide", "c1")
    asked = [p for p in during if p.suffix == ".png"]
    await finished(job)
    assert asked == [] and job.state == "done"


HANGING = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE/docker"
if [ "$1" = kill ]; then touch "$FAKE/killed-$2"; exit 0; fi
prev=
for arg in "$@"; do
  if [ "$prev" = "-v" ]; then case "$arg" in *:/work) touch "${arg%%:*}/left-behind"; mkdir "${arg%%:*}/dir";; esac; fi
  prev=$arg
done
exec sleep 60
"""


@pytest.mark.parametrize("ended", ["stopped", "timed out"])
async def test_a_likeness_that_is_stopped_or_too_long_kills_its_container(rig: Rig, ended: str) -> None:
    Path(rig.settings.docker).write_text(HANGING)
    rig.built()
    jobs = rig.jobs(timeout_s=0.5 if ended == "timed out" else 30)
    job = jobs.start("guide", "c1")
    if ended == "stopped":
        for _ in range(500):
            if (rig.fake / "docker").exists():
                break
            await asyncio.sleep(0.02)
        await jobs.stop(job.id)
        assert job.state == "stopped"
    else:
        await finished(job)
        assert "took more than" in job.error
    name = f"argus-mask-likeness-{job.id}"
    assert f"--name {name}" in (rig.fake / "docker").read_text()
    assert (rig.fake / f"killed-{name}").exists()
    assert not list((rig.store.root / ".work").glob("likeness-*"))
