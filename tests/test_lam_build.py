"""The LAM build of a guided face (#155): the job around ``lam/make_heads.sh``.

No LAM, GPU or docker here: the script, docker and Blender are small stand-ins, written by each test, that leave
what the real ones leave (a log, a packed head) and say when they ran. The real pipeline is run by hand
(``lam/README.md``).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from argus_cortex.gpu import GpuLeaseRefused
from starlette.testclient import TestClient

from argus_mask import gpu as mask_gpu
from argus_mask.config import EditorSettings, LamSettings, Settings
from argus_mask.editor import lam_build
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedError, GuidedStore, NotReady, StepResult
from argus_mask.editor.lam_build import Busy, LamBuilds, LamError, LamJob, host_free_gb
from argus_mask.faces import sha256_file
from tests.helpers import make_splat_zip

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

# What the stand-in script does around its body, as make_heads.sh does: one name=portrait, a log, a packed head.
SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
pair=$1; name=${pair%%=*}; img=${pair#*=}
mkdir -p "$WORK" "$HEADS/$name"
echo "$img|${FIT_SHAPE:-}|$LAM_CONTAINER" >> "$FAKE/runs"
if [ -e "$FAKE/leased" ]; then echo held >> "$FAKE/lease"; else echo free >> "$FAKE/lease"; fi
mkdir "$FAKE/lock" 2>/dev/null || echo overlap >> "$FAKE/overlaps"
cp "$img" "$WORK/$name.png"
checked="FLAME core: argus-mask's; FLAME model: 2023 Open; painted texture: grey"
BODY
rmdir "$FAKE/lock"
"""
BUILDS = r"""
printf '%s\n' "$checked" "DONE (519, 512, 512, 3) peak VRAM GB 8.68" "EXPORTED /work/out-$name/oac" > "$WORK/$name.log"
cp "$FAKE/ready.zip" "$HEADS/$name/$name.zip"
echo "$name: $HEADS/$name/$name.zip"
"""
# LAM loading until its container is killed; then, as the real script, no EXPORTED in the log: status 1.
HANGS = r"""
echo "loading LAM" > "$WORK/$name.log"
while [ ! -e "$FAKE/killed-$LAM_CONTAINER" ]; do sleep 0.02; done
rmdir "$FAKE/lock"
echo "$name: LAM failed, see $WORK/$name.log"; exit 1
"""
DOCKER = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE/docker"
case "$1" in
  image) [ -e "$FAKE/no-image" ] && exit 1; echo "sha256:$(cat "$FAKE/image")";;
  kill) touch "$FAKE/killed-$2";;
esac
"""


class Gpu:
    """A coordinator that says, in a file the script can see, while a lease is held."""

    def __init__(self, fake: Path, refuse: bool = False) -> None:
        self.fake, self.refuse, self.asked = fake, refuse, []

    @contextlib.asynccontextmanager
    async def alease(self, **options: object) -> AsyncIterator[None]:
        self.asked.append(options)
        if self.refuse:
            raise GpuLeaseRefused("the GPU is busy")
        (self.fake / "leased").touch()
        try:
            yield
        finally:
            (self.fake / "leased").unlink()


class Rig:
    """An assets root with a guided build, and the stand-ins a LAM build runs."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fake = tmp_path / "fake"
        self.lam = tmp_path / "lam"
        for directory in (self.fake, self.lam, tmp_path / "LAM" / "model_zoo", tmp_path / "bin"):
            directory.mkdir(parents=True)
        monkeypatch.setenv("FAKE", str(self.fake))
        monkeypatch.setattr(lam_build, "WATCH_S", 0.02)
        monkeypatch.setattr(lam_build, "HALT_WAIT_S", 5.0)
        (self.fake / "image").write_text("11" * 32)
        make_splat_zip(self.fake / "ready.zip", folder="head")
        (tmp_path / "flame.pkl").write_bytes(b"FLAME 2023 Open")
        self.script(BUILDS)
        for name, text in (("docker", DOCKER), ("blender", "#!/usr/bin/env bash\necho 'Blender 4.5.3 LTS'\n")):
            (tmp_path / "bin" / name).write_text(text)
            (tmp_path / "bin" / name).chmod(0o755)
        self.settings = LamSettings(
            script=self.lam / "make_heads.sh",
            lam_dir=tmp_path / "LAM",
            flame_open=tmp_path / "flame.pkl",
            blender=str(tmp_path / "bin" / "blender"),
            docker=str(tmp_path / "bin" / "docker"),
            timeout_s=20,
        )
        self.free = 16.0
        self.store = GuidedStore(tmp_path / "assets")
        self.store.create("guide")
        self.gpu = Gpu(self.fake)

    def script(self, body: str) -> None:
        (self.lam / "make_heads.sh").write_text(SCRIPT.replace("BODY", body))

    def builds(self, **over: object) -> LamBuilds:
        settings = LamSettings(**{**self.settings.__dict__, **over})
        return LamBuilds(self.store, settings, self.gpu, free_gb=lambda: self.free)

    def candidate(self, name: str = "c1", portrait: bytes = b"a portrait", *, checks: tuple[Check, ...] = ()) -> str:
        """A candidate with its portrait kept and its portrait checks run (passed, unless *checks* say otherwise)."""
        kept = self.store.keep("guide", "portrait", f"candidates/{name}/portrait.png", portrait)
        self.store.put("guide", StepResult(step="source", candidate=name, outputs=[kept]))
        self.store.put("guide", StepResult(step="portrait", candidate=name, checks=list(checks)))
        return name

    def said(self, name: str) -> list[str]:
        path = self.fake / name
        return path.read_text().splitlines() if path.exists() else []

    def built(self, candidate: str = "c1") -> StepResult | None:
        return self.store.load("guide").get("build", candidate)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


async def loading(job: LamJob) -> None:
    """Wait until the stand-in for LAM is running in *job* and has been seen to be."""
    for _ in range(1000):
        if job.stage == "building the head: loading LAM":
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"{job.id} never got to LAM: {job.state}, {job.stage}")


async def finished(job: LamJob) -> LamJob:
    await asyncio.wait_for(asyncio.gather(job.task, return_exceptions=True), 30)
    await asyncio.sleep(0)  # the task's done callback
    return job


async def build(rig: Rig, candidate: str = "c1", **options: object) -> tuple[LamJob, StepResult | None]:
    job = await finished(rig.builds().start("guide", candidate, **options))
    return job, rig.built(candidate)


# ── A portrait becomes a head in the record ───────────────────────────────────────────────────────────────────


async def test_a_portrait_becomes_a_head_in_the_record(rig: Rig) -> None:
    rig.candidate()
    job, found = await build(rig)
    assert (job.state, job.status, job.error, job.cached) == ("done", "passed", None, False)
    assert rig.store.load("guide").status("build", "c1") == "passed"
    assert [(c.name, c.verdict) for c in found.checks] == [
        ("memory", "passed"),
        ("pipeline", "passed"),
        ("lam", "passed"),
        ("head", "passed"),
    ]
    # The portrait in, by the sha256 the source recorded; the head out, kept in the build and as it was packed.
    assert [(a.name, a.sha256) for a in found.inputs] == [("portrait", hashlib.sha256(b"a portrait").hexdigest())]
    head = found.outputs[0]
    assert (head.name, head.file) == ("head", "candidates/c1/head.zip")
    assert sha256_file(rig.store.path("guide", head.file)) == head.sha256 == sha256_file(rig.fake / "ready.zip")
    assert found.checks[-1].measure == {"splats": 3.0, "shapes": 6.0}
    # How it went is beside what it found: the log, its tail, the time, the memory free, LAM's peak VRAM.
    assert found.run["peak_vram_gb"] == 8.7 and found.run["free_gb"] == 16.0 and found.run["job"] == job.id
    assert found.run["log_tail"][-1] == "EXPORTED /work/out-head/oac"
    assert "EXPORTED" in rig.store.path("guide", found.run["log"]).read_text()
    assert found.started <= found.finished
    # The script got the build's own copy of the portrait, a container name to be stopped by, and no shape fit.
    portrait, fit, container = rig.said("runs")[0].split("|")
    assert Path(portrait) == rig.store.path("guide", "candidates/c1/portrait.png")
    assert (fit, container) == ("", f"argus-mask-lam-{job.id}")
    assert not (rig.store.root / ".work" / job.id).exists()  # its scratch is gone


async def test_every_part_a_head_depends_on_is_listed_with_its_licence(rig: Rig) -> None:
    rig.candidate()
    _, found = await build(rig)
    tools = {t.name: t for t in found.tools}
    assert tools["FLAME 2023 Open"].sha256 == sha256_file(rig.settings.flame_open)
    assert tools["FLAME 2023 Open"].license == "CC BY 4.0"
    assert tools["LAM weights (lam-20k)"].license == "CC BY-NC 4.0"
    assert tools["image argus-mask-lam"].version == "sha256:" + "11" * 32
    assert tools["Blender"].version == "Blender 4.5.3 LTS"
    assert len(tools["argus-mask lam/ pipeline"].sha256) == 64
    unlicensed = [t.name for t in found.tools if not t.license and not t.name.startswith("image ")]
    assert unlicensed == []
    # lam/README.md's table is where these come from: every part it lists as used is one of them.
    readme = (Path(__file__).resolve().parent.parent / "lam" / "README.md").read_text(encoding="utf-8")
    for part in ("LAM weights", "VHAP tracker", "VGGHeads", "MediaPipe Face Landmarker", "BiRefNet-portrait"):
        assert part in readme and any(part in name for name in tools), part


async def test_the_shape_fit_is_asked_for_by_the_request_alone(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    """``fit_shape`` reaches the script as FIT_SHAPE=1 and its misfits come back as scores; the editor's own
    environment never turns it on."""
    monkeypatch.setenv("FIT_SHAPE", "1")
    fit = r"""
mkdir -p "$WORK/out-$name/fit"; echo '{"best": {}}' > "$WORK/out-$name/fit/fit.json"; echo png > "$WORK/out-$name/fit/best.png"
echo 'FIT {"start": {"total": 0.0334, "eyes": 0.02}, "best": {"total": 0.0261, "eyes": 0.01}}' > "$WORK/$name.fit"
""" + BUILDS.replace('> "$WORK/$name.log"', '> "$WORK/$name.log"; cat "$WORK/$name.fit" >> "$WORK/$name.log"')
    rig.script(fit)
    rig.candidate()
    _, plain = await build(rig)
    assert rig.said("runs")[0].split("|")[1] == ""
    _, fitted = await build(rig, fit_shape=True)
    assert rig.said("runs")[1].split("|")[1] == "1"
    assert fitted.scores == {"shape_misfit_start": 0.0334, "shape_misfit": 0.0261}
    assert [a.name for a in fitted.outputs] == ["head", "shape-fit", "shape-fit-best"]
    assert fitted.data["fit_shape"] is True and fitted.data["key"] != plain.data["key"]


# ── It starts only on settled ground, with the portrait its source kept ───────────────────────────────────────


async def test_a_build_never_starts_on_a_portrait_a_check_refused(rig: Rig) -> None:
    """Refused, advisory nobody overruled, or not checked: nothing runs, and nothing is asked of docker or the GPU."""
    for name, verdict in (("c1", "refused"), ("c2", "advisory"), ("c3", "not_checked"), ("c4", "failed")):
        rig.candidate(name, checks=(Check(name="frontal", verdict=verdict, reason="turned away"),))
        with pytest.raises(NotReady, match="portrait"):
            rig.builds().start("guide", name)
    kept = rig.store.keep("guide", "portrait", "candidates/c5/portrait.png", b"p")
    rig.store.put("guide", StepResult(step="source", candidate="c5", outputs=[kept]))
    with pytest.raises(NotReady, match="portrait has not run"):
        rig.builds().start("guide", "c5")
    assert rig.said("runs") == rig.said("docker") == [] and rig.gpu.asked == []
    # An advisory someone went on past is settled ground.
    rig.store.overrule("guide", "portrait", "frontal", "it is close enough", candidate="c2")
    job = await finished(rig.builds().start("guide", "c2"))
    assert job.state == "done"


async def test_a_portrait_refused_while_the_build_waited_is_not_built(rig: Rig) -> None:
    """The record is asked again when the job leaves the queue: a check run meanwhile stops it."""
    rig.script(HANGS)
    rig.candidate("c1")
    rig.candidate("c2")
    builds = rig.builds()
    first, second = builds.start("guide", "c1"), builds.start("guide", "c2")
    await loading(first)
    refused = Check(name="frontal", verdict="refused", reason="turned away")
    rig.store.put("guide", StepResult(step="portrait", candidate="c2", checks=[refused]))
    await builds.stop(first.id)
    await finished(second)
    assert second.state == "failed" and "portrait refused" in second.error
    assert ["/c1/" in line for line in rig.said("runs")] == [True] and rig.built("c2") is None


async def test_the_portrait_is_the_one_its_source_recorded(rig: Rig) -> None:
    rig.candidate()
    path = rig.store.path("guide", "candidates/c1/portrait.png")
    path.write_bytes(b"another face")
    with pytest.raises(GuidedError, match="changed since its source"):
        rig.builds().start("guide", "c1")
    path.unlink()
    with pytest.raises(GuidedError, match="not where its source kept it"):
        rig.builds().start("guide", "c1")
    rig.store.put("guide", StepResult(step="source", candidate="c9"))
    rig.store.put("guide", StepResult(step="portrait", candidate="c9"))
    with pytest.raises(GuidedError, match="has no portrait"):
        rig.builds().start("guide", "c9")
    assert rig.said("runs") == []


# ── Host memory: checked before, watched during ───────────────────────────────────────────────────────────────


async def test_a_build_needs_twelve_gb_of_host_memory_free(rig: Rig) -> None:
    rig.candidate()
    assert rig.settings.min_free_gb == 12.0
    rig.free = 11.9
    with pytest.raises(LamError, match=r"11\.9 GB of host memory is free and a LAM build needs 12 GB"):
        rig.builds().start("guide", "c1")
    rig.free = None  # it cannot be read: never taken for enough
    with pytest.raises(LamError, match="could not be read"):
        rig.builds().start("guide", "c1")
    assert rig.said("runs") == [] and rig.gpu.asked == [] and rig.built() is None


async def test_memory_is_asked_for_again_when_a_build_leaves_the_queue(rig: Rig) -> None:
    """Enough when it was queued, short when its turn came: it does not run, and the record is left as it was."""
    rig.script(HANGS)
    rig.candidate("c1")
    rig.candidate("c2")
    builds = rig.builds()
    first, second = builds.start("guide", "c1"), builds.start("guide", "c2")
    await loading(first)
    assert (first.state, second.state) == ("running", "queued")
    rig.free = 8.0  # above the floor a running build is stopped at, under what a new one needs
    await builds.stop(first.id)
    await finished(second)
    assert second.state == "failed" and "8.0 GB of host memory is free" in second.error
    assert len(rig.said("runs")) == 1 and rig.built("c2") is None


async def test_a_build_is_stopped_when_host_memory_runs_low(rig: Rig) -> None:
    rig.script(HANGS)
    rig.candidate()
    job = rig.builds().start("guide", "c1")
    await loading(job)  # the page is told what LAM's log last said
    assert job.state == "running"
    rig.free = 1.4
    await finished(job)
    assert f"kill argus-mask-lam-{job.id}" in rig.said("docker")  # LAM's container, by the name it was given
    found = rig.built()
    assert job.state == "failed" and found.standing == "failed"
    assert found.named("failed") == "lam: stopped: host memory fell to 1.4 GB free (under 2 GB)"
    assert found.outputs == [] and rig.said("lease") == ["held"] and not (rig.fake / "leased").exists()


async def test_a_build_that_takes_too_long_is_stopped(rig: Rig) -> None:
    rig.script(HANGS)
    rig.candidate()
    job = await finished(rig.builds(timeout_s=0.2).start("guide", "c1"))
    assert f"kill argus-mask-lam-{job.id}" in rig.said("docker")
    assert rig.built().named("failed") == "lam: stopped: it took more than 0.2 s"


async def test_a_script_that_will_not_end_is_killed_with_its_container(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The script still running after its container was killed (it had not started it yet): its whole process
    group is killed, and the container is killed again, since its client is gone and it would run on."""
    monkeypatch.setattr(lam_build, "HALT_WAIT_S", 0.2)
    rig.script('echo starting > "$WORK/$name.log"\nsleep 60 & wait\n')
    rig.candidate()
    job = await finished(rig.builds(timeout_s=0.2).start("guide", "c1"))
    assert rig.said("docker").count(f"kill argus-mask-lam-{job.id}") == 2
    assert job.state == "failed" and rig.built().standing == "failed"


async def test_stopping_a_build_kills_its_container_and_writes_nothing(rig: Rig) -> None:
    rig.script(HANGS)
    rig.candidate()
    builds = rig.builds()
    job = builds.start("guide", "c1")
    await loading(job)
    with pytest.raises(Busy):
        builds.start("guide", "c1")
    stopped = await builds.stop(job.id)
    await asyncio.sleep(0)
    assert stopped is job and job.state == "stopped" and not job.live
    assert f"kill argus-mask-lam-{job.id}" in rig.said("docker")
    assert rig.built() is None and not (rig.fake / "leased").exists()
    assert not (rig.store.root / ".work" / job.id).exists()
    rig.script(BUILDS)
    assert (await finished(builds.start("guide", "c1"))).state == "done"  # and the candidate can be built again


# ── The lease: only while the pipeline runs ───────────────────────────────────────────────────────────────────


async def test_the_lam_head_lease_is_held_only_while_the_pipeline_runs(rig: Rig) -> None:
    rig.candidate()
    job, _ = await build(rig)
    assert rig.said("lease") == ["held"] and not (rig.fake / "leased").exists()
    assert rig.gpu.asked == [{"caller": "argus-mask:lam-head:guide:c1", **mask_gpu.LEASES[mask_gpu.LAM_HEAD]}]
    # A head that is kept is copied, not built: no lease is asked for.
    again, _ = await build(rig)
    assert again.cached and len(rig.gpu.asked) == 1 and len(rig.said("runs")) == 1
    assert job.state == again.state == "done"


async def test_no_gpu_fails_the_job_and_leaves_the_record_as_it_was(rig: Rig) -> None:
    rig.candidate()
    rig.gpu.refuse = True
    job, found = await build(rig)
    assert job.state == "failed" and job.error == "no GPU for the head yet: the GPU is busy"
    assert found is None and rig.said("runs") == []


# ── One head at a time ────────────────────────────────────────────────────────────────────────────────────────


async def test_heads_are_built_one_at_a_time_in_the_order_asked(rig: Rig) -> None:
    builds = rig.builds()
    jobs = [builds.start("guide", rig.candidate(f"c{i}", f"portrait {i}".encode())) for i in range(4)]
    assert jobs[0].stage == "waiting for its turn"
    for job in jobs:
        await finished(job)
    assert [j.state for j in jobs] == ["done"] * 4
    assert rig.said("overlaps") == []
    ran = [Path(line.split("|")[0]).parent.name for line in rig.said("runs")]
    assert ran == ["c0", "c1", "c2", "c3"]
    assert builds.state() == {"ready": True, "reason": None, "jobs": []}


# ── The same portrait gives the same head ─────────────────────────────────────────────────────────────────────


async def test_the_same_portrait_through_the_same_pipeline_is_not_built_twice(rig: Rig) -> None:
    rig.candidate("c1")
    rig.candidate("c2")  # the same portrait, as another candidate
    rig.candidate("c3", b"another portrait")
    _, first = await build(rig, "c1")
    job, again = await build(rig, "c2")
    assert job.cached and again.run == {"job": job.id, "cached": True}
    assert len(rig.said("runs")) == 1
    assert again.outputs[0].sha256 == first.outputs[0].sha256
    assert rig.store.path("guide", "candidates/c2/head.zip").read_bytes() == (rig.fake / "ready.zip").read_bytes()
    assert [c.model_dump() for c in again.checks] == [c.model_dump() for c in first.checks]
    # What the build found is the same, so nothing that ran on it goes stale when it is built again.
    _, rebuilt = await build(rig, "c1")
    assert rebuilt.digest == first.digest and len(rig.said("runs")) == 1
    _, other = await build(rig, "c3")
    assert len(rig.said("runs")) == 2 and other.data["key"] != first.data["key"]


@pytest.mark.parametrize("change", ["script", "image", "flame"])
async def test_a_changed_pipeline_builds_the_head_again(rig: Rig, change: str) -> None:
    rig.candidate()
    _, first = await build(rig)
    if change == "script":
        (rig.lam / "make_head.py").write_text("# the pipeline's own code, edited\n")
    elif change == "image":
        (rig.fake / "image").write_text("22" * 32)
    else:
        rig.settings.flame_open.write_bytes(b"another FLAME")
    job, again = await build(rig)
    assert not job.cached and len(rig.said("runs")) == 2
    assert again.data["key"] != first.data["key"]


async def test_a_kept_head_that_was_changed_is_dropped_and_built_again(rig: Rig) -> None:
    rig.candidate()
    _, first = await build(rig)
    kept = rig.store.root / ".cache" / "lam" / first.data["key"]
    (kept / "head.zip").write_bytes(b"not the head that was kept")
    job, again = await build(rig)
    assert not job.cached and len(rig.said("runs")) == 2
    assert again.outputs[0].sha256 == first.outputs[0].sha256


async def test_only_the_latest_heads_are_kept(rig: Rig) -> None:
    for i in range(3):
        rig.candidate(f"c{i}", f"portrait {i}".encode())
        await finished(rig.builds(cache=2).start("guide", f"c{i}"))
    kept = [p for p in (rig.store.root / ".cache" / "lam").iterdir()]
    assert len(kept) == 2 and rig.built("c0").data["key"] not in [p.name for p in kept]
    job = await finished(rig.builds(cache=0).start("guide", "c2"))
    assert not job.cached  # 0: nothing is kept, and nothing kept is used


# ── What a run that went wrong leaves in the record ───────────────────────────────────────────────────────────


async def test_the_pipelines_own_refusal_is_a_refusal_with_its_reason(rig: Rig) -> None:
    """make_head.py's preflight refusing (not FLAME 2023 Open): the build is refused, never merely failed."""
    why = "model_zoo/x/flame2023.pkl is not FLAME 2023 Open (sha256 0123456789ab…): mount flame2023_Open.pkl over it"
    rig.script(f'echo "{why}" > "$WORK/$name.log"\nrmdir "$FAKE/lock"\necho "$name: LAM failed"; exit 1\n')
    rig.candidate()
    job, found = await build(rig)
    assert (job.state, found.standing) == ("failed", "refused")
    assert found.named("refused") == f"pipeline: {why}"
    assert found.outputs == [] and not (rig.store.root / ".cache").exists()
    assert rig.store.load("guide").blockers("likeness", "c1") == [f"build refused (pipeline: {why})"]


async def test_a_head_built_without_the_pipelines_checks_showing_is_refused(rig: Rig) -> None:
    rig.script(BUILDS.replace('"$checked" ', ""))
    rig.candidate()
    _, found = await build(rig)
    assert found.standing == "refused" and "does not show make_head.py's own checks passing" in found.named("refused")
    assert not (rig.store.root / ".cache").exists()


async def test_a_crash_is_a_failure_with_the_log_s_last_lines(rig: Rig) -> None:
    # In a terminal's colours and behind a progress bar's redraws, as LAM's own log is.
    said = r'"$checked" "t:  64%|###   | 331/520\rt: 100%|#####| 520/520" "\033[92mRuntimeError: CUDA out of memory\033[0m"'
    rig.script(f'printf "%b\\n" {said} > "$WORK/$name.log"\nexit 1\n')
    rig.candidate()
    job, found = await build(rig)
    assert (job.state, found.standing) == ("failed", "failed")
    assert "ended with status 1" in job.error and "CUDA out of memory" in job.error
    assert found.run["log_tail"] == [
        "FLAME core: argus-mask's; FLAME model: 2023 Open; painted texture: grey",
        "t:  64%|###   | 331/520",
        "t: 100%|#####| 520/520",
        "RuntimeError: CUDA out of memory",
    ]


async def test_a_packed_head_the_server_would_not_serve_is_refused(rig: Rig) -> None:
    make_splat_zip(rig.fake / "ready.zip", {"skin.glb": b"not a glb"}, folder="head")
    rig.candidate()
    _, found = await build(rig)
    assert found.standing == "refused"
    assert found.named("refused").startswith("head: head.zip: missing offset.ply")
    assert found.outputs == []


# ── Whether a build can run here ──────────────────────────────────────────────────────────────────────────────


def test_a_host_that_cannot_build_says_why(rig: Rig, tmp_path: Path) -> None:
    rig.candidate()
    for over, why in (
        ({"script": None}, "lam/make_heads.sh was not found"),
        ({"script": tmp_path / "nowhere.sh"}, "lam/make_heads.sh was not found"),
        ({"lam_dir": None}, "no LAM checkout"),
        ({"lam_dir": tmp_path}, "not a LAM checkout with its model_zoo"),
        ({"flame_open": None}, "no FLAME 2023 Open: set ARGUS_MASK_FLAME_OPEN"),
        ({"flame_open": tmp_path / "nowhere.pkl"}, "no FLAME 2023 Open at"),
        ({"docker": "no-such-docker"}, "docker not found"),
        ({"blender": "no-such-blender"}, "Blender not found"),
    ):
        builds = rig.builds(**over)
        assert why in builds.check(), over
        assert builds.state()["ready"] is False
        with pytest.raises(LamError, match=why):
            builds.start("guide", "c1")
    assert rig.builds().check() is None


async def test_no_image_fails_the_job_before_the_gpu_is_asked(rig: Rig) -> None:
    (rig.fake / "no-image").touch()
    rig.candidate()
    job, found = await build(rig)
    assert job.state == "failed" and "the image 'argus-mask-lam' is not built" in job.error
    assert found is None and rig.gpu.asked == [] and rig.said("runs") == []


def test_free_memory_is_what_the_kernel_says_is_available(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       32000000 kB\nMemFree:         1000000 kB\nMemAvailable:   14680064 kB\n")
    assert host_free_gb(meminfo) == 14.0  # available, not free: the page cache can be given up
    meminfo.write_text("MemTotal: 32000000 kB\n")
    assert host_free_gb(meminfo) is None
    assert host_free_gb(tmp_path / "nowhere") is None


def test_the_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("LAM_SCRIPT", "LAM_DIR", "FLAME_OPEN", "LAM_IMAGE", "LAM_BLENDER", "BLENDER", "LAM_MIN_FREE_GB"):
        monkeypatch.delenv(f"ARGUS_MASK_{name}", raising=False)
    found = LamSettings.from_env()
    # In a checkout the pipeline is the one beside src/; LAM itself and FLAME are never guessed at.
    assert found.script == Path(__file__).resolve().parent.parent / "lam" / "make_heads.sh"
    assert (found.lam_dir, found.flame_open, found.image, found.min_free_gb) == (None, None, "argus-mask-lam", 12.0)
    monkeypatch.setenv("ARGUS_MASK_LAM_DIR", str(tmp_path))
    monkeypatch.setenv("ARGUS_MASK_BLENDER", "blender-4.5")
    monkeypatch.setenv("ARGUS_MASK_LAM_MIN_FREE_GB", "16")
    found = LamSettings.from_env()
    assert (found.lam_dir, found.blender, found.min_free_gb) == (tmp_path, "blender-4.5", 16.0)
    monkeypatch.setenv("ARGUS_MASK_LAM_MIN_FREE_GB", "lots")
    with pytest.raises(ValueError, match="LAM_MIN_FREE_GB 'lots' is not a number"):
        LamSettings.from_env()


# ── The editor's routes ───────────────────────────────────────────────────────────────────────────────────────


def test_the_editor_builds_a_candidates_head_and_reports_the_job(rig: Rig, tmp_path: Path) -> None:
    rig.candidate()
    rig.candidate("c2", checks=(Check(name="frontal", verdict="refused", reason="turned away"),))
    assets = Settings(assets_dir=tmp_path / "assets")
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    with TestClient(create_editor_app(assets, ed, lam=rig.builds())) as client:
        assert client.get("/guided").json()["lam"] == {"ready": True, "reason": None, "jobs": []}
        started = client.post("/guided/guide/candidates/c1/build")
        assert started.status_code == 200, started.text
        job = started.json()
        assert (job["build"], job["candidate"], job["fit_shape"]) == ("guide", "c1", False)
        for _ in range(200):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        assert (job["state"], job["status"]) == ("done", "passed")
        view = client.get("/guided/guide").json()
        assert view["candidates"]["c1"]["steps"]["build"] == {"status": "passed", "blockers": []}
        assert client.post("/guided/guide/candidates/c1/build", json={"fit_shape": True}).json()["fit_shape"] is True

        refused = client.post("/guided/guide/candidates/c2/build")
        assert refused.status_code == 409 and "portrait refused" in refused.json()["detail"]
        assert client.post("/guided/nowhere/candidates/c1/build").status_code == 404
        for not_a_candidate in (".hidden", "C1", "c1;rm"):
            assert client.post(f"/guided/guide/candidates/{not_a_candidate}/build").status_code == 422
        assert client.post("/guided/guide/candidates/c1/build", json={"portrait": "/etc/passwd"}).status_code == 422
        assert client.get(f"/guided/other/jobs/{job['id']}").status_code == 404  # a job is its own build's
        assert client.get("/guided/guide/jobs/nothing").status_code == 404
        assert client.post("/guided/guide/jobs/nothing/stop").status_code == 404
        assert client.post(f"/guided/guide/jobs/{job['id']}/stop").json()["state"] == "done"


def test_the_editor_says_why_a_head_cannot_be_built_here(rig: Rig, tmp_path: Path) -> None:
    rig.candidate()
    assets = Settings(assets_dir=tmp_path / "assets")
    ed = EditorSettings(voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    with TestClient(create_editor_app(assets, ed, lam=rig.builds(lam_dir=None))) as client:
        state = client.get("/guided").json()["lam"]
        assert state["ready"] is False and "ARGUS_MASK_LAM_DIR" in state["reason"]
        reply = client.post("/guided/guide/candidates/c1/build")
        assert reply.status_code == 503 and "ARGUS_MASK_LAM_DIR" in reply.json()["detail"]


def test_the_real_script_names_lams_container_only_when_asked() -> None:
    """The watchdog stops LAM by the container's name: make_heads.sh gives the GPU step that name, and no other."""
    script = (Path(__file__).resolve().parent.parent / "lam" / "make_heads.sh").read_text(encoding="utf-8")
    named = [line for line in script.splitlines() if "LAM_CONTAINER" in line and not line.startswith("#")]
    assert len(named) == 1 and '--gpus all ${LAM_CONTAINER:+--name "$LAM_CONTAINER"}' in named[0]
