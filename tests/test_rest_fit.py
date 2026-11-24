"""The rest set of a guided face's head, fitted to its portrait (#169): the job around ``lam/fit_rest.py``.

No browser or MediaPipe here: uv and the script are stand-ins that write what the real fit writes (a rest set
with a smile of at least 0.15 and its misfit scores). The real fit is run by hand (``lam/fit_rest.py``).
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from argus_mask.config import EditorSettings, RestSettings, Settings
from argus_mask.editor.app import create_editor_app
from argus_mask.editor.guided import Check, GuidedStore, NotReady, StepResult
from argus_mask.editor.rest_fit import Busy, RestFits
from tests.helpers import make_splat_zip

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")

UV = r"""#!/usr/bin/env bash
echo "$*" >> "$FAKE/uv"
while [ $# -gt 0 ] && [ "$1" != "python" ]; do shift; done
exec "$@"
"""

SCRIPT = r"""
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
out = Path(argv[argv.index("--out") + 1])
name = argv[argv.index("--name") + 1] if "--name" in argv else "fit"
smile = float(os.environ.get("FAKE_SMILE", "0.2"))
Path(os.environ["FAKE"], "runs").write_text(" ".join(sys.argv) + "\n")
weights = {
    "eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05,
    "mouthSmileLeft": smile, "mouthSmileRight": smile,
}
payload = {
    "name": name,
    "weights": weights,
    "life": 1.0,
    "score": {
        "warm": {"total": 0.12, "aperture": 0.1, "lip_ratio": 0.05},
        "fit": {"total": 0.04, "aperture": 0.02, "lip_ratio": 0.01},
    },
}
out.write_text(json.dumps(payload))
out.with_suffix(".fit.png").write_bytes(b"\x89PNG\r\n\x1a\nfit")
print(json.dumps({"name": name, "weights": weights, "score": payload["score"]}))
"""


class Rig:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.fake = tmp_path / "fake"
        self.lam = tmp_path / "lam"
        for directory in (self.fake, self.lam, tmp_path / "bin"):
            directory.mkdir(parents=True)
        monkeypatch.setenv("FAKE", str(self.fake))
        script = self.lam / "fit_rest.py"
        script.write_text(SCRIPT)
        uv = tmp_path / "bin" / "uv"
        uv.write_text(UV)
        uv.chmod(0o755)
        landmarker = tmp_path / "face_landmarker.task"
        landmarker.write_bytes(b"landmarker")
        self.settings = RestSettings(
            script=script,
            landmarker=landmarker,
            uv=str(uv),
            timeout_s=30,
            generations=2,
        )
        self.store = GuidedStore(tmp_path / "assets")
        self.store.create("guide")

    def fits(self, **over: object) -> RestFits:
        return RestFits(self.store, RestSettings(**{**self.settings.__dict__, **over}))

    def built(self, name: str = "c1") -> str:
        portrait = self.store.keep("guide", "portrait", f"candidates/{name}/portrait.png", b"a portrait")
        self.store.put("guide", StepResult(step="source", candidate=name, outputs=[portrait]))
        self.store.put("guide", StepResult(step="portrait", candidate=name, checks=[]))
        head = self.store.root / "guide" / f"candidates/{name}/head.zip"
        head.parent.mkdir(parents=True, exist_ok=True)
        make_splat_zip(head, folder="head")
        self.store.put(
            "guide",
            StepResult(
                step="build",
                candidate=name,
                checks=[Check(name="head", verdict="passed")],
                outputs=[self.store.keep("guide", "head", f"candidates/{name}/head.zip", head.read_bytes())],
                inputs=[portrait],
            ),
        )
        return name


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


async def finished(job):  # noqa: ANN001
    await asyncio.wait_for(asyncio.gather(job.task, return_exceptions=True), 60)
    await asyncio.sleep(0)
    return job


async def fit(rig: Rig, candidate: str = "c1", **options: object):
    job = await finished(rig.fits(**options).start("guide", candidate))
    return job, rig.store.load("guide").get("rest", candidate)


async def test_a_built_head_gets_a_fitted_rest_set(rig: Rig) -> None:
    rig.built()
    job, found = await fit(rig)
    assert (job.state, job.status, job.error) == ("done", "passed", None)
    assert found.scores["rest_misfit_fit"] == 0.04
    assert found.scores["rest_misfit_warm"] == 0.12
    assert found.data["weights"]["mouthSmileLeft"] == 0.2
    assert [(c.name, c.verdict) for c in found.checks] == [("smile", "passed"), ("rest", "passed")]
    assert {a.name for a in found.outputs} >= {"rest", "rest-fit"}
    assert any("fit_rest.py" in t.name for t in found.tools)
    assert (rig.fake / "runs").exists()


async def test_rest_never_starts_before_the_build(rig: Rig) -> None:
    portrait = rig.store.keep("guide", "portrait", "candidates/c1/portrait.png", b"p")
    rig.store.put("guide", StepResult(step="source", candidate="c1", outputs=[portrait]))
    rig.store.put("guide", StepResult(step="portrait", candidate="c1", checks=[]))
    with pytest.raises(NotReady, match="build"):
        rig.fits().start("guide", "c1")


async def test_fitting_the_same_candidate_twice_at_once_is_busy(rig: Rig) -> None:
    rig.built()
    fits = rig.fits()
    first = fits.start("guide", "c1")
    with pytest.raises(Busy):
        fits.start("guide", "c1")
    await finished(first)


async def test_a_rest_set_under_the_smile_floor_fails(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_SMILE", "0.05")
    rig.built()
    job, found = await fit(rig)
    assert job.state == "failed" and found.standing == "failed"
    assert "0.15" in found.named("failed")


async def test_a_script_failure_fails_the_step(rig: Rig) -> None:
    Path(rig.settings.script).write_text("import sys\nprint('boom', file=sys.stderr)\nsys.exit(1)\n")
    rig.built()
    job, found = await fit(rig)
    assert job.state == "failed" and "boom" in found.named("failed")
    assert found.scores == {} and found.outputs == []


def test_the_editor_fits_a_candidates_rest_set(rig: Rig, tmp_path: Path) -> None:
    rig.built()
    app = create_editor_app(
        Settings(assets_dir=tmp_path / "assets"),
        EditorSettings(),
        rest=rig.fits(),
    )
    with TestClient(app) as client:
        assert client.get("/guided").json()["rest"]["ready"] is True
        job = client.post("/guided/guide/candidates/c1/rest").json()
        assert job["step"] == "rest"
        for _ in range(400):
            job = client.get(f"/guided/guide/jobs/{job['id']}").json()
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.02)
        assert job["state"] == "done" and job["status"] == "passed"
        rest = next(r for r in client.get("/guided/guide").json()["results"] if r["step"] == "rest")
        assert rest["scores"]["rest_misfit_fit"] == 0.04


@pytest.mark.parametrize("smile", ["nan", "inf"])
async def test_weights_that_are_not_finite_fail(rig: Rig, monkeypatch: pytest.MonkeyPatch, smile: str) -> None:
    """json reads NaN as a number, and NaN is under no floor."""
    monkeypatch.setenv("FAKE_SMILE", smile)
    rig.built()
    job, found = await fit(rig)
    assert job.state == "failed" and found.standing == "failed"
    assert "mouthSmileLeft" in found.named("failed") and "weights" not in found.data
