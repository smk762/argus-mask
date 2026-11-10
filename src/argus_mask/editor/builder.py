"""Build a head from a preset and install it as a face.

One build at a time. Blender runs ``build_head.py`` into a hidden directory
under the assets root; only a build that succeeds, and whose model passes the
face contract, is renamed into ``<assets>/<id>``. A failed build leaves nothing.

The editor only replaces faces it made: a face directory holding
``preset.json``. A stock face, or any face put there by hand, is refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog

from argus_mask import __version__
from argus_mask.editor.library import AssetLibrary, credits, licence_expression
from argus_mask.editor.preset import Preset
from argus_mask.faces import MANIFEST_NAME, build_manifest, inspect_glb
from argus_mask.models import FACE_ID_PATTERN, FaceManifest, Framing, Provenance

log = structlog.get_logger(__name__)

SCRIPT = Path(__file__).resolve().with_name("build_head.py")
PRESET_NAME = "preset.json"
MODEL_NAME = "head.glb"
RECIPE_NAME = "head.recipe.json"
# Where the stock heads put the camera: a bust, not a face filling the screen.
FRAMING = Framing(distance=0.9, height=-0.08)
_ERROR_CHARS = 2000
_FACE_ID = re.compile(FACE_ID_PATTERN)


class BuildError(RuntimeError):
    """The build could not run, failed, or made something that is not a face."""


class Busy(BuildError):
    """Another build is running."""


class NotOwned(BuildError):
    """The face id is taken by a face the editor did not make."""


class Refused(BuildError):
    """The recipe itself refused the preset (``build_head.py`` exit status 2)."""


@dataclass
class BuildResult:
    manifest: FaceManifest
    recipe: dict
    seconds: float


def _why(stderr: str, stdout: str) -> str:
    """The recipe's own error line if it printed one, else the end of Blender's output."""
    lines = [ln for ln in stderr.splitlines() if ln.strip()]
    mine = [ln for ln in lines if ln.startswith("build_head:")]
    if mine:
        return mine[-1].removeprefix("build_head:").strip()[:_ERROR_CHARS]
    tail = "\n".join(lines[-8:] or [ln for ln in stdout.splitlines() if ln.strip()][-8:])
    return tail[-_ERROR_CHARS:] or "Blender exited without saying why"


class Builder:
    def __init__(
        self,
        *,
        blender: str,
        blender_user: Path | None,
        library: AssetLibrary,
        assets_dir: Path,
        timeout_s: float = 300.0,
    ) -> None:
        self.blender = blender
        self.blender_user = blender_user
        self.library = library
        self.assets_dir = Path(assets_dir)
        self.timeout_s = timeout_s
        self._lock = asyncio.Lock()

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def blender_path(self) -> str | None:
        return shutil.which(self.blender)

    def check(self) -> str | None:
        """Why no build can run, or None."""
        if self.blender_path() is None:
            return f"Blender not found: {self.blender!r} (set ARGUS_MASK_BLENDER)"
        if self.library.data_dir is None:
            return "MPFB's data directory is not set (set ARGUS_MASK_BLENDER_USER or ARGUS_MASK_MPFB_DATA)"
        if not self.library.data_dir.is_dir():
            return f"MPFB's data directory {self.library.data_dir} does not exist (set ARGUS_MASK_BLENDER_USER)"
        if not self.library.has_faceunits():
            return "the faceunits01 asset pack is not installed: it holds the ARKit face shapes"
        return None

    def owned(self, face_id: str) -> bool:
        return bool(_FACE_ID.match(face_id)) and (self.assets_dir / face_id / PRESET_NAME).is_file()

    def preset(self, face_id: str) -> Preset | None:
        if not _FACE_ID.match(face_id):
            return None
        path = self.assets_dir / face_id / PRESET_NAME
        if not path.is_file():
            return None
        try:
            return Preset.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def presets(self) -> list[Preset]:
        """The presets behind every face the editor made."""
        if not self.assets_dir.is_dir():
            return []
        found = (self.preset(p.name) for p in sorted(self.assets_dir.iterdir()) if p.is_dir())
        return [p for p in found if p is not None]

    async def build(self, preset: Preset) -> BuildResult:
        if self._lock.locked():
            raise Busy("another build is running")
        async with self._lock:
            return await self._build(preset)

    async def _build(self, preset: Preset) -> BuildResult:
        if reason := self.check():
            raise BuildError(reason)
        assets = self.library.resolve(preset)
        target = self.assets_dir / preset.id
        if target.exists() and not self.owned(preset.id):
            raise NotOwned(f"{preset.id!r} is a face the editor did not make; choose another id")

        self.assets_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".build-", dir=self.assets_dir))
        started = time.monotonic()
        try:
            preset_text = preset.model_dump_json(indent=2, exclude_none=True) + "\n"
            (work / PRESET_NAME).write_text(preset_text, encoding="utf-8")
            await self._run_blender(work / PRESET_NAME, work / MODEL_NAME)
            model = work / MODEL_NAME
            if not model.is_file():
                raise BuildError("Blender finished but wrote no model")
            report = inspect_glb(model)
            if report.missing:
                raise BuildError(f"the model is missing {len(report.missing)} of the 52 ARKit shapes")
            try:
                recipe = json.loads((work / RECIPE_NAME).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                recipe = {}
            provenance = Provenance(
                kind="synthetic",
                source=(
                    f"Built by the argus-mask {__version__} face editor from {PRESET_NAME} beside the model: "
                    f"Blender {recipe.get('blender', 'unknown')}, MPFB {recipe.get('mpfb', 'unknown')}"
                ),
                license=licence_expression(assets),
                notes=(
                    f"{credits(assets)}; ARKit face units from the faceunits01 pack (Mika Suominen, CC0-1.0). "
                    f"Phenotype: gender {preset.phenotype.gender:g}, age about {preset.age_years()} years. "
                    "A generated character, not a real person."
                ),
            )
            manifest = build_manifest(
                model, face_id=preset.id, name=preset.name, provenance=provenance, framing=FRAMING
            )
            (work / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            self._install(work, target)
        finally:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
        seconds = round(time.monotonic() - started, 2)
        log.info("face_built", face=preset.id, seconds=seconds, triangles=recipe.get("triangles_total"))
        return BuildResult(manifest=manifest, recipe=recipe, seconds=seconds)

    async def _run_blender(self, preset_path: Path, out: Path) -> None:
        env = dict(os.environ)
        if self.blender_user is not None:
            env["BLENDER_USER_RESOURCES"] = str(self.blender_user)
        blender = self.blender_path()
        if blender is None:
            raise BuildError(f"Blender not found: {self.blender!r}")
        # --python-exit-code: without it Blender exits 0 when the script raises.
        argv = [blender, "-b", "--python-exit-code", "3", "-P", str(SCRIPT), "--"]
        argv += ["--preset", str(preset_path), "--out", str(out)]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise BuildError(f"Blender took more than {self.timeout_s:g} s and was stopped") from None
        except asyncio.CancelledError:
            proc.kill()
            await proc.wait()
            raise
        if proc.returncode == 0:
            return
        why = _why(stderr.decode(errors="replace"), stdout.decode(errors="replace"))
        if proc.returncode == 2:
            raise Refused(why)
        raise BuildError(f"Blender exited with status {proc.returncode}: {why}")

    def _install(self, work: Path, target: Path) -> None:
        """Put the finished build where the server reads faces, replacing the editor's earlier build of it."""
        if not target.exists():
            work.rename(target)
            return
        old = self.assets_dir / f".old-{uuid.uuid4().hex}"
        target.rename(old)
        try:
            work.rename(target)
        except OSError:
            old.rename(target)
            raise
        shutil.rmtree(old, ignore_errors=True)
