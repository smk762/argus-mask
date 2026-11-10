"""Build a head from a preset as a preview, and export a preview as a face.

A build is a preview: ``<assets>/.preview/<id>``. The server skips
dot-directories, so previews never reach ``/faces``; the editor shows them.
Export copies a preview to ``<assets>/<id>``, where the server serves it.

One build or export at a time. Each is made in a hidden directory and renamed
into place only once it is whole, so a failed build or export leaves nothing
and keeps what was there.

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
from argus_mask.faces import MANIFEST_NAME, FaceError, FaceStore, build_manifest, inspect_glb
from argus_mask.models import FACE_ID_PATTERN, FaceManifest, Framing, Provenance

log = structlog.get_logger(__name__)

SCRIPT = Path(__file__).resolve().with_name("build_head.py")
PRESET_NAME = "preset.json"
MODEL_NAME = "head.glb"
RECIPE_NAME = "head.recipe.json"
PREVIEW_DIR = ".preview"
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


class NoPreview(BuildError):
    """Export was asked for a face that has no finished preview."""


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
        self.preview_dir = self.assets_dir / PREVIEW_DIR
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
        """Whether ``<assets>/<id>`` is a face the editor exported (so it may replace it)."""
        return bool(_FACE_ID.match(face_id)) and (self.assets_dir / face_id / PRESET_NAME).is_file()

    def previewed(self, face_id: str) -> bool:
        return bool(_FACE_ID.match(face_id)) and (self.preview_dir / face_id / PRESET_NAME).is_file()

    @staticmethod
    def _read_preset(directory: Path) -> Preset | None:
        path = directory / PRESET_NAME
        if not path.is_file():
            return None
        try:
            return Preset.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def presets(self) -> list[dict]:
        """Every face the editor made, previewed or exported: its preset (the preview's, if both), and which."""
        found: dict[str, dict] = {}
        for root, key in ((self.assets_dir, "exported"), (self.preview_dir, "previewed")):
            if not root.is_dir():
                continue
            for directory in sorted(p for p in root.iterdir() if p.is_dir() and _FACE_ID.match(p.name)):
                preset = self._read_preset(directory)
                if preset is None:
                    continue
                entry = found.setdefault(directory.name, {"exported": False, "previewed": False})
                entry["preset"] = preset  # the preview, read second, wins: it is what the editor shows
                entry[key] = True
        return [{"preset": e["preset"], "exported": e["exported"], "previewed": e["previewed"]} for e in found.values()]

    async def build(self, preset: Preset) -> BuildResult:
        if self._lock.locked():
            raise Busy("another build or export is running")
        async with self._lock:
            return await self._build(preset)

    async def export(self, face_id: str) -> FaceManifest:
        """Copy the preview of *face_id* to ``<assets>/<id>``, where the server serves it."""
        if self._lock.locked():
            raise Busy("another build or export is running")
        async with self._lock:
            if not _FACE_ID.match(face_id):
                raise NoPreview(f"{face_id!r} is not a face id")
            try:
                manifest, _ = FaceStore(self.preview_dir).load(face_id)
            except FaceError as exc:
                raise NoPreview(f"there is no preview of {face_id!r} to export ({exc})") from exc
            target = self.assets_dir / face_id
            if target.exists() and not self.owned(face_id):
                raise NotOwned(f"{face_id!r} is a face the editor did not make; build it under another id")
            work = Path(tempfile.mkdtemp(prefix=".export-", dir=self.assets_dir))
            try:
                shutil.copytree(self.preview_dir / face_id, work, dirs_exist_ok=True)
                self._install(work, target, self.assets_dir)
            finally:
                if work.exists():
                    shutil.rmtree(work, ignore_errors=True)
            log.info("face_exported", face=face_id)
            return manifest

    async def _build(self, preset: Preset) -> BuildResult:
        if reason := self.check():
            raise BuildError(reason)
        assets = self.library.resolve(preset)
        # Say so now, not at export: a preview under an id that can never be exported is wasted.
        if (self.assets_dir / preset.id).exists() and not self.owned(preset.id):
            raise NotOwned(f"{preset.id!r} is a face the editor did not make; choose another id")
        target = self.preview_dir / preset.id

        self.preview_dir.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".build-", dir=self.preview_dir))
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
            shaping = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(preset.targets.items()) if v)
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
                    + (f"Shaped with MakeHuman targets (CC0): {shaping}. " if shaping else "")
                    + "A generated character, not a real person."
                ),
            )
            manifest = build_manifest(
                model, face_id=preset.id, name=preset.name, provenance=provenance, framing=FRAMING
            )
            (work / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            self._install(work, target, self.preview_dir)
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

    @staticmethod
    def _install(work: Path, target: Path, root: Path) -> None:
        """Rename a finished directory into place under *root*, replacing the editor's earlier one."""
        if not target.exists():
            work.rename(target)
            return
        old = root / f".old-{uuid.uuid4().hex}"
        target.rename(old)
        try:
            work.rename(target)
        except OSError:
            old.rename(target)
            raise
        shutil.rmtree(old, ignore_errors=True)
