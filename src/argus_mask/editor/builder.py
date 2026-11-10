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
import hashlib
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
from argus_mask.faces import MANIFEST_NAME, FaceError, FaceStore, build_manifest, inspect_glb, sha256_file
from argus_mask.models import FACE_ID_PATTERN, FaceManifest, Framing, Provenance

log = structlog.get_logger(__name__)

SCRIPT = Path(__file__).resolve().with_name("build_head.py")
PRESET_NAME = "preset.json"
MODEL_NAME = "head.glb"
RECIPE_NAME = "head.recipe.json"
PREVIEW_DIR = ".preview"
# Every build is also kept here, by what it was built from, so asking for the same face again (undo,
# redo, or a slider put back) copies it instead of running Blender. Under the preview area: dot-named,
# so neither the server nor the preview listing sees it.
CACHE_DIR = ".cache"
DEFAULT_CACHE_SIZE = 40  # builds kept, least recently used first out: about 6 MB each
_KEY = re.compile(r"^[0-9a-f]{64}$")
TRASH_DIR = ".trash"  # deleted faces go here, not away: the server skips it, and a move restores one
PLACES = ("export", "preview")
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


class Taken(BuildError):
    """A rename to an id another face already has."""


class Refused(BuildError):
    """The recipe itself refused the preset (``build_head.py`` exit status 2)."""


@dataclass
class BuildResult:
    manifest: FaceManifest
    recipe: dict
    seconds: float
    key: str = ""  # the cache key: GET /cache/<key> shows this build later without building it
    cached: bool = False


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
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self.blender = blender
        self.blender_user = blender_user
        self.library = library
        self.assets_dir = Path(assets_dir)
        self.preview_dir = self.assets_dir / PREVIEW_DIR
        self.timeout_s = timeout_s
        self.cache_dir = self.preview_dir / CACHE_DIR
        self.cache_size = max(0, cache_size)
        self._lock = asyncio.Lock()
        # What else decides the model: the recipe, the slider catalogue and the Blender it runs in. A
        # change to any of them is a different key, so a cache never serves a model the recipe no longer makes.
        recipe = hashlib.sha256()
        for part in (SCRIPT, SCRIPT.with_name("modifiers.json")):
            recipe.update(part.read_bytes() if part.is_file() else b"")
        recipe.update(str(blender).encode())
        self._recipe = recipe.hexdigest()

    # ---- the build cache ---------------------------------------------------------------------------

    def key(self, preset: Preset) -> str:
        """What a build is made from: the preset without its id and name (they change no vertex), and the recipe."""
        content = preset.model_dump(mode="json", exclude_none=True, exclude={"id", "name"})
        text = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self._recipe}\n{text}".encode()).hexdigest()

    def cached(self, key: str) -> Path | None:
        """The cached build for *key*, if it is whole: its model still matches its manifest."""
        if not _KEY.match(key):
            return None
        directory = self.cache_dir / key
        try:
            manifest = FaceManifest.model_validate_json((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
            model = directory / manifest.model.file
            return directory if model.is_file() and sha256_file(model) == manifest.model.sha256 else None
        except (OSError, ValueError):
            return None

    def cached_face(self, key: str) -> tuple[FaceManifest, Path] | None:
        directory = self.cached(key)
        if directory is None:
            return None
        manifest = FaceManifest.model_validate_json((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
        return manifest, directory / manifest.model.file

    def _keep(self, work: Path, key: str) -> None:
        """Copy a finished build into the cache, then drop the least recently used beyond the cap."""
        if self.cache_size == 0:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / key
        if target.exists():
            os.utime(target)
            return
        tmp = Path(tempfile.mkdtemp(prefix=".keep-", dir=self.cache_dir))
        try:
            shutil.copytree(work, tmp, dirs_exist_ok=True)
            tmp.rename(target)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        entries = sorted((p for p in self.cache_dir.iterdir() if _KEY.match(p.name)), key=lambda p: p.stat().st_mtime)
        for old in entries[: max(0, len(entries) - self.cache_size)]:
            shutil.rmtree(old, ignore_errors=True)

    def _from_cache(self, source: Path, work: Path, preset: Preset) -> tuple[FaceManifest, dict]:
        """Fill *work* from a cached build, with this preset's id and name."""
        shutil.copytree(source, work, dirs_exist_ok=True)
        os.utime(source)  # recently used: kept longest
        manifest = FaceManifest.model_validate_json((work / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest = manifest.model_copy(update={"id": preset.id, "name": preset.name})
        (work / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (work / PRESET_NAME).write_text(preset.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        try:
            recipe = json.loads((work / RECIPE_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recipe = {}
        return manifest, recipe

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
                # Off the event loop: a face is megabytes, and the editor's other routes wait on the loop.
                await asyncio.to_thread(shutil.copytree, self.preview_dir / face_id, work, dirs_exist_ok=True)
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
        key = self.key(preset)
        work = Path(tempfile.mkdtemp(prefix=".build-", dir=self.preview_dir))
        started = time.monotonic()
        source = await asyncio.to_thread(self.cached, key)
        if source is not None:
            try:
                manifest, recipe = await asyncio.to_thread(self._from_cache, source, work, preset)
                self._install(work, target, self.preview_dir)
            finally:
                if work.exists():
                    shutil.rmtree(work, ignore_errors=True)
            seconds = round(time.monotonic() - started, 2)
            log.info("face_built", face=preset.id, seconds=seconds, cached=True)
            return BuildResult(manifest=manifest, recipe=recipe, seconds=seconds, key=key, cached=True)
        try:
            preset_text = preset.model_dump_json(indent=2, exclude_none=True) + "\n"
            (work / PRESET_NAME).write_text(preset_text, encoding="utf-8")
            await self._run_blender(work / PRESET_NAME, work / MODEL_NAME)
            model = work / MODEL_NAME
            if not model.is_file():
                raise BuildError("Blender finished but wrote no model")
            report = await asyncio.to_thread(inspect_glb, model)
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
            manifest = await asyncio.to_thread(
                build_manifest, model, face_id=preset.id, name=preset.name, provenance=provenance, framing=FRAMING
            )
            (work / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            try:
                await asyncio.to_thread(self._keep, work, key)
            except OSError as exc:  # a full disk loses the cache entry, not the build
                log.warning("build_not_cached", face=preset.id, reason=str(exc))
            self._install(work, target, self.preview_dir)
        finally:
            if work.exists():
                shutil.rmtree(work, ignore_errors=True)
        seconds = round(time.monotonic() - started, 2)
        log.info("face_built", face=preset.id, seconds=seconds, triangles=recipe.get("triangles_total"))
        return BuildResult(manifest=manifest, recipe=recipe, seconds=seconds, key=key)

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

    # ---- saved faces: list, rename, delete -------------------------------------------------------

    def root(self, place: str) -> Path:
        return self.preview_dir if place == "preview" else self.assets_dir

    def editable(self, face_id: str, place: str) -> bool:
        return self.previewed(face_id) if place == "preview" else self.owned(face_id)

    def preset_in(self, face_id: str, place: str) -> Preset | None:
        if not _FACE_ID.match(face_id):
            return None
        return self._read_preset(self.root(place) / face_id)

    def saved(self) -> list[dict]:
        """Every face in the assets root (the editor's and any other) and every preview, newest first."""
        out: dict[str, dict] = {}
        for place in PLACES:
            root = self.root(place)
            if not root.is_dir():
                continue
            store = FaceStore(root)
            accepted, refused = store.scan()
            for manifest in accepted:
                model = root / manifest.id / manifest.model.file
                entry = out.setdefault(manifest.id, {"id": manifest.id, "export": None, "preview": None})
                entry[place] = {
                    "name": manifest.name,
                    "kind": manifest.provenance.kind,
                    "license": manifest.provenance.license,
                    "notes": manifest.provenance.notes,
                    "editable": self.editable(manifest.id, place),
                    "updated": model.stat().st_mtime,
                    "bytes": model.stat().st_size,
                    "sha256": manifest.model.sha256,
                }
            if place == "export":
                for face_id, why in refused.items():
                    if _FACE_ID.match(face_id):
                        out.setdefault(face_id, {"id": face_id, "export": None, "preview": None})["refused"] = why

        def newest(e: dict) -> float:
            return max((e[p]["updated"] for p in PLACES if e.get(p)), default=0.0)

        return sorted(out.values(), key=newest, reverse=True)

    async def rename(self, face_id: str, *, new_id: str | None = None, name: str | None = None) -> str:
        """Give a face the editor made a new id, a new name, or both: its export and its preview together."""
        if self._lock.locked():
            raise Busy("another build or export is running")
        async with self._lock:
            new_id = new_id or face_id
            if bad := next((i for i in (face_id, new_id) if not _FACE_ID.match(i)), None):
                raise NoPreview(f"{bad!r} is not a face id")
            places = [p for p in PLACES if (self.root(p) / face_id).exists()]
            if not places:
                raise NoPreview(f"there is no face {face_id!r}")
            for place in places:
                if not self.editable(face_id, place):
                    raise NotOwned(f"{face_id!r} is a face the editor did not make")
            if new_id != face_id and any((self.root(p) / new_id).exists() for p in PLACES):
                raise Taken(f"there is already a face {new_id!r}")
            # Every copy is rewritten before any is moved: a failure part way leaves both as they were,
            # not the export under the new id and the preview under the old one.
            staged: list[tuple[Path, Path]] = []
            try:
                for place in places:
                    root = self.root(place)
                    staged.append((root, await asyncio.to_thread(self._rewrite, root, face_id, new_id, name)))
                for root, work in staged:
                    if new_id == face_id:
                        self._install(work, root / face_id, root)
                    else:
                        work.rename(root / new_id)
                        shutil.rmtree(root / face_id, ignore_errors=True)
            finally:
                for _, work in staged:
                    if work.exists():
                        shutil.rmtree(work, ignore_errors=True)
            log.info("face_renamed", face=face_id, to=new_id, name=name)
            return new_id

    def _rewrite(self, root: Path, face_id: str, new_id: str, name: str | None) -> Path:
        """A copy of ``<root>/<face_id>`` under *root*, with the new id and name written in; not yet in place."""
        work = Path(tempfile.mkdtemp(prefix=".rename-", dir=root))
        try:
            shutil.copytree(root / face_id, work, dirs_exist_ok=True)
            manifest = FaceManifest.model_validate_json((work / MANIFEST_NAME).read_text(encoding="utf-8"))
            manifest = manifest.model_copy(update={"id": new_id, "name": name or manifest.name})
            (work / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
            preset = self._read_preset(work)
            if preset is not None:
                preset = preset.model_copy(update={"id": new_id, "name": name or preset.name})
                (work / PRESET_NAME).write_text(
                    preset.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
                )
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return work

    async def delete(self, face_id: str, places: tuple[str, ...] = PLACES) -> Path:
        """Move a face the editor made (its export, its preview, or both) to the trash; returns where it went."""
        if self._lock.locked():
            raise Busy("another build or export is running")
        async with self._lock:
            if not _FACE_ID.match(face_id):
                raise NoPreview(f"{face_id!r} is not a face id")
            found = [p for p in places if (self.root(p) / face_id).exists()]
            if not found:
                raise NoPreview(f"there is no face {face_id!r}")
            for place in found:
                if not self.editable(face_id, place):
                    raise NotOwned(f"{face_id!r} is a face the editor did not make")
            bin_ = self.assets_dir / TRASH_DIR / f"{face_id}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
            bin_.mkdir(parents=True)
            for place in found:
                (self.root(place) / face_id).rename(bin_ / place)
            log.info("face_deleted", face=face_id, places=found, trash=str(bin_))
            return bin_

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
