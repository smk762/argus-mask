"""Faces: a ``.glb`` and a ``manifest.json`` in a directory under the assets root.

* :func:`read_glb` and :func:`inspect_glb` check a model against the contract:
  glTF 2.0 binary, morph targets named for the ARKit blendshapes.
* :class:`FaceStore` lists the faces the server may serve. A face whose manifest
  has no provenance, names another id, points outside its directory or no longer
  matches its model's sha256 is refused, with the reason kept for ``inspect``.

Only the standard library reads the model: the JSON chunk says everything the
contract needs, and image sizes come from the PNG and JPEG headers.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from argus_mask.arkit import ARKIT_52
from argus_mask.models import FACE_ID_PATTERN, FaceManifest, Framing, ModelFile, Provenance

MANIFEST_NAME = "manifest.json"
# The stock head's budget: a Raspberry Pi renders it once a frame, a Looking Glass Go 66 times.
MAX_TRIANGLES = 30_000
MAX_TEXTURE = 2048

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FACE_ID = re.compile(FACE_ID_PATTERN)


class FaceError(ValueError):
    """A model or manifest that cannot be used."""


@dataclass
class MeshReport:
    name: str
    triangles: int
    targets: list[str]


@dataclass
class GlbReport:
    meshes: list[MeshReport] = field(default_factory=list)
    textures: list[tuple[int, int] | None] = field(default_factory=list)  # None: size not readable
    present: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def triangles(self) -> int:
        return sum(m.triangles for m in self.meshes)

    @property
    def static_meshes(self) -> list[str]:
        """Meshes with no morph targets: fine for hair, wrong for teeth or eyelashes."""
        return [m.name for m in self.meshes if not m.targets]

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.missing:
            out.append(f"{len(self.missing)} of the 52 ARKit shapes are missing; they will be ignored")
        if self.triangles > MAX_TRIANGLES:
            out.append(f"{self.triangles} triangles is over the {MAX_TRIANGLES} budget for weak screens")
        big = [f"{w}x{h}" for size in self.textures if size for w, h in [size] if max(w, h) > MAX_TEXTURE]
        if big:
            out.append(f"textures over {MAX_TEXTURE}px: {', '.join(big)}")
        if self.static_meshes and self.present:
            out.append(f"meshes with no morph targets (they will not move): {', '.join(self.static_meshes)}")
        return out


def read_glb(path: str | Path) -> tuple[dict, bytes]:
    """The JSON chunk and the binary chunk of a glTF 2.0 binary file."""
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise FaceError(f"{path}: too short to be a .glb")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise FaceError(f"{path}: not a .glb (binary glTF) file")
    if version != 2:
        raise FaceError(f"{path}: glTF version {version}, need 2")
    doc: dict | None = None
    binary = b""
    pos = 12
    end = min(length, len(data))
    while pos + 8 <= end:
        size, kind = struct.unpack_from("<II", data, pos)
        body = data[pos + 8 : pos + 8 + size]
        if kind == _CHUNK_JSON and doc is None:
            try:
                doc = json.loads(body)
            except ValueError as exc:
                raise FaceError(f"{path}: the JSON chunk does not parse: {exc}") from exc
        elif kind == _CHUNK_BIN and not binary:
            binary = body
        pos += 8 + size
    if not isinstance(doc, dict):
        raise FaceError(f"{path}: no JSON chunk")
    return doc, binary


def _image_size(blob: bytes) -> tuple[int, int] | None:
    if blob[:8] == b"\x89PNG\r\n\x1a\n" and len(blob) >= 24:
        w, h = struct.unpack_from(">II", blob, 16)
        return w, h
    if blob[:2] == b"\xff\xd8":
        pos = 2
        while pos + 9 <= len(blob):
            if blob[pos] != 0xFF:
                return None
            marker = blob[pos + 1]
            if marker in (0xC0, 0xC1, 0xC2):  # baseline, extended, progressive
                h, w = struct.unpack_from(">HH", blob, pos + 5)
                return w, h
            pos += 2 + struct.unpack_from(">H", blob, pos + 2)[0]
    return None


def _target_names(mesh: dict, count: int) -> list[str]:
    # Blender, three.js and the glTF sample models all put the names in extras.targetNames.
    for holder in (mesh, *(p for p in mesh.get("primitives", []) if isinstance(p, dict))):
        names = (holder.get("extras") or {}).get("targetNames") if isinstance(holder.get("extras"), dict) else None
        if isinstance(names, list) and len(names) == count and all(isinstance(n, str) for n in names):
            return names
    return [f"target_{i}" for i in range(count)]


def inspect_glb(path: str | Path) -> GlbReport:
    """What a model has, against the contract."""
    doc, binary = read_glb(path)
    accessors = doc.get("accessors") or []
    views = doc.get("bufferViews") or []
    report = GlbReport()

    def count(index: object) -> int:
        if isinstance(index, int) and 0 <= index < len(accessors):
            return int(accessors[index].get("count", 0))
        return 0

    for i, mesh in enumerate(doc.get("meshes") or []):
        primitives = [p for p in mesh.get("primitives", []) if isinstance(p, dict)]
        triangles = 0
        for prim in primitives:
            if prim.get("mode", 4) == 4:
                n = (
                    count(prim["indices"])
                    if "indices" in prim
                    else count((prim.get("attributes") or {}).get("POSITION"))
                )
                triangles += n // 3
        targets = max((len(p.get("targets") or []) for p in primitives), default=0)
        report.meshes.append(
            MeshReport(
                name=str(mesh.get("name") or f"mesh_{i}"), triangles=triangles, targets=_target_names(mesh, targets)
            )
        )

    for image in doc.get("images") or []:
        size = None
        view = image.get("bufferView")
        if isinstance(view, int) and 0 <= view < len(views):
            start = int(views[view].get("byteOffset", 0))
            size = _image_size(binary[start : start + int(views[view].get("byteLength", 0))])
        report.textures.append(size)

    found = {name for mesh in report.meshes for name in mesh.targets}
    report.present = [n for n in ARKIT_52 if n in found]
    report.missing = [n for n in ARKIT_52 if n not in found]
    report.extra = sorted(found - set(ARKIT_52))
    return report


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    model: str | Path, *, face_id: str, name: str, provenance: Provenance, framing: Framing | None = None
) -> FaceManifest:
    """A manifest for *model*, with its sha256 and the ARKit shapes it actually has."""
    model = Path(model)
    return FaceManifest(
        id=face_id,
        name=name,
        model=ModelFile(file=model.name, sha256=sha256_file(model)),
        blendshapes=inspect_glb(model).present,
        framing=framing or Framing(),
        provenance=provenance,
    )


def _why(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        if where == "provenance" and err["type"] == "missing":
            return "the manifest has no provenance"
        parts.append(f"{where}: {err['msg']}" if where else err["msg"])
    return "; ".join(parts)


class FaceStore:
    """The faces under one assets root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._hashes: dict[tuple[str, int, int], str] = {}  # (path, mtime_ns, size) -> sha256

    def _sha256(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._hashes:
            self._hashes[key] = sha256_file(path)
        return self._hashes[key]

    def load(self, face_id: str) -> tuple[FaceManifest, Path]:
        """A face's manifest and model path, or :class:`FaceError` saying why it is refused."""
        if not _FACE_ID.match(face_id):
            raise FaceError(f"{face_id!r} is not a face id")
        directory = self.root / face_id
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FaceError(f"no {MANIFEST_NAME}")
        try:
            manifest = FaceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise FaceError(_why(exc)) from exc
        except (ValueError, RuntimeError) as exc:  # bad JSON, or a manifest version this build does not read
            raise FaceError(str(exc)) from exc
        if manifest.id != face_id:
            raise FaceError(f"the manifest's id is {manifest.id!r}, the directory is {face_id!r}")
        if manifest.provenance.kind == "curated":
            raise FaceError("curated faces are not accepted yet")
        model = directory / manifest.model.file
        # The file name is pattern-checked, so it has no separators; a symlink could still point out.
        if not model.is_file() or model.resolve().parent != directory.resolve():
            raise FaceError(f"the model {manifest.model.file} is not in the face's directory")
        if self._sha256(model) != manifest.model.sha256:
            raise FaceError("the model's sha256 does not match the manifest")
        return manifest, model

    def scan(self) -> tuple[list[FaceManifest], dict[str, str]]:
        """Every face that may be served, and for each directory that may not, the reason."""
        accepted: list[FaceManifest] = []
        refused: dict[str, str] = {}
        if not self.root.is_dir():
            return accepted, refused
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            try:
                accepted.append(self.load(directory.name)[0])
            except FaceError as exc:
                refused[directory.name] = str(exc)
        return accepted, refused
