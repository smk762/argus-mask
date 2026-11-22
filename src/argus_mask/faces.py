"""Faces: a model and a ``manifest.json`` in a directory under the assets root.

* :func:`read_glb` and :func:`inspect_glb` check a model against the contract:
  glTF 2.0 binary, morph targets named for the ARKit blendshapes.
* :func:`inspect_splat` checks a LAM Gaussian-splat head (a ``.zip``) without
  unpacking it to disk: one folder of known members, bounded in count and size,
  no path tricks, and a ``skin.glb`` whose morph targets are ARKit shapes.
* :func:`inspect_bound_splats` checks the splats a mesh model carries (``ModelFile.splats``): an SPZ whose
  header and length agree with the manifest, unpacked in memory under a cap, and a binding whose every triangle is
  one of the mesh's, counted through the parts the manifest names, which must be the mesh's own.
* :class:`FaceStore` lists the faces the server may serve. A face whose manifest
  has no provenance, names another id, points outside its directory or no longer
  matches its model's sha256 is refused, with the reason kept for ``inspect``. So is a
  ``consented`` face without a consent record that covers it (:mod:`argus_mask.consent`).

Only the standard library reads the model (numpy reads a binding's numbers): the JSON chunk says everything the
contract needs, and image sizes come from the PNG and JPEG headers.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import threading
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
from pydantic import ValidationError

from argus_mask.arkit import ARKIT_52
from argus_mask.consent import ConsentStore, ConsentUse, refusal
from argus_mask.models import (
    MAX_BOUND_SPLATS,
    MODEL_SUFFIXES,
    RESERVED_FACE_IDS,
    BoundSplats,
    FaceManifest,
    Framing,
    ModelFile,
    Provenance,
    is_face_id,
    is_safe_name,
)

MANIFEST_NAME = "manifest.json"
# The stock head's budget: a Raspberry Pi renders it once a frame, a Looking Glass Go 66 times.
# 45k: room for a dressed head (a uniform is ~5k triangles, MakeHuman's teeth 7k) with long hair or a hat.
# The weakest panels (a Raspberry Pi, the Go's 66 views a frame) are not targets for now; when they are,
# their faces will want a smaller budget of their own.
# A face over WEAK_SCREEN_TRIANGLES is flagged by inspect, and it is the default budget; a preset may ask for up to
# MAX_TRIANGLES (a big screen, a recording). The recipe hard-codes the same ceiling (build_head.py: MAX_TRIANGLES).
WEAK_SCREEN_TRIANGLES = 45_000
MAX_TRIANGLES = 120_000
MAX_TEXTURE = 2048

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942


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
        if self.triangles > WEAK_SCREEN_TRIANGLES:
            out.append(f"{self.triangles} triangles is over the {WEAK_SCREEN_TRIANGLES} budget for weak screens")
        big = [f"{w}x{h}" for size in self.textures if size for w, h in [size] if max(w, h) > MAX_TEXTURE]
        if big:
            out.append(f"textures over {MAX_TEXTURE}px: {', '.join(big)}")
        if self.static_meshes and self.present:
            out.append(f"meshes with no morph targets (they will not move): {', '.join(self.static_meshes)}")
        return out


def read_glb(path: str | Path) -> tuple[dict, bytes]:
    """The JSON chunk and the binary chunk of a glTF 2.0 binary file."""
    return parse_glb(Path(path).read_bytes(), str(path))


def parse_glb(data: bytes | bytearray, path: str = "model") -> tuple[dict, bytes]:
    """:func:`read_glb` of bytes already in memory (*path* names them in errors)."""
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
            binary = bytes(body)
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


_MALFORMED = "not a well-formed glTF"


def _objects(holder: dict, key: str) -> list[dict]:
    """*holder*'s list of objects under *key* (none if absent); :class:`FaceError` for any other shape."""
    value = holder.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise FaceError(f"{_MALFORMED}: {key} is not a list of objects")
    return value


def _whole(value: object, what: str) -> int:
    """A glTF count, offset or index: a whole number, not negative; :class:`FaceError` for anything else."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FaceError(f"{_MALFORMED}: {what} is not a whole number")
    return value


def _target_names(mesh: dict, count: int) -> list[str]:
    # Blender, three.js and the glTF sample models all put the names in extras.targetNames.
    for holder in (mesh, *(p for p in mesh.get("primitives", []) if isinstance(p, dict))):
        names = (holder.get("extras") or {}).get("targetNames") if isinstance(holder.get("extras"), dict) else None
        if isinstance(names, list) and len(names) == count and all(isinstance(n, str) for n in names):
            return names
    return [f"target_{i}" for i in range(count)]


def inspect_glb(path: str | Path) -> GlbReport:
    """What a model has, against the contract."""
    return glb_report(*read_glb(path))


def glb_report(doc: dict, binary: bytes) -> GlbReport:
    """:func:`inspect_glb` of a parsed glTF binary. The JSON is the file's own claim: an entry of the wrong shape is
    :class:`FaceError`, never another exception."""
    accessors = _objects(doc, "accessors")
    views = _objects(doc, "bufferViews")
    report = GlbReport()

    def count(index: object) -> int:
        if isinstance(index, int) and 0 <= index < len(accessors):
            return _whole(accessors[index].get("count", 0), "an accessor's count")
        return 0

    for i, mesh in enumerate(_objects(doc, "meshes")):
        primitives = mesh.get("primitives", [])
        if not isinstance(primitives, list):
            raise FaceError(f"{_MALFORMED}: a mesh's primitives are not a list")
        primitives = [p for p in primitives if isinstance(p, dict)]
        triangles = 0
        for prim in primitives:
            attributes = prim.get("attributes", {})
            if not isinstance(attributes, dict) or not isinstance(prim.get("targets", []), list):
                raise FaceError(f"{_MALFORMED}: a primitive's attributes or targets are the wrong shape")
            if prim.get("mode", 4) == 4:
                n = count(prim["indices"]) if "indices" in prim else count(attributes.get("POSITION"))
                triangles += n // 3
        targets = max((len(p.get("targets") or []) for p in primitives), default=0)
        report.meshes.append(
            MeshReport(
                name=str(mesh.get("name") or f"mesh_{i}"), triangles=triangles, targets=_target_names(mesh, targets)
            )
        )

    for image in _objects(doc, "images"):
        size = None
        view = image.get("bufferView")
        if isinstance(view, int) and 0 <= view < len(views):
            start = _whole(views[view].get("byteOffset", 0), "a buffer view's byteOffset")
            length = _whole(views[view].get("byteLength", 0), "a buffer view's byteLength")
            size = _image_size(binary[start : start + length])
        report.textures.append(size)

    found = {name for mesh in report.meshes for name in mesh.targets}
    report.present = [n for n in ARKIT_52 if n in found]
    report.missing = [n for n in ARKIT_52 if n not in found]
    report.extra = sorted(found - set(ARKIT_52))
    return report


# ── LAM Gaussian-splat heads ──────────────────────────────────────────────────

# A LAM avatar archive: one folder, these members and nothing else. The browser unpacks it (JSZip, in the
# renderer), so the server refuses anything a careless unpacker could be hurt by before serving it.
SPLAT_REQUIRED = ("skin.glb", "offset.ply", "animation.glb", "vertex_order.json")
SPLAT_OPTIONAL = ("iris_occlusion.json",)
SPLAT_MAX_MEMBERS = 8  # the folder and five files, with room to spare
SPLAT_MAX_MEMBER = 64 << 20  # bytes unpacked, each (C101's largest, skin.glb, is 3.6 MB)
SPLAT_MAX_TOTAL = 160 << 20  # bytes unpacked, all together; also the cap on the archive itself
SPLAT_MAX_RATIO = 100  # unpacked to packed, for members over SPLAT_RATIO_FROM bytes: a zip bomb's tell
SPLAT_RATIO_FROM = 64 << 10
_SPLAT_FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._ -]{0,127}")
_S_IFMT, _S_IFREG, _S_IFDIR = 0o170000, 0o100000, 0o040000


@dataclass
class SplatReport:
    """What a LAM splat archive holds. Its ARKit shapes are its ``skin.glb``'s morph targets."""

    folder: str
    members: dict[str, int]  # member name (in the folder) -> unpacked bytes
    splats: int | None  # offset.ply's vertex count: the number of Gaussians
    skin: GlbReport

    @property
    def present(self) -> list[str]:
        return self.skin.present

    @property
    def missing(self) -> list[str]:
        return self.skin.missing

    @property
    def extra(self) -> list[str]:
        return self.skin.extra

    @property
    def meshes(self) -> list[MeshReport]:
        return self.skin.meshes

    @property
    def textures(self) -> list[tuple[int, int] | None]:
        return self.skin.textures

    @property
    def triangles(self) -> int:
        return self.skin.triangles

    @property
    def warnings(self) -> list[str]:
        # The triangle budget is a mesh face's: a splat head's cost is its Gaussians, and LAM's are fixed.
        if self.missing:
            return [f"{len(self.missing)} of the 52 ARKit shapes are missing; they will be ignored"]
        return []


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, path: str) -> bytearray:
    """A member's bytes, counted as they unpack: the sizes in a zip's headers are the archive's own claim."""
    out = bytearray()
    try:
        with archive.open(info) as handle:
            while block := handle.read(1 << 20):
                out += block
                if len(out) > min(info.file_size, SPLAT_MAX_MEMBER):
                    raise FaceError(f"{path}: {info.filename} unpacks to more than it says")
    except (zipfile.BadZipFile, zlib.error, OSError, EOFError, RuntimeError, ValueError) as exc:  # bad CRC or stream
        raise FaceError(f"{path}: {info.filename} does not unpack: {exc}") from exc
    return out  # not copied to bytes: the archive may unpack to SPLAT_MAX_TOTAL, held once


def _member_name(name: str, path: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise FaceError(f"{path}: the member name {name!r} has a backslash or a NUL")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name):
        raise FaceError(f"{path}: the member name {name!r} is an absolute path")
    parts = name.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise FaceError(f"{path}: the member name {name!r} climbs out of its folder or has an empty part")
    return PurePosixPath(*parts)


def inspect_splat(path: str | Path) -> SplatReport:
    """Check a LAM splat archive against the contract, reading it in memory, never unpacking it to disk.

    Refused (:class:`FaceError`, saying why): not a zip; more than one top-level folder, or files outside
    it; a member the format does not have, or one missing; a duplicate name, an absolute path, ``..``, a
    backslash; a symlink or anything but a plain file or folder; an encrypted member, or one compressed
    other than stored or deflated; a member over ``SPLAT_MAX_MEMBER`` unpacked, all of them over
    ``SPLAT_MAX_TOTAL``, or one that unpacks over a hundred times its packed size; an ``offset.ply`` that is
    not a PLY; JSON that does not parse; a ``skin.glb`` that is not a glTF binary, or whose morph targets
    are not ARKit shapes.
    """
    label = str(path)
    path = Path(path)
    if path.stat().st_size > SPLAT_MAX_TOTAL:
        raise FaceError(f"{label}: the archive is over {SPLAT_MAX_TOTAL >> 20} MiB")
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, EOFError, ValueError, NotImplementedError) as exc:  # a crafted directory
        raise FaceError(f"{label}: not a zip archive ({exc})") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > SPLAT_MAX_MEMBERS:
            raise FaceError(
                f"{label}: {len(infos)} members; a LAM head has at most {SPLAT_MAX_MEMBERS}"
                " (zip the folder alone, without extras such as __MACOSX/ or .DS_Store)"
            )
        names = [i.filename for i in infos]
        if len(set(names)) != len(names):
            raise FaceError(f"{label}: a member name appears twice")
        folders: set[str] = set()
        files: dict[str, zipfile.ZipInfo] = {}
        total = 0
        for info in infos:
            name = _member_name(info.filename, label)
            if info.flag_bits & 0x1:
                raise FaceError(f"{label}: {info.filename} is encrypted")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise FaceError(f"{label}: {info.filename} is compressed in a way this reader does not take")
            mode = info.external_attr >> 16
            kind = mode & _S_IFMT
            is_dir = info.is_dir()
            if kind and kind != (_S_IFDIR if is_dir else _S_IFREG):
                raise FaceError(f"{label}: {info.filename} is a link or a special file, not a plain file")
            folders.add(name.parts[0])
            if is_dir:
                if len(name.parts) != 1:
                    raise FaceError(f"{label}: {info.filename} is a folder inside the head's folder")
                continue
            if len(name.parts) != 2:
                raise FaceError(f"{label}: {info.filename} is not a file directly in the head's folder")
            member = name.parts[1]
            if member not in SPLAT_REQUIRED + SPLAT_OPTIONAL:
                raise FaceError(f"{label}: {info.filename} is not part of a LAM head")
            if info.file_size > SPLAT_MAX_MEMBER:
                raise FaceError(f"{label}: {info.filename} unpacks to over {SPLAT_MAX_MEMBER >> 20} MiB")
            if info.file_size > SPLAT_RATIO_FROM and info.file_size > SPLAT_MAX_RATIO * max(1, info.compress_size):
                raise FaceError(f"{label}: {info.filename} unpacks to over {SPLAT_MAX_RATIO} times its packed size")
            total += info.file_size
            if total > SPLAT_MAX_TOTAL:
                raise FaceError(f"{label}: the members unpack to over {SPLAT_MAX_TOTAL >> 20} MiB together")
            files[member] = info
        if len(folders) != 1:
            raise FaceError(f"{label}: {len(folders)} top-level entries; a LAM head is one folder")
        folder = folders.pop()
        if not is_safe_name(folder, _SPLAT_FOLDER):
            raise FaceError(f"{label}: the folder name {folder!r} is not a plain name")
        if missing := [m for m in SPLAT_REQUIRED if m not in files]:
            raise FaceError(f"{label}: missing {', '.join(missing)}")

        # Every member is unpacked once, in memory, counting what really comes out.
        data = {member: _read_member(archive, info, label) for member, info in files.items()}

    ply = data["offset.ply"]
    if not re.match(rb"^ply\r?\n", ply) or b"end_header" not in ply[: 64 << 10]:
        raise FaceError(f"{label}: offset.ply is not a PLY file")
    # Bounded: int() refuses a run of thousands of digits with a ValueError, never a FaceError.
    count = re.search(rb"\nelement vertex (\d{1,12})\r?\n", ply[: 64 << 10])
    for member in ("vertex_order.json", *SPLAT_OPTIONAL):
        if member in data:
            try:
                json.loads(data[member])
            except (ValueError, RecursionError) as exc:  # RecursionError: nesting deep enough to exhaust the stack
                raise FaceError(f"{label}: {member} is not JSON: {exc}") from exc
    # The glTF readers trust a .glb's JSON to have glTF's shape; this one comes from an upload, so a member that
    # does not is refused here, never an AttributeError out of FaceStore.scan() for every listing.
    try:
        parse_glb(data["animation.glb"], f"{label}: animation.glb")
        skin = glb_report(*parse_glb(data["skin.glb"], f"{label}: skin.glb"))
    except FaceError:
        raise
    except (AttributeError, TypeError, KeyError, IndexError, ValueError, RecursionError, struct.error) as exc:
        raise FaceError(f"{label}: skin.glb or animation.glb is not a well-formed glTF: {exc!r}") from exc
    if skin.extra:
        raise FaceError(f"{label}: skin.glb has morph targets that are not ARKit shapes: {', '.join(skin.extra)}")
    if not skin.present:
        raise FaceError(f"{label}: skin.glb has no ARKit morph targets")
    return SplatReport(
        folder=folder,
        members={m: len(b) for m, b in sorted(data.items())},
        splats=int(count.group(1)) if count else None,
        skin=skin,
    )


def model_format(path: str | Path) -> str:
    """The format a model file is in, by its suffix: ``gltf`` (.glb) or ``lam-splat`` (.zip)."""
    suffix = Path(path).suffix.lower()
    for fmt, ending in MODEL_SUFFIXES.items():
        if suffix == ending:
            return fmt
    raise FaceError(f"{path}: a model is a .glb (glTF binary) or a .zip (LAM splat head)")


def inspect_model(path: str | Path) -> GlbReport | SplatReport:
    """:func:`inspect_glb` or :func:`inspect_splat`, by the file's suffix."""
    return inspect_splat(path) if model_format(path) == "lam-splat" else inspect_glb(path)


# ── Splats bound to a mesh (#126 route A) ─────────────────────────────────────

SPZ_MAGIC = 0x5053474E  # "NGSP"
SPZ_HEADER = 16
SPZ_SH_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}
SPZ_MAX_BYTES = 64 << 20  # the file, packed; stock-01's 149,007 splats at SH 3 are 2.9 MiB
BINDING_BYTES = 12  # uint32 triangle, float32 u, float32 v
_BARY_SLACK = 1e-4  # float32 rounding on a point on an edge


def spz_point_bytes(version: int, sh_degree: int) -> int:
    """Bytes a splat takes in an SPZ body: position (3 x 24 bit), alpha, colour, scale, rotation, SH."""
    return 9 + 1 + 3 + 3 + (4 if version >= 3 else 3) + 3 * SPZ_SH_COEFFS[sh_degree]


@dataclass
class SpzReport:
    version: int
    count: int
    sh_degree: int
    unpacked: int


def inspect_spz(data: bytes, label: str = "splats") -> SpzReport:
    """Check an SPZ (gzip'd: a 16-byte header, then the splats) without decoding a splat.

    Refused (:class:`FaceError`): not gzip; unpacking past what the header's count could need (a bomb stops at the
    cap); a magic, version (2 or 3), SH degree (0 to 3) or count (1 to ``MAX_BOUND_SPLATS``) it should not have;
    a body whose length is not exactly the count's.
    """
    if len(data) > SPZ_MAX_BYTES:
        raise FaceError(f"{label}: over {SPZ_MAX_BYTES >> 20} MiB")
    try:
        unzip = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)  # gzip only
        head = unzip.decompress(data, SPZ_HEADER)
    except zlib.error as exc:
        raise FaceError(f"{label}: not an SPZ file (not gzip: {exc})") from exc
    if len(head) < SPZ_HEADER:
        raise FaceError(f"{label}: too short to be an SPZ file")
    magic, version, count, sh_degree, _bits, _flags, _reserved = struct.unpack_from("<IIIBBBB", head)
    if magic != SPZ_MAGIC:
        raise FaceError(f"{label}: not an SPZ file")
    if version not in (2, 3):
        raise FaceError(f"{label}: SPZ version {version}; this reader takes 2 and 3")
    if sh_degree not in SPZ_SH_COEFFS:
        raise FaceError(f"{label}: SH degree {sh_degree}; at most 3")
    if not 1 <= count <= MAX_BOUND_SPLATS:
        raise FaceError(f"{label}: {count} splats; between 1 and {MAX_BOUND_SPLATS}")
    want = count * spz_point_bytes(version, sh_degree)
    got = 0
    try:  # never more than one byte past the count's body: a bomb stops there
        while got <= want and not unzip.eof:
            chunk = unzip.decompress(unzip.unconsumed_tail, want + 1 - got)
            if not chunk and not unzip.unconsumed_tail:
                break  # the input ran out before the gzip stream ended
            got += len(chunk)
    except zlib.error as exc:
        raise FaceError(f"{label}: the SPZ body does not unpack: {exc}") from exc
    if got != want or not unzip.eof:
        raise FaceError(f"{label}: the body holds {'more' if got > want else 'less'} than {count} splats")
    return SpzReport(version=version, count=count, sh_degree=sh_degree, unpacked=SPZ_HEADER + got)


def glb_parts(doc: dict) -> list[tuple[str, int, int]]:
    """Each mesh node of a glTF, in file order: its name, vertices and triangles (the bake's vertex space).

    An entry of the wrong shape is :class:`FaceError`, as in :func:`glb_report`, never another exception.
    """
    accessors = _objects(doc, "accessors")
    meshes = _objects(doc, "meshes")

    def count(index: object) -> int:
        if isinstance(index, int) and 0 <= index < len(accessors):
            return _whole(accessors[index].get("count", 0), "an accessor's count")
        return 0

    def attributes(prim: dict) -> dict:
        found = prim.get("attributes") or {}
        if not isinstance(found, dict):
            raise FaceError(f"{_MALFORMED}: a primitive's attributes are the wrong shape")
        return found

    out = []
    for node in _objects(doc, "nodes"):
        if not isinstance(node.get("mesh"), int) or not 0 <= node["mesh"] < len(meshes):
            continue
        prims = meshes[node["mesh"]].get("primitives") or []
        if not isinstance(prims, list):
            raise FaceError(f"{_MALFORMED}: a mesh's primitives are not a list")
        prims = [p for p in prims if isinstance(p, dict)]
        vertices = sum(count(attributes(p).get("POSITION")) for p in prims)
        triangles = sum(
            (count(p["indices"]) if "indices" in p else count(attributes(p).get("POSITION"))) // 3
            for p in prims
            if p.get("mode", 4) == 4
        )
        out.append((str(node.get("name", "")), vertices, triangles))
    return out


def check_binding(data: bytes, splats: BoundSplats, label: str = "binding") -> None:
    """Every splat's triangle is one of the mesh's and its (u, v) lie on it; :class:`FaceError` if not."""
    if len(data) != splats.count * BINDING_BYTES:
        raise FaceError(f"{label}: {len(data)} bytes; {splats.count} splats take {splats.count * BINDING_BYTES}")
    rows = np.frombuffer(data, np.dtype([("tri", "<u4"), ("u", "<f4"), ("v", "<f4")]))
    if int(rows["tri"].max()) >= splats.triangles:
        raise FaceError(f"{label}: a splat rides triangle {int(rows['tri'].max())}; the mesh has {splats.triangles}")
    u, v = rows["u"], rows["v"]
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise FaceError(f"{label}: a splat's place on its triangle is not a number")
    if (u < -_BARY_SLACK).any() or (v < -_BARY_SLACK).any() or (u + v > 1 + _BARY_SLACK).any():
        raise FaceError(f"{label}: a splat's place is off its triangle (u, v and u + v lie in 0..1)")


@dataclass
class BoundSplatsReport:
    spz: SpzReport


def inspect_bound_splats(directory: Path, model: ModelFile, doc: dict) -> BoundSplatsReport:
    """Check a mesh model's splats (``model.splats``) against the files beside it and the mesh's glTF *doc*.

    The files must be in *directory* (no link out) and match their sha256; the parts the manifest names must be
    the mesh's own, in order, vertex for vertex and triangle for triangle; then :func:`inspect_spz` (its count the
    manifest's) and :func:`check_binding`.
    """
    splats = model.splats
    if splats is None:
        raise FaceError("the model carries no splats")
    found = glb_parts(doc)
    named = [(p.name, p.vertices, p.triangles) for p in splats.parts]
    if found != named:
        raise FaceError("the splats' parts are not the mesh's (name, vertices and triangles, in file order)")
    data = {}
    for name, sha in ((splats.file, splats.sha256), (splats.binding, splats.binding_sha256)):
        path = directory / name
        if not path.is_file() or path.resolve().parent != directory.resolve():
            raise FaceError(f"the splats' {name} is not in the face's directory")
        if path.stat().st_size > SPZ_MAX_BYTES:
            raise FaceError(f"the splats' {name} is over {SPZ_MAX_BYTES >> 20} MiB")
        data[name] = path.read_bytes()
        if hashlib.sha256(data[name]).hexdigest() != sha:
            raise FaceError(f"the splats' {name} does not match its sha256")
    spz = inspect_spz(data[splats.file], splats.file)
    if spz.count != splats.count:
        raise FaceError(f"{splats.file}: {spz.count} splats; the manifest says {splats.count}")
    check_binding(data[splats.binding], splats, splats.binding)
    return BoundSplatsReport(spz=spz)


# What a face's model is served as, by its format.
MEDIA_TYPES: dict[str, str] = {"gltf": "model/gltf-binary", "lam-splat": "application/zip"}
SPLAT_MEDIA_TYPE = "application/octet-stream"  # an SPZ and a binding; neither has a registered type

# Where the camera sits for a LAM head when the manifest does not say: LAM's heads stand at about 1.65 m
# (FLAME's origin is at the feet). The viewer aims at ``height`` from a tenth of the distance above it.
SPLAT_FRAMING = Framing(distance=0.7, height=1.65, fov=50.0)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    model: str | Path, *, face_id: str, name: str, provenance: Provenance, framing: Framing | None = None
) -> FaceManifest:
    """A manifest for *model* (a .glb or a LAM .zip), with its sha256 and the ARKit shapes it actually has."""
    model = Path(model)
    fmt = model_format(model)
    return FaceManifest(
        id=face_id,
        name=name,
        model=ModelFile(file=model.name, sha256=sha256_file(model), format=fmt),  # type: ignore[arg-type]
        blendshapes=inspect_model(model).present,
        framing=framing or (SPLAT_FRAMING if fmt == "lam-splat" else Framing()),
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


def local_now() -> datetime:
    """Now, aware, in this machine's time zone: the day a consent record's dates are read in."""
    return datetime.now().astimezone()


class FaceStore:
    """The faces under one assets root.

    *clock* gives the time a consented face's record is checked at (a test passes its own).
    """

    def __init__(self, root: str | Path, *, clock: Callable[[], datetime] = local_now) -> None:
        self.root = Path(root)
        self.consents = ConsentStore(self.root)
        self.clock = clock
        self._hashes: dict[str, tuple[tuple[int, ...], str]] = {}  # path -> (its version, sha256)
        self._splats: dict[tuple[str, tuple[int, ...]], SplatReport | str] = {}  # (path, version) -> report, or why
        # The model's path -> (the splats and each file's version, as _sha256 takes it; the report, or why refused)
        self._bound: dict[str, tuple[tuple, BoundSplatsReport | str]] = {}

    def _sha256(self, path: Path) -> str:
        """The file's sha256, hashed again once anything about it changes. The version includes the inode and its
        ctime, which no tool sets: a same-size copy with its mtime kept (``cp -p``, ``rsync -t``) is still new."""
        stat = path.stat()
        version = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        cached = self._hashes.get(str(path))
        if cached is None or cached[0] != version:  # one entry per path: an earlier version's goes
            cached = (version, sha256_file(path))
            self._hashes[str(path)] = cached
        return cached[1]

    def load(self, face_id: str, *, use: ConsentUse = "avatar") -> tuple[FaceManifest, Path]:
        """A face's manifest and model path, or :class:`FaceError` saying why it is refused.

        A consented face must have a consent record that covers *use* now: served on a screen is ``avatar``.
        A file that cannot be read (its permissions, or removed mid-check) refuses the face, never the listing.
        """
        try:
            return self._load(face_id, use)
        except OSError as exc:
            raise FaceError(f"a file cannot be read: {exc.strerror or exc}") from exc

    def _load(self, face_id: str, use: ConsentUse) -> tuple[FaceManifest, Path]:
        if not is_face_id(face_id):
            raise FaceError(f"{face_id!r} is not a face id")
        if face_id in RESERVED_FACE_IDS:
            raise FaceError(f"{face_id!r} is reserved: the directory holds no faces")
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
        if manifest.provenance.kind == "consented":
            why = refusal(self.consents.check_face(manifest.provenance, use=use, at=self.clock()))
            if why is not None:
                raise FaceError(f"consent: {why}")
        model = self._model(directory, manifest.model, "the model")
        self._bound_splats(directory, manifest.model, model, "the model")
        for name, framed in manifest.framings.items():  # each framing is checked as the head is
            path = self._model(directory, framed.model, f"the {name} model")
            self._bound_splats(directory, framed.model, path, f"the {name} model")
        if manifest.model.format == "lam-splat":
            report = self._splat(model)
            if unknown := [n for n in manifest.blendshapes if n not in report.present]:
                raise FaceError(f"the manifest names shapes the head's skin.glb does not have: {', '.join(unknown)}")
        return manifest, model

    def _model(self, directory: Path, model: ModelFile, what: str) -> Path:
        """A model file of the face, refused unless it is in the face's directory and matches its sha256."""
        path = directory / model.file
        # The file name is pattern-checked, so it has no separators; a symlink could still point out.
        if not path.is_file() or path.resolve().parent != directory.resolve():
            raise FaceError(f"{what} {model.file} is not in the face's directory")
        if self._sha256(path) != model.sha256:
            raise FaceError(f"{what}'s sha256 does not match the manifest")
        return path

    def _bound_splats(self, directory: Path, model: ModelFile, path: Path, what: str) -> None:
        """A mesh model's splats, checked once per version of the three files; :class:`FaceError` if refused."""
        if model.splats is None:
            return
        files = [path] + [directory / n for n in (model.splats.file, model.splats.binding)]
        try:
            # Each file's version as _sha256 takes it: a same-size copy with its mtime kept is still checked again.
            versions = []
            for f in files:
                stat = f.stat()
                versions.append((stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
        except OSError:
            raise FaceError(f"{what}'s splats are missing a file") from None
        key = (model.splats.model_dump_json(), *versions)
        cached = self._bound.get(str(path))
        if cached is None or cached[0] != key:  # one entry per model: an earlier version's goes
            try:
                found: BoundSplatsReport | str = inspect_bound_splats(directory, model, read_glb(path)[0])
            except FaceError as exc:
                found = f"{what}'s splats: {exc}"
            cached = self._bound[str(path)] = (key, found)
        found = cached[1]
        if isinstance(found, str):
            raise FaceError(found)

    def _splat(self, path: Path) -> SplatReport:
        """The archive's report, checked once per version of the file; :class:`FaceError` if it is refused."""
        stat = path.stat()  # the version as _sha256 takes it: a same-size copy with its mtime kept is still new
        key = (str(path), (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
        if key not in self._splats:
            for old in [k for k in self._splats if k[0] == key[0]]:  # an earlier version of this file
                del self._splats[old]
            try:
                self._splats[key] = inspect_splat(path)
            except FaceError as exc:
                self._splats[key] = str(exc).replace(f"{path}: ", "", 1)
        found = self._splats[key]
        if isinstance(found, str):
            raise FaceError(found)
        return found

    def scan(self) -> tuple[list[FaceManifest], dict[str, str]]:
        """Every face that may be served, and for each directory that may not, the reason."""
        accepted: list[FaceManifest] = []
        refused: dict[str, str] = {}
        if not self.root.is_dir():
            return accepted, refused
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if directory.name in RESERVED_FACE_IDS:
                continue
            try:
                accepted.append(self.load(directory.name)[0])
            except FaceError as exc:
                refused[directory.name] = str(exc)
        return accepted, refused


# ── Aliases ───────────────────────────────────────────────────────────────────

ALIASES_NAME = "aliases.json"


class AliasError(ValueError):
    """An alias that may not be set, with the reason."""


class Aliases:
    """Stable names for faces: ``holly`` stays the name a persona and a kiosk use while the face behind it changes.

    Kept in ``<assets>/aliases.json`` as ``{"aliases": {"holly": "ginger-3"}}``, written by the editor
    and read by the server (whose assets mount may be read-only). The file is read again whenever it
    changes, so repointing an alias needs no restart. An entry is used only if it is sound: its name is
    a face id and not the name of a face directory (an alias never shadows a face), and it points at
    a face the store serves, never at another alias. Anything else in the file is ignored, and
    `problems` says why.
    """

    def __init__(self, store: FaceStore) -> None:
        self.store = store
        self.path = store.root / ALIASES_NAME
        # (the file's version, its entries), replaced in one assignment: request threads read it unlocked.
        self._cache: tuple[tuple[int, int, int] | None, dict[str, str]] = (None, {})
        self._lock = threading.Lock()  # the editor's routes run in threads: one read-modify-write at a time

    def raw(self) -> dict[str, str]:
        """The file's entries as written, valid or not."""
        try:
            stat = self.path.stat()
        except OSError:
            self._cache = (None, {})
            return {}
        # The inode too: every write replaces the file, so a same-size rewrite within one mtime tick is still seen.
        key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        seen, entries = self._cache
        if key != seen:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                found = data.get("aliases") if isinstance(data, dict) else None
                entries = {str(k): str(v) for k, v in found.items()} if isinstance(found, dict) else {}
            except (OSError, ValueError):
                entries = {}
            self._cache = (key, entries)  # read after the stat: a newer write only makes the next call read again
        return dict(entries)

    def _problem(self, name: str, target: str, raw: dict[str, str]) -> str | None:
        if not is_face_id(name):
            return f"{name!r} is not a face id"
        if (self.store.root / name).is_dir():
            return f"{name!r} is a face: an alias may not hide it"
        # An entry named after a face directory is ignored, so it does not make its name an alias.
        if target in raw and not (self.store.root / target).is_dir():
            return f"{target!r} is itself an alias: point at a face"
        try:
            self.store.load(target)
        except FaceError as exc:
            return f"{target!r} is not a face this server serves ({exc})"
        return None

    def problems(self) -> dict[str, str]:
        """The entries that are ignored, and why."""
        raw = self.raw()
        return {n: why for n, t in raw.items() if (why := self._problem(n, t, raw)) is not None}

    def all(self) -> dict[str, str]:
        """The aliases in use: name -> face id, sorted by name."""
        raw = self.raw()
        return {n: t for n, t in sorted(raw.items()) if self._problem(n, t, raw) is None}

    def resolve(self, name: str) -> str:
        """The face a name stands for: a face's own id is itself, an alias its face; anything else as given."""
        if not is_face_id(name) or (self.store.root / name).is_dir():
            return name
        raw = self.raw()
        target = raw.get(name)
        return target if target is not None and self._problem(name, target, raw) is None else name

    def _write(self, entries: dict[str, str]) -> None:
        tmp = self.path.with_name(ALIASES_NAME + ".part")
        tmp.write_text(json.dumps({"aliases": dict(sorted(entries.items()))}, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def set(self, name: str, target: str) -> None:
        """Point *name* at the face *target* (made or repointed), or `AliasError` saying why not."""
        with self._lock:
            raw = self.raw()
            rest = {k: v for k, v in raw.items() if k != name}
            if (why := self._problem(name, target, rest)) is not None:
                raise AliasError(why)
            if any(v == name for v in rest.values()):
                raise AliasError(f"{name!r} is what another alias points at")
            self._write({**rest, name: target})

    def remove(self, name: str) -> bool:
        with self._lock:
            raw = self.raw()
            if name not in raw:
                return False
            self._write({k: v for k, v in raw.items() if k != name})
            return True

    def pointing_at(self, face_id: str) -> list[str]:
        return sorted(n for n, t in self.raw().items() if t == face_id)

    def repoint(self, old: str, new: str | None) -> None:
        """A face was renamed (*new*) or deleted (None): its aliases follow it, or go."""
        with self._lock:
            raw = self.raw()
            if not any(t == old for t in raw.values()):
                return
            # An entry named *new* goes: that name is a face now, and an alias never shadows one.
            self._write(
                {n: (new if t == old else t) for n, t in raw.items() if (t != old or new is not None) and n != new}
            )
