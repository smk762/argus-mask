"""The MPFB assets installed on this machine, and the licence each is shared under.

MPFB keeps its assets under ``<data>/<part dir>/<asset>/<asset>.<mhmat|mhclo>``,
and each asset pack lists its assets in ``<data>/packs/<pack>.json`` with a
licence and an author. An asset no pack lists has no stated licence, so the
editor does not offer it: a face built from it could not carry provenance.

Nothing here is bundled. The user installs the packs (see ``scripts/README.md``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from argus_mask.editor.preset import CLOTHES_DIR, PART_DIRS, Preset

FACEUNITS_PACK = "faceunits01"
CLOTHES = "clothes"  # the library's part name for everything worn
# Clothes a bust never shows.
_UNSEEN_TAGS = {"shoes"}
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# Pack licences as SPDX ids. Anything else is kept as the pack states it.
_SPDX = {"CC0": "CC0-1.0", "CC-0": "CC0-1.0", "CC0-1.0": "CC0-1.0"}


class LibraryError(ValueError):
    """A preset names an asset that is not installed, or not licensed."""


@dataclass(frozen=True)
class Asset:
    part: str  # the preset's part name: skin, eyes, hair ...
    name: str  # the asset's directory name
    ref: str  # what a preset holds: <part dir>/<name>/<name>.<ext>
    license: str
    author: str
    pack: str
    thumb: Path | None
    tags: tuple[str, ...] = ()

    @property
    def slot(self) -> str:
        """For clothes, where it goes: ``hat`` or ``outfit``. Empty for body parts."""
        if self.part != CLOTHES:
            return ""
        return "hat" if {"hats", "hat"} & set(self.tags) else "outfit"

    def public(self) -> dict:
        out = {"part": self.part, "name": self.name, "ref": self.ref, "license": self.license, "author": self.author}
        if self.slot:
            out["slot"] = self.slot
        return out


def _licence(raw: str) -> str:
    return _SPDX.get(raw.strip(), raw.strip())


def _tags(path: Path) -> tuple[str, ...]:
    """The ``tag`` lines of an ``.mhclo``, lower-cased."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    return tuple(ln.split(None, 1)[1].strip().lower() for ln in lines if ln.startswith("tag ") and len(ln.split()) > 1)


def _hair_as_clothes(entries: dict, directory: str, name: str) -> tuple[str, dict]:
    """A hairstyle's pack entry when the pack types it ``clothes``, as MakeHuman 1.x hair packs do (hair02, hair03).

    The files are in ``hair/``, where MPFB loads them as hair; only the index's type is old. The entry is the
    same asset by name, with its licence and author, so it is offered as hair.
    """
    if directory != PART_DIRS["hair"]:
        return "", {}
    return entries.get((CLOTHES_DIR, name), ("", {}))


class AssetLibrary:
    """One MPFB data directory: ``$BLENDER_USER_RESOURCES/extensions/.user/user_default/mpfb/data``."""

    def __init__(self, data_dir: str | Path | None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self._scanned: tuple[tuple, dict[str, list[Asset]]] | None = None

    def _signature(self) -> tuple:
        """What a scan depends on, cheaply: the pack files and the part directories, by mtime.

        Installing or removing a pack or an asset changes it, so a cached scan is only reused while
        the library is as it was.
        """
        if self.data_dir is None:
            return ()
        paths = [self.data_dir / "packs", *(self.data_dir / d for d in (*PART_DIRS.values(), CLOTHES_DIR))]
        packs = self.data_dir / "packs"
        if packs.is_dir():
            paths += sorted(packs.glob("*.json"))
        out = []
        for path in paths:
            try:
                st = path.stat()
            except OSError:
                out.append((str(path), None))
                continue
            out.append((str(path), st.st_mtime_ns, st.st_size))
        return tuple(out)

    def _packs(self) -> tuple[dict[tuple[str, str], tuple[str, dict]], set[str]]:
        """(MPFB type, asset name) to (pack, entry), and the names of the packs installed."""
        entries: dict[tuple[str, str], tuple[str, dict]] = {}
        names: set[str] = set()
        if self.data_dir is None:
            return entries, names
        packs = self.data_dir / "packs"
        if not packs.is_dir():
            return entries, names
        for path in sorted(packs.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict):
                continue
            names.add(path.stem)
            for asset, entry in doc.items():
                if isinstance(entry, dict) and isinstance(entry.get("type"), str):
                    entries.setdefault((entry["type"], asset), (path.stem, entry))
        return entries, names

    def has_faceunits(self) -> bool:
        """Whether the ARKit face shapes are installed: without them no head can be built."""
        return FACEUNITS_PACK in self._packs()[1]

    def scan(self) -> dict[str, list[Asset]]:
        """Every licensed asset, by part; clothes under ``clothes``, without the ones a bust never shows.

        Cached until the library changes: the editor asks once per thumbnail, and a full scan reads every
        pack and every garment's ``.mhclo``.
        """
        signature = self._signature()
        if self._scanned is not None and self._scanned[0] == signature:
            return self._scanned[1]
        found = self._scan()
        self._scanned = (signature, found)
        return found

    def _scan(self) -> dict[str, list[Asset]]:
        entries, _ = self._packs()
        out: dict[str, list[Asset]] = {part: [] for part in (*PART_DIRS, CLOTHES)}
        if self.data_dir is None:
            return out
        for part, directory in (*PART_DIRS.items(), (CLOTHES, CLOTHES_DIR)):
            ext = ".mhmat" if part == "skin" else ".mhclo"
            found: list[Asset] = []
            root = self.data_dir / directory
            if root.is_dir():
                for sub in sorted(p for p in root.iterdir() if p.is_dir() and _NAME.match(p.name)):
                    if not (sub / f"{sub.name}{ext}").is_file():
                        continue  # eyes/materials and the like: not an asset of this part
                    pack, entry = entries.get((directory, sub.name)) or _hair_as_clothes(entries, directory, sub.name)
                    licence = _licence(str(entry.get("license") or ""))
                    if not licence:
                        continue
                    thumb = sub / f"{sub.name}.thumb"
                    tags = _tags(sub / f"{sub.name}{ext}") if part == CLOTHES else ()
                    if _UNSEEN_TAGS & set(tags):
                        continue
                    found.append(
                        Asset(
                            part=part,
                            name=sub.name,
                            ref=f"{directory}/{sub.name}/{sub.name}{ext}",
                            license=licence,
                            author=str(entry.get("author") or ""),
                            pack=pack,
                            thumb=thumb if thumb.is_file() else None,
                            tags=tags,
                        )
                    )
            out[part] = found
        return out

    def find(self, part: str, name: str) -> Asset | None:
        return next((a for a in self.scan().get(part, []) if a.name == name), None)

    def resolve(self, preset: Preset) -> dict[str, Asset]:
        """The installed asset behind each part and each item of clothing, or :class:`LibraryError` naming the first that is not.

        A preset with no outfit among its clothes is refused too.

        Keys are the part names, and ``clothes <name>`` for each item of clothing.
        """
        index = {a.ref: a for assets in self.scan().values() for a in assets}
        wanted = [*preset.assets.refs().items(), *((CLOTHES, ref) for ref in preset.assets.clothes)]
        resolved: dict[str, Asset] = {}
        for part, ref in wanted:
            asset = index.get(ref)
            if asset is None or asset.part != part:
                raise LibraryError(f"{part}: {ref} is not an installed asset with a stated licence")
            resolved[f"{CLOTHES} {asset.name}" if part == CLOTHES else part] = asset
        if not any(a.slot == "outfit" for a in resolved.values()):
            raise LibraryError(f"{CLOTHES}: every head is dressed; choose an outfit, not only a hat")
        return resolved


def licence_expression(assets: dict[str, Asset]) -> str:
    """One licence line for a face built from *assets* (the face units are CC0 and not listed)."""
    licences = sorted({a.license for a in assets.values()})
    return " AND ".join(licences)


def credits(assets: dict[str, Asset]) -> str:
    """Which asset, from which pack, by whom, under what licence: for the manifest's notes."""
    out = []
    for key, a in assets.items():
        label = key if key.startswith(CLOTHES) else f"{key} {a.name}"
        out.append(f"{label} ({a.pack or 'no pack'}, {a.author or 'author not stated'}, {a.license})")
    return "; ".join(out)
