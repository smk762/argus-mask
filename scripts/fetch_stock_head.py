#!/usr/bin/env python3
"""Download a prebuilt stock head into the assets directory.

The repo carries no model files. Each stock head has a lock file,
``scripts/stock_heads/<id>.lock.json``, naming where the built model (a ``.glb``, or a LAM splat head's ``.zip``) is
published, its size, and the manifest that goes beside it (which holds the
model's sha256). A head built at more than one framing (``waist``) lists each
other model under ``framings``, by url and size; the manifest names its file and
sha256. This script downloads every model, checks each against the lock, and
writes them and ``manifest.json`` into ``<assets>/<id>/``. A lock may also carry
the head's ``rest`` set (a splat head's is fitted to its portrait, #151): it is
added to ``<assets>/expressions.json`` and chosen for the head, unless a set of
that name is already there and differs, or the head already has a choice.

    python scripts/fetch_stock_head.py                 # stock-01 into ./assets
    python scripts/fetch_stock_head.py --id stock-01 --assets-dir /srv/faces

Nothing is written unless the download matches the lock. To build the head
yourself instead, see ``scripts/README.md``. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parent / "stock_heads"
TIMEOUT_S = 60
MAX_BYTES = 64 * 1024 * 1024  # no stock head is near this; a lock that says otherwise is wrong
# A mesh head is a .glb; a LAM Gaussian-splat head (#126) is a .zip, its manifest's model.format "lam-splat".
MODEL_SUFFIX = {"gltf": ".glb", "lam-splat": ".zip"}
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHAPE = re.compile(r"^[a-zA-Z]{3,32}$")  # an ARKit blendshape name; the server checks it against the 52
EXPRESSIONS = "expressions.json"


class FetchError(RuntimeError):
    """The head could not be fetched, or what arrived is not what the lock describes."""


def read_lock(face_id: str, lock_dir: Path = LOCK_DIR) -> dict:
    if not _ID.fullmatch(face_id):
        raise FetchError(f"{face_id!r} is not a face id")
    path = lock_dir / f"{face_id}.lock.json"
    if not path.is_file():
        known = sorted(p.name.removesuffix(".lock.json") for p in lock_dir.glob("*.lock.json"))
        raise FetchError(f"no lock file for {face_id!r}; known heads: {', '.join(known) or 'none'}")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
        manifest = lock["manifest"]
        if manifest["id"] != face_id:
            raise FetchError(f"{path}: the manifest is for {manifest['id']!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest["model"]["sha256"]):
            raise FetchError(f"{path}: the manifest's sha256 is not a sha256")
        suffix = MODEL_SUFFIX.get(manifest["model"].get("format", "gltf"))
        if suffix is None:
            raise FetchError(f"{path}: the model's format {manifest['model'].get('format')!r} is not one mask serves")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*" + re.escape(suffix), manifest["model"]["file"]):
            raise FetchError(f"{path}: the manifest's model file name is not a plain {suffix} name")
        if not isinstance(lock["url"], str) or not isinstance(lock["size"], int) or not 0 < lock["size"] <= MAX_BYTES:
            raise FetchError(f"{path}: needs a url and a size between 1 and {MAX_BYTES} bytes")
        for name, framed in (manifest.get("framings") or {}).items():
            where = (lock.get("framings") or {}).get(name)
            if not where:
                raise FetchError(f"{path}: the manifest has a {name} framing but the lock says nowhere to get it")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.glb", framed["model"]["file"]):
                raise FetchError(f"{path}: the {name} model's file name is not a plain .glb name")
            if not re.fullmatch(r"[0-9a-f]{64}", framed["model"]["sha256"]):
                raise FetchError(f"{path}: the {name} model's sha256 is not a sha256")
            if (
                not isinstance(where["url"], str)
                or not isinstance(where["size"], int)
                or not 0 < where["size"] <= MAX_BYTES
            ):
                raise FetchError(f"{path}: the {name} framing needs a url and a size between 1 and {MAX_BYTES} bytes")
        if "rest" in lock:
            check_rest(lock["rest"], path)
    except (ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"{path}: not a usable lock file ({exc})") from exc
    return lock


def check_rest(rest: object, path: Path) -> None:
    """A lock's rest set: a name like a face id, ARKit weights from 0 to 1, life from 0 to 2."""
    if not isinstance(rest, dict) or not isinstance(rest.get("name"), str) or not _ID.fullmatch(rest["name"]):
        raise FetchError(f"{path}: the rest set needs a name like a face id")
    weights, life = rest.get("weights"), rest.get("life", 1.0)
    if not isinstance(weights, dict) or not all(
        isinstance(k, str) and _SHAPE.fullmatch(k) and isinstance(v, (int, float)) and 0 <= v <= 1
        for k, v in weights.items()
    ):
        raise FetchError(f"{path}: the rest set's weights are ARKit names with weights from 0 to 1")
    if not isinstance(life, (int, float)) or not 0 <= life <= 2:
        raise FetchError(f"{path}: the rest set's life is from 0 to 2")


def install_rest(assets_dir: Path, face_id: str, rest: dict) -> str:
    """Add *rest* to the assets' expressions.json and choose it for *face_id*. Never replaces a set or a choice.

    Returns what happened, for the message. The file is replaced whole, as the server writes it.
    """
    path = assets_dir / EXPRESSIONS
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError) as exc:
        raise FetchError(f"{path}: cannot read it ({exc}); the head is in place, its rest set is not") from exc
    if not isinstance(raw, dict):
        raise FetchError(f"{path}: not a JSON object; the head is in place, its rest set is not")
    sets = raw.setdefault("sets", {})
    faces = raw.setdefault("faces", {})
    if not isinstance(sets, dict) or not isinstance(faces, dict):
        raise FetchError(f"{path}: sets and faces must be objects; the head is in place, its rest set is not")
    entry = {"weights": rest["weights"], "life": rest.get("life", 1.0)}
    name = rest["name"]
    if name in sets and sets[name] != entry:
        return f"rest set {name} not installed: {path} has a different set of that name"
    if face_id in faces and faces[face_id] != name:
        sets.setdefault(name, entry)
        note = f"rest set {name} added; {face_id} keeps its own choice ({faces[face_id]})"
    else:
        sets[name], faces[face_id] = entry, name
        note = f"rest set {name} chosen for {face_id}"
    handle, partial = tempfile.mkstemp(dir=assets_dir, prefix=".expressions-", suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(json.dumps(raw, indent=2) + "\n")
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(partial, 0o666 & ~mask)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)
    return note


def download(url: str, size: int, sha256: str, target: Path) -> None:
    """Fetch *url* to *target*, which appears only if the bytes match *size* and *sha256*."""
    if not url.startswith("https://"):
        raise FetchError(f"{url}: only https downloads are allowed")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    got = 0
    handle, partial = tempfile.mkstemp(dir=target.parent, prefix=".fetch-", suffix=".part")
    try:
        with os.fdopen(handle, "wb") as out:
            try:
                with urllib.request.urlopen(url, timeout=TIMEOUT_S) as reply:  # noqa: S310 - https only, checked above
                    while block := reply.read(1 << 16):
                        got += len(block)
                        if got > size:
                            raise FetchError(f"{url}: more than the {size} bytes the lock expects")
                        digest.update(block)
                        out.write(block)
            except urllib.error.HTTPError as exc:
                raise FetchError(f"{url}: HTTP {exc.code}") from exc
            except (urllib.error.URLError, OSError) as exc:
                raise FetchError(f"{url}: {exc}") from exc
        if got != size:
            raise FetchError(f"{url}: got {got} bytes, the lock expects {size}")
        if digest.hexdigest() != sha256:
            raise FetchError(f"{url}: sha256 is {digest.hexdigest()}, the lock expects {sha256}")
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(partial, 0o666 & ~mask)  # mkstemp's 0600 would otherwise stay on the head
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(face_id: str, assets_dir: Path, *, force: bool = False, lock_dir: Path = LOCK_DIR) -> tuple[Path, bool]:
    """Put a stock head under *assets_dir*. Returns its directory and whether anything was downloaded."""
    lock = read_lock(face_id, lock_dir)
    manifest = lock["manifest"]
    directory = assets_dir / face_id
    manifest_path = directory / "manifest.json"
    # The head, then each other framing: (file, sha256, url, size).
    models = [(manifest["model"]["file"], manifest["model"]["sha256"], lock["url"], lock["size"])]
    for name, framed in (manifest.get("framings") or {}).items():
        where = lock["framings"][name]
        models.append((framed["model"]["file"], framed["model"]["sha256"], where["url"], where["size"]))
    wanted_files = []
    for file, wanted, _url, _size in models:  # check every file before downloading any
        model = directory / file
        have = sha256_file(model) if model.is_file() else None
        if have is not None and have != wanted and not force:
            raise FetchError(f"{model} exists and is a different file; pass --force to replace it")
        wanted_files.append(have != wanted)
    downloaded = any(wanted_files)
    for (file, wanted, url, size), needed in zip(models, wanted_files, strict=True):
        if needed:
            download(url, size, wanted, directory / file)
    text = json.dumps(manifest, indent=2) + "\n"
    if manifest_path.is_file() and manifest_path.read_text(encoding="utf-8") != text and not force and not downloaded:
        raise FetchError(f"{manifest_path} exists and differs from the lock; pass --force to replace it")
    manifest_path.write_text(text, encoding="utf-8")
    if "rest" in lock:
        print(install_rest(assets_dir, face_id, lock["rest"]))
    return directory, downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a prebuilt stock head into the assets directory.")
    parser.add_argument("--id", default="stock-01", help="which head (default: stock-01)")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"), help="the assets root (default: ./assets)")
    parser.add_argument(
        "--force", action="store_true", help="replace a model or manifest that is already there and differs"
    )
    args = parser.parse_args(argv)
    try:
        directory, downloaded = fetch(args.id, args.assets_dir, force=args.force)
    except FetchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    provenance = read_lock(args.id)["manifest"]["provenance"]
    print(f"{'downloaded' if downloaded else 'already present'}: {directory}")
    print(f"licence: {provenance['license']} ({provenance['kind']})")
    print(f"check it with: argus-mask inspect {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
