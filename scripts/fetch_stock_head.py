#!/usr/bin/env python3
"""Download a prebuilt stock head into the assets directory.

The repo carries no model files. Each stock head has a lock file,
``scripts/stock_heads/<id>.lock.json``, naming where the built ``.glb`` is
published, its size, and the manifest that goes beside it (which holds the
model's sha256). This script downloads the model, checks it against the lock,
and writes ``<assets>/<id>/head.glb`` and ``manifest.json``.

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
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class FetchError(RuntimeError):
    """The head could not be fetched, or what arrived is not what the lock describes."""


def read_lock(face_id: str, lock_dir: Path = LOCK_DIR) -> dict:
    if not _ID.match(face_id):
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
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.glb", manifest["model"]["file"]):
            raise FetchError(f"{path}: the manifest's model file name is not a plain .glb name")
        if not isinstance(lock["url"], str) or not isinstance(lock["size"], int) or not 0 < lock["size"] <= MAX_BYTES:
            raise FetchError(f"{path}: needs a url and a size between 1 and {MAX_BYTES} bytes")
    except (ValueError, KeyError, TypeError) as exc:
        raise FetchError(f"{path}: not a usable lock file ({exc})") from exc
    return lock


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
    model = directory / manifest["model"]["file"]
    manifest_path = directory / "manifest.json"
    wanted = manifest["model"]["sha256"]
    have = sha256_file(model) if model.is_file() else None
    if have is not None and have != wanted and not force:
        raise FetchError(f"{model} exists and is a different file; pass --force to replace it")
    downloaded = have != wanted
    if downloaded:
        download(lock["url"], lock["size"], wanted, model)
    text = json.dumps(manifest, indent=2) + "\n"
    if manifest_path.is_file() and manifest_path.read_text(encoding="utf-8") != text and not force and not downloaded:
        raise FetchError(f"{manifest_path} exists and differs from the lock; pass --force to replace it")
    manifest_path.write_text(text, encoding="utf-8")
    return directory, downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a prebuilt stock head into the assets directory.")
    parser.add_argument("--id", default="stock-01", help="which head (default: stock-01)")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"), help="the assets root (default: ./assets)")
    parser.add_argument("--force", action="store_true", help="replace a model or manifest that is already there and differs")
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
