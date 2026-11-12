"""MakeHuman community asset packs: the catalogue the editor browses, and installing one into MPFB's data.

The catalogue (``asset_catalogue.json``, written by ``scripts/make_asset_catalogue.py`` and
reviewed like code) lists each pack's zip, its sha256 and its assets. A pack is installed by its id
alone: the request never names a URL or a path. The zip comes only from a URL in the catalogue, on a
host in :data:`HOSTS`, and is refused unless its hash matches. Every member is fenced under the data
directory (no absolute paths, no ``..``, no links) and the whole is capped in size.

A pack never overwrites a file already installed: a face built from an installed asset must build
the same tomorrow. A file the pack shares with what is installed is skipped if it is the same, and
the install is refused before anything is written if it differs. The pack's own index
(``packs/<id>.json``, where :class:`~argus_mask.editor.library.AssetLibrary` reads each asset's
licence) must be in the zip, or its assets would have no licence and the editor would not offer them.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import structlog

log = structlog.get_logger(__name__)

CATALOGUE = Path(__file__).with_name("asset_catalogue.json")
HOSTS = ("files.makehumancommunity.org", "files2.makehumancommunity.org")
PACK_ID = r"^[a-z0-9][a-z0-9_]{0,63}$"
_PACK_ID = re.compile(PACK_ID)
# Licences a pack may carry to be offered: what the editor can credit. As the pack pages state them.
LICENCES = {"CC0", "CC-BY"}
MAX_UNPACKED = 8 * 2**30  # a pack unpacked; the largest are a few hundred MB
MAX_MEMBERS = 20_000
DOWNLOAD_SLACK = 1.05  # a download may run this much over the catalogued size before it is cut
TIMEOUT_S = 60
JOBS_KEEP = 20
AGENT = "argus-mask editor (+https://github.com/smk762/argus-mask)"
_STAGING = ".argus-mask-staging"  # under the data directory, so the final moves stay on one file system


class PackError(ValueError):
    """A pack that cannot be installed: unknown, not pinned, a bad download or an unsafe zip."""


class Busy(PackError):
    """A pack is already being installed."""


@dataclass(frozen=True)
class Pack:
    id: str
    description: str
    license: str
    urls: tuple[str, ...]
    bytes: int | None
    sha256: str | None
    page: str
    assets: tuple[dict, ...]

    @property
    def licences(self) -> set[str]:
        return {p.strip() for p in self.license.split(" AND ") if p.strip()}

    def why_not(self) -> str | None:
        """Why this pack cannot be installed from here, or None."""
        if not self.sha256:
            return "not pinned: the catalogue has no hash for it (run scripts/make_asset_catalogue.py)"
        if not self.licences or not self.licences <= LICENCES:
            return f"licence {self.license or 'not stated'}: only {', '.join(sorted(LICENCES))} packs are offered"
        return None

    def public(self, installed: bool) -> dict:
        why = self.why_not()
        return {
            "id": self.id,
            "description": self.description,
            "license": self.license,
            "bytes": self.bytes,
            "page": self.page,
            "installed": installed,
            "installable": why is None and not installed,
            "reason": None if installed else why,
            "assets": list(self.assets),
        }


def _asset(raw: object) -> dict | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        return None
    out = {k: str(raw.get(k) or "")[:200] for k in ("name", "type", "author", "license")}
    thumb = raw.get("thumb")
    # A thumbnail is shown by the browser, straight from the community's site: an https URL there, or none.
    if isinstance(thumb, str) and urllib.parse.urlsplit(thumb).scheme == "https":
        host = urllib.parse.urlsplit(thumb).hostname or ""
        if host.endswith(".makehumancommunity.org"):
            out["thumb"] = thumb
    return out


def load_catalogue(path: Path = CATALOGUE) -> dict[str, Pack]:
    """The catalogue's packs by id. Entries that are not well formed, or point anywhere but :data:`HOSTS`, are dropped."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    packs: dict[str, Pack] = {}
    for raw in doc.get("packs", []) if isinstance(doc, dict) else []:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str) or not _PACK_ID.match(raw["id"]):
            continue
        urls = tuple(
            u
            for u in raw.get("urls") or []
            if isinstance(u, str)
            and urllib.parse.urlsplit(u).scheme == "https"
            and urllib.parse.urlsplit(u).hostname in HOSTS
        )
        if not urls:
            continue
        sha = raw.get("sha256")
        size = raw.get("bytes")
        packs[raw["id"]] = Pack(
            id=raw["id"],
            description=str(raw.get("description") or "")[:1000],
            license=str(raw.get("license") or ""),
            urls=urls,
            bytes=size if isinstance(size, int) and size > 0 else None,
            sha256=sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha) else None,
            page=str(raw.get("page") or ""),
            assets=tuple(a for a in map(_asset, raw.get("assets") or []) if a is not None),
        )
    return packs


def installed(data_dir: Path | None) -> set[str]:
    """The packs whose index is in the data directory: what MPFB, and the editor's library, read."""
    packs = data_dir / "packs" if data_dir is not None else None
    if packs is None or not packs.is_dir():
        return set()
    return {p.stem for p in packs.glob("*.json")}


# ---- the zip ---------------------------------------------------------------------------------------


def _member_path(info: zipfile.ZipInfo) -> PurePosixPath | None:
    """A member's path inside the data directory, None for a directory, or :class:`PackError` if it would escape."""
    name = info.filename
    if "\\" in name or "\x00" in name:
        raise PackError(f"the zip has a member with an unsafe name: {name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise PackError(f"the zip has a link: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise PackError(f"the zip has a member outside the pack: {name!r}")
    if info.is_dir():
        return None
    if not path.parts:
        raise PackError(f"the zip has a member with no name: {name!r}")
    return path


def _same(path: Path, zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != info.file_size:
        return False
    a, b = hashlib.sha256(), hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            a.update(chunk)
    with zf.open(info) as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            b.update(chunk)
    return a.digest() == b.digest()


def plan(zip_path: Path, data_dir: Path, pack_id: str) -> tuple[list[tuple[zipfile.ZipInfo, PurePosixPath]], int]:
    """What unpacking *zip_path* would write, and how many files it would skip as already there and the same.

    Refused (:class:`PackError`) if any member would land outside *data_dir*, is a link, the whole is
    too big, the pack's own index is missing, or a file it would write is already installed and differs.
    """
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackError(f"not a zip file: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise PackError(f"the zip has {len(infos)} members; more than {MAX_MEMBERS} is not a pack")
        if sum(i.file_size for i in infos) > MAX_UNPACKED:
            raise PackError(f"the zip unpacks to more than {MAX_UNPACKED // 2**30} GB")
        root = data_dir.resolve()
        todo: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        skipped = 0
        conflicts: list[str] = []
        names: set[str] = set()
        for info in infos:
            rel = _member_path(info)
            if rel is None:
                continue
            target = (root / rel).resolve()
            if not target.is_relative_to(root) or target == root:
                raise PackError(f"the zip has a member outside the pack: {info.filename!r}")
            if str(rel) in names:
                raise PackError(f"the zip has {info.filename!r} twice")
            names.add(str(rel))
            if target.exists() or target.is_symlink():
                if target.is_file() and not target.is_symlink() and _same(target, zf, info):
                    skipped += 1
                    continue
                conflicts.append(str(rel))
                continue
            todo.append((info, rel))
        if f"packs/{pack_id}.json" not in names:
            raise PackError(
                f"the zip has no packs/{pack_id}.json: its assets would have no stated licence, so none would be offered"
            )
        if conflicts:
            shown = ", ".join(conflicts[:5]) + (f" and {len(conflicts) - 5} more" if len(conflicts) > 5 else "")
            raise PackError(f"the pack would replace installed files that differ ({shown}); nothing was installed")
        return todo, skipped


def unpack(zip_path: Path, data_dir: Path, pack_id: str) -> dict:
    """Unpack a checked zip into *data_dir*, all or nothing: staged beside it, then moved in, the pack's index last.

    The index last: until it is in place the library offers none of the pack's assets, so a half-moved
    pack is never offered.
    """
    todo, skipped = plan(zip_path, data_dir, pack_id)
    root = data_dir.resolve()
    staging = root / _STAGING / uuid.uuid4().hex[:12]
    staging.mkdir(parents=True)
    moved: list[Path] = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info, rel in todo:
                out = staging / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, out.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
        index = PurePosixPath("packs", f"{pack_id}.json")
        for _, rel in sorted(todo, key=lambda t: t[1] == index):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():  # appeared since the plan: never replace
                raise PackError(f"{rel} appeared while installing; nothing more was installed")
            os.replace(staging / rel, target)
            moved.append(target)
    except BaseException:
        for path in moved:  # all or nothing
            with contextlib.suppress(OSError):
                path.unlink()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        with contextlib.suppress(OSError):
            (root / _STAGING).rmdir()
    return {"files": len(todo), "skipped": skipped}


# ---- the download ----------------------------------------------------------------------------------


Opener = Callable[[str], object]  # url -> a response with .read(n); the tests pass their own


def _open(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})  # noqa: S310 - https, HOSTS only
    return urllib.request.urlopen(req, timeout=TIMEOUT_S)  # noqa: S310


def download(pack: Pack, dest: Path, progress: Callable[[int], None], opener: Opener = _open) -> str:
    """Fetch *pack*'s zip to *dest*, trying each catalogued URL in turn. Returns the URL it came from.

    Cut at the catalogued size plus a little, and refused unless its sha256 is the catalogue's.
    """
    if pack.sha256 is None:
        raise PackError(f"{pack.id} is not pinned")
    limit = int((pack.bytes or MAX_UNPACKED) * DOWNLOAD_SLACK)
    errors = []
    for url in pack.urls:
        host = urllib.parse.urlsplit(url).hostname
        if urllib.parse.urlsplit(url).scheme != "https" or host not in HOSTS:
            continue
        digest, size = hashlib.sha256(), 0
        try:
            reply = opener(url)
            try:
                with dest.open("wb") as out:
                    while chunk := reply.read(1 << 20):  # type: ignore[attr-defined]
                        size += len(chunk)
                        if size > limit:
                            raise PackError(f"{host} sent more than the catalogued {pack.bytes} bytes")
                        digest.update(chunk)
                        out.write(chunk)
                        progress(size)
            finally:
                with contextlib.suppress(Exception):
                    reply.close()  # type: ignore[attr-defined]
        except (urllib.error.URLError, OSError, PackError) as exc:
            errors.append(f"{host}: {getattr(exc, 'reason', exc)}")
            continue
        if digest.hexdigest() != pack.sha256:
            errors.append(f"{host}: the download's sha256 is not the catalogue's")
            continue
        return url
    dest.unlink(missing_ok=True)
    raise PackError("; ".join(errors) or "no URL to download from")


# ---- jobs ------------------------------------------------------------------------------------------


@dataclass
class Install:
    id: str
    pack: str
    state: str = "downloading"  # downloading, unpacking, done, failed
    done_bytes: int = 0
    total_bytes: int | None = None
    error: str | None = None
    result: dict | None = None
    started: float = field(default_factory=time.monotonic)
    ended: float | None = None
    task: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self.state in ("downloading", "unpacking")

    def public(self) -> dict:
        return {
            "id": self.id,
            "pack": self.pack,
            "state": self.state,
            "done_bytes": self.done_bytes,
            "total_bytes": self.total_bytes,
            "error": self.error,
            "result": self.result,
            "seconds": round((self.ended or time.monotonic()) - self.started, 1),
        }


class Installs:
    """Pack installs, one at a time, into one MPFB data directory."""

    def __init__(self, data_dir: Path | None, catalogue: dict[str, Pack], opener: Opener = _open) -> None:
        self.data_dir = data_dir
        self.catalogue = catalogue
        self.opener = opener
        self.jobs: dict[str, Install] = {}

    def listing(self) -> list[dict]:
        have = installed(self.data_dir)
        return [p.public(p.id in have) for p in self.catalogue.values()]

    def start(self, pack_id: str) -> Install:
        pack = self.catalogue.get(pack_id)
        if pack is None:
            raise PackError(f"no pack {pack_id!r} in the catalogue")
        if self.data_dir is None or not self.data_dir.is_dir():
            raise PackError("MPFB's data directory is not set or does not exist (set ARGUS_MASK_BLENDER_USER)")
        if pack_id in installed(self.data_dir):
            raise PackError(f"{pack_id} is already installed")
        if (why := pack.why_not()) is not None:
            raise PackError(f"{pack_id}: {why}")
        if any(j.running for j in self.jobs.values()):
            raise Busy("a pack is already being installed: one at a time")
        job = Install(id=uuid.uuid4().hex[:12], pack=pack_id, total_bytes=pack.bytes)
        self.jobs[job.id] = job
        for old in [j for j in self.jobs.values() if not j.running][:-JOBS_KEEP]:
            self.jobs.pop(old.id, None)
        data_dir = self.data_dir

        def work() -> dict:
            downloads = data_dir / _STAGING
            downloads.mkdir(exist_ok=True)
            zip_path = downloads / f"{pack_id}-{job.id}.zip"
            try:
                url = download(pack, zip_path, lambda n: setattr(job, "done_bytes", n), self.opener)
                job.state = "unpacking"
                out = unpack(zip_path, data_dir, pack_id)
            finally:
                zip_path.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    downloads.rmdir()
            return out | {"from": urllib.parse.urlsplit(url).hostname}

        async def run() -> None:
            try:
                job.result = await asyncio.to_thread(work)
                job.state = "done"
                log.info("pack_installed", pack=pack_id, **job.result)
            except PackError as exc:
                job.state, job.error = "failed", str(exc)
                log.warning("pack_install_failed", pack=pack_id, reason=str(exc))
            except Exception as exc:  # noqa: BLE001 - the page must hear of it
                job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
                log.warning("pack_install_failed", pack=pack_id, reason=str(exc))
            finally:
                job.ended = time.monotonic()

        job.task = asyncio.create_task(run())
        return job

    async def stop_all(self) -> None:
        tasks = [j.task for j in self.jobs.values() if j.task is not None and not j.task.done()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
