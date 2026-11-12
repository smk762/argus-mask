#!/usr/bin/env python3
"""Write ``src/argus_mask/editor/asset_catalogue.json`` from the MakeHuman community's asset pack pages.

The community's old JSON index (``www.makehumancommunity.org/sites/default/files/assets.json``) is
gone with the old site, and there is no API. The asset packs are listed on a static page, and each
pack has a page of its own with a table of its assets (type, thumbnail, name, author, source,
licence) and its zip on two mirrors. This reads those pages and writes the catalogue the editor's
Asset packs page browses and installs from.

Each pack is pinned by its zip's sha256: the editor installs a pack only if the download matches.
Hashing means downloading every zip (a few GB). A hash already in the catalogue is kept while the
pack's URL and size are unchanged, so a refresh only downloads what is new or changed.

    python scripts/make_asset_catalogue.py            # refresh; hash only what is new or changed
    python scripts/make_asset_catalogue.py --no-hash  # pages only: new packs get no hash (not installable)
    python scripts/make_asset_catalogue.py --only hair02 --only hair03

A refresh is a reviewed commit: read the diff. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "argus_mask" / "editor" / "asset_catalogue.json"
SITE = "https://static.makehumancommunity.org"
INDEX = f"{SITE}/assets/downloadassets.html"
# Where pack zips may come from. The editor checks the same list before it downloads anything.
HOSTS = ("files.makehumancommunity.org", "files2.makehumancommunity.org")
AGENT = "argus-mask asset catalogue (+https://github.com/smk762/argus-mask)"
_PACK_LINK = re.compile(r'href="/assets/assetpacks/([a-z0-9_]+)\.html"')
_ZIP = re.compile(r'href="(https://[^"]+\.zip)"')
_SIZE = re.compile(r"\((\d+(?:\.\d+)?)\s*([kmg])b\)", re.I)
TIMEOUT_S = 60
# The download links at the end of a pack's first paragraph: ": mirror1, mirror2 (68 mb)".
_MIRRORS = re.compile(r"[:,]?\s*mirror1\s*,\s*mirror2\s*(\([^)]*\))?\.?\s*$", re.I)
# Some pages name a type in the singular; MPFB's directory (and the editor) use the plural.
_TYPES = {"skin": "skins"}
PAUSE_S = 2.0  # the site refuses a burst: about 70 pages take a few minutes
# Packs whose page has no asset table and whose zip's name states no licence, with the licence the
# asset packs page gives them ("Functional asset packs": CC0) and that their packs/<id>.json confirms.
STATED = {"faceunits01": "CC0", "visemes01": "CC0", "visemes02": "CC0"}


def fetch(url: str, tries: int = 4) -> bytes:
    """A page, retried with a backoff: the site resets the odd connection."""
    for attempt in range(tries):
        time.sleep(PAUSE_S)  # one page at a time, politely
        req = urllib.request.Request(url, headers={"User-Agent": AGENT})  # noqa: S310 - https, fixed hosts
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as reply:  # noqa: S310
                return reply.read()
        except (urllib.error.URLError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": AGENT})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as reply:  # noqa: S310
            size = reply.headers.get("Content-Length")
            return int(size) if size and size.isdigit() else None
    except (urllib.error.URLError, OSError):
        return None


def sha256_of(url: str) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={"User-Agent": AGENT})  # noqa: S310
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as reply:  # noqa: S310
        while chunk := reply.read(1 << 20):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


class _Page(HTMLParser):
    """A pack page: its first paragraph and the rows of its asset table."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[dict]] = []
        self.headers: list[str] = []
        self.intro = ""
        self._in_article = False
        self._p_depth = 0
        self._p_text: list[str] = []
        self._cell: dict | None = None
        self._row: list[dict] | None = None
        self._in_th = False
        self._th: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        if tag == "article":
            self._in_article = True
        elif tag == "p" and self._in_article and not self.intro:
            self._p_depth += 1
        elif tag == "tr":
            self._row = []
        elif tag == "th":
            self._in_th, self._th = True, []
        elif tag == "td" and self._row is not None:
            self._cell = {"text": [], "img": None, "href": None}
        elif self._cell is not None:
            if tag == "img" and self._cell["img"] is None and "figure-image" in (a.get("class") or ""):
                self._cell["img"] = a.get("src")
            elif tag == "a" and a.get("rel") == "external":
                self._cell["href"] = a.get("href")

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._p_depth:
            self._p_depth -= 1
            self.intro = " ".join("".join(self._p_text).split())
        elif tag == "th" and self._in_th:
            self._in_th = False
            self.headers.append(" ".join("".join(self._th).split()).lower())
        elif tag == "td" and self._cell is not None and self._row is not None:
            self._cell["text"] = " ".join("".join(self._cell["text"]).split())
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._p_depth:
            self._p_text.append(data)
        if self._in_th:
            self._th.append(data)
        if self._cell is not None:
            self._cell["text"].append(data)


def _megabytes(text: str) -> float | None:
    m = _SIZE.search(text)
    if not m:
        return None
    n = float(m.group(1))
    return n * {"k": 1 / 1024, "m": 1, "g": 1024}[m.group(2).lower()]


def pack_ids() -> list[str]:
    page = fetch(INDEX).decode("utf-8", "replace")
    seen: list[str] = []
    for name in _PACK_LINK.findall(page):
        if not name.startswith("release_") and name not in seen:
            seen.append(name)
    return seen


def read_pack(pack: str) -> dict | None:
    """A pack's page as a catalogue entry (without the hash), or None if it offers no zip."""
    url = f"{SITE}/assets/assetpacks/{pack}.html"
    raw = fetch(url).decode("utf-8", "replace")
    page = _Page()
    page.feed(raw)
    zips = []
    for z in _ZIP.findall(raw):
        z = html.unescape(z)
        if urllib.parse.urlsplit(z).hostname in HOSTS and z not in zips:
            zips.append(z)
    if not zips:
        return None
    # files.* first: files2 is the mirror the index says to use only if the other fails.
    zips.sort(key=lambda z: urllib.parse.urlsplit(z).hostname != HOSTS[0])
    col = {h: i for i, h in enumerate(page.headers)}
    assets = []
    for row in page.rows:
        if len(row) < len(col) or "asset name" not in col:
            continue

        def cell(name: str, row: list[dict] = row) -> dict:
            i = col.get(name)
            return row[i] if i is not None else {"text": "", "img": None, "href": None}

        name = cell("asset name")["text"]
        if not name:
            continue
        thumb = cell("thumbnail")["img"]
        entry = {
            "name": name,
            "type": _TYPES.get(kind := cell("asset type")["text"].lower(), kind),
            "author": cell("author")["text"],
            "license": cell("license")["text"],
        }
        if thumb:
            entry["thumb"] = urllib.parse.urljoin(url, thumb)
        if source := cell("source")["href"]:
            entry["source"] = source
        assets.append(entry)
    licences = sorted({a["license"] for a in assets if a["license"]})
    if not licences:  # a pack without a table: the zip's name says its licence
        stem = Path(urllib.parse.urlsplit(zips[0]).path).stem.lower()
        by_name = "CC0" if stem.endswith("_cc0") else "CC-BY" if stem.endswith("_ccby") else STATED.get(pack)
        licences = [by_name] if by_name else []
    return {
        "id": pack,
        "page": url,
        "description": _MIRRORS.sub("", page.intro).strip(),
        "license": " AND ".join(licences),
        "urls": zips,
        "megabytes": _megabytes(page.intro),
        "assets": assets,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-hash", action="store_true", help="read the pages only; do not download zips to hash them")
    ap.add_argument("--only", action="append", default=[], help="refresh only this pack (repeatable)")
    args = ap.parse_args(argv)

    old: dict[str, dict] = {}
    if args.out.is_file():
        old = {p["id"]: p for p in json.loads(args.out.read_text(encoding="utf-8")).get("packs", [])}

    ids = pack_ids()
    print(f"{len(ids)} packs listed on {INDEX}", file=sys.stderr)
    packs: list[dict] = []
    for pack in ids:
        if args.only and pack not in args.only:
            if pack in old:
                packs.append(old[pack])
            continue
        entry = read_pack(pack)
        if entry is None:
            print(f"  {pack}: no zip on its page, skipped", file=sys.stderr)
            continue
        size = head_size(entry["urls"][0])
        before = old.get(pack, {})
        if size and before.get("bytes") == size and before.get("urls", [None])[0] == entry["urls"][0]:
            entry["bytes"], entry["sha256"] = size, before.get("sha256")
        elif not args.no_hash:
            t = time.monotonic()
            entry["sha256"], entry["bytes"] = sha256_of(entry["urls"][0])
            print(f"  {pack}: hashed {entry['bytes'] / 2**20:.0f} MB in {time.monotonic() - t:.0f} s", file=sys.stderr)
        else:
            entry["bytes"], entry["sha256"] = size, before.get("sha256") if before.get("bytes") == size else None
        entry.pop("megabytes")
        packs.append(entry)
        print(f"  {pack}: {len(entry['assets'])} assets, {entry['license'] or 'no licence'}", file=sys.stderr)

    out = {
        "source": INDEX,
        "hosts": list(HOSTS),
        "note": "Generated by scripts/make_asset_catalogue.py. Do not edit by hand.",
        "packs": packs,
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(packs)} packs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
