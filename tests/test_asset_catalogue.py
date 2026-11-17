"""scripts/make_asset_catalogue.py: a refresh never loses what the catalogue already knows to a page or HEAD that
fails once, and reads a source link whatever else its rel says."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "make_asset_catalogue_t", Path(__file__).resolve().parent.parent / "scripts" / "make_asset_catalogue.py"
)
cat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cat)

ZIP = "https://files.makehumancommunity.org/asset_packs/hair09/hair09_ccby.zip"
PAGE = f"""<html><body><main><article class="default">
<p>Buns: <a href="{ZIP}" rel="external">mirror1</a>, <a href="{ZIP}" rel="external">mirror2</a> (12 mb)</p>
<table><thead><tr><th>Asset type</th><th>Asset name</th><th>Author</th><th>Source</th><th>License</th></tr></thead>
<tbody><tr><td>hair</td><td>neat_bun</td><td>Some One</td>
<td><a href="http://www.makehumancommunity.org/node/1" rel="noopener external">asset repo</a></td><td>CC-BY</td></tr>
</tbody></table></article></main></body></html>"""
KNOWN = {"id": "hair09", "urls": [ZIP], "bytes": 1234, "sha256": "a" * 64, "assets": []}


def _run(monkeypatch, out: Path, *args: str, page: str = PAGE, size: int | None = 1234) -> list[dict]:
    monkeypatch.setattr(cat, "fetch", lambda url: page.encode())
    monkeypatch.setattr(cat, "pack_ids", lambda: ["hair09"])
    monkeypatch.setattr(cat, "head_size", lambda url: size)
    monkeypatch.setattr(cat, "sha256_of", lambda url: pytest.fail("nothing is hashed here"))
    assert cat.main(["--out", str(out), *args]) == 0
    return json.loads(out.read_text())["packs"]


def test_a_source_link_is_read_whatever_else_its_rel_says(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cat, "fetch", lambda url: PAGE.encode())
    [asset] = cat.read_pack("hair09")["assets"]
    assert asset["source"] == "http://www.makehumancommunity.org/node/1"


def test_a_failed_head_under_no_hash_keeps_the_known_size_and_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "catalogue.json"
    out.write_text(json.dumps({"packs": [KNOWN]}))
    [pack] = _run(monkeypatch, out, "--no-hash", size=None)
    assert pack["bytes"] == 1234 and pack["sha256"] == "a" * 64


def test_a_pack_whose_page_lacks_its_zip_once_keeps_its_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "catalogue.json"
    out.write_text(json.dumps({"packs": [KNOWN]}))
    assert _run(monkeypatch, out, "--no-hash", page="<html><article><p>down for now</p></article></html>") == [KNOWN]
    # A pack the catalogue never had is still skipped.
    out.write_text(json.dumps({"packs": []}))
    assert _run(monkeypatch, out, "--no-hash", page="<html></html>") == []
