"""The asset library: its licence line and its cached scan."""

from __future__ import annotations

import json
from pathlib import Path

from argus_mask.editor.library import Asset, AssetLibrary, licence_expression


def asset(name: str, licence: str) -> Asset:
    return Asset(
        part="hair", name=name, ref=f"hair/{name}/{name}.mhclo", license=licence, author="", pack="p", thumb=None
    )


def test_a_face_of_no_listed_assets_is_cc0_not_an_empty_licence() -> None:
    assert licence_expression({}) == "CC0-1.0"


def test_a_compound_licence_keeps_its_meaning_when_joined() -> None:
    joined = licence_expression({"hair": asset("h1", "MIT OR Apache-2.0"), "skin": asset("s1", "CC-BY-4.0")})
    assert joined == "CC-BY-4.0 AND (MIT OR Apache-2.0)"
    # Alone, or a plain id with a space in it, it is written as it is.
    assert licence_expression({"hair": asset("h1", "MIT OR Apache-2.0")}) == "MIT OR Apache-2.0"
    plain = {"hair": asset("h1", "CC-BY 4.0"), "skin": asset("s1", "CC0-1.0")}
    assert licence_expression(plain) == "CC-BY 4.0 AND CC0-1.0"


def test_a_file_written_into_an_existing_asset_directory_is_seen(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "packs").mkdir(parents=True)
    (data / "packs" / "p.json").write_text(json.dumps({"h1": {"type": "hair", "license": "CC0"}}))
    sub = data / "hair" / "h1"
    sub.mkdir(parents=True)  # an install part way: the directory is there, its asset file not yet
    library = AssetLibrary(data)
    assert library.scan()["hair"] == []
    (sub / "h1.mhclo").write_text("# asset\n")
    assert [a.name for a in library.scan()["hair"]] == ["h1"], "the cached scan is not reused"
    (sub / "h1.thumb").write_bytes(b"\x89PNG")
    assert library.scan()["hair"][0].thumb == sub / "h1.thumb"
