"""Asset packs: the catalogue, and installing a pack by id. Zips are built here; nothing reaches the network."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import stat
import time
import zipfile
from pathlib import Path

import pytest

from argus_mask.editor.library import AssetLibrary
from argus_mask.editor.packs import Installs, PackError, download, load_catalogue, plan, unpack

URL = "https://files.makehumancommunity.org/asset_packs/hair09/hair09_cc0.zip"
MIRROR = "https://files2.makehumancommunity.org/asset_packs/hair09/hair09_cc0.zip"


def pack_zip(pack: str = "hair09", extra: dict[str, bytes] | None = None, index: bool = True) -> bytes:
    """A pack as the community zips one: part directories and packs/<pack>.json at the top."""
    files = {
        "hair/neat_bun/neat_bun.mhclo": b"name neat_bun\n",
        "hair/neat_bun/neat_bun.thumb": b"\x89PNG",
        "hair/neat_bun/neat_bun.mhmat": b"name neat_bun\n",
    }
    if index:
        files[f"packs/{pack}.json"] = json.dumps(
            {"neat_bun": {"type": "hair", "license": "CC0", "author": "someone"}}
        ).encode()
    files |= extra or {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("hair/", b"")
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def write_catalogue(path: Path, data: bytes, *, pack: str = "hair09", **over) -> Path:
    entry = {
        "id": pack,
        "page": f"https://static.makehumancommunity.org/assets/assetpacks/{pack}.html",
        "description": "A bun.",
        "license": "CC0",
        "urls": [URL, MIRROR],
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "assets": [
            {
                "name": "neat_bun",
                "type": "hair",
                "author": "someone",
                "license": "CC0",
                "thumb": "https://static.makehumancommunity.org/assets/assetpacks/neat_bun.png",
            }
        ],
    } | over
    path.write_text(json.dumps({"packs": [entry]}), encoding="utf-8")
    return path


def serving(blobs: dict[str, bytes]):
    """An opener that answers each URL with its bytes, and refuses any other."""
    asked: list[str] = []

    def opener(url: str):
        asked.append(url)
        if url not in blobs:
            raise OSError(f"no route to {url}")
        return io.BytesIO(blobs[url])

    opener.asked = asked  # type: ignore[attr-defined]
    return opener


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "mpfb" / "data"
    (d / "packs").mkdir(parents=True)
    return d


def files_under(d: Path) -> set[str]:
    return {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()}


# ---- the catalogue ---------------------------------------------------------------------------------


def test_the_shipped_catalogue_loads_and_points_only_at_the_community_hosts() -> None:
    catalogue = load_catalogue()
    for pack in catalogue.values():
        assert all(u.startswith("https://files") for u in pack.urls)
    if catalogue:  # filled by scripts/make_asset_catalogue.py
        assert "suits02" in catalogue


def test_a_catalogue_entry_pointing_anywhere_else_is_dropped(tmp_path: Path) -> None:
    data = pack_zip()
    path = write_catalogue(
        tmp_path / "c.json", data, urls=["https://evil.example/hair09.zip", "http://files.makehumancommunity.org/x.zip"]
    )
    assert load_catalogue(path) == {}


def test_a_thumbnail_off_the_community_site_is_not_shown(tmp_path: Path) -> None:
    data = pack_zip()
    path = write_catalogue(
        tmp_path / "c.json",
        data,
        assets=[{"name": "x", "type": "hair", "license": "CC0", "thumb": "https://tracker.example/x.png"}],
    )
    assert "thumb" not in load_catalogue(path)["hair09"].assets[0]


def test_a_pack_without_a_hash_or_with_another_licence_is_not_installable(tmp_path: Path) -> None:
    data = pack_zip()
    assert load_catalogue(write_catalogue(tmp_path / "a.json", data, sha256=None))["hair09"].why_not()
    assert (
        "licence" in load_catalogue(write_catalogue(tmp_path / "b.json", data, license="CC-BY-NC"))["hair09"].why_not()
    )
    assert (
        load_catalogue(write_catalogue(tmp_path / "c.json", data, license="CC0 AND CC-BY"))["hair09"].why_not() is None
    )


# ---- the download ----------------------------------------------------------------------------------


def test_a_download_with_the_wrong_hash_is_refused_and_the_mirror_tried(tmp_path: Path) -> None:
    good = pack_zip()
    pack = load_catalogue(write_catalogue(tmp_path / "c.json", good))["hair09"]
    opener = serving({URL: good[:-1] + b"!", MIRROR: good})
    dest = tmp_path / "p.zip"
    assert download(pack, dest, lambda n: None, opener) == MIRROR
    assert dest.read_bytes() == good

    opener = serving({URL: good[:-1] + b"!", MIRROR: good[:-1] + b"?"})
    with pytest.raises(PackError, match="sha256"):
        download(pack, dest, lambda n: None, opener)
    assert not dest.exists()


def test_a_download_that_runs_long_is_cut(tmp_path: Path) -> None:
    good = pack_zip()
    pack = load_catalogue(write_catalogue(tmp_path / "c.json", good, bytes=100))["hair09"]
    with pytest.raises(PackError, match="more than the catalogued"):
        download(pack, tmp_path / "p.zip", lambda n: None, serving({URL: good, MIRROR: good}))


# ---- the zip ---------------------------------------------------------------------------------------


def zip_file(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "pack.zip"
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("name", ["../escape.txt", "/etc/escape", "hair/../../escape", "C:/escape", "hair\\..\\escape"])
def test_a_member_outside_the_data_directory_is_refused(tmp_path: Path, data_dir: Path, name: str) -> None:
    path = zip_file(tmp_path, pack_zip(extra={name: b"x"}))
    with pytest.raises(PackError):
        unpack(path, data_dir, "hair09")
    assert files_under(data_dir) == set()
    assert not (tmp_path / "escape.txt").exists()


def test_a_link_in_the_zip_is_refused(tmp_path: Path, data_dir: Path) -> None:
    buf = io.BytesIO(pack_zip())
    with zipfile.ZipFile(buf, "a") as zf:
        info = zipfile.ZipInfo("hair/neat_bun/link")
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(PackError, match="link"):
        unpack(zip_file(tmp_path, buf.getvalue()), data_dir, "hair09")
    assert files_under(data_dir) == set()


def test_a_zip_without_its_pack_index_is_refused(tmp_path: Path, data_dir: Path) -> None:
    with pytest.raises(PackError, match="packs/hair09.json"):
        unpack(zip_file(tmp_path, pack_zip(index=False)), data_dir, "hair09")


def test_an_installed_file_is_never_replaced(tmp_path: Path, data_dir: Path) -> None:
    mine = data_dir / "hair" / "neat_bun" / "neat_bun.mhmat"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"my own material\n")
    with pytest.raises(PackError, match="replace installed files"):
        unpack(zip_file(tmp_path, pack_zip()), data_dir, "hair09")
    assert mine.read_bytes() == b"my own material\n"
    assert files_under(data_dir) == {"hair/neat_bun/neat_bun.mhmat"}


def test_a_file_already_there_and_the_same_is_skipped(tmp_path: Path, data_dir: Path) -> None:
    same = data_dir / "hair" / "neat_bun" / "neat_bun.mhmat"
    same.parent.mkdir(parents=True)
    same.write_bytes(b"name neat_bun\n")
    todo, skipped = plan(zip_file(tmp_path, pack_zip()), data_dir, "hair09")
    assert skipped == 1 and len(todo) == 3
    assert unpack(zip_file(tmp_path, pack_zip()), data_dir, "hair09") == {"files": 3, "skipped": 1}


def test_unpacking_leaves_no_staging_behind(tmp_path: Path, data_dir: Path) -> None:
    unpack(zip_file(tmp_path, pack_zip()), data_dir, "hair09")
    assert files_under(data_dir) == {
        "packs/hair09.json",
        "hair/neat_bun/neat_bun.mhclo",
        "hair/neat_bun/neat_bun.mhmat",
        "hair/neat_bun/neat_bun.thumb",
    }
    assert not any(p.name.startswith(".argus") for p in data_dir.iterdir())


# ---- the routes ------------------------------------------------------------------------------------


def editor_client(tmp_path: Path, data_dir: Path, installs: Installs):
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from argus_mask.config import EditorSettings, Settings  # noqa: PLC0415
    from argus_mask.editor.app import create_editor_app  # noqa: PLC0415

    ed = EditorSettings(mpfb_data=data_dir, voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    return TestClient(create_editor_app(Settings(assets_dir=tmp_path / "assets"), ed, packs=installs))


def wait(client, job: dict) -> dict:
    for _ in range(200):
        if job["state"] not in ("downloading", "unpacking"):
            return job
        time.sleep(0.02)
        job = client.get(f"/packs/jobs/{job['id']}").json()
    raise AssertionError("the install never finished")


def test_a_pack_installs_by_id_and_its_assets_are_offered(tmp_path: Path, data_dir: Path) -> None:
    data = pack_zip()
    catalogue = load_catalogue(write_catalogue(tmp_path / "c.json", data))
    opener = serving({URL: data})
    with editor_client(tmp_path, data_dir, Installs(data_dir, catalogue, opener)) as c:
        listed = c.get("/packs").json()
        assert listed["ready"] and listed["packs"][0]["installable"] and not listed["packs"][0]["installed"]
        assert c.get("/library").json()["hair"] == []

        job = c.post("/packs/hair09/install")
        assert job.status_code == 200, job.text
        job = wait(c, job.json())
        assert job["state"] == "done", job
        assert job["result"]["files"] == 4

        assert c.get("/packs").json()["packs"][0]["installed"]
        hair = c.get("/library").json()["hair"]
        assert [h["name"] for h in hair] == ["neat_bun"] and hair[0]["license"] == "CC0-1.0"
        assert c.post("/packs/hair09/install").status_code == 422  # already installed
    assert opener.asked == [URL]
    assert AssetLibrary(data_dir).find("hair", "neat_bun") is not None


def test_a_bad_download_installs_nothing(tmp_path: Path, data_dir: Path) -> None:
    data = pack_zip()
    catalogue = load_catalogue(write_catalogue(tmp_path / "c.json", data))
    with editor_client(tmp_path, data_dir, Installs(data_dir, catalogue, serving({URL: data + b"x"}))) as c:
        job = wait(c, c.post("/packs/hair09/install").json())
        assert job["state"] == "failed" and "sha256" in job["error"]
    assert files_under(data_dir) == set()


def test_only_a_catalogued_pinned_pack_installs(tmp_path: Path, data_dir: Path) -> None:
    data = pack_zip()
    catalogue = load_catalogue(write_catalogue(tmp_path / "c.json", data, sha256=None))
    opener = serving({})
    with editor_client(tmp_path, data_dir, Installs(data_dir, catalogue, opener)) as c:
        assert c.post("/packs/nothere/install").status_code == 404
        assert c.post("/packs/..%2F..%2Fetc/install").status_code in (404, 405)  # no route, or no pack
        refused = c.post("/packs/hair09/install")
        assert refused.status_code == 422 and "not pinned" in refused.json()["detail"]
    assert opener.asked == []


def test_installing_is_a_write_the_read_only_guard_refuses(tmp_path: Path, data_dir: Path) -> None:
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from argus_mask.config import EditorSettings, Settings  # noqa: PLC0415
    from argus_mask.editor.app import create_editor_app  # noqa: PLC0415

    data = pack_zip()
    installs = Installs(data_dir, load_catalogue(write_catalogue(tmp_path / "c.json", data)), serving({URL: data}))
    ed = EditorSettings(mpfb_data=data_dir, voices_dir=tmp_path / "voices", recordings_dir=tmp_path / "rec")
    app = create_editor_app(Settings(assets_dir=tmp_path / "assets", read_only=True), ed, packs=installs)
    with TestClient(app) as c:
        assert c.get("/packs").status_code == 200
        assert c.post("/packs/hair09/install").status_code == 403
    assert files_under(data_dir) == set()


# ---- the catalogue script --------------------------------------------------------------------------

_SPEC = importlib.util.spec_from_file_location(
    "make_asset_catalogue", Path(__file__).resolve().parent.parent / "scripts" / "make_asset_catalogue.py"
)
make_asset_catalogue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(make_asset_catalogue)

# A pack page as the community's static site writes one (trimmed).
PAGE = """<html><body><nav><p>Site menu</p></nav><main><article class="default">
<h1 id="hair09">hair09</h1>
<p>A set of buns by Someone: <a href="https://files2.makehumancommunity.org/asset_packs/hair09/hair09_ccby.zip" rel="external">mirror1</a>,
<a href="https://files.makehumancommunity.org/asset_packs/hair09/hair09_ccby.zip" rel="external">mirror2</a> (12 mb)</p>
<table><thead><tr><th>Asset type</th><th>Thumbnail</th><th>Asset name</th><th>Author</th><th>Source</th><th>License</th></tr></thead>
<tbody><tr><td>hair</td><td><a href="#R-1" class="lightbox-link"><img alt="neat_bun.png" class="lazy lightbox figure-image" src="neat_bun.png"></a>
<a href="javascript:history.back();" class="lightbox-back" id="R-1"><img class="lazy lightbox lightbox-image" src="neat_bun.png"></a></td>
<td>neat_bun</td><td>Some&nbsp;One</td><td><a href="http://www.makehumancommunity.org/node/1" rel="external">asset repo</a></td><td>CC-BY</td></tr>
</tbody></table>
<a href="https://evil.example/hair09.zip">elsewhere</a>
</article></main></body></html>"""


def test_the_script_reads_a_pack_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(make_asset_catalogue, "fetch", lambda url: PAGE.encode())
    entry = make_asset_catalogue.read_pack("hair09")
    assert entry["urls"] == [
        "https://files.makehumancommunity.org/asset_packs/hair09/hair09_ccby.zip",
        "https://files2.makehumancommunity.org/asset_packs/hair09/hair09_ccby.zip",
    ]
    assert entry["license"] == "CC-BY"
    assert entry["description"] == "A set of buns by Someone"
    assert entry["megabytes"] == 12
    assert entry["assets"] == [
        {
            "name": "neat_bun",
            "type": "hair",
            "author": "Some One",
            "license": "CC-BY",
            "thumb": "https://static.makehumancommunity.org/assets/assetpacks/neat_bun.png",
            "source": "http://www.makehumancommunity.org/node/1",
        }
    ]


def test_the_script_keeps_a_hash_while_the_zip_is_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "catalogue.json"
    out.write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "id": "hair09",
                        "urls": ["https://files.makehumancommunity.org/asset_packs/hair09/hair09_ccby.zip"],
                        "bytes": 1234,
                        "sha256": "a" * 64,
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(make_asset_catalogue, "fetch", lambda url: PAGE.encode())
    monkeypatch.setattr(make_asset_catalogue, "pack_ids", lambda: ["hair09"])
    monkeypatch.setattr(make_asset_catalogue, "head_size", lambda url: 1234)
    monkeypatch.setattr(make_asset_catalogue, "sha256_of", lambda url: pytest.fail("hashed again"))
    assert make_asset_catalogue.main(["--out", str(out)]) == 0
    [pack] = json.loads(out.read_text())["packs"]
    assert pack["sha256"] == "a" * 64 and pack["bytes"] == 1234 and "megabytes" not in pack

    monkeypatch.setattr(make_asset_catalogue, "head_size", lambda url: 999)  # the zip changed: hash it
    monkeypatch.setattr(make_asset_catalogue, "sha256_of", lambda url: ("b" * 64, 999))
    assert make_asset_catalogue.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text())["packs"][0]["sha256"] == "b" * 64


def test_hair_a_pack_types_as_clothes_is_offered_as_hair(tmp_path: Path, data_dir: Path) -> None:
    """MakeHuman 1.x hair packs (hair02, hair03) put styles in hair/ but type them clothes in their index."""
    index = {"elvs_bun": {"type": "clothes", "license": "CC-BY", "author": "Elvaerwyn"}}
    data = pack_zip(
        extra={
            "packs/hair09.json": json.dumps(index).encode(),
            "hair/elvs_bun/elvs_bun.mhclo": b"name elvs_bun\n",
        }
    )
    # The extra index replaces pack_zip's own: build it without one.
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(buf, "w") as out:
        seen = set()
        for info in reversed(src.infolist()):
            if info.filename not in seen:
                seen.add(info.filename)
                out.writestr(info.filename, src.read(info))
    unpack(zip_file(tmp_path, buf.getvalue()), data_dir, "hair09")
    library = AssetLibrary(data_dir)
    hair = library.find("hair", "elvs_bun")
    assert hair is not None and hair.license == "CC-BY" and hair.author == "Elvaerwyn" and hair.pack == "hair09"
    assert library.scan()["clothes"] == []  # never an outfit
