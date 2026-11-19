from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from argus_mask.faces import FaceStore, build_manifest
from argus_mask.models import FaceManifest
from tests.helpers import STOCK, make_glb, make_splat_zip

_SPEC = importlib.util.spec_from_file_location(
    "fetch_stock_head", Path(__file__).resolve().parent.parent / "scripts" / "fetch_stock_head.py"
)
fetch_stock_head = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fetch_stock_head)


@pytest.fixture
def published(tmp_path, monkeypatch):
    """A lock directory for one head, and a stand-in for the download that serves *served* bytes."""
    model = make_glb()
    source = tmp_path / "built" / "head.glb"
    source.parent.mkdir()
    source.write_bytes(model)
    manifest = build_manifest(source, face_id="stock-test", name="Test head", provenance=STOCK)
    locks = tmp_path / "locks"
    locks.mkdir()
    url = "https://example.invalid/stock-test.glb"
    (locks / "stock-test.lock.json").write_text(
        json.dumps({"url": url, "size": len(model), "manifest": manifest.model_dump(mode="json")})
    )
    served = {"body": model, "calls": 0}

    class Reply:
        def __init__(self) -> None:
            self._rest = served["body"]

        def read(self, n: int) -> bytes:
            block, self._rest = self._rest[:n], self._rest[n:]
            return block

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    def urlopen(requested: str, timeout: float):
        assert requested == url
        served["calls"] += 1
        return Reply()

    monkeypatch.setattr(fetch_stock_head.urllib.request, "urlopen", urlopen)
    return locks, served


def test_a_fetched_head_is_one_the_server_will_serve(tmp_path, published) -> None:
    locks, served = published
    directory, downloaded = fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks)
    assert downloaded and directory == tmp_path / "assets" / "stock-test"
    manifest, model = FaceStore(tmp_path / "assets").load("stock-test")
    assert manifest.provenance.kind == "stock" and model.read_bytes() == served["body"]
    # A second run finds the same file and downloads nothing.
    assert fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks) == (directory, False)
    assert served["calls"] == 1


@pytest.mark.parametrize("tamper", ["changed", "truncated", "padded"])
def test_a_download_that_does_not_match_the_lock_leaves_nothing(tmp_path, published, tamper: str) -> None:
    locks, served = published
    body = served["body"]
    served["body"] = {"changed": body[:-1] + bytes([body[-1] ^ 1]), "truncated": body[:-10], "padded": body + b"extra"}[
        tamper
    ]
    with pytest.raises(fetch_stock_head.FetchError):
        fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks)
    assert not (tmp_path / "assets" / "stock-test" / "manifest.json").exists()
    assert list((tmp_path / "assets").rglob("*.glb")) == [] and list((tmp_path / "assets").rglob("*.part")) == []


def test_an_existing_different_model_is_kept_unless_forced(tmp_path, published) -> None:
    locks, _ = published
    mine = tmp_path / "assets" / "stock-test" / "head.glb"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(make_glb(["jawOpen"]))
    with pytest.raises(fetch_stock_head.FetchError, match="different file"):
        fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks)
    assert mine.read_bytes() == make_glb(["jawOpen"])
    fetch_stock_head.fetch("stock-test", tmp_path / "assets", force=True, lock_dir=locks)
    FaceStore(tmp_path / "assets").load("stock-test")


def test_only_https_and_only_known_ids(tmp_path, published) -> None:
    locks, _ = published
    with pytest.raises(fetch_stock_head.FetchError, match="only https"):
        fetch_stock_head.download("http://example.invalid/x.glb", 1, "0" * 64, tmp_path / "x.glb")
    with pytest.raises(fetch_stock_head.FetchError, match="known heads: stock-test"):
        fetch_stock_head.fetch("nobody", tmp_path / "assets", lock_dir=locks)
    with pytest.raises(fetch_stock_head.FetchError, match="not a face id"):
        fetch_stock_head.fetch("../escape", tmp_path / "assets", lock_dir=locks)


def test_every_committed_lock_is_a_valid_manifest() -> None:
    locks = sorted(fetch_stock_head.LOCK_DIR.glob("*.lock.json"))
    assert locks, "the repo should lock at least one stock head"
    for path in locks:
        lock = fetch_stock_head.read_lock(path.name.removesuffix(".lock.json"))
        manifest = FaceManifest.model_validate(lock["manifest"])
        assert manifest.provenance.kind in ("stock", "synthetic")
        assert lock["url"].startswith("https://github.com/smk762/argus-mask/releases/download/")


def _framed_lock(tmp_path: Path, monkeypatch, *, waist_served: bytes | None = None) -> Path:
    """A lock for a head with a waist-up model, and a download that serves both (the waist's bytes overridable)."""
    import hashlib  # noqa: PLC0415

    head, waist = make_glb(), make_glb(("jawOpen", "mouthClose"))
    built = tmp_path / "built"
    built.mkdir()
    (built / "head.glb").write_bytes(head)
    manifest = build_manifest(built / "head.glb", face_id="stock-test", name="Test head", provenance=STOCK)
    body = manifest.model_dump(mode="json")
    body["framings"] = {
        "waist": {
            "model": {"file": "waist.glb", "sha256": hashlib.sha256(waist).hexdigest()},
            "framing": {"distance": 1.9, "height": -0.3, "fov": 30},
        }
    }
    urls = {"https://example.invalid/head.glb": head, "https://example.invalid/waist.glb": waist_served or waist}
    locks = tmp_path / "locks"
    locks.mkdir()
    lock = {
        "url": "https://example.invalid/head.glb",
        "size": len(head),
        "framings": {"waist": {"url": "https://example.invalid/waist.glb", "size": len(waist)}},
        "manifest": body,
    }
    (locks / "stock-test.lock.json").write_text(json.dumps(lock))

    class Reply:
        def __init__(self, data: bytes) -> None:
            self._rest = data

        def read(self, n: int) -> bytes:
            block, self._rest = self._rest[:n], self._rest[n:]
            return block

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    monkeypatch.setattr(fetch_stock_head.urllib.request, "urlopen", lambda url, timeout: Reply(urls[url]))
    return locks


def test_a_head_with_a_waist_framing_fetches_both_models(tmp_path, monkeypatch) -> None:
    locks = _framed_lock(tmp_path, monkeypatch)
    directory, downloaded = fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks)
    assert downloaded and sorted(p.name for p in directory.iterdir()) == ["head.glb", "manifest.json", "waist.glb"]
    manifest = FaceStore(tmp_path / "assets").load("stock-test")[0]
    assert manifest.framings["waist"].model.file == "waist.glb"


def test_a_waist_download_that_does_not_match_writes_nothing_of_it(tmp_path, monkeypatch) -> None:
    locks = _framed_lock(tmp_path, monkeypatch, waist_served=make_glb(("jawOpen",)))
    with pytest.raises(fetch_stock_head.FetchError, match="waist.glb"):
        fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks)
    assert not (tmp_path / "assets" / "stock-test" / "waist.glb").exists()


def test_a_framing_the_lock_cannot_fetch_is_refused(tmp_path, monkeypatch) -> None:
    locks = _framed_lock(tmp_path, monkeypatch)
    path = locks / "stock-test.lock.json"
    lock = json.loads(path.read_text())
    del lock["framings"]
    path.write_text(json.dumps(lock))
    with pytest.raises(fetch_stock_head.FetchError, match="nowhere to get it"):
        fetch_stock_head.read_lock("stock-test", locks)


def _splat_lock(tmp_path: Path, monkeypatch, *, file: str | None = None) -> tuple[Path, bytes]:
    """A lock for a LAM splat head (#126): a .zip model, its manifest's format lam-splat."""
    built = tmp_path / "built"
    built.mkdir()
    model = make_splat_zip(built / "splat-test.zip").read_bytes()
    manifest = build_manifest(built / "splat-test.zip", face_id="splat-test", name="Splat", provenance=STOCK)
    body = manifest.model_dump(mode="json")
    if file is not None:
        body["model"]["file"] = file
    locks = tmp_path / "locks"
    locks.mkdir()
    url = "https://example.invalid/splat-test.zip"
    (locks / "splat-test.lock.json").write_text(json.dumps({"url": url, "size": len(model), "manifest": body}))

    class Reply:
        def __init__(self) -> None:
            self._rest = model

        def read(self, n: int) -> bytes:
            block, self._rest = self._rest[:n], self._rest[n:]
            return block

        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    monkeypatch.setattr(fetch_stock_head.urllib.request, "urlopen", lambda requested, timeout: Reply())
    return locks, model


def test_a_fetched_splat_head_is_one_the_server_will_serve(tmp_path, monkeypatch) -> None:
    locks, model = _splat_lock(tmp_path, monkeypatch)
    directory, downloaded = fetch_stock_head.fetch("splat-test", tmp_path / "assets", lock_dir=locks)
    assert downloaded
    manifest, served = FaceStore(tmp_path / "assets").load("splat-test")
    assert manifest.model.format == "lam-splat" and served.suffix == ".zip" and served.read_bytes() == model


def test_a_splat_lock_must_name_a_zip(tmp_path, monkeypatch) -> None:
    locks, _ = _splat_lock(tmp_path, monkeypatch, file="splat-test.glb")
    with pytest.raises(fetch_stock_head.FetchError, match=r"plain \.zip name"):
        fetch_stock_head.read_lock("splat-test", locks)


def test_a_fetched_head_gets_the_usual_file_mode_not_mkstemps(tmp_path, published) -> None:
    """mkstemp makes the download 0600; another user (a service, a container) must still read the head."""
    import os  # noqa: PLC0415
    import stat  # noqa: PLC0415

    locks, _ = published
    old = os.umask(0o022)
    try:
        directory, _ = fetch_stock_head.fetch("stock-test", tmp_path / "assets", lock_dir=locks)
    finally:
        os.umask(old)
    model = next(directory.glob("*.glb"))
    assert stat.S_IMODE(model.stat().st_mode) == 0o644
