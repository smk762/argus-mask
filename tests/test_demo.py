"""The hosted demo (#118): the committed clips agree with their spec, the contract and the stock heads, and the
page's build pieces hold together. No network: the faces are not fetched here (demo/build.py does that)."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from argus_mask.models import CLEARED_KINDS
from argus_mask.take import read_take

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
SPEC = json.loads((DEMO / "clips.json").read_text(encoding="utf-8"))
INDEX = json.loads((DEMO / "clips" / "index.json").read_text(encoding="utf-8"))
BUILT_IN_SCENES = {"ftl", "garden", "lake"}  # viewer/backdrop.js SCENES: a plate of one's own is not in the demo


def _build():
    spec = importlib.util.spec_from_file_location("demo_build", DEMO / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _lock(face: str) -> dict:
    return json.loads((ROOT / "scripts" / "stock_heads" / f"{face}.lock.json").read_text(encoding="utf-8"))


def test_every_clip_in_the_spec_is_a_cleared_stock_head_on_a_built_in_scene() -> None:
    ids = [c["id"] for c in SPEC["clips"]]
    assert len(set(ids)) == len(ids)
    for clip in SPEC["clips"]:
        manifest = _lock(clip["face"])["manifest"]
        assert manifest["provenance"]["kind"] in CLEARED_KINDS
        assert clip["framing"] == "head" or clip["framing"] in manifest.get("framings", {})
        assert clip["scene"] in BUILT_IN_SCENES
        assert clip["voice"].startswith("speaker:"), "a service's own speaker, never a clone"


def test_the_index_lists_every_clip_once_in_the_spec_order() -> None:
    order = [c["id"] for c in SPEC["clips"]]
    assert INDEX, "the demo has clips"
    assert [i for i in order if i in INDEX] == INDEX and len(set(INDEX)) == len(INDEX)


@pytest.mark.parametrize("clip_id", INDEX)
def test_a_committed_clip_is_what_its_spec_says(clip_id: str) -> None:
    spec = next(c for c in SPEC["clips"] if c["id"] == clip_id)
    folder = DEMO / "clips" / clip_id
    meta = json.loads((folder / "clip.json").read_text(encoding="utf-8"))
    assert {k: meta[k] for k in ("face", "framing", "scene", "title")} == {
        k: spec[k] for k in ("face", "framing", "scene", "title")
    }
    assert [c["text"] for c in meta["captions"]] == spec["sentences"]
    spans = [(c["start_ms"], c["end_ms"]) for c in meta["captions"]]
    assert all(a < b for a, b in spans) and all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:], strict=False))
    assert spans[-1][1] <= meta["duration_ms"]
    header, frames, end = read_take(folder / "take.ndjson")
    assert header.face_id == spec["face"] and header.seed == spec["seed"] and header.fps == SPEC["fps"]
    assert header.driver.name == "a2f", "the published clips are Audio2Face's, not the energy stand-in"
    assert end is not None and end.frames == len(frames) == meta["frames"] and end.duration_ms == meta["duration_ms"]
    assert any(f.cue for f in frames), "the frames carry their cues, for the gestures"
    assert (folder / "audio.m4a").stat().st_size > 10_000
    assert meta["made_with"]["voice"] == spec["voice"].removeprefix("speaker:")
    assert set(p.name for p in folder.iterdir()) == {"clip.json", "take.ndjson", "audio.m4a"}


def test_the_page_pastes_the_strip_and_allows_its_import_map_by_hash() -> None:
    build = _build()
    page = (DEMO / "index.html").read_text(encoding="utf-8")
    assert build.MARKER in page and 'data-dh-site="mask"' in page
    assert build.strip_html().startswith('<nav class="dh-strip"') and build.strip_html().endswith("</nav>")
    importmap = re.search(r'<script type="importmap">(.*?)</script>', page, re.S).group(1)
    digest = base64.b64encode(hashlib.sha256(importmap.encode()).digest()).decode()
    policy = build.csp(page)
    assert f"'sha256-{digest}'" in policy and "unsafe" not in policy
    assert not re.search(r"<script(?![^>]*\bsrc=)(?![^>]*importmap)[^>]*>", page), "no other inline script"


# ---- /docs/ (#123) ---------------------------------------------------------------------------------------------


def _clips() -> list[dict]:
    return [json.loads((DEMO / "clips" / i / "clip.json").read_text(encoding="utf-8")) for i in INDEX]


def _has_commit(commit: str) -> bool:
    return (
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"], check=False).returncode == 0
    )  # noqa: S603, S607


@pytest.fixture(scope="module")
def docs_site(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """The docs pages, built without clips: the clips' drift check needs history a shallow CI clone lacks."""
    build = _build()
    out = tmp_path_factory.mktemp("site")
    return out, build.docs_page.build(out, build.shell, [], env={})


def test_the_docs_are_every_public_doc_recorded_by_commit_and_sha256(docs_site: tuple[Path, dict]) -> None:
    out, record = docs_site
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)  # noqa: S603, S607
    assert record["commit"] == head.stdout.strip() and record["repo"] == "smk762/argus-mask"
    assert set(record["docs"]) == {"README.md", *(f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md"))}
    assert not {"CLAUDE.md", "AGENTS.md", "spec.md"} & set(record["docs"]), "agent and planning files never go out"
    for path, digest in record["docs"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in record["images"].items():
        assert hashlib.sha256((out / "docs" / "files" / path).read_bytes()).hexdigest() == digest
    assert json.loads((out / "docs" / "sources.json").read_text(encoding="utf-8")) == record
    assert (out / "docs" / "index.html").is_file() and (out / "docs" / "consent" / "index.html").is_file()


def test_a_docs_page_is_csp_neutral_and_links_stay_on_site(docs_site: tuple[Path, dict]) -> None:
    out, record = docs_site
    for page in (out / "docs").rglob("index.html"):
        text = page.read_text(encoding="utf-8")
        assert re.findall(r"<script[^>]*>", text) == ['<script src="/vendor/dh/dh-theme.js">'], page
        assert " style=" not in text and "<style" not in text, page
        assert not re.search(r'<img[^>]+src="(?!/docs/files/)', text), "images are copied in, never hot-linked"
        assert "&lt;picture" not in text and "&lt;!--" not in text, "raw HTML blocks are dropped, not shown"
        assert f"@ <code>{record['commit'][:7]}</code>" in text and "pypi.org/project/argus-mask/" in text
        assert "/blob/main/" not in text, "the repo's own files are linked at the imported commit"
    readme = (out / "docs" / "index.html").read_text(encoding="utf-8")
    assert 'id="hosted-demo"' in readme and 'href="/docs/consent/' in readme


def test_the_opsec_gate_refuses_and_never_echoes_the_denylist() -> None:
    docs = _build().docs_page
    text = "see 192.168.1.5 and /home/someone/x\nmail ops@example.com about Mithril\nfine: docs/media/a.jpg, 127.0.0.1"
    hits = docs.opsec_hits(text, ["mithril"])
    assert sorted((n, kind) for n, kind, _ in hits) == [
        (1, "home directory path"),
        (1, "private IPv4 address"),
        (2, "denylisted term"),
        (2, "email address"),
    ]
    assert "Mithril" not in str(hits) and "mithril" not in str(hits)
    with pytest.raises(docs.Refused, match="opsec gate refused"):
        docs.gate("x.md", text, [])
    assert docs.denylist({"OPSEC_DENYLIST": "a, b\nc"}) == ["a", "b", "c"]


def test_versions_and_clip_commits_read_as_hatch_and_git_do() -> None:
    docs = _build().docs_page
    assert docs.version_from_describe("v0.2.0-65-g1cdb94f") == "0.2.1.dev65+g1cdb94f"
    assert docs.version_from_describe("v0.3.0-0-gabcdef1") == "0.3.0"
    assert docs.clip_commit("v0.2.0-56-g0c6a9ec") == "0c6a9ec"
    with pytest.raises(docs.Refused, match="uncommitted"):
        docs.clip_commit("v0.2.0-56-g0c6a9ec-dirty")


def test_the_committed_clips_were_made_by_the_engine_at_head() -> None:
    clips = _clips()
    docs = _build().docs_page
    if not all(_has_commit(docs.clip_commit(c["made_with"]["argus_mask"])) for c in clips):
        pytest.skip("a shallow clone: the clips' commits are not here (the deploy fetches the full history)")
    made = docs.clip_drift(clips)
    assert set(made) == set(INDEX)


def test_the_site_has_a_real_404_and_the_demo_links_its_docs() -> None:
    build = _build()
    page = build.shell("Not found · Argus Mask", build.NOT_FOUND)
    assert 'href="/docs/"' in page and re.findall(r"<script[^>]*>", page) == ['<script src="/vendor/dh/dh-theme.js">']
    demo = (DEMO / "index.html").read_text(encoding="utf-8")
    assert 'href="docs/"' in demo and 'id="made-line"' in demo
