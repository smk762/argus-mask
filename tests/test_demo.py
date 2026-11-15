"""The hosted demo (#118): the committed clips agree with their spec, the contract and the stock heads, and the
page's build pieces hold together. No network: the faces are not fetched here (demo/build.py does that)."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import re
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
