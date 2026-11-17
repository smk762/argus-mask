"""The demo player's JavaScript, run in node with stubs where it can be held without a browser."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_viewer_js import _block

PLAYER = Path(__file__).resolve().parent.parent / "demo" / "player.js"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _function(source: str, head: str) -> str:
    """The function that starts with ``head``, to its closing brace (its parameters may hold braces)."""
    start = source.index(head)
    body = source.index(") {", start) + 2
    return source[start:body] + _block(source, body)


@needs_node
def test_a_splat_head_that_fails_to_come_up_leaves_the_clip_on_show() -> None:
    """``SplatHead.create`` throwing leaves the stage as it was: the clip before is still shown, nothing cleared."""
    source = PLAYER.read_text(encoding="utf-8")
    make = _function(source, "async function makeSplat(")
    script = f"""
let loadSeq = 1;
const stale = (token) => token !== loadSeq;
let shown = {{ clip: 'before' }};
let cleared = 0;
function clear() {{ cleared += 1; shown = null; }}
const BACKGROUND = '#000';
const hosts = [];
const document = {{ createElement: () => ({{ hidden: false, remove() {{ this.removed = true; }} }}) }};
const stageEl = {{ prepend: (h) => hosts.push(h) }};
{make}
const mod = {{
  readSkin: async () => ({{ budget: null }}),
  SplatHead: {{ create: async () => {{ throw new Error('did not load in 60 s'); }} }},
}};
makeSplat('f', {{ model: {{ file: 'head.zip' }}, framing: {{}} }}, 1, mod).then(
  () => console.log(JSON.stringify({{ threw: false }})),
  (err) => console.log(JSON.stringify({{ threw: true, message: err.message, shown, cleared, removed: hosts.every((h) => h.removed) }})));
"""
    run = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30)
    out = json.loads(run.stdout)
    assert out == {
        "threw": True,
        "message": "did not load in 60 s",
        "shown": {"clip": "before"},
        "cleared": 0,
        "removed": True,
    }
