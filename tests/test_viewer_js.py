"""The viewer's JavaScript, where it can be held without a browser.

The clock is run in node (skipped where there is none) with a stub AudioContext and a stub page clock. The splat
head's clean-up needs a WebGL renderer, so its shape is checked in the source instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parent.parent / "src" / "argus_mask" / "viewer"


def _block(source: str, start: int) -> str:
    """The text from ``start`` to the close of the first brace block after it."""
    depth, i = 0, source.index("{", start)
    for j in range(i, len(source)):
        depth += {"{": 1, "}": -1}.get(source[j], 0)
        if depth == 0:
            return source[start : j + 1]
    raise AssertionError("no end")


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_timeline_runs_on_when_the_audio_context_is_suspended_and_resumed() -> None:
    """The audio clock and the page clock have unrelated origins: a switch between them carries the anchor across."""
    source = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    functions = "\n".join(_block(source, source.index(f"function {n}(")) for n in ("clockNow", "timelineNow"))
    script = f"""
const page = {{ t: 1000 }};
const performance = {{ now: () => page.t * 1000 }};
let audio = {{ state: 'running', currentTime: 10 }};
let anchor = null;
let clockDomain = null;
{functions}
const out = [];
clockNow();
anchor = {{ clock: 10.25, t: 0 }};
audio.currentTime = 10.5; page.t = 1000.25; out.push(timelineNow());   // playing: 250 ms in
audio.state = 'suspended'; out.push(timelineNow());                    // suspended: the audio clock stands
page.t = 1001.25; out.push(timelineNow());                             // a second on the page's clock
audio.state = 'running'; out.push(timelineNow());                      // resumed: the audio clock again
audio.currentTime = 11; out.push(timelineNow());
console.log(JSON.stringify(out));
"""
    run = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30)
    times = json.loads(run.stdout)
    assert times == pytest.approx([250, 250, 1250, 1250, 1750], abs=1e-6), times


def test_a_splat_head_that_fails_to_come_up_is_disposed() -> None:
    """Past ``head.r = r`` the renderer holds a canvas, a context and a worker: a throw there disposes the head."""
    source = (VIEWER / "splat.js").read_text(encoding="utf-8")
    create = source[source.index("static async create(") : source.index("get canvas()")]
    after = create[create.index("head.r = r;") :]
    assert "try {" in after, "the work after the renderer is made is unguarded"
    guarded = after[after.index("try {") :]
    body = _block(guarded, 0)
    rest = guarded[len(body) :]
    assert "did not load in 60 s" in body
    assert rest.lstrip().startswith("catch (")
    handler = _block(rest, 0)
    assert handler.index("head.dispose()") < handler.index("throw ")
