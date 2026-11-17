"""Reading a take: nothing after its end is taken as part of it."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus_mask.models import Frame, StreamEnd, StreamHeader
from argus_mask.take import TakeError, line, read_take, render_take

HEADER = StreamHeader(
    face_id="f", session_id="s", fps=30, seed=0, blendshapes=[], audio={"sample_rate": 16000}, driver={"name": "e"}
)
FRAMES = [Frame(seq=i, t_ms=i * 33, weights=[]) for i in range(2)]
END = StreamEnd(frames=2, duration_ms=66)


@pytest.mark.parametrize("extra", [line(Frame(seq=2, t_ms=66, weights=[])), line(END)])
def test_a_message_after_the_end_is_refused(tmp_path: Path, extra: str) -> None:
    path = tmp_path / "t.ndjson"
    path.write_text(render_take(HEADER, FRAMES, END) + extra + "\n", encoding="utf-8")
    with pytest.raises(TakeError, match=r":5: unexpected"):
        read_take(path)


def test_a_whole_take_reads_back(tmp_path: Path) -> None:
    path = tmp_path / "t.ndjson"
    path.write_text(render_take(HEADER, FRAMES, END), encoding="utf-8")
    assert read_take(path) == (HEADER, FRAMES, END)
