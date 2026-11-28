"""What the rest, turns and speech jobs share (``editor/scripts.py``): the script they run ends whole, and so does the
server they run it against."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from argus_mask.config import RestSettings, SpeechSettings, TurnsSettings
from argus_mask.editor.guided import GuidedStore
from argus_mask.editor.rest_fit import RestError, RestFits
from argus_mask.editor.speech import SpeechError, SpeechJobs
from argus_mask.editor.turns import TurnsError, TurnsJobs

# uv, as a stand-in: it starts a child of its own (the browser's part) and waits, as uv does for the script.
UV = """#!/usr/bin/env bash
sleep 60 &
echo $! > "$FAKE/child"
echo $$ > "$FAKE/pid"
wait
"""


def _alive(pid: int) -> bool:
    """Running: not gone, and not a zombie."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z"
    except OSError:
        return False


def _written(path: Path) -> bool:
    """The stand-in has written its whole number here (it ends each with a newline)."""
    return path.is_file() and path.read_text().endswith("\n")


def _run(kind: str, tmp_path: Path, timeout_s: float):  # noqa: ANN202 - a coroutine, and the error a timeout gives
    files = {n: tmp_path / n for n in ("script.py", "talk.json", "model.task", "portrait.png", "out.json", "uv")}
    for path in files.values():
        path.write_text("")
    files["uv"].write_text(UV)
    files["uv"].chmod(0o755)
    common = {"script": files["script.py"], "uv": str(files["uv"]), "timeout_s": timeout_s}
    store, job, url, out = GuidedStore(tmp_path), SimpleNamespace(id="j", candidate="c1"), "http://x", files["out.json"]
    if kind == "rest":
        jobs = RestFits(store, RestSettings(**common, landmarker=files["model.task"]))
        return jobs._script(job, url, files["portrait.png"], out), RestError
    if kind == "turns":
        return TurnsJobs(store, TurnsSettings(**common))._script(job, url, out), TurnsError
    settings = SpeechSettings(**common, take=files["talk.json"], landmarker=files["model.task"])
    return SpeechJobs(store, settings)._script(job, url, "face", out), SpeechError


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.parametrize("ended", ["stopped", "timed out"])
@pytest.mark.parametrize("kind", ["rest", "turns", "speech"])
async def test_a_script_ends_with_everything_it_started(
    kind: str, ended: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAKE", str(tmp_path))
    coro, error = _run(kind, tmp_path, 0.5 if ended == "timed out" else 30)
    try:
        if ended == "timed out":
            with pytest.raises(error, match="took more than"):
                await asyncio.wait_for(coro, 10)  # it does not wait for what it left running
        else:
            task = asyncio.create_task(coro)
            deadline = time.monotonic() + 10
            while not all(_written(tmp_path / n) for n in ("child", "pid")) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            task.cancel()
            done, _ = await asyncio.wait({task}, timeout=10)
            assert done and task.cancelled()
        pids = [int((tmp_path / n).read_text()) for n in ("child", "pid")]
        for _ in range(100):  # a killed child is init's to reap
            if not any(map(_alive, pids)):
                break
            await asyncio.sleep(0.02)
        assert not any(map(_alive, pids))
    finally:
        # Whatever it started, even if the test stopped before reading both. Only what still runs: a pid found gone
        # may be another process's by now.
        for name in ("child", "pid"):
            if _written(tmp_path / name) and _alive(pid := int((tmp_path / name).read_text())):
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, 9)


def test_a_server_that_does_not_come_up_in_time_is_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argus_mask.editor import scripts  # noqa: PLC0415

    monkeypatch.setattr(scripts, "START_S", 0.0)
    server = scripts.HeadServer(tmp_path)
    assert server.start() is False
    assert server.server.should_exit
    server.thread.join(10)
    assert not server.thread.is_alive()


def test_a_server_stopped_before_it_started_never_serves(tmp_path: Path) -> None:
    """A job cancelled while its server's start waits for a thread: the start that comes later does nothing."""
    from argus_mask.editor.scripts import HeadServer  # noqa: PLC0415

    server = HeadServer(tmp_path)
    server.stop()
    assert server.start() is False and not server.thread.is_alive()


def test_a_server_serves_and_stops(tmp_path: Path) -> None:
    from urllib.request import urlopen  # noqa: PLC0415

    from argus_mask.editor.scripts import HeadServer  # noqa: PLC0415

    server = HeadServer(tmp_path)
    try:
        assert server.start() is True
        with urlopen(f"{server.url}/health", timeout=5) as reply:  # noqa: S310
            assert reply.status == 200
    finally:
        server.stop()
    assert not server.thread.is_alive()
