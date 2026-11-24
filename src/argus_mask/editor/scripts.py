"""What the rest, turns and speech jobs share: a short-lived mask server for one head, and a ``lam/`` script's run.

Both end with the job that started them, however it ends. A script runs under ``uv``, which starts Python, which
starts a browser: killing ``uv`` alone would leave the others running, so the whole process group is killed.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import threading
import time
from pathlib import Path

import uvicorn

from argus_mask.config import Settings
from argus_mask.server.app import create_app

START_S = 15.0
JOIN_S = 5.0


def browsers_env() -> dict[str, str]:
    """This process's environment, with where Playwright keeps its browsers."""
    found = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", str(Path.home() / ".cache" / "ms-playwright"))
    return {**os.environ, "PLAYWRIGHT_BROWSERS_PATH": found}


class HeadServer:
    """A mask server on a free loopback port, serving only *assets*. Made before it is started, so whoever made it can
    always stop it, even where the start failed or its wait was cancelled."""

    def __init__(self, assets: Path) -> None:
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))  # kept bound: nothing can take the port before the server has it
        self.url = f"http://127.0.0.1:{self._sock.getsockname()[1]}"
        app = create_app(Settings(assets_dir=assets, driver="energy"))
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self._sock]}, daemon=True)

    def start(self) -> bool:
        """Serve; whether it came up in time. Blocks: run it off the event loop. Stopped already, it never starts."""
        if self.server.should_exit:
            return False
        self.thread.start()
        deadline = time.monotonic() + START_S
        while not self.server.started and self.thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self.server.started:
            self.stop()
            return False
        return True

    def stop(self) -> None:
        """Stop serving, whatever state it is in. A server still starting exits once it has started."""
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(JOIN_S)
        self._sock.close()


async def run_script(argv: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str]) -> tuple[int, str]:
    """Run *argv* to its end: its status, and what it said. ``TimeoutError`` past *timeout_s*. Timed out or cancelled,
    it is killed with everything it started, and reaped."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        env=env,
        cwd=str(cwd),
        start_new_session=True,  # its own process group: killed whole
    )
    try:
        said, _ = await asyncio.wait_for(proc.communicate(), timeout_s)
    except BaseException:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):  # what left the group may hold the pipe open: not waited for
            await asyncio.wait_for(proc.wait(), JOIN_S)
        raise
    return proc.returncode or 0, (said or b"").decode(errors="replace")
