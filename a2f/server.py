"""The Audio2Face server: audio in, ARKit blendshape weights out, over one WebSocket.

It is the other end of argus-mask's ``a2f`` driver (``src/argus_mask/drivers/a2f.py``):

* ``GET /health`` -> 200 ``{"ready": bool, "model": str, "reason": str?}``
* ``WS /drive``, one stream at a time:

  - client -> ``{"fps": 30, "sample_rate": 16000}``
  - server -> ``{"blendshapes": [name, ...]}``: the order of every frame's weights
  - client -> 16 kHz mono ``f32le`` PCM (binary), as it arrives
  - server -> ``{"i": n, "weights": [...]}``, one per finished frame, in order
  - client -> ``{"end": true}``; the server sends the remaining frames, then ``{"end": true}``
  - any failure -> ``{"error": "..."}`` and the socket closes

This process owns the network and the first-start work (download the model, build the
TensorRT engine). The model itself runs in ``a2f-worker``, a child process that keeps it
loaded and is reset between streams. Nothing here imports CUDA.

Config (environment): ``A2F_MODEL`` (mark | claire | james), ``A2F_GPU_SOLVER`` (0 | 1),
``A2F_PORT`` (8110), ``A2F_HOST`` (0.0.0.0), ``A2F_DATA`` (/data), ``A2F_FPS`` (the frame
rate the worker is warmed for, 30).

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import websockets
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

SAMPLE_RATE = 16000
MAX_FPS = 240
HELLO_TIMEOUT_S = 10.0
WORKER_START_TIMEOUT_S = 120.0
FLUSH_TIMEOUT_S = 60.0
RETRY_S = 30.0
MAX_MESSAGE = 2**22  # 4 MiB: about a minute of audio in one chunk

# The regression models, one per character. Revisions are pinned so a changed upstream repo
# cannot change what a deployment runs. None of these is gated; Audio2Emotion (which is) is
# never downloaded.
# james, not mark: on real speech mark holds the jaw to one side (see README, "Which model").
DEFAULT_MODEL = "james"
MODELS = {
    "mark": ("nvidia/Audio2Face-3D-v2.3-Mark", "5451728e07378df93b04523279e134a9993ae71b"),
    "claire": ("nvidia/Audio2Face-3D-v2.3.1-Claire", "a46eafb067adfdd7cc8f5c7941586b309004561c"),
    "james": ("nvidia/Audio2Face-3D-v2.3.1-James", "327d000d9f76e370014a9b7467b23ea36846b680"),
}
MODEL_FILES = (
    "network_info.json",
    "trt_info.json",
    "model_config.json",
    "bs_skin_config.json",
    "bs_tongue_config.json",
    "bs_tongue.npz",
    "implicit_emo_db.npz",
    "bs_skin.npz",
    "model_data.npz",
    "network.onnx",
)


def log(message: str) -> None:
    print(f"a2f-server: {message}", file=sys.stderr, flush=True)


def env_flag(name: str) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("", "0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    raise SystemExit(f"{name}={raw!r} is not a flag: use 0 or 1")


def env_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a whole number") from None
    if not low <= value <= high:
        raise SystemExit(f"{name}={value} is outside {low}..{high}")
    return value


class Config:
    def __init__(self) -> None:
        self.model = os.environ.get("A2F_MODEL", DEFAULT_MODEL).strip().lower() or DEFAULT_MODEL
        if self.model not in MODELS:
            raise SystemExit(f"A2F_MODEL={self.model!r}: choose one of {', '.join(MODELS)}")
        self.repo, self.revision = MODELS[self.model]
        self.gpu_solver = env_flag("A2F_GPU_SOLVER")
        self.port = env_int("A2F_PORT", 8110, 1, 65535)
        self.host = os.environ.get("A2F_HOST", "0.0.0.0").strip() or "0.0.0.0"  # noqa: S104 - a container
        self.fps = env_int("A2F_FPS", 30, 1, MAX_FPS)
        self.data = Path(os.environ.get("A2F_DATA", "/data"))
        self.worker = os.environ.get("A2F_WORKER", "a2f-worker")
        self.trtexec = os.environ.get("A2F_TRTEXEC", "trtexec")
        self.trt_version = os.environ.get("A2F_TRT_VERSION", "unknown")


# ---- first start: the model and its engine ------------------------------------------------


def _download(url: str, dest: Path, progress) -> None:  # noqa: ANN001
    """Fetch *url* to *dest* through a ``.part`` file, so a killed download is never mistaken for a model."""
    part = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "argus-mask-a2f"})  # noqa: S310 - https, fixed host
    with urllib.request.urlopen(request, timeout=60) as reply, part.open("wb") as out:  # noqa: S310
        total = int(reply.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = reply.read(1 << 20)
            if not block:
                break
            out.write(block)
            done += len(block)
            progress(done, total)
    if total and done != total:
        part.unlink(missing_ok=True)
        raise OSError(f"{dest.name}: got {done} of {total} bytes")
    part.replace(dest)


def _gpu_name() -> str:
    try:
        out = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"no NVIDIA GPU is visible (run the container with --gpus all): {exc}") from exc
    name = out.strip().splitlines()[0].strip() if out.strip() else ""
    if not name:
        raise RuntimeError("no NVIDIA GPU is visible (run the container with --gpus all)")
    return name


def prepare(config: Config, phase) -> Path:  # noqa: ANN001
    """Download the model and build its engine if either is missing. Returns the model JSON the worker loads.

    Blocking; run in a thread. *phase* is called with a sentence describing what is happening.
    """
    model_dir = config.data / "models" / f"{config.repo.split('/')[-1]}@{config.revision[:12]}"
    model_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    for number, name in enumerate(missing, 1):
        label = f"downloading {config.repo} ({number} of {len(missing)}: {name})"
        phase(label)
        log(label)

        def progress(done: int, total: int, label: str = label) -> None:
            phase(f"{label}, {done >> 20} of {total >> 20} MB" if total else label)

        _download(f"https://huggingface.co/{config.repo}/resolve/{config.revision}/{name}", model_dir / name, progress)

    # A TensorRT engine only runs on the GPU model and TensorRT version it was built with,
    # so both are in its file name: a new card or image builds a new engine beside the old one.
    gpu = _gpu_name()
    key = re.sub(r"[^a-z0-9.]+", "-", f"b1-trt{config.trt_version}-{gpu}".lower()).strip("-")
    engine = model_dir / f"network-{key}.trt"
    if not engine.is_file():
        info = json.loads((model_dir / "trt_info.json").read_text())
        # The SDK's defaults build for up to 128 simultaneous tracks. This server runs one,
        # and a batch-1 engine holds 0.6 GB of VRAM where the default holds 1.4 GB.
        sizes = {**info.get("defaults", {}), "MAX_BATCH_SIZE": 1, "OPT_BATCH_SIZE": 1}
        args = [arg.format(**sizes) for group in info["trt_build_param"].values() for arg in group]
        part = engine.with_name(engine.name + ".part")
        log_path = model_dir / f"trtexec-{key}.log"
        label = f"building the TensorRT engine for {gpu} (about a minute)"
        phase(label)
        log(label)
        started = time.monotonic()
        with log_path.open("w") as out:
            result = subprocess.run(  # noqa: S603
                [config.trtexec, f"--onnx={model_dir / 'network.onnx'}", f"--saveEngine={part}", "--device=0", *args],
                stdout=out,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not part.is_file():
            part.unlink(missing_ok=True)
            lines = log_path.read_text(errors="replace").splitlines()
            first = next((line for line in lines if "[E]" in line), lines[-1] if lines else "no output")
            raise RuntimeError(f"trtexec failed (exit {result.returncode}): {first[:300]} (full log: {log_path})")
        part.replace(engine)
        log(f"engine built in {time.monotonic() - started:.0f} s: {engine}")

    model_json = model_dir / f"a2f-{key}.json"
    model_json.write_text(
        json.dumps(
            {
                "networkInfoPath": "network_info.json",
                "networkPath": engine.name,
                "emotionDatabasePath": "implicit_emo_db.npz",
                "modelConfigPath": "model_config.json",
                "modelDataPath": "model_data.npz",
                "blendshapePaths": {
                    "skin": {"config": "bs_skin_config.json", "data": "bs_skin.npz"},
                    "tongue": {"config": "bs_tongue_config.json", "data": "bs_tongue.npz"},
                },
            },
            indent=2,
        )
    )
    return model_json


# ---- the worker process -------------------------------------------------------------------


class WorkerDied(RuntimeError):
    pass


class Worker:
    """One ``a2f-worker`` child: a model held loaded, driven over its stdin and stdout."""

    def __init__(self, process: asyncio.subprocess.Process, fps: int) -> None:
        self.process = process
        self.fps = fps
        self.names: list[str] = []

    @classmethod
    async def start(cls, config: Config, model_json: Path, fps: int) -> Worker:
        args = [str(model_json), "--fps", str(fps)] + (["--gpu-solver"] if config.gpu_solver else [])
        process = await asyncio.create_subprocess_exec(
            config.worker, *args, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        worker = cls(process, fps)
        try:
            kind, payload = await asyncio.wait_for(worker.read(), WORKER_START_TIMEOUT_S)
            if kind != b"H":
                raise WorkerDied(f"the worker opened with {kind!r}, not its hello")
            worker.names = [str(name) for name in json.loads(payload)["names"]]
        except BaseException:
            await worker.stop()
            raise
        return worker

    async def read(self) -> tuple[bytes, bytes]:
        """The worker's next message. Raises :class:`WorkerDied` on its error message or exit."""
        assert self.process.stdout is not None  # noqa: S101
        try:
            header = await self.process.stdout.readexactly(5)
            payload = await self.process.stdout.readexactly(struct.unpack("<I", header[1:])[0])
        except asyncio.IncompleteReadError as exc:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.process.wait(), 2)
            raise WorkerDied(f"the worker exited (code {self.process.returncode})") from exc
        if header[:1] == b"X":
            raise WorkerDied(payload.decode(errors="replace"))
        return header[:1], payload

    async def write(self, kind: bytes, payload: bytes = b"") -> None:
        assert self.process.stdin is not None  # noqa: S101
        try:
            self.process.stdin.write(kind + struct.pack("<I", len(payload)) + payload)
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise WorkerDied("the worker is gone") from exc

    async def stop(self) -> None:
        stdin = self.process.stdin
        if stdin is not None:
            with contextlib.suppress(OSError):
                stdin.close()
            # Retrieve the pipe's own error (a dead worker breaks it) so asyncio does not log it.
            with contextlib.suppress(OSError):
                await stdin.wait_closed()
        if self.process.returncode is not None:
            return
        try:
            await asyncio.wait_for(self.process.wait(), 5)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()


# ---- the server ---------------------------------------------------------------------------


class Server:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.phase = "starting"
        self.model_json: Path | None = None
        self.worker: Worker | None = None
        self.busy = asyncio.Lock()
        self.streams = 0
        self._recovery: asyncio.Task | None = None
        self._watchers: set[asyncio.Task] = set()
        self._closing = False

    @property
    def ready(self) -> bool:
        return self.worker is not None and self.worker.process.returncode is None

    def health(self) -> dict:
        body: dict = {
            "ready": self.ready,
            "model": self.config.repo.split("/")[-1],
            "fps": self.worker.fps if self.worker else self.config.fps,
            "gpu_solver": self.config.gpu_solver,
            "busy": self.busy.locked(),
            "streams": self.streams,
        }
        if not body["ready"]:
            body["reason"] = self.phase
        return body

    async def bring_up(self) -> None:
        """Download, build and load, retrying until it works. ``/health`` says where it is."""
        loop = asyncio.get_running_loop()

        def phase(text: str) -> None:
            loop.call_soon_threadsafe(setattr, self, "phase", text)

        while True:
            try:
                if self.model_json is None:
                    self.model_json = await asyncio.to_thread(prepare, self.config, phase)
                self.phase = "loading the model"
                worker = await Worker.start(self.config, self.model_json, self.config.fps)
            except (OSError, RuntimeError, ValueError, KeyError, TimeoutError, urllib.error.URLError) as exc:
                self.phase = f"failed, retrying in {RETRY_S:g} s: {exc}"
                log(self.phase)
                await asyncio.sleep(RETRY_S)
                continue
            self._adopt(worker)
            log(f"ready: {self.config.repo} at {worker.fps} fps, {len(worker.names)} weights")
            return

    def _adopt(self, worker: Worker) -> None:
        """Make *worker* the loaded model, and notice if it exits while nothing is using it."""
        self.worker = worker

        async def watch() -> None:
            code = await worker.process.wait()
            if self.worker is worker and not self._closing:
                self.worker = None
                self._recover(f"the worker exited on its own (code {code})")

        task = asyncio.get_running_loop().create_task(watch())
        self._watchers.add(task)
        task.add_done_callback(self._watchers.discard)

    async def worker_for(self, fps: int) -> Worker:
        """The loaded worker, reloaded first if the stream wants another frame rate."""
        assert self.model_json is not None and self.worker is not None  # noqa: S101
        if self.worker.fps == fps:
            return self.worker
        old, self.worker = self.worker, None
        self.phase = f"loading the model at {fps} fps"
        log(f"reloading the worker for {fps} fps (was {old.fps})")
        await old.stop()
        try:
            worker = await Worker.start(self.config, self.model_json, fps)
        except (OSError, RuntimeError, ValueError, KeyError, TimeoutError) as exc:
            self._recover(f"could not load the model at {fps} fps: {exc}")
            raise WorkerDied(f"could not load the model at {fps} fps: {exc}") from exc
        self._adopt(worker)
        return worker

    def _recover(self, why: str) -> None:
        """Bring a fresh worker up in the background; ``/health`` is not ready until it is."""
        if self._closing or (self._recovery is not None and not self._recovery.done()):
            return
        log(f"restarting the model: {why}")
        self.phase = "restarting the model after a failure"
        self._recovery = asyncio.get_running_loop().create_task(self.bring_up())

    def process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        def reply(status: int, text: str, body: dict) -> Response:
            raw = json.dumps(body).encode()
            headers = Headers({"Content-Type": "application/json", "Content-Length": str(len(raw))})
            return Response(status, text, headers, raw)

        path = request.path.split("?", 1)[0]
        if path == "/health":
            return reply(200, "OK", self.health())
        if path != "/drive":
            return reply(404, "Not Found", {"error": "not found"})
        # A browser always names the page that opened a socket; the mask server never does.
        # Refusing every Origin keeps a web page from driving the GPU through a visitor's browser.
        if request.headers.get("Origin"):
            return reply(403, "Forbidden", {"error": "browsers may not open this socket"})
        return None

    async def drive(self, ws: ServerConnection) -> None:
        if self.busy.locked():
            # Take the client's opening message first, so the refusal arrives as the answer to
            # it and not as a socket that closed while the client was still writing.
            with contextlib.suppress(TimeoutError, websockets.ConnectionClosed):
                await asyncio.wait_for(ws.recv(), 2)
            await self._refuse(ws, "busy: this server runs one stream at a time", code=1013)
            return
        async with self.busy:
            try:
                await self._stream(ws)
            except websockets.ConnectionClosed:
                pass
            except WorkerDied as exc:
                log(f"the worker failed: {exc}")
                await self._refuse(ws, f"the model failed: {exc}", code=1011)

    @staticmethod
    async def _refuse(ws: ServerConnection, reason: str, *, code: int = 1008) -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            await ws.send(json.dumps({"error": reason}))
            await ws.close(code, reason[:120])

    async def _stream(self, ws: ServerConnection) -> None:
        try:
            hello = json.loads(await asyncio.wait_for(ws.recv(), HELLO_TIMEOUT_S))
        except (TimeoutError, ValueError, UnicodeDecodeError):
            await self._refuse(ws, 'the first message must be {"fps": n, "sample_rate": 16000}')
            return
        fps = hello.get("fps") if isinstance(hello, dict) else None
        if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= MAX_FPS:
            await self._refuse(ws, f"fps must be a whole number from 1 to {MAX_FPS}")
            return
        if hello.get("sample_rate") != SAMPLE_RATE:
            await self._refuse(ws, f"sample_rate must be {SAMPLE_RATE}: resample before sending")
            return
        if not self.ready:
            await self._refuse(ws, f"not ready: {self.phase}", code=1013)
            return

        worker = await self.worker_for(fps)
        await ws.send(json.dumps({"blendshapes": worker.names}))
        self.streams += 1
        # The client's messages and the worker's frames are watched together, so a worker that dies
        # while the client is quiet (a pause in speech) is reported at once, not on the next write.
        pump = asyncio.create_task(self._pump(worker, ws))
        receive = asyncio.create_task(self._receive(ws, worker))
        try:
            done, _ = await asyncio.wait({pump, receive}, return_when=asyncio.FIRST_COMPLETED)
            if receive in done:
                try:
                    receive.result()
                except ValueError as exc:
                    await self._refuse(ws, str(exc))
            # Otherwise the pump ended first: the worker failed, and the flush below reports it.
        finally:
            if not receive.done():
                receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)
            # Whether the client finished, vanished or sent nonsense, the worker's stream is
            # ended and drained, so the next stream starts from a reset model.
            failure = None
            try:
                if not pump.done():
                    await worker.write(b"E")
                await asyncio.wait_for(pump, FLUSH_TIMEOUT_S)
            except TimeoutError:
                failure = f"the model did not finish within {FLUSH_TIMEOUT_S:g} s"
            except WorkerDied as exc:
                failure = str(exc)
            if failure is not None:
                pump.cancel()
                if self.worker is worker:
                    self.worker = None
                await worker.stop()
                self._recover(failure)
                raise WorkerDied(failure)

    @staticmethod
    async def _receive(ws: ServerConnection, worker: Worker) -> None:
        """Pass the client's audio to the worker until it sends ``{"end": true}`` or goes away."""
        async for message in ws:
            if isinstance(message, bytes):
                if len(message) % 4:
                    raise ValueError("audio must be whole float32 samples")
                if message:
                    await worker.write(b"A", message)
                continue
            body = json.loads(message)
            if isinstance(body, dict) and body.get("end"):
                return
            raise ValueError('a text message must be {"end": true}')

    async def _pump(self, worker: Worker, ws: ServerConnection) -> None:
        """Forward the worker's frames until it reports the stream flushed.

        Keeps reading after the client has gone: the worker must reach its end marker.
        """
        gone = False
        while True:
            kind, payload = await worker.read()
            if kind == b"E":
                if not gone:
                    with contextlib.suppress(websockets.ConnectionClosed):
                        await ws.send(json.dumps({"end": True}))
                return
            if kind != b"F" or gone:
                continue
            index = struct.unpack_from("<I", payload)[0]
            weights = struct.unpack_from(f"<{(len(payload) - 12) // 4}f", payload, 12)
            message = {"i": index, "weights": [round(w, 4) if math.isfinite(w) else 0.0 for w in weights]}
            try:
                await ws.send(json.dumps(message, separators=(",", ":")))
            except websockets.ConnectionClosed:
                gone = True

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        stop = loop.create_future()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: stop.done() or stop.set_result(None))
        bring_up = asyncio.create_task(self.bring_up())
        async with serve(
            self.drive, self.config.host, self.config.port, process_request=self.process_request, max_size=MAX_MESSAGE
        ):
            log(f"listening on {self.config.host}:{self.config.port} (model {self.config.model})")
            await stop
        self._closing = True
        bring_up.cancel()
        if self._recovery is not None:
            self._recovery.cancel()
        if self.worker is not None:
            await self.worker.stop()


def main() -> None:
    config = Config()
    if shutil.which(config.worker) is None:
        raise SystemExit(f"{config.worker} is not on PATH")
    asyncio.run(Server(config).run())


if __name__ == "__main__":
    main()
