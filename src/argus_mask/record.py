"""Record a session as a video clip, offline and frame-exact.

Screen capture of the viewer competes with the viewer for the machine and gets the
audio wherever the operating system puts it. A take is deterministic, so a clip
need not be captured at all: ``record`` listens on a face's viewer socket, keeps
the next session's header, frames and audio, then renders every frame in a
headless browser, one screenshot per frame, and muxes them with the audio. The
result is exactly what the stream said, whatever the recording machine was doing.

Needs the ``record`` extra (playwright and imageio-ffmpeg), and a browser for
playwright: ``playwright install chromium-headless-shell`` once.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import struct
import subprocess
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from argus_mask.client import ClientError, view_url
from argus_mask.models import Frame, StreamEnd, StreamHeader
from argus_mask.take import render_take

DEFAULT_WIDTH, DEFAULT_HEIGHT = 1280, 720
QUALITIES = ("low", "medium", "high")  # the viewer's look profiles (look.js)
SCENES = ("ftl",)  # the viewer's built-in backdrops (backdrop.js: SCENES); a test holds the two together
# Chromium's rendering backends, tried in order: the host's GPU over EGL, then SwiftShader in software.
BACKENDS = (
    ("gpu", ["--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"]),
    ("software", ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]),
)
WEBGL_RENDERER = """(() => {
  const gl = document.createElement('canvas').getContext('webgl2');
  const info = gl && gl.getExtension('WEBGL_debug_renderer_info');
  return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'unknown renderer';
})()"""


class RecordError(Exception):
    """A clip that cannot be made."""


@dataclass
class Recording:
    header: StreamHeader
    frames: list[Frame] = field(default_factory=list)
    end: StreamEnd | None = None
    audio: list[tuple[float, np.ndarray]] = field(default_factory=list)  # (start_ms, mono f32)

    @property
    def duration_ms(self) -> float:
        rate = self.header.audio.sample_rate
        sound = max((start + x.size * 1000.0 / rate for start, x in self.audio), default=0.0)
        return max(sound, (len(self.frames) * 1000.0 / self.header.fps) if self.frames else 0.0)

    def pcm(self) -> np.ndarray:
        """The audio laid out on the timeline from 0, as mono float32."""
        rate = self.header.audio.sample_rate
        out = np.zeros(int(round(self.duration_ms * rate / 1000)), dtype=np.float32)
        for start, x in self.audio:
            at = int(round(start * rate / 1000))
            out[at : at + x.size] = x[: max(0, out.size - at)]
        return out


async def capture(base: str, face_id: str, *, on_wait: Callable[[], None] | None = None) -> Recording:
    """Wait on *face_id*'s viewer socket for the next whole session and keep it.

    A session already under way when the socket opens is skipped: its first audio is gone.
    """
    try:
        import websockets  # noqa: PLC0415
    except ImportError as exc:
        raise RecordError("record needs websockets: install argus-mask[cli]") from exc
    recording: Recording | None = None
    try:
        async with websockets.connect(view_url(base, face_id), max_size=2**22, proxy=None) as ws:
            # Only now: a session that starts before the socket is open is skipped, so whoever starts one
            # on this signal (the editor's Broadcast page does) must not be told any earlier.
            if on_wait:
                on_wait()
            async for raw in ws:
                if isinstance(raw, bytes):
                    if recording is not None:
                        (start_ms,) = struct.unpack_from("<d", raw, 0)
                        recording.audio.append((start_ms, np.frombuffer(raw[8:], dtype="<f4")))
                    continue
                message = json.loads(raw)
                kind = message.get("type")
                if kind == "header":
                    recording = Recording(StreamHeader.model_validate(message))
                elif kind == "frame" and recording is not None:
                    frame = Frame.model_validate(message)
                    if frame.seq != len(recording.frames):
                        recording = None  # joined mid-stream (or lost a frame): wait for the next session
                        continue
                    recording.frames.append(frame)
                elif kind == "end" and recording is not None:
                    if message.get("error"):
                        raise RecordError(f"the session failed: {message['error']}")
                    recording.end = StreamEnd.model_validate(message)
                    return recording
                elif kind == "error":
                    raise RecordError(str(message.get("detail")))
    except ClientError as exc:
        raise RecordError(str(exc)) from exc
    except websockets.ConnectionClosed as exc:
        raise RecordError(f"the server closed the viewer socket: {exc.reason or exc.code}") from exc
    except (OSError, websockets.InvalidURI, websockets.InvalidHandshake) as exc:
        raise RecordError(f"cannot reach {base}: {exc}") from exc
    raise RecordError("the server closed the viewer socket before a session ended")


def write_wav(path: Path, recording: Recording) -> None:
    pcm = np.clip(recording.pcm(), -1.0, 1.0)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(recording.header.audio.sample_rate)
        out.writeframes((pcm * 32767).astype("<i2").tobytes())


def write_take(path: Path, recording: Recording) -> None:
    end = recording.end or StreamEnd(frames=len(recording.frames), duration_ms=round(recording.duration_ms))
    path.write_text(render_take(recording.header, recording.frames, end), encoding="utf-8")


async def render_frames(
    page_url: str,
    recording: Recording,
    frames_dir: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    on_progress: Callable[[int, int], None] | None = None,
    on_renderer: Callable[[str], None] | None = None,
) -> None:
    """One PNG per frame, named by sequence number, from the viewer in record mode."""
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415
    except ImportError as exc:
        raise RecordError(
            "record needs playwright: install argus-mask[record] and run `playwright install chromium-headless-shell`"
        ) from exc
    frames_dir.mkdir(parents=True, exist_ok=True)
    names = list(recording.header.blendshapes)
    async with async_playwright() as p:
        browser = page = None
        failures = []
        # The machine's GPU through EGL is about ten times faster than SwiftShader on a head this size.
        # A host with no usable GPU fails to load the face on the first try and falls through.
        for backend, args in BACKENDS:
            try:
                browser = await p.chromium.launch(args=args)
                page = await browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
                await page.goto(page_url)
                await page.wait_for_function(
                    "window.__mask && (window.__mask.ready || window.__mask.error)", timeout=60_000
                )
                error = await page.evaluate("window.__mask.error")
                if error:
                    raise RecordError(f"the viewer could not load the face: {error}")
                renderer = await page.evaluate(WEBGL_RENDERER)
                look = await page.evaluate("window.__mask.look && window.__mask.look.quality")
                if on_renderer:
                    on_renderer(f"{backend}: {renderer}" + (f", look {look}" if look else ""))
                break
            except Exception as exc:  # playwright's own error types vary by version
                failures.append(f"{backend}: {str(exc).splitlines()[0]}")
                if browser is not None:
                    await browser.close()
                browser = page = None
        if page is None or browser is None:
            raise RecordError("no browser could render the face: " + "; ".join(failures))
        try:
            total = len(recording.frames)
            for i, frame in enumerate(recording.frames):
                await page.evaluate(
                    "([w, h, n, t]) => window.__mask.pose(w, h, n, t)",
                    [list(frame.weights), frame.head.model_dump(), names if i == 0 else None, frame.t_ms],
                )
                await page.screenshot(path=str(frames_dir / f"{frame.seq:06d}.png"), type="png")
                if on_progress:
                    on_progress(i + 1, total)
        finally:
            await browser.close()


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: PLC0415

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RecordError("record needs ffmpeg on PATH, or the record extra's imageio-ffmpeg") from exc


def mux(frames_dir: Path, wav: Path, out: Path, fps: int) -> None:
    """Frames and audio into one H.264/AAC MP4 (or VP9/Opus WebM, by the output's suffix)."""
    webm = out.suffix.lower() == ".webm"
    command = [
        find_ffmpeg(),
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-i",
        str(wav),
        *(
            ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30"]
            if webm
            else ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
        ),
        "-pix_fmt",
        "yuv420p",
        *(["-c:a", "libopus", "-b:a", "128k"] if webm else ["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]),
        "-shortest",
        str(out),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RecordError(f"ffmpeg failed: {result.stderr.strip() or result.returncode}")


async def record(
    base: str,
    face_id: str,
    out: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    display: str = "flat",
    bg: str = "",
    scene: str = "",
    quality: str = "high",
    keep: bool = False,
    on_wait: Callable[[], None] | None = None,
    on_captured: Callable[[Recording], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_renderer: Callable[[str], None] | None = None,
) -> Recording:
    """Capture the next session on *face_id* and write *out*, with the take and WAV beside it."""
    if width % 2 or height % 2:
        raise RecordError(f"{width}x{height}: the clip's width and height must be even (4:2:0 video)")
    if quality not in QUALITIES:  # before the wait: a typo should not cost the session
        raise RecordError(f"quality {quality!r}: choose from {', '.join(QUALITIES)}")
    if scene and scene not in SCENES:
        raise RecordError(f"scene {scene!r}: choose from {', '.join(SCENES)}")
    recording = await capture(base, face_id, on_wait=on_wait)
    if not recording.frames:
        raise RecordError("the session had no frames")
    if on_captured:
        on_captured(recording)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    wav, take = Path(f"{stem}.wav"), Path(f"{stem}.take.ndjson")
    # Off the event loop: the editor records inside its own server, and must keep answering meanwhile.
    await asyncio.to_thread(write_wav, wav, recording)
    await asyncio.to_thread(write_take, take, recording)
    frames_dir = Path(f"{stem}.frames")
    shutil.rmtree(frames_dir, ignore_errors=True)  # frames kept from an earlier, longer clip would join this one
    # The viewer's look profile (look.js). Offline, there is time for all of it: high by default.
    params = {"face": face_id, "record": "1", "hud": "0", "display": display, "quality": quality}
    params |= {"bg": bg} if bg else {}
    params |= {"scene": scene} if scene else {}
    try:
        await render_frames(
            f"{base.rstrip('/')}/?{urlencode(params)}",
            recording,
            frames_dir,
            width=width,
            height=height,
            on_progress=on_progress,
            on_renderer=on_renderer,
        )
        await asyncio.to_thread(mux, frames_dir, wav, out, recording.header.fps)
    finally:  # a failed or stopped render must not leave a directory of PNGs behind either
        if not keep:
            shutil.rmtree(frames_dir, ignore_errors=True)
    return recording
