"""Make the hosted demo's clips (#118): demo/clips.json in, demo/clips/<id>/ out.

For each clip, each sentence is spoken through the TTS service exactly as ``argus-mask say`` would (``speech.py``'s
tidying, ``client.synthesise``), with ``say``'s pause between sentences, which gives the captions their times. The
whole utterance becomes a take through the driver (``performance.perform``, as ``argus-mask drive`` does), and the
audio is encoded to AAC for the page. ``clip.json`` records the text, the times and what made it.

    uv run --no-sync python demo/make_clips.py --a2f-url http://127.0.0.1:8110 [--only hello]

Needs the TTS service, the Audio2Face server (or ``--driver energy`` for a quick local try, which the page labels)
and ffmpeg. Run it again after a driver or engine change, and commit demo/clips/.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from argus_mask import __version__, client
from argus_mask.audio import PcmDecoder
from argus_mask.drivers import build_driver
from argus_mask.expressions import BUILTIN, DEFAULT_SET
from argus_mask.performance import perform
from argus_mask.take import render_take

HERE = Path(__file__).resolve().parent


async def speak(tts_url: str, voice: str, sentences: list[str]) -> tuple[np.ndarray, int, list[dict]]:
    """The sentences as one mono float32 utterance, its rate, and each sentence's span (ms) in it."""
    pieces: list[np.ndarray] = []
    captions: list[dict] = []
    rate = 0
    at = 0
    for i, text in enumerate(sentences):
        pcm, fmt, _ = await client.synthesise(tts_url, client.voiced({"text": text}, voice, tts_url))
        rate = rate or fmt.sample_rate
        if fmt.sample_rate != rate:
            raise SystemExit(f"the TTS service changed its rate mid-clip ({fmt.sample_rate} after {rate})")
        mono = PcmDecoder(fmt).feed(pcm)
        captions.append({"start_ms": at * 1000 // rate, "end_ms": (at + mono.size) * 1000 // rate, "text": text})
        pieces.append(mono)
        at += mono.size
        if i + 1 < len(sentences):
            gap = np.zeros(client.SAY_PAUSE_MS * rate // 1000, dtype=np.float32)
            pieces.append(gap)
            at += gap.size
    return np.concatenate(pieces), rate, captions


def made_by() -> str:
    """The code that made the clips: git's description of the checkout (an editable install's version file can be
    stale), else the package's version. Long, so it always names the commit, and marked ``-dirty`` if the checkout
    had uncommitted changes: demo/build.py refuses such a clip, and one whose engine has changed since its commit."""
    try:
        return subprocess.run(  # noqa: S603 - fixed arguments
            ["git", "describe", "--tags", "--long", "--always", "--dirty"],  # noqa: S607
            cwd=HERE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return __version__


def write_wav(path: Path, mono: np.ndarray, rate: int) -> None:
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)
    path.write_bytes(buffer.getvalue())


def encode(wav: Path, out: Path) -> None:
    """AAC in an .m4a: every browser plays it. Bit-exact flags so a re-run changes nothing it need not."""
    ffmpeg = shutil.which("ffmpeg") or sys.exit("needs ffmpeg on PATH")
    subprocess.run(  # noqa: S603 - fixed arguments
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(wav),
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-fflags",
            "+bitexact",
            "-map_metadata",
            "-1",
            "-flags:a",
            "+bitexact",
            str(out),
        ],
        check=True,
    )


async def make(spec: dict, clip: dict, driver_name: str, a2f_url: str, made: str) -> None:
    out = HERE / "clips" / clip["id"]
    out.mkdir(parents=True, exist_ok=True)
    mono, rate, captions = await speak(spec["tts_url"], clip["voice"], clip["sentences"])
    driver = build_driver(driver_name, a2f_url=a2f_url)
    header, frames, end = await perform(
        mono, rate, driver, fps=spec["fps"], seed=clip["seed"], face_id=clip["face"], session_id=clip["id"]
    )
    (out / "take.ndjson").write_text(render_take(header, frames, end), encoding="utf-8")
    wav = out / "audio.wav"
    write_wav(wav, mono, rate)
    encode(wav, out / "audio.m4a")
    wav.unlink()
    meta = {
        "id": clip["id"],
        "title": clip["title"],
        "face": clip["face"],
        "framing": clip["framing"],
        "scene": clip["scene"],
        "duration_ms": end.duration_ms,
        "frames": end.frames,
        "captions": captions,
        "rest": {
            "name": DEFAULT_SET,
            **BUILTIN[DEFAULT_SET].model_dump(exclude={"name"}),
        },  # what a screen holds at rest
        "made_with": {
            "argus_mask": made,
            "driver": header.driver.model_dump(),
            "fps": header.fps,
            "seed": header.seed,
            "voice": clip["voice"].removeprefix("speaker:"),
            "voice_licence": spec["voice_licence"],
        },
    }
    (out / "clip.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    cues = sum(1 for f in frames if f.cue)
    print(f"{clip['id']}: {end.duration_ms / 1000:.1f} s, {end.frames} frames, {cues} cues, {header.driver.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--driver", default="a2f", help="a2f (the demo's) or energy (a quick local try)")
    parser.add_argument("--a2f-url", default="http://127.0.0.1:8110")
    parser.add_argument("--only", action="append", help="make only this clip (repeatable)")
    args = parser.parse_args()
    spec = json.loads((HERE / "clips.json").read_text(encoding="utf-8"))
    # The checkout as it is before any clip is written: a clip's own files would make it read as -dirty.
    made = made_by()
    if made.endswith("-dirty"):
        sys.exit(
            f"the checkout has uncommitted changes ({made}): commit them first, or demo/build.py refuses the clips"
        )
    for clip in spec["clips"]:
        if not args.only or clip["id"] in args.only:
            asyncio.run(make(spec, clip, args.driver, args.a2f_url, made))
    index = [c["id"] for c in spec["clips"] if (HERE / "clips" / c["id"] / "clip.json").is_file()]
    (HERE / "clips" / "index.json").write_text(json.dumps(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
