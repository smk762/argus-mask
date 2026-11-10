"""Check a running Audio2Face server against the /drive protocol, with real audio.

    python a2f/check.py http://127.0.0.1:8110 speech-16k-mono.wav

Needs ``websockets`` (argus-mask's ``[a2f]`` extra has it) and a 16 kHz mono 16-bit WAV of
speech. It is a live check, not a unit test: it needs the GPU and the model, so it is run by
hand. Exits non-zero if any check fails.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import array
import asyncio
import json
import sys
import time
import urllib.error
import urllib.request
import wave

import websockets

RATE = 16000


def load(path: str) -> bytes:
    with wave.open(path) as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise SystemExit(f"{path}: need 16 kHz mono 16-bit")
        samples = array.array("h")
        samples.frombytes(wav.readframes(wav.getnframes()))
    return array.array("f", (s / 32768.0 for s in samples)).tobytes()


def get(url: str) -> tuple[int, dict]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {}


async def stream(ws_url: str, pcm: bytes, *, fps: int = 30, chunk_ms: int = 40, realtime: bool = False) -> dict:
    """One whole stream. Returns the names, the frames and, per frame, how late it arrived."""
    async with websockets.connect(ws_url, proxy=None, max_size=2**22) as ws:
        await ws.send(json.dumps({"fps": fps, "sample_rate": RATE}))
        hello = json.loads(await ws.recv())
        if "error" in hello:
            return hello
        names = hello["blendshapes"]
        frames: list[list[float]] = []
        lag: list[float] = []
        start = time.monotonic()

        async def reader() -> None:
            async for raw in ws:
                message = json.loads(raw)
                if message.get("error"):
                    raise RuntimeError(f"the server failed mid-stream: {message['error']}")
                if message.get("end"):
                    return
                assert message["i"] == len(frames), f"frame {message['i']} arrived as number {len(frames)}"
                # Frame n shows the face at n / fps seconds into the audio.
                lag.append(time.monotonic() - start - len(frames) / fps)
                frames.append(message["weights"])

        task = asyncio.create_task(reader())
        step = RATE * chunk_ms // 1000 * 4
        for offset in range(0, len(pcm), step):
            if realtime:  # a chunk exists once its last sample has been spoken
                await asyncio.sleep(max(0.0, start + (offset + step) / 4 / RATE - time.monotonic()))
            await ws.send(pcm[offset : offset + step])
        await ws.send(json.dumps({"end": True}))
        await asyncio.wait_for(task, 60)
        return {"names": names, "frames": frames, "lag": lag}


async def main(base: str, wav: str) -> int:
    ws_url = base.replace("http", "ws", 1).rstrip("/") + "/drive"
    pcm = load(wav)
    seconds = len(pcm) / 4 / RATE
    failures = 0

    def check(ok: bool, what: str) -> None:
        nonlocal failures
        failures += not ok
        print(("ok    " if ok else "FAIL  ") + what)

    status, health = get(base.rstrip("/") + "/health")
    check(status == 200 and health.get("ready") is True, f"/health is ready: {health}")
    check(get(base.rstrip("/") + "/nope")[0] == 404, "another path is 404")

    first = await stream(ws_url, pcm)
    names, frames = first["names"], first["frames"]
    expected = -(-len(pcm) // 4 * 30 // RATE)
    check(
        len(names) == 68 and names[0] == "eyeBlinkLeft" and names[52] == "tongueTipUp",
        f"{len(names)} names, skin then tongue",
    )
    check(len(frames) == expected, f"{len(frames)} frames for {seconds:.2f} s at 30 fps (expected {expected})")
    jaw = [f[names.index("jawOpen")] for f in frames]
    check(max(jaw) - min(jaw) > 0.2, f"jawOpen moves: {min(jaw):.3f} to {max(jaw):.3f}")
    check(all(0.0 <= w <= 1.5 for f in frames for w in f), "every weight is a sane number")

    again = await stream(ws_url, pcm, chunk_ms=250)
    check(again["frames"] == frames, "a second stream, chunked differently, gives the same frames")

    # A client that vanishes mid-stream must not leave its audio in the model.
    async with websockets.connect(ws_url, proxy=None) as ws:
        await ws.send(json.dumps({"fps": 30, "sample_rate": RATE}))
        await ws.recv()
        await ws.send(pcm[: len(pcm) // 3 // 4 * 4])
    await asyncio.sleep(0.5)
    after = await stream(ws_url, pcm)
    check(after.get("frames") == frames, "a stream after an abandoned one gives the same frames")

    # One stream at a time.
    async with websockets.connect(ws_url, proxy=None) as holder:
        await holder.send(json.dumps({"fps": 30, "sample_rate": RATE}))
        await holder.recv()
        second = await stream(ws_url, pcm)
        check("busy" in str(second.get("error", "")), f"a second stream is refused: {second.get('error')}")
        check(get(base.rstrip("/") + "/health")[1].get("busy") is True, "/health says busy meanwhile")
        await holder.send(json.dumps({"end": True}))
        async for _ in holder:
            pass

    wrong = await stream(ws_url, pcm[:6400], fps=0)
    check("fps" in str(wrong.get("error", "")), f"fps 0 is refused: {wrong.get('error')}")
    async with websockets.connect(ws_url, proxy=None) as ws:
        await ws.send(json.dumps({"fps": 30, "sample_rate": 24000}))
        reply = json.loads(await ws.recv())
        check("sample_rate" in str(reply.get("error", "")), f"24 kHz is refused: {reply.get('error')}")
    try:
        async with websockets.connect(ws_url, proxy=None, origin="https://example.com"):
            check(False, "a socket with an Origin header was accepted")
    except websockets.InvalidStatus as exc:
        check(exc.response.status_code == 403, "a browser's socket (Origin header) is refused with 403")

    sixty = await stream(ws_url, pcm, fps=60)
    want = -(-len(pcm) // 4 * 60 // RATE)
    check(len(sixty.get("frames", [])) == want, f"60 fps: {len(sixty.get('frames', []))} frames (expected {want})")
    back = await stream(ws_url, pcm)
    check(back.get("frames") == frames, "back at 30 fps the frames are the same again")

    live = await stream(ws_url, pcm, realtime=True)
    lag = sorted(live["lag"])
    check(live["frames"] == frames, "fed at the speed of speech, the frames are the same")
    print(
        f"      lag behind the audio, 40 ms chunks at real time: "
        f"mean {sum(lag) / len(lag):.3f} s, median {lag[len(lag) // 2]:.3f} s, max {lag[-1]:.3f} s"
    )
    check(lag[-1] < 1.0, "no frame arrives more than a second late")

    print("FAILED" if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2])))
