"""A contact sheet of LAM heads, to pick one (#151): each head from the front, turned, tipped and with its mouth open.

    python lam/sheet.py --url http://127.0.0.1:8109 --out sheet.jpg head-a head-b ...

The heads are faces a running argus-mask server already serves. Each is drawn by the viewer in record mode with its
limits off (``?turn=0&jaw=1``), so the sheet shows where a head gives out: the far temple and the hairline as it
turns and tips back, the inside of the mouth as the jaw opens. One row per head. Needs playwright and pillow.
"""

from __future__ import annotations

import argparse
import asyncio
import io
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.async_api import async_playwright

# (label, head turn in degrees, jawOpen). Positive pitch tips the head forward.
POSES = [
    ("front", {}, 0.0),
    ("yaw -20", {"yaw": -20}, 0.0),
    ("yaw -12", {"yaw": -12}, 0.0),
    ("yaw 12", {"yaw": 12}, 0.0),
    ("yaw 20", {"yaw": 20}, 0.0),
    ("back 8", {"pitch": -8}, 0.0),
    ("forward 15", {"pitch": 15}, 0.0),
    ("jaw 0.35", {}, 0.35),
    ("jaw 0.6", {}, 0.6),
]
SHAPES = ["jawOpen", "mouthLowerDownLeft", "mouthLowerDownRight"]
POSE = "([w, h, n]) => window.__mask.pose(w, h, n, 0)"


async def row(browser, url: str, face: str, size: int) -> list[Image.Image]:
    page = await browser.new_page(viewport={"width": size, "height": size}, device_scale_factor=1)
    try:
        await page.goto(f"{url}/?face={face}&hud=0&record=1&life=0&turn=0&jaw=1")
        await page.wait_for_function("window.__mask && (__mask.ready || __mask.error)", timeout=120_000)
        if error := await page.evaluate("window.__mask.error"):
            raise SystemExit(f"{face}: {error}")
        shots = []
        for _label, turn, jaw in POSES:
            head = {"yaw": 0, "pitch": 0, "roll": 0, **turn}
            await page.evaluate(POSE, [[jaw, jaw / 2, jaw / 2], head, SHAPES])
            shots.append(Image.open(io.BytesIO(await page.screenshot())).convert("RGB"))
        return shots
    finally:
        await page.close()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("faces", nargs="+")
    ap.add_argument("--url", default="http://127.0.0.1:8109")
    ap.add_argument("--out", type=Path, default=Path("sheet.jpg"))
    ap.add_argument("--size", type=int, default=512, help="each picture's side, before the crop to the head")
    a = ap.parse_args()
    cell = a.size * 5 // 8  # the head and shoulders, out of the standard shot
    left, top = (a.size - cell) // 2, a.size * 3 // 20
    sheet = Image.new("RGB", (cell * len(POSES), cell * len(a.faces)))
    draw = ImageDraw.Draw(sheet)
    async with async_playwright() as p:
        for r, face in enumerate(a.faces):  # a browser per head: several splat renderers in one stall each other
            browser = await p.chromium.launch(args=["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
            try:
                shots = await row(browser, a.url.rstrip("/"), face, a.size)
            finally:
                await browser.close()
            for c, shot in enumerate(shots):
                sheet.paste(shot.crop((left, top, left + cell, top + cell)), (c * cell, r * cell))
                draw.text(
                    (c * cell + 6, r * cell + 6), f"{face}  {POSES[c][0]}" if c == 0 else POSES[c][0], fill="yellow"
                )
            print(f"{face}: drawn", flush=True)
    sheet.save(a.out, quality=90)
    print(a.out)


if __name__ == "__main__":
    asyncio.run(main())
