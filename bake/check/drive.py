"""Draw camera views in the check page and compare them with the trainer's renders and Cycles' (PSNR).

    MODE=splat|mesh [SPLATS=splats.spz] python bake/check/drive.py <run> <work dir> [view,view,...]

Needs the page served at http://127.0.0.1:8779/ from the work directory, and playwright, pillow and numpy. Saves
``trained/<run>/view<i>_<mode>.png``; by default draws the held-out views the trainer saved.
"""

import asyncio
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.async_api import async_playwright

MODE = os.environ.get("MODE", "splat")
SPLATS = os.environ.get("SPLATS", "splats.ply")  # or an SPZ beside it


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(-10 * np.log10(((a.astype(float) - b.astype(float)) ** 2).mean() / 255**2))


async def main(run: str, root: Path, views: list[int] | None) -> None:
    out = root / "trained" / run
    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--use-angle=gl-egl", "--ignore-gpu-blocklist"])
        page = await browser.new_page(viewport={"width": 768, "height": 768})
        page.on("pageerror", lambda e: print("pageerror:", e))
        await page.goto(f"http://127.0.0.1:8779/?run={run}&mode={MODE}&splats={SPLATS}")
        await page.wait_for_function("window.__bake && __bake.ready", timeout=120_000)
        for i in views or [int(t.name[4:8]) for t in sorted(out.glob("view*_trained.png"))]:
            await page.evaluate(f"__bake.view({i})")
            shot = np.asarray(Image.open(io.BytesIO(await page.screenshot())).convert("RGB"))
            Image.fromarray(shot).save(
                out / f"view{i:04d}_{MODE}{'' if SPLATS == 'splats.ply' else '_' + Path(SPLATS).stem}.png"
            )
            trained, gt = out / f"view{i:04d}_trained.png", out / f"view{i:04d}_gt.png"
            if MODE != "splat" or not trained.exists():
                print(f"view {i}: drawn")
                continue
            tr, g = np.asarray(Image.open(trained).convert("RGB")), np.asarray(Image.open(gt).convert("RGB"))
            print(
                f"view {i}: viewer~trainer {psnr(shot, tr):.1f} dB, viewer~cycles {psnr(shot, g):.1f} dB, "
                f"trainer~cycles {psnr(tr, g):.1f} dB"
            )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(
        main(sys.argv[1], Path(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None)
    )
