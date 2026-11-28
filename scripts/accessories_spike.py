#!/usr/bin/env python3
"""Spike #67: pose splat-01 with ?accessory=earpiece twice; compare frame digests (docs/accessories-spike.md).

python scripts/accessories_spike.py [server URL]   # writes PNGs and summary.json under work/accessories-spike/
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8109"
OUT = Path(__file__).resolve().parents[1] / "work" / "accessories-spike"  # gitignored
ARKIT = [
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueOut",
]


def digest(png: bytes) -> str:
    return hashlib.sha256(png).hexdigest()[:16]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    page_url = f"{URL.rstrip('/')}/?face=splat-01&record=1&hud=0&life=0&accessory=earpiece&scene=none"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 960, "height": 720})
        page.goto(page_url, wait_until="networkidle", timeout=120_000)
        page.wait_for_function("() => window.__mask && window.__mask.ready", timeout=120_000)
        accessory = page.evaluate("() => window.__mask.accessory")
        results = {"url": page_url, "accessory": accessory, "poses": []}

        def pose(yaw: float, tag: str) -> str:
            page.evaluate(
                """([weights, head, names]) => window.__mask.pose(weights, head, names, 0)""",
                [[0.0] * len(ARKIT), {"yaw": yaw, "pitch": 0, "roll": 0}, ARKIT],
            )
            # pose() returns a Promise; wait for the next paint after it
            page.wait_for_timeout(800)
            png = page.screenshot(type="png")
            path = OUT / f"{tag}.png"
            path.write_bytes(png)
            d = digest(png)
            results["poses"].append({"tag": tag, "yaw": yaw, "sha256_16": d, "bytes": len(png)})
            return d

        a1 = pose(0.0, "yaw0-a")
        a2 = pose(0.0, "yaw0-b")
        b1 = pose(18.0, "yaw18-a")
        b2 = pose(18.0, "yaw18-b")
        results["same_pose_match"] = {"yaw0": a1 == a2, "yaw18": b1 == b2}
        results["turn_changed_pixels"] = a1 != b1
        (OUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
        browser.close()
        if not accessory or accessory.get("error"):
            raise SystemExit(2)
        if not results["same_pose_match"]["yaw0"] or not results["same_pose_match"]["yaw18"]:
            raise SystemExit(3)
        if not results["turn_changed_pixels"]:
            raise SystemExit(4)


if __name__ == "__main__":
    main()
