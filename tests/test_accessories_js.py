"""CPU checks for the #67 accessories spike helpers (no WebGL)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "src" / "argus_mask" / "viewer"


def test_earpiece_factory_and_overlay_are_shipped() -> None:
    assert (VIEWER / "accessories.js").is_file()
    assert (VIEWER / "accessories" / "earpiece.js").is_file()
    text = (VIEWER / "accessories.js").read_text()
    assert "export function attach" in text
    assert "export class BoneOverlay" in text
    assert "holder.scale.set(1, 1, 1)" in text  # strip LAM bone scale
    ear = (VIEWER / "accessories" / "earpiece.js").read_text()
    assert "export function createEarpiece" in ear
    assert "Math.random" not in ear
    assert "Date.now" not in ear
    assert "performance.now" not in ear


def test_viewer_wires_accessory_query() -> None:
    viewer = (VIEWER / "viewer.js").read_text()
    assert "?accessory=" in viewer
    assert "mountAccessory" in viewer
    splat = (VIEWER / "splat.js").read_text()
    assert "_drawAccessories" in splat
    assert "setAccessoryOverlay" in splat
