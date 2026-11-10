#!/usr/bin/env python3
"""Write ``src/argus_mask/editor/modifiers.json`` from MPFB's own target catalogue.

MakeHuman's modelling targets (CC0, bundled with MPFB) are listed in
``<mpfb>/data/targets/target.json``, with each slider's opposite files and its
left and right sides. This keeps the parts a bust shows (head, face, neck),
joins left and right into one symmetric slider, and writes the allow-list both
the editor and ``build_head.py`` read. A preset can only name what is in it.

    python scripts/make_modifiers.py --mpfb $BLENDER_USER_RESOURCES/extensions/user_default/mpfb

Run it again after an MPFB upgrade. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src" / "argus_mask" / "editor" / "modifiers.json"

# MPFB's groups that a bust shows, in the order the editor lists them, with the editor's heading.
GROUPS = {
    "head": "Head shape",
    "forehead": "Forehead",
    "eyebrows": "Brow ridge",
    "eyes": "Eyes",
    "ears": "Ears",
    "cheek": "Cheeks",
    "nose": "Nose",
    "mouth": "Lips and mouth",
    "chin": "Chin and jaw",
    "neck": "Neck",
}
# A two-way category's name ends with its two directions; "-decr-incr" is dropped from the slider's name,
# the others stay (they tell "move down-up" from "move in-out") and become the slider's two ends.
_PAIR = re.compile(r"-([a-z]+)-([a-z]+)$")
_ENDS = {"backward": "back", "uncompress": "open", "compress": "pinched", "decr": "less", "incr": "more"}
_DIRECTIONS = {
    "down",
    "up",
    "in",
    "out",
    "backward",
    "forward",
    "concave",
    "convex",
    "compress",
    "uncompress",
    "square",
    "round",
    "pointed",
    "triangle",
}
_WORDS = {
    "trans": "move",
    "scale": "",
    "horiz": "width",
    "vert": "height",
    "volume": "size",
    "upperlip": "upper lip",
    "lowerlip": "lower lip",
    "cupidsbow": "Cupid's bow",
    "invertedtriangular": "heart",
    "septumangle": "septum angle",
    "eyefold": "eye fold",
    "prognathism": "jaw thrust",
    "prominent": "forward",
    "point": "tip",
    "rot": "rotate",
    "ext": "outer",
    "circ": "circumference",
    "measure": "",
    "bones": "bone width",
}


# What a two-way slider's ends mean, for the page to say "45% wider" rather than "+0.45". Taken from the
# direction words a name ends with, else from the measure it changes, else "less" and "more".
_MEASURES = {
    "horiz": ("narrower", "wider"),
    "width": ("narrower", "wider"),
    "vert": ("shorter", "taller"),
    "height": ("shorter", "taller"),
    "depth": ("shallower", "deeper"),
    "volume": ("smaller", "bigger"),
    "scale": ("smaller", "bigger"),
    "fat": ("thinner", "fatter"),
    "age": ("younger", "older"),
    "lobe": ("smaller", "bigger"),
    "flap": ("smaller", "bigger"),
    "wing": ("flatter", "further out"),
    "bag": ("smaller", "bigger"),
    "temple": ("narrower", "wider"),
    "double": ("less", "more"),
    "bones": ("narrower", "wider"),
    "circ": ("thinner", "thicker"),
    "prominent": ("receding", "prominent"),
    "prognathism": ("back", "forward"),
}
# Labels the generated ones say badly.
_LABELS = {"head-age": "Apparent age"}


def ends(name: str) -> list[str]:
    """``["narrower", "wider"]`` for ``nose-scale-horiz``; ``["down", "up"]`` for ``nose-trans-down-up``."""
    m = _PAIR.search(name)
    if m and set(m.groups()) <= _DIRECTIONS and m.group(1) != m.group(2):
        return [_ENDS.get(w, w) for w in m.groups()]
    for word in reversed(re.sub(r"\d+$", "", name).split("-")):
        if word in _MEASURES:
            return list(_MEASURES[word])
    return ["less", "more"]


def label(name: str, group: str) -> str:
    """``mouth-upperlip-volume`` to ``Upper lip size``; ``nose-trans-down-up`` to ``Move: down – up``."""
    if name in _LABELS:
        return _LABELS[name]
    suffix = ""
    m = _PAIR.search(name)
    if m and set(m.groups()) <= _DIRECTIONS and m.group(1) != m.group(2):
        lo, hi = (_ENDS.get(w, w) for w in m.groups())
        suffix = f": {lo} – {hi}"
        name = name[: m.start()]
    parts = name.split("-")
    while parts and parts[0] in (group, group.rstrip("s"), "eye", "ear", "measure"):
        parts = parts[1:]
    words = []
    for p in parts:
        n = re.fullmatch(r"([a-z]+)(\d+)", p)
        word = _WORDS.get(p, p) if not n else f"{_WORDS.get(n.group(1), n.group(1))} {n.group(2)}"
        if word:
            words.append(word)
    text = (" ".join(words) or name.split("-")[-1]) + suffix
    return text[0].upper() + text[1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mpfb", required=True, type=Path, help="MPFB's extension directory (holds data/targets)")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    root = args.mpfb / "data" / "targets"
    catalogue = json.loads((root / "target.json").read_text(encoding="utf-8"))

    out: list[dict] = []
    seen: set[str] = set()
    for group in GROUPS:
        for cat in catalogue.get(group, {}).get("categories", []):
            ops = cat.get("opposites")
            if ops:
                sides = ("left", "right") if cat.get("has_left_and_right") else ("unsided",)
                negative = [ops[f"negative-{k}"] for k in sides if ops.get(f"negative-{k}")]
                positive = [ops[f"positive-{k}"] for k in sides if ops.get(f"positive-{k}")]
                name = cat["name"].removesuffix("-decr-incr")
            else:  # one-way: its own file(s), 0 to 1
                negative, positive = [], list(cat.get("targets") or [])
                name = cat["name"]
            if not positive or (ops and not negative):
                print(f"make_modifiers: skipping {group}/{cat.get('name')}: no files", file=sys.stderr)
                continue
            if name in seen:
                print(f"make_modifiers: two sliders would both be called {name!r}", file=sys.stderr)
                return 1
            for stem in negative + positive:
                if not (root / group / f"{stem}.target.gz").is_file():
                    print(f"make_modifiers: {group}/{stem}.target.gz is listed but not installed", file=sys.stderr)
                    return 1
            seen.add(name)
            out.append(
                {
                    "name": name,
                    "group": group,
                    "label": label(name, group),
                    "ends": ends(name) if negative else [],
                    "negative": [f"{group}/{s}" for s in negative],
                    "positive": [f"{group}/{s}" for s in positive],
                }
            )
    doc = {
        "about": "MakeHuman modelling targets a preset may set (CC0, bundled with MPFB). Generated by "
        "scripts/make_modifiers.py from MPFB's data/targets/target.json; do not edit by hand.",
        "groups": GROUPS,
        "modifiers": out,
    }
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"make_modifiers: wrote {len(out)} sliders to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
