"""A batch of source portraits for LAM heads, from argus-darkroom: FLUX.1 schnell, no LoRA, one per seed (#151).

    python lam/portraits.py --subject "a woman in her thirties, short dark hair with a fringe over the forehead, \
        green eyes, wearing a charcoal crew-neck top" --seeds 501-508 --out work/portraits

Writes ``<out>/<seed>.png`` and ``<out>/<seed>.generation.json`` (darkroom's record: the face's provenance). A
seed darkroom did not pass, or a record that names a LoRA, writes nothing: such a face would not be `synthetic`.
The prompt around the subject is the recipe in docs/synthetic-faces.md § For a LAM head.

Uses :mod:`argus_mask.portraits` (the same templates and checks the guided editor's source step uses, #167).
Standard library + the installed package only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from argus_mask.portraits import (
    CHARACTER,
    CHECKPOINT,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    SEMIREAL,
    STYLISED,
    TEMPLATE,
    TEMPLATES,
    generate_body,
    prompt_for,
    refusal,
    seeds,
)

# Re-exported so tests/test_lam.py and callers that import lam/portraits keep the same names.


def last_event(reply) -> dict:
    """The last JSON event of darkroom's stream: the one that carries the record."""
    last = ""
    for raw in reply:
        line = raw.decode().strip()
        if line.startswith("data:"):
            last = line[5:].strip()
    return json.loads(last) if last else {}


def generate(darkroom: str, prompt: str, seed: int, width: int, height: int, policy: str) -> dict:
    body = generate_body(prompt, seed, width=width, height=height, policy=policy, checkpoint=CHECKPOINT)
    request = urllib.request.Request(  # noqa: S310 - the caller's own darkroom
        f"{darkroom.rstrip('/')}/generate/stream",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as reply:  # noqa: S310
        return last_event(reply).get("record") or {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="who: age (18 or over), hair, eyes, clothes; no names")
    ap.add_argument("--seeds", type=seeds, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--darkroom", default="http://localhost:8106")
    ap.add_argument("--policy", default="default")
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--style", choices=sorted(TEMPLATES), default="photo")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    prompt = prompt_for(a.subject, a.style)
    for seed in a.seeds:
        record = generate(a.darkroom, prompt, seed, a.width, a.height, a.policy)
        if why := refusal(record):
            print(f"{seed}: skipped: {why}", file=sys.stderr)
            continue
        with urllib.request.urlopen(f"{a.darkroom.rstrip('/')}/image/{record['id']}", timeout=120) as reply:  # noqa: S310
            (a.out / f"{seed}.png").write_bytes(reply.read())
        (a.out / f"{seed}.generation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{seed}: {a.out / f'{seed}.png'}")


if __name__ == "__main__":
    main()
