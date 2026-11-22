"""A batch of source portraits for LAM heads, from argus-darkroom: FLUX.1 schnell, no LoRA, one per seed (#151).

    python lam/portraits.py --subject "a woman in her thirties, short dark hair with a fringe over the forehead, \
        green eyes, wearing a charcoal crew-neck top" --seeds 501-508 --out work/portraits

Writes ``<out>/<seed>.png`` and ``<out>/<seed>.generation.json`` (darkroom's record: the face's provenance). A
seed darkroom did not pass, or a record that names a LoRA, writes nothing: such a face would not be `synthetic`.
The prompt around the subject is the recipe in docs/synthetic-faces.md § For a LAM head. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

CHECKPOINT = "flux/flux1-schnell-fp8-e4m3fn.safetensors"
SAMPLING = {"sampler": "euler", "scheduler": "simple", "steps": 4, "cfg": 1.0, "guidance": 3.5}
# What LAM needs of a portrait: square on, neutral, flat light (a smile or a shadow is baked into the head).
TEMPLATE = (
    "close-up studio portrait photograph of {subject}, head and shoulders, facing the camera squarely, head level, "
    "calm neutral expression, mouth closed and relaxed, not smiling, eyes open looking into the lens, "
    "very soft flat lighting from a large softbox directly in front, face evenly lit, no shadows on the face, "
    "plain dark grey background, sharp focus, natural skin texture, 85mm lens, f/8"
)


def seeds(text: str) -> list[int]:
    """``501-508`` or ``501,503,507``."""
    out: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        out.extend(range(int(first), int(last or first) + 1))
    return out


def last_event(reply) -> dict:
    """The last JSON event of darkroom's stream: the one that carries the record."""
    last = ""
    for raw in reply:
        line = raw.decode().strip()
        if line.startswith("data:"):
            last = line[5:].strip()
    return json.loads(last) if last else {}


def generate(darkroom: str, prompt: str, seed: int, width: int, height: int, policy: str) -> dict:
    body = {
        "user_prompt": prompt,
        "seed": seed,
        "enforcing_policy": policy,
        "checkpoint": CHECKPOINT,
        "sampling": {**SAMPLING, "width": width, "height": height},
    }
    request = urllib.request.Request(  # noqa: S310 - the caller's own darkroom
        f"{darkroom}/generate/stream", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=900) as reply:  # noqa: S310
        return last_event(reply).get("record") or {}


def refusal(record: dict) -> str | None:
    """Why this record's image is not a cleared source, or None."""
    if record.get("outcome") != "passed":
        return f"darkroom: {record.get('outcome', 'no record')} ({record.get('outcome_reason') or 'no reason given'})"
    if (record.get("request") or {}).get("loras"):
        return "the record names a LoRA: the face would be personal, not synthetic"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="who: age (18 or over), hair, eyes, clothes; no names")
    ap.add_argument("--seeds", type=seeds, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--darkroom", default="http://localhost:8106")
    ap.add_argument("--policy", default="default")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1280)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    prompt = TEMPLATE.format(subject=a.subject)
    for seed in a.seeds:
        record = generate(a.darkroom, prompt, seed, a.width, a.height, a.policy)
        if why := refusal(record):
            print(f"{seed}: skipped: {why}", file=sys.stderr)
            continue
        with urllib.request.urlopen(f"{a.darkroom}/image/{record['id']}", timeout=120) as reply:  # noqa: S310
            (a.out / f"{seed}.png").write_bytes(reply.read())
        (a.out / f"{seed}.generation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"{seed}: {a.out / f'{seed}.png'}")


if __name__ == "__main__":
    main()
