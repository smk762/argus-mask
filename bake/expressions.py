"""Pick the expressions a bake renders: frames from real takes, spread out by farthest-point sampling on their
ARKit weights, neutral first. A held-out set comes from takes the training set never saw.

    python bake/expressions.py --train recordings/*.take.ndjson demo/clips/hello/take.ndjson \
        --holdout demo/clips/how/take.ndjson --n 24 --n-holdout 6 -o work/expressions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def frames(paths: list[Path]) -> tuple[list[str], np.ndarray]:
    names, rows = None, []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            if names is None:
                names = header["blendshapes"]
            if header["blendshapes"] != names:
                raise SystemExit(f"{path}: another blendshape order")
            rows += [json.loads(line)["weights"] for line in fh if '"type":"frame"' in line]
    return names, np.asarray(rows, np.float32)


def spread(weights: np.ndarray, n: int, start: np.ndarray | None = None) -> list[int]:
    """Farthest-point sampling: each pick is the frame farthest from every pick so far (and from *start*).

    Picks are distinct frames, so fewer than *n* come back once every frame left repeats a pick (or *start*)."""
    picked: list[int] = []
    seen = np.full(len(weights), np.inf)
    if start is not None and len(weights):
        seen = np.linalg.norm(weights - start, axis=1)
    for _ in range(min(n, len(weights))):
        i = int(np.argmax(seen))
        if seen[i] <= 0:
            break
        picked.append(i)
        seen = np.minimum(seen, np.linalg.norm(weights - weights[i], axis=1))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--holdout", type=Path, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-holdout", type=int, default=6)
    ap.add_argument("-o", type=Path, required=True)
    a = ap.parse_args()
    names, train = frames(a.train)
    names2, held = frames(a.holdout)
    if names2 != names:
        raise SystemExit("the held-out takes use another blendshape order")
    neutral = np.zeros(len(names), np.float32)
    # Half the extremes (farthest-point), half the ordinary (evenly through the frames): extremes alone are mostly
    # blinks and rolled lips, and most of a take is open eyes and a moderate mouth.
    far = (a.n - 1) // 2
    even = np.linspace(0, len(train) - 1, a.n - 1 - far + 2)[1:-1].round().astype(int)
    picks = [neutral] + [train[i] for i in spread(train, far, start=neutral)] + [train[i] for i in even]
    holdout = [held[i] for i in spread(held, a.n_holdout)]
    out = {
        "blendshapes": names,
        "train": [[round(float(x), 4) for x in w] for w in picks],
        "holdout": [[round(float(x), 4) for x in w] for w in holdout],
        "sources": {"train": [str(p) for p in a.train], "holdout": [str(p) for p in a.holdout]},
    }
    a.o.parent.mkdir(parents=True, exist_ok=True)
    a.o.write_text(json.dumps(out) + "\n", encoding="utf-8")
    for label, ws in (("train", picks), ("holdout", holdout)):
        for w in ws[:8]:
            top = np.argsort(w)[::-1][:3]
            print(label, ", ".join(f"{names[k]} {w[k]:.2f}" for k in top))


if __name__ == "__main__":
    main()
