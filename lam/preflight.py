"""The checks `make_head.py` runs before any LAM code (#126): the FLAME model is FLAME 2023 Open, and the landmark
embedding LAM and VHAP will load is ours. Kept apart from `make_head.py`, which runs LAM as it is imported, so CI can
test the refusals without LAM or a GPU (#136).

Run from the LAM checkout: the paths are where LAM and VHAP look for their files.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
from landmarks import embedding

# FLAME 2023 Open (CC BY 4.0), as downloaded from flame.is.tue.mpg.de. The academic FLAME 2023 is a different file.
FLAME_OPEN_SHA256 = "e75a0990728ba038c7da2a420ae4396f7ddd7781366c13026066c5b24f127623"
FLAME_PATHS = (
    "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl",
    "model_zoo/human_parametric_models/flame_vhap/flame2023.pkl",
)
# Ours, measured on cleared heads for MediaPipe's landmarks (landmarks/measure.py), in place of MPI's.
EMBEDDING_PATHS = (
    "model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy",
    "model_zoo/human_parametric_models/flame_vhap/landmark_embedding_with_eyes.npy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def problems(root: Path = Path(), flame_sha256: str = FLAME_OPEN_SHA256) -> list[str]:
    """Why a head must not be made from the checkout at *root*: empty when every file is the one we checked."""
    found = []
    for rel in FLAME_PATHS:
        path = root / rel
        if not path.is_file():
            found.append(f"{rel} is missing: mount flame2023_Open.pkl there")
        elif (digest := _sha256(path)) != flame_sha256:
            found.append(f"{rel} is not FLAME 2023 Open (sha256 {digest[:12]}…): mount flame2023_Open.pkl over it")
    ours = embedding.load()
    for rel in EMBEDDING_PATHS:
        path = root / rel
        try:
            theirs = np.load(path, allow_pickle=True, encoding="latin1")[()]
            mine = isinstance(theirs, dict) and embedding.same(theirs, ours)
        except (OSError, ValueError, KeyError, TypeError, EOFError, pickle.UnpicklingError):
            mine = False
        if not mine:
            found.append(f"{rel} is not argus-mask's landmark embedding: mount the one embedding.py writes over it")
    return found
