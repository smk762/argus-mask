"""argus-mask's FLAME landmark embedding: kept as JSON in the repo, written as LAM's `.npy` for a run (#126).

    python lam/landmarks/embedding.py npy <out.npy>      # LAM's format, from embedding.json

`embedding.json` holds, per landmark (68 MediaPipe points in iBUG order, then the right and left pupils), a face of
FLAME 2023 Open and barycentric weights on it, with how they were measured (`measure.py`). LAM and VHAP load an
embedding with `np.load(path, allow_pickle=True)[()]` and read `full_lmk_faces_idx` (1, 70) and
`full_lmk_bary_coords` (1, 70, 3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JSON = HERE / "embedding.json"


def load(path: Path = JSON) -> dict:
    data = json.loads(Path(path).read_text())
    faces = np.array([data["faces"]], dtype=np.int64)
    bary = np.array([data["barycentric"]], dtype=np.float64)
    if faces.shape != (1, 70) or bary.shape != (1, 70, 3):
        raise ValueError(f"{path}: want 70 landmarks, got faces {faces.shape}, barycentric {bary.shape}")
    return {"full_lmk_faces_idx": faces, "full_lmk_bary_coords": bary}


def save(embedding: dict, about: dict, path: Path = JSON) -> None:
    data = {
        **about,
        "faces": embedding["full_lmk_faces_idx"][0].tolist(),
        "barycentric": [[round(float(w), 8) for w in row] for row in embedding["full_lmk_bary_coords"][0]],
    }
    Path(path).write_text(json.dumps(data, indent=1) + "\n")


def same(a: dict, b: dict) -> bool:
    return np.array_equal(a["full_lmk_faces_idx"], b["full_lmk_faces_idx"]) and np.allclose(
        a["full_lmk_bary_coords"], b["full_lmk_bary_coords"], atol=1e-7
    )


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "npy":
        sys.exit(__doc__)
    np.save(sys.argv[2], load(), allow_pickle=True)
    print("EMBEDDING", sys.argv[2])
