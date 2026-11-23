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


# All a pickled embedding may name: numpy's array reconstruction (`numpy.core` before numpy 2). Anything else is not
# ours, and is refused before it can run: the check that vets a file must not execute it.
_ARRAY_GLOBALS = {
    (module, name)
    for module in ("numpy.core.multiarray", "numpy._core.multiarray")
    for name in ("_reconstruct", "scalar")
} | {("numpy", "ndarray"), ("numpy", "dtype")}


class _ArraysOnly(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) not in _ARRAY_GLOBALS:
            raise pickle.UnpicklingError(f"{module}.{name} is not part of an embedding")
        return super().find_class(module, name)


def _load_embedding(path: Path) -> object:
    """A `.npy` as LAM loads it (`np.load(allow_pickle=True)[()]`), but unpickling nothing beyond numpy arrays.

    Versions 1.0, 2.0 and 3.0 are read as numpy reads them (3.0 is 2.0's layout with a UTF-8 header). A
    Fortran-ordered file is transposed back, as ``np.load`` does; any other version is refused.
    """
    # 3.0's UTF-8 header lives on numpy's private reader; the public ones stop at 2.0 (latin1).
    from numpy.lib._format_impl import _read_array_header  # noqa: PLC0415

    with path.open("rb") as f:
        version = np.lib.format.read_magic(f)
        if version not in ((1, 0), (2, 0), (3, 0)):
            raise ValueError(f"{path.name}: .npy format {version[0]}.{version[1]} is not supported")
        shape, fortran_order, dtype = _read_array_header(f, version)
        if dtype.hasobject:
            array = _ArraysOnly(f, encoding="latin1").load()
        else:
            count = 1 if len(shape) == 0 else int(np.multiply.reduce(shape, dtype=np.int64))
            array = np.fromfile(f, dtype, count)
            array = array.reshape(shape[::-1]).transpose() if fortran_order else array.reshape(shape)
    return array[()]


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
            theirs = _load_embedding(path)
            mine = isinstance(theirs, dict) and embedding.same(theirs, ours)
        except Exception:  # noqa: BLE001 - a file that is not ours, however it fails, is a problem to list
            mine = False
        if not mine:
            found.append(f"{rel} is not argus-mask's landmark embedding: mount the one embedding.py writes over it")
    return found
