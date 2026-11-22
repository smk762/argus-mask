"""Read a 3DGS PLY and write SPZ (version 3), the format bound splats ship in (``ModelFile.splats``).

SPZ is Niantic's (MIT; github.com/nianticlabs/spz): gzip of a 16-byte header, then each attribute for every splat in
turn: positions (24-bit fixed point), alphas, colours, scales, rotations (smallest three, 32 bits), SH. The
quantisation follows the reference encoder, and the splats keep the PLY's order, which the binding indexes. No
coordinate change: the viewer reads the file as written.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

MAGIC = 0x5053474E  # "NGSP"
VERSION = 3
COLOR_SCALE = 0.15
SH_COEFFS = {0: 0, 1: 3, 2: 8, 3: 15}
SH1_BUCKET, SH_REST_BUCKET = 8, 16  # 5 bits for degree 1, 4 for the rest


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """A binary little-endian PLY of float32 vertex properties, as {name: (N,) float32}."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"{path}: not a binary little-endian PLY")
    elements = [line.split()[1] for line in header if line.startswith("element")]
    if elements != ["vertex"]:
        raise ValueError(f"{path}: elements {elements}; expected vertex alone")
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    names = []
    for line in header:
        if line.startswith("property"):
            kind, name = line.split()[1:3]
            if kind != "float":
                raise ValueError(f"{path}: property {name} is {kind}; expected float")
            names.append(name)
    table = np.frombuffer(data, np.float32, count * len(names), end).reshape(count, len(names))
    return {name: table[:, i] for i, name in enumerate(names)}


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def _sh(x: np.ndarray, bucket: int) -> np.ndarray:
    q = np.round(x * 128.0).astype(np.int32) + 128
    q = (q + bucket // 2) // bucket * bucket
    return np.clip(q, 0, 255).astype(np.uint8)


def _rotations(q_xyzw: np.ndarray) -> np.ndarray:
    """Smallest three: the largest component's index in the top two bits, then the others' sign and 9 bits each."""
    q = q_xyzw / np.linalg.norm(q_xyzw, axis=1, keepdims=True)
    largest = np.abs(q).argmax(axis=1)
    negate = q[np.arange(len(q)), largest] < 0
    comp = largest.astype(np.uint32)
    for i in range(4):
        keep = largest != i
        neg = ((q[:, i] < 0) ^ negate).astype(np.uint32)
        mag = (511.0 * (np.abs(q[:, i]) / np.sqrt(0.5)) + 0.5).astype(np.uint32)
        comp = np.where(keep, (comp << 10) | (neg << 9) | mag, comp)
    return comp.astype("<u4").view(np.uint8).reshape(-1, 4)


def encode(
    ply: dict[str, np.ndarray], sh_degree: int = 3, fractional_bits: int = 12, antialiased: bool = False
) -> bytes:
    """SPZ bytes (gzip'd) for a 3DGS PLY's properties, at *sh_degree* (at most the PLY's)."""
    n = len(ply["x"])
    if n == 0:
        raise ValueError("the PLY has no splats")
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float64)
    scaled = np.round(xyz * (1 << fractional_bits))
    if not np.isfinite(scaled).all() or np.abs(scaled).max() >= 1 << 23:  # before the cast, which would wrap
        raise ValueError("a splat lies beyond SPZ's 24-bit range at these fractional bits")
    fixed = scaled.astype(np.int32)
    positions = fixed.astype("<i4").view(np.uint8).reshape(n, 3, 4)[:, :, :3].reshape(n, 9)
    alphas = _u8(255.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float64))))
    colours = _u8(np.stack([ply[f"f_dc_{c}"] for c in range(3)], 1) * (COLOR_SCALE * 255) + 0.5 * 255)
    scales = _u8((np.stack([ply[f"scale_{c}"] for c in range(3)], 1) + 10.0) * 16.0)
    rotations = _rotations(np.stack([ply["rot_1"], ply["rot_2"], ply["rot_3"], ply["rot_0"]], 1).astype(np.float64))
    dim = SH_COEFFS[sh_degree]
    have = sum(1 for k in ply if k.startswith("f_rest_")) // 3
    if dim > have:
        raise ValueError(f"the PLY has {have} SH coefficients a channel; degree {sh_degree} needs {dim}")
    # The PLY is channel-major (every red coefficient, then green, then blue); SPZ is coefficient-major.
    if dim == 0:  # colour only: no SH bytes
        sh = np.zeros((n, 0), np.uint8)
    else:
        rest = np.stack([ply[f"f_rest_{c * have + j}"] for j in range(dim) for c in range(3)], 1).reshape(n, dim, 3)
        sh = np.concatenate([_sh(rest[:, :3], SH1_BUCKET), _sh(rest[:, 3:], SH_REST_BUCKET)], 1).reshape(n, dim * 3)
    header = struct.pack("<IIIBBBB", MAGIC, VERSION, n, sh_degree, fractional_bits, int(antialiased), 0)
    body = b"".join(a.tobytes() for a in (positions, alphas, colours, scales, rotations, sh))
    return gzip.compress(header + body, mtime=0)
