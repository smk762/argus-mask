"""A head's meshes, read from its .glb with numpy: every part's vertices at rest, its ARKit morph deltas and its
triangles, concatenated into one vertex space (parts in file order). Splats are bound to these triangles, so the
trainer, the exporter and the viewer must all build this space the same way: the viewer gets ``parts`` (name, vertex
offset, count) beside the splats and fills its vertex texture in that order.

Vertices are in the glTF scene's space at the bind pose (the stock heads' nodes are identity: ``load`` refuses one
that isn't). A frame's vertices are ``rest + weights @ deltas``: morphs only. The head's turn is the rig's at run
time, and rides the triangles with everything else.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass
class Head:
    names: list[str]  # the parts, in file order
    offsets: np.ndarray  # (P,) first vertex of each part
    counts: np.ndarray  # (P,) vertices per part
    rest: np.ndarray  # (V, 3) float32
    uv: np.ndarray  # (V, 2) float32
    tris: np.ndarray  # (T, 3) int32, into the concatenated vertices
    tri_part: np.ndarray  # (T,) part index of each triangle
    blendshapes: list[str]  # ARKit names, the order of ``deltas``
    deltas: np.ndarray  # (B, V, 3) float32; zero where a part has no such target
    textures: dict[str, bytes]  # part name -> its base colour image (PNG/JPEG bytes), if any

    def vertices(self, weights: np.ndarray) -> np.ndarray:
        """(V, 3) for one frame's weights (B,), or (N, V, 3) for (N, B)."""
        return self.rest + np.tensordot(weights, self.deltas, axes=(-1, 0))


def _glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path}: not a .glb")
    n, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A:
        raise ValueError(f"{path}: first chunk is not JSON")
    gltf = json.loads(data[20 : 20 + n])
    m, _ = struct.unpack_from("<II", data, 20 + n)
    return gltf, data[28 + n : 28 + n + m]


def _view(gltf: dict, blob: bytes, index: int) -> bytes:
    view = gltf["bufferViews"][index]
    start = view.get("byteOffset", 0)
    return blob[start : start + view["byteLength"]]


def _accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    dtype, width, count = COMPONENT[acc["componentType"]], WIDTH[acc["type"]], acc["count"]
    if "bufferView" in acc:
        view = gltf["bufferViews"][acc["bufferView"]]
        raw = _view(gltf, blob, acc["bufferView"])
        stride = view.get("byteStride")
        item = np.dtype(dtype).itemsize * width
        start = acc.get("byteOffset", 0)
        if stride and stride != item:
            rows = [np.frombuffer(raw, dtype, width, start + i * stride) for i in range(count)]
            out = np.stack(rows).astype(np.float32 if dtype == np.float32 else dtype)
        else:
            out = np.frombuffer(raw, dtype, count * width, start).reshape(count, width).copy()
    else:
        out = np.zeros((count, width), dtype)
    if sparse := acc.get("sparse"):
        idx = np.frombuffer(
            _view(gltf, blob, sparse["indices"]["bufferView"]),
            COMPONENT[sparse["indices"]["componentType"]],
            sparse["count"],
            sparse["indices"].get("byteOffset", 0),
        )
        values = np.frombuffer(
            _view(gltf, blob, sparse["values"]["bufferView"]),
            dtype,
            sparse["count"] * width,
            sparse["values"].get("byteOffset", 0),
        ).reshape(-1, width)
        out[idx.astype(np.int64)] = values
    return out


def load(path: Path, blendshapes: list[str]) -> Head:
    gltf, blob = _glb(Path(path))
    meshes = {i for i, node in enumerate(gltf["nodes"]) if "mesh" in node}
    parents = {i for i, node in enumerate(gltf["nodes"]) if meshes & set(node.get("children", []))}
    for i in meshes | parents:
        node = gltf["nodes"][i]
        if any(k in node for k in ("translation", "rotation", "scale", "matrix")):
            raise ValueError(f"{path}: node {node.get('name')!r} has a transform; the bake expects identity nodes")
    names, offsets, counts, rests, uvs, tris, parts, deltas, textures = [], [], [], [], [], [], [], [], {}
    base = 0
    for node in gltf["nodes"]:
        if "mesh" not in node:
            continue
        mesh = gltf["meshes"][node["mesh"]]
        if len(mesh["primitives"]) != 1:
            raise ValueError(f"{path}: {mesh['name']} has {len(mesh['primitives'])} primitives; expected one")
        prim = mesh["primitives"][0]
        pos = _accessor(gltf, blob, prim["attributes"]["POSITION"]).astype(np.float32)
        uv = _accessor(gltf, blob, prim["attributes"]["TEXCOORD_0"]).astype(np.float32)
        idx = _accessor(gltf, blob, prim["indices"]).reshape(-1, 3).astype(np.int64)
        target_names = mesh.get("extras", {}).get("targetNames", [])
        d = np.zeros((len(blendshapes), len(pos), 3), np.float32)
        for name, target in zip(target_names, prim.get("targets", []), strict=True):
            if name in blendshapes:
                d[blendshapes.index(name)] = _accessor(gltf, blob, target["POSITION"])
        names.append(node["name"])
        offsets.append(base)
        counts.append(len(pos))
        rests.append(pos)
        uvs.append(uv)
        tris.append(idx + base)
        parts.append(np.full(len(idx), len(names) - 1, np.int32))
        deltas.append(d)
        material = gltf["materials"][prim["material"]] if "material" in prim else {}
        tex = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if tex is not None:
            image = gltf["images"][gltf["textures"][tex["index"]]["source"]]
            textures[node["name"]] = _view(gltf, blob, image["bufferView"])
        base += len(pos)
    return Head(
        names=names,
        offsets=np.array(offsets, np.int32),
        counts=np.array(counts, np.int32),
        rest=np.concatenate(rests),
        uv=np.concatenate(uvs),
        tris=np.concatenate(tris).astype(np.int32),
        tri_part=np.concatenate(parts),
        blendshapes=list(blendshapes),
        deltas=np.concatenate(deltas, axis=1),
        textures=textures,
    )
