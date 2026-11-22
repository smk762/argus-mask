"""Put a trained run's splats beside a face's mesh model: SPZ, binding, and ``model.splats`` in its manifest (#126).

    python bake/package.py --run work/trained --face assets/stock-01 --license CC0-1.0

Writes ``<model stem>.splats.spz`` and ``<model stem>.binding.bin`` into the face's directory and records them in
``manifest.json`` (:class:`argus_mask.models.BoundSplats`), with every input's licence: the mesh's (from the face's
provenance) and the bake's own (``BAKE_INPUTS``). Refuses a run trained on another build of the mesh, or one whose
parts are not the mesh's. Then loads the face as the server would, so a face it writes is one the server serves.

Runs in the repo's environment (it imports argus_mask), not the trainer's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import spz

from argus_mask.faces import MANIFEST_NAME, FaceStore, glb_parts, read_glb, sha256_file
from argus_mask.models import BoundSplats, Credit, FaceManifest, SplatPart

# What render_views.py and train.py put into every bake, beside the mesh itself.
BAKE_INPUTS = [
    Credit(
        name="Expressions: frames of argus-mask's own Audio2Face takes (NVIDIA Audio2Face-3D models)",
        license="NVIDIA Open Model License (the models; NVIDIA claims no ownership rights in outputs)",
        url="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    ),
    Credit(
        name="Environment light: interior.exr, Poly Haven 'Hotel Room' by Greg Zaal, as bundled with Blender",
        license="CC0-1.0",
        url="https://polyhaven.com/a/hotel_room",
    ),
    Credit(
        name="Renderer: Blender 4.5 LTS, Cycles (a tool: its renders are not covered by its licence)",
        license="GPL-3.0-or-later",
        url="https://www.blender.org/about/license/",
    ),
    Credit(name="Rasteriser: gsplat 1.5.3", license="Apache-2.0", url="https://github.com/nerfstudio-project/gsplat"),
    Credit(name="Trainer: PyTorch 2.4.1", license="BSD-3-Clause", url="https://github.com/pytorch/pytorch"),
    Credit(
        name="Bake scripts: argus-mask bake/ (views, training, binding)",
        license="MIT",
        url="https://github.com/smk762/argus-mask/tree/dev/bake",
    ),
]


def args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, required=True, help="train.py's output directory")
    ap.add_argument("--face", type=Path, required=True, help="the face's directory (manifest.json and its .glb)")
    ap.add_argument("--license", required=True, help="the splats' licence (the mesh's, for a stock head)")
    ap.add_argument("--sh", type=int, default=3, choices=[0, 1, 2, 3], help="SH degree kept (3: view-dependent)")
    return ap.parse_args()


def main() -> None:
    a = args()
    face = a.face.resolve()
    manifest = FaceManifest.model_validate_json((face / MANIFEST_NAME).read_text(encoding="utf-8"))
    model = manifest.model
    glb = face / model.file
    if model.format != "gltf" or sha256_file(glb) != model.sha256:
        raise SystemExit(f"{glb}: not the mesh the manifest names")
    info = json.loads((a.run / "head.json").read_text(encoding="utf-8"))
    parts = glb_parts(read_glb(glb)[0])
    if [(p["name"], p["count"]) for p in info["parts"]] != [(name, v) for name, v, _ in parts]:
        raise SystemExit("the run's parts are not this mesh's: it was trained on another build; bake again")
    if info["triangles"] != sum(t for _, _, t in parts):
        raise SystemExit("the run's triangle count is not this mesh's: bake again")

    raw = np.fromfile(a.run / "binding.bin", np.float32).reshape(-1, 3)  # train.py: float32 triangle, u, v
    tri = raw[:, 0]
    if len(raw) != info["splats"] or not np.array_equal(tri, np.round(tri)):
        raise SystemExit("binding.bin does not hold one whole triangle index per splat")
    if not ((tri >= 0) & (tri < info["triangles"])).all():  # before the uint32 cast, where a negative would wrap
        raise SystemExit(f"binding.bin names a triangle outside 0..{info['triangles'] - 1}")
    binding = np.empty(len(raw), np.dtype([("tri", "<u4"), ("u", "<f4"), ("v", "<f4")]))
    binding["tri"], binding["u"], binding["v"] = raw[:, 0].astype(np.uint32), raw[:, 1], raw[:, 2]
    splat_bytes = spz.encode(spz.read_ply(a.run / "splats.ply"), a.sh)
    binding_bytes = binding.tobytes()

    stem = Path(model.file).stem
    names = (f"{stem}.splats.spz", f"{stem}.binding.bin")
    model.splats = BoundSplats(
        file=names[0],
        sha256=hashlib.sha256(splat_bytes).hexdigest(),
        binding=names[1],
        binding_sha256=hashlib.sha256(binding_bytes).hexdigest(),
        count=len(binding),
        mesh_sha256=model.sha256,
        parts=[SplatPart(name=n, vertices=v, triangles=t) for n, v, t in parts],
        license=a.license,
        inputs=[
            Credit(
                name=f"Mesh: {manifest.id} {model.file} ({manifest.provenance.source})"[:300],
                license=manifest.provenance.license or "personal (no licence stated)",
            ),
            *BAKE_INPUTS,
        ],
    )
    manifest = FaceManifest.model_validate(manifest.model_dump())  # the validators, on the whole manifest
    outputs = {
        names[0]: splat_bytes,
        names[1]: binding_bytes,
        MANIFEST_NAME: (manifest.model_dump_json(indent=2) + "\n").encode(),
    }
    # Load the result as the server will, from a copy, before the face itself changes: a re-bake that is refused
    # (raising with the reason) leaves the face as it was, still matching the manifest it has.
    with tempfile.TemporaryDirectory() as staging:
        staged = Path(staging) / manifest.id
        shutil.copytree(face, staged)
        for name, data in outputs.items():
            (staged / name).write_bytes(data)
        FaceStore(staged.parent).load(manifest.id)
    for name, data in outputs.items():  # each written aside and moved in whole; the manifest last
        part = face / f".{name}.part"
        part.write_bytes(data)
        os.replace(part, face / name)
    print(f"{manifest.id}: {len(binding)} splats, {len(splat_bytes) / 2**20:.2f} MiB SPZ at SH {a.sh}; served")


if __name__ == "__main__":
    main()
