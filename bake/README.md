# bake: splat heads from our own mesh heads (#126 route A)

This folder turns a mesh head built by mask's recipe (a `.glb` such as `stock-01`'s `head.glb`) into Gaussian splats
bound to that head's triangles. It is an offline asset build, a GPU tenant like Blender or Audio2Face. It is not part
of the package, and CI doesn't lint or test it.

The look comes from Cycles: path-traced subsurface skin, soft hair and light baked into each splat, with
view-dependent colour (SH degree 3). The motion stays the mesh's: every splat rides its own triangle, so the head's
ARKit shape keys and its rig drive the splats exactly as they drive the mesh. There's no FLAME and no per-splat
blendshape table.

## Steps

```bash
# 1. Expressions: frames from real Audio2Face takes, half extremes (farthest-point) and half ordinary; a held-out set
#    from takes the training never sees.
python bake/expressions.py --train recordings/*.take.ndjson demo/clips/hello/take.ndjson \
    demo/clips/screens/take.ndjson --holdout demo/clips/how/take.ndjson --n 24 --n-holdout 6 -o work/expressions.json

# 2. Cycles renders: each expression from many seeded views (mostly in front), RGBA, plus cameras.json.
blender -b -P bake/render_views.py -- --glb assets/stock-01/head.glb --expressions work/expressions.json \
    --out work/renders --views 40 --holdout-views 8          # resumable: rerun to finish

# 3. Training: splats bound to triangles, gsplat's rasteriser.
python bake/train.py --glb assets/stock-01/head.glb --expressions work/expressions.json \
    --renders work/renders --out work/trained --splats 150000 --steps 30000
```

```bash
# 4. Package: SPZ and binding beside the face's mesh, `model.splats` in its manifest (repo environment).
PYTHONPATH=bake python bake/package.py --run work/trained --face assets/stock-01 --license CC0-1.0
```

`train.py` writes the following:

| File | Contents |
|---|---|
| `splats.ply` | A standard 3DGS PLY at the rest pose, SH degree 3. |
| `binding.bin` | Each splat's triangle, u and v as float32, in the PLY's order. |
| `head.json` | The parts, in the order their vertices are concatenated. |
| `metrics.json` | PSNR on training views and on held-out views. |
| `view*_gt.png`, `view*_trained.png` | A few held-out views, to compare against the viewer. |

## Checking in the browser

`check/` draws a trained run in Spark with the binding (`ride.glsl`, a dyno), or the mesh head as the demo draws it,
at any camera in `cameras.json`, so the viewer can be compared pixel for pixel with the trainer and Cycles. Serve it
from a work directory:

```bash
mkdir -p work/check/trained && cd work/check
ln -s ../../bake/check/{index.html,check.js,ride.glsl} . && ln -s ../../src/argus_mask/viewer viewer
ln -s ../../assets/stock-01/head.glb . && cp ../renders/cameras.json ../expressions.json . && cp -r ../trained trained/a
npm pack @sparkjsdev/spark@2.4.0 && tar xzf sparkjsdev-spark-2.4.0.tgz package/dist/spark.module.js --strip-components 2
python3 -m http.server 8779 &
MODE=splat python ../../bake/check/drive.py a .   # PSNR: viewer~trainer, viewer~cycles, trainer~cycles
MODE=mesh python ../../bake/check/drive.py a .    # the same views as the mesh head
```

## Results

The first run on `stock-01` (2026-11-16): 24 training expressions × 40 views and 6 held-out expressions (from a take
the training never saw) × 8 views, 768², 64 spp; 149,007 splats, 30,000 steps, 11.5 minutes on an RTX 4060.

| | PSNR |
|---|---|
| Trainer ~ Cycles, training views | 37.7 dB |
| Trainer ~ Cycles, held-out views | 37.3 dB |
| Viewer ~ trainer, 6 held-out views | 35.0–37.8 dB |
| Viewer ~ Cycles, 6 held-out views | 30.0–37.0 dB |

![Cycles, trainer, viewer (Spark) and the mesh head, at six held-out views](results/route-a-1.jpg)

Columns: Cycles, the trainer, the splats in the viewer, the mesh head (look.js high). The splats keep the subsurface
skin, the soft light and the open mouth; fine detail (lids, hair edges) is a little soft, and the mesh's hair cards
still give a ragged hairline. The PLY is 36 MB; SPZ size with SH degree 3 is not measured yet (the format task).

## The format

`package.py` writes the head's `ModelFile.splats` (`BoundSplats` in `models.py`, wire 1.6), two files beside the
`.glb`:

- `<stem>.splats.spz`: the splats at rest, SPZ version 3 (`spz.py`, Niantic's format, MIT), SH degree 3, in the
  PLY's order. `spz.py` matches Spark's own encoder to one quantisation step, and draws the same.
- `<stem>.binding.bin`: per splat, little-endian uint32 triangle, float32 u, float32 v.

The manifest records the mesh build (`mesh_sha256`, which must be the model's sha256: a rebuilt head drops its
splats), the parts the triangles are counted through, the splats' licence, and `inputs`, every input with its
licence (the mesh, from the face's provenance; then `BAKE_INPUTS`). The server refuses a face whose splats don't
match (`inspect_bound_splats` in `faces.py`) and serves the two files at `/face/{id}/model/{file}`.

On `stock-01` (the held-out views, as the viewer draws them against the trainer):

| SPZ | Size | PSNR |
|---|---|---|
| (the PLY, 36 MB) | | 35.0–37.8 dB |
| SH 3 | 2.85 MiB | 33.5–35.4 dB |
| SH 0 | 2.01 MiB | 24.9–28.1 dB |

SH 3 it is: view-dependent colour carries the baked light. Quantisation costs about 2 dB.

## The binding

`mesh.py` builds one vertex space for the head: every part's vertices, with parts in file order, and a frame's
vertices are `rest + weights @ deltas`.

A splat has a triangle and fixed barycentrics (u, v). At expression e, take that triangle's point p_e and its frame
R_e (columns: the first edge, normal × that edge, the normal). The splat then sits at `p_e + R_e @ offset`, turned
by `R_e ∘ q_local`. The offset, q_local, the scales, the opacity and the SH coefficients are learned.

The viewer applies the same thing as a change from rest, in a Spark dyno (see the spike comment on #126):

- `centre = p_e + R_e R_0ᵀ (c_0 − p_0)`
- `quat = (R_e R_0ᵀ) q_0`

Its vertex texture is filled from three's `getVertexPosition` for each part, in `head.json`'s order, so the rig's
head turn rides along too.

Scales don't follow a triangle's stretch, in training or in the viewer.

## Environment

| Part | Version | Licence |
|---|---|---|
| Blender (Cycles) | 4.5 LTS | GPL-3.0 (a tool; the renders are ours) |
| `interior.exr` (Blender's bundled studio light, Poly Haven "Hotel Room") | 1K | CC0 (`datafiles/studiolights/world/license.txt`) |
| gsplat | 1.5.3 (`+pt24cu124` wheel, CPython 3.10) | Apache-2.0 |
| PyTorch | 2.4.1 + CUDA 12.4 | BSD-3-Clause |
| numpy, Pillow | | BSD, MIT-CMU |
| The head | `stock-01` | CC0 MakeHuman/MPFB assets, our recipe |

gsplat needs `packaging`, `setuptools`, `jaxtyping` and `rich` at import time, beyond its wheel's own requirements.

On a GPU shared with the suite's services, take the room that iris shows as admissible.
