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

`train.py` writes the following:

| File | Contents |
|---|---|
| `splats.ply` | A standard 3DGS PLY at the rest pose, SH degree 3. |
| `binding.bin` | Each splat's triangle, u and v as float32, in the PLY's order. |
| `head.json` | The parts, in the order their vertices are concatenated. |
| `metrics.json` | PSNR on training views and on held-out views. |
| `view*_gt.png`, `view*_trained.png` | A few held-out views, to compare against the viewer. |

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
