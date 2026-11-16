# LAM heads

A portrait in, a Gaussian-splat head out: the `lam-splat` faces that mask serves and records ([#66](https://github.com/smk762/argus-mask/issues/66)). This directory runs [LAM](https://github.com/aigc3d/LAM) (Alibaba's one-shot animatable head) on one NVIDIA GPU. It does so with argus-mask's own FLAME code and FLAME 2023 Open, so that a head made from a cleared portrait can go on the hosted demo ([#126](https://github.com/smk762/argus-mask/issues/126)).

It is not part of the `argus-mask` Python package, it is not linted or tested by CI, and it runs offline: the server never needs a GPU.

## Licences, and what this directory changes

| Part | Licence | Here |
|---|---|---|
| LAM code | Apache-2.0 (the repo's label) | used as is, from a checkout |
| LAM weights (`lam-20k`) | CC BY-NC 4.0 | used; credit them wherever a head is shown |
| VHAP tracker, vendored in LAM (`vhap/`) | CC BY-NC-SA 4.0 | used as is (never copied into this repo) |
| FLAME's academic models (2017 to 2023) | research only, not to be made available | **replaced** by FLAME 2023 Open (CC BY 4.0) |
| FLAME's skinning code in LAM and VHAP (`lbs.py`, and its copies in `FlameHeadSubdivided`) | Max-Planck proprietary notice | **replaced** by `flame_core.py` (MIT, ours) |
| FLAME's painted mean texture (`tex_mean_painted.png`) | FLAME texture space, non-commercial | **replaced** by a flat grey (it is loaded but never fitted to) |
| `FLAME_masks.pkl`, `landmark_embedding_with_eyes.npy` | MPI downloads, research only | **still used, offline**: to be rebuilt on the Open mesh (#126) |

- `flame_core.py` is FLAME's maths written from the paper: blendshapes, pose correctives, a five-joint chain and linear blend skinning. It has the function names and signatures LAM calls. Checked against LAM's originals on random inputs in float64, it agrees to 1e-15.
- `overrides.py` registers `flame_core` under the names of LAM's and VHAP's skinning modules before LAM is imported. It then points the remaining helpers and `FlameHeadSubdivided`'s posing and skinning at it, and swaps the painted texture for grey. The LAM checkout is never edited.
- `make_head.py` stops before doing any work if the FLAME file is not FLAME 2023 Open (by sha256), or if `overrides.verify()` finds any LAM FLAME function still in use.

FLAME 2023 Open is the academic FLAME 2023 with new expression components (from SUPR). The template, faces, skinning weights, joints, pose correctives and identity shape space are identical. A head made with it looks the same as one made with the academic model ([#126](https://github.com/smk762/argus-mask/issues/126)). Its licence bars pornographic use and "fake, libelous, misleading or defamatory content": keep mask's disclosure label on, and feed it SFW portraits only.

## What it needs

- **A LAM checkout** (`LAM_DIR`) with its `model_zoo` (weights and tracking models, from LAM's README: `LAM_human_model.tar`, `LAM_assets.tar`, `lam-20k`).
- **FLAME 2023 Open** (`FLAME_OPEN`): `flame2023_Open.pkl` from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/download.php) (sign in, then *FLAME 2023 Open*). It is mounted read-only over both of LAM's `flame2023.pkl`.
- **The container image** (`LAM_IMAGE`): CUDA 12.1, torch 2.3, LAM's requirements, pytorch3d, the rasterisers and the FBX SDK wheel. That is the spike's `lam-spike:dev` (about 25 GB); its Dockerfile is not in this repo yet.
- **Blender 4.5** (`BLENDER`) for the FBX-to-GLB step and the vertex order.
- About 9 GB of VRAM and 2 minutes per head.

## Run it

```bash
lam/make_heads.sh my-head=portrait.png
```

The portrait should be frontal, neutral, evenly lit, head and shoulders ([docs/synthetic-faces.md](../docs/synthetic-faces.md)). The head is packed as `$HEADS/my-head/my-head.zip`. Add it to the assets with a manifest whose `model.format` is `lam-splat`, and give its provenance: every part above that the head depends on, with the portrait's own record.

The same portrait gives the same head, byte for byte.
