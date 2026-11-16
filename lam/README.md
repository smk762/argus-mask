# LAM heads

A portrait in, a Gaussian-splat head out: the `lam-splat` faces that mask serves and records ([#66](https://github.com/smk762/argus-mask/issues/66)). This directory runs [LAM](https://github.com/aigc3d/LAM) (Alibaba's one-shot animatable head) on one NVIDIA GPU. It uses argus-mask's own FLAME code, FLAME 2023 Open, our own landmark embedding and a cleared tracker front end, so that a head made from a cleared portrait can go on the hosted demo ([#126](https://github.com/smk762/argus-mask/issues/126)).

It is not part of the `argus-mask` Python package, it is not linted or tested by CI, and it runs offline: the server never needs a GPU.

## Licences, and what this directory changes

| Part | Licence | Here |
|---|---|---|
| LAM code | Apache-2.0 (the repo's label) | used as is, from a checkout |
| LAM weights (`lam-20k`) | CC BY-NC 4.0 | used; credit them wherever a head is shown |
| VHAP tracker, vendored in LAM (`vhap/`): the FLAME fit | CC BY-NC-SA 4.0 | used as is (never copied into this repo) |
| VGGHeads head detector (LAM's `external/vgghead_detector`) | MIT code, CC BY-NC 4.0 weights | used for the crop; credit it |
| FLAME's academic models (2017 to 2023) | research only, not to be made available | **replaced** by FLAME 2023 Open (CC BY 4.0) |
| FLAME's skinning code in LAM and VHAP (`lbs.py`, and its copies in `FlameHeadSubdivided`) | Max-Planck proprietary notice | **replaced** by `flame_core.py` (MIT, ours) |
| FLAME's painted mean texture (`tex_mean_painted.png`) | FLAME texture space, non-commercial | **replaced** by a flat grey (it is loaded but never fitted to) |
| FLAME's vertex masks (`FLAME_masks.pkl`) | MPI download, research only | **replaced** by placeholders (they feed only photometric tracking, which is off) |
| FLAME's landmark embedding (`landmark_embedding_with_eyes.npy`) | MPI research data (via DECA/VHAP) | **replaced** by `landmarks/embedding.json` (ours, measured on cleared heads) |
| STAR landmarks with FaceBoxes (`68_keypoints_model.pkl`, `FaceBoxesV2.pth`) | STAR: no licence, trained on 300W (research only) | **replaced** by MediaPipe Face Landmarker (Apache-2.0) |
| StyleMatte matting (`stylematte_synth.pt`) | no licence | **replaced** by BiRefNet-portrait (MIT) |

- `flame_core.py` is FLAME's maths written from the paper: blendshapes, pose correctives, a five-joint chain and linear blend skinning. It has the function names and signatures LAM calls. Checked against LAM's originals on random inputs in float64, it agrees to 1e-15.
- `overrides.py` registers `flame_core` under the names of LAM's and VHAP's skinning modules before LAM is imported. It then points the remaining helpers and `FlameHeadSubdivided`'s posing and skinning at it. It swaps the painted texture for grey and FLAME's vertex masks for placeholders, and refuses photometric tracking, the only stage that would read either. The LAM checkout is never edited.
- `frontend.py` replaces the tracker's constructor and `preprocess`. The crop is LAM's (VGGHeads' head box, expanded 1.65 times, 1024 px); the matte is BiRefNet's, on white as before; the 68 landmarks are MediaPipe's (`landmarks/ibug68.py`). MediaPipe needs numpy 2, so it runs in its own environment in the image. LAM's fit and export are unchanged.
- `make_head.py` stops before doing any work unless the FLAME file is FLAME 2023 Open (by sha256), the mounted landmark embedding is ours, `overrides.verify()` finds no LAM FLAME function still in use, and the front end is ours.

Checked on #126: a head built with garbage bytes mounted over STAR's, FaceBoxes', StyleMatte's and the vertex masks' files builds, and looks the same as one built with LAM's own tracker.

FLAME 2023 Open is the academic FLAME 2023 with new expression components (from SUPR). The template, faces, skinning weights, joints, pose correctives and identity shape space are identical. Its licence bars pornographic use and "fake, libelous, misleading or defamatory content": keep mask's disclosure label on, and feed it SFW portraits only.

## The landmark embedding

The tracker fits FLAME to a portrait through 68 landmarks, so it must know where each lies on FLAME's surface. `landmarks/embedding.json` says that, for MediaPipe's points: a FLAME face and barycentric weights per landmark, plus the two pupils. It was measured on cleared material only (`landmarks/measure.py`):

1. `render_head.py` (Blender) renders mask's MakeHuman heads from 15 views. The heads are `stock-01` and `stock-02`, and four synthetic editor presets.
2. `detect.py` finds MediaPipe's landmarks in each render, and each is cast back onto the head's surface.
3. FLAME 2023 Open is fitted to each head's visible surface (identity, a little expression, jaw and a similarity transform; eye centres and nose tip pinned). The fit comes to 2.3 to 2.4 mm RMS.
4. Each landmark is carried to the fitted FLAME surface, and the median over the heads is placed on FLAME's template.

Across the six heads, landmarks agree with the consensus to a median of 0.6 mm (eyes), 1.1 mm (brows), 1.2 mm (nose) and 1.8 mm (lips). The jaw is looser (4.7 mm, worse at its angle, where the surface folds), as it is in any 68-point fit. To measure it again:

```bash
blender --background --python lam/landmarks/render_head.py -- assets/<head>/head.glb <dir>/<head>
docker run --rm -v $PWD/lam:/argus:ro -v <dir>:/d argus-mask-lam \
  bash -c '/opt/mediapipe/bin/python /argus/landmarks/detect.py $ARGUS_LAM_MODELS/face_landmarker.task /d/<head>/mediapipe.npz /d/<head>/*.png'
docker run --rm --gpus all -v $PWD/lam:/argus:ro -v <dir>:/d -v <flame dir>:/flame:ro argus-mask-lam \
  python /argus/landmarks/measure.py /flame/flame2023_Open.pkl /d/embedding.json /d/<head> ...
```

## What it needs

- **A LAM checkout** (`LAM_DIR`) with its `model_zoo` (weights and the VGGHeads model, from LAM's README: `LAM_human_model.tar`, `LAM_assets.tar`, `lam-20k`).
- **FLAME 2023 Open** (`FLAME_OPEN`): `flame2023_Open.pkl` from [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/download.php) (sign in, then *FLAME 2023 Open*). It is mounted read-only over both of LAM's `flame2023.pkl`.
- **The image** (`LAM_IMAGE`, default `argus-mask-lam`). `Dockerfile.base` is LAM's environment (CUDA 12.1, torch 2.3, LAM's requirements, pytorch3d, the rasterisers and the FBX SDK wheel, which you supply; about 25 GB). `Dockerfile` adds the front end's models on top, each pinned by revision or sha256:

  ```bash
  docker build -f lam/Dockerfile.base -t argus-mask-lam-base --build-context lam=$LAM_DIR --build-context fbx=<wheel dir> lam/
  docker build -t argus-mask-lam lam/
  ```
- **Blender 4.5** (`BLENDER`) for the FBX-to-GLB step and the vertex order.
- About 9 GB of VRAM and 2 minutes per head.

## Run it

```bash
lam/make_heads.sh my-head=portrait.png
```

The portrait should be frontal, neutral, evenly lit, head and shoulders ([docs/synthetic-faces.md](../docs/synthetic-faces.md)). The head is packed as `$HEADS/my-head/my-head.zip`. Add it to the assets with a manifest whose `model.format` is `lam-splat`, and give its provenance: every part above that the head depends on, with the portrait's own record.

The same portrait gives the same head, byte for byte.
