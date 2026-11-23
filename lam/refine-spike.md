# Gaussian refine spike (#173)

Optimise a packed LAM head's `offset.ply` so a frontal render matches its portrait, with a pull back to LAM's own attributes. Script: [`refine_gaussians.py`](refine_gaussians.py). Driver: [`run_refine_spike.sh`](run_refine_spike.sh).

## Method

- **Rasteriser:** `diff_gaussian_rasterization` in `argus-mask-lam` (looks down **+Z** in view space).
- **Camera:** aim at the Gaussian centroid; distance fits the head at 23° FOV (viewer long lens).
- **Loss:** face-masked L1 vs the portrait (resized) + `λ=2` L2 pull to the original attributes.
- **Stages (seeded Adam):** colour (`f_dc_*`) → opacity + scale → small xyz (hard-capped).
- **Constraints held:** same Gaussian count/order (20018), same PLY property layout, normals unchanged; only portrait + LAM output as inputs.

Likeness (#169) on these spike `start.png` / `best.png` frames failed: MediaPipe found no face (the spike camera is a centroid frontal, not LAM’s white-backdrop portrait crop). Turns (#170) and speech (#171) were **not** re-measured on a live viewer (needs a server with the refined zip installed). The pull and position caps are there so off-axis and motion stay close to LAM; that still needs a real check before any default-on path.

## Numbers (seed 0, full stages)

| Face | Style | Photo L1 before | Photo L1 after | Δ photo | Mean pull (attrs) |
|---|---|---:|---:|---:|---:|
| `splat-01` | photo | 0.779 | 0.258 | **−0.520** | 0.0029 |
| `lam-s622` | photo | 0.980 | 0.370 | **−0.610** | 0.0033 |
| `lam-c1022` | character | 0.694 | 0.244 | **−0.450** | 0.0019 |

Outputs under `work/refine-spike/<face>/`: `start.png`, `best.png`, `offset.ply`, `refine.json`, repacked zip. Summary: `work/refine-spike/summary.json`.

## Decision

**Do not add a guided `STEPS` entry or default `FIT_REFINE` yet.**

Frontal photometric error drops a lot on all three heads with a small attribute pull, so the lever is real. Without turns/speech (and without #169 likeness on MediaPipe-aligned crops) we cannot clear the issue's bar for shipping it in the flow. Keep the spike scripts; treat refine as opt-in research (`lam/refine_gaussians.py`) until a follow-up measures turns + speech on the refined zips and, if still good, wires `FIT_REFINE=1` like `FIT_SHAPE`.

## Reproduce

```bash
./lam/run_refine_spike.sh --steps-scale 1
# or one head:
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$PWD/assets/splat-01:/head:ro" -v "$PWD/lam:/argus:ro" -v "$PWD/work/refine-spike:/out" \
  argus-mask-lam python /argus/refine_gaussians.py \
  --zip /head/splat-01.zip --portrait /head/source/portrait.png \
  --out /out/splat-01 --seed 0 --repack
```
