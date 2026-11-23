#!/usr/bin/env bash
# Measure #173's Gaussian refine on three heads: likeness proxy (photometric) + optional
# likeness.py when MediaPipe is available. Writes work/refine-spike/<face>/ and a summary table.
#
#   lam/run_refine_spike.sh [--steps-scale 1] [--image argus-mask-lam]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-argus-mask-lam}
STEPS_SCALE=1
SEED=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --steps-scale) STEPS_SCALE=$2; shift 2 ;;
    --image) IMAGE=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT=$ROOT/work/refine-spike
mkdir -p "$OUT"
FACES=(splat-01 lam-s622 lam-c1022)

for face in "${FACES[@]}"; do
  dir=$ROOT/assets/$face
  zip=$(ls "$dir"/*.zip | head -1)
  portrait=$dir/source/portrait.png
  echo "=== refine $face ==="
  docker run --rm --gpus all \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$dir:/head:ro" \
    -v "$ROOT/lam:/argus:ro" \
    -v "$OUT:/out" \
    "$IMAGE" \
    python /argus/refine_gaussians.py \
      --zip "/head/$(basename "$zip")" \
      --portrait /head/source/portrait.png \
      --out "/out/$face" \
      --seed "$SEED" \
      --steps-scale "$STEPS_SCALE" \
      --repack
done

python3 - <<'PY' "$OUT" "${FACES[@]}"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
faces = sys.argv[2:]
rows = []
for face in faces:
    rec = json.loads((out / face / "refine.json").read_text())
    hist = rec["history"]
    first = next(h for h in hist if h["stage"] == "colour" and h["step"] == 1)
    last = hist[-1]
    rows.append(
        {
            "face": face,
            "photo_before": first["photo"],
            "photo_after": last["photo"],
            "delta_photo": last["photo"] - first["photo"],
            "pull": rec["start_pull"],
            "ply_sha256": rec["ply_sha256"][:12],
            "count": rec["gaussian_count"],
        }
    )
(out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
