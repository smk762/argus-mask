#!/usr/bin/env bash
# Portraits to LAM Gaussian heads (.zip, the layout argus-mask serves as `lam-splat`). See README.md.
#   lam/make_heads.sh <name>=<portrait.png> ...
# Environment (defaults in brackets):
#   LAM_DIR      the aigc3d/LAM checkout with its model_zoo and assets [~/lam-spike/LAM]
#   FLAME_OPEN   flame2023_Open.pkl from flame.is.tue.mpg.de [~/Downloads/FLAME2023Open/flame2023_Open.pkl]
#   LAM_IMAGE    the container image, built from lam/Dockerfile [argus-mask-lam]
#   BLENDER      Blender 4.5 [~/argus-mask-head/blender/blender]
#   WORK         scratch for portraits, logs and LAM's output [~/lam-spike/work]
#   HEADS        where the packed heads go [~/mask-examples/heads]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LAM_DIR=${LAM_DIR:-~/lam-spike/LAM}
FLAME_OPEN=${FLAME_OPEN:-~/Downloads/FLAME2023Open/flame2023_Open.pkl}
LAM_IMAGE=${LAM_IMAGE:-argus-mask-lam}
BLENDER=${BLENDER:-~/argus-mask-head/blender/blender}
WORK=${WORK:-~/lam-spike/work}
HEADS=${HEADS:-~/mask-examples/heads}
[ -f "$FLAME_OPEN" ] || { echo "no FLAME 2023 Open at $FLAME_OPEN (set FLAME_OPEN)"; exit 1; }
mkdir -p "$WORK" "$HEADS"

# LAM's FBX-to-GLB converter, minus the one argument Blender 4.5's glTF exporter no longer takes.
convert="$WORK/convertFBX2GLB_45.py"
sed 's/export_colors=False,//' "$LAM_DIR/tools/convertFBX2GLB.py" > "$convert"

# Our landmark embedding (landmarks/embedding.json) in LAM's format, written by the image's numpy so LAM reads it.
docker run --rm -v "$here":/argus:ro -v "$WORK":/work "$LAM_IMAGE" \
  python /argus/landmarks/embedding.py npy /work/.landmark_embedding.npy > /dev/null
docker run --rm -v "$WORK":/work alpine:3.21 chown "$(id -u):$(id -g)" /work/.landmark_embedding.npy
embedding="$WORK/.landmark_embedding.npy"

flame=/lam/model_zoo/human_parametric_models
for pair in "$@"; do
  name=${pair%%=*}; img=${pair#*=}
  cp "$img" "$WORK/$name.png"
  out="$WORK/out-$name"
  # The container writes as root: clear and re-own its output through a throwaway container.
  docker run --rm -v "$WORK":/work alpine:3.21 rm -rf "/work/out-$name"
  docker run --rm --gpus all -e USER=lam -e PYTHONPATH=/lam -e IMG="/work/$name.png" -e OUT="/work/out-$name" \
    -e FIT_SHAPE -e FIT_DIMS -e FIT_GENERATIONS \
    -v "$LAM_DIR":/lam -v "$here":/argus:ro -v "$WORK":/work \
    -v "$FLAME_OPEN":$flame/flame_assets/flame/flame2023.pkl:ro -v "$FLAME_OPEN":$flame/flame_vhap/flame2023.pkl:ro \
    -v "$embedding":$flame/flame_assets/flame/landmark_embedding_with_eyes.npy:ro \
    -v "$embedding":$flame/flame_vhap/landmark_embedding_with_eyes.npy:ro \
    -w /lam "$LAM_IMAGE" python /argus/make_head.py > "$WORK/$name.log" 2>&1 || true
  docker run --rm -v "$WORK":/work alpine:3.21 chown -R "$(id -u):$(id -g)" "/work/out-$name"
  grep -q EXPORTED "$WORK/$name.log" || { echo "$name: LAM failed, see $WORK/$name.log"; tail -3 "$WORK/$name.log"; exit 1; }
  oac="$out/oac"; pkg="$HEADS/$name"; rm -rf "$pkg"; mkdir -p "$pkg/$name"
  "$BLENDER" --background --python "$convert" -- "$oac/temp_bin.fbx" "$pkg/$name/skin.glb" > "$WORK/$name.blender.log" 2>&1
  mesh=$(ls "$oac"/*.obj | head -1)
  (cd "$LAM_DIR" && "$BLENDER" --background --python tools/generateVertexIndices.py -- "$mesh" "$pkg/$name/vertex_order.json" \
    >> "$WORK/$name.blender.log" 2>&1)
  cp "$oac/offset.ply" "$oac/animation.glb" "$pkg/$name/"
  (cd "$pkg" && rm -f "$name.zip" && python3 -c 'import shutil, sys; shutil.make_archive(sys.argv[1], "zip", ".", sys.argv[1])' "$name")
  echo "$name: $pkg/$name.zip"
done
