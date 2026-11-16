# argus-mask's head builder: the LAM environment (Dockerfile.base) plus the tracker's cleared front end (#126):
# BiRefNet-portrait for the matte (MIT) and MediaPipe Face Landmarker for the landmarks (Apache-2.0), each pinned.
#
#   docker build -t argus-mask-lam lam/
#
# MediaPipe needs numpy 2 and LAM numpy 1.23, so MediaPipe lives in its own environment (/opt/mediapipe) and
# frontend.py runs it as a subprocess.
ARG BASE=argus-mask-lam-base
FROM ${BASE}

# BiRefNet's model code wants timm and kornia; --no-deps keeps LAM's numpy and torch as they are.
RUN pip install --no-deps "timm==1.0.9" "kornia==0.7.3" "kornia_rs==0.1.5" \
 && python -c "import numpy; assert numpy.__version__ == '1.23.0', numpy.__version__"
RUN apt-get update && apt-get install -y --no-install-recommends python3.10-venv && rm -rf /var/lib/apt/lists/* \
 && python -m venv /opt/mediapipe && /opt/mediapipe/bin/pip install --no-cache-dir "mediapipe==0.10.14"

# The models, each checked against the hash it was tested with.
ENV ARGUS_LAM_MODELS=/opt/argus-models ARGUS_MEDIAPIPE_PYTHON=/opt/mediapipe/bin/python
ARG BIREFNET_REVISION=b6561965a70070d9143fd9e558f6ca3c481510db
RUN python - <<'PY'
import hashlib, os, urllib.request
root = "/opt/argus-models"
rev = os.environ.get("BIREFNET_REVISION", "b6561965a70070d9143fd9e558f6ca3c481510db")
files = {
    "birefnet-portrait/model.safetensors": (
        f"https://huggingface.co/ZhengPeng7/BiRefNet-portrait/resolve/{rev}/model.safetensors",
        "4a4eb3a5469b75f0cccaec6772c22fc30e6c12ed429c2c0ba71b43e3d8d97182",
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    ),
}
for name in ("birefnet.py", "BiRefNet_config.py", "config.json"):
    files[f"birefnet-portrait/{name}"] = (f"https://huggingface.co/ZhengPeng7/BiRefNet-portrait/resolve/{rev}/{name}", None)
for path, (url, sha) in files.items():
    dest = os.path.join(root, path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    got = hashlib.sha256(open(dest, "rb").read()).hexdigest()
    if sha and got != sha:
        raise SystemExit(f"{path}: sha256 {got}, want {sha}")
    print(path, got[:16])
PY
