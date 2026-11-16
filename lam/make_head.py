"""One portrait to a LAM Gaussian head, inside the LAM container (see README.md). Run from the LAM checkout.

    IMG=/work/portrait.png OUT=/work/out-name python /argus/make_head.py

Tracks FLAME to the portrait, runs LAM once, and exports the OpenAvatarChat layout (`oac/`: offset.ply, an ASCII
then binary FBX, animation.glb) that `make_heads.sh` packs into a mask face. Refuses to run unless the FLAME model
is FLAME 2023 Open, the landmark embedding is ours (`landmarks/embedding.json`), argus-mask's FLAME core has replaced
every Max-Planck-noticed function (`overrides.py`) and the tracker's front end is ours (`frontend.py`).
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overrides  # noqa: E402  (must run before anything from LAM is imported)

overrides.install()
import frontend  # noqa: E402
from landmarks import embedding  # noqa: E402

# FLAME 2023 Open (CC BY 4.0), as downloaded from flame.is.tue.mpg.de. The academic FLAME 2023 is a different file.
FLAME_OPEN_SHA256 = "e75a0990728ba038c7da2a420ae4396f7ddd7781366c13026066c5b24f127623"
FLAME_PATHS = (
    "model_zoo/human_parametric_models/flame_assets/flame/flame2023.pkl",
    "model_zoo/human_parametric_models/flame_vhap/flame2023.pkl",
)
for path in FLAME_PATHS:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != FLAME_OPEN_SHA256:
        sys.exit(f"{path} is not FLAME 2023 Open (sha256 {digest[:12]}…): mount flame2023_Open.pkl over it")

# Ours, measured on cleared heads for MediaPipe's landmarks (landmarks/measure.py), in place of MPI's.
EMBEDDING_PATHS = (
    "model_zoo/human_parametric_models/flame_assets/flame/landmark_embedding_with_eyes.npy",
    "model_zoo/human_parametric_models/flame_vhap/landmark_embedding_with_eyes.npy",
)
ours = embedding.load()
for path in EMBEDDING_PATHS:
    if not embedding.same(np.load(path, allow_pickle=True, encoding="latin1")[()], ours):
        sys.exit(f"{path} is not argus-mask's landmark embedding: mount the one embedding.py writes over it")

sys.argv = [sys.argv[0]]
os.environ.update(
    {
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
        "APP_TYPE": "infer.lam",
        "NUMBA_THREADING_LAYER": "omp",
    }
)
import torch  # noqa: E402
from app_lam import _build_model, parse_configs, save_images2video  # noqa: E402
from lam.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper  # noqa: E402
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image  # noqa: E402
from PIL import Image  # noqa: E402
from tools.flame_tracking_single_image import FlameTrackingSingleImage  # noqa: E402
from tools.generateARKITGLBWithBlender import convert_ascii_to_binary, update_flame_shape  # noqa: E402

overrides.patch()
frontend.install()
left = overrides.verify() + ([] if frontend.installed() else ["the tracker's front end"])
if left:
    sys.exit("LAM's own code is still in use: " + ", ".join(left))
print(
    "FLAME core: argus-mask's; FLAME model: 2023 Open; painted texture: grey; vertex masks: placeholders; "
    "landmarks: MediaPipe, our embedding; matte: BiRefNet"
)

image_in = os.environ.get("IMG", "/work/portrait.png")
motion = os.environ.get("MOTION", "Look_In_My_Eyes")
out_dir = os.environ.get("OUT", "/work/out")
os.makedirs(out_dir, exist_ok=True)

# The checkpoint carries the whole DINOv2 encoder, so skip the ImageNet download (its server's certificate expired).
_build_dinov2 = Dinov2FusionWrapper._build_dinov2
Dinov2FusionWrapper._build_dinov2 = staticmethod(
    lambda model_name, modulation_dim=None, pretrained=True: _build_dinov2(model_name, modulation_dim, pretrained=False)
)
cfg, _ = parse_configs()
lam = _build_model(cfg).to("cuda").eval()

tracker = FlameTrackingSingleImage(output_dir="output/tracking")  # frontend.py's constructor
raw = os.path.join(out_dir, "raw.png")
Image.open(image_in).convert("RGB").save(raw)
assert tracker.preprocess(raw) == 0, "tracking: preprocessing failed"
assert tracker.optimize() == 0, "tracking: FLAME fit failed"
rc, tracked = tracker.export()
assert rc == 0, "tracking: export failed"

image, _, _, shape_param = preprocess_image(
    os.path.join(tracked, "images/00000_00.png"),
    mask_path=os.path.join(tracked, "fg_masks/00000_00.png"),
    intr=None,
    pad_ratio=0,
    bg_color=1.0,
    max_tgt_size=None,
    aspect_standard=1.0,
    enlarge_ratio=[1.0, 1.0],
    render_tgt_size=cfg.source_size,
    multiply=14,
    need_mask=True,
    get_shape_param=True,
)
seq = prepare_motion_seqs(
    os.path.join("./assets/sample_motion/export", motion, "flame_param"),
    None,
    save_root=out_dir,
    fps=30,
    bg_color=1.0,
    aspect_standard=1.0,
    enlarge_ratio=[1.0, 1, 0],
    render_image_res=cfg.render_size,
    multiply=16,
    need_mask=False,
    vis_motion=False,
    shape_param=shape_param,
    test_sample=False,
    cross_id=False,
    src_driven=["portrait", motion],
)
seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
with torch.no_grad():
    res = lam.infer_single_view(
        image.unsqueeze(0).to("cuda"),
        None,
        None,
        render_c2ws=seq["render_c2ws"].to("cuda"),
        render_intrs=seq["render_intrs"].to("cuda"),
        render_bg_colors=seq["render_bg_colors"].to("cuda"),
        flame_params={k: v.to("cuda") for k, v in seq["flame_params"].items()},
    )

# A short preview of the head moving, to look at before packing.
rgb = res["comp_rgb"].detach().cpu().numpy()
mask = res["comp_mask"].detach().cpu().numpy()
mask[mask < 0.5] = 0
rgb = (np.clip(rgb * mask + (1 - mask), 0, 1) * 255).astype(np.uint8)
save_images2video(rgb, os.path.join(out_dir, "motion.mp4"), 30)
print("DONE", rgb.shape, "peak VRAM GB", round(torch.cuda.max_memory_allocated() / 1e9, 1))

# The ARKit avatar (the OpenAvatarChat layout): the shaped FLAME mesh into LAM's template, ASCII FBX to binary.
oac = Path(out_dir) / "oac"
oac.mkdir(exist_ok=True)
mesh_path = lam.renderer.flame_model.save_shaped_mesh(shape_param.unsqueeze(0).cuda(), fd=str(oac))
res["cano_gs_lst"][0].save_ply(str(oac / "offset.ply"), rgb2sh=False, offset2xyz=True)
update_flame_shape(Path(mesh_path), oac / "temp_ascii.fbx", Path("./assets/sample_oac/template_file.fbx"))
convert_ascii_to_binary(oac / "temp_ascii.fbx", oac / "temp_bin.fbx")
shutil.copy("./assets/sample_oac/animation.glb", oac / "animation.glb")
print("EXPORTED", oac)
