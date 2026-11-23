"""One portrait to a LAM Gaussian head, inside the LAM container (see README.md). Run from the LAM checkout.

    IMG=/work/portrait.png OUT=/work/out-name python /argus/make_head.py

Tracks FLAME to the portrait, runs LAM once, and exports the OpenAvatarChat layout (`oac/`: offset.ply, an ASCII
then binary FBX, animation.glb) that `make_heads.sh` packs into a mask face. Refuses to run unless the FLAME model
is FLAME 2023 Open, the landmark embedding is ours (`landmarks/embedding.json`), argus-mask's FLAME core has replaced
every Max-Planck-noticed function (`overrides.py`) and the tracker's front end is ours (`frontend.py`).
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overrides  # noqa: E402  (must run before anything from LAM is imported)

overrides.install()
import frontend  # noqa: E402
import preflight  # noqa: E402

if refused := preflight.problems():
    sys.exit("\n".join(refused))

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


def portrait_view(shape_param, tracked, save_root):
    """The portrait's own camera, pose and expression (the tracker's), for a head of this shape."""
    return prepare_motion_seqs(
        os.path.join(tracked, "flame_param"), None, save_root=str(save_root), fps=30, bg_color=1.0,
        aspect_standard=1.0, enlarge_ratio=[1.0, 1, 0], render_image_res=cfg.render_size, multiply=16,
        need_mask=False, vis_motion=False, shape_param=shape_param, test_sample=False, cross_id=False,
        src_driven=["", ""],
    )  # fmt: skip


def render_at(view, image, shape, path):
    """The head of this shape as *view* sees it, on white."""
    import copy

    params = copy.deepcopy(dict(view["flame_params"]))
    params["betas"] = torch.as_tensor(shape, dtype=torch.float32)[None]
    with torch.no_grad():
        res = lam.infer_single_view(
            image.unsqueeze(0).to("cuda"), None, None,
            render_c2ws=view["render_c2ws"].to("cuda"), render_intrs=view["render_intrs"].to("cuda"),
            render_bg_colors=view["render_bg_colors"].to("cuda"),
            flame_params={k: v.to("cuda") for k, v in params.items()},
        )  # fmt: skip
    rgb, mask = res["comp_rgb"][0].cpu().numpy(), res["comp_mask"][0].cpu().numpy()
    Image.fromarray((np.clip(rgb * mask + (1 - mask), 0, 1) * 255).astype(np.uint8)).save(path)


def fit_portrait_shape(image, shape_param, tracked):
    """FIT_SHAPE=1: search the shape for the head that looks most like its portrait (fit_shape.py). Writes
    ``fit/`` beside the head: the start's and the best's renders at the portrait's camera, and fit.json."""
    import json
    import subprocess
    import tempfile

    import fit_shape

    fit_dir = Path(out_dir) / "fit"
    fit_dir.mkdir(exist_ok=True)
    source = portrait_view(shape_param, tracked, fit_dir)
    detector = Path(frontend.__file__).parent / "landmarks" / "detect.py"

    def detect(paths):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "points.npz"
            subprocess.run(
                [frontend.MEDIAPIPE_PYTHON, str(detector), str(frontend.MODELS / "face_landmarker.task"), str(out), *paths],
                check=True, capture_output=True,
            )  # fmt: skip
            return np.load(out)["points"]

    def render(shape, path):
        render_at(source, image, shape, path)

    reference = detect([os.path.join(tracked, "images/00000_00.png")])[0]
    if np.isnan(reference[0, 0]):
        sys.exit("fit: MediaPipe found no face in the portrait")
    count = [0]

    def score(shapes):
        paths = []
        for shape in shapes:
            paths.append(str(fit_dir / f"try-{count[0]:03d}.png"))
            render(shape, paths[-1])
            count[0] += 1
        return [fit_shape.misfit(reference, points)["total"] for points in detect(paths)]

    start = shape_param.cpu().numpy().astype(np.float64)
    best, best_score, history = fit_shape.search(
        start, score, dims=int(os.environ.get("FIT_DIMS", 20)), generations=int(os.environ.get("FIT_GENERATIONS", 20))
    )
    render(start, fit_dir / "start.png")
    render(best, fit_dir / "best.png")
    points = detect([str(fit_dir / "start.png"), str(fit_dir / "best.png")])
    report = {
        "start": fit_shape.misfit(reference, points[0]),
        "best": fit_shape.misfit(reference, points[1]),
        "best_score": best_score,
        "history": history,
        "start_shape": start.tolist(),
        "best_shape": best.tolist(),
    }
    (fit_dir / "fit.json").write_text(json.dumps(report, indent=1))
    for path in fit_dir.glob("try-*.png"):
        path.unlink()
    print("FIT", json.dumps({k: report[k] for k in ("start", "best")}))
    return torch.as_tensor(best, dtype=shape_param.dtype)


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
# Explicit checks, not assert: under python -O the steps themselves would be skipped.
if tracker.preprocess(raw) != 0:
    sys.exit("tracking: preprocessing failed")
if tracker.optimize() != 0:
    sys.exit("tracking: FLAME fit failed")
rc, tracked = tracker.export()
if rc != 0:
    sys.exit("tracking: export failed")

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
if os.environ.get("FIT_SHAPE"):
    shape_param = fit_portrait_shape(image, shape_param, tracked)

# What the likeness is scored on (likeness.py, #169): the portrait as LAM was given it (the tracker's crop, matted
# on white) and the head as the portrait's own camera sees it, in the portrait's pose and expression.
like_dir = Path(out_dir) / "likeness"
like_dir.mkdir(exist_ok=True)
crop = np.asarray(Image.open(os.path.join(tracked, "images/00000_00.png")).convert("RGB"), dtype=np.float32) / 255
matte = np.asarray(Image.open(os.path.join(tracked, "fg_masks/00000_00.png")).convert("L"), dtype=np.float32) / 255
Image.fromarray(((crop * matte[..., None] + (1 - matte[..., None])) * 255).round().astype(np.uint8)).save(
    like_dir / "portrait.png"
)
render_at(portrait_view(shape_param, tracked, like_dir), image, shape_param.cpu().numpy(), like_dir / "render.png")
print("LIKENESS", like_dir)

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
