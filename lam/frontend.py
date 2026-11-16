"""The FLAME tracker's front end, from cleared models: crop, matte and landmarks for one portrait (#126).

LAM's single-image tracker (`tools/flame_tracking_single_image.py`) crops the head with VGGHeads, mattes it with
StyleMatte and finds 68 landmarks with FaceBoxes and STAR. StyleMatte and STAR carry no licence (and STAR was trained
on research-only data). `install()` replaces the tracker's constructor and `preprocess` so that:

- the crop is VGGHeads' head box, expanded and resized exactly as LAM does it (MIT code; CC BY-NC weights);
- the matte is BiRefNet-portrait (MIT), composited on white as LAM's was;
- the landmarks are MediaPipe Face Mesh's (Apache-2.0), the 68 of `landmarks/ibug68.py`, which
  `landmarks/embedding.json` places on FLAME. MediaPipe runs in its own environment (it needs numpy 2).

What it writes is what LAM's fit reads: `images/`, `mask/`, `alpha_maps/` and `landmark2d/landmarks.npz` under
`<output>/preprocess/<name>/`. The fit (`optimize`) and the export are LAM's, unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "landmarks"))
from ibug68 import MEDIAPIPE_68  # noqa: E402

MODELS = Path(os.environ.get("ARGUS_LAM_MODELS", "/opt/argus-models"))
MEDIAPIPE_PYTHON = os.environ.get("ARGUS_MEDIAPIPE_PYTHON", "/opt/mediapipe/bin/python")
CROP = 1024  # LAM's crop, square
BBOX_SCALE = 1.65  # LAM's expansion of VGGHeads' head box
FAILED_TO_DETECT = 1  # LAM's error code


class Matte:
    """BiRefNet-portrait: a portrait's alpha matte."""

    def __init__(self, path: Path, device: str):
        import torch
        from transformers import AutoModelForImageSegmentation

        self.torch, self.device = torch, device
        self.model = AutoModelForImageSegmentation.from_pretrained(str(path), trust_remote_code=True)
        self.model = self.model.to(device).eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]

    def __call__(self, rgb):
        """rgb (3, H, W) in [0, 1] -> alpha (H, W) in [0, 1]."""
        torch = self.torch
        x = rgb.to(self.device).float()
        h, w = x.shape[-2:]
        x = torch.nn.functional.interpolate(x[None], size=(1024, 1024), mode="bilinear", align_corners=False)[0]
        with torch.no_grad():
            alpha = self.model(((x - self.mean) / self.std)[None].half())[-1].sigmoid().float()
        return torch.nn.functional.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu()


def landmarks(image_path: str) -> np.ndarray | None:
    """MediaPipe's 478 points on one image, normalised to [0, 1]; None if it found no face."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "points.npz"
        subprocess.run(
            [
                MEDIAPIPE_PYTHON,
                str(HERE / "landmarks" / "detect.py"),
                str(MODELS / "face_landmarker.task"),
                str(out),
                image_path,
            ],
            check=True,
            capture_output=True,
        )
        points = np.load(out)["points"][0]
    return None if np.isnan(points[0, 0]) else points


def _init(self, output_dir, vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd", **_):
    from external.vgghead_detector import VGGHeadDetector

    self.output_dir = output_dir
    self.output_preprocess = os.path.join(output_dir, "preprocess")
    self.output_tracking = os.path.join(output_dir, "tracking")
    self.output_export = os.path.join(output_dir, "export")
    self.device = "cuda:0"
    self.vgghead_encoder = VGGHeadDetector(device=self.device, vggheadmodel_path=vgghead_model_path)
    self.matte = Matte(MODELS / "birefnet-portrait", self.device)
    self.detect_iris_landmarks_flag = False


def _preprocess(self, input_image_path):
    import cv2
    import torchvision
    from tools.flame_tracking_single_image import expand_bbox

    if not os.path.exists(input_image_path):
        return FAILED_TO_DETECT
    frame = torchvision.io.read_image(input_image_path)[:3, ...]
    try:
        _, bbox, _ = self.vgghead_encoder(frame, 0)
    except Exception:
        bbox = None
    if bbox is None:
        print("frontend: no head found", file=sys.stderr)
        return FAILED_TO_DETECT
    bbox = expand_bbox(bbox, scale=BBOX_SCALE).long()
    crop = torchvision.transforms.functional.crop(
        frame, top=bbox[1], left=bbox[0], height=bbox[3] - bbox[1], width=bbox[2] - bbox[0]
    )
    crop = torchvision.transforms.functional.resize(crop, (CROP, CROP), antialias=True)

    rgb = crop / 255.0
    alpha = self.matte(rgb)
    matted = rgb * alpha + (1 - alpha)  # on white, as LAM's matting returned it
    saved = np.round(matted.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)[:, :, ::-1]  # BGR for cv2

    self.sub_output_dir = os.path.join(self.output_preprocess, os.path.splitext(os.path.basename(input_image_path))[0])
    dirs = {k: os.path.join(self.sub_output_dir, k) for k in ("images", "mask", "alpha_maps", "landmark2d")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    image_path = os.path.join(dirs["images"], "00000.png")
    cv2.imwrite(image_path, saved)
    cv2.imwrite(os.path.join(dirs["mask"], "00000.png"), (alpha.numpy() * 255.0).astype(np.uint8))
    cv2.imwrite(os.path.join(dirs["alpha_maps"], "00000.jpg"), np.full_like(saved, 255))

    points = landmarks(image_path)
    if points is None:
        print("frontend: MediaPipe found no face in the crop", file=sys.stderr)
        return FAILED_TO_DETECT
    lmk = np.zeros((68, 3))
    lmk[:, :2] = points[list(MEDIAPIPE_68), :2]  # already in [0, 1] of the square crop
    np.savez(os.path.join(dirs["landmark2d"], "landmarks.npz"), bounding_box=[], face_landmark_2d=lmk[None])
    return 0


def install() -> None:
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    FlameTrackingSingleImage.__init__ = _init
    FlameTrackingSingleImage.preprocess = _preprocess


def installed() -> bool:
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    return FlameTrackingSingleImage.__init__ is _init and FlameTrackingSingleImage.preprocess is _preprocess
