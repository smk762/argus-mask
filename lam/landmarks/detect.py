"""MediaPipe Face Landmarker on images: all 478 points per image, normalised to [0, 1] (#126).

Runs in the image's MediaPipe environment (/opt/mediapipe), apart from LAM's: MediaPipe needs numpy 2, LAM 1.23.

    /opt/mediapipe/bin/python lam/landmarks/detect.py <model.task> <out.npz> <image> ...

Writes `names` (each image's file stem) and `points` (images, 478, 3): x and y in [0, 1] of the image's width and
height, z MediaPipe's relative depth. NaN where no face was found.
"""

import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

POINTS = 478


def detect(model: str, images: list[str]) -> np.ndarray:
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model), num_faces=1)
    out = np.full((len(images), POINTS, 3), np.nan)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, path in enumerate(images):
            found = landmarker.detect(mp.Image.create_from_file(path)).face_landmarks
            if found:
                out[i] = [(p.x, p.y, p.z) for p in found[0]]
    return out


if __name__ == "__main__":
    model, out, *images = sys.argv[1:]
    points = detect(model, images)
    np.savez(out, names=np.array([Path(p).stem for p in images]), points=points)
    print("DETECTED", int((~np.isnan(points[:, 0, 0])).sum()), "of", len(images))
