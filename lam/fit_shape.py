"""Fit a LAM head's FLAME shape to its portrait by what the head looks like, not by what the tracker guessed (#151).

The tracker fits FLAME's shape to landmarks on the portrait; LAM then builds its Gaussians on that shape, and its
prior pulls a face longer, narrower and heavier at the jaw: the head looks older than its portrait. Here the shape is
searched for the head whose render, at the portrait's own camera and pose, puts MediaPipe's landmarks where the
portrait has them. The score is the landmarks' distance in units of the eyes' spacing, the face's outline weighted
most; a small pull towards the tracker's shape keeps the search from wandering where the score can't see.

Pure numpy: ``make_head.py`` hands in the render and the landmark detector, so this file runs without LAM or torch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# MediaPipe Face Mesh indices.
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
        150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]  # fmt: skip
EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]  # fmt: skip
BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSE = [1, 2, 98, 327, 4, 5, 195, 197, 6, 168, 48, 278, 64, 294]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# The outline carries the shape (face length, cheeks, jaw); the features place it.
REGIONS = {"oval": (OVAL, 2.0), "eyes": (EYES, 1.0), "brows": (BROWS, 0.5), "nose": (NOSE, 1.0), "lips": (LIPS, 1.0)}
EYE_CORNERS = (33, 263)
PULL = 0.002  # per unit of shape, squared: the pull back towards the tracker's shape


def misfit(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Each region's RMS landmark distance, in units of the reference's eye spacing; ``total`` their weighted mean.

    Both are MediaPipe's points, x and y in [0, 1] of the same frame (the portrait's crop and the head's render at
    the portrait's camera). A candidate with no face found scores infinity.
    """
    if candidate is None or np.isnan(candidate[:, :2]).any():
        return {"total": float("inf")}
    ref, cand = reference[:, :2], candidate[:, :2]
    unit = float(np.linalg.norm(ref[EYE_CORNERS[0]] - ref[EYE_CORNERS[1]]))
    out, weighted, weights = {}, 0.0, 0.0
    for name, (idx, weight) in REGIONS.items():
        rms = float(np.sqrt(((ref[idx] - cand[idx]) ** 2).sum(axis=1).mean())) / unit
        out[name] = rms
        weighted += weight * rms
        weights += weight
    out["total"] = weighted / weights
    return out


def search(
    start: np.ndarray,
    score: Callable[[list[np.ndarray]], list[float]],
    dims: int = 20,
    sigma: float = 0.4,
    population: int = 8,
    generations: int = 20,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, float, list[dict]]:
    """A seeded (mu/mu, lambda) evolution strategy over the first ``dims`` shape components, from ``start``.

    ``score`` takes a batch of shapes and returns each one's misfit (lower is better); the pull towards ``start``
    is added here. Returns the best shape seen, its score, and one record per generation.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=np.float64)
    mean = start[:dims].copy()
    elite = max(1, population // 2)
    ranks = np.log(elite + 0.5) - np.log(np.arange(1, elite + 1))
    ranks /= ranks.sum()

    def full(head: np.ndarray) -> np.ndarray:
        shape = start.copy()
        shape[:dims] = head
        return shape

    def pulled(shapes: list[np.ndarray], raw: list[float]) -> list[float]:
        return [r + PULL * float(((s - start) ** 2).sum()) for s, r in zip(shapes, raw, strict=True)]

    best_shape = start.copy()
    best = pulled([start], score([start]))[0]
    history = [{"generation": 0, "best": best, "sigma": sigma}]
    log(f"start: {best:.4f}")
    for generation in range(1, generations + 1):
        steps = rng.standard_normal((population, dims))
        shapes = [full(mean + sigma * step) for step in steps]
        scores = pulled(shapes, score(shapes))
        order = np.argsort(scores)
        if scores[order[0]] < best:
            best, best_shape = float(scores[order[0]]), shapes[order[0]]
        mean = mean + sigma * (ranks[:, None] * steps[order[:elite]]).sum(axis=0)
        # A generation that improves on the mean keeps its step; one that doesn't shrinks it.
        sigma *= 1.1 if scores[order[0]] <= best else 0.85
        history.append({"generation": generation, "best": best, "sigma": sigma, "worst": float(scores[order[-1]])})
        log(f"generation {generation}: best {best:.4f}, this one {scores[order[0]]:.4f}..{scores[order[-1]]:.4f}, "
            f"sigma {sigma:.3f}")  # fmt: skip
    return best_shape, best, history
