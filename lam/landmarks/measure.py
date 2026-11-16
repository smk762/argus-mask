"""Measure where MediaPipe's 68 landmarks lie on FLAME 2023 Open: argus-mask's landmark embedding (#126).

The FLAME tracker fits FLAME to a portrait through 68 landmarks plus two pupils. It needs to know where each one sits
on FLAME's surface: a triangle and barycentric weights, the "landmark embedding". LAM ships MPI's (research only).
This measures our own from cleared material only, with no MPI file involved:

1. `render_head.py` renders mask's stock heads (MakeHuman, CC0) from 15 views and saves their surfaces.
2. `detect.py` finds MediaPipe's landmarks in every render.
3. Here, each landmark of `ibug68.MEDIAPIPE_68` is cast as a ray back onto the head's surface, view by view, and the
   hits' median is its point on the head.
4. FLAME 2023 Open is fitted to the head's surface: identity and expression components, jaw, and a similarity
   transform, matching the two surfaces over the face (both directions) with the eyeball centres pinned.
5. Each landmark goes to the closest point of the fitted FLAME surface, kept as triangle and barycentric weights,
   which are the same on every FLAME mesh. Over several heads, the points are averaged on FLAME's template.
6. The pupils (68, 69) are the front poles of FLAME's eyeballs, from the geometry alone.

    python lam/landmarks/measure.py <flame2023_Open.pkl> <out.json> <head dir> ...

Each head dir holds `scene.npz` (step 1) and `mediapipe.npz` (step 2). Runs in LAM's environment (torch, pytorch3d).
The output is `embedding.json` (`embedding.py` turns it into LAM's `.npy`).
"""

from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch3d.ops import knn_points

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import flame_core  # noqa: E402
from ibug68 import MEDIAPIPE_68  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64
N_SHAPE, N_EXPR = 100, 50
STEPS = 1500
FACE_RADIUS = 0.085  # metres around the face's centre that both surfaces must match
REGIONS = {
    "jaw": range(0, 17),
    "brows": range(17, 27),
    "nose": range(27, 36),
    "eyes": range(36, 48),
    "lips": range(48, 68),
}


# ── FLAME ───────────────────────────────────────────────────────────────────────────────────────────────────────


class Flame:
    """FLAME 2023 Open: template, components, skinning; forward with identity, expression and jaw."""

    def __init__(self, path: str):
        with open(path, "rb") as fh:
            m = flame_core.Struct(**pickle.load(fh, encoding="latin1"))
        t = lambda a, dt=DT: flame_core.to_tensor(flame_core.to_np(a, np.float64), dt).to(DEVICE)  # noqa: E731
        self.template = t(m.v_template)
        self.faces = t(m.f, torch.long)
        self.shapedirs = t(m.shapedirs)
        self.posedirs = t(np.reshape(flame_core.to_np(m.posedirs, np.float64), (-1, m.posedirs.shape[-1])).T)
        self.J_regressor = t(m.J_regressor)
        self.weights = t(m.weights)
        parents = flame_core.to_np(m.kintree_table[0], np.int64)
        parents[0] = -1
        self.parents = torch.as_tensor(parents, device=DEVICE)
        template_x = flame_core.to_np(m.v_template, np.float64)[:, 0]
        # (right, left): the subject's right is at x < 0.
        self.eyeballs = tuple(sorted(eyeball_components(m.f, len(m.v_template)), key=lambda c: template_x[c].mean()))
        eye_vids = np.concatenate(self.eyeballs)
        keep = ~np.isin(flame_core.to_np(m.f, np.int64), eye_vids).any(1)
        self.skin_faces = self.faces[torch.as_tensor(keep, device=DEVICE)]

    def forward(self, shape, expr, jaw):
        betas = torch.zeros(1, 400, dtype=DT, device=DEVICE)
        betas[:, : shape.numel()] = shape
        betas[:, 300 : 300 + expr.numel()] = expr
        v_shaped = self.template[None] + flame_core.blend_shapes(betas, self.shapedirs)
        pose = torch.zeros(1, 15, dtype=DT, device=DEVICE)
        pose[:, 6:9] = jaw
        verts, _, _ = flame_core.lbs(
            pose, v_shaped, self.posedirs, self.J_regressor, self.parents, self.weights, dtype=DT
        )
        return verts[0]


def eyeball_components(faces, n_verts):
    """FLAME's two eyeballs: the mesh's two separate components of a few hundred vertices (546 each)."""
    parent = np.arange(n_verts)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2])):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(n_verts)])
    comps = [np.flatnonzero(roots == r) for r in np.unique(roots)]
    eyes = sorted((c for c in comps if 300 < len(c) < 1000), key=len)[-2:]
    if len(eyes) != 2:
        raise SystemExit("FLAME's eyeballs not found as two separate components")
    return eyes


# ── geometry ────────────────────────────────────────────────────────────────────────────────────────────────────


def rays(scene, view: int, uv: np.ndarray):
    """World-space rays (origins, unit directions) through normalised image points of one rendered view."""
    cam_to_world = np.linalg.inv(scene["world_to_cam"][view])
    half = math.tan(float(scene["fov_y"]) / 2)
    d = np.stack([(2 * uv[:, 0] - 1) * half, (1 - 2 * uv[:, 1]) * half, -np.ones(len(uv))], 1)
    d = d @ cam_to_world[:3, :3].T
    return np.broadcast_to(cam_to_world[:3, 3], d.shape), d / np.linalg.norm(d, axis=1, keepdims=True)


def ray_hits(origins, dirs, tris, chunk=1024, with_triangle=False):
    """Nearest hit of each ray on triangles (T, 3, 3), Möller-Trumbore; NaN where a ray misses.

    With `with_triangle`, also each hit's triangle index (-1 for a miss).
    """
    a, b, c = (torch.as_tensor(np.ascontiguousarray(tris[:, i]), device=DEVICE, dtype=torch.float32) for i in range(3))
    e1, e2 = b - a, c - a
    hits, which = [], []
    for start in range(0, len(origins), chunk):
        o = torch.as_tensor(np.ascontiguousarray(origins[start : start + chunk]), device=DEVICE, dtype=torch.float32)
        d = torch.as_tensor(np.ascontiguousarray(dirs[start : start + chunk]), device=DEVICE, dtype=torch.float32)
        p = torch.cross(d[:, None], e2[None].expand(len(d), -1, -1), dim=-1)
        det = (e1[None] * p).sum(-1)
        inv = 1 / torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
        s = o[:, None] - a[None]
        u = (s * p).sum(-1) * inv
        q = torch.cross(s, e1[None].expand_as(s), dim=-1)
        v = (d[:, None] * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        ok = (det.abs() > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        t = torch.where(ok, t, torch.full_like(t, math.inf))
        best, idx = t.min(1)
        hit = (o + d * best[:, None]).double()
        miss = torch.isinf(best)
        hit[miss] = math.nan
        idx[miss] = -1
        hits.append(hit.cpu().numpy())
        which.append(idx.cpu().numpy())
    hits, which = np.concatenate(hits), np.concatenate(which)
    return (hits, which) if with_triangle else hits


def visible_points(scene, n_head: int, grid: int = 96):
    """The head's surface as the cameras see it: hits of a grid of rays from every view, on the head mesh only.

    Cavities (the mouth's inside, the eye sockets) and the far side never enter the fit this way.
    """
    tris = scene["verts"][scene["faces"]]
    head_tri = (scene["faces"] < n_head).all(1)
    u = (np.arange(grid) + 0.5) / grid
    uv = np.stack(np.meshgrid(u, u), -1).reshape(-1, 2)
    out = []
    for v in range(len(scene["views"])):
        o, d = rays(scene, v, uv)
        hit, tri = ray_hits(np.array(o), d, tris, with_triangle=True)
        keep = tri >= 0
        keep[keep] = head_tri[tri[keep]]
        out.append(hit[keep])
    return np.concatenate(out)


def closest_on_triangles(points, verts, faces, k=64):
    """For each point (P, 3), the closest point on a mesh: (face index (P,), barycentric weights (P, 3))."""
    tri = verts[faces]  # (F, 3, 3)
    centroids = tri.mean(1)
    near = knn_points(points[None].float(), centroids[None].float(), K=min(k, len(faces))).idx[0]  # (P, k)
    cand = tri[near]  # (P, k, 3, 3)
    bary = _closest_bary(points[:, None].expand(-1, near.shape[1], -1), cand)
    closest = (bary[..., None] * cand).sum(-2)
    dist = (closest - points[:, None]).norm(dim=-1)
    pick = dist.argmin(1)
    rows = torch.arange(len(points), device=points.device)
    return near[rows, pick], bary[rows, pick]


def _closest_bary(p, tri):
    """Barycentric weights of the closest point to p on each triangle (Ericson, Real-Time Collision Detection 5.1.5)."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp, cp = p - b, p - c
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    one, zero = torch.ones_like(d1), torch.zeros_like(d1)
    denom = va + vb + vc
    denom = torch.where(denom.abs() < 1e-30, one, denom)
    v_in, w_in = vb / denom, vc / denom
    out = torch.stack([1 - v_in - w_in, v_in, w_in], -1)  # inside the face

    def setw(mask, w):
        nonlocal out
        out = torch.where(mask[..., None], w, out)

    # Edges, then vertices: later regions override earlier ones, in Ericson's order reversed.
    t_bc = (d4 - d3) / torch.where(((d4 - d3) + (d5 - d6)).abs() < 1e-30, one, (d4 - d3) + (d5 - d6))
    setw((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), torch.stack([zero, 1 - t_bc, t_bc], -1))
    t_ac = d2 / torch.where((d2 - d6).abs() < 1e-30, one, d2 - d6)
    setw((vb <= 0) & (d2 >= 0) & (d6 <= 0), torch.stack([1 - t_ac, zero, t_ac], -1))
    t_ab = d1 / torch.where((d1 - d3).abs() < 1e-30, one, d1 - d3)
    setw((vc <= 0) & (d1 >= 0) & (d3 <= 0), torch.stack([1 - t_ab, t_ab, zero], -1))
    setw((d6 >= 0) & (d5 <= d6), torch.stack([zero, zero, one], -1))
    setw((d3 >= 0) & (d4 <= d3), torch.stack([zero, one, zero], -1))
    setw((d1 <= 0) & (d2 <= 0), torch.stack([one, zero, zero], -1))
    return out


def sample_surface(verts, faces, n, seed=0):
    """n points spread over a mesh by area, from a fixed seed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tri = verts[faces]
    area = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    pick = torch.multinomial(area.cpu().float(), n, replacement=True, generator=g).to(verts.device)
    r = torch.rand(n, 2, generator=g, dtype=DT).to(verts.device)
    s = r[:, 0].sqrt()
    w = torch.stack([1 - s, s * (1 - r[:, 1]), s * r[:, 1]], 1)
    return (w[..., None] * tri[pick]).sum(1)


def axis_angle_matrix(r):
    return flame_core.batch_rodrigues(r[None], dtype=DT)[0]


# ── one head ────────────────────────────────────────────────────────────────────────────────────────────────────


def blender_to_flame(p):
    """Blender's Z-up, face toward -Y, to FLAME's Y-up, face toward +Z."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def head_landmarks(head: Path):
    """The 68 landmarks on one head's surface (68, 3, FLAME axes), each the median of its hits over the views."""
    scene, found = np.load(head / "scene.npz"), np.load(head / "mediapipe.npz")
    tris = scene["verts"][scene["faces"]]
    order = {n: i for i, n in enumerate(found["names"])}
    hits = []
    for v, name in enumerate(scene["views"]):
        pts = found["points"][order[str(name)]]
        if np.isnan(pts[0, 0]):
            continue
        o, d = rays(scene, v, pts[list(MEDIAPIPE_68), :2])
        hits.append(ray_hits(o, d, tris))
    hits = np.stack(hits)  # (views, 68, 3)
    seen = (~np.isnan(hits[..., 0])).sum(0)
    if seen.min() < 3:
        raise SystemExit(f"{head}: landmark {int(seen.argmin())} hit the head in only {int(seen.min())} views")
    return blender_to_flame(np.nanmedian(hits, axis=0)), seen


# The fit's stages: (steps, what is free). Pose and scale first, then identity, then a little expression and jaw.
STAGES = ((400, ("pose",)), (1200, ("pose", "shape")), (600, ("pose", "shape", "expr")))
TRUNCATE = 0.015  # metres: surface pairs further apart than this are ignored (hair line, neck cut, ears)
PRIOR = {"shape": 5e-8, "expr": 1e-7, "jaw": 1e-4}  # weights on the squared coefficients, against squared metres


def nose_tip(points, eyes):
    """The nose's tip: the front-most point on the midline, between 1.5 and 6 cm below the eyes (FLAME axes)."""
    y = eyes[:, 1].mean()
    band = (points[:, 0].abs() < 0.01) & (points[:, 1] < y - 0.015) & (points[:, 1] > y - 0.06)
    cand = torch.nonzero(band)[:, 0]
    return cand[torch.argmax(points[cand, 2])]


def fit(flame: Flame, head: Path, log=print):
    """FLAME fitted to a head's surface: (FLAME vertices, transform head->FLAME as a function, residual in mm).

    The head's surface is what the cameras saw of it (`visible_points`). Its eye centres and nose tip pin FLAME's,
    so the face cannot slide; the surfaces are matched both ways over the face, robustly (`TRUNCATE`).
    """
    scene = np.load(head / "scene.npz")
    n_head = int(scene["part_verts"][list(scene["parts"]).index("head")])
    verts = torch.as_tensor(blender_to_flame(scene["verts"]), device=DEVICE)
    eye_v = verts[n_head:]
    mh_eyes = torch.stack([eye_v[eye_v[:, 0] < 0].mean(0), eye_v[eye_v[:, 0] >= 0].mean(0)])  # right, left
    mh_points = torch.as_tensor(blender_to_flame(visible_points(scene, n_head)), device=DEVICE)

    fl_eye_ids = [torch.as_tensor(e, device=DEVICE) for e in flame.eyeballs]
    skin_ids = torch.unique(flame.skin_faces)

    # Start: match the eyes' centres (scale and position); both faces already look down +Z.
    fl0 = flame.template
    fl_eyes0 = torch.stack([fl0[i].mean(0) for i in fl_eye_ids])
    scale0 = (fl_eyes0[1] - fl_eyes0[0]).norm() / (mh_eyes[1] - mh_eyes[0]).norm()
    shift0 = fl_eyes0.mean(0) - scale0 * mh_eyes.mean(0)
    fl_nose = skin_ids[nose_tip(fl0[skin_ids], fl_eyes0)]  # a FLAME vertex id
    mh0 = scale0 * mh_points + shift0
    mh_nose = mh_points[nose_tip(mh0, fl_eyes0)]  # a point on the head, in its own frame

    params = {
        "log_s": torch.zeros(1, dtype=DT, device=DEVICE),
        "rot": torch.zeros(3, dtype=DT, device=DEVICE),
        "shift": torch.zeros(3, dtype=DT, device=DEVICE),
        "shape": torch.zeros(N_SHAPE, dtype=DT, device=DEVICE),
        "expr": torch.zeros(N_EXPR, dtype=DT, device=DEVICE),
        "jaw": torch.zeros(3, dtype=DT, device=DEVICE),
    }
    groups = {"pose": ("log_s", "rot", "shift"), "shape": ("shape",), "expr": ("expr", "jaw")}
    rates = {"log_s": 2e-3, "rot": 2e-3, "shift": 1e-3, "shape": 3e-2, "expr": 2e-2, "jaw": 3e-3}

    def to_flame(p):
        R = axis_angle_matrix(params["rot"])
        return (scale0 * torch.exp(params["log_s"])) * (p @ R.T) + shift0 + params["shift"]

    def matched(a, b):
        """Mean squared distance from each of a to its nearest in b, ignoring pairs beyond TRUNCATE."""
        near = knn_points(a[None].float(), b[None].float(), K=1).idx[0, :, 0]
        d2 = ((a - b[near]) ** 2).sum(1)
        keep = d2 < TRUNCATE**2
        return d2[keep].mean(), keep.float().mean()

    rms = math.nan
    for stage, (steps, free) in enumerate(STAGES):
        names = [n for g in free for n in groups[g]]
        for n in names:
            params[n].requires_grad_(True)
        opt = torch.optim.Adam([{"params": [params[n]], "lr": rates[n]} for n in names])
        for step in range(steps + 1):
            fl = flame.forward(params["shape"], params["expr"], params["jaw"])
            fl_eyes = torch.stack([fl[i].mean(0) for i in fl_eye_ids])
            centre = fl_eyes.mean(0) + torch.tensor([0, -0.03, 0.0], dtype=DT, device=DEVICE)
            mh = to_flame(mh_points)
            fl_skin = fl[skin_ids]
            with torch.no_grad():
                fl_in = ((fl_skin - centre).norm(dim=1) < FACE_RADIUS) & (fl_skin[:, 2] > centre[2] - 0.04)
                mh_in = ((mh - centre).norm(dim=1) < FACE_RADIUS) & (mh[:, 2] > centre[2] - 0.04)
            loss_ab, kept_ab = matched(fl_skin[fl_in], mh[mh_in])
            loss_ba, kept_ba = matched(mh[mh_in], fl_skin)
            loss_eyes = ((fl_eyes - to_flame(mh_eyes)) ** 2).sum(1).mean()
            loss_nose = ((fl[fl_nose] - to_flame(mh_nose[None])[0]) ** 2).sum()
            prior = sum(PRIOR[k] * (params[k] ** 2).sum() for k in PRIOR)
            loss = loss_ab + loss_ba + loss_eyes + loss_nose + prior
            if step == steps:
                rms = 1000 * math.sqrt(float((loss_ab + loss_ba) / 2))
                log(
                    f"  {head.name} stage {stage + 1} ({'+'.join(free)}): surface rms {rms:.2f} mm "
                    f"(kept {100 * float(kept_ab):.0f}%/{100 * float(kept_ba):.0f}%), "
                    f"eyes {1000 * math.sqrt(float(loss_eyes)):.2f} mm, nose {1000 * math.sqrt(float(loss_nose)):.2f} mm"
                )
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
        for n in names:
            params[n].requires_grad_(False)
    with torch.no_grad():
        fitted = flame.forward(params["shape"], params["expr"], params["jaw"])
        transform = lambda p: to_flame(torch.as_tensor(p, device=DEVICE))  # noqa: E731
        return fitted, transform, rms


def front_pole(verts, ids):
    """An eyeball's front-most vertex (largest z, FLAME looks down +Z)."""
    return int(ids[int(torch.argmax(verts[torch.as_tensor(ids, device=DEVICE), 2]))])


def vertex_on_face(faces, vid):
    """A face holding a vertex, and barycentric weights that put the point exactly on it."""
    fidx = int(torch.nonzero((faces == vid).any(1))[0, 0])
    w = (faces[fidx] == vid).to(DT)
    return fidx, w


# ── all heads ───────────────────────────────────────────────────────────────────────────────────────────────────


def measure(flame_path: str, heads: list[Path], log=print):
    flame = Flame(flame_path)
    on_template = []
    for head in heads:
        points, seen = head_landmarks(head)
        log(f"{head.name}: 68 landmarks hit in {int(seen.min())} to {int(seen.max())} of the views")
        fitted, transform, rms = fit(flame, head, log)
        with torch.no_grad():
            p = transform(torch.as_tensor(points, device=DEVICE))
            fidx, bary = closest_on_triangles(p, fitted, flame.skin_faces)
            gap = ((bary[..., None] * fitted[flame.skin_faces[fidx]]).sum(1) - p).norm(dim=1)
            log(
                f"{head.name}: landmarks to the fitted surface: median {1000 * float(gap.median()):.2f} mm, max {1000 * float(gap.max()):.2f} mm"
            )
            on_template.append((bary[..., None] * flame.template[flame.skin_faces[fidx]]).sum(1))
    # Per landmark, the median over the heads: a head whose fit is off at one spot (the jaw's angle folds sharply)
    # does not move it.
    stack = torch.stack(on_template)
    centre = stack.median(0).values if len(heads) > 2 else stack.mean(0)
    spread = (stack - centre).norm(dim=2).median(0).values * 1000  # each landmark's median distance, mm
    for name, part in REGIONS.items():
        s = spread[list(part)]
        log(
            f"  {name:6s} heads' median distance from the consensus: median {float(s.median()):.1f} mm, worst {float(s.max()):.1f} mm"
        )
    fidx, bary = closest_on_triangles(centre, flame.template, flame.skin_faces)
    faces_idx = flame.skin_faces[fidx]  # back to indices into FLAME's own face list
    face_ids = [_face_index(flame.faces, tri) for tri in faces_idx]
    bary = list(bary)

    for ids in flame.eyeballs:  # right pupil (68), then left (69)
        f, w = vertex_on_face(flame.faces, front_pole(flame.template, ids))
        face_ids.append(f)
        bary.append(w)
    return {
        "full_lmk_faces_idx": np.array([face_ids], dtype=np.int64),
        "full_lmk_bary_coords": torch.stack(bary).cpu().numpy()[None].astype(np.float64),
    }, flame


def _face_index(faces, tri):
    return int(torch.nonzero((faces == tri).all(1))[0, 0])


def template_points(flame: Flame, embedding: dict) -> np.ndarray:
    f = flame.faces[torch.as_tensor(embedding["full_lmk_faces_idx"][0], device=DEVICE)]
    w = torch.as_tensor(embedding["full_lmk_bary_coords"][0], device=DEVICE)
    return (w[..., None] * flame.template[f]).sum(1).cpu().numpy()


if __name__ == "__main__":
    import hashlib

    from embedding import save

    flame_path, out, *heads = sys.argv[1:]
    embedding, flame = measure(flame_path, [Path(h) for h in heads])
    pts = template_points(flame, embedding)
    print(f"pupils on the template: right x {pts[68, 0]:+.4f}, left x {pts[69, 0]:+.4f}")
    about = {
        "about": "Where 68 MediaPipe Face Mesh landmarks (lam/landmarks/ibug68.py, iBUG order) and the two pupils lie on "
        "FLAME 2023 Open: a face index and barycentric weights each. Measured by lam/landmarks/measure.py.",
        "flame_sha256": hashlib.sha256(Path(flame_path).read_bytes()).hexdigest(),
        "heads": [Path(h).name for h in heads],
        "views_per_head": 15,
    }
    save(embedding, about, Path(out))
    print("EMBEDDING", out)
