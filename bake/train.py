"""Train Gaussian splats bound to a head's triangles, from Cycles renders at many expressions (#126 route A).

    python bake/train.py --glb head.glb --expressions expressions.json --renders renders --out trained \
        [--splats 150000] [--steps 30000]

Each splat belongs to one triangle, at fixed barycentrics (u, v) with a learned offset and rotation in the triangle's
frame, so at expression e it sits at  p_e + R_e @ offset,  turned by  R_e ∘ q_local, where p_e is the barycentric
point and R_e the frame (first edge, normal) of its triangle. The viewer's modifier (Spark dyno) applies the same
thing as a change from rest: centre = p_e + R_e R_0^T (c_0 - p_0), quat = (R_e R_0^T) q_0. Scales are not changed
by the triangle's stretch, in training or in the viewer.

Writes ``<out>/splats.ply`` (standard 3DGS PLY, SH degree 3, at rest: no morphs), ``<out>/binding.bin``
(float32 triangle, u, v per splat, in the PLY's order), ``<out>/head.json`` (the parts, in the order the vertex
space concatenates them) and ``<out>/metrics.json``. Deterministic for a seed, up to CUDA's atomics.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mesh as headmesh  # noqa: E402

SH_C0 = 0.28209479177387814


# ---- geometry ---------------------------------------------------------------------------------------------------


def frames(v: torch.Tensor, tris: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per splat: its triangle's point (N, 3) and frame (N, 3, 3) (columns: first edge, normal x edge, normal)."""
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    p = a + u[:, None] * (b - a) + w[:, None] * (c - a)
    e1 = F.normalize(b - a, dim=-1)
    n = F.normalize(torch.cross(b - a, c - a, dim=-1), dim=-1)
    e2 = torch.cross(n, e1, dim=-1)
    return p, torch.stack([e1, e2, n], dim=-1)


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) rotations -> (N, 4) quaternions, w first (gsplat's order)."""
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.stack(
        [
            1 + t,
            1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
            1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2],
        ],
        -1,
    )
    k = q.argmax(-1)
    w = torch.empty(len(m), 4, device=m.device)
    # Each branch picks the largest component for stability (Shepperd's method).
    r = torch.sqrt(q.gather(1, k[:, None]).clamp_min(1e-12))[:, 0] * 2  # 4 * the largest component
    m01, m02, m10 = m[:, 0, 1], m[:, 0, 2], m[:, 1, 0]
    m12, m20, m21 = m[:, 1, 2], m[:, 2, 0], m[:, 2, 1]
    cand = [
        torch.stack([r / 4, (m21 - m12) / r, (m02 - m20) / r, (m10 - m01) / r], -1),
        torch.stack([(m21 - m12) / r, r / 4, (m01 + m10) / r, (m02 + m20) / r], -1),
        torch.stack([(m02 - m20) / r, (m01 + m10) / r, r / 4, (m12 + m21) / r], -1),
        torch.stack([(m10 - m01) / r, (m02 + m20) / r, (m12 + m21) / r, r / 4], -1),
    ]
    for i, c in enumerate(cand):
        w[k == i] = c[k == i]
    return F.normalize(w, dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # w first
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


# ---- data -------------------------------------------------------------------------------------------------------


def load_views(renders: Path, device: str) -> list[dict]:
    views = []
    for cam in json.loads((renders / "cameras.json").read_text()):
        img_u8 = np.asarray(Image.open(renders / cam["file"]).convert("RGBA"), np.uint8).copy()
        c2w = np.array(cam["c2w"], np.float64) @ np.diag([1.0, -1.0, -1.0, 1.0])  # OpenGL -> OpenCV
        h, w = cam["height"], cam["width"]
        fy = (h / 2) / math.tan(math.radians(cam["fov_y"]) / 2)
        K = np.array([[fy, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], np.float32)
        views.append(
            {
                "set": cam["set"],
                "expression": cam["expression"],
                "file": cam["file"],
                "image": torch.from_numpy(img_u8),  # host memory; one view at a time goes to the GPU
                "viewmat": torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).to(device),
                "K": torch.from_numpy(K).to(device),
                "w": w,
                "h": h,
            }
        )
    return views


def ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # (1, 3, H, W) in [0, 1]
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    win = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)
    f = lambda t: F.conv2d(t, win, padding=5, groups=3)  # noqa: E731
    mx, my = f(x), f(y)
    sx, sy, sxy = f(x * x) - mx**2, f(y * y) - my**2, f(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sx + sy + c2))).mean()


# ---- the model ----------------------------------------------------------------------------------------------------


def allocate(head: headmesh.Head, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splats per triangle in proportion to area (at least one each); barycentrics on a low-discrepancy sequence."""
    a, b, c = (head.rest[head.tris[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    per = np.maximum(1, np.floor(area / area.sum() * n)).astype(int)
    tri = np.repeat(np.arange(len(per)), per)
    k = np.concatenate([np.arange(p) for p in per])
    g = 1.32471795724474602596
    off = rng.random(len(per))[tri]
    u, w = (off + (k + 1) / g) % 1, (off * 1.618 + (k + 1) / g**2) % 1
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    return tri.astype(np.int64), u.astype(np.float32), w.astype(np.float32)


def albedo_dc(head: headmesh.Head, tri: np.ndarray, u: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Initial colour: the part's base colour texture at the splat's UV (grey where a part has none)."""
    colour = np.full((len(tri), 3), 0.5, np.float32)
    part = head.tri_part[tri]
    t = head.tris[tri]
    uv = (
        head.uv[t[:, 0]]
        + u[:, None] * (head.uv[t[:, 1]] - head.uv[t[:, 0]])
        + w[:, None] * (head.uv[t[:, 2]] - head.uv[t[:, 0]])
    )
    for i, name in enumerate(head.names):
        if name not in head.textures:
            continue
        img = np.asarray(Image.open(io.BytesIO(head.textures[name])).convert("RGB"), np.float32) / 255.0
        sel = part == i
        x = np.clip((uv[sel, 0] % 1) * img.shape[1], 0, img.shape[1] - 1).astype(int)
        y = np.clip((uv[sel, 1] % 1) * img.shape[0], 0, img.shape[0] - 1).astype(int)
        colour[sel] = img[y, x]
    return (colour - 0.5) / SH_C0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", type=Path, required=True)
    ap.add_argument("--expressions", type=Path, required=True)
    ap.add_argument("--renders", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splats", type=int, default=150_000)
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = "cuda"
    ex = json.loads(a.expressions.read_text())
    head = headmesh.load(a.glb, ex["blendshapes"])
    sets = {
        s: torch.from_numpy(head.vertices(np.asarray(ex[s], np.float32)).astype(np.float32)).to(dev)
        for s in ("train", "holdout")
    }
    tri, u, w = allocate(head, a.splats, rng)
    n = len(tri)
    tris = torch.from_numpy(head.tris[tri]).to(dev, dtype=torch.long)
    tu, tw = torch.from_numpy(u).to(dev), torch.from_numpy(w).to(dev)
    a0, b0, c0 = (torch.from_numpy(head.rest[head.tris[tri][:, k]]).to(dev) for k in range(3))
    area = 0.5 * torch.linalg.norm(torch.cross(b0 - a0, c0 - a0, dim=-1), dim=-1)
    per = torch.bincount(torch.from_numpy(tri).to(dev), minlength=len(head.tris)).float()[torch.from_numpy(tri).to(dev)]
    sigma = torch.sqrt(area / per / math.pi).clamp(1e-4, 0.01)
    params = torch.nn.ParameterDict(
        {
            "offset": torch.zeros(n, 3, device=dev),
            "log_scale": torch.log(torch.stack([sigma, sigma, sigma * 0.3], -1)),
            "quat": torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n, 1),
            "opacity": torch.full((n,), 1.5, device=dev),  # logit: 0.82
            "sh0": torch.from_numpy(albedo_dc(head, tri, u, w)).to(dev)[:, None, :],
            "shN": torch.zeros(n, 15, 3, device=dev),
        }
    )
    lrs = {"offset": 5e-5, "log_scale": 5e-3, "quat": 1e-3, "opacity": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opt = torch.optim.Adam([{"params": [params[k]], "lr": lr, "name": k} for k, lr in lrs.items()], eps=1e-15)
    decay = 0.01 ** (1 / a.steps)  # offsets' lr falls to 1% over the run

    views = load_views(a.renders, dev)
    train = [v for v in views if v["set"] == "train"]
    held = [v for v in views if v["set"] == "holdout"]
    print(f"{n} splats on {len(head.tris)} triangles; {len(train)} training views, {len(held)} held out", flush=True)

    def render(verts: torch.Tensor, view: dict, background: torch.Tensor, sh_degree: int = 3):
        p, R = frames(verts, tris, tu, tw)
        means = p + (R @ params["offset"][:, :, None])[:, :, 0]
        quats = quat_mul(mat_to_quat(R), F.normalize(params["quat"], dim=-1))
        colours, alphas, _ = rasterization(
            means,
            quats,
            torch.exp(params["log_scale"]),
            torch.sigmoid(params["opacity"]),
            torch.cat([params["sh0"], params["shN"]], 1),
            view["viewmat"][None],
            view["K"][None],
            view["w"],
            view["h"],
            sh_degree=sh_degree,
            backgrounds=background[None],
            packed=False,
        )
        return colours[0], alphas[0]

    def image(view: dict) -> torch.Tensor:
        return view["image"].to(dev, non_blocking=True).float() / 255.0

    def target(view: dict, background: torch.Tensor) -> torch.Tensor:
        img = image(view)
        return img[..., :3] * img[..., 3:] + background * (1 - img[..., 3:])

    def evaluate(subset: list[dict], verts_of) -> float:
        psnrs = []
        with torch.no_grad():
            bg = torch.zeros(3, device=dev)
            for view in subset:
                out, _ = render(verts_of[view["expression"]], view, bg)
                mse = F.mse_loss(out.clamp(0, 1), target(view, bg))
                psnrs.append(float(-10 * torch.log10(mse)))
        return float(np.mean(psnrs))

    t0 = time.time()
    order = rng.permutation(np.arange(a.steps) % len(train))
    for step in range(a.steps):
        view = train[order[step]]
        bg = torch.rand(3, device=dev)  # random backgrounds: the alpha has to be right, not just the colour
        degree = min(3, step // 1000)
        out, alpha = render(sets["train"][view["expression"]], view, bg, degree)
        gt = target(view, bg)
        loss = 0.8 * (out - gt).abs().mean() + 0.2 * (1 - ssim(out.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        loss = loss + 0.01 * (alpha[..., 0] - image(view)[..., 3]).abs().mean()
        loss = loss + 1e-2 * params["offset"].norm(dim=-1).mean()  # stay near the triangle
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        opt.param_groups[0]["lr"] *= decay
        if step and step % 2000 == 0 and step < a.steps - 3000:
            relocate(params, opt)
        if step % 1000 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    metrics = {
        "splats": n,
        "steps": a.steps,
        "seconds": round(time.time() - t0),
        "psnr_train": evaluate(train[::5], sets["train"]),
        "psnr_holdout": evaluate(held, sets["holdout"]),
    }
    print(metrics, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    export(a.out, head, params, tris, tu, tw, torch.from_numpy(head.rest).to(dev), tri, u, w)
    (a.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # A few held-out views as the trainer draws them, on the viewer's background, by their index in cameras.json:
    # the viewer draws the same camera and weights, so viewer-vs-trainer checks the binding and the colours.
    files = [c["file"] for c in json.loads((a.renders / "cameras.json").read_text())]
    bg = torch.tensor([16, 18, 22], device=dev, dtype=torch.float32) / 255
    for view in held[:: max(1, len(held) // 6)]:
        with torch.no_grad():
            out, _ = render(sets["holdout"][view["expression"]], view, bg)
            i = files.index(view["file"])
            for name, img in (("gt", target(view, bg)), ("trained", out.clamp(0, 1))):
                Image.fromarray((img.cpu().numpy() * 255 + 0.5).astype(np.uint8)).save(
                    a.out / f"view{i:04d}_{name}.png"
                )


def relocate(params: torch.nn.ParameterDict, opt: torch.optim.Adam) -> None:
    """Revive dead splats (opacity < 0.005) in place at opacity 0.1, their optimiser state cleared. A splat's
    triangle is fixed, so this keeps every triangle covered rather than moving splats to where detail is wanted
    (MCMC-style relocation across triangles is a later step)."""
    with torch.no_grad():
        op = torch.sigmoid(params["opacity"])
        dead = op < 0.005
        if not dead.any():
            return
        # Revive in place: a dead splat comes back faint with its neighbours' colour, so the area is not lost.
        params["opacity"][dead] = torch.logit(torch.tensor(0.1, device=op.device))
        rows = dead.reshape(dead.shape[0], -1).any(dim=1)  # one per splat, whatever opacity's shape: (N,) or (N, 1)
        for state in opt.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape[0] == rows.shape[0]:
                    state[key][rows] = 0
        print(f"revived {int(rows.sum())} splats", flush=True)


def export(out: Path, head, params, tris, tu, tw, rest, tri, u, w) -> None:
    with torch.no_grad():
        p, R = frames(rest, tris, tu, tw)
        means = (p + (R @ params["offset"][:, :, None])[:, :, 0]).cpu().numpy()
        quats = quat_mul(mat_to_quat(R), F.normalize(params["quat"], dim=-1)).cpu().numpy()  # w, x, y, z
        scales = params["log_scale"].cpu().numpy()
        opacity = params["opacity"].cpu().numpy()
        sh0 = params["sh0"][:, 0, :].cpu().numpy()
        shN = params["shN"].permute(0, 2, 1).reshape(len(means), 45).cpu().numpy()  # channel-major, as 3DGS
    n = len(means)
    fields = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(45)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    data = np.concatenate([means, np.zeros((n, 3)), sh0, shN, opacity[:, None], scales, quats], 1).astype(np.float32)
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {f}\n" for f in fields) + "end_header\n"
    (out / "splats.ply").write_bytes(header.encode() + data.tobytes())
    np.stack([tri.astype(np.float32), u, w], 1).astype(np.float32).tofile(out / "binding.bin")
    parts = [
        {"name": nm, "offset": int(o), "count": int(c)}
        for nm, o, c in zip(head.names, head.offsets, head.counts, strict=True)
    ]
    (out / "head.json").write_text(
        json.dumps(
            {"parts": parts, "vertices": int(len(head.rest)), "triangles": int(len(head.tris)), "splats": n}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
