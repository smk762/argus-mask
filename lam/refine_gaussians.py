"""Spike (#173): refine a LAM head's canonical Gaussians so its frontal render matches the portrait.

Optimises ``offset.ply`` in place (count, order and property layout fixed). Stages: colour (``f_dc_*``), then
opacity + scale, then small position offsets. Every stage pulls toward LAM's own attributes so the unseen sides
stay put. Seeded Adam. Photometric L1 inside the face mask; no extra loss network.

    python lam/refine_gaussians.py --zip head.zip --portrait portrait.png --out refine/ --seed 0

Runs in the ``argus-mask-lam`` image (needs CUDA + ``diff_gaussian_rasterization``). Pure PLY helpers at the top
are importable without torch for CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# offset.ply is head-local (centroid near the origin); the viewer lifts it onto FLAME. The spike camera
# aims at the Gaussian centroid with FOV matching the viewer's long lens.
FOV_DEG = 23.0
SH_C0 = 0.28209479177387814
# Fit the head in frame: half-extent / tan(fov/2), with a little margin.
FRAME_MARGIN = 1.35
REQUIRED = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


# ---- PLY I/O (no torch) -------------------------------------------------------------------------------------


def read_ply(data: bytes) -> tuple[list[str], dict[str, np.ndarray]]:
    """Binary little-endian float32 vertex PLY → (property names in order, {name: (N,) float32})."""
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in header:
        raise ValueError("not a binary little-endian PLY")
    elements = [line.split()[1] for line in header if line.startswith("element")]
    if elements != ["vertex"]:
        raise ValueError(f"elements {elements}; expected vertex alone")
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    names: list[str] = []
    for line in header:
        if line.startswith("property"):
            kind, name = line.split()[1:3]
            if kind != "float":
                raise ValueError(f"property {name} is {kind}; expected float")
            names.append(name)
    table = np.frombuffer(data, np.float32, count * len(names), end).reshape(count, len(names)).copy()
    return names, {name: table[:, i] for i, name in enumerate(names)}


def write_ply(names: list[str], attrs: dict[str, np.ndarray]) -> bytes:
    """Bytes of a binary little-endian PLY with *names* in order; count and order preserved."""
    count = len(attrs[names[0]])
    if any(len(attrs[n]) != count for n in names):
        raise ValueError("every property must have the same count")
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {count}\n"
    header += "".join(f"property float {n}\n" for n in names) + "end_header\n"
    table = np.stack([attrs[n].astype("<f4", copy=False) for n in names], axis=1)
    return header.encode("ascii") + table.tobytes(order="C")


def ply_round_trip(data: bytes) -> bytes:
    """Read then write: used to prove layout survives. Bit-identical on float32 payloads."""
    names, attrs = read_ply(data)
    return write_ply(names, attrs)


def extract_offset_ply(zip_path: Path) -> tuple[str, bytes, list[str]]:
    """``(member path, ply bytes, other zip member names)`` from a LAM head archive."""
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()
        ply_name = next((n for n in members if n.endswith("offset.ply") and not n.endswith("/")), None)
        if ply_name is None:
            raise ValueError(f"{zip_path}: no offset.ply")
        return ply_name, archive.read(ply_name), members


def repack_zip(src: Path, ply_name: str, ply_bytes: bytes, dest: Path) -> str:
    """Copy *src* to *dest* with *ply_name* replaced. Returns sha256 of *dest*. Refused where *dest* is *src*, or
    *src* has no *ply_name*; *dest* is written whole or not at all."""
    if dest.resolve() == src.resolve():
        raise ValueError(f"{dest} is the archive being refined: write the refined one somewhere else")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part")
    try:
        with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            if ply_name not in zin.namelist():
                raise ValueError(f"{src}: no {ply_name} to replace")
            for info in zin.infolist():
                data = ply_bytes if info.filename == ply_name else zin.read(info.filename)
                zout.writestr(info, data)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    return digest


def pull_loss(current: dict[str, np.ndarray], start: dict[str, np.ndarray], keys: tuple[str, ...]) -> float:
    """Mean squared pull of *current* toward *start* over *keys* (numpy, for tests)."""
    total = 0.0
    for key in keys:
        total += float(((current[key] - start[key]) ** 2).mean())
    return total / max(len(keys), 1)


# ---- GPU refine ---------------------------------------------------------------------------------------------


def _aim_from_attrs(attrs: dict[str, np.ndarray] | dict[str, "torch.Tensor"]) -> tuple[np.ndarray, float]:
    """Centroid and camera distance so the head fills the frame at FOV_DEG."""
    import torch

    def np_of(v):
        return v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

    xyz = np.stack([np_of(attrs["x"]), np_of(attrs["y"]), np_of(attrs["z"])], axis=1)
    center = xyz.mean(axis=0).astype(np.float32)
    half = float(np.max(np.linalg.norm(xyz - center, axis=1)))
    distance = max(FRAME_MARGIN * half / math.tan(math.radians(FOV_DEG) / 2), 0.2)
    return center, distance


def _camera(
    height: int,
    width: int,
    *,
    center: np.ndarray,
    distance: float,
    fov_deg: float = FOV_DEG,
):
    """World-to-camera and projection for the LAM image's rasterizer (looks down **+Z** in view space)."""
    import torch

    fov = math.radians(fov_deg)
    tanfovy = math.tan(fov / 2)
    tanfovx = tanfovy * width / height
    # Camera sits on -Z, looking toward +Z at the head.
    cam = np.array([center[0], center[1], center[2] - distance], dtype=np.float32)
    look = center.astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = look - cam  # +Z world
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)  # X = up × forward so a right-handed +Z-forward frame
    right /= max(np.linalg.norm(right), 1e-8)
    up2 = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up2
    view[2, :3] = forward  # +Z forward (not OpenGL's -Z)
    view[:3, 3] = -view[:3, :3] @ cam
    znear, zfar = 0.01, 100.0
    # graphdeco-style projection (matches the rasterizer build in argus-mask-lam).
    top = tanfovy * znear
    right_e = tanfovx * znear
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 * znear / (2.0 * right_e)
    proj[1, 1] = 2.0 * znear / (2.0 * top)
    proj[3, 2] = 1.0
    proj[2, 2] = zfar / (zfar - znear)
    proj[2, 3] = -(zfar * znear) / (zfar - znear)
    # Stored transposed for GLM, as 3DGS does; projmatrix setting is view @ proj in that layout.
    view_t = view.T
    proj_t = proj.T
    full_t = view_t @ proj_t
    return {
        "view": torch.tensor(view_t, device="cuda"),
        "full": torch.tensor(full_t, device="cuda"),
        "campos": torch.tensor(cam, device="cuda"),
        "tanfovx": float(tanfovx),
        "tanfovy": float(tanfovy),
    }


def _load_portrait(path: Path, size: int) -> "torch.Tensor":
    from PIL import Image
    import torch

    img = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.tensor(arr, device="cuda").permute(2, 0, 1)  # 3,H,W


def _face_mask(rgb: "torch.Tensor", alpha: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Soft mask of the head: raster alpha if given, else non-white pixels."""
    import torch

    if alpha is not None:
        return alpha.clamp(0, 1)
    # rgb is 3,H,W on white
    white = (rgb > 0.94).all(dim=0).float()
    return (1.0 - white).clamp(0, 1)


def render_gaussians(
    attrs: dict[str, "torch.Tensor"],
    *,
    size: int = 512,
    bg: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Front render → ``(rgb 3,H,W in 0..1, alpha H,W)``."""
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    center, distance = _aim_from_attrs(attrs)
    cam = _camera(size, size, center=center, distance=distance)
    means3D = torch.stack([attrs["x"], attrs["y"], attrs["z"]], dim=1)
    means2D = torch.zeros_like(means3D, requires_grad=True)
    colours = (torch.stack([attrs["f_dc_0"], attrs["f_dc_1"], attrs["f_dc_2"]], dim=1) * SH_C0 + 0.5).clamp(0, 1)
    opacities = torch.sigmoid(attrs["opacity"]).unsqueeze(1)
    scales = torch.exp(torch.stack([attrs["scale_0"], attrs["scale_1"], attrs["scale_2"]], dim=1))
    rotations = torch.stack([attrs["rot_0"], attrs["rot_1"], attrs["rot_2"], attrs["rot_3"]], dim=1)
    rotations = rotations / rotations.norm(dim=1, keepdim=True).clamp_min(1e-8)
    settings = GaussianRasterizationSettings(
        image_height=size,
        image_width=size,
        tanfovx=cam["tanfovx"],
        tanfovy=cam["tanfovy"],
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=cam["view"],
        projmatrix=cam["full"],
        sh_degree=0,
        campos=cam["campos"],
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(settings)
    color, _radii, _depth, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        opacities=opacities,
        colors_precomp=colours,
        scales=scales,
        rotations=rotations,
    )
    return color, alpha.squeeze(0) if alpha.ndim == 3 else alpha


def _as_torch(attrs: dict[str, np.ndarray], keys: tuple[str, ...]) -> dict[str, "torch.Tensor"]:
    import torch

    out = {}
    for name, arr in attrs.items():
        t = torch.tensor(arr, device="cuda", dtype=torch.float32)
        if name in keys:
            t = t.clone().requires_grad_(True)
        out[name] = t
    return out


def _numpy_attrs(torch_attrs: dict[str, "torch.Tensor"], names: list[str]) -> dict[str, np.ndarray]:
    return {n: torch_attrs[n].detach().float().cpu().numpy().astype(np.float32) for n in names}


STAGES = (
    ("colour", ("f_dc_0", "f_dc_1", "f_dc_2"), 80, 0.05),
    ("opacity_scale", ("opacity", "scale_0", "scale_1", "scale_2"), 40, 0.02),
    ("position", ("x", "y", "z"), 30, 0.005),
)


def stage_steps(steps: int, scale: float) -> int:
    """How many steps a stage of *steps* runs at *scale*: at least one."""
    return max(1, int(steps * scale))


def refine(
    names: list[str],
    attrs: dict[str, np.ndarray],
    portrait: Path,
    *,
    seed: int = 0,
    size: int = 512,
    pull: float = 2.0,
    steps_scale: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run the three stages. Returns refined attrs and a record of losses."""
    import torch

    missing = [k for k in REQUIRED if k not in attrs]
    if missing:
        raise ValueError(f"offset.ply missing {missing}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    target = _load_portrait(portrait, size)
    start = {k: v.copy() for k, v in attrs.items()}
    current = {k: v.copy() for k, v in attrs.items()}
    history: list[dict] = []

    # Cap position drift relative to head size.
    extent = float(np.linalg.norm(np.stack([attrs["x"], attrs["y"], attrs["z"]], 1).std(axis=0)))
    pos_cap = 0.02 * max(extent, 1e-3)

    for stage_name, keys, steps, lr in STAGES:
        steps = stage_steps(steps, steps_scale)
        params = _as_torch(current, keys)
        # freeze non-stage tensors as plain data
        for k, t in list(params.items()):
            if k not in keys:
                params[k] = t.detach()
        optim = torch.optim.Adam([params[k] for k in keys], lr=lr)
        best_loss = float("inf")
        best_snap = {k: params[k].detach().clone() for k in names}
        for step in range(steps):
            optim.zero_grad(set_to_none=True)
            rgb, alpha = render_gaussians(params, size=size)
            mask = _face_mask(rgb, alpha)
            weight = mask.sum().clamp_min(1.0)
            photo = ((rgb - target).abs() * mask).sum() / weight
            # If the head missed the frame entirely, fall back to full-frame L1 so the stage still moves.
            if float(mask.max()) < 1e-4:
                photo = (rgb - target).abs().mean()
            reg = sum(((params[k] - torch.tensor(start[k], device="cuda")) ** 2).mean() for k in keys)
            loss = photo + pull * reg
            value = float(loss.detach())
            if value < best_loss:  # before the step: the parameters this loss is of
                best_loss = value
                best_snap = {k: params[k].detach().clone() for k in names}
            loss.backward()
            optim.step()
            if stage_name == "position":
                with torch.no_grad():
                    for ax in ("x", "y", "z"):
                        base = torch.tensor(start[ax], device="cuda")
                        params[ax].copy_(base + (params[ax] - base).clamp(-pos_cap, pos_cap))
            if step == 0 or step + 1 == steps or (step + 1) % 20 == 0:
                history.append(
                    {
                        "stage": stage_name,
                        "step": step + 1,
                        "loss": value,
                        "photo": float(photo.detach()),
                        "pull": float(reg.detach()),
                    }
                )
        current = _numpy_attrs({k: best_snap[k] for k in names}, names)
        # Keep normals / anything not optimised from the last snap
        for n in names:
            if n not in current:
                current[n] = start[n]

    record = {
        "seed": seed,
        "size": size,
        "pull": pull,
        "stages": [{"name": s[0], "keys": list(s[1]), "steps": stage_steps(s[2], steps_scale), "lr": s[3]} for s in STAGES],
        "history": history,
        "start_pull": pull_loss(current, start, REQUIRED),
        "inputs": "portrait + LAM offset.ply only; photometric L1 + L2 pull; no extra network",
        "licences": "diff_gaussian_rasterization (inria, Gaussian rasterizer in the LAM image)",
    }
    return current, record


def save_render(rgb: "torch.Tensor", path: Path) -> None:
    from PIL import Image

    arr = (rgb.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--zip", type=Path, required=True, help="LAM head archive (…/face.zip)")
    parser.add_argument("--portrait", type=Path, required=True, help="source portrait PNG")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pull", type=float, default=2.0, help="weight of the pull toward LAM's attributes")
    parser.add_argument("--steps-scale", type=float, default=1.0, help="multiply stage step counts (1 = full)")
    parser.add_argument("--repack", action="store_true", help="write a refined head.zip beside the outputs")
    args = parser.parse_args(argv)

    dest = args.out / args.zip.name
    if args.repack and dest.resolve() == args.zip.resolve():  # before any work
        parser.error(f"--out {args.out} holds the archive being refined: choose another directory")
    ply_name, ply_bytes, _ = extract_offset_ply(args.zip)
    names, attrs = read_ply(ply_bytes)
    rt = ply_round_trip(ply_bytes)
    names_rt, attrs_rt = read_ply(rt)
    if names_rt != names or any(not np.array_equal(attrs[n], attrs_rt[n]) for n in names):
        raise SystemExit("PLY round-trip changed the floats; refusing to refine")

    args.out.mkdir(parents=True, exist_ok=True)
    import torch

    with torch.no_grad():
        t0 = _as_torch(attrs, ())
        rgb0, _ = render_gaussians(t0, size=args.size)
        save_render(rgb0, args.out / "start.png")

    refined, record = refine(
        names, attrs, args.portrait, seed=args.seed, size=args.size, pull=args.pull, steps_scale=args.steps_scale
    )
    # Restore any properties refine did not carry (e.g. normals)
    for n in names:
        if n not in refined:
            refined[n] = attrs[n]
        elif n.startswith("n"):
            refined[n] = attrs[n]  # normals unchanged

    out_ply = write_ply(names, {n: refined[n] for n in names})
    (args.out / "offset.ply").write_bytes(out_ply)

    with torch.no_grad():
        t1 = _as_torch(refined, ())
        rgb1, _ = render_gaussians(t1, size=args.size)
        save_render(rgb1, args.out / "best.png")

    record["ply_sha256"] = hashlib.sha256(out_ply).hexdigest()
    record["source_zip"] = str(args.zip)
    record["portrait"] = str(args.portrait)
    record["gaussian_count"] = len(attrs[names[0]])
    if args.repack:
        record["zip_sha256"] = repack_zip(args.zip, ply_name, out_ply, dest)
        record["refined_zip"] = str(dest)
    (args.out / "refine.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "ply_sha256": record["ply_sha256"], "steps": len(record["history"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
