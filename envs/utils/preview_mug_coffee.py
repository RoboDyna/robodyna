#!/usr/bin/env python3
"""Graphics-only preview: coffee disk vs ``039_mug`` cavity (not wired into clean_table).

PartNet mugs have a handle that stretches the AABB. The pose origin sits near the
cup bottom, but the circular opening is offset away from the handle. The old
``clean_table`` heuristic used a too-small inner radius and a too-small offset,
so the coffee read as a tiny circle inside the cup head.

This script offscreen-renders top-down + 3/4 views for:
  - ``current`` — clean_table's existing 0.19/0.25 heuristics
  - ``fixed``   — mesh-measured cavity center + inner radius (full opening)

Run (robotwin_bench / domino env + Vulkan):
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
  python envs/utils/preview_mug_coffee.py --mug-id 0
  python envs/utils/preview_mug_coffee.py --mug-id 0 --all-ids
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)

import numpy as np
import sapien
import sapien.render
import transforms3d.quaternions as tq
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MUG_DIR = ROOT / "assets" / "objects" / "039_mug"
MUG_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]
COFFEE_COLOR = [0.55, 0.38, 0.22, 0.95]
# Thin cyan rim marker so the measured opening is visible in the preview.
RIM_COLOR = [0.15, 0.85, 0.95, 0.85]


def _load_model_data(mug_id: int) -> dict:
    return json.loads((MUG_DIR / f"model_data{mug_id}.json").read_text())


def _final_scale(data: dict, scale_mult: float) -> float:
    sc = data.get("scale", [1, 1, 1])
    sc0 = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
    return sc0 * float(scale_mult)


def _world_extents(data: dict, scale_mult: float) -> np.ndarray:
    ext = np.asarray(data.get("extents", [1, 1, 1]), dtype=float)
    sc = data.get("scale", [1, 1, 1])
    sc = np.asarray(sc if isinstance(sc, (list, tuple)) else [sc] * 3, dtype=float)
    return ext * sc * float(scale_mult)


def current_coffee_params(data: dict, scale_mult: float):
    """Existing clean_table heuristics (intentionally left as-is for comparison)."""
    world = _world_extents(data, scale_mult)
    wmax = float(max(world[0], world[2]))
    inner_r = 0.19 * wmax
    coffee_r = max(0.008, 0.88 * inner_r)
    off = np.zeros(2, dtype=float)
    fmat = data.get("functional_matrix") or []
    if fmat:
        final = _final_scale(data, scale_mult)
        h = np.array(fmat[0], dtype=float)[:3, 3] * final
        R = tq.quat2mat(np.asarray(MUG_UPRIGHT_Q, dtype=float))
        h_xy = (R @ h)[:2]
        n = float(np.linalg.norm(h_xy))
        if n > 1e-6:
            off = (-h_xy / n) * (0.25 * inner_r)
    return {
        "offset_xy": off,
        "radius": coffee_r,
        "inner_r": inner_r,
        "label": "current",
    }


def measure_cavity(mug_id: int, scale_mult: float) -> dict:
    """Largest empty circle in a mid-height horizontal slice (cup head, not handle)."""
    data = _load_model_data(mug_id)
    final = _final_scale(data, scale_mult)
    mesh = trimesh.load(str(MUG_DIR / "visual" / f"base{mug_id}.glb"), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    v = np.asarray(mesh.vertices, dtype=float) * final
    R = tq.quat2mat(np.asarray(MUG_UPRIGHT_Q, dtype=float))
    vw = (R @ v.T).T
    z = vw[:, 2]
    top = float(z.max())
    band = vw[np.abs(z - 0.45 * top) < 0.0018]
    xy = band[:, :2]
    if len(xy) < 40:
        raise RuntimeError(f"mug {mug_id}: too few mid-slice points for cavity fit")
    rng = np.random.default_rng(0)
    if len(xy) > 2500:
        xy = xy[rng.choice(len(xy), 2500, replace=False)]

    xs = np.linspace(xy[:, 0].min() - 0.005, xy[:, 0].max() + 0.005, 51)
    ys = np.linspace(xy[:, 1].min() - 0.005, xy[:, 1].max() + 0.005, 71)
    xx, yy = np.meshgrid(xs, ys)
    cand = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dmin = np.full(len(cand), np.inf)
    for i in range(0, len(cand), 200):
        c = cand[i : i + 200]
        dmin[i : i + 200] = np.linalg.norm(c[:, None, :] - xy[None, :, :], axis=2).min(1)

    best_i, best_r = None, -1.0
    for i in range(len(cand)):
        r = float(dmin[i])
        if r < 0.008 or r > 0.05:
            continue
        ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ok = 0
        for a in ang:
            p = cand[i] + (r + 0.002) * np.array([np.cos(a), np.sin(a)])
            if np.linalg.norm(xy - p, axis=1).min() < 0.006:
                ok += 1
        if ok >= 12 and r > best_r:
            best_r = r
            best_i = i
    if best_i is None:
        raise RuntimeError(f"mug {mug_id}: cavity fit failed")

    # Match the inner opening (same dimension as the cup-head circle).
    coffee_r = max(0.008, float(best_r))
    return {
        "offset_xy": cand[best_i].astype(float),
        "radius": float(coffee_r),
        "inner_r": float(best_r),
        "label": "fixed",
    }


def fixed_heuristic_params(data: dict, scale_mult: float) -> dict:
    """Cheap closed-form fallback tuned on mesh fits across base0–12."""
    world = _world_extents(data, scale_mult)
    wmax = float(max(world[0], world[2]))
    inner_r = 0.26 * wmax
    coffee_r = max(0.008, float(inner_r))
    off = np.zeros(2, dtype=float)
    fmat = data.get("functional_matrix") or []
    if fmat:
        final = _final_scale(data, scale_mult)
        h = np.array(fmat[0], dtype=float)[:3, 3] * final
        R = tq.quat2mat(np.asarray(MUG_UPRIGHT_Q, dtype=float))
        h_xy = (R @ h)[:2]
        n = float(np.linalg.norm(h_xy))
        if n > 1e-6:
            # ~0.12 * AABB width away from the handle into the cup head.
            off = (-h_xy / n) * (0.12 * wmax)
    return {
        "offset_xy": off,
        "radius": coffee_r,
        "inner_r": inner_r,
        "label": "fixed_heur",
    }


def _coffee_material(rgba):
    mat = sapien.render.RenderMaterial(base_color=list(rgba))
    try:
        mat.set_roughness(0.35)
        mat.set_metallic(0.0)
        mat.set_transmission(0.0)
    except Exception:
        mat.roughness = 0.35
        mat.metallic = 0.0
    return mat


def _add_disk(scene, xy, z, radius, half_h, rgba, name):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_cylinder_visual(
        pose=sapien.Pose([0, 0, 0], VERTICAL_CYL_Q),
        radius=float(radius),
        half_length=float(half_h),
        material=_coffee_material(rgba),
    )
    builder.set_initial_pose(sapien.Pose(p=[float(xy[0]), float(xy[1]), float(z)]))
    return builder.build(name=name)


def _add_coffee_column(scene, xy, z_bottom, z_top, radius, rgba, name):
    """Opaque column from cup floor up to the fill surface (pour_beer-style)."""
    hh = max(0.0025, 0.5 * (float(z_top) - float(z_bottom)))
    z_c = float(z_bottom) + hh
    return _add_disk(scene, xy, z_c, radius, hh, rgba, name)


def _add_rim_marker(scene, xy, z, radius):
    """Thin torus-like ring via a slightly larger hollow-looking disk edge (flat annulus proxy)."""
    # Outer ring disk + punch visually by stacking a hole-colored? Simpler: thin tall ring wall.
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    # Narrow vertical cylinder shell approximation: two cylinders (outer visual only).
    builder.add_cylinder_visual(
        pose=sapien.Pose([0, 0, 0], VERTICAL_CYL_Q),
        radius=float(radius),
        half_length=0.0012,
        material=_coffee_material(RIM_COLOR),
    )
    builder.set_initial_pose(sapien.Pose(p=[float(xy[0]), float(xy[1]), float(z)]))
    return builder.build(name="rim_marker")


def _lookat_pose(eye, target, up=np.array([0.0, 0.0, 1.0])):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    left = np.cross(up, fwd)
    left /= np.linalg.norm(left) + 1e-12
    up_n = np.cross(fwd, left)
    m = np.eye(4)
    m[:3, 0] = fwd
    m[:3, 1] = left
    m[:3, 2] = up_n
    m[:3, 3] = eye
    return sapien.Pose(m)


def render_variant(
    mug_id: int,
    scale_mult: float,
    fill_frac: float,
    params: dict,
    out_dir: Path,
    show_rim: bool,
    column: bool,
    size: int = 640,
):
    data = _load_model_data(mug_id)
    final_sc = _final_scale(data, scale_mult)
    world = _world_extents(data, scale_mult)
    height = float(world[1])

    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0.3, 0.4, -1.0], [0.9, 0.9, 0.85])
    scene.add_directional_light([-0.5, -0.2, -0.6], [0.35, 0.35, 0.4])

    # Ground card so the mug casts a bit of context.
    ground = scene.create_actor_builder()
    ground.set_physx_body_type("static")
    gmat = sapien.render.RenderMaterial(base_color=[0.82, 0.80, 0.76, 1.0])
    ground.add_box_visual(half_size=[0.18, 0.18, 0.002], material=gmat)
    ground.set_initial_pose(sapien.Pose([0, 0, -0.002]))
    ground.build(name="ground")

    scale = [final_sc, final_sc, final_sc]
    mug_b = scene.create_actor_builder()
    mug_b.set_physx_body_type("kinematic")
    mug_b.add_visual_from_file(
        filename=str(MUG_DIR / "visual" / f"base{mug_id}.glb"),
        scale=scale,
    )
    mug_b.set_initial_pose(sapien.Pose([0, 0, 0.001], MUG_UPRIGHT_Q))
    mug_b.build(name="mug")

    off = np.asarray(params["offset_xy"], dtype=float)
    z_top = 0.001 + float(fill_frac) * height
    # Keep a small floor clearance so the column sits inside the cavity.
    z_bottom = 0.001 + 0.08 * height
    if column and params["label"] != "current":
        _add_coffee_column(
            scene,
            xy=off,
            z_bottom=z_bottom,
            z_top=z_top,
            radius=params["radius"],
            rgba=COFFEE_COLOR,
            name="coffee_column",
        )
    else:
        _add_disk(
            scene,
            xy=off,
            z=z_top,
            radius=params["radius"],
            half_h=0.0035,
            rgba=COFFEE_COLOR,
            name="coffee_disk",
        )
    if show_rim:
        _add_rim_marker(scene, xy=off, z=z_top + 0.004, radius=params["inner_r"])

    target = np.array([float(off[0]), float(off[1]), 0.5 * height])
    tag = params["label"]

    # Orthographic-ish topdown: far camera + narrow FOV so the fill surface
    # isn't perspective-shrunk relative to the rim above it.
    top_eye = np.array([off[0], off[1], height + 1.20])
    top_tgt = np.array([off[0], off[1], 0.0])
    cam = scene.add_camera(f"{tag}_topdown", size, size, 0.12, 0.01, 5.0)
    try:
        cam.entity.set_pose(_lookat_pose(top_eye, top_tgt))
    except Exception:
        cam.set_pose(_lookat_pose(top_eye, top_tgt))
    scene.step()
    scene.update_render()
    cam.take_picture()
    rgb = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
    path = out_dir / f"mug{mug_id}_{tag}_topdown.png"
    Image.fromarray(rgb).save(path)
    print(f"  wrote {path}")

    obl_eye = np.array([0.14, -0.12, height + 0.10])
    cam2 = scene.add_camera(f"{tag}_oblique", size, size, 0.70, 0.001, 10.0)
    try:
        cam2.entity.set_pose(_lookat_pose(obl_eye, target))
    except Exception:
        cam2.set_pose(_lookat_pose(obl_eye, target))
    scene.step()
    scene.update_render()
    cam2.take_picture()
    rgb = (cam2.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
    path = out_dir / f"mug{mug_id}_{tag}_oblique.png"
    Image.fromarray(rgb).save(path)
    print(f"  wrote {path}")

    print(
        f"  [{tag}] offset_xy={np.round(off, 4).tolist()}  "
        f"coffee_r={params['radius']:.4f}  inner_r={params['inner_r']:.4f}  "
        f"fill_z={z_top:.4f}  mug_h={height:.4f}  column={column and params['label'] != 'current'}"
    )


def side_by_side(out_dir: Path, mug_id: int, view: str = "topdown"):
    a = out_dir / f"mug{mug_id}_current_{view}.png"
    b = out_dir / f"mug{mug_id}_fixed_{view}.png"
    if not (a.exists() and b.exists()):
        return
    ia, ib = Image.open(a), Image.open(b)
    w, h = ia.size
    canvas = Image.new("RGB", (w * 2 + 8, h), (30, 30, 30))
    canvas.paste(ia, (0, 0))
    canvas.paste(ib, (w + 8, 0))
    path = out_dir / f"mug{mug_id}_compare_{view}.png"
    canvas.save(path)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mug-id", type=int, default=0)
    ap.add_argument("--all-ids", action="store_true")
    ap.add_argument("--scale-mult", type=float, default=0.60)
    ap.add_argument("--fill-frac", type=float, default=0.40)
    ap.add_argument(
        "--mode",
        choices=("both", "current", "fixed", "fixed_heur"),
        default="both",
    )
    ap.add_argument("--show-rim", action="store_true", help="draw measured rim marker")
    ap.add_argument(
        "--column",
        action="store_true",
        default=True,
        help="fixed mode: fill with a coffee column (default on)",
    )
    ap.add_argument("--disk-only", action="store_true", help="fixed mode: thin disk only")
    ap.add_argument(
        "--out",
        default=str(ROOT / "docs/final_task_demos" / "clean_table" / "coffee_graphics"),
    )
    args = ap.parse_args()
    use_column = bool(args.column) and not bool(args.disk_only)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(range(13)) if args.all_ids else [int(args.mug_id)]

    for mid in ids:
        data = _load_model_data(mid)
        print(f"\n=== mug {mid} ===")
        modes = []
        if args.mode in ("both", "current"):
            modes.append(current_coffee_params(data, args.scale_mult))
        if args.mode in ("both", "fixed"):
            try:
                modes.append(measure_cavity(mid, args.scale_mult))
            except Exception as e:
                print(f"  mesh fit failed ({e}); falling back to fixed_heur")
                modes.append(fixed_heuristic_params(data, args.scale_mult))
        if args.mode == "fixed_heur":
            modes.append(fixed_heuristic_params(data, args.scale_mult))

        for params in modes:
            render_variant(
                mid,
                args.scale_mult,
                args.fill_frac,
                params,
                out_dir,
                show_rim=bool(args.show_rim and params["label"] != "current"),
                column=use_column,
            )
        if args.mode == "both":
            side_by_side(out_dir, mid, "topdown")
            side_by_side(out_dir, mid, "oblique")


if __name__ == "__main__":
    main()
