#!/usr/bin/env python3
"""Rebuild open-cup collision for 076_breadbasket base3 and base4.

Those variants ship with a closed / slit collision hull, so dropped fruit
hits an invisible lid. Meshes are authored in raw units (model_data scale 0.1).
"""
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/objects/076_breadbasket/collision"
RX = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1, 0, 0])


def _export(geoms, path: Path) -> None:
    scene = trimesh.Scene({f"p{i}": g for i, g in enumerate(geoms)})
    scene.export(str(path))
    comb = trimesh.util.concatenate(geoms)
    print(path.name, "n=", len(geoms), "aabb", np.round(comb.bounds, 3))


def rect_cup(half_x, half_z, height, wall=0.09, floor_h=0.055):
    geoms = []
    floor = trimesh.creation.box(
        extents=[2 * (half_x - wall * 0.5), floor_h, 2 * (half_z - wall * 0.5)])
    floor.apply_translation([0, floor_h / 2.0, 0])
    geoms.append(floor)
    for sx in (+1.0, -1.0):
        w = trimesh.creation.box(extents=[wall, height, 2 * half_z])
        w.apply_translation([sx * (half_x - wall / 2.0), height / 2.0, 0])
        geoms.append(w)
    inner_x = 2 * (half_x - wall)
    for sz in (+1.0, -1.0):
        w = trimesh.creation.box(extents=[inner_x, height, wall])
        w.apply_translation([0, height / 2.0, sz * (half_z - wall / 2.0)])
        geoms.append(w)
    return geoms


def round_cup(radius, height, wall=0.10, floor_h=0.06, n_wall=24):
    geoms = []
    r_in = radius - wall
    floor = trimesh.creation.cylinder(radius=r_in, height=floor_h, sections=32)
    floor.apply_transform(RX)
    floor.apply_translation([0, floor_h / 2.0, 0])
    geoms.append(floor)
    tang = 2.0 * radius * np.sin(np.pi / n_wall) * 1.08
    r_mid = radius - wall / 2.0
    for i in range(n_wall):
        ang = 2.0 * np.pi * i / n_wall
        box = trimesh.creation.box(extents=[wall, height, tang])
        box.apply_translation([r_mid, height / 2.0, 0.0])
        box.apply_transform(trimesh.transformations.rotation_matrix(ang, [0, 1, 0]))
        geoms.append(box)
    return geoms


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Match visual AABBs (raw): id3 rectangular tray, id4 round bowl.
    _export(rect_cup(0.950, 0.685, 0.445), OUT / "base3.glb")
    _export(round_cup(0.90, 0.746), OUT / "base4.glb")


if __name__ == "__main__":
    main()
