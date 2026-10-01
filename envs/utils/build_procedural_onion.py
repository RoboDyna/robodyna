#!/usr/bin/env python3
"""Build a simple white onion stand-in: cylinder with 3 top-surface rings.

Cylinder: 4.68 cm diameter, 1.2 cm height.
Three centered groove rings on the top face; each ring diameter is 70% of
the previous (30% smaller), starting from the cylinder diameter, plus one
extra ring halfway between the cylinder edge and the first scaled ring.

Writes assets/dyna_assets/270_onion_half/
"""
from __future__ import annotations

import json
import os

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OBJ_NAME = "270_onion_half"
DEST = os.path.join(ROOT, "assets", "dyna_assets", OBJ_NAME)

RADIUS = 0.0234  # 4.68 cm diameter (+30% from 3.6 cm)
HEIGHT = 0.012   # 1.2 cm tall (+20% from 1 cm)
N_RINGS = 3
RING_SCALE = 0.70          # each ring diameter = 70% of the previous
GROOVE_DEPTH = 0.0010      # 1 mm shallow groove
LINE_HALF_W = 0.0007       # half-width of each ring line
WHITE = [245, 245, 245, 255]

HAMBURG_GRASP_ROTS = [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
]
FUNCTIONAL_ROT = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
_ROT_X = trimesh.transformations.rotation_matrix(-0.5 * np.pi, [1.0, 0.0, 0.0])


def mat(R, t):
    m = np.eye(4)
    m[:3, :3] = np.array(R, float)
    m[:3, 3] = t
    return m.tolist()


def ring_diameters() -> list[float]:
    """Ring diameters on the top face (cm-scale logic, returned in metres).

    - Three inner rings: each 30% smaller than the previous, starting from the
      cylinder diameter.
    - One outer ring halfway between the cylinder edge and the first inner ring.
    """
    cyl_d = 2.0 * RADIUS
    inner: list[float] = []
    d = cyl_d
    for _ in range(N_RINGS):
        d *= RING_SCALE
        inner.append(d)
    mid_d = 0.5 * (cyl_d + inner[0])
    # Outermost → innermost for stable groove placement.
    return [mid_d, *inner]


def _groove_annulus(center_radius: float, top_y: float) -> trimesh.Trimesh | None:
    r_in = max(0.0008, center_radius - LINE_HALF_W)
    r_out = min(RADIUS - 0.0004, center_radius + LINE_HALF_W)
    if r_out <= r_in:
        return None
    groove = trimesh.creation.annulus(r_in, r_out, GROOVE_DEPTH, sections=48)
    groove.apply_transform(_ROT_X)
    groove.apply_translation([0.0, top_y - 0.5 * GROOVE_DEPTH, 0.0])
    return groove


def build_cylinder() -> trimesh.Trimesh:
    body = trimesh.creation.cylinder(radius=RADIUS, height=HEIGHT, sections=64)
    body.apply_transform(_ROT_X)
    body.apply_translation(-body.centroid)

    top_y = float(body.bounds[1][1])
    parts: list[trimesh.Trimesh] = [body]
    for d in ring_diameters():
        groove = _groove_annulus(0.5 * d, top_y)
        if groove is not None:
            parts.append(groove)

    mesh = trimesh.util.concatenate(parts)
    mesh.merge_vertices()
    col = [c / 255.0 for c in WHITE[:3]] + [1.0]
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=col)
    )
    return mesh


def build_collision_cylinder() -> trimesh.Trimesh:
    """Smooth cylinder for stable physics / grasp (no surface grooves)."""
    mesh = trimesh.creation.cylinder(radius=RADIUS, height=HEIGHT, sections=32)
    mesh.apply_transform(_ROT_X)
    mesh.apply_translation(-mesh.centroid)
    return mesh


def main() -> None:
    mesh = build_cylinder()
    col_mesh = build_collision_cylinder()
    ext = np.asarray(mesh.extents, float)
    center = (mesh.bounds[1] + mesh.bounds[0]) / 2.0

    os.makedirs(os.path.join(DEST, "visual"), exist_ok=True)
    os.makedirs(os.path.join(DEST, "collision"), exist_ok=True)
    mesh.export(os.path.join(DEST, "visual", "base0.glb"))
    col_mesh.export(os.path.join(DEST, "collision", "base0.glb"))

    ymin, ymax = float(mesh.bounds[0][1]), float(mesh.bounds[1][1])
    cx, cz = float(center[0]), float(center[2])

    model_data = {
        "center": center.tolist(),
        "extents": ext.tolist(),
        "scale": [1.0, 1.0, 1.0],
        "transform_matrix": np.eye(4).tolist(),
        "target_pose": [],
        "contact_points_pose": [mat(R, [cx, ymax, cz]) for R in HAMBURG_GRASP_ROTS],
        "functional_matrix": [mat(FUNCTIONAL_ROT, [cx, ymin, cz])],
        "orientation_point": [],
        "contact_points_group": [],
        "contact_points_mask": [],
        "target_point_discription": [],
        "contact_points_discription": ["Top-center, grasped from above."],
        "functional_point_discription": ["Bottom-center; placed onto / lift off a target."],
        "orientation_point_discription": [],
        "stable": True,
    }
    with open(os.path.join(DEST, "model_data0.json"), "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)

    points_info = {
        "contact_points": [
            {"id": [0, 1, 2, 3], "description": "Top-center", "usage": "Grasp from above."}
        ],
        "functional_points": [
            {"id": 0, "description": "Bottom-center", "usage": "Place onto / lift off a target."}
        ],
    }
    with open(os.path.join(DEST, "points_info.json"), "w", encoding="utf-8") as f:
        json.dump(points_info, f, indent=2)

    diams = ring_diameters()
    with open(os.path.join(DEST, "NOTICE"), "w", encoding="utf-8") as f:
        f.write(
            f"Asset: {OBJ_NAME}\n"
            f"Source: procedural cylinder Ø{2*RADIUS*100:.1f} cm × h{HEIGHT*100:.1f} cm\n"
            f"Top rings (cm): {', '.join(f'{d*100:.2f}' for d in diams)}\n"
            "Author: RoboDynaExp\n"
        )

    print(f"wrote {DEST}")
    print(f"  cylinder: Ø{2*RADIUS*100:.1f} cm  h={HEIGHT*100:.1f} cm")
    print(f"  ring Ø (cm): {[round(d*100, 2) for d in diams]}")


if __name__ == "__main__":
    main()
