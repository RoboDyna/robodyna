"""Flat keycap icons built from thin visual boxes (same style as catch_shelf_marble)."""
from __future__ import annotations

import numpy as np
import sapien

from .create_actor import create_visual_box, preprocess

ARROW_COLOR = [0.95, 0.95, 0.95]
DECAL_HALF_Z = 0.001
DECAL_LIFT = 0.0015


def actor_entity(actor):
    if actor is None:
        return None
    return actor.actor if hasattr(actor, "actor") else actor


def _sapien_scene(scene):
    sapien_scene, _ = preprocess(scene, sapien.Pose())
    return sapien_scene


def _place_box(scene, xyz, yaw, half_size, name):
    q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    return create_visual_box(
        scene,
        sapien.Pose([float(xyz[0]), float(xyz[1]), float(xyz[2])], q),
        half_size=list(half_size),
        color=list(ARROW_COLOR),
        name=name,
    )


def _rot_xy(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * x - s * y, s * x + c * y


def _on_off_world_parts(scene, cx, cy, cz, key_half, name, yaw=0.0):
    """IEC standby: open ring + stem, authored in XY like shelf-marble arrows."""
    s = min(float(key_half[0]), float(key_half[1]))
    radius = s * 0.52
    thick = s * 0.11
    hz = DECAL_HALF_Z
    n_pts = 11
    angles = np.deg2rad(np.linspace(-50.0, 230.0, n_pts))
    pts = [np.array([radius * np.cos(a), radius * np.sin(a)]) for a in angles]
    parts = []

    def segment(p0, p1, half_thick, suffix):
        mid = (p0 + p1) / 2.0
        delta = p1 - p0
        seg_len = float(np.linalg.norm(delta))
        if seg_len < 1e-6:
            return
        lx, ly = _rot_xy(mid[0], mid[1], yaw)
        heading = float(np.arctan2(delta[1], delta[0])) + float(yaw)
        ent = _place_box(
            scene,
            [cx + lx, cy + ly, cz],
            heading,
            [seg_len / 2.0 + half_thick * 0.35, half_thick, hz],
            f"{name}_{suffix}",
        )
        parts.append(ent)

    for i, (p0, p1) in enumerate(zip(pts[:-1], pts[1:])):
        segment(p0, p1, thick, f"ring_{i}")

    y0 = -radius * 0.12
    y1 = -radius * 1.08
    stem_mid = 0.5 * (y0 + y1)
    stem_len = abs(y1 - y0)
    lx, ly = _rot_xy(0.0, stem_mid, yaw)
    ent = _place_box(
        scene,
        [cx + lx, cy + ly, cz],
        np.pi / 2.0 + float(yaw),
        [stem_len / 2.0, thick * 0.95, hz],
        f"{name}_stem",
    )
    parts.append(ent)
    return parts


def _push_world_parts(scene, cx, cy, cz, key_half, name, yaw=0.0):
    """Download-style push arrow: shelf-marble chevron plus a tip bar, pointing -Y."""
    s = min(float(key_half[0]), float(key_half[1])) / 0.028
    # Author a left-pointing arrow (same parts as catch_shelf_marble), then rotate
    # +90° so it points toward the robot, and add the landing bar at the tip.
    rotation = np.pi / 2.0 + float(yaw)
    templates = [
        ("shaft", [0.003, 0.0], 0.0, [0.013, 0.0025, 0.001]),
        ("head_upper", [-0.011, 0.005], 0.75, [0.009, 0.0025, 0.001]),
        ("head_lower", [-0.011, -0.005], -0.75, [0.009, 0.0025, 0.001]),
        ("bar", [-0.021, 0.0], np.pi / 2.0, [0.008, 0.0025, 0.001]),
    ]
    parts = []
    for part_name, (lx, ly), local_yaw, half in templates:
        x, y = _rot_xy(lx * s, ly * s, rotation)
        box_yaw = local_yaw + rotation
        half_size = [half[0] * s, half[1] * s, half[2]]
        parts.append(
            _place_box(
                scene,
                [cx + x, cy + y, cz],
                box_yaw,
                half_size,
                f"{name}_{part_name}",
            )
        )
    return parts


def attach_key_symbol(scene, actor, key_half, kind: str, name: str):
    """Build an on-key icon; returns ``(parts, local_poses)`` in the actor frame."""
    entity = actor_entity(actor)
    if entity is None:
        return [], []
    try:
        wp = entity.get_pose()
    except Exception:
        return [], []
    cz = float(wp.p[2]) + float(key_half[2]) + DECAL_LIFT
    sapien_scene = _sapien_scene(scene)
    kind = "on_off" if kind in ("on_off", "on-off", "switch") else "push"
    if kind == "on_off":
        world_parts = _on_off_world_parts(
            sapien_scene,
            float(wp.p[0]),
            float(wp.p[1]),
            cz,
            key_half,
            name,
            yaw=np.pi,
        )
    else:
        world_parts = _push_world_parts(
            sapien_scene,
            float(wp.p[0]),
            float(wp.p[1]),
            cz,
            key_half,
            name,
        )
    locals_ = []
    inv = wp.inv()
    for part in world_parts:
        local = inv * part.get_pose()
        locals_.append(sapien.Pose(list(local.p), list(local.q)))
    return world_parts, locals_


def sync_key_symbol(parts, local_poses, actor) -> None:
    if not parts:
        return
    entity = actor_entity(actor)
    if entity is None:
        return
    try:
        wp = entity.get_pose()
    except Exception:
        return
    for part, local in zip(parts, local_poses or []):
        try:
            part.set_pose(wp * local)
        except Exception:
            pass
