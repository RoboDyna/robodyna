"""Shared head-camera framing and episode cutoff for suite tasks / demos.

Used by household *and* base interactive viewers, GUI ``scene_snapshot`` stills,
and household RGB demos. Keep these numbers aligned with
``assets/embodiments/ur5-wsg/config.yml`` ``head_camera`` and D435 fovy in
``task_config/_camera_config.yml``.
"""
from __future__ import annotations

import numpy as np
import sapien


class EpisodeTimeLimit(Exception):
    """Raised when a household episode hits the step cutoff while saving."""


# Fixed elevated head view (GUI snapshots + interactive free-fly default).
HEAD_CAMERA_POS = np.array([0.0, -0.50, 2.0], dtype=np.float64)
HEAD_CAMERA_FORWARD = np.array([0.0, 0.45, -1.0], dtype=np.float64)
HEAD_CAMERA_LEFT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
# D435 vertical FOV (degrees → radians).
HEAD_CAMERA_FOVY_DEG = 37.0
HEAD_CAMERA_FOVY = float(np.deg2rad(HEAD_CAMERA_FOVY_DEG))
# Sapien FPS free-fly controller values that recreate HEAD_CAMERA_* axes.
HEAD_VIEWER_XYZ = (0.0, -0.50, 2.0)
HEAD_VIEWER_RPY = (0.0, -1.14794242382892, float(-np.pi / 2.0))

# Back-compat aliases (older household call sites).
HOUSEHOLD_HEAD_POS = HEAD_CAMERA_POS
HOUSEHOLD_HEAD_FORWARD = HEAD_CAMERA_FORWARD
HOUSEHOLD_HEAD_LEFT = HEAD_CAMERA_LEFT

HOUSEHOLD_TASKS = frozenset(
    {
        "boil_milk",
        "catch_cup",
        "clean_table",
        "cook_food",
        "cook_food_timer",
        "fill_coffee_jar",
        "make_soup",
        "measure_ingredient",
        "catch_mouse_object_drop",
        "pour_beer",
        "stop_ball",
        "trap_bug",
    }
)

# Fallback episode cutoff if yaml ``eval_step_limit`` is missing (sim / control steps).
EPISODE_MAX_STEPS = 15000
# Back-compat alias used by record_demo / older callers.
HOUSEHOLD_MAX_STEPS = EPISODE_MAX_STEPS
# Kept for callers that still pass --max-seconds; unused as the household default.
HOUSEHOLD_MAX_SECONDS = 60.0


def configure_standard_head_camera(task) -> None:
    """Force ``head_camera`` to the shared suite pose + D435 fovy (base + household)."""
    cams = getattr(task, "cameras", None)
    if cams is None:
        return
    names = list(getattr(cams, "static_camera_name", []) or [])
    clist = list(getattr(cams, "static_camera_list", []) or [])
    if "head_camera" not in names:
        return
    camera = clist[names.index("head_camera")]
    forward = HEAD_CAMERA_FORWARD / np.linalg.norm(HEAD_CAMERA_FORWARD)
    left = HEAD_CAMERA_LEFT / np.linalg.norm(HEAD_CAMERA_LEFT)
    up = np.cross(forward, left)
    m = np.eye(4)
    m[:3, :3] = np.stack([forward, left, up], axis=1)
    m[:3, 3] = HEAD_CAMERA_POS
    camera.entity.set_pose(sapien.Pose(m))
    # Reset FOV in case a task previously widened/narrowed the sensor.
    try:
        camera.set_fovy(HEAD_CAMERA_FOVY)
    except Exception:
        try:
            camera.fovy = HEAD_CAMERA_FOVY
        except Exception:
            pass


def configure_household_head_camera(task) -> None:
    """Alias for :func:`configure_standard_head_camera` (legacy name)."""
    configure_standard_head_camera(task)
