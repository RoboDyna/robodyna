"""Spring-back push buttons matching ``fill_coffee_jar``'s blue key.

Position-controlled grippers do not produce reliable PhysX push-through on a
static keycap.  ``fill_coffee_jar`` therefore uses a *virtual spring*: once the
gripper tip enters a band above the key top, force grows with penetration, the
keycap pose tracks that force, and with no force the key springs back up.

This helper keeps that same visual/spring model, but fires a single action edge
when the keycap travels past a depth threshold (not four force levels).

Important: ``create_box(env, ...)`` applies ``table_z_bias`` to the entity while
callers often pass pre-bias homes.  Homes / tops are therefore taken from the
live entity pose at init so spring math and ``set_pose`` share one world frame.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import sapien

from .create_actor import create_box

# Hollow bezel defaults matching ``catch_shelf_marble`` (not a solid under-cube).
KEY_BASE_WALL_T = 0.006
KEY_BASE_HALF_Z = 0.004
KEY_BASE_MARGIN = 0.008
KEY_BASE_COLOR = (0.28, 0.28, 0.31)


def add_key_base_border(
    scene,
    cx: float,
    cy: float,
    table_z: float,
    key_half,
    *,
    wall_t: float = KEY_BASE_WALL_T,
    half_z: float = KEY_BASE_HALF_Z,
    margin: float = KEY_BASE_MARGIN,
    color=KEY_BASE_COLOR,
    name_prefix: str = "key_base",
    is_static: bool = True,
) -> list:
    """Hollow dark bezel around a keycap (four walls, open center).

    The keycap should sit on the table (center at ``table_z + key_half[2]``);
    this rim frames it without a solid black cube underneath.
    ``scene`` may be a task env or a sapien Scene (same as ``create_box``).
    """
    hx = float(key_half[0]) + float(margin)
    hy = float(key_half[1]) + float(margin)
    hz = float(half_z)
    wt = float(wall_t)
    z = float(table_z) + hz
    rgba = list(color)
    walls = [
        ("x_pos", [cx + hx - 0.5 * wt, cy, z], [0.5 * wt, hy, hz]),
        ("x_neg", [cx - hx + 0.5 * wt, cy, z], [0.5 * wt, hy, hz]),
        ("y_pos", [cx, cy + hy - 0.5 * wt, z], [hx - wt, 0.5 * wt, hz]),
        ("y_neg", [cx, cy - hy + 0.5 * wt, z], [hx - wt, 0.5 * wt, hz]),
    ]
    out = []
    for name, pos, half in walls:
        out.append(
            create_box(
                scene,
                pose=sapien.Pose(list(pos)),
                half_size=list(half),
                color=rgba,
                is_static=is_static,
                name=f"{name_prefix}_{name}",
            )
        )
    return out


class ReactivePushButtons:
    """Track one or more keycaps with fill_coffee-style spring visuals."""

    # Mirrors fill_coffee_jar.FORCE_STIFFNESS / FORCE_ENGAGE_SLACK / FORCE_THRESHOLDS[-1].
    DEFAULT_STIFFNESS = 800.0
    DEFAULT_ENGAGE_SLACK = 0.05
    DEFAULT_FORCE_FULL = 14.0
    DEFAULT_EE_TO_TCP = 0.12
    DEFAULT_VISUAL_STEP = 0.0007

    def __init__(
        self,
        env,
        *,
        actors: Sequence,
        home_poses: Sequence[sapien.Pose],
        max_depth: float,
        ids: Sequence | None = None,
        # Per-button arms that may press (e.g. ``(("left",), ("right",), ("right",))``).
        # When omitted, ids ``"left"``/``"right"`` accept only the matching arm;
        # all other ids accept either arm.
        press_arms: Sequence[Sequence[str]] | None = None,
        xy_tol: float = 0.055,
        visual_step: float = DEFAULT_VISUAL_STEP,
        force_stiffness: float = DEFAULT_STIFFNESS,
        force_engage_slack: float = DEFAULT_ENGAGE_SLACK,
        force_full: float = DEFAULT_FORCE_FULL,
        ee_to_tcp: float = DEFAULT_EE_TO_TCP,
        # Fire once the keycap has traveled this fraction of max_depth while
        # moving down.  Do not require the tip to reach the physical top — the
        # gripper usually collides before that, while the spring already moves.
        trigger_depth_frac: float = 0.35,
        # Must spring back nearly all the way before the next press edge arms.
        release_frac: float = 0.08,
    ):
        if len(actors) != len(home_poses):
            raise ValueError("actors and home_poses must have the same length")
        self.env = env
        self.visual_step = float(visual_step)
        self.force_stiffness = float(force_stiffness)
        self.force_engage_slack = float(force_engage_slack)
        self.force_full = float(max(force_full, 1e-6))
        self.ee_to_tcp = float(ee_to_tcp)
        self.max_depth = float(max(max_depth, 1e-6))
        self.xy_tol = float(xy_tol)
        self.trigger_depth = float(
            self.max_depth * np.clip(trigger_depth_frac, 0.05, 0.95)
        )
        self.release_depth = float(self.max_depth * np.clip(release_frac, 0.0, 0.95))
        if self.release_depth >= self.trigger_depth:
            self.release_depth = 0.5 * self.trigger_depth

        # Unwrap Actor wrappers and snapshot WORLD homes from the live entities.
        # ``create_box(env, pose)`` adds ``table_z_bias``; caller homes often omit it.
        self.actors = []
        self.home_poses = []
        for actor, home in zip(actors, home_poses):
            entity = actor.actor if hasattr(actor, "actor") else actor
            self.actors.append(entity)
            try:
                pose = entity.get_pose()
                self.home_poses.append(sapien.Pose(list(pose.p), list(pose.q)))
            except Exception:
                self.home_poses.append(sapien.Pose(list(home.p), list(home.q)))

        n = len(self.actors)
        if ids is None:
            self.ids = list(range(n))
        else:
            if len(ids) != n:
                raise ValueError("ids length must match button count")
            self.ids = list(ids)
        self._id_to_i = {button_id: i for i, button_id in enumerate(self.ids)}
        if press_arms is None:
            self._press_arms = None
        else:
            if len(press_arms) != n:
                raise ValueError("press_arms length must match button count")
            self._press_arms = [tuple(str(s) for s in arms) for arms in press_arms]
        self.visual_depth = [0.0] * n
        self.target_depth = [0.0] * n
        self._forced_depth = [None] * n
        self._latched = [False] * n
        self._tip_z = [None] * n
        # Key top = center + half-height (max_depth is the keycap half-z for these tasks).
        self.tops_z = [float(pose.p[2]) + self.max_depth for pose in self.home_poses]
        self._n = n

    def set_tops_z(self, tops: Sequence[float]) -> None:
        """Set absolute world-frame key-top Z.

        Values that fall below the corresponding world home (typical when a
        caller passes pre-``table_z_bias`` coordinates) are ignored in favor of
        ``home.z + max_depth``.
        """
        if len(tops) != self._n:
            raise ValueError("tops length must match button count")
        resolved = []
        for i, z in enumerate(tops):
            z = float(z)
            home_z = float(self.home_poses[i].p[2])
            if z < home_z - 1e-4:
                resolved.append(home_z + self.max_depth)
            else:
                resolved.append(z)
        self.tops_z = resolved

    def resolve_index(self, button_id) -> int:
        if isinstance(button_id, int) and button_id not in self._id_to_i:
            if 0 <= button_id < self._n:
                return button_id
            raise KeyError(button_id)
        return self._id_to_i[button_id]

    def set_forced(self, button_id, pressed: bool) -> None:
        """Force a key fully down (expert latch) or clear the override."""
        idx = self.resolve_index(button_id)
        self._forced_depth[idx] = self.max_depth if bool(pressed) else None

    def reset(self) -> None:
        self.visual_depth = [0.0] * self._n
        self.target_depth = [0.0] * self._n
        self._forced_depth = [None] * self._n
        self._latched = [False] * self._n
        self._tip_z = [None] * self._n
        for i in range(self._n):
            self._apply_pose(i, 0.0)

    def _tip_xyz(self, side: str) -> np.ndarray | None:
        """Virtual tip = EE pose lowered by ``ee_to_tcp`` (fill_coffee_jar).

        Prefer the teleop command EE while a key is held so Q/E tracks without
        link lag; otherwise use the measured EE (same as fill_coffee_jar).
        Keyboard+mouse buries the arms; their leftover EE pose must not press keys.
        """
        if bool(getattr(self.env, "_interactive_arms_removed", False)):
            return None
        side = str(side)
        cmd = getattr(self.env, "_interactive_cmd_pose", None)
        if isinstance(cmd, dict) and side in cmd:
            ee = np.asarray(cmd[side][:3], dtype=float)
            tip = ee.copy()
            tip[2] -= self.ee_to_tcp
            return tip
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return None
        try:
            pose_fn = robot.get_left_ee_pose if side == "left" else robot.get_right_ee_pose
            ee = np.asarray(pose_fn(), dtype=float)
            tip = np.asarray(ee[:3], dtype=float).copy()
            tip[2] -= self.ee_to_tcp
            return tip
        except Exception:
            try:
                pose_fn = (
                    robot.get_left_tcp_pose if side == "left" else robot.get_right_tcp_pose
                )
                return np.asarray(pose_fn(), dtype=float)[:3]
            except Exception:
                return None

    def _contact_force(self, idx: int) -> float:
        """Optional PhysX contact force against this keycap (fill_coffee style)."""
        if bool(getattr(self.env, "_interactive_arms_removed", False)):
            return 0.0
        entity = self.actors[idx]
        try:
            btn_name = str(entity.get_name() or "")
        except Exception:
            return 0.0
        if not btn_name:
            return 0.0
        robot = getattr(self.env, "robot", None)
        grip = set(getattr(robot, "gripper_name", []) or []) if robot is not None else set()
        dt = float(getattr(self.env.scene, "get_timestep", lambda: 1.0 / 250.0)())
        imp = 0.0
        try:
            for contact in self.env.scene.get_contacts():
                n0 = contact.bodies[0].entity.name
                n1 = contact.bodies[1].entity.name
                if btn_name not in (n0, n1):
                    continue
                other = n1 if n0 == btn_name else n0
                if grip and other not in grip:
                    o = other.lower()
                    if not (
                        o.startswith("left")
                        or o.startswith("right")
                        or "finger" in o
                        or "pad" in o
                        or "hand" in o
                        or "gripper" in o
                    ):
                        continue
                for point in contact.points:
                    imp += float(np.linalg.norm(point.impulse))
        except Exception:
            return 0.0
        return float(imp / max(dt, 1e-6))

    def _sides_for_button(self, idx: int) -> tuple[str, ...]:
        """Arms that may press this key.

        Default: left/right-labeled keys only accept the matching arm so an
        approaching opposite gripper cannot actuate them early.  Callers that
        put both direction keys under one arm (e.g. dispense_gummy belt keys)
        pass ``press_arms`` to override.
        """
        if self._press_arms is not None:
            return self._press_arms[idx]
        button_id = self.ids[idx]
        if button_id in ("left", "right"):
            return (str(button_id),)
        return ("left", "right")

    def _spring_depth_for_button(self, idx: int) -> tuple[float, float | None]:
        """Return (target_depth, best_tip_z) using fill_coffee spring proxy."""
        forced = self._forced_depth[idx]
        if forced is not None:
            return float(forced), None
        home_xy = np.asarray(self.home_poses[idx].p[:2], dtype=float)
        top_z = float(self.tops_z[idx])
        engage_z = top_z + self.force_engage_slack
        best_depth = 0.0
        best_tip_z = None
        best_force = 0.0
        contact_force = self._contact_force(idx)
        for side in self._sides_for_button(idx):
            tip = self._tip_xyz(side)
            if tip is None:
                continue
            if float(np.linalg.norm(tip[:2] - home_xy)) > self.xy_tol:
                continue
            tip_z = float(tip[2])
            spring_force = self.force_stiffness * max(0.0, engage_z - tip_z)
            # Contact only counts when the tip is already over the key (XY lock).
            force = max(spring_force, contact_force)
            depth = float(np.clip(force / self.force_full * self.max_depth, 0.0, self.max_depth))
            if force >= best_force:
                best_force = force
                best_depth = depth
                best_tip_z = tip_z
        return best_depth, best_tip_z

    def _apply_pose(self, idx: int, depth: float) -> None:
        entity = self.actors[idx]
        home = self.home_poses[idx]
        if entity is None or home is None:
            return
        pose = sapien.Pose(
            [float(home.p[0]), float(home.p[1]), float(home.p[2] - depth)],
            list(home.q),
        )
        try:
            entity.set_pose(pose)
        except Exception:
            pass

    def update(self) -> list:
        """Advance spring visuals; return ids that newly triggered this frame."""
        triggered = []
        for i in range(self._n):
            target, tip_z = self._spring_depth_for_button(i)
            self.target_depth[i] = target
            self._tip_z[i] = tip_z
            prev = float(self.visual_depth[i])
            step = self.visual_step
            if target > prev:
                cur = min(target, prev + step)
            elif target < prev:
                cur = max(target, prev - step)
            else:
                cur = prev
            self.visual_depth[i] = cur
            self._apply_pose(i, cur)

            # Action edge: keycap crosses the press threshold while going down.
            # Stay latched until the key fully springs back (and force is gone)
            # so holding never re-fires; release completely to press again.
            going_lower = cur > prev + 1e-9
            if going_lower and cur >= self.trigger_depth and not self._latched[i]:
                self._latched[i] = True
                triggered.append(self.ids[i])
            fully_released = cur <= self.release_depth and target <= self.release_depth
            if fully_released:
                self._latched[i] = False
        return triggered

    def held_mask(self) -> list[bool]:
        # Require a real press (past trigger depth), not a tiny approach depression.
        return [d >= self.trigger_depth for d in self.visual_depth]

    def is_held(self, button_id) -> bool:
        idx = self.resolve_index(button_id)
        return bool(self.visual_depth[idx] >= self.trigger_depth)

    def is_engaged(self, button_id) -> bool:
        """True from press trigger until the key fully springs back.

        Unlike ``is_held`` (depth ≥ trigger right now), this keeps the press
        session alive through brief tip wobble above the trigger while the
        keycap is still mid-travel — needed for hold-to-cook tasks.
        """
        idx = self.resolve_index(button_id)
        if self._forced_depth[idx] is not None:
            return True
        return bool(self._latched[idx]) or float(self.visual_depth[idx]) >= self.trigger_depth

    def held_ids(self) -> list:
        return [self.ids[i] for i, on in enumerate(self.held_mask()) if on]

    def tip_z_at_depth(self, idx: int, depth: float) -> float:
        """World tip Z that produces ``depth`` via the spring model."""
        depth = float(np.clip(depth, 0.0, self.max_depth))
        force_needed = (depth / self.max_depth) * self.force_full
        return float(
            self.tops_z[idx]
            + self.force_engage_slack
            - force_needed / max(self.force_stiffness, 1e-6)
        )

    def tip_z_at_trigger(self, idx: int) -> float:
        """World tip Z that just reaches ``trigger_depth`` via the spring model."""
        return self.tip_z_at_depth(idx, self.trigger_depth)

    def tip_z_at_full_press(self, idx: int) -> float:
        """World tip Z that fully depresses the keycap (``max_depth``)."""
        return self.tip_z_at_depth(idx, self.max_depth)

    def min_ee_z_over_key(self, xy, *, margin: float = 0.003) -> float | None:
        """Lowest allowed EE Z while ``xy`` is over any keycap, else ``None``.

        Floors teleop Q at the tip height that just reaches ``trigger_depth`` so
        the press can finish after fingers contact the keycap, without the
        continued drive down to full key travel (which collapses other joints).
        """
        xy = np.asarray(xy, dtype=float)[:2]
        floor = None
        for i in range(self._n):
            home_xy = np.asarray(self.home_poses[i].p[:2], dtype=float)
            if float(np.linalg.norm(xy - home_xy)) > self.xy_tol:
                continue
            ee_floor = self.tip_z_at_trigger(i) - float(margin) + self.ee_to_tcp
            floor = ee_floor if floor is None else min(floor, ee_floor)
        return floor

    def min_ee_z_over_pressed(self, xy, *, margin: float = 0.003) -> float | None:
        """Lowest allowed EE Z while ``xy`` is over a pressed key, else ``None``.

        Once a key has triggered (latched) or is held past the press threshold,
        keep the same trigger-depth floor — enough to hold the press, not drive
        through the keycap toward full travel.
        """
        xy = np.asarray(xy, dtype=float)[:2]
        floor = None
        for i in range(self._n):
            pressed = self._latched[i] or float(self.visual_depth[i]) >= self.trigger_depth
            if not pressed:
                continue
            home_xy = np.asarray(self.home_poses[i].p[:2], dtype=float)
            if float(np.linalg.norm(xy - home_xy)) > self.xy_tol:
                continue
            ee_floor = self.tip_z_at_trigger(i) - float(margin) + self.ee_to_tcp
            floor = ee_floor if floor is None else min(floor, ee_floor)
        return floor
