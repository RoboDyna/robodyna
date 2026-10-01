"""Empty table + dual UR5 playground for the interactive tutorial.

Parts 1–2 leave the table bare. Part 3 spawns one prop at a time
(cube, hold button, on/off switch, push box) via ``tutorial_set_stage``.
Part 4 reuses the same hook for advanced actions (rolling ball, stove
knob, mallet, multi-stage force key).
"""
from __future__ import annotations

import sapien
import sapien.physx
import sapien.render
import numpy as np

from ._base_task import Base_Task
from ._kitchens_base_task import KitchenS_base_task
from .utils.actor_utils import Actor
from .utils.create_actor import create_box, create_sphere, preprocess
from .utils.reactive_button import ReactivePushButtons, add_key_base_border
from .utils.key_symbol import attach_key_symbol, sync_key_symbol


class _ForceKeyBank(ReactivePushButtons):
    """fill_coffee-style spring key that reports live force and allows full travel."""

    def live_force(self, button_id=0) -> float:
        idx = self.resolve_index(button_id)
        home_xy = np.asarray(self.home_poses[idx].p[:2], dtype=float)
        top_z = float(self.tops_z[idx])
        engage_z = top_z + self.force_engage_slack
        best = 0.0
        contact = self._contact_force(idx)
        for side in self._sides_for_button(idx):
            tip = self._tip_xyz(side)
            if tip is None:
                continue
            if float(np.linalg.norm(tip[:2] - home_xy)) > self.xy_tol:
                continue
            spring = self.force_stiffness * max(0.0, engage_z - float(tip[2]))
            best = max(best, spring, contact)
        return float(best)

    def min_ee_z_over_key(self, xy, *, margin: float = 0.003) -> float | None:
        """Floor Q at full force (level 4), not the first trigger depth."""
        xy = np.asarray(xy, dtype=float)[:2]
        floor = None
        for i in range(self._n):
            home_xy = np.asarray(self.home_poses[i].p[:2], dtype=float)
            if float(np.linalg.norm(xy - home_xy)) > self.xy_tol:
                continue
            ee_floor = self.tip_z_at_full_press(i) - float(margin) + self.ee_to_tcp
            floor = ee_floor if floor is None else min(floor, ee_floor)
        return floor


class tutorial_empty(Base_Task):
    """Bare tabletop with both arms at home. Used by tutorial parts 1–4."""

    # Left-side workspace so the selected left arm can reach every prop.
    PROP_X = -0.16
    PROP_Y = -0.08
    TABLE_Z = 0.74  # create_box adds table_z_bias
    EE_TO_TCP = 0.12

    CUBE_HALF = 0.022
    CUBE_COLOR = (0.95, 0.55, 0.12)
    CUBE_MASS = 0.05
    CUBE_LIFT = 0.035  # m above rest before a pick counts
    GRASP_CONFIRM_STEPS = 20

    KEY_HALF = (0.020, 0.020, 0.014)
    KEY_COLOR_UP = (0.18, 0.78, 0.28)
    KEY_COLOR_DOWN = (0.85, 0.10, 0.10)
    KEY_XY_TOL = 0.06
    HOLD_MIN_STEPS = 60  # ~0.24 s at 250 Hz — a tap does not count
    HOLD_RELEASE_STEPS = 12

    PUSH_HALF = (0.035, 0.035, 0.022)
    PUSH_COLOR = (0.28, 0.68, 0.82)
    PUSH_MASS = 0.10
    PUSH_Y = -0.14
    PUSH_GOAL_DY = 0.10
    TARGET_COLOR = (0.15, 0.85, 0.25)
    PUSH_CONFIRM_STEPS = 20
    SETTLE_STEPS = 45  # ignore success for a beat after a new prop appears

    # Part 4 — rolling ball (toward the robot, −Y). Pack-fruits style: keep
    # sliding until grasped; only respawn if it falls off the table.
    BALL_RADIUS = 0.024
    BALL_COLOR = (0.92, 0.18, 0.16)
    BALL_MASS = 0.12
    BALL_X = -0.16
    BALL_Y0 = 0.16
    BALL_Y_END = -0.28
    BALL_SPEED = 0.10  # m/s along −Y
    BALL_LIFT = 0.030
    BALL_TABLE_X_LIM = 0.45
    BALL_TABLE_Y_MIN = -0.38
    BALL_TABLE_Y_MAX = 0.38
    BALL_RESPAWN_SETTLE = 15
    BALL_OFF_Z = 0.10  # below rest before counting as fallen

    # Part 4 — KitchenS cooktop (knob on the left workspace).
    STOVE_XY = (-0.28, 0.10)
    STOVE_BURNER = "left_front"
    KNOB_MAX_ANGLE = float(np.pi / 2)
    STOVE_ON_FRAC = 0.25
    STOVE_OFF_FRAC = 0.08

    # Part 4 — mallet (same geometry as whack_moles).
    MALLET_HEAD_RADIUS = 0.050
    MALLET_HEAD_HALF_X = 0.030
    MALLET_HANDLE_RADIUS = 0.012
    MALLET_HANDLE_HALF_Y = 0.130
    MALLET_HEAD_Y = -0.105
    MALLET_HANDLE_CENTER_Y = -0.025
    MALLET_WOOD_COLOR = [0.48, 0.25, 0.10]
    MALLET_REST_HEIGHT = 0.115
    MALLET_LEVEL_Q = [0.0, 0.70710678, 0.0, 0.70710678]
    MALLET_REST_POST_Y = (-0.065, 0.035)
    MALLET_MASS = 0.06
    MALLET_LIFT = 0.040

    # Part 4 — force key: clear levels 1→2→3→4 (yellow band advances).
    FORCE_THRESHOLDS = (3.0, 6.0, 10.0, 14.0)
    FORCE_TARGET_LEVEL = 1
    FORCE_KEY_COLOR = (0.08, 0.36, 0.95)
    FORCE_LEVEL_COLORS = (
        (0.08, 0.36, 0.95),
        (0.18, 0.78, 0.28),
        (0.92, 0.78, 0.12),
        (0.95, 0.50, 0.10),
        (0.85, 0.10, 0.10),
    )
    # Mild slowdown over the key so bands are readable (fill_coffee uses 0.22).
    DISPENSE_Z_DOWN_SCALE = 0.70
    # Park unused props off-table. Never scene.remove_entity at runtime —
    # deleting a body still in gripper contact segfaults PhysX (exit -11).
    HIDE_XY = (1.85, 1.85)

    def setup_demo(self, **kwags):
        self._tutorial_complete = False
        self._tutorial_stage = None
        self._props_ready = False
        self.arm = "left"
        self._tutorial_actors = []
        self._reactive_buttons = None
        self._cube = None
        self._cube_rest_z = None
        self._cube_table_pose = None
        self._push_box = None
        self._push_goal_y = None
        self._push_table_pose = None
        self._push_goal = None
        self._push_goal_table_pose = None
        self._key_actor = None
        self._key_bezel = None
        self._key_table_pose = None
        self._bezel_table_poses = []
        self._key_shapes = []
        self._key_color_down = None
        self._key_symbol_parts = []
        self._key_symbol_locals = []
        self._key_symbol_kind = None
        self._hold_was_pressed = False
        self._hold_press_steps = 0
        self._hold_release_steps = 0
        self._hold_complete = False
        self._switch_on = False
        self._switch_complete = False
        self._switch_touch_latched = False
        self._grasp_hold_steps = 0
        self._grasp_complete = False
        self._push_hold_steps = 0
        self._push_complete = False
        self._stage_settle = 0
        self._defer_hide_key = False
        self._defer_hide_ball = False
        self._ball = None
        self._ball_rigid = None
        self._ball_start = None
        self._ball_rest_z = None
        self._ball_complete = False
        self._ball_ever_lifted = False
        self._ball_grasp_armed = False
        self._ball_grip_mat = None
        self._mallet = None
        self._mallet_rests = []
        self._mallet_table_pose = None
        self._mallet_rest_poses = []
        self._mallet_rest_z = None
        self._mallet_complete = False
        self._force_key_actor = None
        self._force_key_bezel = None
        self._force_key_table_pose = None
        self._force_key_bezel_poses = []
        self._stove_loaded = False
        self._stove_range_home = None
        self._stove_knob_home = None
        self._stove_cover_homes = {}
        self._stove_complete = False
        self._stove_turned_on = False
        self._stove_turned_off = False
        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self.stove_knob_articulation = None
        self._knob_grasp_active = False
        self._knob_clutch_engaged = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._policy_controlling_knob = False
        self._knob_grasp_arm = None
        self._last_committed_knob_angle = None
        self._force_complete = False
        self._force_target_level = int(self.FORCE_TARGET_LEVEL)
        self._force_cleared = 0
        self._force_step_armed = True  # must release after a hit before next counts
        self._key_force_n = 0.0
        self._press_peak_force = 0.0
        self._press_active = False
        self._force_feedback = ""
        self._prop_side = -1.0
        self._cube_reset_pose = None
        self._push_reset_pose = None
        self._mallet_reset_pose = None
        self._mallet_grasp_armed = False
        self._ball_lane_x = float(self.BALL_X)
        self._burner_xy_home = None
        super()._init_task_env_(**kwags)

    def load_actors(self):
        return

    def play_once(self):
        pass

    def check_success(self):
        return bool(getattr(self, "_tutorial_complete", False))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._maybe_park_key()
        self._maybe_park_ball()
        stage = getattr(self, "_tutorial_stage", None)
        if not stage:
            return
        if stage == "grasp":
            self._tick_grasp()
        elif stage in ("hold", "switch"):
            self._tick_buttons()
        elif stage == "push":
            self._tick_push()
        elif stage == "ball":
            self._tick_ball()
        elif stage == "stove":
            self._tick_stove()
        elif stage == "mallet":
            self._tick_mallet()
        elif stage == "force_key":
            self._tick_force_key()

    # ---------------------------------------------------------- stage props
    def tutorial_set_stage(self, stage: str | None) -> None:
        """Show the next prop on the side opposite the gripper.

        First stage of a part stays on the left slot. Later stages flip so
        the new prop does not spawn under the current EE. Hold → switch
        keeps the same key in place.
        """
        prev = self._tutorial_stage
        self._reactive_buttons = None
        self._hold_was_pressed = False
        self._hold_press_steps = 0
        self._hold_release_steps = 0
        self._hold_complete = False
        self._switch_on = False
        self._switch_complete = False
        self._switch_touch_latched = False
        self._grasp_hold_steps = 0
        self._grasp_complete = False
        self._push_hold_steps = 0
        self._push_complete = False
        self._ball_complete = False
        self._ball_ever_lifted = False
        self._mallet_complete = False
        self._stove_complete = False
        self._stove_turned_on = False
        self._stove_turned_off = False
        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self._knob_grasp_active = False
        self._knob_clutch_engaged = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._policy_controlling_knob = False
        self._knob_grasp_arm = None
        self._last_committed_knob_angle = None
        self._force_complete = False
        self._force_target_level = 1
        self._force_cleared = 0
        self._force_step_armed = True
        self._key_force_n = 0.0
        self._press_peak_force = 0.0
        self._press_active = False
        self._force_feedback = ""
        self._stage_settle = int(self.SETTLE_STEPS) if stage else 0
        self._tutorial_stage = stage
        keep_side = stage == "switch" and prev == "hold"
        if not keep_side:
            self._prop_side = -1.0 if prev is None else self._side_sign()
        if stage in ("grasp", "hold", "switch", "push"):
            self._ensure_props()
        # Do not teleport the key out from under the gripper — that PhysX
        # contact break is the green-screen SIGSEGV (-11) after a hold release.
        if stage == "switch" and prev == "hold" and self._key_actor is not None:
            self._init_key_bank("switch")
            self._set_key_color(False)
            self._attach_key_symbol("on_off")
            return
        if stage == "push" and prev in ("hold", "switch") and self._key_actor is not None:
            self._show_push_box()
            self._defer_hide_key = True
            return
        # Do not teleport the ball out of a closed gripper — same PhysX SIGSEGV.
        leaving_ball = prev == "ball" and stage not in (None, "ball")
        self._defer_hide_ball = bool(
            leaving_ball and self._ball is not None and self._ball_gripper_contact()
        )
        self._defer_hide_key = False
        self._hide_all_props()
        if stage == "grasp":
            self._show_cube()
        elif stage == "hold":
            self._show_key("hold")
        elif stage == "switch":
            self._show_key("switch")
        elif stage == "push":
            self._show_push_box()
        elif stage == "ball":
            self._defer_hide_ball = False
            # Preload later stages while the table is empty — never build a
            # cooktop / mallet / key mid-episode under a held gripper.
            self._ensure_ball()
            self._ensure_stove()
            self._hide_stove()
            self._ensure_mallet()
            self._hide_obj(self._mallet, 11)
            self._hide_obj(getattr(self, "_mallet_rests", None), 12)
            self._ensure_force_key()
            self._hide_obj(getattr(self, "_force_key_actor", None), 16)
            self._hide_obj(getattr(self, "_force_key_bezel", None), 17)
            self._respawn_ball()
        elif stage == "stove":
            # Stove must already be loaded (from ball stage). Only unhide.
            if not self._stove_loaded:
                self._ensure_stove()
            self._show_stove()
        elif stage == "mallet":
            self._ensure_mallet()
            self._show_mallet()
        elif stage == "force_key":
            self._ensure_force_key()
            self._show_force_key()

    def tutorial_stage_complete(self) -> bool:
        stage = self._tutorial_stage
        if stage == "grasp":
            return bool(self._grasp_complete)
        if stage == "hold":
            return bool(self._hold_complete)
        if stage == "switch":
            return bool(self._switch_complete)
        if stage == "push":
            return bool(self._push_complete)
        if stage == "ball":
            return bool(self._ball_complete)
        if stage == "stove":
            return bool(self._stove_complete)
        if stage == "mallet":
            return bool(self._mallet_complete)
        if stage == "force_key":
            return bool(self._force_complete)
        return False

    def _ensure_props(self) -> None:
        """Spawn every prop once, then hide all but the active stage."""
        if self._props_ready:
            return
        self._spawn_key_hidden()
        self._spawn_push_hidden()
        self._spawn_cube()
        self._props_ready = True

    def _iter_entities(self, obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._iter_entities(item)
            return
        yield obj.actor if hasattr(obj, "actor") else obj

    def _pose_copy(self, ent) -> sapien.Pose:
        pose = ent.get_pose()
        return sapien.Pose(list(pose.p), list(pose.q))

    def _hide_pose(self, ent, slot: int = 0) -> sapien.Pose:
        z = 0.05 + 0.04 * slot
        return sapien.Pose(
            [self.HIDE_XY[0] + 0.08 * slot, self.HIDE_XY[1], z], [1, 0, 0, 0]
        )

    def _zero_velocity(self, ent) -> None:
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_linear_velocity([0.0, 0.0, 0.0])
                    comp.set_angular_velocity([0.0, 0.0, 0.0])
        except Exception:
            pass

    def _hide_entity(self, ent, slot: int = 0) -> None:
        try:
            ent.set_pose(self._hide_pose(ent, slot))
        except Exception:
            return
        self._zero_velocity(ent)

    def _hide_obj(self, obj, slot0: int = 0) -> None:
        for i, ent in enumerate(self._iter_entities(obj)):
            self._hide_entity(ent, slot0 + i)

    def _apply_pose(self, obj, pose: sapien.Pose | None) -> None:
        if obj is None or pose is None:
            return
        ent = next(self._iter_entities(obj), None)
        if ent is None:
            return
        try:
            ent.set_pose(pose)
        except Exception:
            return
        self._zero_velocity(ent)

    def _gripper_xy(self):
        robot = getattr(self, "robot", None)
        if robot is None:
            return None
        sides = tuple(getattr(self, "_interactive_selected_arms", ()) or ())
        if not sides:
            sides = ("left",)
        pts = []
        for side in sides:
            getter = getattr(robot, f"get_{side}_ee_pose", None)
            if getter is None:
                continue
            try:
                xyz = np.asarray(getter(), dtype=float)[:3]
            except Exception:
                continue
            pts.append(xyz[:2])
        if not pts:
            return None
        return np.mean(np.stack(pts, axis=0), axis=0)

    def _side_sign(self) -> float:
        """+1 = right table slot, −1 = left. Opposite the current gripper."""
        xy = self._gripper_xy()
        if xy is None:
            return -1.0
        return 1.0 if float(xy[0]) < 0.0 else -1.0

    def _dx_for(self, home_x: float) -> float:
        side = float(getattr(self, "_prop_side", -1.0))
        target = abs(float(home_x)) * side
        return target - float(home_x)

    def _offset_x(self, pose: sapien.Pose | None, dx: float) -> sapien.Pose | None:
        if pose is None:
            return None
        p = list(pose.p)
        p[0] = float(p[0]) + float(dx)
        return sapien.Pose(p, list(pose.q))

    def _actor_xyz(self, obj):
        ent = next(self._iter_entities(obj), None) if obj is not None else None
        if ent is None:
            return None
        try:
            return np.asarray(ent.get_pose().p, dtype=float)
        except Exception:
            return None

    def _off_workspace(self, p) -> bool:
        if p is None:
            return True
        if abs(float(p[0])) > float(self.BALL_TABLE_X_LIM):
            return True
        if float(p[1]) < float(self.BALL_TABLE_Y_MIN) or float(p[1]) > float(
            self.BALL_TABLE_Y_MAX
        ):
            return True
        table_z = float(self.TABLE_Z) + float(getattr(self, "table_z_bias", 0.0))
        if float(p[2]) < table_z - 0.05:
            return True
        return False

    def _gripper_touching(self, obj) -> bool:
        name = self._object_name(obj)
        if not name:
            return False
        try:
            return len(self.get_gripper_actor_contact_position(name)) > 0
        except Exception:
            return False

    def _reset_if_lost(self, obj, pose: sapien.Pose | None, *, in_hand: bool) -> None:
        """Put a thrown / fallen primary object back on its current slot."""
        if in_hand or obj is None or pose is None:
            return
        if self._gripper_touching(obj):
            return
        p = self._actor_xyz(obj)
        if p is not None and not self._off_workspace(p):
            return
        self._apply_pose(obj, pose)

    def _hide_all_props(self) -> None:
        self._hide_obj(self._cube, 0)
        self._hide_obj(self._key_actor, 1)
        self._hide_obj(self._key_bezel, 2)
        self._hide_key_symbols()
        self._hide_obj(self._push_box, 8)
        self._hide_obj(self._push_goal, 9)
        if not getattr(self, "_defer_hide_ball", False):
            self._park_ball_safe()
        self._hide_obj(self._mallet, 11)
        self._hide_obj(getattr(self, "_mallet_rests", None), 12)
        self._hide_obj(getattr(self, "_force_key_actor", None), 16)
        self._hide_obj(getattr(self, "_force_key_bezel", None), 17)
        hide_stove = getattr(self, "_hide_stove", None)
        if callable(hide_stove):
            hide_stove()

    def _gripper_near_key(self) -> bool:
        ent = next(self._iter_entities(self._key_actor), None)
        if ent is None:
            return False
        try:
            home = np.asarray(ent.get_pose().p[:2], dtype=float)
        except Exception:
            return False
        robot = getattr(self, "robot", None)
        if robot is None:
            return False
        for getter in (
            getattr(robot, "get_left_ee_pose", None),
            getattr(robot, "get_right_ee_pose", None),
        ):
            if getter is None:
                continue
            try:
                tip = np.asarray(getter(), dtype=float)[:3].copy()
            except Exception:
                continue
            tip[2] -= float(self.EE_TO_TCP)
            if float(np.linalg.norm(tip[:2] - home)) < float(self.KEY_XY_TOL) + 0.03:
                return True
        return False

    def _maybe_park_key(self) -> None:
        if not getattr(self, "_defer_hide_key", False):
            return
        if self._gripper_near_key():
            return
        self._hide_obj(self._key_actor, 1)
        self._hide_obj(self._key_bezel, 2)
        self._hide_key_symbols()
        self._defer_hide_key = False

    def _park_ball_safe(self) -> None:
        """Make the ball kinematic, then park it off-table (safe under a gripper)."""
        ball = self._ball
        rigid = self._ball_rigid
        if ball is None:
            return
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                rigid.set_linear_velocity([0.0, 0.0, 0.0])
                rigid.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass
        self._ball_grasp_armed = False
        self._hide_obj(ball, 10)

    def _maybe_park_ball(self) -> None:
        if not getattr(self, "_defer_hide_ball", False):
            return
        # Wait until jaws no longer contact the ball, then park kinematically.
        if self._ball_gripper_contact():
            return
        self._park_ball_safe()
        self._defer_hide_ball = False

    def _show_cube(self) -> None:
        dx = self._dx_for(self.PROP_X)
        pose = self._offset_x(self._cube_table_pose, dx)
        self._apply_pose(self._cube, pose)
        self._cube_reset_pose = pose
        if self._cube is not None:
            self._cube_rest_z = float(self._cube.get_pose().p[2])

    def _show_key(self, button_id: str) -> None:
        dx = self._dx_for(self.PROP_X)
        self._apply_pose(self._key_actor, self._offset_x(self._key_table_pose, dx))
        if self._key_bezel is not None and self._bezel_table_poses:
            for ent, pose in zip(
                self._iter_entities(self._key_bezel), self._bezel_table_poses
            ):
                try:
                    ent.set_pose(self._offset_x(pose, dx))
                except Exception:
                    pass
        self._init_key_bank(button_id)
        self._key_color_down = None
        self._set_key_color(False)
        if button_id == "hold":
            self._attach_key_symbol("push")
        elif button_id == "switch":
            self._attach_key_symbol("on_off")
        else:
            self._hide_key_symbols()

    def _show_push_box(self) -> None:
        dx = self._dx_for(self.PROP_X)
        pose = self._offset_x(self._push_table_pose, dx)
        self._apply_pose(self._push_box, pose)
        self._apply_pose(self._push_goal, self._offset_x(self._push_goal_table_pose, dx))
        self._push_reset_pose = pose
        if self._push_goal_y is not None and self._push_table_pose is not None:
            self._push_goal_y = float(self._push_table_pose.p[1]) + float(self.PUSH_GOAL_DY)

    def _init_key_bank(self, button_id: str) -> None:
        key = self._key_actor
        if key is None:
            self._reactive_buttons = None
            return
        live = self._pose_copy(next(self._iter_entities(key)))
        hz = float(self.KEY_HALF[2])
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[key],
            home_poses=[live],
            max_depth=hz,
            ids=[button_id],
            press_arms=(("left", "right"),),
            xy_tol=float(self.KEY_XY_TOL),
        )
        self._reactive_buttons.set_tops_z([float(live.p[2]) + hz])

    def _hide_key_symbols(self) -> None:
        for i, part in enumerate(self._key_symbol_parts or []):
            self._hide_entity(part, 40 + i)
        self._key_symbol_parts = []
        self._key_symbol_locals = []
        self._key_symbol_kind = None

    def _attach_key_symbol(self, kind: str) -> None:
        if self._key_symbol_kind == kind and self._key_symbol_parts:
            return
        self._hide_key_symbols()
        if self._key_actor is None:
            return
        parts, locals_ = attach_key_symbol(
            self, self._key_actor, self.KEY_HALF, kind, f"tutorial_key_symbol_{kind}"
        )
        self._key_symbol_parts = parts
        self._key_symbol_locals = locals_
        self._key_symbol_kind = kind

    def _sync_key_symbol(self) -> None:
        sync_key_symbol(
            self._key_symbol_parts, self._key_symbol_locals, self._key_actor
        )

    def _spawn_cube(self) -> None:
        hz = float(self.CUBE_HALF)
        cube = create_box(
            self,
            pose=sapien.Pose(
                [self.PROP_X, self.PROP_Y, self.TABLE_Z + hz], [1, 0, 0, 0]
            ),
            half_size=[hz, hz, hz],
            color=list(self.CUBE_COLOR),
            name="tutorial_cube",
            is_static=False,
        )
        cube.set_name("tutorial_cube")
        cube.set_mass(float(self.CUBE_MASS))
        self._cube = cube
        self._cube_table_pose = self._pose_copy(cube.actor)
        self._cube_rest_z = float(self._cube_table_pose.p[2])

    def _spawn_key_hidden(self) -> None:
        hz = float(self.KEY_HALF[2])
        x, y = float(self.PROP_X), float(self.PROP_Y)
        z0 = float(self.TABLE_Z)
        bezel = add_key_base_border(
            self,
            x,
            y,
            z0,
            self.KEY_HALF,
            name_prefix="tutorial_key_base",
        )
        home = sapien.Pose([x, y, z0 + hz], [1, 0, 0, 0])
        key = create_box(
            self,
            pose=home,
            half_size=list(self.KEY_HALF),
            color=list(self.KEY_COLOR_UP),
            name="tutorial_key",
            is_static=True,
        )
        self._key_actor = key
        self._key_bezel = bezel
        self._key_table_pose = self._pose_copy(key.actor)
        self._bezel_table_poses = [
            self._pose_copy(ent) for ent in self._iter_entities(bezel)
        ]
        self._key_shapes = self._render_shapes(key)
        self._hide_obj(key, 1)
        self._hide_obj(bezel, 2)

    def _spawn_push_hidden(self) -> None:
        hx, hy, hz = (float(v) for v in self.PUSH_HALF)
        y0 = float(self.PUSH_Y)
        goal_y = y0 + float(self.PUSH_GOAL_DY)
        box = create_box(
            self,
            pose=sapien.Pose(
                [self.PROP_X, y0, self.TABLE_Z + hz], [1, 0, 0, 0]
            ),
            half_size=[hx, hy, hz],
            color=list(self.PUSH_COLOR),
            name="tutorial_push_box",
            is_static=False,
        )
        box.set_name("tutorial_push_box")
        box.set_mass(float(self.PUSH_MASS))
        self._push_box = box
        self._push_table_pose = self._pose_copy(box.actor)
        self._push_goal_y = goal_y
        goal = self._spawn_visual_box(
            sapien.Pose(
                [self.PROP_X, goal_y, self.TABLE_Z + 0.0015], [1, 0, 0, 0]
            ),
            half_size=[hx + 0.04, 0.008, 0.0015],
            color=list(self.TARGET_COLOR),
            name="tutorial_push_goal",
        )
        self._push_goal = goal
        self._push_goal_table_pose = self._pose_copy(goal)
        self._hide_obj(box, 8)
        self._hide_obj(goal, 9)

    def _spawn_visual_box(self, pose, half_size, color, name: str):
        scene, pose = preprocess(self, pose)
        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(pose)
        material = sapien.render.RenderMaterial(base_color=[*list(color)[:3], 1])
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(list(half_size), material))
        entity.add_component(render)
        scene.add_entity(entity)
        entity.set_pose(pose)
        return entity

    # ------------------------------------------------------- success ticks
    def _tick_grasp(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        self._reset_if_lost(
            self._cube, self._cube_reset_pose, in_hand=self._cube_picked_up()
        )
        if self._cube_picked_up():
            self._grasp_hold_steps += 1
            if self._grasp_hold_steps >= int(self.GRASP_CONFIRM_STEPS):
                self._grasp_complete = True
        else:
            self._grasp_hold_steps = 0

    def _cube_picked_up(self) -> bool:
        cube = self._cube
        rest = self._cube_rest_z
        if cube is None or rest is None:
            return False
        if float(cube.get_pose().p[2]) < float(rest) + float(self.CUBE_LIFT):
            return False
        closing = False
        try:
            robot = self.robot
            closing = (
                float(robot.left_gripper_val) < 0.55
                or float(robot.right_gripper_val) < 0.55
            )
        except Exception:
            try:
                closing = bool(
                    self.is_left_gripper_close() or self.is_right_gripper_close()
                )
            except Exception:
                return False
        if not closing:
            return False
        try:
            contacts = self.get_gripper_actor_contact_position(cube.get_name())
        except Exception:
            return False
        return len(contacts) > 0

    def _tick_push(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        box = self._push_box
        goal = self._push_goal_y
        if box is None or goal is None:
            return
        self._reset_if_lost(box, self._push_reset_pose, in_hand=False)
        if float(box.get_pose().p[1]) >= float(goal) - 0.01:
            self._push_hold_steps += 1
            if self._push_hold_steps >= int(self.PUSH_CONFIRM_STEPS):
                self._push_complete = True
        else:
            self._push_hold_steps = 0

    def _tick_buttons(self) -> None:
        bank = self._reactive_buttons
        if bank is None:
            return
        self._sync_key_symbol()
        stage = self._tutorial_stage
        if self._stage_settle > 0:
            self._stage_settle -= 1
            if stage == "switch":
                bank.set_forced("switch", False)
            bank.update()
            if stage == "switch":
                still = False
                try:
                    still = bool(bank.is_engaged("switch"))
                except Exception:
                    still = False
                self._switch_touch_latched = still or self._key_tip_pressing("switch")
            return
        if stage == "hold":
            bank.update()
            engaged = bool(bank.is_engaged("hold"))
            self._set_key_color(engaged or float(bank.visual_depth[0]) > 1e-4)
            if engaged:
                self._hold_was_pressed = True
                self._hold_press_steps += 1
                self._hold_release_steps = 0
            elif self._hold_was_pressed and not self._hold_complete:
                self._hold_release_steps += 1
                held_long_enough = self._hold_press_steps >= int(self.HOLD_MIN_STEPS)
                if (
                    held_long_enough
                    and self._hold_release_steps >= int(self.HOLD_RELEASE_STEPS)
                ):
                    self._hold_complete = True
            return

        # Latch on/off switch (cook_meat default): press → ON (stays down,
        # red); press again → OFF (springs up, green).
        bank.set_forced("switch", self._switch_on)
        triggered = set(bank.update())
        touching = self._key_tip_pressing("switch")
        if not self._switch_on:
            if "switch" in triggered:
                self._switch_on = True
                self._switch_touch_latched = True
            else:
                self._switch_touch_latched = touching
        else:
            if touching and not self._switch_touch_latched:
                self._switch_on = False
                self._switch_complete = True
                bank.set_forced("switch", False)
            self._switch_touch_latched = touching
        down = bool(self._switch_on)
        if bank.visual_depth:
            down = down or float(bank.visual_depth[0]) > 1e-4
        self._set_key_color(down)

    def _key_tip_pressing(self, button_id: str) -> bool:
        bank = self._reactive_buttons
        if bank is None:
            return False
        try:
            idx = bank.resolve_index(button_id)
        except Exception:
            return False
        tip = None
        for side in bank._sides_for_button(idx):
            candidate = bank._tip_xyz(side)
            if candidate is None:
                continue
            home_xy = np.asarray(bank.home_poses[idx].p[:2], dtype=float)
            if float(np.linalg.norm(candidate[:2] - home_xy)) > float(bank.xy_tol):
                continue
            tip = candidate
            break
        if tip is None:
            return False
        top_z = float(bank.tops_z[idx])
        force = float(bank.force_stiffness) * max(
            0.0, top_z + float(bank.force_engage_slack) - float(tip[2])
        )
        engage = float(bank.force_full) * (
            float(bank.trigger_depth) / max(float(bank.max_depth), 1e-6)
        )
        return force >= engage

    def _render_shapes(self, actor) -> list:
        entity = actor.actor if hasattr(actor, "actor") else actor
        shapes = []
        try:
            for comp in entity.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    shapes.extend(list(comp.render_shapes))
        except Exception:
            return []
        return shapes

    def _set_key_color(self, down: bool) -> None:
        down = bool(down)
        if self._key_color_down is not None and bool(self._key_color_down) == down:
            return
        self._key_color_down = down
        rgb = self.KEY_COLOR_DOWN if down else self.KEY_COLOR_UP
        color = list(rgb) + [1.0]
        for shape in self._key_shapes:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _get_rigid(self, obj):
        ent = obj.actor if hasattr(obj, "actor") else obj
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    return comp
        except Exception:
            return None
        return None

    def _object_name(self, obj) -> str:
        if obj is None:
            return ""
        try:
            if hasattr(obj, "get_name"):
                return str(obj.get_name() or "")
        except Exception:
            pass
        ent = obj.actor if hasattr(obj, "actor") else obj
        try:
            return str(ent.get_name() or "")
        except Exception:
            return ""

    def _gripper_holding(self, obj, rest_z, lift: float) -> bool:
        if obj is None or rest_z is None:
            return False
        try:
            z = float(obj.get_pose().p[2])
        except Exception:
            return False
        if z < float(rest_z) + float(lift):
            return False
        closing = False
        try:
            robot = self.robot
            closing = (
                float(robot.left_gripper_val) < 0.55
                or float(robot.right_gripper_val) < 0.55
            )
        except Exception:
            try:
                closing = bool(
                    self.is_left_gripper_close() or self.is_right_gripper_close()
                )
            except Exception:
                return False
        if not closing:
            return False
        name = self._object_name(obj)
        if not name:
            return False
        try:
            contacts = self.get_gripper_actor_contact_position(name)
        except Exception:
            return False
        return len(contacts) > 0

    # ------------------------------------------------------- part 4: ball
    def _ensure_ball(self) -> None:
        if self._ball is not None:
            return
        r = float(self.BALL_RADIUS)
        ball = create_sphere(
            self,
            pose=sapien.Pose(
                [float(self.BALL_X), float(self.BALL_Y0), float(self.TABLE_Z) + r]
            ),
            radius=r,
            color=list(self.BALL_COLOR),
            is_static=False,
            name="tutorial_ball",
        )
        try:
            ball.set_name("tutorial_ball")
        except Exception:
            pass
        self._ball = ball
        self._ball_start = self._pose_copy(ball)
        self._ball_rest_z = float(self._ball_start.p[2])
        rigid = self._get_rigid(ball)
        self._ball_rigid = rigid
        if rigid is not None:
            try:
                rigid.mass = float(self.BALL_MASS)
            except Exception:
                pass
            self._configure_ball_collision(rigid)
            # Pack-fruits style: kinematic slide until the user grasps it.
            try:
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
            except Exception:
                pass

    def _configure_ball_collision(self, rigid) -> None:
        """High-friction, no-bounce material so closed jaws can hold the ball."""
        if rigid is None:
            return
        try:
            if self._ball_grip_mat is None:
                self._ball_grip_mat = self.scene.create_physical_material(
                    2.8, 2.4, 0.0
                )
            mat = self._ball_grip_mat
            for shape in rigid.get_collision_shapes():
                shape.set_physical_material(mat)
        except Exception:
            pass
        try:
            rigid.set_linear_damping(0.8)
            rigid.set_angular_damping(4.0)
        except Exception:
            pass

    def _ball_kinematic(self) -> bool:
        rigid = self._ball_rigid
        if rigid is None:
            return False
        try:
            return bool(rigid.kinematic)
        except Exception:
            try:
                return bool(rigid.is_kinematic)
            except Exception:
                return False

    def _calm_ball(self, damping=(1.0, 5.0)) -> None:
        rigid = self._ball_rigid
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity([0.0, 0.0, 0.0])
            rigid.set_angular_velocity([0.0, 0.0, 0.0])
            rigid.set_linear_damping(float(damping[0]))
            rigid.set_angular_damping(float(damping[1]))
        except Exception:
            pass

    def _release_ball_physics(self) -> None:
        """Hand the ball to PhysX for a contact grasp (pack_fruits pattern).

        Keep gravity off until the jaws finish closing and lift — otherwise the
        ball drops through the fingers the frame it goes dynamic.
        """
        ball = self._ball
        rigid = self._ball_rigid
        if rigid is None or ball is None:
            return
        try:
            pose = ball.get_pose()
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(True)
            ball.set_pose(pose)
            self._configure_ball_collision(rigid)
            self._calm_ball(damping=(1.2, 6.0))
            self._ball_grasp_armed = True
        except Exception:
            pass

    def _enable_ball_gravity(self) -> None:
        rigid = self._ball_rigid
        if rigid is None:
            return
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            self._calm_ball(damping=(0.6, 3.0))
        except Exception:
            pass

    def _drive_ball_step(self) -> None:
        """Advance the kinematic ball toward the robot along −Y."""
        ball = self._ball
        rigid = self._ball_rigid
        if ball is None or rigid is None or not self._ball_kinematic():
            return
        try:
            pose = ball.get_pose()
            p = np.asarray(pose.p, dtype=float)
        except Exception:
            return
        try:
            dt = float(self.scene.get_timestep())
        except Exception:
            dt = 1.0 / 250.0
        y = float(p[1]) - float(self.BALL_SPEED) * dt
        y_end = float(self.BALL_Y_END)
        lane_x = float(getattr(self, "_ball_lane_x", self.BALL_X))
        if y <= y_end:
            # Slide off the near edge still ungripped → fall and respawn.
            try:
                fall = sapien.Pose(
                    [lane_x, y_end - 0.04, float(p[2])],
                    list(pose.q),
                )
                ball.set_pose(fall)
                rigid.set_kinematic_target(fall)
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_velocity([0.0, -float(self.BALL_SPEED), -0.05])
            except Exception:
                pass
            return
        new_pose = sapien.Pose(
            [lane_x, y, float(self._ball_rest_z)], list(pose.q)
        )
        try:
            ball.set_pose(new_pose)
            rigid.set_kinematic_target(new_pose)
        except Exception:
            try:
                ball.set_pose(new_pose)
            except Exception:
                pass

    def _launch_ball(self) -> None:
        ball = self._ball
        start = self._ball_start
        rigid = self._ball_rigid
        if ball is None or start is None:
            return
        dx = self._dx_for(self.BALL_X)
        start = self._offset_x(start, dx)
        self._ball_lane_x = float(self.BALL_X) + dx
        self._ball_grasp_armed = False
        try:
            ball.set_pose(start)
        except Exception:
            return
        if rigid is None:
            return
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(start)
            rigid.set_linear_velocity([0.0, 0.0, 0.0])
            rigid.set_angular_velocity([0.0, 0.0, 0.0])
        except Exception:
            pass

    def _respawn_ball(self) -> None:
        self._ball_ever_lifted = False
        self._ball_grasp_armed = False
        self._grasp_hold_steps = 0
        self._stage_settle = int(self.BALL_RESPAWN_SETTLE)
        self._launch_ball()

    def _ball_off_table(self) -> bool:
        ball = self._ball
        if ball is None:
            return False
        p = np.asarray(ball.get_pose().p, dtype=float)
        table_z = float(self._ball_rest_z or (self.TABLE_Z + self.BALL_RADIUS))
        if float(p[2]) < table_z - float(self.BALL_OFF_Z):
            return True
        if abs(float(p[0])) > float(self.BALL_TABLE_X_LIM):
            return True
        if float(p[1]) < float(self.BALL_TABLE_Y_MIN) or float(p[1]) > float(
            self.BALL_TABLE_Y_MAX
        ):
            return True
        return False

    def _ball_gripper_contact(self) -> bool:
        """Closed jaws physically touching the ball (grasp start, before lift)."""
        ball = self._ball
        if ball is None:
            return False
        closing = False
        try:
            robot = self.robot
            closing = (
                float(robot.left_gripper_val) < 0.55
                or float(robot.right_gripper_val) < 0.55
            )
        except Exception:
            try:
                closing = bool(
                    self.is_left_gripper_close() or self.is_right_gripper_close()
                )
            except Exception:
                return False
        if not closing:
            return False
        name = self._object_name(ball)
        if not name:
            return False
        try:
            return len(self.get_gripper_actor_contact_position(name)) > 0
        except Exception:
            return False

    def _tick_ball(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        # Pack-fruits: free the stream object once jaws close on it (gravity off).
        if self._ball_kinematic() and self._ball_gripper_contact():
            self._release_ball_physics()
        holding = self._gripper_holding(
            self._ball, self._ball_rest_z, self.BALL_LIFT
        )
        if holding:
            if self._ball_grasp_armed:
                self._enable_ball_gravity()
                self._ball_grasp_armed = False
            self._ball_ever_lifted = True
            self._grasp_hold_steps += 1
            if self._grasp_hold_steps >= int(self.GRASP_CONFIRM_STEPS):
                self._ball_complete = True
            return
        # Closed on the ball but not lifted yet — keep gravity off so it stays
        # pinched between the pads.
        if self._ball_grasp_armed and self._ball_gripper_contact():
            return
        if self._ball_grasp_armed and not self._ball_gripper_contact():
            # Missed grasp: restore kinematic slide from current XY.
            self._ball_grasp_armed = False
            rigid = self._ball_rigid
            ball = self._ball
            if rigid is not None and ball is not None:
                try:
                    pose = ball.get_pose()
                    rigid.set_disable_gravity(True)
                    rigid.set_kinematic(True)
                    rigid.set_kinematic_target(pose)
                    self._calm_ball()
                except Exception:
                    pass
        self._grasp_hold_steps = 0
        # Touch / nudge must not despawn — only a fall off the table does.
        if self._ball_off_table():
            self._respawn_ball()
            return
        if self._ball_kinematic():
            self._drive_ball_step()
            if self._ball_off_table():
                self._respawn_ball()

    # ------------------------------------------------------- part 4: stove
    def _ensure_stove(self) -> None:
        if self._stove_loaded:
            return
        self.range_scale_mult = 1.0
        self.range_position_override = list(self.STOVE_XY)
        self.stove_side = "left"
        self.knob_contact_radius = float(self.KNOB_CONTACT_RADIUS_DEFAULT)
        table_h = float(self.TABLE_Z) + float(getattr(self, "table_z_bias", 0.0))
        self._load_cooking_range(table_h, [0.0, 0.0])
        name = str(self.STOVE_BURNER)
        if name not in getattr(self, "burner_positions", {}):
            name = "left_front"
        self.burner_name = name
        bx, by = self.burner_positions[name]
        self.burner_xy = (float(bx), float(by))
        self._burner_home_pose = sapien.Pose(
            p=[float(bx), float(by), float(self.range_top_z) + 0.002]
        )
        self._build_stove_fire_ring(
            float(bx),
            float(by),
            float(self.range_top_z) + 0.004,
            0.058,
            half_size=[0.010, 0.005, 0.004],
        )
        self._set_stove_fire(False, intensity=0.0)
        if getattr(self, "range_body", None) is not None:
            self._stove_range_home = self._pose_copy(self.range_body.actor)
        art = getattr(self, "stove_knob_articulation", None)
        if art is not None:
            try:
                self._stove_knob_home = sapien.Pose(
                    list(art.get_pose().p), list(art.get_pose().q)
                )
            except Exception:
                try:
                    pose = art.get_root_pose()
                    self._stove_knob_home = sapien.Pose(list(pose.p), list(pose.q))
                except Exception:
                    self._stove_knob_home = None
        self._stove_cover_homes = dict(getattr(self, "_burner_cover_home_poses", {}) or {})
        self._burner_xy_home = tuple(self.burner_xy)
        self._ring_home_poses_base = list(getattr(self, "_ring_home_poses", []) or [])
        self._disc_home_poses_base = list(getattr(self, "_disc_home_poses", []) or [])
        self._cover_homes_base = dict(self._stove_cover_homes)
        self._burner_positions_base = dict(getattr(self, "burner_positions", {}) or {})
        self._set_knob_joint_angle(float(self.KNOB_OFF_ANGLE), hard=True)
        self._stove_loaded = True

    def _hide_stove(self) -> None:
        if not self._stove_loaded:
            return
        try:
            self._set_stove_fire(False, intensity=0.0)
        except Exception:
            pass
        hidden = sapien.Pose([0.0, 0.0, -1.6], [1, 0, 0, 0])
        try:
            if getattr(self, "range_body", None) is not None:
                self.range_body.actor.set_pose(hidden)
        except Exception:
            pass
        art = getattr(self, "stove_knob_articulation", None)
        if art is not None:
            try:
                art.set_pose(hidden)
            except Exception:
                try:
                    art.set_root_pose(hidden)
                except Exception:
                    pass
        for cover in getattr(self, "_burner_covers", []) or []:
            try:
                cover.set_pose(hidden)
            except Exception:
                pass
        burner = getattr(self, "active_burner", None)
        if burner is not None:
            try:
                burner.set_pose(hidden)
            except Exception:
                pass

    def _show_stove(self) -> None:
        if not self._stove_loaded:
            return
        dx = self._dx_for(self.STOVE_XY[0])
        if self._stove_range_home is not None and getattr(self, "range_body", None):
            try:
                self.range_body.actor.set_pose(
                    self._offset_x(self._stove_range_home, dx)
                )
            except Exception:
                pass
        art = getattr(self, "stove_knob_articulation", None)
        if art is not None and self._stove_knob_home is not None:
            posed = self._offset_x(self._stove_knob_home, dx)
            try:
                art.set_root_pose(posed)
            except Exception:
                try:
                    art.set_pose(posed)
                except Exception:
                    pass
        home_xy = getattr(self, "_burner_xy_home", None) or getattr(
            self, "burner_xy", (float(self.STOVE_XY[0]), float(self.STOVE_XY[1]))
        )
        self.burner_xy = (float(home_xy[0]) + dx, float(home_xy[1]))
        try:
            z = float(self.range_top_z) + 0.002
        except Exception:
            z = float(self.TABLE_Z) + 0.002
        self._burner_home_pose = sapien.Pose(
            p=[float(self.burner_xy[0]), float(self.burner_xy[1]), z]
        )
        ring_base = getattr(self, "_ring_home_poses_base", None)
        if ring_base:
            self._ring_home_poses = [self._offset_x(p, dx) for p in ring_base]
        disc_base = getattr(self, "_disc_home_poses_base", None)
        if disc_base:
            self._disc_home_poses = [self._offset_x(p, dx) for p in disc_base]
        cover_base = getattr(self, "_cover_homes_base", None)
        if cover_base:
            covers = {k: self._offset_x(p, dx) for k, p in cover_base.items()}
            self._burner_cover_home_poses = covers
            self._stove_cover_homes = covers
        pos_base = getattr(self, "_burner_positions_base", None)
        if pos_base:
            self.burner_positions = {
                k: (float(v[0]) + dx, float(v[1])) for k, v in pos_base.items()
            }
        for cover in getattr(self, "_burner_covers", []) or []:
            # Covers stay buried until fire turns on.
            try:
                cover.set_pose(self._fire_hidden_pose())
            except Exception:
                pass
        try:
            self._set_knob_joint_angle(float(self.KNOB_OFF_ANGLE), hard=True)
        except Exception:
            pass
        self.knob_angle = float(self.KNOB_OFF_ANGLE)
        self.fire_intensity = 0.0
        self.stove_on = False
        # Avoid fire flush nesting a render; just mark off and bump bodies.
        try:
            self._stove_fire_visual = None
            self._set_stove_fire(False, intensity=0.0)
        except Exception:
            pass

    def tutorial_ball_released(self) -> bool:
        """True when the ball is no longer pinched (safe to spawn the next prop)."""
        if self._ball is None:
            return True
        return not self._ball_gripper_contact()

    def tutorial_force_open_grippers(self) -> None:
        """Open selected grippers so a held ball can be parked safely."""
        robot = getattr(self, "robot", None)
        if robot is None or not hasattr(robot, "set_gripper"):
            return
        sides = tuple(getattr(self, "_interactive_selected_arms", ()) or ())
        if not sides:
            sides = ("left", "right")
        for side in sides:
            try:
                robot.set_gripper(1.0, str(side), gripper_eps=0.0)
            except Exception:
                pass

    def _set_knob_angle(self, angle: float, *, drive_fire: bool = True) -> None:
        """cook_food mapping: 0 = off, −π/2 = full fire (tick left)."""
        angle = float(np.clip(angle, -float(self.KNOB_MAX_ANGLE), 0.0))
        self.knob_angle = angle
        if not drive_fire:
            return
        frac = float(-angle / max(float(self.KNOB_MAX_ANGLE), 1e-6))
        self.fire_intensity = float(np.clip(frac, 0.0, 1.0))
        self.stove_on = self.fire_intensity > 0.02
        if self.fire_intensity >= float(self.STOVE_ON_FRAC):
            self._stove_turned_on = True
        if self._stove_turned_on and self.fire_intensity <= float(self.STOVE_OFF_FRAC):
            self._stove_turned_off = True
        self._set_knob_joint_angle(
            angle, hard=not bool(getattr(self, "_knob_grasp_active", False))
        )
        self._set_stove_fire(self.stove_on, intensity=self.fire_intensity)

    def _tick_stove(self) -> None:
        if getattr(self, "stove_knob_articulation", None) is None:
            return
        self._update_knob_from_physics()
        self._update_stove_knob_control()
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        if self._stove_turned_on and self._stove_turned_off:
            self._stove_complete = True

    # ------------------------------------------------------- part 4: mallet
    def _ensure_mallet(self) -> None:
        if self._mallet is not None:
            return
        x, y = float(self.PROP_X), float(self.PROP_Y)
        rest_h = float(self.MALLET_REST_HEIGHT)
        table_z = float(self.TABLE_Z) + float(getattr(self, "table_z_bias", 0.0))
        rails = []
        rest_poses = []
        for post_idx, y_offset in enumerate(self.MALLET_REST_POST_Y):
            # Same two-post cradle as whack_moles (table_z includes bias).
            rail = create_box(
                self,
                sapien.Pose(
                    [x, y + float(y_offset), table_z + 0.5 * rest_h]
                ),
                half_size=[0.016, 0.010, 0.5 * rest_h],
                color=[0.32, 0.30, 0.28],
                is_static=True,
                name=f"tutorial_mallet_rest_{post_idx}",
            )
            rails.append(rail)
            rest_poses.append(self._pose_copy(next(self._iter_entities(rail))))
        z = table_z + rest_h + float(self.MALLET_HANDLE_RADIUS)
        mallet = self._build_mallet(
            sapien.Pose([x, y, z], list(self.MALLET_LEVEL_Q)),
            name="tutorial_mallet",
        )
        mallet.set_name("tutorial_mallet")
        mallet.set_mass(float(self.MALLET_MASS))
        self._mallet = mallet
        self._mallet_rests = rails
        self._mallet_table_pose = self._pose_copy(mallet.actor)
        self._mallet_rest_poses = rest_poses
        self._mallet_rest_z = float(self._mallet_table_pose.p[2])
        self._mallet_grasp_armed = False
        # whack_moles: staged mallets stay kinematic on the cradle until pickup.
        self._seat_mallet(self._mallet_table_pose)

    def _build_mallet(self, pose, name: str):
        """T mallet: handle along Y, cylindrical head across X (whack_moles)."""
        builder = self.scene.create_actor_builder()
        mat = self.scene.default_physical_material
        wood = sapien.render.RenderMaterial(
            base_color=[*self.MALLET_WOOD_COLOR, 1.0]
        )
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, self.MALLET_HEAD_Y, 0.0]),
            radius=self.MALLET_HEAD_RADIUS,
            half_length=self.MALLET_HEAD_HALF_X,
            material=mat,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0.0, self.MALLET_HEAD_Y, 0.0]),
            radius=self.MALLET_HEAD_RADIUS,
            half_length=self.MALLET_HEAD_HALF_X,
            material=wood,
        )
        handle_pose = sapien.Pose(
            [0.0, self.MALLET_HANDLE_CENTER_Y, 0.0],
            [0.70710678, 0.0, 0.0, 0.70710678],
        )
        builder.add_cylinder_collision(
            pose=handle_pose,
            radius=self.MALLET_HANDLE_RADIUS,
            half_length=self.MALLET_HANDLE_HALF_Y,
            material=mat,
        )
        builder.add_cylinder_visual(
            pose=handle_pose,
            radius=self.MALLET_HANDLE_RADIUS,
            half_length=self.MALLET_HANDLE_HALF_Y,
            material=wood,
        )
        builder.set_initial_pose(pose)
        entity = builder.build(name=name)
        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [
                self.MALLET_HEAD_HALF_X,
                self.MALLET_HANDLE_HALF_Y + abs(self.MALLET_HEAD_Y),
                self.MALLET_HEAD_RADIUS,
            ],
            "scale": [
                self.MALLET_HANDLE_RADIUS,
                self.MALLET_HANDLE_HALF_Y,
                self.MALLET_HANDLE_RADIUS,
            ],
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
        }
        return Actor(entity, data, mass=float(self.MALLET_MASS))

    def _seat_mallet(self, pose: sapien.Pose | None = None) -> None:
        """Park level on the cradle — kinematic, no gravity (whack_moles staging)."""
        mallet = self._mallet
        if mallet is None:
            return
        if pose is None:
            pose = self._mallet_reset_pose or self._mallet_table_pose
        if pose is not None:
            self._apply_pose(mallet, pose)
        rigid = self._get_rigid(mallet)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                rigid.set_linear_velocity([0.0, 0.0, 0.0])
                rigid.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass
        self._mallet_grasp_armed = False

    def _release_mallet_for_grasp(self) -> None:
        """Free the staged mallet so jaw friction can lift it."""
        rigid = self._get_rigid(self._mallet)
        if rigid is None:
            return
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_damping(20.0)
            rigid.set_angular_damping(20.0)
        except Exception:
            pass
        self._mallet_grasp_armed = True

    def _show_mallet(self) -> None:
        dx = self._dx_for(self.PROP_X)
        pose = self._offset_x(self._mallet_table_pose, dx)
        self._mallet_reset_pose = pose
        if self._mallet_rests and self._mallet_rest_poses:
            for ent, rest in zip(
                self._iter_entities(self._mallet_rests), self._mallet_rest_poses
            ):
                try:
                    ent.set_pose(self._offset_x(rest, dx))
                except Exception:
                    pass
        self._seat_mallet(pose)
        if self._mallet is not None:
            self._mallet_rest_z = float(self._mallet.get_pose().p[2])

    def _tick_mallet(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            # Keep it seated while the stage settles after arm reset.
            if self._mallet_reset_pose is not None:
                self._seat_mallet(self._mallet_reset_pose)
            return
        mallet = self._mallet
        if mallet is None:
            return
        closing = False
        try:
            robot = self.robot
            closing = (
                float(robot.left_gripper_val) < 0.55
                or float(robot.right_gripper_val) < 0.55
            )
        except Exception:
            try:
                closing = bool(
                    self.is_left_gripper_close() or self.is_right_gripper_close()
                )
            except Exception:
                closing = False
        touching = self._gripper_touching(mallet)
        rigid = self._get_rigid(mallet)
        # Stay kinematic on the cradle until the jaws close on the handle.
        if (
            rigid is not None
            and bool(getattr(rigid, "kinematic", False))
            and touching
            and closing
        ):
            self._release_mallet_for_grasp()
        holding = self._gripper_holding(
            mallet, self._mallet_rest_z, self.MALLET_LIFT
        )
        if not holding and not touching:
            p = self._actor_xyz(mallet)
            tipped = False
            try:
                z = float(mallet.get_pose().p[2])
                rest_z = float(self._mallet_rest_z or z)
                tipped = z < rest_z - 0.015
            except Exception:
                tipped = False
            if tipped or self._off_workspace(p):
                self._seat_mallet(self._mallet_reset_pose)
                return
        if holding:
            self._grasp_hold_steps += 1
            if self._grasp_hold_steps >= int(self.GRASP_CONFIRM_STEPS):
                self._mallet_complete = True
        else:
            self._grasp_hold_steps = 0
    # ------------------------------------------------------- part 4: force key
    def _ensure_force_key(self) -> None:
        if self._force_key_actor is not None:
            return
        hz = float(self.KEY_HALF[2])
        x, y = float(self.PROP_X), float(self.PROP_Y)
        z0 = float(self.TABLE_Z)
        bezel = add_key_base_border(
            self,
            x,
            y,
            z0,
            self.KEY_HALF,
            name_prefix="tutorial_force_key_base",
        )
        home = sapien.Pose([x, y, z0 + hz], [1, 0, 0, 0])
        key = create_box(
            self,
            pose=home,
            half_size=list(self.KEY_HALF),
            color=list(self.FORCE_KEY_COLOR),
            name="tutorial_force_key",
            is_static=True,
        )
        self._force_key_actor = key
        self._force_key_bezel = bezel
        self._force_key_table_pose = self._pose_copy(key.actor)
        self._force_key_bezel_poses = [
            self._pose_copy(ent) for ent in self._iter_entities(bezel)
        ]
        self._key_shapes = self._render_shapes(key)
        self._hide_obj(key, 16)
        self._hide_obj(bezel, 17)

    def _show_force_key(self) -> None:
        dx = self._dx_for(self.PROP_X)
        self._apply_pose(
            self._force_key_actor, self._offset_x(self._force_key_table_pose, dx)
        )
        if self._force_key_bezel is not None and self._force_key_bezel_poses:
            for ent, pose in zip(
                self._iter_entities(self._force_key_bezel),
                self._force_key_bezel_poses,
            ):
                try:
                    ent.set_pose(self._offset_x(pose, dx))
                except Exception:
                    pass
        key = self._force_key_actor
        if key is None:
            self._reactive_buttons = None
            return
        live = self._pose_copy(next(self._iter_entities(key)))
        hz = float(self.KEY_HALF[2])
        self._key_shapes = self._render_shapes(key)
        self._key_color_down = None
        self._set_force_key_color(0.0)
        self._reactive_buttons = _ForceKeyBank(
            self,
            actors=[key],
            home_poses=[live],
            max_depth=hz,
            ids=["force"],
            press_arms=(("left", "right"),),
            xy_tol=float(self.KEY_XY_TOL),
            force_full=float(self.FORCE_THRESHOLDS[-1]),
        )
        self._reactive_buttons.set_tops_z([float(live.p[2]) + hz])

    def _force_level(self, force_n: float) -> int:
        level = 0
        for i, thr in enumerate(self.FORCE_THRESHOLDS):
            if float(force_n) >= float(thr):
                level = i + 1
        return int(level)

    def _set_force_key_color(self, force_n: float) -> None:
        level = self._force_level(force_n)
        rgb = self.FORCE_LEVEL_COLORS[min(level, len(self.FORCE_LEVEL_COLORS) - 1)]
        if self._key_color_down is not None and self._key_color_down == level:
            return
        self._key_color_down = level
        color = list(rgb) + [1.0]
        for shape in self._key_shapes:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _force_yellow_band(self, target: int) -> tuple[float, float]:
        """Inclusive force band for the current yellow target (N)."""
        thr = self.FORCE_THRESHOLDS
        n = len(thr)
        t = max(1, min(int(target), n))
        lo = float(thr[t - 1])
        hi = float(thr[t]) if t < n else float(thr[-1]) * 1.15
        return lo, hi

    def _force_in_yellow(self, force_n: float, target: int) -> bool:
        lo, hi = self._force_yellow_band(target)
        f = float(force_n)
        if int(target) >= len(self.FORCE_THRESHOLDS):
            return f >= lo
        return lo <= f <= hi

    def _advance_force_step(self, force_n: float) -> None:
        """Yellow reached — clear this step and move the band (or finish)."""
        target = int(self._force_target_level)
        n_levels = len(self.FORCE_THRESHOLDS)
        self._force_cleared = int(target)
        self._force_step_armed = False  # lift off before the next yellow counts
        if target >= n_levels:
            self._force_complete = True
            self._force_feedback = f"full force ({force_n:.1f} N) — done"
            return
        self._force_target_level = target + 1
        self._force_feedback = (
            f"yellow hit ({force_n:.1f} N) — release, then press the next band"
        )

    def _tick_force_key(self) -> None:
        bank = self._reactive_buttons
        if bank is None or not isinstance(bank, _ForceKeyBank):
            return
        if self._stage_settle > 0:
            self._stage_settle -= 1
            bank.update()
            return
        bank.update()
        force = float(bank.live_force("force"))
        self._key_force_n = force
        self._set_force_key_color(force)
        engaged = bool(bank.is_engaged("force"))
        if engaged:
            self._press_active = True
            self._press_peak_force = max(float(self._press_peak_force), force)
            # Success as soon as the live bar enters the yellow region.
            if (
                not self._force_complete
                and bool(getattr(self, "_force_step_armed", True))
                and self._force_in_yellow(force, int(self._force_target_level))
            ):
                self._advance_force_step(force)
            return
        # Off the key: re-arm so the next press can clear the new yellow.
        if self._press_active or not getattr(self, "_force_step_armed", True):
            self._press_active = False
            self._press_peak_force = 0.0
            if not self._force_complete:
                self._force_step_armed = True
            return

    def interactive_ee_z_floor(self, side, pose):
        if getattr(self, "_tutorial_stage", None) != "force_key":
            return None
        bank = self._reactive_buttons
        if not isinstance(bank, _ForceKeyBank):
            return None
        try:
            return bank.min_ee_z_over_key(np.asarray(pose[:2], dtype=float))
        except Exception:
            return None

    def interactive_teleop_z_speed_scale(self, side, pose, z_delta: float):
        if getattr(self, "_tutorial_stage", None) != "force_key":
            return None
        if float(z_delta) >= 0.0:
            return None
        bank = self._reactive_buttons
        if not isinstance(bank, _ForceKeyBank) or not bank.home_poses:
            return None
        xy = np.asarray(pose[:2], dtype=float)
        home = np.asarray(bank.home_poses[0].p[:2], dtype=float)
        if float(np.linalg.norm(xy - home)) > float(bank.xy_tol) * 1.35:
            return None
        return float(self.DISPENSE_Z_DOWN_SCALE)


# Reuse KitchenS cooktop / knob helpers without swapping the empty table scene.
_KS_ATTRS = (
    "COOKTOP_ASSET",
    "RANGE_BURNER_OFFSETS",
    "KNOB_LOCAL_XY",
    "KNOB_RADIUS",
    "KNOB_HEIGHT",
    "KNOB_OFF_ANGLE",
    "KNOB_ON_ANGLE",
    "KNOB_CONTACT_RADIUS_DEFAULT",
    "KNOB_GRASP_GRIPPER_MAX",
    "EE_TO_TCP_DEFAULT",
    "FIRE_BLUE",
    "FIRE_DISC_BLUE",
    "TOP_KNOB_APPROACH_PATH",
    "KNOB_APPROACH_PATH",
    "KNOB_GRASP_STANDOFF",
)
_KS_METHODS = (
    "_load_cooking_range",
    "_fire_hidden_pose",
    "_clear_stove_fire_ring",
    "_build_stove_fire_ring",
    # Do NOT mix in ``_flush_stove_fire_viewer`` — its viewer.render() nests
    # inside the interactive on_step and SIGSEGVs when the stove appears.
    "_set_stove_fire",
    "_get_knob_joint_angle",
    "_set_knob_articulation_qpos",
    "_hold_knob_joint",
    "_set_knob_joint_angle",
    "_ee_knob_twist",
    "_boost_gripper_knob_friction",
    "_knob_candidate_arms",
    "_begin_knob_turn",
    "_end_knob_turn",
    "_coupled_knob_angle",
    "_update_knob_from_physics",
    "_knob_pinch_near",
    "_knob_gripper_closed",
    "_knob_has_gripper_contact",
    "_knob_is_grasped",
    "_commit_stove_from_knob_angle",
    "_update_stove_knob_control",
)
# Keep @staticmethod — a bare copy becomes a bound method and breaks
# ``self._dim_cooktop_burner_materials(entity)`` (extra self → TypeError crash
# the moment the cooktop loads on ball→stove).
_KS_STATICMETHODS = (
    "_dim_cooktop_burner_materials",
    "_bump_render_bodies",
)
for _name in _KS_ATTRS:
    setattr(tutorial_empty, _name, getattr(KitchenS_base_task, _name))
for _name in _KS_METHODS:
    setattr(tutorial_empty, _name, getattr(KitchenS_base_task, _name))
for _name in _KS_STATICMETHODS:
    setattr(
        tutorial_empty,
        _name,
        staticmethod(getattr(KitchenS_base_task, _name)),
    )


def _flush_stove_fire_viewer(self) -> None:
    """Dirty the scene only — never nest ``viewer.render()`` during on_step."""
    try:
        self.scene.update_render()
    except Exception:
        pass


tutorial_empty._flush_stove_fire_viewer = _flush_stove_fire_viewer
