"""Shared interactive runner for the household task sandboxes.

The household environments deliberately keep their normal physics and success
checks.  This module only adds the same viewer/arm teleoperation used by
``interactive/_interactive_common.py``: arrows move the selected end-effector in XY, Q/E move it in Z,
F/G tip it left/right about world Y, and 1/2/3 select the left/right/both arms.
Space opens/closes the selected gripper(s) only (shared ``ViewerViewToggle``)
in robot mode. Keyboard+mouse hides the arms and stays on head_camera (no V
cycle). Space never auto-grasps, teleports, or runs planner pick/place shortcuts.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import sapien.physx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = ROOT / "script" / "bench_script"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from sapien import internal_renderer as R  # noqa: E402
from sapien.utils.viewer.plugin import Plugin  # noqa: E402

from interactive._interactive_common import (  # noqa: E402
    ESCAPE_QUIT_DETAIL,
    RealtimePhysicsPacer,
    action_failed,
    add_record_data_arg,
    click_hits_actor,
    configure_task,
    edge_pressed,
    flash_gripper_failure,
    format_ps_suffix,
    gripper_failure_feedback,
    is_robot_control,
    make_viewer_view_toggle,
    normalize_control_mode,
    prepare_interactive_control,
    print_episode_condition,
    print_failure,
    print_instructions,
    print_mode_controls,
    print_success,
    report_task_result,
    require_selected_arms,
    resolve_action_arm,
    table_xy_from_click,
    _env_partial_score,
)


PROFILES = {
    "trap_bug": ("envs.trap_bug", "trap_bug", "demo_dynamic", "trap"),
    "boil_milk": ("envs.boil_milk", "boil_milk", "demo_dynamic", None),
    "fill_coffee_jar": ("envs.fill_coffee_jar", "fill_coffee_jar", "demo_dynamic", None),
    "pour_beer": ("envs.pour_beer", "pour_beer", "demo_dynamic", None),
    "cook_food": ("envs.cook_food", "cook_food", "demo_dynamic", "food"),
    "cook_food_timer": ("envs.cook_food_timer", "cook_food_timer", "demo_dynamic", "food"),
    "measure_ingredient": ("envs.measure_ingredient", "measure_ingredient", "demo_dynamic", "jar"),
    "make_soup": ("envs.make_soup", "make_soup", "demo_dynamic", "board"),
    "catch_cup": ("envs.catch_cup", "catch_cup", "demo_dynamic", "pillow"),
    "catch_mouse_object_drop": ("envs.catch_mouse_object_drop", "catch_mouse_object_drop", "demo_dynamic", "basket"),
    "stop_ball": ("envs.stop_ball", "stop_ball", "demo_dynamic", None),
    "clean_table": ("envs.clean_table", "clean_table", "demo_dynamic", "sponge"),
}


class FillTargetHud(Plugin):
    """Top-right overlay showing the episode's required jar fill level."""

    _WINDOW_W = 220
    _WINDOW_H = 56

    def __init__(self, env, task: str):
        self.env = env
        self.task = task
        self.ui_window = None

    def _required_pct(self) -> str:
        tgt = 100.0 * float(getattr(self.env, "target_fill", 0.0))
        return f"Required fill: {tgt:.0f}%"

    def get_ui_windows(self):
        if self.ui_window is None:
            self.ui_window = (
                R.UIWindow()
                .Label("Fill target")
                .append(R.UIDisplayText().Bind(self._required_pct))
            )
        ww = 1280
        try:
            ww = int(self.viewer.window.size[0])
        except Exception:
            pass
        pw, ph = self._WINDOW_W, self._WINDOW_H
        self.ui_window.Size(pw, ph)
        self.ui_window.Pos(max(16, ww - pw - 24), 16)
        return [self.ui_window]


def _rigid(actor):
    obj = getattr(actor, "actor", actor)
    for comp in getattr(obj, "get_components", lambda: ())():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_pose(actor, xyz, quat=None, kinematic=True):
    """Set an Actor pose while preserving its dynamic/kinematic state."""
    if actor is None:
        return
    pose = actor.get_pose()
    q = pose.q if quat is None else quat
    target = sapien.Pose([float(v) for v in xyz], list(q))
    obj = getattr(actor, "actor", actor)
    obj.set_pose(target)
    body = _rigid(actor)
    if body is None:
        return
    try:
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
        body.set_disable_gravity(bool(kinematic))
        body.set_kinematic(bool(kinematic))
        if kinematic and hasattr(body, "set_kinematic_target"):
            body.set_kinematic_target(target)
    except Exception:
        pass


def _arm_tag(env, *, exactly_one: bool = True):
    """Return the highlighted arm, or ``None`` when no action should run."""
    from envs.utils.action import ArmTag
    return resolve_action_arm(env, ArmTag, exactly_one=exactly_one)


class HouseholdController:
    """Small adapter around each task's existing actors and expert hooks."""

    FILL_PRESS_STEP = 0.006
    FILL_MAX_PRESS_DEPTH = 0.12
    FILL_CONTACT_DZ = 0.006
    FILL_STALL_DZ = 0.012
    FILL_XY_TOL = 0.040
    FILL_RETURN_STEPS = 25

    def __init__(self, env, task, robot=False):
        self.env, self.task, self.robot = env, task, robot
        actor_attr = PROFILES[task][3]
        self.actor = getattr(env, actor_attr, None) if actor_attr else None
        self.holding = False
        self._fill_press_state = None
        self.trap_released = False
        self.scenario_started = False
        if task in (
            "fill_coffee_jar", "pour_beer", "measure_ingredient", "catch_cup",
        ) and robot:
            self._close_grippers_at_start()
        elif task == "stop_ball" and robot:
            self._open_stop_ball_grippers_at_start()
        elif task in ("clean_table", "make_soup") and robot:
            # Start open so the WSG can close around the handle cube.
            self._open_clean_table_grippers_at_start()
        self._key_prev = {}
        self._pour_held = False
        self._board_tilt = 0.0
        self._board_placed = False
        self._board_flat_pose = None
        self._last_tilt_t = None
        self._orig_button_press_signal = None
        self._click_via_handler = False
        self._prev_mouse = False
        self._catcher_placed = False
        self._stop_ball_bridge = None
        self._stop_ball_bridge_placed = False
        # catch_cup / make_soup robot: keep the prop dynamic so gripper contact is real.
        # Keyboard+mouse drives props via click / keys (arms are hidden).
        if robot and task == "catch_cup" and self.actor is not None:
            try:
                self.env._enable_pillow_physics()
            except Exception:
                pass
        elif not robot:
            self._prepare_keyboard_props()

    def _close_grippers_at_start(self):
        """Start robot mode with both grippers closed at once."""
        try:
            self.env.plan_success = True
            self.env.together_close_gripper(save_freq=None)
        except Exception as exc:
            print(f"[{self.task}] could not pre-close grippers: {exc}")
        self.env.plan_success = True

    def _open_stop_ball_grippers_at_start(self):
        """stop_ball: start open so the fingers can catch the rolling ball."""
        try:
            self.env.plan_success = True
            opener = getattr(self.env, "together_open_gripper", None)
            if callable(opener):
                opener(save_freq=None, left_pos=1.0, right_pos=1.0)
            else:
                from envs.utils.action import ArmTag

                for side in ("left", "right"):
                    self.env.plan_success = True
                    self.env.move(self.env.open_gripper(ArmTag(side), pos=1.0))
        except Exception as exc:
            print(f"[stop_ball] could not pre-open grippers: {exc}")
        self.env.plan_success = True

    def _open_clean_table_grippers_at_start(self):
        """Start robot mode with both grippers fully open (handle pinch tasks)."""
        if self.task not in ("clean_table", "make_soup"):
            return
        try:
            self.env.plan_success = True
            opener = getattr(self.env, "together_open_gripper", None)
            if callable(opener):
                opener(save_freq=None, left_pos=1.0, right_pos=1.0)
            else:
                from envs.utils.action import ArmTag

                for side in ("left", "right"):
                    self.env.plan_success = True
                    self.env.move(self.env.open_gripper(ArmTag(side), pos=1.0))
        except Exception as exc:
            print(f"[{self.task}] could not pre-open grippers: {exc}")
        self.env.plan_success = True

    def _prepare_keyboard_props(self):
        """Seat keyboard-mode props without changing env files or robot play."""
        e, t = self.env, self.task
        if t == "catch_cup" and getattr(e, "pillow", None) is not None:
            e._push_active = False
            try:
                e._freeze_pillow()
            except Exception:
                _set_pose(e.pillow, e.pillow.get_pose().p, kinematic=True)
        elif t == "catch_mouse_object_drop" and getattr(e, "basket", None) is not None:
            p = np.asarray(e.basket.get_pose().p, dtype=float)
            hz = float(getattr(e, "basket_hz", 0.5 * float(getattr(e, "basket_height", 0.07))))
            p[2] = float(e.table_top) + hz
            _set_pose(e.basket, p, quat=list(e.basket.get_pose().q), kinematic=True)
        elif t in ("cook_food", "cook_food_timer") and getattr(e, "food", None) is not None:
            self._seat_food_in_pan()
            # Keyboard has no gripper on the knob — don't let buried-arm
            # contacts re-drive fire from the joint after a snap.
            e._knob_is_grasped = lambda: False
            e._knob_has_gripper_contact = lambda: False
            e._knob_grasp_active = False
            e._policy_controlling_knob = False
            e._ignore_knob = False
        elif t == "clean_table" and getattr(e, "sponge", None) is not None:
            self._hover_sponge()
        elif t == "make_soup" and getattr(e, "board", None) is not None:
            _set_pose(e.board, e.board.get_pose().p, kinematic=True)
        elif t == "stop_ball":
            self._create_stop_ball_bridge()
        elif self.actor is not None and t not in ("stop_ball",):
            _set_pose(self.actor, self.actor.get_pose().p, kinematic=True)
        if t == "pour_beer":
            self._orig_button_press_signal = getattr(e, "_button_press_signal", None)

            def _keyboard_button_signal():
                if getattr(self, "_pour_held", False):
                    on_n = float(getattr(e, "PRESS_FORCE_ON", 1.0))
                    return {
                        "side": "left",
                        "tcp": np.array([0.0, 0.0, 0.0], dtype=float),
                        "force": on_n * 2.0,
                    }
                orig = self._orig_button_press_signal
                return orig() if callable(orig) else None

            e._button_press_signal = _keyboard_button_signal

    def _create_stop_ball_bridge(self):
        """Create a movable U-shaped WSG surrogate for keyboard control."""
        e = self.env
        builder = e.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        # Match the open WSG silhouette: two fingers with a connecting palm.
        # The ball is slightly wider than this inner gap, as in robot mode.
        gap = float(getattr(e, "WSG_OPEN_FINGER_GAP", 0.110))
        finger_half = np.array([0.012, 0.022, 0.045], dtype=float)
        finger_x = 0.5 * gap + finger_half[0]
        palm_half = np.array([finger_x + finger_half[0], 0.022, 0.010], dtype=float)
        palm_z = 2.0 * finger_half[2] - palm_half[2]
        material = sapien.render.RenderMaterial(
            base_color=[0.16, 0.20, 0.24, 1.0],
        )
        try:
            material.set_roughness(0.42)
            material.set_metallic(0.55)
        except Exception:
            pass

        parts = (
            (sapien.Pose([-finger_x, 0.0, finger_half[2]]), finger_half),
            (sapien.Pose([finger_x, 0.0, finger_half[2]]), finger_half),
            (sapien.Pose([0.0, 0.0, palm_z]), palm_half),
        )
        for pose, half_size in parts:
            builder.add_box_collision(
                pose=pose,
                half_size=half_size.tolist(),
                material=e.scene.default_physical_material,
            )
            builder.add_box_visual(
                pose=pose,
                half_size=half_size.tolist(),
                material=material,
            )

        builder.set_initial_pose(sapien.Pose([0.0, 0.0, -10.0]))
        self._stop_ball_bridge = builder.build(name="keyboard_stop_ball_bridge")
        _set_pose(self._stop_ball_bridge, [0.0, 0.0, -10.0], kinematic=True)

    def _seat_food_in_pan(self):
        e = self.env
        tgt = e._pan_place_target()
        pose = sapien.Pose(
            [float(tgt[0]), float(tgt[1]), float(tgt[2])],
            list(getattr(e, "FOOD_QPOS", [1.0, 0.0, 0.0, 0.0])),
        )
        try:
            e.food.actor.set_pose(pose)
        except Exception:
            e.food.set_pose(pose)
        rigid = getattr(e, "_food_rigid", None)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        e._food_in_pan = True
        try:
            e._lock_food_to_pan()
        except Exception:
            pass
        print(f"[{self.task}] food starts in the pan")

    def _hover_sponge(self, xy=None):
        e = self.env
        if xy is None:
            p = np.asarray(e.sponge.get_pose().p, dtype=float)
            xy = (float(p[0]), float(p[1]))
        z = float(e.table_top) + float(e.SPONGE_HALF[2]) + 0.05
        e._sponge_welded = False
        e._freeze_sponge(sapien.Pose([float(xy[0]), float(xy[1]), z], [1.0, 0.0, 0.0, 0.0]))

    def _contact_sponge(self, x, y):
        e = self.env
        z = float(e._contact_z()) if callable(getattr(e, "_contact_z", None)) else (
            float(e.table_top) + float(e.SPONGE_HALF[2]) + 0.002
        )
        e._sponge_welded = False
        e._freeze_sponge(sapien.Pose([float(x), float(y), z], [1.0, 0.0, 0.0, 0.0]))
        print(f"[clean_table] sponge contacting table at ({x:.3f}, {y:.3f})")

    def _keyboard_action(self):
        """Operate task state directly without moving either robot arm."""
        e, t = self.env, self.task
        try:
            if t in ("boil_milk", "cook_food", "cook_food_timer"):
                want_on = not bool(getattr(e, "stove_on", False))
                self._snap_stove_knob(want_on)
                print(f"[{t}] stove {'ON' if want_on else 'OFF'}")
            elif t == "fill_coffee_jar":
                self._fill_coffee_press(1)
            elif t == "measure_ingredient":
                self._toggle_measure_nozzle()
            elif t == "pour_beer":
                pass
        except Exception as exc:
            print(f"[{t}] action unavailable: {exc}")

    def _toggle_measure_nozzle(self):
        e = self.env
        want = not bool(getattr(e, "tab_open", False))
        e._set_tab_open(want)
        print(f"[measure_ingredient] nozzle {'ON' if want else 'OFF'}")

    def _toggle_stove(self):
        e, t = self.env, self.task
        lit = bool(getattr(e, "stove_on", False)) or float(
            getattr(e, "fire_intensity", 0.0)
        ) > 0.02
        want_on = not lit
        self._snap_stove_knob(want_on)
        print(f"[{t}] stove {'ON' if want_on else 'OFF'}")

    def _mouse_picture_xy(self, viewer):
        window = viewer.window
        mx, my = window.mouse_position
        ww, wh = window.size
        if ww <= 0 or wh <= 0 or mx < 0 or my < 0 or mx >= ww or my >= wh:
            return None
        tw, th = window.get_picture_size("Segmentation")
        return int(mx * tw / ww), int(my * th / wh)

    def _click_hits(self, viewer, pixel_x, pixel_y, actor) -> bool:
        if actor is None:
            return False
        try:
            return bool(click_hits_actor(viewer, pixel_x, pixel_y, actor))
        except Exception:
            return False

    def _click_hits_any(self, viewer, pixel_x, pixel_y, actors) -> bool:
        for actor in actors:
            if actor is None:
                continue
            if isinstance(actor, (list, tuple)):
                if self._click_hits_any(viewer, pixel_x, pixel_y, actor):
                    return True
                continue
            if self._click_hits(viewer, pixel_x, pixel_y, actor):
                return True
        return False

    def _table_z(self):
        return float(getattr(self.env, "table_top", 0.74))

    def on_click(self, viewer, pixel_x, pixel_y):
        """Keyboard+mouse click dispatch (ignored in robot mode)."""
        if self.robot:
            return False
        t, e = self.task, self.env
        try:
            if t == "trap_bug":
                return self._click_drop_trap(viewer, pixel_x, pixel_y)
            if t in ("boil_milk", "cook_food", "cook_food_timer"):
                knob = getattr(e, "stove_knob", None)
                if self._click_hits(viewer, pixel_x, pixel_y, knob):
                    self._toggle_stove()
                    return True
                hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
                kxy = getattr(e, "knob_xy", None)
                if hit is not None and kxy is not None:
                    if float(np.linalg.norm(np.asarray(hit) - np.asarray(kxy[:2]))) < 0.06:
                        self._toggle_stove()
                        return True
                return False
            if t == "pour_beer":
                if self._click_hits(viewer, pixel_x, pixel_y, getattr(e, "bell", None)):
                    e._bell_pressed = True
                    print("[pour_beer] finish bell pressed")
                    return True
                return False
            if t == "measure_ingredient":
                return self._click_teleport_jar(viewer, pixel_x, pixel_y)
            if t == "make_soup":
                return self._click_place_board(viewer, pixel_x, pixel_y)
            if t == "catch_cup":
                return self._click_teleport_pillow(viewer, pixel_x, pixel_y)
            if t == "catch_mouse_object_drop":
                return self._click_teleport_basket(viewer, pixel_x, pixel_y)
            if t == "stop_ball":
                return self._click_stop_ball(viewer, pixel_x, pixel_y)
            if t == "clean_table":
                return self._click_sponge(viewer, pixel_x, pixel_y)
        except Exception as exc:
            print(f"[{t}] click unavailable: {exc}")
        return False

    def _click_drop_trap(self, viewer, pixel_x, pixel_y):
        e = self.env
        if bool(getattr(e, "_trap_released", False)) or bool(getattr(e, "_trap_falling", False)):
            return False
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        hz = float(e.trap_half[2])
        z = float(e.table_top) + hz + 0.04
        q = list(e.trap.get_pose().q)
        e._set_trap_pose(sapien.Pose([float(hit[0]), float(hit[1]), z], q))
        e._trap_welded = False
        e._trap_released = True
        e._trap_falling = True
        print(f"[trap_bug] trap dropping from 4 cm at ({hit[0]:.3f}, {hit[1]:.3f})")
        return True

    def _click_place_board(self, viewer, pixel_x, pixel_y):
        e = self.env
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        hz = float(e.board_half[2])
        top_z = float(e.pot_rim_z) + 0.02
        z = top_z - hz
        old = e.board.get_pose()
        new = sapien.Pose([float(hit[0]), float(hit[1]), z], [1.0, 0.0, 0.0, 0.0])
        rels = []
        for veg in getattr(e, "veggies", []) or []:
            try:
                rels.append((veg, old.inv() * veg.get_pose()))
            except Exception:
                rels.append((veg, None))
        e._set_entity_pose(e.board, new, snap=True)
        for veg, rel in rels:
            if rel is None:
                continue
            try:
                e._set_entity_pose(veg, new * rel, snap=True)
            except Exception:
                pass
        self._board_placed = True
        self._board_tilt = 0.0
        self._board_flat_pose = new
        e._pour_armed = True
        e._score_veg_spill = True
        try:
            e._ensure_veggies_dynamic()
        except Exception:
            pass
        print(
            f"[make_soup] board top-center at ({hit[0]:.3f}, {hit[1]:.3f}), "
            f"2 cm above pot rim"
        )
        return True

    def _click_teleport_jar(self, viewer, pixel_x, pixel_y):
        e = self.env
        if getattr(e, "jar", None) is None:
            return False
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        z = float(e.table_top) + 0.001
        q = list(e.jar.get_pose().q)
        pose = sapien.Pose([float(hit[0]), float(hit[1]), z], q)
        if callable(getattr(e, "_freeze_jar", None)):
            e._push_active = False
            e._freeze_jar(pose)
        else:
            _set_pose(e.jar, pose.p, quat=q, kinematic=True)
        if callable(getattr(e, "_sync_jar_followers", None)):
            e._sync_jar_followers()
        print(f"[measure_ingredient] jar at ({hit[0]:.3f}, {hit[1]:.3f})")
        return True

    def _click_teleport_pillow(self, viewer, pixel_x, pixel_y):
        e = self.env
        if self._catcher_placed or getattr(e, "pillow", None) is None:
            return False
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        z = float(e.table_top) + 0.5 * float(getattr(e, "pillow_height", 0.04))
        q = getattr(e, "PILLOW_Q", None)
        quat = q.tolist() if hasattr(q, "tolist") else list(e.pillow.get_pose().q)
        pose = sapien.Pose([float(hit[0]), float(hit[1]), z], quat)
        e._push_active = False
        e._pillow_placed = True
        if callable(getattr(e, "_freeze_pillow", None)):
            e._freeze_pillow(pose)
        else:
            _set_pose(e.pillow, pose.p, quat=quat, kinematic=True)
        self._catcher_placed = True
        print(f"[catch_cup] pillow on table at ({hit[0]:.3f}, {hit[1]:.3f})")
        return True

    def _click_teleport_basket(self, viewer, pixel_x, pixel_y):
        e = self.env
        if self._catcher_placed or getattr(e, "basket", None) is None:
            return False
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        hz = float(getattr(e, "basket_hz", 0.5 * float(getattr(e, "basket_height", 0.07))))
        q = list(e.basket.get_pose().q)
        xyz = [float(hit[0]), float(hit[1]), float(e.table_top) + hz]
        _set_pose(e.basket, xyz, quat=q, kinematic=True)
        e._basket_placed = True
        self._catcher_placed = True
        print(f"[catch_mouse_object_drop] basket on table at ({hit[0]:.3f}, {hit[1]:.3f})")
        return True

    def _click_stop_ball(self, viewer, pixel_x, pixel_y):
        e = self.env
        if self._stop_ball_bridge_placed or self._stop_ball_bridge is None:
            return False
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        x, y = float(hit[0]), float(hit[1])

        # Turn the bridge across the sampled roll heading, with its opening
        # facing the incoming ball. Local X spans across the path.
        direction = np.asarray(getattr(e, "_roll_dir", [0.0, -1.0]), dtype=float)
        yaw = float(np.arctan2(direction[1], direction[0]) - np.pi / 2.0)
        quat = [float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw))]
        _set_pose(
            self._stop_ball_bridge,
            [x, y, float(e.table_top)],
            quat=quat,
            kinematic=True,
        )
        self._stop_ball_bridge_placed = True
        e._stopper_placed = True
        print(f"[stop_ball] gripper bridge placed at ({x:.3f}, {y:.3f})")
        return True

    def _click_sponge(self, viewer, pixel_x, pixel_y):
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, self._table_z())
        if hit is None:
            return False
        self._contact_sponge(hit[0], hit[1])
        return True

    def _tilt_board(self, window):
        if not self._board_placed or self._board_flat_pose is None:
            return
        now = time.perf_counter()
        dt = 0.016 if self._last_tilt_t is None else max(0.0, min(0.05, now - self._last_tilt_t))
        self._last_tilt_t = now
        direction = float(window.key_down("right")) - float(window.key_down("left"))
        if abs(direction) < 1e-6:
            return
        self._board_tilt = float(np.clip(self._board_tilt + direction * 0.9 * dt, -1.2, 1.2))
        e = self.env
        pose = e._rot_about_y(self._board_flat_pose, self._board_flat_pose.p, self._board_tilt)
        e._set_entity_pose(e.board, pose, snap=True)
        e._pour_armed = True
        e._score_veg_spill = True

    def _pour_held_now(self, viewer, window) -> bool:
        if bool(window.key_down("space")):
            return True
        if not bool(window.mouse_down(0)):
            return False
        pix = self._mouse_picture_xy(viewer)
        if pix is None:
            return False
        e = self.env
        # Red jewel is the visible press target; also accept brass cap / chrome rim.
        if self._click_hits_any(
            viewer,
            pix[0],
            pix[1],
            (
                getattr(e, "_button_jewel", None),
                getattr(e, "_button_rim", None),
                getattr(e, "_button_actor", None),
                getattr(e, "_button_bezel", None),
            ),
        ):
            return True
        # Fallback: click near the button XY on the tap head.
        hit = table_xy_from_click(viewer, pix[0], pix[1], self._table_z())
        touch = getattr(e, "touch_xy", None)
        if hit is not None and touch is not None:
            r = float(getattr(e, "BTN_HALF", [0.018, 0.018, 0.01])[0]) + 0.012
            if float(np.linalg.norm(np.asarray(hit)[:2] - np.asarray(touch)[:2])) <= r:
                return True
        return False

    def _fill_coffee_press(self, force_level):
        """Dispense at a chosen force level (keyboard 1–4), no arm press."""
        e = self.env
        level = max(1, min(4, int(force_level)))
        orig_force = getattr(e, "_lid_press_force", None)
        try:
            force_n = float(e.force_thresholds[level - 1])
            e._lid_press_force = lambda: force_n
            e._awaiting_release = False
            e._start_press(require_release=False)
            if not bool(getattr(e, "_press_active", False)):
                return
            e._press_peak_force = force_n
            e._press_dispense_level = level
            e._press_force_level = level
            e._end_press()
            print(f"[fill_coffee_jar] completed force level {level} press")
        except Exception as exc:
            print(f"[fill_coffee_jar] level {level} press unavailable: {exc}")
        finally:
            if orig_force is not None:
                e._lid_press_force = orig_force
            e._awaiting_release = False

    @staticmethod
    def _fill_force_level(held_seconds):
        """Use one force level per 0.5 s, with level 4 at 2 s and above."""
        return max(1, min(4, int(np.ceil(max(0.0, float(held_seconds)) / 0.5))))

    def _fill_tcp_pose(self, side):
        getter = (
            self.env.robot.get_left_tcp_pose
            if side == "left"
            else self.env.robot.get_right_tcp_pose
        )
        return np.asarray(getter(), dtype=np.float64)

    def _fill_arm_qpos(self, side):
        joints = (
            self.env.robot.left_arm_joints
            if side == "left"
            else self.env.robot.right_arm_joints
        )
        entity = (
            self.env.robot.left_entity
            if side == "left"
            else self.env.robot.right_entity
        )
        try:
            qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
            active = list(entity.get_active_joints())
            return np.asarray([qpos[active.index(joint)] for joint in joints], dtype=np.float64)
        except Exception:
            return np.asarray(
                [float(np.asarray(joint.get_drive_target()).reshape(-1)[0]) for joint in joints],
                dtype=np.float64,
            )

    def _fill_over_dispenser(self, side):
        tcp = self._fill_tcp_pose(side)
        distance = float(np.linalg.norm(tcp[:2] - np.asarray(self.env.touch_xy, dtype=float)))
        return distance <= self.FILL_XY_TOL

    def _fill_contact(self, state):
        """Detect real contact, or a collision-limited stop at the button top."""
        side = state["side"]
        if not self._fill_over_dispenser(side):
            return False
        tcp_z = float(self._fill_tcp_pose(side)[2])
        dz = tcp_z - float(self.env.touch_top_z)
        real_contact = float(self.env._lid_contact_force()) > 0.5
        geometric_contact = dz <= self.FILL_CONTACT_DZ
        stalled = (
            state["descended"] >= 0.010
            and dz <= self.FILL_STALL_DZ
            and abs(tcp_z - state["last_tcp_z"]) < 0.0008
        )
        state["last_tcp_z"] = tcp_z
        return bool(real_contact or geometric_contact or stalled)

    def _start_fill_press(self):
        selected = require_selected_arms(self.env, exactly_one=True)
        if not selected:
            return
        side = selected[0]
        from envs.utils.action import ArmTag

        arm = ArmTag(side)
        return_qpos = self._fill_arm_qpos(side)
        self.env._pressing_arm_side = side
        self.env._interactive_teleop_locked = True
        high_dis = float(getattr(self.env, "KEY_HOVER_DIS", 0.06)) + 0.08
        hover_dis = float(getattr(self.env, "KEY_HOVER_DIS", 0.06))
        self.env.plan_success = True
        moved = self.env.move(self.env.move_to_pose(arm, self.env._touch_tip_pose(high_dis)))
        if moved is not False and bool(getattr(self.env, "plan_success", True)):
            self.env.move(self.env.move_by_displacement(arm, z=-(high_dis - hover_dis)))
        if moved is False or not bool(getattr(self.env, "plan_success", True)):
            self.env.plan_success = True
            moved = self.env.move(self.env.move_to_pose(arm, self.env._touch_tip_pose(hover_dis)))
        if moved is False or not bool(getattr(self.env, "plan_success", True)):
            self.env._interactive_teleop_locked = False
            action_failed(
                self.env, (side,),
                detail="could not move above the blue button",
            )
            return
        tcp_z = float(self._fill_tcp_pose(side)[2])
        self._fill_press_state = {
            "side": side,
            "arm": arm,
            "return_qpos": return_qpos,
            "descended": 0.0,
            "last_tcp_z": tcp_z,
            "contact_started_at": None,
            "descent_stopped": False,
        }
    def _hold_fill_press(self):
        state = self._fill_press_state
        if state is None or state["descent_stopped"]:
            return
        remaining = self.FILL_MAX_PRESS_DEPTH - float(state["descended"])
        if remaining <= 1e-6:
            state["descent_stopped"] = True
            print("[fill_coffee_jar] press-depth cap reached without dispenser contact")
            return
        contacting = self._fill_contact(state)
        if contacting and state["contact_started_at"] is None:
            state["contact_started_at"] = time.perf_counter()
            print("[fill_coffee_jar] dispenser contact; button is physically pressed")
        step = min(self.FILL_PRESS_STEP, remaining)
        if contacting:
            step = min(max(step * 0.5, 0.002), remaining)
        self.env.plan_success = True
        moved = self.env.move(self.env.move_by_displacement(state["arm"], z=-step))
        if moved is False or not bool(getattr(self.env, "plan_success", True)):
            state["descent_stopped"] = True
            print("[fill_coffee_jar] downward press plan stopped; release Space to return")
            return
        state["descended"] += step
        if state["descended"] >= self.FILL_MAX_PRESS_DEPTH - 1e-6:
            state["descent_stopped"] = True
            print("[fill_coffee_jar] press-depth cap reached; release Space to return")

    def _return_fill_arm(self, state):
        side = state["side"]
        target = np.asarray(state["return_qpos"], dtype=np.float64)
        start = self._fill_arm_qpos(side)
        for index in range(1, self.FILL_RETURN_STEPS + 1):
            alpha = index / float(self.FILL_RETURN_STEPS)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            qpos = start + (target - start) * smooth
            velocity = (target - start) / max(self.FILL_RETURN_STEPS, 1)
            self.env.robot.set_arm_joints(qpos, velocity, side)
            if hasattr(self.env, "_update_kinematic_tasks"):
                self.env._update_kinematic_tasks()
            self.env.scene.step()
        self.env.robot.set_arm_joints(target, np.zeros_like(target), side)
        self.env.plan_success = True

    def _finish_fill_press(self):
        state = self._fill_press_state
        self._fill_press_state = None
        if state is None:
            self.env._interactive_teleop_locked = False
            return
        try:
            self._return_fill_arm(state)
        finally:
            self.env._interactive_teleop_locked = False
        print("[fill_coffee_jar] released; dispensing is driven by measured button press")

    def _robot_action(self):
        """Use the environment's physical robot helper for discrete actions."""
        e, t = self.env, self.task
        arm = None
        try:
            # A failed plan latches this flag false. Each deliberate key press
            # starts a fresh attempt so the same action can be retried.
            e.plan_success = True
            if t == "boil_milk":
                self._turn_boil_milk_knob()
            elif t == "fill_coffee_jar":
                arm = _arm_tag(e)
                if arm is None:
                    return
                e._press_dispenser(arm, force_level=4)
                if not bool(getattr(e, "plan_success", True)):
                    action_failed(e, (str(arm),), detail="dispenser press failed")
            elif t == "measure_ingredient":
                # No C assist — oil key is pressed by lowering the closed gripper.
                print(
                    "[measure_ingredient] push jar under nozzle, then lower onto "
                    "the green key; success is checked after the key turns OFF"
                )
        except Exception as exc:
            if arm is not None:
                action_failed(e, (str(arm),), detail=f"action unavailable: {exc}")
            else:
                print(f"[{t}] action unavailable: {exc}")

    def _snap_stove_knob(self, want_on: bool, *, continuous_angle: float | None = None):
        """Instant interactive knob/fire update (no planner reach).

        Matches the snappy base-suite control feel: teleop stays on UniversalRobotControls;
        Space/C only flips task state instead of running a multi-second expert path.
        """
        e = self.env
        # Drop debounce caches so the snap is applied even if the joint was near
        # the previous committed angle.
        e._last_committed_knob_angle = None
        e._stove_fire_visual = None
        # cook_food*: fire intensity + blue ring live on ``_set_knob_angle``.
        # Only setting the joint left the halo lit after a keyboard shutoff.
        if callable(getattr(e, "_set_knob_angle", None)):
            if want_on:
                if continuous_angle is not None:
                    angle = float(continuous_angle)
                else:
                    max_ang = float(
                        getattr(e, "KNOB_MAX_ANGLE", abs(float(e.KNOB_ON_ANGLE)))
                    )
                    angle = -float(getattr(e, "cook_intensity", 1.0)) * max_ang
            else:
                angle = 0.0
            e._knob_grasp_active = False
            e._policy_controlling_knob = False
            e._expert_holding_knob = False
            e._set_knob_angle(angle, drive_fire=True)
            if callable(getattr(e, "_set_knob_joint_angle", None)):
                e._set_knob_joint_angle(angle, hard=True)
            if callable(getattr(e, "_set_knob_articulation_qpos", None)):
                e._set_knob_articulation_qpos(angle)
            if callable(getattr(e, "_hold_knob_joint", None)):
                e._hold_knob_joint(stiff=True)
            e.fire_intensity = 0.0 if not want_on else float(getattr(e, "fire_intensity", 1.0))
            e.stove_on = bool(want_on)
            if not want_on:
                e.fire_intensity = 0.0
                e.stove_on = False
                if callable(getattr(e, "_set_burner_visuals", None)):
                    e._set_burner_visuals(0.0)
                elif callable(getattr(e, "_set_stove_fire", None)):
                    e._set_stove_fire(False, intensity=0.0)
            e._last_committed_knob_angle = float(angle)
            return
        angle = e.KNOB_ON_ANGLE if want_on else e.KNOB_OFF_ANGLE
        if callable(getattr(e, "_set_knob_joint_angle", None)):
            e._set_knob_joint_angle(angle, hard=True)
        if callable(getattr(e, "_set_stove", None)):
            e._set_stove(bool(want_on))
        if callable(getattr(e, "_set_stove_fire", None)):
            e._set_stove_fire(bool(want_on), intensity=1.0 if want_on else 0.0)

    def _return_arm_after_failure(self, arm):
        """Best-effort recovery for a failed scripted reach."""
        e = self.env
        try:
            e.plan_success = True
            e.move(e.open_gripper(arm))
        except Exception as exc:
            print(f"[boil_milk] could not open {arm} gripper during recovery: {exc}")
        try:
            # A failed gripper command must not prevent the return plan.
            e.plan_success = True
            e.move(e.back_to_origin(arm))
        except Exception as exc:
            print(f"[boil_milk] could not return {arm} arm to origin: {exc}")

    def _show_failed_arm_red(self, arm):
        """Tint the failed arm's gripper red for two seconds."""
        flash_gripper_failure(self.env, (str(arm),))

    def _update_failure_visual(self):
        gripper_failure_feedback(self.env).update()

    def _turn_boil_milk_knob(self):
        """Turn the knob with exactly the arm selected by interactive controls."""
        e = self.env
        arm = _arm_tag(e)
        if arm is None:
            return

        previous_arm = getattr(e, "arm", None)
        stove_was_on = bool(getattr(e, "stove_on", False))
        wanted_on = not stove_was_on
        action_error = None
        try:
            e.arm = arm
            e.plan_success = True
            e._turn_knob(wanted_on)
        except Exception as exc:
            action_error = exc
        finally:
            e.arm = previous_arm
            e._ignore_knob = False

        # The physical stove state is authoritative. If it did not reach the
        # requested state, the next Space press requests that same action again.
        reached_target = bool(getattr(e, "stove_on", False)) == wanted_on
        if reached_target:
            print(f"[boil_milk] knob turned with {arm} arm")
            return

        detail = (
            f"could not turn the knob: {action_error}"
            if action_error is not None
            else "could not turn the knob"
        )
        print(f"[boil_milk] {arm} arm failed; returning it to origin")
        self._return_arm_after_failure(arm)
        action_failed(e, (str(arm),), detail=detail)

    def _task_action(self):
        if self.robot:
            self._robot_action()
        else:
            self._keyboard_action()

    def update(self, window):
        self._update_failure_visual()
        if self.robot:
            return
        e, t = self.env, self.task
        viewer = getattr(e, "viewer", None)
        if t in ("boil_milk", "cook_food", "cook_food_timer"):
            if edge_pressed(window, "space", self._key_prev):
                self._toggle_stove()
        elif t == "fill_coffee_jar":
            for n, key in enumerate(("1", "2", "3", "4"), start=1):
                if edge_pressed(window, key, self._key_prev):
                    e._awaiting_release = False
                    self._fill_coffee_press(n)
                    break
        elif t == "pour_beer":
            self._pour_held = bool(self._pour_held_now(viewer, window))
        elif t == "measure_ingredient":
            if edge_pressed(window, "space", self._key_prev):
                self._toggle_measure_nozzle()
        elif t == "make_soup":
            self._tilt_board(window)
        down = bool(window.mouse_down(0)) if viewer is not None else False
        if (
            not self._click_via_handler
            and down
            and not self._prev_mouse
            and viewer is not None
        ):
            pix = self._mouse_picture_xy(viewer)
            if pix is not None:
                self.on_click(viewer, pix[0], pix[1])
        self._prev_mouse = down

    def start_scenario(self):
        """Start time-sensitive scene motion after the first rendered frame."""
        if self.scenario_started:
            return
        self.scenario_started = True
        e = self.env
        try:
            if self.task == "trap_bug":
                e._start_bug()
            elif self.task == "catch_cup":
                e._release_cup()
            elif self.task == "catch_mouse_object_drop":
                # Release shelf objects and start the mouse immediately (no
                # stand-off wait for basket placement).
                e._activate_target()
                e._allow_shove = True
                e._release_mouse()
            elif self.task == "stop_ball":
                # Ball goes live at handoff; a miss keeps rolling until fall-off.
                e._release_ball()
            elif self.task == "clean_table" and not bool(e.cup_tipped):
                e._animate_tip()
            else:
                return
            print(f"[{self.task}] scenario started automatically")
        except Exception as exc:
            print(f"[{self.task}] automatic start failed: {exc}")

    def after_step(self):
        """Post-physics hooks (release arming, catch/miss latch, trap freeze)."""
        env = self.env
        if self.task == "stop_ball":
            self._poll_stop_ball_bridge_contact()
            return
        if self.task == "catch_mouse_object_drop":
            # Catch/miss is evaluated before scene.step inside
            # _update_kinematic_tasks; re-check after the step so a table
            # landing latches immediately for terminal failure.
            try:
                env._update_catch_state()
            except Exception:
                pass
            return
        if self.task != "trap_bug":
            return
        if env.trap is None:
            return
        # Gripper open / slip is detected in env._maybe_auto_weld_or_release.
        if bool(getattr(env, "_trap_released", False)):
            self.trap_released = True
        if bool(getattr(env, "_trap_anchored", False)) and not getattr(self, "_trap_land_logged", False):
            self._trap_land_logged = True
            print("[trap_bug] trap landed; pose frozen as-is")

    def _poll_stop_ball_bridge_contact(self):
        """Latch real PhysX contact between the ball and placed bridge."""
        if (
            not self._stop_ball_bridge_placed
            or bool(getattr(self.env, "_stopper_contacted", False))
            or getattr(self.env, "ball", None) is None
        ):
            return
        ball_name = self.env.ball.get_name()
        bridge_name = self._stop_ball_bridge.get_name()
        try:
            for contact in self.env.scene.get_contacts():
                names = {
                    contact.bodies[0].entity.name,
                    contact.bodies[1].entity.name,
                }
                if ball_name in names and bridge_name in names:
                    self.env._stopper_contacted = True
                    print("[stop_ball] ball contacted the gripper bridge")
                    return
        except Exception:
            pass


def _fill_level_detail(env, task: str) -> str:
    """Compact fill readout for terminal SUCCESS/FAILURE lines."""
    if task == "pour_beer":
        liq = 100.0 * float(getattr(env, "liquid_level", 0.0))
        foam = 100.0 * float(getattr(env, "foam_level", 0.0))
        try:
            total = 100.0 * float(env._total_fill())
        except Exception:
            total = liq + foam
        tgt = 100.0 * float(getattr(env, "target_liquid", 0.90))
        band = f"need>{tgt:.0f}%"
        if bool(getattr(env, "overflowed", False)):
            return f"OVERFLOW beer={liq:.0f}% foam={foam:.0f}% total={total:.0f}%"
        return f"beer={liq:.0f}% foam={foam:.0f}% total={total:.0f}% {band}"
    if task == "boil_milk":
        lvl = 100.0 * float(getattr(env, "liquid_level", 0.0))
        tgt = 100.0 * float(getattr(env, "target_level", 1.0))
        if bool(getattr(env, "overflowed", False)):
            return f"OVERFLOW level={lvl:.0f}%"
        return f"level={lvl:.0f}% target={tgt:.0f}%"
    if task == "measure_ingredient":
        lvl = 100.0 * float(getattr(env, "liquid_level", 0.0))
        try:
            lo, hi = env._fill_band()
            lo_pct, hi_pct = 100.0 * float(lo), 100.0 * float(hi)
        except Exception:
            tgt = 100.0 * float(getattr(env, "target_fill", 0.0))
            tol = 100.0 * float(getattr(env, "fill_tol", 0.05))
            lo_pct, hi_pct = tgt - tol, tgt + tol
        if bool(getattr(env, "overflowed", False)):
            return f"OVERFLOW fill={lvl:.0f}%"
        return f"fill={lvl:.0f}% target={lo_pct:.0f}–{hi_pct:.0f}%"
    if task == "fill_coffee_jar":
        try:
            fill = 100.0 * float(env._current_fill())
        except Exception:
            fill = 0.0
        tgt = 100.0 * float(getattr(env, "target_fill", 0.0))
        tol = 100.0 * float(getattr(env, "fill_tol", 0.05))
        idle = float(getattr(env, "_press_idle_s", 0.0))
        need = float(getattr(env, "idle_score_sec", 3.0))
        return (
            f"fill={fill:.0f}% target={tgt - tol:.0f}–{tgt + tol:.0f}% "
            f"idle={idle:.1f}/{need:.0f}s"
        )
    return ""


def _terminal_failure(env, task):
    """Return an irreversible task failure reason, or ``None`` while playable."""
    if task == "trap_bug":
        if bool(getattr(env, "_bug_escaped", False)):
            return "bug went back into hiding"
        if bool(getattr(env, "_trap_anchored", False)):
            return "trap missed the bug"
    elif task in ("boil_milk", "pour_beer", "measure_ingredient"):
        if bool(getattr(env, "overflowed", False)):
            return "liquid overflowed"
        # pour_beer: score only when the finish bell is pressed.
        if task == "pour_beer" and bool(getattr(env, "_bell_pressed", False)):
            try:
                if bool(env._pour_quality_ok()):
                    return None
            except Exception:
                pass
            liq = 100.0 * float(getattr(env, "liquid_level", 0.0))
            tgt = 100.0 * float(getattr(env, "target_liquid", 0.85))
            if not bool(getattr(env, "opened_once", False)):
                return "finish bell pressed before pouring"
            if liq + 1e-3 <= tgt:
                return f"finish bell pressed with underfill beer={liq:.0f}% need>{tgt:.0f}%"
            return "finish bell pressed without meeting pour criteria"
        if task == "boil_milk":
            turned_on = bool(getattr(env, "turned_on_once", False))
            stove_off = not bool(getattr(env, "stove_on", False))
            reached_target = bool(getattr(env, "reached_target", False))
            max_level = float(getattr(env, "max_liquid_level", 0.0))
            target_level = float(getattr(env, "target_level", 1.0))
            if turned_on and stove_off and not reached_target and max_level < target_level - 1e-3:
                return "stove turned off before milk reached the target"
        if task == "measure_ingredient":
            if float(getattr(env, "spill_amount", 0.0)) > 1e-4:
                return "ingredient spilled"
            # Score only after the nozzle key is turned OFF (success is checked
            # first in the viewer loop; this path is the fail branch).
            if bool(getattr(env, "closed_after_pour", False)):
                lvl = float(getattr(env, "liquid_level", 0.0))
                try:
                    under = bool(env._jar_under_nozzle())
                except Exception:
                    under = False
                if not under:
                    return "switch turned off with jar not under the nozzle"
                try:
                    lo, hi = env._fill_band()
                except Exception:
                    tgt = float(getattr(env, "target_fill", 0.0))
                    tol = float(getattr(env, "fill_tol", 0.05))
                    lo, hi = tgt - tol, tgt + tol
                if lvl + 1e-3 < float(lo) or lvl - 1e-3 > float(hi):
                    return "switch turned off with incorrect fill level"
                return "switch turned off without meeting success criteria"
    elif task == "fill_coffee_jar":
        try:
            if not bool(env._fill_ready_to_score()):
                return None
            fill = float(env._current_fill())
            lo, hi = env._fill_band()
            if fill > float(hi) + 1e-3:
                return "coffee jar overfilled after idle"
            if fill + 1e-3 < float(lo):
                return "coffee jar underfilled after idle"
            return "fill outside target after idle"
        except Exception:
            pass
    elif task in ("cook_food", "cook_food_timer"):
        try:
            lo, hi = env.target_doneness_range
            # Overcook is irreversible — fail as soon as doneness passes the band.
            if float(env.doneness) > float(hi) + 1e-3:
                return "food overcooked"
            # Under/out-of-band after shutoff only (success is checked first).
            stove_off = (
                not bool(getattr(env, "stove_on", False))
                and float(getattr(env, "fire_intensity", 1.0)) <= 0.02
                and float(getattr(env, "knob_angle", -1.0)) >= -0.05
            )
            if not stove_off or not bool(getattr(env, "turned_off_after_cook", False)):
                return None
            grasped = getattr(env, "_grasp_doneness", None)
            score = float(env.doneness) if grasped is None else float(grasped)
            if score + 1e-3 < float(lo):
                return "food undercooked"
            if not env._doneness_in_target_range(score):
                return "food outside target doneness"
        except Exception:
            pass
    elif task == "make_soup":
        if bool(getattr(env, "_arm_veg_contact", False)):
            return getattr(env, "_fail_reason", None) or "arm contacted vegetables"
        if bool(getattr(env, "_veg_fallen", False)):
            return getattr(env, "_fail_reason", None) or "vegetables dropped on the table"
        # Live re-check so a mid-episode spill latches before the next success poll.
        try:
            if callable(getattr(env, "_check_veg_fallen", None)):
                env._check_veg_fallen()
            if callable(getattr(env, "_check_arm_veg_contact", None)):
                env._check_arm_veg_contact()
        except Exception:
            pass
        if bool(getattr(env, "_arm_veg_contact", False)):
            return getattr(env, "_fail_reason", None) or "arm contacted vegetables"
        if bool(getattr(env, "_veg_fallen", False)):
            return getattr(env, "_fail_reason", None) or "vegetables dropped on the table"
    elif task == "catch_cup":
        if bool(getattr(env, "_fell_on_table", False)) or getattr(env, "_cup_state", "") == "fallen":
            return "cup fell on the table"
    elif task == "catch_mouse_object_drop":
        # Live re-check: tipped landings used to miss pose-based contact, so
        # latch failure from the AABB table test before reading the flags.
        try:
            if callable(getattr(env, "_object_touches_table", None)) and env._object_touches_table():
                env._fell_on_table = True
                env._caught = False
                env._obj_state = "fallen"
        except Exception:
            pass
        if bool(getattr(env, "_fell_on_table", False)) or getattr(env, "_obj_state", "") == "fallen":
            return "object fell on the table"
    elif task == "stop_ball":
        if bool(getattr(env, "_fell_off", False)) or getattr(env, "_ball_state", "") == "fallen":
            return "ball fell off the table without being stopped"
    elif task == "clean_table":
        if bool(getattr(env, "laptop_reached", False)):
            return "spill reached the laptop"
    return None


def _terminal_hold_seconds(env, task: str) -> float:
    """Wall-clock seconds to keep rendering after a terminal verdict.

    stop_ball closes 1 s after the ball leaves the table: once it is falling
    there is no settle left to watch, so it must not sit on the shared 2 s hold.
    """
    if task == "stop_ball" and (
        bool(getattr(env, "_fell_off", False))
        or str(getattr(env, "_ball_state", "")) == "fallen"
    ):
        return 1.0
    return 2.0


def run_task(task, args, keyboard_controls, robot_controls, post_setup=None):
    module_name, class_name, _, _ = PROFILES[task]
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    from envs import CONFIGS_PATH

    globals()["CONFIGS_PATH"] = CONFIGS_PATH
    use_robot = is_robot_control(args.control)
    args.control = normalize_control_mode(args.control)
    config = configure_task(task, args.config, args.seed, use_robot)
    task_args = config.setdefault("task_args", {}).setdefault(task, {})
    for item in args.task_arg:
        if "=" not in item:
            raise SystemExit(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                value = raw
        task_args[key.strip()] = value
    env = task_cls()
    env._interactive_robot_mode = use_robot
    # Viewer sessions (keyboard or robot teleop) — tasks use this for live physics
    # handoff / miss-continues-until-fall behavior (e.g. stop_ball). Set before
    # setup_demo so interactive warning suppression applies during load.
    env._interactive_session = True
    env.setup_demo(**config)
    prepare_interactive_control(env, args.control)
    if post_setup is not None:
        post_setup(env)
    print_episode_condition(env, task)
    print_mode_controls(task, args.control, keyboard=keyboard_controls, robot=robot_controls)
    controller = HouseholdController(env, task, robot=use_robot)
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    if not use_robot:
        cw = getattr(viewer, "control_window", None)
        if cw is None:
            for plugin in getattr(viewer, "plugins", []) or []:
                if hasattr(plugin, "click_handlers") and hasattr(
                    plugin, "register_click_handler"
                ):
                    cw = plugin
                    break
        if cw is not None:
            cw.register_click_handler(controller.on_click)
            controller._click_via_handler = True
    views = make_viewer_view_toggle(env, viewer)
    # Append after declutter so this HUD stays visible (stock ImGui is hidden).
    if task in ("measure_ingredient", "fill_coffee_jar"):
        hud = FillTargetHud(env, task)
        try:
            hud.init(viewer)
        except Exception:
            pass
        try:
            viewer.plugins.append(hud)
        except Exception:
            pass
    rendered_frames = 0
    terminal_result = None  # True=success, False=failure, None=manual close/smoke
    terminal_started_at = None
    terminal_fill_detail = ""
    terminal_failure_reason = None
    terminal_partial_score = None  # freeze PS at the terminal decision frame
    terminal_hold_s = 2.0
    escape_quit = False
    # Match interactive/_interactive_common run_viewer_loop: teleop once per display frame, then
    # fixed-dt physics catch-up so 60 Hz / 240 Hz monitors feel the same speed.
    # Success checks every few *physics* steps keep kitchen eval cost down.
    SUCCESS_CHECK_EVERY = 5
    pacer = RealtimePhysicsPacer(env)
    physics_steps = 0
    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    if terminal_result is None:
                        terminal_result = False
                        terminal_failure_reason = ESCAPE_QUIT_DETAIL
                        escape_quit = True
                    break
                if (
                    terminal_started_at is not None
                    and time.perf_counter() - terminal_started_at >= terminal_hold_s
                ):
                    print(
                        f"[{task}] closing after {terminal_hold_s:g}-second "
                        "terminal-result display"
                    )
                    break
                continue

            controller.update(viewer.window)
            for _ in range(n_steps):
                if hasattr(env, "_update_kinematic_tasks"):
                    env._update_kinematic_tasks()
                env.scene.step()
                controller.after_step()
                physics_steps += 1

                if terminal_started_at is not None:
                    continue
                if physics_steps % SUCCESS_CHECK_EVERY == 0:
                    try:
                        succeeded = bool(env.check_success())
                    except Exception as exc:
                        print_failure(f"[{task}] success check failed: {exc}")
                        succeeded = False
                    failure = None if succeeded else _terminal_failure(env, task)
                    fill = _fill_level_detail(env, task)
                    if fill:
                        terminal_fill_detail = fill
                    if succeeded:
                        terminal_partial_score = _env_partial_score(env)
                        msg = f"[{task}] terminal result: SUCCESS"
                        if fill:
                            msg = f"{msg} ({fill})"
                        msg = f"{msg}{format_ps_suffix(terminal_partial_score)}"
                        print_success(msg)
                        terminal_result = True
                        terminal_hold_s = _terminal_hold_seconds(env, task)
                        terminal_started_at = time.perf_counter()
                    if failure is not None:
                        terminal_partial_score = _env_partial_score(env)
                        if fill and fill not in failure:
                            terminal_failure_reason = f"{failure}; {fill}"
                        else:
                            terminal_failure_reason = failure
                        print_failure(
                            f"[{task}] terminal result: FAILURE "
                            f"({terminal_failure_reason})"
                            f"{format_ps_suffix(terminal_partial_score)}"
                        )
                        terminal_result = False
                        terminal_hold_s = _terminal_hold_seconds(env, task)
                        terminal_started_at = time.perf_counter()

            env.scene.update_render()
            viewer.render()
            rendered_frames += 1
            if rendered_frames == 1 and not args.smoke_test:
                controller.start_scenario()
            if viewer.window.key_down("escape"):
                if terminal_result is None:
                    terminal_result = False
                    terminal_failure_reason = ESCAPE_QUIT_DETAIL
                    escape_quit = True
                break
            if args.smoke_test and rendered_frames >= 3:
                print(f"[{task}] smoke test rendered {rendered_frames} frames")
                break
            if (
                terminal_started_at is not None
                and time.perf_counter() - terminal_started_at >= terminal_hold_s
            ):
                print(
                    f"[{task}] closing after {terminal_hold_s:g}-second "
                    "terminal-result display"
                )
                break
    finally:
        try:
            if not terminal_fill_detail:
                terminal_fill_detail = _fill_level_detail(env, task)
            if terminal_result is False:
                detail = terminal_failure_reason or terminal_fill_detail or None
            else:
                detail = terminal_fill_detail or None
            # Prefer the latched terminal verdict — do not re-score after the
            # 2s hold (player can keep pouring and flip check_success / PS).
            if terminal_partial_score is not None:
                try:
                    env._latched_partial_score = float(terminal_partial_score)
                except (TypeError, ValueError):
                    pass
            if escape_quit:
                report_ok = False
            elif terminal_result is not None:
                report_ok = bool(terminal_result)
            else:
                report_ok = None
            report_task_result(env, detail=detail, ok=report_ok)
        finally:
            try:
                viewer.close()
            except Exception:
                pass
            env.close_env()
    return terminal_result


def make_parser(task, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=PROFILES[task][2], help="task_config name without .yml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control", choices=("keyboard", "keyboard+mouse", "robot"), default="robot")
    p.add_argument("--robot-motion", choices=("planner", "interpolate"), default="interpolate")
    p.add_argument("--task-arg", action="append", default=[], help="override task_args entry (key=value)")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="initialize, render three frames, report state, and exit",
    )
    add_record_data_arg(p)
    return p
