#!/usr/bin/env python3
"""Interactive viewer for ``stop_valley_ball``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_stop_valley_ball.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_stop_valley_ball.py --control robot

Close the gripper (Space) on the bat handle to latch it, then teleop to the intercept.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    print_instructions,
    action_failed,
    add_record_data_arg,
    escape_quit_requested,
    gripper_width,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    resolve_action_arm,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Mouse click       move the bat to that XY (height stays at a stoppable intercept height)
"""

CONTROLS_ROBOT = """
  Space             open / close selected gripper only (close on bat handle to latch; 1/2 pick arm)
  Arrow keys        move selected arm in XY
  E / Q             move selected arm in Z
"""

# EE/TCP must be roughly this close to the handle to latch without contact.
# Wrist EE sits above the fingers, so this is looser than a finger-pad check.
_LATCH_PROXIMITY_M = 0.11


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_task(config_name: str, seed: int, use_robot: bool = False):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.update(
        task_name="stop_valley_ball",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as handle:
        embodiments = yaml.safe_load(handle)
    embodiment_names = config.get("embodiment", ["aloha-agilex"])
    if len(embodiment_names) == 1:
        left_name = right_name = embodiment_names[0]
        config["dual_arm_embodied"] = True
    elif len(embodiment_names) == 3:
        left_name, right_name, config["embodiment_dis"] = embodiment_names
        config["dual_arm_embodied"] = False
    else:
        raise SystemExit("Expected one embodiment or [left_embodiment, right_embodiment, separation].")

    config["left_robot_file"] = embodiments[left_name]["file_path"]
    config["right_robot_file"] = embodiments[right_name]["file_path"]
    config["left_embodiment_config"] = _embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = _embodiment_config(config["right_robot_file"])
    return config


def _intercept_xyz(env):
    if getattr(env, "intercept", None) is None:
        env._compute_intercept()
    return np.asarray(env.intercept, dtype=float)


def _get_rigid(actor):
    import sapien
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _bat_face_quat(env):
    if hasattr(env, "_bat_face_quat"):
        return list(env._bat_face_quat())
    return [1.0, 0.0, 0.0, 0.0]


def _clamp_bat_xyz(env, x, y, z):
    """Keep the bat mid-air past the red line and within a reachable band."""
    if hasattr(env, "_bat_min_center_z"):
        min_z = float(env._bat_min_center_z())
    else:
        min_clear = float(getattr(env, "INTERCEPT_MIN_CLEARANCE_DEFAULT", 0.06))
        min_z = float(env.table_top + min_clear + 0.5 * float(env.panel_radius))
    max_z = float(env.table_top + 0.28)
    z = float(np.clip(z, min_z, max_z))
    y = float(np.clip(y, -0.45, 0.30))
    # Stay on the exit side of the red line (same spirit as catch_valley place).
    side = float(getattr(env, "side", 1.0))
    red = float(env.red_line_x)
    if side > 0.0:
        x = float(np.clip(x, red + 0.02, 0.38))
    else:
        x = float(np.clip(x, -0.38, red - 0.02))
    return x, y, z


def _set_bat_xyz(env, x, y, z=None):
    import sapien
    pose = env.panel.get_pose()
    if z is None:
        z = float(pose.p[2])
    x, y, z = _clamp_bat_xyz(env, x, y, z)
    new_pose = sapien.Pose([float(x), float(y), float(z)], _bat_face_quat(env))
    try:
        env.panel.set_pose(new_pose)
    except Exception:
        env.panel.actor.set_pose(new_pose)
    # Keep aliases in sync (weld helpers use self.bowl).
    env.bowl = env.panel
    rigid = _get_rigid(env.panel)
    if rigid is not None:
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass
    return x, y, z


def _nudge_from_keys(window, xy_step=0.045, z_step=0.030):
    """Map arrows/E/Q to world-frame held-bat movement."""

    dx = xy_step * (window.key_down("right") - window.key_down("left"))
    dy = xy_step * (window.key_down("up") - window.key_down("down"))
    dz = z_step * (window.key_down("e") - window.key_down("q"))
    return float(dx), float(dy), float(dz)


def _bat_handle_segment(env):
    """World endpoints of the graspable handle (head rim → tip along −Y)."""
    head = np.asarray(env.panel.get_pose().p, dtype=float)
    head_r = float(getattr(env, "panel_radius", env.PANEL_RADIUS_DEFAULT))
    h_len = float(getattr(env, "handle_half_len", env.HANDLE_HALF_LEN_DEFAULT))
    # Handle center is at −(head_r + h_len) in Y; tip is a further h_len along −Y.
    start = head + np.array([0.0, -head_r, -0.012], dtype=float)
    end = head + np.array([0.0, -(head_r + 2.0 * h_len), -0.012], dtype=float)
    return start, end


def _point_segment_distance(point, start, end) -> float:
    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    span = end - start
    denom = float(np.dot(span, span))
    if denom < 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, span) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * span)))


def _gripper_probe_points(env, arm) -> list:
    """World points to test nearness (TCP preferred, then EE). Works for left or right."""
    side = str(arm)
    points = []
    robot = getattr(env, "robot", None)
    if robot is not None:
        tcp_fn = getattr(robot, f"get_{side}_tcp_pose", None)
        if callable(tcp_fn):
            try:
                points.append(np.asarray(tcp_fn()[:3], dtype=float))
            except Exception:
                pass
        ee_fn = getattr(robot, f"get_{side}_ee_pose", None)
        if callable(ee_fn):
            try:
                points.append(np.asarray(ee_fn()[:3], dtype=float))
            except Exception:
                pass
    if not points:
        try:
            points.append(np.asarray(env._end_effector_position(arm), dtype=float))
        except Exception:
            pass
    return points


def _selected_arm_gripper_link_names(env, arm) -> set[str]:
    """Link names on the selected arm that count as gripper contact."""
    side = str(arm)
    names = set()
    robot = getattr(env, "robot", None)
    if robot is None:
        return names
    art = getattr(robot, f"{side}_entity", None)
    if art is None:
        return names
    for link in art.get_links():
        name = link.get_name()
        if name in (
            "finger_left",
            "finger_right",
            "gripper_left",
            "gripper_right",
            "wsg_50_base_link",
            "base_link_gripper_left",
            "base_link_gripper_right",
        ) or "finger" in name or "gripper" in name:
            names.add(name)
    return names


def _gripper_can_latch_bat(env, arm) -> bool:
    """True when the selected arm's fingers contact the bat or sit on the handle.

    Applies to left and right — uses the currently selected arm only.
    """
    bat_name = getattr(env, "BAT_NAME", "pingpong_bat")
    arm_links = _selected_arm_gripper_link_names(env, arm)
    try:
        for contact in env.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if bat_name not in (name0, name1):
                continue
            other = name1 if name0 == bat_name else name0
            if other in arm_links or other in getattr(env.robot, "gripper_name", []):
                return True
    except Exception:
        pass

    start, end = _bat_handle_segment(env)
    for point in _gripper_probe_points(env, arm):
        if _point_segment_distance(point, start, end) <= _LATCH_PROXIMITY_M:
            return True
    return False


def _keyboard_bat_z(env) -> float:
    """Fixed bat height that can still intercept the ball."""
    try:
        _ix, _iy, iz = _intercept_xyz(env)
        return float(iz)
    except Exception:
        pass
    if hasattr(env, "_bat_min_center_z"):
        return float(env._bat_min_center_z()) + 0.02
    return float(env.table_top + 0.10)


class KeyboardBatController:
    """Mouse-click bat teleport at a fixed intercept-ready height."""

    def __init__(self, env):
        self.env = env
        env._bowl_ready = True
        self._z = _keyboard_bat_z(env)
        # Park at the current XY with the standard stoppable height.
        p = np.asarray(env.panel.get_pose().p, dtype=float)
        _set_bat_xyz(env, float(p[0]), float(p[1]), self._z)

    def update(self, _window):
        return

    def on_click(self, viewer, pixel_x, pixel_y):
        from _interactive_common import table_xy_from_click

        hit = table_xy_from_click(viewer, pixel_x, pixel_y, float(self.env.table_top))
        if hit is None:
            return False
        x, y, z = _set_bat_xyz(self.env, hit[0], hit[1], self._z)
        print(f"Bat → ({x:.3f}, {y:.3f}, {z:.3f}).")
        return True


class RobotBatController:
    """Latch the bat when the gripper closes on/near the handle (Space = gripper only)."""

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.busy = False
        self._prev_width = {"left": 1.0, "right": 1.0}

    def _latch(self):
        self.busy = True
        self.arm = resolve_action_arm(self.env, self.ArmTag, exactly_one=True)
        if self.arm is None:
            self.busy = False
            return
        side = str(self.arm)
        # Let the shared Space close settle contacts before judging proximity.
        try:
            self.env.robot.set_gripper(0.0, side, gripper_eps=0.0)
        except Exception:
            pass
        self.env._dwell(12)
        if not _gripper_can_latch_bat(self.env, self.arm):
            action_failed(self.env, (side,), detail="close on the bat handle to latch")
            self.busy = False
            return
        self.env.bowl = self.env.panel
        self.env._weld_bowl_to_end_effector(self.arm)
        self.holding = True
        self.env._bowl_ready = True
        print(f"Latched bat to {side} gripper. Teleop to the intercept.")
        self.busy = False

    def update(self, window):
        if self.busy or self.holding:
            return
        selected = tuple(getattr(self.env, "_interactive_selected_arms", ()) or ())
        arms = list(selected) if selected else []
        closing = False
        for side in ("left", "right"):
            width = gripper_width(self.env, side)
            prev = self._prev_width.get(side, 1.0)
            if side in arms and prev > 0.5 and width <= 0.5:
                closing = True
            self._prev_width[side] = width
        if closing:
            self._latch()


def main():
    parser = argparse.ArgumentParser(description="Interactive stop_valley_ball viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "keyboard+mouse", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="interpolate",
        help="Retained for compatibility; arm teleop uses UniversalRobotControls.",
    )
    parser.add_argument(
        "--task-arg",
        action="append",
        default=[],
        help="Override task_args.stop_valley_ball entry, e.g. --task-arg wall_bounce_enabled=true",
    )
    add_record_data_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.stop_valley_ball import stop_valley_ball
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("stop_valley_ball", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    config = _configure_task(args.config, args.seed, use_robot=args.control == "robot")
    # Optional option toggles (same as record_demo --task-arg).
    targs = config.setdefault("task_args", {}).setdefault("stop_valley_ball", {})
    for item in args.task_arg:
        if "=" not in item:
            raise SystemExit(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            targs[key] = raw.lower() == "true"
        else:
            try:
                targs[key] = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                targs[key] = raw

    env = stop_valley_ball()
    env.setup_demo(**config)
    print_episode_condition(env)
    try:
        env.together_open_gripper(save_freq=None)
    except Exception:
        pass
    # setup_demo already starts ball motion with expert_demo=False.
    ix, iy, iz = _intercept_xyz(env)
    print(
        f"Predicted intercept ≈ ({ix:.3f}, {iy:.3f}, {iz:.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}."
    )

    controller = (
        RobotBatController(env, ArmTag) if args.control == "robot" else KeyboardBatController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if args.control == "robot":
        print_instructions(
            "Teleop to the bat handle, close with Space to latch, then move to the intercept."
        )
    else:
        viewer.register_click_handler(controller.on_click)
        print_instructions(
            f"Click to move the bat (height fixed at z≈{float(controller._z):.3f})."
        )

    settle_after = None
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)
    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            controller.update(viewer.window)

            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if escape_quit_requested(env, viewer.window):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            for _ in range(n_steps):
                env._update_kinematic_tasks()
                env.scene.step()
            env.scene.update_render()
            viewer.render()

            if escape_quit_requested(env, viewer.window):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if getattr(env, "_ball_phase", None) == "released":
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball left the valley exit; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    terminal_started_at = time.perf_counter()
                    continue
            # Early fail if the ball hit the table before a head contact.
            if getattr(env, "_ball_table_before_hit", False) and not getattr(env, "_panel_hit", False):
                report_task_result(env)
                terminal_started_at = time.perf_counter()
                continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
