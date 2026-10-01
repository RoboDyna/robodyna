#!/usr/bin/env python3
"""Interactive viewer for ``dispense_gummy``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_dispense_gummy.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_dispense_gummy.py --control robot

Keyboard mode: Space dispenses; arrows or hold-mouse on keycaps (no latch). Robot mode: select an arm,
move over a key, lower with Q to press (left → red dispense; right → belt keys). Sandbox only — not data collection.
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
    UniversalRobotControls,
    actor_scene_id,
    click_hits_actor_map,
    escape_quit_requested,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_mode_controls,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Space             hold to press the red dispense key
  Left / Right      move bowl left / right (belt keys)
  Mouse             hold click on a keycap to press it (releases when you let go)
"""

CONTROLS_ROBOT = """
  Left arm — hover over the red dispense key, lower with Q.
  Right arm — hover over a left/right belt key, lower with Q.
  Continuous belt (Opt 2): hold the gripper down on a belt key to slide.
  Discrete belt (default): press edge hops one station.
"""


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
        task_name="dispense_gummy",
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


def _belt_side(window):
    left = window.key_down("left")
    right = window.key_down("right")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _mouse_picture_xy(viewer):
    """Map window mouse position into Segmentation picture coordinates."""
    window = viewer.window
    mx, my = window.mouse_position
    ww, wh = window.size
    if ww <= 0 or wh <= 0 or mx < 0 or my < 0 or mx >= ww or my >= wh:
        return None
    tw, th = window.get_picture_size("Segmentation")
    return int(mx * tw / ww), int(my * th / wh)


def _remaining_in_tubes(env):
    left = list(env._tube_stack_colors["left"][env._dispensed_count["left"]:])
    right = list(env._tube_stack_colors["right"][env._dispensed_count["right"]:])
    return left, right


def _unrecoverable(env):
    """Any target miss or distractor catch ends the episode as a failure."""
    target_missed = env.yellow_missed if env.target_color == "yellow" else env.blue_dropped
    distractor_caught = env.blue_caught if env.target_color == "yellow" else env.yellow_caught
    return int(target_missed) > 0 or int(distractor_caught) > 0


def _episode_done(env):
    """Return ``(done, detail)`` for a definitive success/failure state."""
    if bool(getattr(env, "invalid_pattern", False)):
        return True, "invalid_pattern"
    if _unrecoverable(env):
        target_missed = env.yellow_missed if env.target_color == "yellow" else env.blue_dropped
        distractor_caught = env.blue_caught if env.target_color == "yellow" else env.yellow_caught
        return True, f"unrecoverable (target_missed={target_missed}, distractor_caught={distractor_caught})"
    left, right = _remaining_in_tubes(env)
    if not left and not right and not getattr(env, "_active_drops", None):
        return True, "tubes empty"
    return False, None


# Max TCP→key XY distance (m) to count as "over" a key (button half is ~2 cm).
_KEY_XY_TOL = 0.055


def _tcp_xy(env, side: str) -> np.ndarray:
    getter = env.robot.get_left_tcp_pose if side == "left" else env.robot.get_right_tcp_pose
    return np.asarray(getter()[:2], dtype=np.float64)


def _key_xy(env, name: str) -> np.ndarray:
    if name == "dispense":
        return np.asarray([env.key_x, env.key_y], dtype=np.float64)
    return np.asarray(env.belt_key_xy[name], dtype=np.float64)


def _key_top_z(env, name: str) -> float:
    if name == "dispense":
        return float(env.dispense_key_top_z)
    return float(env.belt_key_top_z)


def _arm_for_key(name: str) -> str:
    return "left" if name == "dispense" else "right"


def _nearest_key_for_arm(env, side: str, max_dist: float = _KEY_XY_TOL):
    """Name of the nearest key under ``side``'s TCP, or None if too far."""
    tcp = _tcp_xy(env, side)
    candidates = ("dispense",) if side == "left" else ("left", "right")
    best_name, best_d = None, float(max_dist)
    for name in candidates:
        d = float(np.linalg.norm(_key_xy(env, name) - tcp))
        if d < best_d:
            best_d, best_name = d, name
    return best_name


class KeyboardState:
    """Space dispenses; arrows or hold-click on keycaps (no latch/toggle)."""

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self._key_ids = {}
        self._last_mouse_hit = None
        self._dispense_was_held = False
        dispense = getattr(env, "dispense_key", None)
        sid = actor_scene_id(dispense)
        if sid is not None:
            self._key_ids[int(sid)] = "dispense"
        for side, key in (getattr(env, "belt_keys", {}) or {}).items():
            sid = actor_scene_id(key)
            if sid is not None:
                self._key_ids[int(sid)] = str(side)

    def _mouse_held_hit(self):
        window = self.viewer.window
        if not bool(window.mouse_down(0)):
            return None
        pix = _mouse_picture_xy(self.viewer)
        if pix is None:
            return None
        return click_hits_actor_map(self.viewer, pix[0], pix[1], self._key_ids)

    def update(self, env, window):
        hit = self._mouse_held_hit()
        if hit != self._last_mouse_hit:
            if hit is not None:
                print(f"Key pressed: {hit}")
            elif self._last_mouse_hit is not None:
                print(f"Key released: {self._last_mouse_hit}")
            self._last_mouse_hit = hit

        side = _belt_side(window)
        if side is None and hit in ("left", "right"):
            side = hit
        env._expert_belt_hold = side
        env._bowl_force_stop = False

        # Hold while Space or mouse is down so the spring can reach trigger depth.
        dispense_held = bool(window.key_down("space")) or hit == "dispense"
        env._expert_dispense = dispense_held
        if dispense_held and not self._dispense_was_held:
            print("Dispense.")
        self._dispense_was_held = dispense_held


class SmoothGummyPressController:
    """Non-blocking vertical key press from the current arm pose.

    Timed on the simulation clock (same approach as catch_marbles_trapdoors):
    the viewer advances a few ms of physics per frame, so wall-timed ramps
    finish before the fingertip arrives.
    """

    PRESS_SPEED = 0.70
    RAISE_SPEED = 1.00
    MIN_TRANSITION_SECONDS = 0.10
    MAX_TRANSITION_SECONDS = 0.90
    MIN_HOLD_SECONDS = 0.05
    MAX_HOLD_SECONDS = 0.25
    # Aim TCP slightly above the keycap so EE enters the task press band.
    TOUCH_DZ = 0.020
    MIN_DESCENT = 0.010
    MAX_DESCENT = 0.50
    MAX_PRESS_JOINT_TRAVEL = 1.80

    def __init__(self, env):
        self.env = env
        self.continuous = bool(getattr(env, "belt_continuous_motion", False))
        self.phase = "idle"
        self.side = None
        self.key = None
        self.start_qpos = None
        self.hover_qpos = None
        self.press_qpos = None
        self.started_at = None
        self.holding_from = None
        self.holding_until = None
        self.transition_seconds = self.MIN_TRANSITION_SECONDS
        self.descent = 0.0
        self._clock = 0.0
        self._hold_while_space = False
        self._space_held = False

    @property
    def busy(self):
        return self.phase != "idle"

    def _drive_qpos(self, side):
        joints = (
            self.env.robot.left_arm_joints
            if side == "left"
            else self.env.robot.right_arm_joints
        )
        return np.asarray(
            [joint.get_drive_target()[0] for joint in joints],
            dtype=np.float64,
        )

    def _tcp_z(self):
        getter = (
            self.env.robot.get_left_tcp_pose
            if self.side == "left"
            else self.env.robot.get_right_tcp_pose
        )
        return float(getter()[2])

    def _ik_joints(self, ee_pose7):
        solver = arm_ik(self.env, self.side)
        if solver is None:
            return None
        solution = solver.solve(ee_pose7)
        return None if solution is None else solution[0]

    def _plan_press_target(self):
        get_ee = (
            self.env.robot.get_left_ee_pose
            if self.side == "left"
            else self.env.robot.get_right_ee_pose
        )
        pose = np.asarray(get_ee(), dtype=np.float64).copy()
        desired_tcp_z = _key_top_z(self.env, self.key) + self.TOUCH_DZ
        descent = float(
            np.clip(
                self._tcp_z() - desired_tcp_z,
                self.MIN_DESCENT,
                self.MAX_DESCENT,
            )
        )
        pose[2] -= descent
        q = self._ik_joints(pose)
        if q is None:
            return None, 0.0
        start = self.hover_qpos
        target = np.asarray(q[: len(start)], dtype=np.float64)
        if float(np.max(np.abs(target - start))) > self.MAX_PRESS_JOINT_TRAVEL:
            return None, 0.0
        return target, descent

    def request(self, key: str, *, hold_while_space: bool = False):
        if self.busy:
            return False
        self.key = str(key)
        self.side = _arm_for_key(self.key)
        self.hover_qpos = self._drive_qpos(self.side)
        self.press_qpos, descent = self._plan_press_target()
        if self.press_qpos is None:
            print("Could not plan a smooth vertical key press.")
            self._reset()
            return False
        self.descent = descent
        self._hold_while_space = bool(hold_while_space)
        self._space_held = True
        self.env._interactive_teleop_locked = True
        self.env._expert_belt_hold = None
        self.env._expert_dispense = False
        self._begin_transition("pressing", self.press_qpos, self.PRESS_SPEED)
        return True

    def _begin_transition(self, phase, target, speed):
        self.phase = phase
        self.start_qpos = self._drive_qpos(self.side)
        self.target_qpos = np.asarray(target, dtype=np.float64)
        self.transition_seconds = float(np.clip(
            self.descent / speed,
            self.MIN_TRANSITION_SECONDS,
            self.MAX_TRANSITION_SECONDS,
        ))
        self.started_at = self._clock

    def _finish_transition(self, now):
        if self.phase == "pressing":
            self.phase = "holding"
            self.started_at = None
            self.holding_from = now
            if self._hold_while_space:
                self.holding_until = None
            else:
                self.holding_until = now + self.MAX_HOLD_SECONDS
        elif self.phase == "raising":
            self._reset()

    def set_space_held(self, held: bool):
        self._space_held = bool(held)

    def update(self):
        if self.phase == "idle":
            return
        self._clock += float(self.env.scene.get_timestep())
        now = self._clock
        if self.phase == "holding":
            self.env.robot.set_arm_joints(
                self.press_qpos,
                np.zeros_like(self.press_qpos),
                self.side,
            )
            if self._hold_while_space:
                if not self._space_held:
                    self._begin_transition("raising", self.hover_qpos, self.RAISE_SPEED)
                return
            settled = now >= self.holding_until
            if settled and now - self.holding_from >= self.MIN_HOLD_SECONDS:
                self._begin_transition("raising", self.hover_qpos, self.RAISE_SPEED)
            return

        progress = min(
            1.0,
            (now - self.started_at) / self.transition_seconds,
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        delta = self.target_qpos - self.start_qpos
        velocity = (
            delta / self.transition_seconds
            if progress < 1.0
            else np.zeros_like(delta)
        )
        self.env.robot.set_arm_joints(
            self.start_qpos + delta * smooth,
            velocity,
            self.side,
        )
        if progress >= 1.0:
            self._finish_transition(now)

    def _reset(self):
        self.env._interactive_teleop_locked = False
        self.phase = "idle"
        self.side = None
        self.key = None
        self.start_qpos = None
        self.hover_qpos = None
        self.press_qpos = None
        self.started_at = None
        self.holding_from = None
        self.holding_until = None
        self._hold_while_space = False
        self._space_held = False

    def release(self):
        if self.busy and self.hover_qpos is not None and self.side is not None:
            self.env.robot.set_arm_joints(
                self.hover_qpos,
                np.zeros_like(self.hover_qpos),
                self.side,
            )
        self.env._expert_belt_hold = None
        self.env._expert_dispense = False
        self._reset()


def main():
    parser = argparse.ArgumentParser(description="Interactive dispense_gummy viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.dispense_gummy import dispense_gummy
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("dispense_gummy", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    use_robot = args.control == "robot"
    env = dispense_gummy()
    env._interactive_robot_mode = use_robot
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=use_robot))
    if use_robot:
        env.together_close_gripper(save_freq=None)
    print_episode_condition(env)
    env._expert_belt_hold = None
    env._expert_dispense = False
    env._bowl_force_stop = False

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    keyboard = (
        KeyboardState(env, viewer)
        if args.control in ("keyboard", "keyboard+mouse")
        else None
    )

    views = make_viewer_view_toggle(env, viewer)
    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)

    mode = "continuous" if getattr(env, "belt_continuous_motion", False) else "discrete"
    print(f"Belt mode: {mode}.")
    if args.control in ("keyboard", "keyboard+mouse"):
        print_instructions(
            "Hold Space or the red keycap to dispense; hold Left/Right or belt "
            "keycaps to move the bowl (releases when you let go)."
        )

    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if keyboard is not None:
                keyboard.update(env, viewer.window)

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
            done, detail = _episode_done(env)
            if done:
                report_task_result(env, detail)
                terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
