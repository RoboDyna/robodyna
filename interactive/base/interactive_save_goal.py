#!/usr/bin/env python3
"""Interactive viewer for ``save_goal``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_save_goal.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_save_goal.py --control robot

Keyboard+mouse: click the table to place the goalkeeper (once).
Robot: teleop grasp / place the keeper in the green zone before the red line.
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
    UniversalRobotControls,
    add_robot_motion_arg,
    escape_quit_requested,
    is_robot_control,
    make_viewer_view_toggle,
    prepare_interactive_control,
    print_instructions,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    table_xy_from_click,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Mouse click       place the goalkeeper at that table XY (once)

  Place it in the green zone before the ball crosses the red line.
"""

CONTROLS_ROBOT = """
  Grasp and place the keeper in the green zone before the red line.
  Space opens/closes the gripper.
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
        task_name="save_goal",
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


def _clip_keeper_xy(env, x, y):
    x_min = float(env.green_area_x_min + env.keeper_half_x)
    x_max = float(env.green_area_x_max - env.keeper_half_x)
    y_min = float(env.green_area_y_min + env.keeper_half_y)
    y_max = float(env.green_area_y_max - env.keeper_half_y)
    return float(np.clip(x, x_min, x_max)), float(np.clip(y, y_min, y_max))


def _set_keeper_xy(env, x, y, clip=True, kinematic=True):
    import sapien
    if clip:
        x, y = _clip_keeper_xy(env, x, y)
    z = float(env.table_top_z + env.keeper_half_z)
    q = list(env.goalkeeper.get_pose().q)
    pose = sapien.Pose([x, y, z], q)
    try:
        env.goalkeeper.set_pose(pose)
    except Exception:
        env.goalkeeper.actor.set_pose(pose)
    rigid = env._get_rigid(env.goalkeeper)
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            if kinematic:
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
            else:
                rigid.set_kinematic(False)
        except Exception:
            pass
    return x, y


def _park_keeper_hidden(env):
    """Stash the keeper off the field until the player clicks a place."""
    import sapien
    z = float(env.table_top_z + env.keeper_half_z + 0.35)
    pose = sapien.Pose([0.0, -0.55, z], [1, 0, 0, 0])
    try:
        env.goalkeeper.set_pose(pose)
    except Exception:
        env.goalkeeper.actor.set_pose(pose)
    rigid = env._get_rigid(env.goalkeeper)
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(pose)
        except Exception:
            pass
    if hasattr(env, "_set_collision_enabled"):
        try:
            env._set_collision_enabled(env.goalkeeper, False)
        except Exception:
            pass


def _place_keeper_at_click(env, x, y):
    x, y = _set_keeper_xy(env, x, y, clip=True, kinematic=True)
    if hasattr(env, "_seat_keeper_dynamic"):
        env._seat_keeper_dynamic()
    else:
        _set_keeper_xy(env, x, y, clip=False, kinematic=False)
        if hasattr(env, "_set_collision_enabled"):
            try:
                env._set_collision_enabled(env.goalkeeper, True)
            except Exception:
                pass
    env._keeper_deployed = True
    print(f"Goalkeeper placed at ({x:.3f}, {y:.3f}).")
    return x, y


def _start_shot(env):
    import sapien
    env._ball_step = 0
    env._ball_blocked = False
    env._ball_live = False
    env._block_was_legal = False
    env._goal_conceded = False
    env._late_failure = False
    env._ball_crossed_goal = False
    env._keeper_deployed = False
    env._keeper_drop_pose = None
    env._ball_motion_active = True
    if hasattr(env, "_set_collision_enabled"):
        env._set_collision_enabled(env.ball, False)
    start = getattr(env, "ball_start_pose", None)
    if start is not None and env.ball is not None:
        pose = sapien.Pose(np.asarray(start, dtype=float).tolist(), [1, 0, 0, 0])
        env.ball.set_pose(pose)
        rigid = getattr(env, "_ball_rigid", None)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass


def _save_ok_without_grippers(env) -> bool:
    """``check_success`` requires open grippers; keyboard mode has no arms."""
    return bool(
        env._keeper_in_zone()
        and getattr(env, "_block_was_legal", False)
        and env._ball_blocked
        and (not env._late_failure)
        and (not env._goal_conceded)
    )


class ClickKeeperController:
    """One click places the keeper; further clicks ignored."""

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self.deployed = False

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.deployed or getattr(self.env, "_keeper_deployed", False):
            return False
        hit = table_xy_from_click(
            viewer, pixel_x, pixel_y, float(self.env.table_top_z)
        )
        if hit is None:
            return False
        _place_keeper_at_click(self.env, hit[0], hit[1])
        self.deployed = True
        return True

    def update(self, _window):
        return


class RobotKeeperController:
    """Teleop only — Space opens/closes the gripper only."""

    def __init__(self, env):
        self.env = env

    def update(self, window):
        del window


def main():
    parser = argparse.ArgumentParser(description="Interactive save_goal viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="planner")
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.save_goal import save_goal
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("save_goal", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    use_robot = is_robot_control(args.control)
    env = save_goal()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=use_robot))
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    _start_shot(env)
    if not use_robot:
        _park_keeper_hidden(env)

    target = env.goalkeeper_target_pose.p if env.goalkeeper_target_pose is not None else [0, 0, 0]
    print(
        f"Shot started. Intercept target ≈ ({target[0]:.3f}, {target[1]:.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}."
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)
        controller = RobotKeeperController(env)
        print_instructions("Teleop the keeper into the green zone before the red line.")
    else:
        controller = ClickKeeperController(env, viewer)
        viewer.register_click_handler(controller.on_click)
        print_instructions("Click once on the table to place the goalkeeper.")

    done_since = None
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

            shot_done = (not getattr(env, "_ball_motion_active", False)) or env._goal_conceded
            if shot_done:
                if done_since is None:
                    done_since = time.perf_counter()
                elif time.perf_counter() - done_since >= 1.0:
                    detail = (
                        f"in_zone={env._keeper_in_zone()}, "
                        f"legal_block={getattr(env, '_block_was_legal', False)}, "
                        f"blocked={env._ball_blocked}, late={env._late_failure}, "
                        f"conceded={env._goal_conceded}"
                    )
                    if use_robot:
                        report_task_result(env, detail)
                    else:
                        report_task_result(env, detail, ok=_save_ok_without_grippers(env))
                    terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
