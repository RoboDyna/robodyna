#!/usr/bin/env python3
"""Interactive viewer for ``catch_ramp_ball``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_catch_ramp_ball.py --control robot
    /path/to/RoboDynaExp/interactive/base/interactive_catch_ramp_ball.py --control keyboard

Robot: teleop and Space-grasp the cup into the catch aim.
Keyboard: click once on empty table space to teleport the cup there (one shot).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
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
    make_viewer_view_toggle,
    print_instructions,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    table_xy_from_click,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Mouse click       teleport the cup to that table XY (once only; further clicks ignored)
"""

CONTROLS_ROBOT = """
  Space             open / close selected gripper(s) to pick / place the cup

  Manually grasp the cup and place it under the predicted catch aim.
  There is no keyboard teleport; Space opens/closes the gripper.
"""


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_task(config_name: str, seed: int, use_robot: bool = True):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.update(
        task_name="catch_ramp_ball",
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


def _aim_xy(env):
    landing, _ = env._predict_landing()
    aim = np.asarray(getattr(env, "catch_aim", landing), dtype=float)
    return float(aim[0]), float(aim[1])


def _set_cup_xy(env, x: float, y: float) -> None:
    """Teleport cup in XY only; keep current Z so it stays on the table."""
    cur = env.cup.get_pose()
    z = float(cur.p[2])
    pose = sapien.Pose([float(x), float(y), z], list(cur.q))
    try:
        env.cup.set_pose(pose)
    except Exception:
        env.cup.actor.set_pose(pose)
    rigid = env._cup_comp() if hasattr(env, "_cup_comp") else None
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(pose)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_ramp_ball viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_ramp_ball import catch_ramp_ball
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls(
        "catch_ramp_ball",
        args.control,
        keyboard=CONTROLS_KEYBOARD,
        robot=CONTROLS_ROBOT,
    )

    use_robot = args.control == "robot"
    env = catch_ramp_ball()
    env._interactive_robot_mode = use_robot
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=use_robot))
    print_episode_condition(env)

    x, y = _aim_xy(env)
    if use_robot:
        try:
            env.together_open_gripper(save_freq=None)
        except Exception:
            pass

    env._start_ball_motion(expert_demo=False)
    print(
        f"Predicted catch aim ≈ ({x:.3f}, {y:.3f}). Ball is rolling."
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    cup_placed = {"done": False}

    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)
        print_instructions("Arrows/E/Q move the arm; Space opens/closes the gripper to pick the cup.")
    else:
        def _on_click(viewer, pixel_x, pixel_y):
            if cup_placed["done"]:
                return False
            hit = table_xy_from_click(viewer, pixel_x, pixel_y, float(env.table_top))
            if hit is None:
                return False
            _set_cup_xy(env, hit[0], hit[1])
            cup_placed["done"] = True
            print(f"Cup teleported to ({hit[0]:.3f}, {hit[1]:.3f}); mouse disabled.")
            return True

        viewer.register_click_handler(_on_click)
        print_instructions("Click once on the table to place the cup; further clicks are ignored.")

    settle_after = None
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)

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
                    print("Ball released from ramp lip; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
