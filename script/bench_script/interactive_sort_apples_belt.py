"""Interactive, top-down default-mode viewer for ``sort_apples_belt``.

Run from the repository root:

    conda run -n robodyna python script/bench_script/interactive_sort_apples_belt.py --control keyboard
    conda run -n robodyna python script/bench_script/interactive_sort_apples_belt.py --control mouse

Keyboard mode: hold Left Arrow or Right Arrow to aim the diverter toward the
corresponding basket; release to reset. Mouse mode: click a red or green button
to toggle its routing direction. This is an interaction sandbox, not a
data-collection or robot-control rollout.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from envs import CONFIGS_PATH
from envs.sort_apples_belt import sort_apples_belt
from envs.utils.action import ArmTag
from _interactive_common import RealtimePhysicsPacer  # noqa: E402


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_default_task(config_name: str, seed: int):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Keep the sandbox on the baseline task regardless of config overrides.
    task_args = config.setdefault("task_args", {}).setdefault("sort_apples_belt", {})
    task_args["color_mode"] = "alternating"
    task_args["rotten_prob"] = 0.0

    config.update(
        task_name="sort_apples_belt",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=False,
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


def _update_keyboard_control(env, window):
    """Map sustained Left/Right Arrow input to the task's existing hold control."""

    left_down = window.key_down("left")
    right_down = window.key_down("right")
    # Default mode has no dump action, so pressing both arrows simply releases the gate.
    env._expert_hold = "left" if left_down and not right_down else (
        "right" if right_down and not left_down else None
    )


def _button_click_handler(env):
    """Return a segmentation-based handler that toggles a clicked task button."""

    button_ids = {button.actor.per_scene_id: side for side, button in env.buttons.items()}

    def handle_click(viewer, pixel_x, pixel_y):
        pixel = viewer.window.get_picture_pixel("Segmentation", pixel_x, pixel_y)
        side = button_ids.get(int(pixel[1]))
        if side is None:
            return False

        env._expert_hold = side if getattr(env, "_expert_hold", None) != side else None
        color = "red" if env._side_color[side] == env.COLOR_RED else "green"
        state = f"diverting to {color}" if env._expert_hold else "released (plank at rest)"
        print(f"Button click: {color}; {state}")
        return True

    return handle_click


def _move_grippers_to_ready_position(env):
    """Park both closed grippers above their buttons without pressing either."""

    env._expert_hold = None
    env.plan_success = True
    env.move(
        env.grasp_actor(
            env.buttons["left"], arm_tag=ArmTag("left"), pre_grasp_dis=0.09,
            grasp_dis=0.09, contact_point_id=0, gripper_pos=0.0,
        ),
        env.grasp_actor(
            env.buttons["right"], arm_tag=ArmTag("right"), pre_grasp_dis=0.09,
            grasp_dis=0.09, contact_point_id=0, gripper_pos=0.0,
        ),
    )
    if not env.plan_success:
        detail = getattr(env, "_last_plan_fail", None)
        print(f"Could not reach the two-gripper ready position: {detail or 'unknown reason'}")
        env.plan_success = True
        env._last_plan_fail = None
    env._expert_hold = None


def main():
    parser = argparse.ArgumentParser(description="Interactive top-down sort-apples viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "mouse"),
        default="keyboard",
        help="Interaction method (default: keyboard)",
    )
    args = parser.parse_args()

    env = sort_apples_belt()
    env.setup_demo(**_configure_default_task(args.config, args.seed))
    env.together_close_gripper(save_freq=None)
    _move_grippers_to_ready_position(env)
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")

    # A fixed overhead camera makes the belt and the selected routing direction clear.
    viewer.set_camera_xyz(0.0, 0.0, 2.1)
    viewer.set_camera_rpy(0.0, -np.pi / 2.0, 0.0)
    if args.control == "mouse":
        viewer.register_click_handler(_button_click_handler(env))
        print("Top-down sort-apples sandbox ready. Click red/green to toggle the plank direction.")
    else:
        print("Top-down sort-apples sandbox ready. Hold Left/Right Arrow to rotate the plank; release to reset.")

    pacer = RealtimePhysicsPacer(env)
    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            if args.control == "keyboard":
                _update_keyboard_control(env, viewer.window)

            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    break
                continue

            for _ in range(n_steps):
                env._update_kinematic_tasks()
                env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
