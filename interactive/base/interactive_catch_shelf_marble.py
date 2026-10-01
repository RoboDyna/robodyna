#!/usr/bin/env python3
"""Interactive viewer for ``catch_shelf_marble``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_catch_shelf_marble.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_catch_shelf_marble.py --control robot

Keyboard mode latches bowl keys directly via arrows. Robot mode: select an arm,
move over the bowl key, lower with Q to press (gripper-Z / ReactivePushButtons).

Default / Opt 2: marble stays parked until a bowl key press edge.
Opt 1 / Opt 1+2 (reactive_marble): marble starts descending at episode start.
Sandbox only — not data collection.
"""

import argparse
import os
import sys
import time
from pathlib import Path

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
  Left Arrow        hold to move bowl left
  Right Arrow       hold to move bowl right
  Mouse             hold click on a keycap to press it (releases when you let go)
"""

CONTROLS_ROBOT = """
  Select an arm, move over a key, lower with Q to press (E to raise).
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
        task_name="catch_shelf_marble",
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


def _requested_side(window):
    # Arrows only — letter-key aliases also orbit the SAPIEN viewer camera.
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


class KeyboardBowlController:
    """Arrows or hold-click on a keycap drive bowl motion (no latch/toggle)."""

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self._key_ids = {}
        self._last_mouse_side = None
        for side, key in (getattr(env, "keys", {}) or {}).items():
            sid = actor_scene_id(key)
            if sid is not None:
                self._key_ids[int(sid)] = str(side)

    def _mouse_held_side(self):
        window = self.viewer.window
        if not bool(window.mouse_down(0)):
            return None
        pix = _mouse_picture_xy(self.viewer)
        if pix is None:
            return None
        return click_hits_actor_map(self.viewer, pix[0], pix[1], self._key_ids)

    def update(self, window):
        side = _requested_side(window)
        if side is None:
            side = self._mouse_held_side()
        if side != self._last_mouse_side:
            if side is not None:
                print(f"Bowl key pressed: {side}")
            elif self._last_mouse_side is not None:
                print(f"Bowl key released: {self._last_mouse_side}")
            self._last_mouse_side = side
        self.env._expert_hold = side
        self.env._bowl_force_stop = False
        if side is not None and getattr(self.env, "_marble_state", None) == "parked":
            self.env._release_marble()


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_shelf_marble viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_shelf_marble import catch_shelf_marble
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("catch_shelf_marble", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    use_robot = args.control == "robot"
    env = catch_shelf_marble()
    env._interactive_robot_mode = use_robot
    # Raster viewer: pour_beer-style plain-alpha shelves (transmission is invisible here).
    env._plain_glass = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=use_robot))
    if use_robot:
        env.together_close_gripper(save_freq=None)
    print_episode_condition(env)
    env._osc_armed = True
    env._bowl_force_stop = False
    env._expert_hold = None
    # Opt 1 (reactive_marble): match play_once — start descent immediately. Key-press
    # release is disabled in that mode, so without this the marble stays parked forever.
    if bool(getattr(env, "reactive_marble", False)):
        env._release_marble()

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    keyboard = None
    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)
        print_instructions("Select an arm, hover a key, lower with Q to press.")
    else:
        keyboard = KeyboardBowlController(env, viewer)
        print_instructions(
            "Hold Left/Right arrows, or hold mouse on a keycap (releases when you let go)."
        )

    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if keyboard is not None:
                keyboard.update(viewer.window)

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
            if getattr(env, "_marble_state", None) == "landed":
                if getattr(env, "_marble_result", None) is None:
                    env._resolve_marble()
                report_task_result(env, env._marble_result)
                terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
