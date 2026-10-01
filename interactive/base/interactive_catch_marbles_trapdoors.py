#!/usr/bin/env python3
"""Interactive viewer for ``catch_marbles_trapdoors``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_catch_marbles_trapdoors.py --control robot
    /path/to/RoboDynaExp/interactive/base/interactive_catch_marbles_trapdoors.py --control keyboard

Robot: select an arm, move over the matching colored key, then lower with Q to press.
Keyboard: press 1–4 or click a keycap to open the matching trapdoor.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    RealtimePhysicsPacer,
    UniversalRobotControls,
    actor_scene_id,
    add_robot_motion_arg,
    click_hits_actor_map,
    configure_task,
    edge_pressed,
    escape_quit_requested,
    make_viewer_view_toggle,
    print_instructions,
    print_mode_controls,
    report_task_result,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  1 / 2 / 3 / 4     open trapdoor for key L→R (same as clicking that keycap)
  Mouse click       click a colored keycap to open its trapdoor
"""

CONTROLS_ROBOT = """
  Select left (1) or right (2) arm, move over the matching colored key, then lower with Q to press (E to raise).
  Left arm covers left-half keys; right arm covers right-half keys.
  Door opens when the keycap is pushed down past its trigger depth.
  Release fully and press again to reopen a door.
"""


def _print_color_map(env):
    names = list(getattr(env, "button_color_names", []) or [])
    target = int(getattr(env, "target_button_idx", -1))
    mapping = ", ".join(f"{i + 1}:{c}" for i, c in enumerate(names))
    target_name = names[target] if 0 <= target < len(names) else "?"
    left_keys = [f"{i + 1}:{c}" for i, c in enumerate(names) if str(env._arm_for_door(i)) == "left"]
    right_keys = [f"{i + 1}:{c}" for i, c in enumerate(names) if str(env._arm_for_door(i)) == "right"]
    print(f"Buttons L→R: {mapping}")
    print(f"Left-arm keys: {', '.join(left_keys) or '(none)'} | Right-arm keys: {', '.join(right_keys) or '(none)'}")
    print(f"Target marble color: {target_name} (index {target})")


def _trigger_door(env, idx: int, *, source: str) -> None:
    if idx < 0 or idx >= int(getattr(env, "n_buttons", 0)):
        return
    bank = getattr(env, "_reactive_buttons", None)
    if bank is not None:
        try:
            bank.set_forced(int(idx), True)
            bank.update()
            bank.set_forced(int(idx), False)
        except Exception:
            pass
    opened = bool(env._open_door_direct(int(idx)))
    names = list(getattr(env, "button_color_names", []) or [])
    color = names[idx] if 0 <= idx < len(names) else "?"
    if opened:
        print(f"Opened door {idx + 1} ({color}) via {source}.")
    else:
        print(f"Door {idx + 1} ({color}) did not open ({source}; budget/lock).")


class KeyboardTrapdoorController:
    """Keys 1–4 and mouse clicks on keycaps open matching trapdoors."""

    def __init__(self, env):
        self.env = env
        self._prev = {}
        self._button_ids = {}
        for i, btn in enumerate(getattr(env, "buttons", []) or []):
            sid = actor_scene_id(btn)
            if sid is not None:
                self._button_ids[int(sid)] = int(i)

    def update(self, window):
        for digit, idx in (("1", 0), ("2", 1), ("3", 2), ("4", 3)):
            if edge_pressed(window, digit, self._prev):
                _trigger_door(self.env, idx, source=f"key {digit}")

    def on_click(self, viewer, pixel_x, pixel_y):
        idx = click_hits_actor_map(viewer, pixel_x, pixel_y, self._button_ids)
        if idx is None:
            return False
        _trigger_door(self.env, int(idx), source="mouse click")
        return True


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_marbles_trapdoors viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.catch_marbles_trapdoors import catch_marbles_trapdoors

    print_mode_controls(
        "catch_marbles_trapdoors",
        args.control,
        keyboard=CONTROLS_KEYBOARD,
        robot=CONTROLS_ROBOT,
    )

    use_robot = args.control == "robot"
    env = catch_marbles_trapdoors()
    env._interactive_robot_mode = use_robot
    env.setup_demo(**configure_task(
        "catch_marbles_trapdoors", args.config, args.seed, use_robot=use_robot,
    ))
    if use_robot:
        env.together_close_gripper(save_freq=None)
    print_episode_condition(env)
    _print_color_map(env)

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
        keyboard = KeyboardTrapdoorController(env)
        viewer.register_click_handler(keyboard.on_click)
        print_instructions("Press 1–4 or click a keycap to open its trapdoor.")

    left_track_since = None
    settle_s = 0.6
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
            mode = str(getattr(env, "_ball_mode", "track"))
            drop_pending = bool(
                getattr(env, "_distractor_drop_still_possible", lambda: False)()
            )
            if mode != "track" and not drop_pending:
                if left_track_since is None:
                    left_track_since = time.perf_counter()
                elif time.perf_counter() - left_track_since >= settle_s:
                    report_task_result(env, f"ball_mode={mode}")
                    terminal_started_at = time.perf_counter()
            elif bool(getattr(env, "doors_open_budget_exhausted", lambda: False)()):
                report_task_result(env, "door_open_budget_exhausted")
                terminal_started_at = time.perf_counter()
            else:
                left_track_since = None
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
