#!/usr/bin/env python3
"""Interactive viewer for ``whack_moles``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_whack_moles.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_whack_moles.py --control robot

Pick up a side-staged mallet by teleoping onto the handle and closing Space, then
jab moles mid-rise by lowering the mallet head. Avoid rabbits (Opt1).
Success = all moles hit and no rabbit touch.
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
    RealtimePhysicsPacer,
    UniversalRobotControls,
    action_failed,
    actor_scene_id,
    add_record_data_arg,
    add_robot_motion_arg,
    click_hits_actor_map,
    escape_quit_requested,
    gripper_width,
    is_robot_control,
    make_viewer_view_toggle,
    prepare_interactive_control,
    print_instructions,
    print_mode_controls,
    report_task_result,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Mouse click       click a mole while it is above the board to hit it
                    (clicking a rabbit fails; clicking a buried mole misses)
"""

CONTROLS_ROBOT = """
  1 / 2 / 3         select left, right, or both arms (selected gripper turns green)
  Space             open / close selected gripper only (close on mallet handle to latch)
  Arrows / E / Q    teleop; lower the mallet head onto rising moles to strike
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
        task_name="whack_moles",
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


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


def _arm_for_mole(env, idx, ArmTag):
    return env._arm_for_hole(env.mole_holes[idx])


def _set_cube_over_hole(env, arm_name, hole_xy, z=None):
    cube = env.hammer_cubes.get(arm_name)
    if cube is None:
        return
    if z is None:
        drop = float(env._mole_raise_drop()) if hasattr(env, "_mole_raise_drop") else 0.02
        z = float(
            env.board_top_z + env.mole_height - drop + float(env.cube_half) + 0.02
        )
    pose = sapien.Pose([float(hole_xy[0]), float(hole_xy[1]), float(z)], cube.get_pose().q)
    cube.actor.set_pose(pose)
    rigid = env._cube_comps.get(arm_name)
    if rigid is not None:
        try:
            rigid.set_kinematic_target(pose)
        except Exception:
            pass


class ClickMoleController:
    """Click moles above the surface to hit; rabbits fail (like catch_cuboid)."""

    def __init__(self, env):
        self.env = env
        self._mole_ids = {}
        self._rabbit_ids = {}
        for idx, mole in enumerate(getattr(env, "moles", []) or []):
            sid = actor_scene_id(mole)
            if sid is not None:
                self._mole_ids[int(sid)] = int(idx)
        for idx, rabbit in enumerate(getattr(env, "rabbits", []) or []):
            sid = actor_scene_id(rabbit)
            if sid is not None:
                self._rabbit_ids[int(sid)] = int(idx)

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.env.distractor_hit or getattr(self.env, "appearances_exhausted", False):
            return False
        rabbit = click_hits_actor_map(viewer, pixel_x, pixel_y, self._rabbit_ids)
        if rabbit is not None:
            self.env._mark_rabbit_touched(int(rabbit))
            print("Rabbit clicked — failure.")
            return True
        idx = click_hits_actor_map(viewer, pixel_x, pixel_y, self._mole_ids)
        if idx is None:
            return False
        idx = int(idx)
        if self.env.touched[idx]:
            return True
        if not self.env._mole_above_surface(idx):
            print(f"Miss — mole {idx} is not above the board.")
            return True
        self.env._mark_touched(idx)
        print(f"Hit mole {idx}.")
        return True

    def update(self, _window):
        return


class KeyboardMoleController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.selected = 0
        self._q = EdgeKey()
        self._e = EdgeKey()
        self._digit = {str(i): EdgeKey() for i in range(1, 10)}
        self._select_next(0)

    def _unhit(self):
        return [i for i in range(self.env.num_moles) if not self.env.touched[i]]

    def _select_next(self, delta):
        unhit = self._unhit()
        if not unhit:
            return
        if self.selected not in unhit:
            self.selected = unhit[0]
        else:
            k = unhit.index(self.selected)
            self.selected = unhit[(k + delta) % len(unhit)]
        arm = _arm_for_mole(self.env, self.selected, self.ArmTag)
        hole = self.env.holes[self.env.mole_holes[self.selected]]
        _set_cube_over_hole(self.env, str(arm), hole)
        print(f"Selected mole {self.selected} ({arm} arm).")

    def update(self, window):
        if self.env.distractor_hit or getattr(self.env, "appearances_exhausted", False):
            return
        if self._q.poll(window.key_down("q")):
            self._select_next(-1)
        if self._e.poll(window.key_down("e")):
            self._select_next(+1)
        for d, edge in self._digit.items():
            if edge.poll(window.key_down(d)):
                idx = int(d) - 1
                if 0 <= idx < self.env.num_moles and not self.env.touched[idx]:
                    self.selected = idx
                    arm = _arm_for_mole(self.env, idx, self.ArmTag)
                    hole = self.env.holes[self.env.mole_holes[idx]]
                    _set_cube_over_hole(self.env, str(arm), hole)
                    print(f"Selected mole {idx} ({arm} arm).")
        # Keep cube hovering over selection while waiting (unless weld rewrites it).
        if not self.env.touched[self.selected]:
            arm = _arm_for_mole(self.env, self.selected, self.ArmTag)
            hole = self.env.holes[self.env.mole_holes[self.selected]]
            # Only re-seat when mole is rising so the player can time the jab.
            if self.env._mole_is_rising(self.selected) or self.env._mole_above_surface(self.selected):
                drop = float(self.env._mole_raise_drop())
                hover_z = float(
                    self.env.board_top_z
                    + self.env.mole_height
                    - drop
                    + float(self.env.cube_half)
                    + 0.03
                )
                _set_cube_over_hole(self.env, str(arm), hole, z=hover_z)


class RobotMoleController:
    """Latch cradle mallets on Space-close; strike by teleoping the head onto moles.

    Gripper highlight is owned by UniversalRobotControls (1/2/3) only.
    """

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.selected_arm = "left"
        self.busy = False
        self._prev_width = {"left": 1.0, "right": 1.0}

    def _try_latch(self, selected):
        self.busy = True
        need = [s for s in selected if s not in self.env.hammer_cubes]
        if not need:
            self.busy = False
            return
        latched = []
        failed = []
        for side in need:
            # Let the shared Space close settle before judging proximity.
            try:
                self.env.robot.set_gripper(0.0, side, gripper_eps=0.0)
            except Exception:
                pass
            self.env._dwell(12)
            if self.env.try_latch_staged_mallet(self.ArmTag(side)):
                latched.append(side)
            else:
                failed.append(side)
        if latched:
            print(
                f"Picked up {', '.join(latched)} mallet(s). "
                "Teleop over rising moles and press the head down to strike."
            )
        if failed:
            action_failed(
                self.env,
                failed,
                detail="close on the mallet handle to latch",
            )
        self.busy = False

    def update(self, window):
        if self.busy or self.env.distractor_hit or getattr(
                self.env, "appearances_exhausted", False):
            return
        # Do not fall back to a default arm — wait for 1 / 2 / 3.
        selected = tuple(getattr(self.env, "_interactive_selected_arms", ()) or ())
        if not selected:
            return
        if len(selected) == 1:
            self.selected_arm = selected[0]

        arms = list(selected)
        closing = False
        for side in ("left", "right"):
            width = gripper_width(self.env, side)
            prev = self._prev_width.get(side, 1.0)
            if side in arms and prev > 0.5 and width <= 0.5:
                closing = True
            self._prev_width[side] = width
        if closing:
            self._try_latch(arms)


def main():
    parser = argparse.ArgumentParser(description="Interactive whack_moles viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="planner")
    add_record_data_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.whack_moles import whack_moles
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("whack_moles", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    use_robot = is_robot_control(args.control)
    env = whack_moles()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=use_robot))
    prepare_interactive_control(env, args.control)
    print(
        f"moles={env.num_moles}; appearances={env.num_appearances}; "
        f"distractors={env.num_distractors}; "
        f"relocating={env.relocating_moles}; difficulty={env.difficulty}."
    )
    print_episode_condition(env)

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)
        controller = RobotMoleController(env, ArmTag)
        print_instructions("Latch a mallet with Space, then jab rising moles.")
    else:
        env._cube_weld = {}
        controller = ClickMoleController(env)
        viewer.register_click_handler(controller.on_click)
        print_instructions(
            "Click moles while they are above the board; rabbits fail."
        )

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

            if env.distractor_hit:
                report_task_result(env, "rabbit touched")
                terminal_started_at = time.perf_counter()
                continue
            if getattr(env, "appearances_exhausted", False):
                detail = getattr(env, "_last_fail_reason", None) or (
                    f"missed after {getattr(env, 'num_appearances', 5)} appearances")
                report_task_result(env, detail)
                terminal_started_at = time.perf_counter()
                continue
            if env.check_success():
                if done_since is None:
                    done_since = time.perf_counter()
                    print("All moles hit; wrapping up…")
                elif time.perf_counter() - done_since >= 1.0:
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
