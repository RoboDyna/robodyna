#!/usr/bin/env python3
"""Interactive viewer for ``hit_target``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_hit_target.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_hit_target.py --control robot

Keyboard+mouse: click the board to plant the dart; click a blocker to fail.
Robot: grasp the dart and jab the yellow center by teleop.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    UniversalRobotControls,
    actor_scene_id,
    add_robot_motion_arg,
    click_hits_actor_map,
    escape_quit_requested,
    is_robot_control,
    make_viewer_view_toggle,
    prepare_interactive_control,
    print_instructions,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Mouse click       click the target to plant the dart; click a blocker to fail
"""

CONTROLS_ROBOT = """
  Space             open / close gripper to grasp the dart
  Drive the tip into the yellow center with teleop to jab / stick.
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
        task_name="hit_target",
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


def _get_rigid(actor):
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _tip(env):
    return np.asarray(env.dart.get_functional_point(0, "list")[:3], dtype=float)


def _set_tip_xyz(env, tip_xyz, kinematic=True):
    """Translate the dart so its tip lands at tip_xyz (keeps orientation)."""
    tip = _tip(env)
    body = np.asarray(env.dart.get_pose().p, dtype=float)
    offset = tip - body
    new_body = np.asarray(tip_xyz, dtype=float) - offset
    pose = env.dart.get_pose()
    new_pose = sapien.Pose(new_body.tolist(), pose.q)
    entity = getattr(env.dart, "actor", env.dart)
    entity.set_pose(new_pose)
    rigid = env._dart_rigid or _get_rigid(env.dart)
    env._dart_rigid = rigid
    if rigid is None:
        return
    try:
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        if kinematic:
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(new_pose)
        else:
            rigid.set_kinematic(False)
    except Exception:
        pass


def _blocker_tip_xyz(env, which: str):
    bz = float(env.blocker_z)
    if which == "static":
        return np.array(
            [float(env._static_blocker_x()), float(env.static_blocker_y), bz],
            dtype=float,
        )
    bx = float(env._dynamic_blocker_x_at(getattr(env, "_step_count", 0)))
    return np.array([bx, float(env.dynamic_blocker_y), bz], dtype=float)


class ClickDartController:
    """One click plants the dart on the target or fails on a blocker."""

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self.done = False
        self._ids = {}
        tid = actor_scene_id(getattr(env, "target", None))
        if tid is not None:
            self._ids[int(tid)] = "target"
        for name in ("static_blocker", "dynamic_blocker", "blocker"):
            actor = getattr(env, name, None)
            sid = actor_scene_id(actor)
            if sid is None:
                continue
            label = "dynamic" if "dynamic" in name else "static"
            if name == "blocker":
                label = "dynamic" if getattr(env, "dynamic_blocker", None) is actor else "static"
            self._ids[int(sid)] = label

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.done or self.env._stuck or self.env._hit_blocker:
            return False
        hit = click_hits_actor_map(viewer, pixel_x, pixel_y, self._ids)
        if hit is None:
            return False
        if hit == "target":
            center = np.asarray(self.env._target_center_world(), dtype=float)
            tip = np.array(
                [center[0], float(self.env._plant_tip_y()), center[2]],
                dtype=float,
            )
            _set_tip_xyz(self.env, tip, kinematic=True)
            self.env._try_form_stick(any_ring=True, exact_pose=True)
            if not self.env._stuck:
                # Force a center plant if spring stick didn't latch yet.
                self.env._record_board_hit()
                self.env._try_form_stick(any_ring=True, exact_pose=True)
            print("Dart planted on the target.")
        else:
            tip = _blocker_tip_xyz(self.env, hit)
            _set_tip_xyz(self.env, tip, kinematic=True)
            self.env._hit_blocker = True
            self.env.hit_score = 0.0
            self.env._check_blocker_hit()
            print(f"Dart hit the {hit} blocker — failure.")
        self.done = True
        return True

    def update(self, _window):
        return


class RobotDartController:
    """Teleop aim only — Space gripper toggle is shared; jab by driving tip in."""

    def __init__(self, env):
        self.env = env

    def update(self, window):
        del window


def main():
    parser = argparse.ArgumentParser(description="Interactive hit_target viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.hit_target import hit_target
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("hit_target", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    use_robot = is_robot_control(args.control)
    env = hit_target()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=use_robot))
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    print(
        f"Arm={'right' if env.dart_side > 0 else 'left'}; "
        f"blocker_static={env.blocker_enabled}; blocker_dyn={env.blocker_dynamic}."
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)
        controller = RobotDartController(env)
        print_instructions("Grasp the dart and jab the yellow center.")
    else:
        controller = ClickDartController(env, viewer)
        viewer.register_click_handler(controller.on_click)
        print_instructions(
            "Click the target to plant the dart, or a blocker to fail."
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
                if not env._stuck and not env._hit_blocker:
                    env._try_form_stick(any_ring=True, exact_pose=True)
            env.scene.update_render()
            viewer.render()

            if escape_quit_requested(env, viewer.window):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if env._hit_blocker:
                report_task_result(env, env.hit_result_detail())
                terminal_started_at = time.perf_counter()
                continue
            if env._stuck or env._hit_color is not None:
                if settle_after is None:
                    settle_after = time.perf_counter()
                elif time.perf_counter() - settle_after >= 1.0:
                    if not env._hit_blocker and env._hit_color is None:
                        env._record_board_hit()
                    report_task_result(env, env.hit_result_detail())
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
