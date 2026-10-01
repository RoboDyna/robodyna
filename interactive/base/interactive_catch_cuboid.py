#!/usr/bin/env python3
"""Interactive viewer for ``catch_cuboid``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_catch_cuboid.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_catch_cuboid.py --control robot

Close the gripper (Space) while the cuboid is rising to latch it, then lift it out.
Opt1 supports dual arms.
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
    print_instructions,
    action_failed,
    actor_scene_id,
    add_record_data_arg,
    click_hits_actor_map,
    escape_quit_requested,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)

CONTROLS_KEYBOARD = """
  Mouse click       each cuboid can be clicked once. Above the board: pull it out.
                    Below the board surface: episode FAILURE.
"""

CONTROLS_ROBOT = """
  Space             open / close gripper (both jaws touching a risen cuboid grips it, then lift)
  Arrow keys        move selected arm in XY
  E / Q             move selected arm in Z
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
        task_name="catch_cuboid",
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


def _cuboid_above_board(env, idx=0):
    """True when any of the cuboid is visibly above the board top."""
    if idx >= len(getattr(env, "cuboids", []) or []):
        return False
    top = float(env.cuboids[idx].get_pose().p[2]) + float(env.cuboid_half[2])
    return top >= float(env.board_top_z) + 0.005


def _cuboid_linear_speed(env, idx):
    if idx >= len(getattr(env, "cuboids", []) or []):
        return 0.0
    cuboid = env.cuboids[idx]
    rigid = None
    try:
        actor = getattr(cuboid, "actor", cuboid)
        rigid = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    except Exception:
        rigid = None
    if rigid is None:
        return 0.0
    try:
        v = np.asarray(rigid.get_linear_velocity(), dtype=float)
        w = np.asarray(rigid.get_angular_velocity(), dtype=float)
        return float(np.linalg.norm(v) + 0.05 * np.linalg.norm(w))
    except Exception:
        return 0.0


class KeyboardCatchController:
    """One click per cuboid: extract if above the board, else episode failure."""

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.dual = bool(env.dual_catch)
        self._latched = set()
        self._clicked = set()
        self.done = False
        self.success = False
        self.fail_detail = None
        self._cuboid_ids = {}
        for i, cuboid in enumerate(getattr(env, "cuboids", []) or []):
            sid = actor_scene_id(cuboid)
            if sid is not None:
                self._cuboid_ids[int(sid)] = int(i)

    def update(self, _window):
        return

    def _extract_cuboid(self, idx: int) -> None:
        import sapien

        env = self.env
        cuboid = env.cuboids[idx]
        env._release_cuboid(idx)
        p = np.asarray(cuboid.get_pose().p, dtype=float)
        # Park just clear of the board so the cuboid "comes out and stays outside".
        clear_z = (
            float(env.board_top_z)
            + float(env.cuboid_half[2])
            + float(getattr(env, "PULL_OUT_CLEARANCE", 0.04))
            + 0.02
        )
        outward_y = float(p[1]) - 0.08
        pose = sapien.Pose([float(p[0]), outward_y, clear_z], list(cuboid.get_pose().q))
        try:
            cuboid.set_pose(pose)
        except Exception:
            cuboid.actor.set_pose(pose)
        try:
            rigid = cuboid.actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if rigid is not None:
                rigid.set_linear_velocity([0, 0, 0])
                rigid.set_angular_velocity([0, 0, 0])
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
        except Exception:
            pass
        env.catches = int(getattr(env, "catches", 0)) + 1
        self._latched.add(idx)
        print(f"Pulled {env._cuboid_names[idx]} out of the hole.")
        need = 2 if self.dual else 1
        if len(self._latched) >= need:
            self.done = True
            self.success = True

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.done:
            return False
        idx = click_hits_actor_map(viewer, pixel_x, pixel_y, self._cuboid_ids)
        if idx is None:
            return False
        idx = int(idx)
        if idx in self._latched or idx in self._clicked:
            return True
        self._clicked.add(idx)
        if not _cuboid_above_board(self.env, idx):
            name = self.env._cuboid_names[idx]
            detail = f"{name} clicked below the board surface"
            self.done = True
            self.success = False
            self.fail_detail = detail
            print(f"Miss — {name} is below the board — episode FAILURE.")
            return True
        self._extract_cuboid(idx)
        return True


class RobotCatchController:
    """Watch the env's contact-driven grasp and end the episode on a drop.

    The grasp itself is physics-only: ``catch_cuboid`` hands a cuboid over to
    dynamics when both jaws touch it, so teleop and any policy behave alike.
    Gripper highlight is owned by UniversalRobotControls (1/2/3) only.
    """

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.dual = bool(env.dual_catch)
        self.busy = False
        self.selected = "right"
        if not self.dual:
            hole = env._cuboid_holes[0]
            self.selected = "right" if env.holes[hole][0] > 0 else "left"
        self._latched = set()
        self._released_at = {}
        self.done = False
        self.success = False
        self.fail_detail = None

    def _fail(self, detail):
        arms = tuple(getattr(self.env, "_interactive_selected_arms", ()) or ())
        if not arms:
            arms = ("left", "right")
        action_failed(self.env, arms, detail=detail)
        self.done = True
        self.success = False
        self.fail_detail = detail
        print(f"Task failed ({detail}).")

    def _note_env_releases(self):
        """Track cuboids the env handed to dynamics after a two-jaw pinch."""
        rigids = getattr(self.env, "_cuboid_rigids", []) or []
        for i, rigid in enumerate(rigids):
            if rigid is not None or i in self._latched:
                continue
            self._latched.add(i)
            self._released_at[i] = time.perf_counter()
            print(
                f"{self.env._cuboid_names[i]} is gripped and free of the board drive; "
                "lift it out of the hole."
            )

    def _check_grasp_failure(self):
        """End the episode once a grasped cuboid is dropped and comes to rest."""
        if self.done or self.env.check_success():
            return
        now = time.perf_counter()
        for i in sorted(self._latched):
            name = self.env._cuboid_names[i]
            started = self._released_at.get(i)
            if started is None or (now - started) < 0.6:
                continue
            try:
                if bool(self.env._cuboid_in_gripper(name)):
                    continue
            except Exception:
                pass
            if _cuboid_linear_speed(self.env, i) > 0.05:
                continue
            self._fail(f"{name} dropped")
            return

    def update(self, window):
        if self.done:
            return
        self._note_env_releases()
        self._check_grasp_failure()

def main():
    parser = argparse.ArgumentParser(description="Interactive catch_cuboid viewer")
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
        default="planner",
        help="Robot motion backend (interpolate = faster joint interp when supported; default planner)",
    )
    add_record_data_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_cuboid import catch_cuboid
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    # Ensure default raster viewer (no leftover RT glass shader from prior runs).
    try:
        default_shader = Path(sapien.__file__).resolve().parent / "vulkan_shader" / "default"
        if default_shader.is_dir():
            sapien.render.set_viewer_shader_dir(str(default_shader))
    except Exception:
        pass

    print_mode_controls("catch_cuboid", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = catch_cuboid()
    env._interactive_robot_mode = args.control == "robot"
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print_episode_condition(env)
    # Start with open grippers so Space close has an effect.
    if args.control == "robot":
        try:
            env.together_open_gripper(save_freq=None)
        except Exception:
            pass
    print(
        f"dual={env.dual_catch}; opaque={env.opaque_surface}; "
        f"holes={env._cuboid_holes}; speed={env._cuboid_pop_speed:.3f} m/s."
    )
    if args.control == "robot":
        print_instructions(
            "Arrows/E/Q move the arm; close with Space to latch, then lift the cuboid out."
        )
    else:
        print_instructions(
            "Click each cuboid once while it is above the board to pull it out; "
            "clicking while it is below the surface is a failure."
        )
    controller = (
        RobotCatchController(env, ArmTag) if args.control == "robot"
        else KeyboardCatchController(env, ArmTag)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if args.control in ("keyboard", "keyboard+mouse"):
        viewer.register_click_handler(controller.on_click)

    settle_after = None
    # Env never increments appearances_done; count completed pop cycles locally.
    num_appearances = int(getattr(env, "num_appearances", 5) or 5)
    cuboid_cycles = [0] * len(getattr(env, "cuboids", []) or [])
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
                prev_motion = list(getattr(env, "_cuboid_auto_motion", []) or [])
                env._update_kinematic_tasks()
                env.scene.step()
                motions = list(getattr(env, "_cuboid_auto_motion", []) or [])
                if len(cuboid_cycles) < len(motions):
                    cuboid_cycles.extend([0] * (len(motions) - len(cuboid_cycles)))
                for i, motion in enumerate(motions):
                    was = prev_motion[i] if i < len(prev_motion) else None
                    # Appearance ends when a fall back into the hole finishes.
                    if was == "falling" and motion != "falling":
                        cuboid_cycles[i] += 1
            if hasattr(controller, "_note_env_releases"):
                controller._note_env_releases()
            if hasattr(controller, "_check_grasp_failure"):
                controller._check_grasp_failure()
            env.scene.update_render()
            viewer.render()

            if escape_quit_requested(env, viewer.window):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            latched = getattr(controller, "_latched", set()) or set()
            uncaught = [i for i in range(len(cuboid_cycles)) if i not in latched]
            appearances_exhausted = (
                bool(uncaught)
                and all(cuboid_cycles[i] >= num_appearances for i in uncaught)
                and not env.check_success()
            )
            if (getattr(controller, "done", False) or env.check_success()
                    or appearances_exhausted):
                if settle_after is None:
                    settle_after = time.perf_counter()
                elif time.perf_counter() - settle_after >= 1.0:
                    if getattr(controller, "success", False):
                        detail = f"catches={env.catches}"
                        report_task_result(env, detail, ok=True)
                    elif getattr(controller, "done", False) and not env.check_success():
                        detail = getattr(controller, "fail_detail", None) or "insufficient contact"
                        report_task_result(env, detail, ok=False)
                    elif appearances_exhausted:
                        n_done = max(cuboid_cycles) if cuboid_cycles else 0
                        detail = f"missed after {n_done}/{num_appearances} appearances"
                        report_task_result(env, detail, ok=False)
                    else:
                        detail = f"catches={env.catches}"
                        report_task_result(env, detail)
                    terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
