#!/usr/bin/env python3
"""Interactive viewer for ``put_cup_belt``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_put_cup_belt.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_put_cup_belt.py --control robot

Grasp / release with Space (open/close selected gripper). Teleop the cup onto the
belt gap; when the cup leaves the fingers (open gripper or slip), landing pose
is scored after a short settle — gripper open is not required for success.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import sapien.physx

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    UniversalRobotControls,
    add_record_data_arg,
    configure_task,
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
  Mouse click       click the belt to teleport the cup onto that belt XY
"""

CONTROLS_ROBOT = """
  Space             open / close selected gripper to grasp or release the cup
  1 / 2 / 3         select left / right / both arms (selected gripper turns green)
  Arrows / E / Q    teleop the selected arm(s)
  When the cup leaves the fingers, landing is scored after a short settle.
"""


def _get_rigid(actor):
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_cup_pose(env, x, y, z=None, kinematic=True, quat=None):
    pose = env.cup.get_pose()
    if z is None:
        z = float(pose.p[2])
    if quat is None:
        quat = pose.q
    new_pose = sapien.Pose([float(x), float(y), float(z)], quat)
    env.cup.actor.set_pose(new_pose)
    rigid = _get_rigid(env.cup)
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


def _mark_deposit(env):
    env._attempt_active = True
    env._deposit_step = int(getattr(env, "_kin_step", 0))
    env._slot_x_at_deposit = float(env.slot_x())


def _table_z(env) -> float:
    """put_cup_belt has no ``table_top``; match env spawn height."""
    return float(0.74 + float(getattr(env, "table_z_bias", 0.0)))


def _teleport_cup_to_belt(env, x: float, y: float) -> None:
    """Place the cup on the belt at ``(x, y)`` with Z raised to the belt surface."""
    z0 = _table_z(env)
    # ``slot_z`` is the belt-top height used by the env for place / success checks.
    belt_z = float(getattr(env, "slot_z", z0 + float(getattr(env, "BELT_Z_OFF", 0.01))))
    half_y = float(getattr(env, "belt_plate_half_size", [0.0, 0.05])[1])
    y = float(np.clip(y, float(env.belt_y) - half_y, float(env.belt_y) + half_y))
    half_x = float(getattr(env, "belt_plate_half_size", [0.2, 0.0])[0])
    center_x = float(getattr(env, "belt_center_x", 0.0))
    x = float(np.clip(x, center_x - half_x, center_x + half_x))
    _set_cup_pose(
        env, x, y, belt_z, kinematic=False, quat=env.CUP_UPRIGHT_QPOS,
    )
    _mark_deposit(env)
    print(
        f"Cup teleported onto belt at ({x:.3f}, {y:.3f}, z={belt_z:.3f}); "
        "evaluating landing…"
    )


class CupReleaseMonitor:
    """Watch finger contact; when the cup leaves the hand, score the landing.

    Mirrors the previous slip / Space-release path: mark deposit on detach,
    settle briefly, then ``check_success`` / placement score.
    """

    def __init__(self, env):
        self.env = env
        self.holding = False
        self.placed = False
        # Require sustained contact before treating a later drop as detach
        # (avoids one-frame PhysX flicker right after grasp).
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def update(self, _window=None):
        if self.placed:
            return
        held = bool(self.env._cup_held())
        if held:
            if not self.holding:
                self.holding = True
                self.env._attempt_active = True
                print("Cup grasped — teleop to the belt gap, then Space to open / release.")
            self._hold_contact_seen = True
            self._no_contact_steps = 0
            return
        if not self.holding:
            return
        self._no_contact_steps += 1
        limit = (
            self._slip_no_contact_steps
            if self._hold_contact_seen
            else self._slip_no_contact_steps * 4
        )
        if self._no_contact_steps < limit:
            return
        p = np.asarray(self.env.cup.get_pose().p, dtype=float)
        _mark_deposit(self.env)
        self.holding = False
        self.placed = True
        print(
            f"Cup detached at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}); "
            "evaluating landing…"
        )


def main():
    parser = argparse.ArgumentParser(description="Interactive put_cup_belt viewer")
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
    add_record_data_arg(parser)
    args = parser.parse_args()

    from envs.put_cup_belt import put_cup_belt

    print_mode_controls("put_cup_belt", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    use_robot = args.control == "robot"
    env = put_cup_belt()
    env._interactive_robot_mode = use_robot
    env.setup_demo(**configure_task("put_cup_belt", args.config, args.seed, use_robot=use_robot))
    print_episode_condition(env)
    print(
        f"Side={'left' if env.mirrored else 'right'}; "
        f"curtains={env.blue_curtains_enabled}; "
        f"gap x≈{env.slot_x():.3f}."
    )

    release_monitor = CupReleaseMonitor(env) if use_robot else None
    cup_placed = {"done": False}

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if use_robot:
        if views.robot_controls is None:
            views.robot_controls = UniversalRobotControls(env)
        print_instructions(
            "Space opens/closes the gripper to grasp/release the cup. "
            "When the cup leaves the fingers, landing is scored."
        )
    else:
        def _on_click(viewer, pixel_x, pixel_y):
            if cup_placed["done"]:
                return False
            hit = table_xy_from_click(viewer, pixel_x, pixel_y, _table_z(env))
            if hit is None:
                return False
            half_y = float(getattr(env, "belt_plate_half_size", [0.0, 0.05])[1]) + 0.04
            if abs(float(hit[1]) - float(env.belt_y)) > half_y:
                print("Click the belt to place the cup.")
                return True
            _teleport_cup_to_belt(env, hit[0], hit[1])
            cup_placed["done"] = True
            return True

        viewer.register_click_handler(_on_click)
        print_instructions("Click the belt to teleport the cup onto it.")

    placed_since = None
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if release_monitor is not None:
                release_monitor.update(viewer.window)

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

            placed = (
                bool(release_monitor.placed) if release_monitor is not None
                else bool(cup_placed["done"])
            )
            if getattr(env, "_curtain_hit", False) and placed_since is None:
                report_task_result(env, "curtain contact")
                terminal_started_at = time.perf_counter()
                continue
            if placed:
                if placed_since is None:
                    placed_since = time.perf_counter()
                    print("Cup on belt; settling…")
                elif time.perf_counter() - placed_since >= 2.0:
                    hit = bool(getattr(env, "_curtain_hit", False))
                    detail = f"score={env.placement_score():.2f}"
                    if hit:
                        detail = f"curtain hit; {detail}"
                    report_task_result(env, detail)
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
