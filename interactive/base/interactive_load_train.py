#!/usr/bin/env python3
"""Interactive sandbox for ``load_train``.

Keyboard+mouse: marble hovers over the near-rail drop; Space releases it.
Robot: pick the ball, hover over a wagon, Space to open / drop.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_load_train.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_load_train.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _interactive_common import (  # noqa: E402
    print_instructions,
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    edge_pressed,
    is_robot_control,
    prepare_interactive_control,
    print_banner,
    release_dynamic,
    run_viewer_loop,
    print_episode_condition,
)

bootstrap_repo()


def _ball_held(env) -> bool:
    """True while fingers still contact the ball (Space-close grasp)."""
    if getattr(env, "ball", None) is None:
        return False
    try:
        return len(env.get_gripper_actor_contact_position(env.ball.get_name())) > 0
    except Exception:
        return False


def _hover_z(env) -> float:
    return float(
        env.car_floor_z
        + env.car_floor_h
        + env.car_wall_h
        + env.ball_radius
        + max(
            float(getattr(env, "transport_clearance_z", 0.02)),
            float(getattr(env, "release_clearance_z", 0.02)),
        )
    )


def _set_ball_pose(env, xyz, *, kinematic=True, gravity_off=True):
    pose = sapien.Pose(
        [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        list(env.ball.get_pose().q),
    )
    try:
        env.ball.set_pose(pose)
    except Exception:
        env.ball.actor.set_pose(pose)
    rigid = getattr(env, "_ball_rigid", None)
    if rigid is None:
        return
    try:
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        if gravity_off:
            rigid.set_disable_gravity(True)
        if kinematic:
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(pose)
        else:
            rigid.set_kinematic(False)
            if not gravity_off:
                rigid.set_disable_gravity(False)
    except Exception:
        pass


class KeyboardHoverRelease:
    """Hold the ball over the drop station; Space drops it into the train."""

    def __init__(self, env):
        self.env = env
        self._prev = {}
        self.released = False
        xy = np.asarray(env._drop_target_xy, dtype=float)
        _set_ball_pose(env, [xy[0], xy[1], _hover_z(env)], kinematic=True, gravity_off=True)
        print(
            "Marble hovering over the train drop — press Space to release "
            "when a wagon is underneath."
        )

    def update(self, window):
        if self.released:
            return
        xy = np.asarray(self.env._drop_target_xy, dtype=float)
        _set_ball_pose(
            self.env,
            [xy[0], xy[1], _hover_z(self.env)],
            kinematic=True,
            gravity_off=True,
        )
        if not edge_pressed(window, "space", self._prev):
            return
        self.released = True
        self.env._interactive_holding = False
        self.env._interactive_released = True
        rigid = getattr(self.env, "_ball_rigid", None)
        if rigid is not None:
            try:
                rigid.set_disable_gravity(False)
            except Exception:
                pass
        release_dynamic(rigid)
        self.env._ball_released = True
        self.env._bed_contact_steps = 0
        print("Ball released — watch for wagon latch.")


class BallReleaseMonitor:
    """When the ball leaves the hand, mark release so wagon latching can run."""

    def __init__(self, env):
        self.env = env
        self.holding = False
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def update(self):
        if getattr(self.env, "_interactive_released", False):
            return
        if _ball_held(self.env):
            if not self.holding:
                self.holding = True
                self.env._interactive_holding = True
                print("Ball grasped — carry it over a wagon, then Space to open / release.")
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
        self.holding = False
        self.env._interactive_holding = False
        self.env._interactive_released = True
        if self.env._ball_rigid is not None:
            try:
                self.env._ball_rigid.set_disable_gravity(False)
            except Exception:
                pass
        release_dynamic(self.env._ball_rigid)
        self.env._ball_released = True
        self.env._bed_contact_steps = 0
        print("Ball released — watch for wagon latch.")


def _outcome_detail(env) -> str:
    """Explain the result in wagon terms (Opt 1: only the red wagon counts)."""
    if getattr(env, "_ball_fell_off_table", False):
        return "ball dropped off the table"
    landed = getattr(env, "_latched_car_idx", None)
    target = getattr(env, "target_wagon_idx", None)
    if not getattr(env, "target_wagon_mode", False) or target is None:
        return "ball in a wagon" if env.ball_in_train else "ball not in any wagon"
    if landed is None:
        return f"ball not in any wagon (target was wagon {int(target)}, the red one)"
    if int(landed) == int(target):
        return f"ball in target wagon {int(target)}"
    return (
        f"ball landed in wagon {int(landed)} (gray distractor); "
        f"target was wagon {int(target)}, the red one"
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive load_train viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs.load_train import load_train

    use_robot = is_robot_control(args.control)
    env = load_train()
    env.setup_demo(**configure_task("load_train", args.config, args.seed, use_robot=use_robot))
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    env._train_running = True

    if use_robot:
        print_banner(
            "load_train — interactive controls",
            [
                f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
                f"config: {args.config}  |  seed: {args.seed}",
                "Goal: drop the ball into an open wagon as it passes under the near rail.",
                "Opt 1 (target wagon): ONLY the RED wagon counts — gray ones are distractors.",
                "1 / 2 / 3 — select left / right / both arms (robot mode; selected gripper turns green)",
                "Space — open / close selected gripper(s) only",
                "Arrows / E / Q — teleop the selected arm(s)",
                "V — cycle view: head_camera ↔ gripper(s)",
                "Esc — close the viewer window to quit",
                "Close Space on the ball to grasp; open Space over a wagon to drop.",
                "--robot-motion planner|interpolate",
            ],
        )
    else:
        print_banner(
            "load_train — keyboard+mouse",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Marble hovers over the low train drop station.",
                "Space — release the marble when a wagon is underneath",
                "Opt 1: only the RED wagon counts",
                "Esc — quit",
            ],
        )

    env._interactive_holding = False
    env._interactive_released = False
    env._ball_released = False
    release_monitor = BallReleaseMonitor(env) if use_robot else None
    hover = KeyboardHoverRelease(env) if not use_robot else None
    if use_robot:
        print_instructions(
            "Ball ready. Teleop to the ball, Space to close/open the gripper. "
            "Drop into a wagon as it passes."
        )
    else:
        print_instructions("Press Space to release the hovering marble into a wagon.")

    post_release = 0

    def on_step(window, step):
        nonlocal post_release
        del step
        if hover is not None:
            hover.update(window)
        if release_monitor is not None:
            release_monitor.update()
        if env._interactive_released:
            post_release += 1

    def is_done(step):
        del step
        if getattr(env, "_ball_fell_off_table", False):
            return True, "ball dropped off the table"
        if env._interactive_released and post_release > 400:
            return True, _outcome_detail(env)
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=20000)


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
