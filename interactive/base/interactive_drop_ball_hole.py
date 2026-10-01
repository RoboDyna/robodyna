#!/usr/bin/env python3
"""Interactive sandbox for ``drop_ball_hole``.

Keyboard+mouse: click the rotating platform → ball drops from elevate height.
Robot: grasp / release with Space when the hole aligns.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_drop_ball_hole.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_drop_ball_hole.py --control robot
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
    gripper_width,
    is_robot_control,
    prepare_interactive_control,
    print_banner,
    release_dynamic,
    run_viewer_loop,
    print_episode_condition,
    selected_robot_arms,
    table_xy_from_click,
)

bootstrap_repo()


def _elevate_z(env) -> float:
    return float(
        env.cap_z
        + env.cap_thickness
        + env.ball_radius
        + float(getattr(env, "transport_clearance_z", env.TRANSPORT_CLEARANCE_Z_DEFAULT))
    )


def _clamp_to_platform(env, x: float, y: float):
    local = env._world_to_cap_local_xy(np.array([x, y], dtype=float))
    if getattr(env, "container_shape", "") == "cylinder":
        r = float(env.cap_radius) - 0.01
        n = float(np.linalg.norm(local))
        if n > r and n > 1e-9:
            local = local * (r / n)
    else:
        half = float(env.cap_half_extent) - 0.01
        local = np.clip(local, -half, half)
    world = env._cap_local_to_world_xy(local)
    return float(world[0]), float(world[1])


def _drop_ball_at(env, x: float, y: float):
    x, y = _clamp_to_platform(env, x, y)
    z = _elevate_z(env)
    pose = sapien.Pose([x, y, z], list(env.ball.get_pose().q))
    try:
        env.ball.set_pose(pose)
    except Exception:
        env.ball.actor.set_pose(pose)
    rigid = getattr(env, "_ball_rigid", None)
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(pose)
        except Exception:
            pass
    release_dynamic(rigid)
    if hasattr(env, "mark_ball_released"):
        env.mark_ball_released()
    else:
        env.ball_released = True
    env._interactive_released = True
    print(f"Ball dropped from z={z:.3f} onto ({x:.3f}, {y:.3f}).")


class PlatformClickDrop:
    def __init__(self, env):
        self.env = env
        self.dropped = False

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.dropped or bool(getattr(self.env, "ball_released", False)):
            return False
        plane_z = float(self.env.cap_z + self.env.cap_thickness)
        hit = table_xy_from_click(viewer, pixel_x, pixel_y, plane_z)
        if hit is None:
            return False
        _drop_ball_at(self.env, hit[0], hit[1])
        self.dropped = True
        return True


def main():
    parser = argparse.ArgumentParser(description="Interactive drop_ball_hole viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.drop_ball_hole import drop_ball_hole

    use_robot = is_robot_control(args.control)
    env = drop_ball_hole()
    env.setup_demo(**configure_task(
        "drop_ball_hole", args.config, args.seed, use_robot=use_robot,
    ))
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    env._interactive_released = False
    env._cap_tracking = True

    if use_robot:
        print_banner(
            "drop_ball_hole — interactive controls",
            [
                f"Mode: {args.control}  |  robot-motion: {args.robot_motion}",
                "Grasp the ball; release when the hole aligns underneath.",
            ],
        )
        print_instructions("Close Space on the ball to grasp, open Space to drop.")
    else:
        print_banner(
            "drop_ball_hole — keyboard+mouse",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Click the rotating platform — the ball drops from elevate height.",
            ],
        )
        viewer = env.viewer
        if viewer is None:
            raise SystemExit("Viewer was not created.")
        clicker = PlatformClickDrop(env)
        viewer.register_click_handler(clicker.on_click)
        print_instructions("Click the platform surface to drop the ball.")

    if env._uses_drop_timeout():
        print(
            f"Drop timeout armed: {float(env.drop_timeout_s):.0f}s after release "
            "(stick_to_surface off)."
        )

    prev_gripper_closed = False
    grasped_ball = False

    def _tcp_xyz(side: str) -> np.ndarray | None:
        robot = getattr(env, "robot", None)
        if robot is None:
            return None
        getter = robot.get_left_tcp_pose if side == "left" else robot.get_right_tcp_pose
        try:
            return np.asarray(getter()[:3], dtype=np.float64)
        except Exception:
            return None

    def _ball_near_gripper(side: str, tol: float = 0.09) -> bool:
        if getattr(env, "ball", None) is None:
            return False
        tcp = _tcp_xyz(side)
        if tcp is None:
            return False
        ball = np.asarray(env.ball.get_pose().p[:3], dtype=np.float64)
        return float(np.linalg.norm(ball - tcp)) <= float(tol)

    def _mark_released(message: str):
        if bool(getattr(env, "ball_released", False)):
            return
        if hasattr(env, "mark_ball_released"):
            env.mark_ball_released()
        else:
            env.ball_released = True
        env._interactive_released = True
        print(message)

    def on_step(window, step):
        nonlocal prev_gripper_closed, grasped_ball
        del window, step
        if use_robot and not bool(getattr(env, "ball_released", False)):
            arms = selected_robot_arms(env)
            if arms:
                closed_now = all(float(gripper_width(env, side)) < 0.5 for side in arms)
                near_now = any(_ball_near_gripper(side) for side in arms)
                if closed_now and near_now:
                    grasped_ball = True
                opening = (not closed_now) and prev_gripper_closed
                if grasped_ball and opening:
                    if env._uses_drop_timeout():
                        _mark_released(
                            f"Released ball — {float(env.drop_timeout_s):.0f}s to land in the box."
                        )
                    else:
                        _mark_released("Released ball — hope the hole was aligned.")
                    grasped_ball = False
                prev_gripper_closed = closed_now

    def is_done(step):
        del step
        if env.check_success():
            return True, "ball in box"
        if getattr(env, "_ball_fell_off_table", False):
            return True, "ball dropped off the table"
        if getattr(env, "_drop_timed_out", False):
            return True, (
                f"drop timeout ({float(env.drop_timeout_s):.0f}s) — ball not in box"
            )
        if env.ball_stuck_on_platform:
            return True, "ball stuck on platform"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=20000)


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
