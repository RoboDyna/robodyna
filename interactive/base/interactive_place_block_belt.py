#!/usr/bin/env python3
"""Interactive sandbox for ``place_block_belt``.

Keyboard+mouse: click the belt to drop the cube from 2 cm above the surface.
Robot: grasp with Space, teleop over the belt, open to drop.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_place_block_belt.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_place_block_belt.py --control robot
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
    is_robot_control,
    prepare_interactive_control,
    print_banner,
    run_viewer_loop,
    print_episode_condition,
    table_xy_from_click,
)

bootstrap_repo()


def _block_held(env) -> bool:
    if getattr(env, "block", None) is None:
        return False
    try:
        return len(env.get_gripper_actor_contact_position(env.block.get_name())) > 0
    except Exception:
        return False


# Click spawn: cube bottom is this far above the belt, then gravity drops it.
_KEYBOARD_DROP_HEIGHT = 0.02


def _teleport_block_to_belt(env, x: float, y: float) -> None:
    """Place the cube 2 cm above the belt at ``(x, y)`` and let it drop."""
    half_x = float(getattr(env, "_belt_half_len_x", 0.2))
    half_y = float(getattr(env, "_belt_half_w", getattr(env, "belt_half_w", 0.05)))
    cx = float(getattr(env, "_belt_cx", 0.0))
    by = float(env.belt_y)
    x = float(np.clip(x, cx - half_x, cx + half_x))
    y = float(np.clip(y, by - half_y, by + half_y))
    z = float(env.belt_surface_z + env.block_half_h + _KEYBOARD_DROP_HEIGHT)
    q = [1.0, 0.0, 0.0, 0.0]
    pose = sapien.Pose([x, y, z], q)
    try:
        env.block.set_pose(pose)
    except Exception:
        env.block.actor.set_pose(pose)
    rigid = getattr(env, "_block_dyn", None)
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
        except Exception:
            pass
    env._release_q = list(q)
    env._released = True
    env._interactive_released = True
    env._interactive_holding = False
    env._belt_active = False
    env._release_delay_left = 0
    env._keyboard_await_drop = True
    print(
        f"Block dropping onto belt from {_KEYBOARD_DROP_HEIGHT * 100:.0f} cm "
        f"at ({x:.3f}, {y:.3f})."
    )


class BlockClickController:
    def __init__(self, env):
        self.env = env
        self.placed = False

    def on_click(self, viewer, pixel_x, pixel_y):
        if self.placed or getattr(self.env, "_interactive_released", False):
            return False
        hit = table_xy_from_click(
            viewer, pixel_x, pixel_y, float(self.env.belt_surface_z)
        )
        if hit is None:
            return False
        _teleport_block_to_belt(self.env, hit[0], hit[1])
        self.placed = True
        return True


class BlockReleaseMonitor:
    def __init__(self, env):
        self.env = env
        self.holding = False
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def update(self):
        if getattr(self.env, "_interactive_released", False):
            return
        held = _block_held(self.env)
        if held:
            if not self.holding:
                self.holding = True
                print("Block grasped — teleop over the belt, then Space to open / release.")
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
        self.env._interactive_released = True
        self.env._interactive_holding = False
        self.env._release_q = [1.0, 0.0, 0.0, 0.0]
        self.env._released = True
        self.env._release_delay_left = int(self.env.belt_release_delay_steps)
        print("Block released — belt will engage after release delay.")


def main():
    parser = argparse.ArgumentParser(description="Interactive place_block_belt viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.place_block_belt import place_block_belt

    use_robot = is_robot_control(args.control)
    env = place_block_belt()
    env.setup_demo(**configure_task(
        "place_block_belt", args.config, args.seed, use_robot=use_robot,
    ))
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)

    if use_robot:
        print_banner(
            "place_block_belt — interactive controls",
            [
                f"Mode: {args.control}  |  robot-motion: {args.robot_motion}",
                "Grasp the block, place before the red line, Space to release.",
            ],
        )
    else:
        print_banner(
            "place_block_belt — keyboard+mouse",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Click the belt to drop the cube from 2 cm above the surface.",
            ],
        )

    env._interactive_holding = False
    env._interactive_released = False
    env._released = False
    env._belt_active = False
    env._release_delay_left = 0
    env._keyboard_await_drop = False

    clicker = None
    release_monitor = None
    if use_robot:
        release_monitor = BlockReleaseMonitor(env)
        print_instructions("Space closes/opens the gripper to grasp/release the block.")
    else:
        viewer = env.viewer
        if viewer is None:
            raise SystemExit("Viewer was not created.")
        clicker = BlockClickController(env)
        viewer.register_click_handler(clicker.on_click)
        print_instructions("Click once on the belt to drop the cube from 2 cm above.")

    off_belt_since = None
    settle_steps = max(1, int(round(2.0 / float(env.scene.get_timestep()))))

    def on_step(window, step):
        nonlocal off_belt_since
        del window
        if release_monitor is not None:
            release_monitor.update()

        if env._interactive_released:
            if getattr(env, "_keyboard_await_drop", False):
                seated_z = float(env.belt_surface_z + env.block_half_h)
                z = float(env.block.get_pose().p[2])
                if z <= seated_z + 0.006:
                    env._keyboard_await_drop = False
                    env._release_delay_left = int(
                        getattr(env, "belt_release_delay_steps", 0)
                    )
                    if env._release_delay_left <= 0:
                        env._belt_active = True
                        print("Belt drive engaged.")
            if env._release_delay_left > 0:
                env._release_delay_left -= 1
                if env._release_delay_left == 0:
                    env._belt_active = True
                    print("Belt drive engaged.")

            left_belt = bool(getattr(env, "_block_dropped", False)) or (
                bool(getattr(env, "placed_on_belt", False)) and not env._on_belt()
            )
            if left_belt and off_belt_since is None:
                off_belt_since = step
                print("Block has left the belt — settling for 2 seconds…")

    def is_done(step):
        if off_belt_since is not None and step - off_belt_since >= settle_steps:
            return True, (
                f"left_belt=True, on_belt={env.placed_on_belt}, in_bowl={env.in_bowl}, "
                f"avoided_blocker={env.avoided_blocker}"
            )
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=25000)


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
