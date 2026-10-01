#!/usr/bin/env python3
"""Interactive sandbox for ``pick_ripe_apple``.

Keyboard+mouse: Left/Right (or click an apple) picks it over the basket center;
Space or a second click releases. Robot: teleop pinch / drop.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_pick_ripe_apple.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_pick_ripe_apple.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _interactive_common import (  # noqa: E402
    actor_scene_id,
    add_robot_motion_arg,
    bootstrap_repo,
    click_hits_actor_map,
    configure_task,
    edge_pressed,
    gripper_width,
    is_robot_control,
    prepare_interactive_control,
    print_banner,
    print_episode_condition,
    print_instructions,
    release_dynamic,
    run_viewer_loop,
)

bootstrap_repo()


def _move_arms_to_pre_grasp_orientation(env) -> None:
    """Reorient both open grippers to the front pre-grasp quat at home XYZ."""
    from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
    from envs.utils.action import ArmTag

    env.together_open_gripper(save_freq=None)
    for side_name in ("left", "right"):
        arm = ArmTag(side_name)
        if side_name == "left":
            origin = np.asarray(env.robot.left_original_pose, dtype=np.float64)
        else:
            origin = np.asarray(env.robot.right_original_pose, dtype=np.float64)
        xyz = origin[:3].tolist()
        moved = False
        for key in env._front_grasp_quat_keys(arm):
            quat = list(GRASP_DIRECTION_DIC[key])
            pose = list(xyz) + quat
            env.plan_success = True
            env._last_plan_fail = None
            env.move(env.move_to_pose(arm, pose))
            if env.plan_success:
                print(f"{side_name} arm at home XYZ with front pre-grasp orientation ({key}).")
                moved = True
                break
        if not moved:
            detail = getattr(env, "_last_plan_fail", None) or "unreachable"
            print(f"Warning: {side_name} arm could not reach pre-grasp orientation ({detail}).")


def _hover_z(env) -> float:
    """Hold above the basket lip so in-basket success cannot fire until the drop."""
    return float(getattr(env, "basket_top_z", 0.8) + 0.14)


def _set_apple_hover(env, side: float) -> None:
    apple = (getattr(env, "apples", {}) or {}).get(side)
    if apple is None:
        return
    x = float(env.basket_x)
    y = float(env.basket_y)
    z = _hover_z(env)
    q = list(apple.get_pose().q)
    pose = sapien.Pose([x, y, z], q)
    try:
        apple.set_pose(pose)
    except Exception:
        apple.actor.set_pose(pose)
    rigid = (getattr(env, "_apple_rigids", {}) or {}).get(side)
    if rigid is None:
        return
    try:
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        rigid.set_disable_gravity(True)
        rigid.set_kinematic(True)
        rigid.set_kinematic_target(pose)
    except Exception:
        pass


class KeyboardAppleController:
    """Arrow / click pick → hover over initial basket; Space / 2nd click release."""

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self.held_side = None
        self._prev = {}
        self._apple_ids = {}
        for side, apple in (getattr(env, "apples", {}) or {}).items():
            sid = actor_scene_id(apple)
            if sid is not None:
                self._apple_ids[int(sid)] = float(side)

    def _still_attached(self, side: float) -> bool:
        if abs(float(side) - float(self.env.apple_side)) < 0.5:
            return bool(getattr(self.env, "_apple_attached", False))
        return bool(getattr(self.env, "_spoiled_attached", False))

    def _pick(self, side: float) -> bool:
        side = float(side)
        if self.held_side is not None:
            return False
        if side not in (getattr(self.env, "apples", {}) or {}):
            return False
        if not self._still_attached(side):
            print("That apple is already detached.")
            return False
        self.env._detach_apple(side=side)
        self.held_side = side
        _set_apple_hover(self.env, side)
        label = "good" if abs(side - float(self.env.apple_side)) < 0.5 else "spoiled"
        print(
            f"Picked {label} apple — hovering over basket. "
            "Space or click again to release."
        )
        return True

    def _release(self) -> bool:
        if self.held_side is None:
            return False
        side = self.held_side
        rigid = (getattr(self.env, "_apple_rigids", {}) or {}).get(side)
        if rigid is None:
            rigid = getattr(self.env, "_apple_rigid", None)
        self.env._dampen_apple_for_basket_drop(rigid=rigid)
        self.env._enable_held_apple_gravity(rigid=rigid)
        release_dynamic(rigid)
        self.held_side = None
        print("Released apple — dropping under gravity.")
        return True

    def poll_space(self, window) -> None:
        """Release on Space. Called from on_step and after_render (0-physics frames)."""
        if self.held_side is None:
            return
        if edge_pressed(window, "space", self._prev):
            self._release()

    def on_click(self, viewer, pixel_x, pixel_y):
        # While carrying, any click releases. The hover pose sits over the basket,
        # so a second click often hits the basket mesh instead of the apple.
        if self.held_side is not None:
            self._release()
            return True
        hit = click_hits_actor_map(viewer, pixel_x, pixel_y, self._apple_ids)
        if hit is None:
            return False
        self._pick(float(hit))
        return True

    def update(self, window):
        if self.held_side is not None:
            _set_apple_hover(self.env, self.held_side)
            self.poll_space(window)
            return
        if edge_pressed(window, "left", self._prev):
            for side in (getattr(self.env, "apples", {}) or {}):
                if float(side) < 0:
                    self._pick(float(side))
                    break
        if edge_pressed(window, "right", self._prev):
            for side in (getattr(self.env, "apples", {}) or {}):
                if float(side) > 0:
                    self._pick(float(side))
                    break


class _AppleSpacePlugin:
    """Poll Space during viewer.render so taps are not lost on 0-physics frames."""

    def __init__(self, controller: KeyboardAppleController):
        self.controller = controller
        self.viewer = None

    def init(self, viewer):
        self.viewer = viewer

    def notify_scene_change(self):
        pass

    def notify_selected_entity_change(self):
        pass

    def notify_window_focus_change(self, focused):
        pass

    def get_ui_windows(self):
        return []

    def before_render(self):
        pass

    def after_render(self):
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        self.controller.poll_space(window)

    def close(self):
        pass

    def clear_scene(self):
        pass


class ApplePinchMonitor:
    """Detach a hanging apple when the matching arm's gripper pinches it."""

    PINCH_DIST = 0.06

    def __init__(self, env):
        self.env = env
        self._gravity_at_step: dict[float, int] = {}
        self._announced: set[float] = set()

    def _still_attached(self, side: float) -> bool:
        if abs(float(side) - float(self.env.apple_side)) < 0.5:
            return bool(getattr(self.env, "_apple_attached", False))
        return bool(getattr(self.env, "_spoiled_attached", False))

    def update(self, step: int) -> None:
        env = self.env
        for side, at in list(self._gravity_at_step.items()):
            if step >= at:
                env._enable_held_apple_gravity(rigid=env._apple_rigids.get(side))
                del self._gravity_at_step[side]

        for side, apple in (getattr(env, "apples", {}) or {}).items():
            if not self._still_attached(side):
                continue
            arm_name = "left" if float(side) < 0 else "right"
            if gripper_width(env, arm_name) > 0.45:
                continue
            tcp = env._tcp_pos(arm_name)
            center = env._apple_grasp_center(apple)
            near = float(np.linalg.norm(tcp - center)) <= self.PINCH_DIST
            contacting = False
            try:
                contacting = len(
                    env.get_gripper_actor_contact_position(apple.get_name())
                ) > 0
            except Exception:
                pass
            if not (near or contacting):
                continue

            env._detach_apple(side=side)
            self._gravity_at_step[float(side)] = (
                step + int(getattr(env, "GRASP_SETTLE_STEPS", 25))
            )
            if float(side) in self._announced:
                continue
            self._announced.add(float(side))
            if abs(float(side) - float(env.apple_side)) < 0.5:
                tol = float(getattr(env, "red_tol", env.RED_TOLERANCE_DEFAULT))
                in_window = abs(float(env.ripeness) - float(env.red_window)) <= tol
                print(
                    f"Detached good apple at ripeness={env.ripeness:.3f} "
                    f"(window={env.red_window:.3f}±{tol:.3f}, "
                    f"{'in window' if in_window else 'OUTSIDE window'}). "
                    "Carry over the basket, then Space to open / release."
                )
            else:
                print(
                    f"Detached spoiled apple at ripeness={env.spoiled_ripeness:.3f} — "
                    "putting it in the basket fails. Space opens to release."
                )


def main():
    parser = argparse.ArgumentParser(description="Interactive pick_ripe_apple viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.pick_ripe_apple import pick_ripe_apple

    use_robot = is_robot_control(args.control)
    env = pick_ripe_apple()
    env.setup_demo(**configure_task(
        "pick_ripe_apple", args.config, args.seed, use_robot=use_robot,
    ))
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    env._ripen_started = True

    if use_robot:
        _move_arms_to_pre_grasp_orientation(env)
        print_banner(
            "pick_ripe_apple — interactive controls",
            [
                f"Mode: {args.control}  |  robot-motion: {args.robot_motion}",
                "Pinch the GOOD apple near peak red; drop in the basket.",
                "Space — close to pinch / open to release",
            ],
        )
    else:
        print_banner(
            "pick_ripe_apple — keyboard+mouse",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Left/Right or click an apple → hover over basket center",
                "Space or click the held apple again → release",
            ],
        )

    n_apples = len(getattr(env, "apples", {}) or {})
    print(
        f"Good side={'left' if env.apple_side < 0 else 'right'}  "
        f"apples={n_apples}"
        + (
            f"  spoiled_side={'left' if env.spoiled_side < 0 else 'right'}"
            if n_apples > 1 else ""
        )
        + f"  red_window={env.red_window:.3f}"
        f"±{float(getattr(env, 'red_tol', env.RED_TOLERANCE_DEFAULT)):.3f}"
    )

    pinch = ApplePinchMonitor(env) if use_robot else None
    clicker = None
    extra_plugins = None
    if not use_robot:
        viewer = env.viewer
        if viewer is None:
            raise SystemExit("Viewer was not created.")
        clicker = KeyboardAppleController(env, viewer)
        extra_plugins = [_AppleSpacePlugin(clicker)]
        viewer.register_click_handler(clicker.on_click)
        print_instructions(
            "Left/Right or click an apple to pick; Space or click again to release."
        )

    in_basket_since = None
    on_table_since = None

    def _keyboard_holding() -> bool:
        return clicker is not None and clicker.held_side is not None

    def on_step(window, step):
        nonlocal in_basket_since, on_table_since
        if pinch is not None:
            pinch.update(step)
        if clicker is not None:
            clicker.update(window)

        if getattr(env, "_apple_attached", False) and step % 150 == 0:
            tol = float(getattr(env, "red_tol", env.RED_TOLERANCE_DEFAULT))
            in_window = abs(float(env.ripeness) - float(env.red_window)) <= tol
            print(
                f"[ripeness] good={env.ripeness:.3f}  "
                f"window={env.red_window:.3f}±{tol:.3f}  "
                f"{'IN window' if in_window else 'outside window'}"
            )

        # Keyboard hold is kinematic hover — score only after the physical drop.
        if _keyboard_holding():
            in_basket_since = None
            on_table_since = None
            return

        basket_xy = env._basket_xy_now()
        good_in = bool(
            env.r_grasp is not None
            and env._pose_in_basket(env._apple_eval_p(env.apple), basket_xy)
        )
        spoiled = getattr(env, "spoiled_apple", None)
        spoiled_in = bool(
            spoiled is not None
            and env._pose_in_basket(env._apple_eval_p(spoiled), basket_xy)
        )
        if good_in or spoiled_in:
            if in_basket_since is None:
                in_basket_since = step
            on_table_since = None
        else:
            in_basket_since = None

        if env._good_apple_dropped_on_table():
            if on_table_since is None:
                on_table_since = step
        else:
            on_table_since = None

    def is_done(step):
        if _keyboard_holding():
            return False
        settle = max(1, int(getattr(env, "DROP_SETTLE_STEPS", 80)))
        # Let check_success own the printed reason (ripeness vs table miss).
        if in_basket_since is not None and step - in_basket_since >= settle:
            return True
        if on_table_since is not None and step - on_table_since >= settle:
            return True
        if (
            getattr(env, "_apple_attached", False)
            and float(env.ripeness) >= 0.95
        ):
            return True, "apple overripe (black) without a grasp"
        return False

    run_viewer_loop(
        env, on_step, is_done=is_done, max_steps=30000, extra_plugins=extra_plugins,
    )


if __name__ == "__main__":
    main()
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
