#!/usr/bin/env python3
"""Interactive, top-down default-mode viewer for ``sort_apples_belt``.

Run directly from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_sort_apples_belt.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_sort_apples_belt.py --control mouse
    /path/to/RoboDynaExp/interactive/base/interactive_sort_apples_belt.py --control robot

Keyboard mode changes the diverter directly. Robot mode uses 1/2/3 to select
gripper and arrows/E/Q to move it; lower with Q onto a button to divert
(hold both for dump). Mouse mode: click a red or green button
to toggle its routing direction. This is an interaction sandbox, not a
data-collection or robot-control rollout.
"""

import argparse
from datetime import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# This copy lives at <repo>/interactive/base/; resolve the repository independently
# of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    UniversalRobotControls,
    add_record_data_arg,
    declutter_interactive_viewer,
    escape_quit_requested,
    make_viewer_view_toggle,
    print_instructions,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_default_task(config_name: str, seed: int, use_robot: bool = False):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Preserve all task settings from the selected config (including its
    # ``rotten_prob``). The interactive launcher needs rendering enabled even
    # though batch demo collection uses ``render_freq: 0``.
    config.setdefault("task_name", "sort_apples_belt")
    config.setdefault("now_ep_num", 0)
    config.setdefault("need_plan", use_robot)
    config.setdefault("save_data", False)
    config["render_freq"] = 1
    config["seed"] = seed

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


def _update_keyboard_control(env, window):
    """Map arrow keys to left, right, or dump holds (both Left+Right = dump)."""

    left_down = window.key_down("left")
    right_down = window.key_down("right")
    env._expert_hold = (
        "dump" if left_down and right_down else
        "left" if left_down else
        "right" if right_down else None
    )


def _mouse_picture_xy(viewer):
    window = viewer.window
    mx, my = window.mouse_position
    ww, wh = window.size
    if ww <= 0 or wh <= 0 or mx < 0 or my < 0 or mx >= ww or my >= wh:
        return None
    tw, th = window.get_picture_size("Segmentation")
    return int(mx * tw / ww), int(my * th / wh)


def _button_hold_from_mouse(env, viewer):
    """While LMB held on a button, return that side; else None (no toggle)."""
    window = viewer.window
    if not bool(window.mouse_down(0)):
        return None
    pix = _mouse_picture_xy(viewer)
    if pix is None:
        return None
    button_ids = {
        int(button.actor.per_scene_id): side for side, button in env.buttons.items()
    }
    pixel = viewer.window.get_picture_pixel("Segmentation", pix[0], pix[1])
    return button_ids.get(int(pixel[1]))


def _update_keyboard_mouse_control(env, viewer):
    """Arrows or hold-click buttons; Left+Right together opens dump for spoiled apples."""
    window = viewer.window
    left_down = window.key_down("left")
    right_down = window.key_down("right")
    if left_down and right_down:
        env._expert_hold = "dump"
        return
    if left_down:
        env._expert_hold = "left"
        return
    if right_down:
        env._expert_hold = "right"
        return
    env._expert_hold = _button_hold_from_mouse(env, viewer)


class FastArmTeleop:
    """World-frame teleoperation for either ready-position gripper."""

    XY_STEP = 0.045
    Z_STEP = 0.030
    DURATION = 0.04

    def __init__(self, env):
        self.env = env
        self.selected = "left"
        self._start = None
        self._target = None
        self._started_at = None
        self._moving_side = None

    def _drive_qpos(self):
        joints = self.env.robot.left_arm_joints if self.selected == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_pose(self):
        getter = self.env.robot.get_left_ee_pose if self.selected == "left" else self.env.robot.get_right_ee_pose
        return np.asarray(getter(), dtype=np.float64)

    def _select_arm(self, window):
        selected = tuple(getattr(self.env, "_interactive_selected_arms", (self.selected,)))
        if len(selected) == 1:
            self.selected = selected[0]

    def _queue_move(self, dx, dy, dz):
        if self._started_at is not None:
            return
        pose = self._ee_pose().copy()
        pose[:3] += np.asarray([dx, dy, dz], dtype=np.float64)
        planner = self.env.robot.left_plan_path if self.selected == "left" else self.env.robot.right_plan_path
        result = planner(pose.tolist(), last_qpos=np.asarray(self._drive_qpos(), dtype=np.float32))
        if result is None or result.get("status") != "Success":
            return
        self._start = self._drive_qpos()
        self._target = np.asarray(result["position"][-1], dtype=np.float64)
        self._moving_side = self.selected
        self._started_at = time.perf_counter()

    def _advance(self):
        if self._started_at is None:
            return
        progress = min(1.0, (time.perf_counter() - self._started_at) / self.DURATION)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        delta = self._target - self._start
        position = self._start + delta * smooth
        velocity = delta / self.DURATION if progress < 1.0 else np.zeros_like(delta)
        self.env.robot.set_arm_joints(position, velocity, self._moving_side)
        if progress >= 1.0:
            self._started_at = None
            self._moving_side = None

    def update(self, window):
        self._select_arm(window)
        self._advance()


def _move_grippers_to_ready_position(env, arm_tag):
    """Park both closed grippers above their buttons without pressing either."""

    env._expert_hold = None
    env.plan_success = True
    env.move(
        env.grasp_actor(
            env.buttons["left"], arm_tag=arm_tag("left"), pre_grasp_dis=0.09,
            grasp_dis=0.09, contact_point_id=0, gripper_pos=0.0,
        ),
        env.grasp_actor(
            env.buttons["right"], arm_tag=arm_tag("right"), pre_grasp_dis=0.09,
            grasp_dis=0.09, contact_point_id=0, gripper_pos=0.0,
        ),
    )
    if not env.plan_success:
        detail = getattr(env, "_last_plan_fail", None)
        print(f"Could not reach the two-gripper ready position: {detail or 'unknown reason'}")
        env.plan_success = True
        env._last_plan_fail = None
    env._expert_hold = None


def _button_click_handler(env):
    """Return a segmentation-based handler that toggles a clicked task button."""

    button_ids = {button.actor.per_scene_id: side for side, button in env.buttons.items()}

    def handle_click(viewer, pixel_x, pixel_y):
        pixel = viewer.window.get_picture_pixel("Segmentation", pixel_x, pixel_y)
        side = button_ids.get(int(pixel[1]))
        if side is None:
            return False

        env._expert_hold = side if getattr(env, "_expert_hold", None) != side else None
        color = "red" if env._side_color[side] == env.COLOR_RED else "green"
        state = f"diverting to {color}" if env._expert_hold else "released (plank at rest)"
        print(f"Button click: {color}; {state}")
        return True

    return handle_click


def _start_recorder(output_path, frame, fps=30):
    """Start an MP4 encoder for RGB frames from the interactive viewer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    ffmpeg = Path(sys.executable).with_name("ffmpeg")
    if not ffmpeg.exists():
        raise SystemExit(f"FFmpeg was not found beside the active Python: {ffmpeg}")
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "-", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2,hflip,vflip",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264",
            "-crf", "20", str(output_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"Recording interactive demo to {output_path}")
    return process


def _viewer_rgb_frame(viewer):
    """Read the viewer Color buffer as a top-left-origin uint8 RGB frame."""

    rgba = viewer.window.get_picture("Color")
    return np.flipud((rgba[..., :3] * 255).clip(0, 255).astype(np.uint8))


def _close_recorder(process):
    if process is None:
        return
    process.stdin.close()
    if process.wait() != 0:
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        print(f"Warning: FFmpeg could not finalize the recording: {error or 'no error output'}")


def _disable_failed_recorder(process):
    """Report an early FFmpeg exit and disable recording without stopping the demo."""

    if process.poll() is None:
        return False
    error = process.stderr.read().decode("utf-8", errors="replace").strip()
    print(f"Recording disabled: FFmpeg exited early: {error or 'no error output'}")
    return True


def _add_second_view(env, primary_viewer):
    """Open a perspective viewer beside the overhead interactive viewer."""

    from sapien.utils.viewer import Viewer

    secondary = Viewer(env.renderer)
    secondary.set_scene(env.scene)
    declutter_interactive_viewer(secondary)
    # The standard simulator perspective, complementary to the overhead view.
    secondary.set_camera_xyz(0.477, 0.253, 1.625)
    secondary.set_camera_rpy(0.0, -0.8, 2.45)

    original_render = primary_viewer.render

    def render_both(*args, **kwargs):
        # Re-apply both poses because the viewer control plugin can otherwise
        # synchronize camera state after a window receives focus.
        primary_viewer.set_camera_xyz(0.08, 0.0, 1.68)
        primary_viewer.set_camera_rpy(0.0, -np.pi / 2.0, -np.pi / 2.0)
        result = original_render(*args, **kwargs)
        if not secondary.closed:
            secondary.set_camera_xyz(0.477, 0.253, 1.625)
            secondary.set_camera_rpy(0.0, -0.8, 2.45)
            secondary.render()
        return result

    # Base-task motions call primary_viewer.render internally; wrapping it keeps
    # the second window live while a planned robot action is executing.
    primary_viewer.render = render_both

    def restore():
        if primary_viewer.render is render_both:
            primary_viewer.render = original_render
        if not secondary.closed:
            secondary.close()

    return secondary, restore


def _camera_pose(sapien, position, look_at, up_hint):
    forward = np.asarray(look_at, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray(up_hint, dtype=np.float64), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, up], axis=1)
    matrix[:3, 3] = position
    return sapien.Pose(matrix)


class CompositeView:
    """One GUI window containing overhead and robot-head camera renders."""

    WINDOW_NAME = "RoboDyna: Overhead | Head camera"
    DISPLAY_FPS = 20

    def __init__(self, env, sapien):
        self.env = env
        self.window_open = True
        self.action = None
        self.overhead = env.scene.add_camera("interactive_overhead", 640, 480, np.deg2rad(55), 0.1, 10.0)
        self.overhead.entity.set_pose(_camera_pose(
            # Reverse the up axis so the overhead image is rotated 180 degrees:
            # the robot is at the bottom, matching the interactive GUI view.
            sapien, [0.08, 0.0, 1.68], [0.0, 0.0, 0.75], [0.0, -1.0, 0.0],
        ))
        self.head = next(
            (camera for camera, name in zip(env.cameras.static_camera_list, env.cameras.static_camera_name)
             if name == "head_camera"),
            None,
        )
        if self.head is None:
            raise RuntimeError("This task configuration does not provide a head_camera.")
        self._next_refresh = 0.0
        self._last_frame = None
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    @staticmethod
    def _rgb(camera):
        camera.take_picture()
        rgba = camera.get_picture("Color")
        return np.flipud((rgba[..., :3] * 255).clip(0, 255).astype(np.uint8))

    def needs_refresh(self):
        return self._last_frame is None or time.perf_counter() >= self._next_refresh

    def render(self, refresh):
        if refresh:
            left = self._rgb(self.overhead)
            right = self._rgb(self.head)
            # Match the overhead panel's orientation without changing the
            # task-owned head camera pose used by observations.
            right = np.rot90(right, 2)
            # The configured head camera is 320x240, while the overhead panel is
            # 640x480. Resize only for display so the two panels align cleanly.
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            composite = np.concatenate((left, right), axis=1)
            cv2.putText(composite, "Overhead", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(composite, "Head camera", (658, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self._last_frame = composite
            self._next_refresh = time.perf_counter() + 1.0 / self.DISPLAY_FPS
            cv2.imshow(self.WINDOW_NAME, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.window_open = False
        elif key == 81:
            # The displayed overhead view is rotated 180 degrees, so screen
            # left maps to the task's right-side actuator (and vice versa).
            self.action = None if self.action == "right" else "right"
        elif key == 83:
            self.action = None if self.action == "left" else "left"
        elif key == 84:
            self.action = None if self.action == "dump" else "dump"
        return self._last_frame

    def close(self):
        cv2.destroyWindow(self.WINDOW_NAME)
        self.env.scene.remove_camera(self.overhead)


def _attach_composite_motion_renderer(viewer, composite_view):
    """Keep the composite window responsive during blocking robot motions."""

    original_render = viewer.render

    def render_composite(*_args, **_kwargs):
        # Robot trajectories call ``viewer.render`` from their internal
        # blocking loop.  The stock viewer is hidden in composite mode, so use
        # that callback to refresh the visible OpenCV window instead.
        composite_view.render(composite_view.needs_refresh())

    viewer.render = render_composite

    def restore():
        if viewer.render is render_composite:
            viewer.render = original_render

    return restore


def main():
    parser = argparse.ArgumentParser(description="Interactive top-down sort-apples viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        metavar="PATH",
        help="Record the overhead viewer to an MP4; omit PATH for a timestamped file under demos/interactive_sort_apples_belt.",
    )
    parser.add_argument(
        "--two-views",
        action="store_true",
        help="Open a second live perspective viewer beside the overhead interaction view.",
    )
    parser.add_argument(
        "--composite-view",
        action="store_true",
        help="Show overhead and robot-head cameras side by side in one OpenCV window.",
    )
    parser.add_argument(
        "--control",
        choices=("keyboard", "keyboard+mouse", "mouse", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="interpolate",
        help="Robot key-press implementation; interpolate is a faster test mode (default: interpolate)",
    )
    add_record_data_arg(parser)
    args = parser.parse_args()
    if args.two_views and args.composite_view:
        raise SystemExit("Choose either --two-views or --composite-view.")
    if args.record == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.record = REPO_ROOT / "demos" / "interactive_sort_apples_belt" / f"sort_apples_belt_{stamp}.mp4"
    elif args.record is not None:
        args.record = Path(args.record).expanduser().resolve()

    # Keep ``--help`` usable on machines without the GPU-only simulator stack.
    from envs import CONFIGS_PATH
    from envs.sort_apples_belt import sort_apples_belt
    from envs.utils.action import ArmTag
    import sapien
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    env = sort_apples_belt()
    use_robot = args.control == "robot"
    if use_robot:
        env._interactive_robot_mode = True
    env.setup_demo(**_configure_default_task(
        args.config, args.seed, use_robot=use_robot
    ))
    if args.control in ("keyboard", "keyboard+mouse"):
        from _interactive_common import prepare_interactive_control
        prepare_interactive_control(env, args.control)
        env._interactive_universal_controls = True
    print_episode_condition(env)
    if use_robot:
        env.together_close_gripper(save_freq=None)
        _move_grippers_to_ready_position(env, ArmTag)
    belt_clear_since = None
    env._interactive_space_click = False
    recorder = None
    record_frame_count = 0
    record_every = max(1, round(1.0 / (env.scene.get_timestep() * 30.0)))
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")

    # A fixed overhead camera with the robot at the bottom keeps its controls in view.
    views = make_viewer_view_toggle(env, viewer)
    if args.control == "robot" and views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)
    composite_view = None
    restore_composite_motion_renderer = lambda: None
    if args.composite_view:
        composite_view = CompositeView(env, sapien)
        viewer.window.hide()
        print("Composite view ready at 20 FPS. Click an action to divert; Escape quits cleanly.")
    second_viewer = None
    restore_second_view = lambda: None
    if args.two_views:
        second_viewer, restore_second_view = _add_second_view(env, viewer)
        print("Two-view mode ready: overhead interaction view + perspective view.")
    if args.control == "mouse":
        viewer.register_click_handler(_button_click_handler(env))
        print_instructions(
            "Top-down sort-apples sandbox ready. Click red/green to toggle the plank direction."
        )
    elif args.control == "robot":
        print_mode_controls(
            "sort_apples_belt",
            "robot",
            keyboard="",
            robot=(
                "\n  Lower with Q onto a button to divert (both arms = dump).\n"
            ),
        )
    else:
        print_mode_controls(
            "sort_apples_belt",
            "keyboard",
            keyboard=(
                "\n"
                "  Left Arrow        hold to divert left\n"
                "  Right Arrow       hold to divert right\n"
                "  Left + Right      hold dump (spoiled apple hatch)\n"
                "  Mouse             hold click on a button (releases when you let go)\n"
                "\n"
                "  Release to let the plank return to rest.\n"
            ),
            robot="",
        )

    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)
    try:
        while (not viewer.closed and (second_viewer is None or not second_viewer.closed)
               and (composite_view is None or composite_view.window_open)):
            n_steps = pacer.begin_frame()
            if composite_view is None:
                views.update(viewer.window)
            if composite_view is not None:
                # Composite mode has no SAPIEN key window; retain its click action.
                env._expert_hold = composite_view.action
            elif args.control == "keyboard+mouse":
                _update_keyboard_mouse_control(env, viewer)
            elif args.control == "keyboard":
                _update_keyboard_control(env, viewer.window)

            if n_steps == 0:
                if composite_view is not None:
                    refresh_composite = composite_view.needs_refresh()
                    if refresh_composite:
                        env.scene.update_render()
                    composite_view.render(refresh_composite)
                else:
                    env.scene.update_render()
                    viewer.render()
                if composite_view is None and escape_quit_requested(env, viewer.window):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            # Robot mode: gripper-Z reactive held_mask drives the diverter (no dedicated latch key).
            for _ in range(n_steps):
                env._update_kinematic_tasks()
                env.scene.step()
            if composite_view is not None:
                refresh_composite = composite_view.needs_refresh()
                if refresh_composite:
                    env.scene.update_render()
                composite_frame = composite_view.render(refresh_composite)
            else:
                env.scene.update_render()
                viewer.render()
                composite_frame = None
            # SAPIEN keeps the window open by default; terminate the launcher
            # explicitly so Escape works consistently across viewer backends.
            if composite_view is None and escape_quit_requested(env, viewer.window):
                break
            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue
            if args.record is not None and record_frame_count % record_every == 0:
                frame = composite_frame if composite_frame is not None else _viewer_rgb_frame(viewer)
                recorder = recorder or _start_recorder(args.record, frame)
                if _disable_failed_recorder(recorder):
                    recorder = None
                    args.record = None
                else:
                    try:
                        recorder.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        # FFmpeg may fail between the poll and the write.
                        _disable_failed_recorder(recorder)
                        recorder = None
                        args.record = None
            record_frame_count += 1
            timed_out = (
                bool(getattr(env, "_timed_out", False))
                or bool(getattr(env, "_episode_timed_out", False))
                or (hasattr(env, "_budget_exhausted") and bool(env._budget_exhausted()))
            )
            if timed_out:
                report_task_result(env, "timed_out")
                terminal_started_at = time.perf_counter()
                continue
            all_spawned = env._spawned >= env.n_apples
            belt_clear = all(not env._apple_on_belt(i) for i in range(env._spawned))
            if all_spawned and belt_clear:
                if belt_clear_since is None:
                    belt_clear_since = time.perf_counter()
                    print("All apples have cleared the belt; stopping in two seconds.")
                elif time.perf_counter() - belt_clear_since >= 2.0:
                    correct = sum(bool(result) for result in env.results)
                    report_task_result(
                        env, f"{correct}/{env.n_apples} apples sorted correctly"
                    )
                    terminal_started_at = time.perf_counter()
                    continue
            else:
                belt_clear_since = None
    finally:
        try:
            _close_recorder(recorder)
        finally:
            restore_second_view()
            restore_composite_motion_renderer()
            if composite_view is not None:
                composite_view.close()
            env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
