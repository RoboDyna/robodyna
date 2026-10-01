#!/usr/bin/env python3
"""Standard head-view demo recorder for RoboDyna / DOMINO tasks.

Records ONE successful expert rollout and writes a single MP4:
  - robot head  -> ``head_camera``  (elevated training cam)

Output layout (under ``tmp/``, versioned, never overwrites prior demos)::

    ./tmp/tmp_<task>/video/vN_head.mp4

Usage (from repo root, with the robodyna / domino env active)::

    python script/bench_script/record_demo.py place_block_belt
    python script/bench_script/record_demo.py place_block_belt --config demo_dynamic
    python script/bench_script/record_demo.py catch_cuboid --task-arg catch_two_cuboids=true
    python script/bench_script/record_demo.py catch_cuboid --task-arg opaque_surface=true
    # legacy shorthand still works: --option 1 (catch_two_cuboids) / --option 2 (opaque_surface)

Agents MUST use this script for task demo videos (not ad-hoc record_*_demo.py
copies, and not the training collector preview).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import numpy as np
import sapien
import yaml

from envs import CONFIGS_PATH
from envs.utils.household_view import (
    HOUSEHOLD_MAX_SECONDS,
    HOUSEHOLD_MAX_STEPS,
    HOUSEHOLD_TASKS,
    configure_household_head_camera,
)
from envs.utils.images_to_video import images_to_video
from envs.utils.pkl2hdf5 import load_pkl_file
from script.collect_data import class_decorator, get_embodiment_config, run


class DemoRenderTimeout(Exception):
    """Raised when a failure-demo render hits the step / wall-clock cap."""

# Bird's-eye framing: above table center, looking straight down.
TOPDOWN_POS = np.array([0.0, 0.05, 1.85], dtype=np.float64)
TOPDOWN_LOOK_AT = np.array([0.0, 0.05, 0.78], dtype=np.float64)
TOPDOWN_FOVY_DEG = 70.0
TOPDOWN_WH = (640, 480)  # (w, h)


def next_version(video_dir: str) -> int:
    os.makedirs(video_dir, exist_ok=True)
    nums = []
    for p in glob.glob(os.path.join(video_dir, "v*_head.mp4")):
        m = re.search(r"v(\d+)_", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    # Also accept legacy vN_episode0.mp4 so versioning stays monotonic.
    for p in glob.glob(os.path.join(video_dir, "v*_episode0.mp4")):
        m = re.search(r"v(\d+)_", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def _encode_mp4(frames: list[np.ndarray], out_mp4: str, fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames to encode for {out_mp4}")
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    stack = np.stack(frames, axis=0)
    if stack.dtype != np.uint8:
        stack = (stack * 255).clip(0, 255).astype(np.uint8)
    n_frames, H, W, _C = stack.shape
    # libx264 yuv420p needs even dims.
    if H % 2 or W % 2:
        import cv2

        H2, W2 = H - (H % 2), W - (W % 2)
        stack = np.stack(
            [cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA) for f in stack],
            axis=0,
        )
        n_frames, H, W, _C = stack.shape
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}", "-framerate", str(fps),
            "-i", "-",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264",
            "-crf", "17", "-preset", "slow",
            out_mp4,
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(stack.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        images_to_video(stack, out_path=out_mp4, fps=fps, is_rgb=True)
    else:
        print(f"Wrote {out_mp4} ({n_frames} frames at {W}x{H}, crf=17)")


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if h == height:
        out = frame
    else:
        w2 = int(round(w * (height / h)))
        out = cv2.resize(frame, (w2, height), interpolation=cv2.INTER_AREA)
    if out.shape[1] % 2:
        out = out[:, :-1]
    return out


def encode_head_video(cache_dir: str, out_head: str, fps: float) -> None:
    """Encode ``head_camera`` frames from a pkl cache into one MP4."""
    pkls = sorted(
        glob.glob(os.path.join(cache_dir, "*.pkl")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    head_frames = []
    for p in pkls:
        data = load_pkl_file(p)
        obs = data.get("observation", {})
        if "head_camera" in obs and "rgb" in obs["head_camera"]:
            head_frames.append(np.asarray(obs["head_camera"]["rgb"]))
    if not head_frames:
        raise RuntimeError(f"No head_camera frames in {cache_dir}")
    _encode_mp4(head_frames, out_head, fps)


def merge_dual_view_videos(cache_dir: str, out_head: str, out_topdown: str, out_side: str, fps: float) -> None:
    """Legacy dual-view encoder kept for older per-task export scripts."""
    pkls = sorted(
        glob.glob(os.path.join(cache_dir, "*.pkl")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    head_frames, top_frames, side_frames = [], [], []
    for p in pkls:
        data = load_pkl_file(p)
        obs = data.get("observation", {})
        head = None
        if "head_camera" in obs and "rgb" in obs["head_camera"]:
            head = np.asarray(obs["head_camera"]["rgb"])
        top = data.get("third_view_rgb", None)
        if top is not None:
            top = np.asarray(top)
        if head is None and top is None:
            continue
        if head is not None:
            head_frames.append(head)
        if top is not None:
            top_frames.append(top)
        if head is not None and top is not None:
            H = max(head.shape[0], top.shape[0])
            H += H % 2
            side_frames.append(np.hstack([_resize_to_height(head, H), _resize_to_height(top, H)]))

    if not head_frames:
        raise RuntimeError(f"No head_camera frames in {cache_dir}")
    if not top_frames:
        raise RuntimeError(
            f"No top-down (third_view_rgb) frames in {cache_dir}. "
            "Ensure data_type.third_view=true and the observer was reframed."
        )
    _encode_mp4(head_frames, out_head, fps)
    _encode_mp4(top_frames, out_topdown, fps)
    if side_frames:
        _encode_mp4(side_frames, out_side, fps)


def configure_topdown_camera(task) -> None:
    """Replace observer_camera with a high-res bird's-eye (true top-down) view."""
    cams = getattr(task, "cameras", None)
    if cams is None or getattr(cams, "observer_camera", None) is None:
        return
    scene = getattr(task, "scene", None)
    if scene is None:
        return

    old = cams.observer_camera
    near = float(getattr(old, "near", 0.1))
    far = float(getattr(old, "far", 100.0))
    try:
        scene.remove_camera(old)
    except Exception:
        pass

    w, h = TOPDOWN_WH
    camera = scene.add_camera(
        name="observer_camera",
        width=w,
        height=h,
        fovy=np.deg2rad(TOPDOWN_FOVY_DEG),
        near=near,
        far=far,
    )
    cams.observer_camera = camera

    forward = TOPDOWN_LOOK_AT - TOPDOWN_POS
    forward = forward / np.linalg.norm(forward)
    # Prefer world +y as the image "up" hint so the table's near edge is consistent.
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    left = np.cross(world_up, forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    m = np.eye(4)
    m[:3, :3] = np.stack([forward, left, up], axis=1)
    m[:3, 3] = TOPDOWN_POS
    camera.entity.set_pose(sapien.Pose(m))


def _parse_task_arg_value(raw: str):
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def save_freq_for_fps(fps: float) -> int:
    """Map target video Hz to sim ``save_freq`` (fps ≈ 250 / save_freq)."""
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return max(1, int(round(250.0 / fps)))


def build_args(
    task_name: str,
    config_name: str,
    save_root: str,
    option: str | None,
    task_arg_overrides: list[str] | None = None,
    fps: float | None = None,
) -> dict:
    with open(f"./task_config/{config_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = config_name
    args["episode_num"] = 1
    args["save_path"] = save_root  # flat; call run() directly (not main())
    args["collect_data"] = True
    args["eval_video_log"] = False
    args["save_failed_cases"] = False
    args["use_seed"] = False
    args["check_render_success"] = True
    args["export_lerobot"] = False
    # Match collect_data / policy rate: ~30 Hz → save_freq=8 (fps ≈ 250/save_freq).
    # yaml demo_dynamic is 15 (~16.7 Hz); policy data is in the 30 fps range.
    if fps is not None:
        args["save_freq"] = save_freq_for_fps(fps)
    else:
        args["save_freq"] = save_freq_for_fps(30.0)

    args.setdefault("camera", {})
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = False  # demos don't need wrists; faster render
    # Same head sensor as collect_data (demo_dynamic.yml).
    args["camera"]["head_camera_type"] = "D435"

    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    # Default keeps third_view for legacy dual-view export scripts. The
    # standard record_demo / record_fail_demo paths turn it off (head-only).
    args["data_type"]["third_view"] = True

    args.setdefault("task_args", {}).setdefault(task_name, {})
    if option is not None:
        args["task_args"][task_name]["option"] = option
    for item in task_arg_overrides or []:
        if "=" not in item:
            raise ValueError(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        args["task_args"][task_name][key.strip()] = _parse_task_arg_value(raw.strip())

    # Failure demos (e.g. force_platform_miss stick) still need a rendered trajectory.
    targs = args["task_args"][task_name]
    if (
        bool(targs.get("force_platform_miss", False))
        or bool(targs.get("allow_fail", False))
        or bool(targs.get("force_overflow", False))
        or bool(targs.get("force_spill", False))
    ):
        args["save_failed_cases"] = True
        args["check_render_success"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return emb[t]["file_path"]

    et = args["embodiment"]
    if len(et) == 1:
        args["left_robot_file"] = emb_file(et[0])
        args["right_robot_file"] = emb_file(et[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(et[0])
    else:
        args["left_robot_file"] = emb_file(et[0])
        args["right_robot_file"] = emb_file(et[1])
        args["embodiment_dis"] = et[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def _cleanup_scratch(save_root: str, task_name: str) -> None:
    for junk in (
        "data",
        ".cache",
        "seed.txt",
        "scene_info.json",
        task_name,
        "_traj_data",
        "seed_archive.txt",
        "video/episode0.mp4",
    ):
        p = os.path.join(save_root, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)


def record_fail_demo(
    task_name: str,
    seed: int,
    config_name: str = "demo_dynamic",
    task_arg_overrides: list[str] | None = None,
    tag: str | None = None,
    max_seconds: float | None = None,
    max_steps: int | None = None,
    dest_dir: str | None = None,
    fps: float | None = None,
) -> dict:
    """Record a failure demo with a step (default) or wall-clock render cap.

    Plans once (saving the traj even when check_success fails), then renders
    until ``max_steps`` saved frames (or ``max_seconds`` if set). On timeout,
    merges whatever frames are in the cache and treats the clip as a failure demo.
    """
    if max_steps is None and max_seconds is None:
        max_steps = int(HOUSEHOLD_MAX_STEPS)
    save_root = os.path.abspath(f"./tmp/tmp_{task_name}_failrec")
    video_dir = os.path.join(save_root, "video")
    os.makedirs(video_dir, exist_ok=True)
    _cleanup_scratch(save_root, task_name)
    os.makedirs(save_root, exist_ok=True)

    overrides = list(task_arg_overrides or [])
    if not any(o.startswith("allow_fail=") for o in overrides):
        overrides.append("allow_fail=true")

    args = build_args(
        task_name, config_name, save_root, None, overrides, fps=fps
    )
    args["save_failed_cases"] = True
    args["check_render_success"] = False
    args["episode_num"] = 1
    args["collect_data"] = True
    args["use_seed"] = False
    args.setdefault("data_type", {})["third_view"] = False  # head-only

    ver = next_version(video_dir)
    stem = f"v{ver}_{tag}" if tag else f"v{ver}_fail"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")

    task = class_decorator(task_name)
    _orig_setup = task.setup_demo

    def _setup_demo(**kwags):
        kwags["seed"] = int(seed)
        _orig_setup(**kwags)
        if task_name in HOUSEHOLD_TASKS:
            configure_household_head_camera(task)

    task.setup_demo = _setup_demo

    # ----- plan pass (no RGB) -----
    plan_args = dict(args)
    plan_args.update(
        collect_data=False,
        save_data=False,
        eval_video_log=False,
        need_plan=True,
        render_freq=0,
    )
    print(f"[fail-demo] plan seed={seed} tag={tag}", flush=True)
    task.setup_demo(now_ep_num=0, seed=int(seed), **plan_args)
    task.play_once()
    plan_ok = bool(task.plan_success and task.check_success())
    print(
        f"[fail-demo] plan done plan_success={task.plan_success} "
        f"check_success={task.check_success()} ok={plan_ok}",
        flush=True,
    )
    task.save_traj_data(0)
    try:
        task.close_env()
    except Exception:
        pass

    # ----- render pass with step / wall-clock cap -----
    from envs.utils.household_view import EpisodeTimeLimit

    render_args = deepcopy(args)
    render_args.update(need_plan=False, render_freq=0, save_data=True)
    task.setup_demo(now_ep_num=0, seed=int(seed), **render_args)
    task._episode_timed_out = False
    if max_steps is not None:
        task._max_episode_steps = int(max_steps)
        task._max_episode_seconds = None
    elif max_seconds is not None:
        task._max_episode_steps = None
        task._max_episode_seconds = float(max_seconds)
        task._episode_t0 = time.monotonic()
    traj = task.load_tran_data(0)
    render_args["left_joint_path"] = traj["left_joint_path"]
    render_args["right_joint_path"] = traj["right_joint_path"]
    task.set_path_lst(render_args)

    t0 = time.monotonic()
    _orig_take = task._take_picture
    timed_out = False

    def _take_picture_capped():
        nonlocal timed_out
        if max_steps is not None and int(getattr(task, "FRAME_IDX", 0) or 0) >= int(max_steps):
            timed_out = True
            raise DemoRenderTimeout(
                f"render hit {int(max_steps)}-step cap "
                f"(frames={getattr(task, 'FRAME_IDX', '?')})"
            )
        if max_seconds is not None and time.monotonic() - t0 >= float(max_seconds):
            timed_out = True
            raise DemoRenderTimeout(
                f"render hit {max_seconds:.0f}s wall-clock cap "
                f"(frames={getattr(task, 'FRAME_IDX', '?')})"
            )
        return _orig_take()

    task._take_picture = _take_picture_capped
    print(
        f"[fail-demo] render seed={seed} max_steps={max_steps} max_seconds={max_seconds}",
        flush=True,
    )
    try:
        task.play_once()
    except (DemoRenderTimeout, EpisodeTimeLimit) as e:
        timed_out = True
        print(f"[fail-demo] {e} — saving partial video as failure", flush=True)
    finally:
        task._take_picture = _orig_take

    cache_path = f"{task.save_dir}/.cache/episode{task.ep_num}/"
    fps = 250.0 / float(task.save_freq) if task.save_freq else 15.0
    n_frames = len(glob.glob(os.path.join(cache_path, "*.pkl")))
    print(f"[fail-demo] merge {n_frames} frames from {cache_path}", flush=True)
    if n_frames == 0:
        raise RuntimeError(f"No frames saved before timeout for {task_name} seed={seed}")
    encode_head_video(cache_path, out_head, fps=fps)

    try:
        task.close_env(clear_cache=True)
    except Exception:
        pass
    try:
        task.remove_data_cache()
    except Exception:
        pass

    copied = {}
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
        if os.path.isfile(out_head):
            name = f"{tag or 'fail'}_seed{seed}_head.mp4"
            dst = os.path.join(dest_dir, name)
            shutil.copy2(out_head, dst)
            copied["head"] = dst
            print(f"[fail-demo] COPIED {dst}", flush=True)

    _cleanup_scratch(save_root, task_name)
    print(
        f"[fail-demo] ready timed_out={timed_out} plan_ok={plan_ok} "
        f"frames~{n_frames}",
        flush=True,
    )
    return {
        "version": ver,
        "tag": tag,
        "seed": seed,
        "timed_out": timed_out,
        "plan_ok": plan_ok,
        "head": out_head,
        "copied": copied,
        "n_frames": n_frames,
    }


def record_demo(
    task_name: str,
    config_name: str = "demo_dynamic",
    option: str | None = None,
    task_arg_overrides: list[str] | None = None,
    tag: str | None = None,
    max_seconds: float | None = None,
    max_steps: int | None = None,
    fps: float | None = None,
) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{task_name}")
    video_dir = os.path.join(save_root, "video")
    ver = next_version(video_dir)
    # Clear condition tags go in the filename, e.g. v5_opt2_blocker_head.mp4
    stem = f"v{ver}_{tag}" if tag else f"v{ver}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")

    _cleanup_scratch(save_root, task_name)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(
        task_name, config_name, save_root, option, task_arg_overrides, fps=fps
    )
    print(
        f"[record_demo] save_freq={args['save_freq']} "
        f"(~{250.0 / float(args['save_freq']):.1f} Hz)",
        flush=True,
    )
    args.setdefault("data_type", {})["third_view"] = False  # head-only
    # Household demos: keep a partial render if the step cutoff fires.
    if task_name in HOUSEHOLD_TASKS:
        args["check_render_success"] = False
        if max_steps is None and max_seconds is None:
            max_steps = int(HOUSEHOLD_MAX_STEPS)
    task = class_decorator(task_name)

    # Shared household head framing after every setup_demo.
    _orig_setup = task.setup_demo

    def _setup_demo(**kwags):
        _orig_setup(**kwags)
        if task_name in HOUSEHOLD_TASKS:
            configure_household_head_camera(task)
            task._episode_timed_out = False
            if max_steps is not None:
                task._max_episode_steps = int(max_steps)
                task._max_episode_seconds = None
            elif max_seconds is not None:
                task._max_episode_steps = None
                task._max_episode_seconds = float(max_seconds)
                task._episode_t0 = time.monotonic()

    task.setup_demo = _setup_demo

    def _merge_head(self):
        if not self.save_data:
            return
        cache_path = f"{self.save_dir}/.cache/episode{self.ep_num}/"
        fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
        encode_head_video(cache_path, out_head, fps=fps)

    task.merge_pkl_to_hdf5_video = _merge_head.__get__(task, task.__class__)

    run(task, args)
    _cleanup_scratch(save_root, task_name)

    # Drop leftover empty dirs / conventional episode mp4 if merge didn't write one.
    for junk in (".cache", "data", "seed.txt", "scene_info.json", "_traj_data"):
        p = os.path.join(save_root, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)

    print("Demo ready:")
    if tag:
        print("  tag      :", tag)
    print("  head     :", out_head)
    return {
        "version": ver,
        "tag": tag,
        "head": out_head,
    }


def main():
    parser = argparse.ArgumentParser(description="Record standard head-view demo videos.")
    parser.add_argument("task", help="Task name (envs/<task>.py class name), e.g. place_block_belt")
    parser.add_argument(
        "--config",
        default="demo_dynamic",
        help="task_config/<config>.yml to load (default: demo_dynamic)",
    )
    parser.add_argument(
        "--option",
        default=None,
        help="Legacy shorthand for task_args.<task>.option "
             "(catch_cuboid: 1/catch_two_cuboids, 2/opaque_surface; "
             "save_goal: 1/players_enabled, 2/cover_enabled; "
             "punch_dual_holes: 1/missing_tile_mode, 2/belt_continous_motion; "
             "catch_valley_ball: 1/wall_bounce_enabled, 2/enable_distractor; "
             "catch_ramp_ball: 1/wall_bounce_enabled, 2/enable_distractor; "
             "play_billiard: 1/specific_hole, 2/enable_distractors; "
             "cook_meat: 1/cook_button_enabled [default on], 2/dual_setup_enabled). "
             "Prefer --task-arg.",
    )
    parser.add_argument(
        "--task-arg",
        action="append",
        default=[],
        help="Override task_args.<task> entry, e.g. --task-arg blocker_enabled=true",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Condition label embedded in output names, e.g. opt2_blocker "
             "→ vN_opt2_blocker_head.mp4",
    )
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Record a failure demo (saves even when check_success fails).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Force episode seed (required with --fail).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Saved-frame / simulation-step render cap. Household tasks "
             f"default to {HOUSEHOLD_MAX_STEPS}. On timeout, merge whatever "
             "frames exist (fail demos, or success demos with "
             "check_render_success disabled).",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional wall-clock render cap in seconds (overrides step cap "
             f"when set without --max-steps). Legacy; prefer --max-steps.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target head-video frame rate in Hz. Sets save_freq via 250/fps. "
             "Default 30 (collect_data / policy rate) → save_freq=8.",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Optional copy destination dir (e.g. rand_demo/<task>/failure).",
    )
    ns = parser.parse_args()
    # Alias: users sometimes say place_block_task
    task = ns.task
    if task in ("place_block_task", "place_block"):
        print(f"Note: '{task}' -> place_block_belt")
        task = "place_block_belt"
    max_steps = ns.max_steps
    max_seconds = ns.max_seconds
    if max_steps is None and max_seconds is None and task in HOUSEHOLD_TASKS:
        max_steps = int(HOUSEHOLD_MAX_STEPS)
    if ns.fail:
        if ns.seed is None:
            parser.error("--fail requires --seed")
        record_fail_demo(
            task,
            seed=ns.seed,
            config_name=ns.config,
            task_arg_overrides=ns.task_arg,
            tag=ns.tag or "fail",
            max_steps=max_steps,
            max_seconds=max_seconds,
            dest_dir=ns.dest,
            fps=ns.fps,
        )
    else:
        record_demo(
            task,
            config_name=ns.config,
            option=ns.option,
            task_arg_overrides=ns.task_arg,
            tag=ns.tag,
            max_steps=max_steps,
            max_seconds=max_seconds,
            fps=ns.fps,
        )


if __name__ == "__main__":
    main()
