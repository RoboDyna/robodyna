"""
Simple visualization script for benchmark task scenes.
Uses environments from bench_envs and the same config layout as collect_data.

USAGE:
    Run this script from the benchmark folder:

    cd benchmark
    source set_env.sh  # or set ROBOTWIN_ROOT and BENCH_ROOT manually
    python bench_script/visualize_task_scene.py <task_name> <task_config> [options]

EXAMPLES:
    # Basic usage with default seed
    python bench_script/visualize_task_scene.py cook_meat bench_demo_clean

    # With custom seed
    python bench_script/visualize_task_scene.py cook_meat bench_demo_clean --seed 42

    # With custom render frequency
    python bench_script/visualize_task_scene.py cook_meat bench_demo_clean --render-freq 5

    # Viewer aligned to a scene camera (see embodiment static_camera_list + bench office_config)
    python bench_script/visualize_task_scene.py put_bottle_in_fridge bench_demo_clean --viewer-camera demo_camera

    # Roll out the task (run play_once) then view
    python bench_script/visualize_task_scene.py put_away_stapler bench_demo_clean --rollout

    # Roll out with the previous viewer behavior (mostly arm motion + final frame)
    python bench_script/visualize_task_scene.py put_away_stapler bench_demo_clean --rollout \\
        --render-mode legacy

    # Slow the visualization to quarter speed
    python bench_script/visualize_task_scene.py put_away_stapler bench_demo_clean --rollout \\
        --speed 0.25

    # With explicit bench subdir (for tasks under office/, study/, etc.)
    python bench_script/visualize_task_scene.py mouse_on_pad bench_demo_clean --bench-subdir office --rollout --seed 0

    # Run headless (no GUI)
    python bench_script/visualize_task_scene.py cook_meat bench_demo_clean --no-render
    python bench_script/visualize_task_scene.py cook_meat bench_demo_clean --no-render --rollout

    # Save RGB on first planning failure (move_step = Nth top-level env.move in play_once)
    python bench_script/visualize_task_scene.py move_books_onto_table bench_demo_clean --bench-subdir study \\
        --rollout --save-plan-fail-dir ./plan_fail_debug --plan-fail-camera head_camera

ARGUMENTS:
    task_name      Task module name from bench_envs (e.g. cook_meat)
    task_config    Task config name without .yml extension (e.g. bench_demo_clean)

OPTIONS:
    --seed N                   Random seed for scene initialization (default: 0)
    --render-freq N            Render every N simulation steps (default: 1)
    --speed S                  Visualization speed multiplier; 1.0 is realtime, lower is slower
    --viewer-camera N          Match viewer to this scene camera name (default: demo_camera; use default to keep built-in pose)
    --rollout                  Run play_once() to roll out the task; if not set, only view initial setup
    --render-mode M            Choose live or legacy viewer updates during rollout
    --bench-subdir S           Subdirectory under bench_envs (e.g. office, study). If set, imports bench_envs.<S>.<task_name>
    --no-render                Disable simulator rendering (run headless). Default: render on.
    --save-plan-fail-dir PATH  With --rollout, save camera RGB when a top-level env.move() fails planning
    --plan-fail-camera NAME    Camera key for save_camera_rgb (default: head_camera)

NOTES:
    - The script automatically changes directory to customized_robotwin for proper path resolution
    - Requires ROBOTWIN_ROOT and BENCH_ROOT environment variables (set via set_env.sh)
    - Close the viewer window to exit the visualization
"""
import sys
import os
import argparse
import importlib
import time
import yaml
from pathlib import Path
import numpy as np
import cv2
import json
sys.path.append("./")

# from setup_paths import setup_paths
# setup_paths()
# Paths: script is run from benchmark folder, but changes to customized_robotwin
current_file_path = os.path.abspath(__file__)
script_dir = os.path.dirname(current_file_path)
# bench_root is the parent of bench_script directory
# bench_root = Path(os.environ["BENCH_ROOT"])
# robotwin_root = Path(os.environ["ROBOTWIN_ROOT"])

# os.chdir(robotwin_root)  # Change to customized_robotwin for proper path resolution

from envs import CONFIGS_PATH  # from customized_robotwin


def sync_viewer_to_scene_camera(env, camera_name: str) -> bool:
    """
    Match the interactive viewer frustum to an existing scene camera (same pose as data-collection cams).
    ``camera_name`` must appear in ``env.cameras.static_camera_name`` (e.g. head_camera, demo_camera).
    """
    if not camera_name or str(camera_name).lower() in ("default", "none", ""):
        return False
    viewer = getattr(env, "viewer", None)
    cams = getattr(env, "cameras", None)
    if viewer is None or cams is None:
        print("Warning: cannot sync viewer (missing viewer or cameras).")
        return False
    names = getattr(cams, "static_camera_name", None) or []
    if camera_name not in names:
        print(
            f"Warning: viewer-camera '{camera_name}' not in env static cameras {names}. "
            "Using default viewer pose; pass a valid name or --viewer-camera default."
        )
        return False
    idx = names.index(camera_name)
    cam = cams.static_camera_list[idx]
    try:
        pose = cam.entity.get_pose()
        viewer.set_camera_pose(pose)
    except Exception as e:
        print(f"Warning: failed to sync viewer to '{camera_name}': {e}")
        return False
    print(f"Viewer pose aligned with scene camera '{camera_name}'.")
    return True


def _extract_task_class(envs_module, task_name):
    """Extract task class from module, handling class names that differ from module name."""
    try:
        return getattr(envs_module, task_name)
    except AttributeError:
        from envs._base_task import Base_Task
        for name in dir(envs_module):
            obj = getattr(envs_module, name)
            if isinstance(obj, type) and issubclass(obj, Base_Task) and obj is not Base_Task:
                return obj
        raise SystemExit(f"No task class found in {envs_module.__name__}")


def get_env_class(task_name, bench_subdir=None):
    """Load task env class from bench_envs, or envs if not in bench_envs."""
    # Known bench_envs subpackages (office, study, etc.)
    BENCH_SUBDIRS = ["office", "study", "kitchenl", "kitchens"]

    if bench_subdir:
        # Explicit subdir: try only bench_envs.{subdir}.{task_name}
        try:
            envs_module = importlib.import_module(f"bench_envs.{bench_subdir}.{task_name}")
            return _extract_task_class(envs_module, task_name)
        except ModuleNotFoundError:
            raise SystemExit(f"Task '{task_name}' not found in bench_envs.{bench_subdir}")

    # Try bench_envs.{task_name} first (flat structure)
    try:
        envs_module = importlib.import_module(f"bench_envs.{task_name}")
        return _extract_task_class(envs_module, task_name)
    except ModuleNotFoundError:
        pass

    # Try bench_envs.{subdir}.{task_name} for each known subdir
    for subdir in BENCH_SUBDIRS:
        try:
            envs_module = importlib.import_module(f"bench_envs.{subdir}.{task_name}")
            return _extract_task_class(envs_module, task_name)
        except ModuleNotFoundError:
            continue

    # Fallback to envs
    try:
        envs_module = importlib.import_module(f"envs.{task_name}")
        return _extract_task_class(envs_module, task_name)
    except ModuleNotFoundError:
        raise SystemExit(f"No task class found for '{task_name}' in bench_envs or envs")


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _install_plan_fail_camera_saver(env, out_dir: Path, task_name: str, seed: int, camera_name: str):
    """
    Wrap env.move: on the first planning failure in a rollout, save RGB. move_step counts each
    top-level move() call in play_once (not inner simulation frames).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    move_step = [0]
    orig_move = env.move

    def wrapped_move(*args, **kwargs):
        move_step[0] += 1
        was_ok = env.plan_success
        ret = orig_move(*args, **kwargs)
        if was_ok and ret is False:
            stem = f"{task_name}_seed{seed}_move_step_{move_step[0]}"
            path = out_dir / f"{stem}.png"
            try:
                env.save_camera_rgb(str(path), camera_name=camera_name)
                print(f"[plan-fail] saved {path} (move step {move_step[0]})")
            except Exception as e:
                print(f"[plan-fail] could not save camera ({camera_name}): {e}")
        return ret

    env.move = wrapped_move


def _install_rollout_render_pacer(env, speed: float):
    """
    Pace viewer updates during ``play_once()`` without changing simulation timing.
    The interval is based on the configured render frequency and the scene timestep.
    """
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return lambda: None

    original_render = viewer.render
    render_every = max(1, int(getattr(env, "render_freq", 1) or 1))
    render_interval = float(env.scene.get_timestep()) * render_every / speed
    last_render_time = [None]

    def wrapped_render(*args, **kwargs):
        now = time.perf_counter()
        if last_render_time[0] is not None:
            remaining = render_interval - (now - last_render_time[0])
            if remaining > 0:
                time.sleep(remaining)
        result = original_render(*args, **kwargs)
        last_render_time[0] = time.perf_counter()
        return result

    viewer.render = wrapped_render

    def restore():
        if viewer.render is wrapped_render:
            viewer.render = original_render

    return restore


def _install_live_step_renderer(env):
    """
    Ensure visualization updates for task-specific loops that call ``scene.step()`` directly
    instead of going through ``env.move(...)``. This keeps the viewer live during dwell /
    conveyor / wait phases inside ``play_once()``.
    """
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return lambda: None

    original_step = env.scene.step
    step_counter = [0]
    render_every = max(1, int(getattr(env, "render_freq", 1) or 1))

    def wrapped_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        step_counter[0] += 1
        if not viewer.closed and step_counter[0] % render_every == 0:
            env._update_render()
            viewer.render()
        return result

    env.scene.step = wrapped_step

    def restore():
        if env.scene.step is wrapped_step:
            env.scene.step = original_step

    return restore


def main():
    parser = argparse.ArgumentParser(description="Visualize a benchmark task scene")
    # parser.add_argument("scene_name", type=str, help="Task module name (e.g. cook_meat)")
    parser.add_argument("task_name", type=str, help="Task module name (e.g. cook_meat)")
    parser.add_argument("task_config", type=str, help="Task config name (e.g. bench_demo_clean)")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed for scene")
    parser.add_argument("--render-freq", type=int, default=3, help="Render every N steps (default 1)")
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help=(
            "Visualization speed multiplier. 1.0 is realtime, 0.5 is half speed, "
            "2.0 is double speed. Omit to render as fast as possible."
        ),
    )
    parser.add_argument("--rollout", action="store_true", help="Run play_once() to roll out the task")
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=("live", "legacy"),
        default="live",
        help=(
            "With --rollout, 'live' renders direct scene.step() phases during play_once; "
            "'legacy' keeps the previous behavior (default: live)."
        ),
    )
    parser.add_argument("--bench-subdir", type=str, default=None,
                       help="Subdirectory under bench_envs (e.g. office, study)")
    parser.add_argument("--save_data", action="store_true", help="Save the visualization")
    parser.add_argument(
        "--viewer-camera",
        type=str,
        default="demo_camera",
        help=(
            "Scene camera name to copy the viewer pose from (see embodiment static_camera_list), "
            "e.g. demo_camera, head_camera, front_camera. Use 'default' to keep setup_scene viewer xyz/rpy."
        ),
    )
    parser.add_argument("--no-render", action="store_true",
                       help="Disable simulator rendering (run headless). Default: render on.")
    parser.add_argument(
        "--save-plan-fail-dir",
        type=str,
        default=None,
        help="If set with --rollout, save camera RGB when a top-level env.move() fails planning",
    )
    parser.add_argument(
        "--plan-fail-camera",
        type=str,
        default="demo_camera",
        help="Camera name passed to save_camera_rgb (default: demo_camera)",
    )

    args = parser.parse_args()
    if args.speed is not None and args.speed <= 0:
        parser.error("--speed must be greater than 0")

    # scene_name=args.scene_name
    task_name = args.task_name
    task_config = args.task_config
    seed = args.seed
    render_freq = args.render_freq
    speed = args.speed
    rollout = args.rollout
    render_mode = args.render_mode
    bench_subdir = args.bench_subdir
    enable_render = not args.no_render

    # Load env class from bench_envs
    # env_class = get_env_class(task_name, bench_subdir=scene_name)

    if os.getenv("ROBOTWIN_BENCH_TASK") == "bench":
        config_path = bench_root / "bench_task_config" / f"{task_config}.yml"
    else:
        config_path = Path(f"./task_config/{task_config}.yml")
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    cfg["task_name"] = task_name
    cfg["render_freq"] = render_freq if enable_render else 0
    cfg["now_ep_num"] = 0
    cfg["seed"] = seed if seed != -1 else np.random.randint(100)
    cfg["need_plan"] = True
    cfg["save_data"] = bool(args.save_data)

    # Embodiment setup (same as collect_data)
    embodiment_type = cfg.get("embodiment", ["aloha-agilex"])
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type_name):
        robot_file = _embodiment_types[embodiment_type_name]["file_path"]
        if robot_file is None:
            raise SystemExit("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        cfg["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        cfg["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        cfg["embodiment_dis"] = embodiment_type[2]
        cfg["dual_arm_embodied"] = False
    else:
        raise SystemExit("embodiment config should have 1 or 3 entries")

    cfg["left_embodiment_config"] = get_embodiment_config(cfg["left_robot_file"])
    cfg["right_embodiment_config"] = get_embodiment_config(cfg["right_robot_file"])

    envs_module = importlib.import_module(f"envs.{task_name}")

    env_class = getattr(envs_module, task_name)

    # Build env and setup scene with viewer
    env = env_class()
    env.setup_demo(**cfg)

    if args.save_data:
        env.save_data = True

    if enable_render and (not getattr(env, "render_freq", 0) or getattr(env, "viewer", None) is None):
        print("Warning: viewer not created (render_freq was 0?). Exiting.")
        env.close_env()
        return

    if enable_render:
        sync_viewer_to_scene_camera(env, args.viewer_camera)

    if rollout:
        print("******* Rolling out task (play_once)...****** ")
        env.ep_num = f"_{env.__class__.__name__}_{env.ep_num}"
        restore_live_step_renderer = lambda: None
        restore_rollout_render_pacer = lambda: None
        if enable_render and render_mode == "live":
            restore_live_step_renderer = _install_live_step_renderer(env)
        if enable_render and speed is not None:
            restore_rollout_render_pacer = _install_rollout_render_pacer(env, speed)

        if args.save_plan_fail_dir:
            _install_plan_fail_camera_saver(
                env,
                Path(args.save_plan_fail_dir),
                task_name,
                cfg["seed"],
                args.plan_fail_camera,
            )
        try:
            env.play_once()
        finally:
            restore_live_step_renderer()
            restore_rollout_render_pacer()
        if env.save_data:
            # env.ep_num = f"_{env.__class__.__name__}_{env.ep_num}"
            env.close_env(clear_cache=True)
            env.merge_pkl_to_hdf5_video()
            env.remove_data_cache()
        print("Rollout done.")

    if enable_render:
        viewer = env.viewer
        if not rollout:
            print("Scene ready. Close the viewer window to exit.")
        else:
            print("Close the viewer window to exit.")
        while not viewer.closed:
            frame_start = time.perf_counter()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            if speed is not None:
                remaining = float(env.scene.get_timestep()) / speed - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        if not rollout:
            print("Scene ready (headless). Press Ctrl+C to exit.")
        else:
            print("Rollout done (headless). Exiting.")
            env.close_env()
            print(f"Success: {env.check_success()}")
            print("Done.")
            return
        try:
            while True:
                env.scene.step()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass

    print(f"Success: {env.check_success()}")
    env.close_env()
    print("Done.")


if __name__ == "__main__":
    main()
