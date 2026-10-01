"""Shared helpers for ``interactive/base`` and ``interactive/household`` sandboxes."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# When set by base_task_gui / household_task_gui, report_task_result
# writes {"ok": bool, "detail": str} here so the launcher can show failure reasons.
TASK_RESULT_ENV = "ROBODYNA_TASK_RESULT_FILE"

# ANSI colors for interactive CLI (TTY only; respect NO_COLOR / FORCE_COLOR).
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def _stdout_supports_color() -> bool:
    """Color when stdout is a TTY, unless ``NO_COLOR`` is set.

    ``FORCE_COLOR`` (any value other than empty / ``0``) forces color even when
    not a TTY — useful for CI demos / piped capture.
    """
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force not in (None, "", "0"):
        return True
    return hasattr(sys.stdout, "isatty") and bool(sys.stdout.isatty())


def colorize(text: str, ansi_code: str) -> str:
    """Wrap ``text`` in ANSI color when stdout is a color-capable TTY."""
    if not text or not _stdout_supports_color():
        return text
    return f"{ansi_code}{text}{_ANSI_RESET}"


def print_instructions(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print controls / how-to text in cyan."""
    msg = sep.join(str(a) for a in args)
    print(colorize(msg, _ANSI_CYAN), end=end, flush=flush)


def print_success(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print a success / SUCCESS result line in green."""
    msg = sep.join(str(a) for a in args)
    print(colorize(msg, _ANSI_GREEN), end=end, flush=flush)


def print_failure(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print a failure / FAILURE result line in red."""
    msg = sep.join(str(a) for a in args)
    print(colorize(msg, _ANSI_RED), end=end, flush=flush)


def bootstrap_repo():
    """chdir to repo root and put it on ``sys.path`` (any caller cwd)."""
    os.chdir(REPO_ROOT)
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    bench = str(REPO_ROOT / "script" / "bench_script")
    if bench not in sys.path:
        sys.path.insert(0, bench)


def embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


CONTROL_ROBOT = "robot"
CONTROL_KEYBOARD_MOUSE = "keyboard+mouse"
_CONTROL_KEYBOARD_ALIASES = {
    "keyboard",
    "keyboard+mouse",
    "keyboardmouse",
    "key+mouse",
    "keymouse",
    "km",
}


def normalize_control_mode(value) -> str:
    """Canonical interactive controller: ``robot`` or ``keyboard+mouse``."""
    text = str(value or CONTROL_ROBOT).strip().lower()
    text = text.replace("_", "+").replace(" ", "")
    if text in _CONTROL_KEYBOARD_ALIASES:
        return CONTROL_KEYBOARD_MOUSE
    return CONTROL_ROBOT


def is_robot_control(value) -> bool:
    return normalize_control_mode(value) == CONTROL_ROBOT


def strip_interactive_arms(env) -> None:
    """Hide dual-arm articulations for keyboard+mouse (do not destroy them).

    Env files stay unchanged: robots still load during ``setup_demo``. Removing
    articulations from the PhysX scene leaves dangling contacts and can SIGSEGV
    on the next ``scene.step()`` / ``get_contacts()``. Hide, disable collision,
    and bury them instead.
    """
    if bool(getattr(env, "_interactive_arms_removed", False)):
        return
    robot = getattr(env, "robot", None)
    scene = getattr(env, "scene", None)
    if robot is None or scene is None:
        return
    import sapien

    for side in ("left", "right"):
        entity = getattr(robot, f"{side}_entity", None)
        if entity is None:
            continue
        # Hide any render bodies that survive a failed remove.
        try:
            for link in entity.get_links():
                for comp in link.entity.get_components():
                    if isinstance(comp, sapien.render.RenderBodyComponent):
                        try:
                            comp.visibility = 0.0
                        except Exception:
                            pass
                        try:
                            comp.disable()
                        except Exception:
                            pass
                    if isinstance(comp, sapien.physx.PhysxArticulationLinkComponent):
                        try:
                            for shape in comp.get_collision_shapes():
                                shape.set_collision_groups([0, 0, 0, 0])
                        except Exception:
                            pass
        except Exception:
            pass
        # Do not remove_articulation: PhysX then SIGSEGVs on scene.step / get_contacts.
        try:
            pose = entity.get_root_pose()
            entity.set_root_pose(sapien.Pose([pose.p[0], pose.p[1], -8.0], pose.q))
        except Exception:
            try:
                entity.set_root_pose(sapien.Pose([0.0, 0.0, -8.0]))
            except Exception:
                pass
        try:
            entity.set_qvel(np.zeros_like(entity.get_qpos()))
        except Exception:
            pass
    env._interactive_arms_removed = True
    env._interactive_robot_mode = False


def prepare_interactive_control(env, control) -> str:
    """Normalize ``control``, set robot-mode flag, and strip arms for keyboard+mouse."""
    mode = normalize_control_mode(control)
    env._interactive_robot_mode = mode == CONTROL_ROBOT
    if mode != CONTROL_ROBOT:
        strip_interactive_arms(env)
    return mode


def configure_task(task_name: str, config_name: str, seed: int, use_robot: bool,
                   task_arg_overrides: dict | None = None):
    """Load ``task_config/<config_name>.yml`` and wire embodiment paths."""
    disable_rt_for_interactive_capture()
    from envs import CONFIGS_PATH

    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if task_arg_overrides:
        task_args = config.setdefault("task_args", {}).setdefault(task_name, {})
        task_args.update(task_arg_overrides)

    config.update(
        task_name=task_name,
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=bool(use_robot),
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
    config["left_embodiment_config"] = embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = embodiment_config(config["right_robot_file"])
    config["task_config"] = config_name
    use_capture_camera(config)
    return config


def add_record_data_arg(parser):
    """Add ``--record-data`` (same HDF5 / LeRobot layout as ``collect_data.py``).

    Idempotent: safe to call after ``add_robot_motion_arg``, which already
    registers this flag.
    """
    if any("--record-data" in getattr(a, "option_strings", ()) for a in parser._actions):
        return parser
    parser.add_argument(
        "--record-data",
        action="store_true",
        help=(
            "Record this episode in collect_data format. State is logged during "
            "play; cameras (head + wrists + static, per demo_dynamic.yml) render "
            "after the viewer closes (HDF5 + preview mp4 + LeRobot under "
            "data/<task>/<config>/)"
        ),
    )
    return parser


def _env_flag(name: str) -> str | None:
    flag = os.environ.get(name, "").strip().lower()
    if flag in ("1", "true", "yes"):
        return "1"
    if flag in ("0", "false", "no"):
        return "0"
    return None


def interactive_record_data_requested() -> bool:
    """True when the GUI env var or ``--record-data`` CLI flag is set."""
    flag = _env_flag("ROBODYNA_RECORD_DATA")
    if flag == "1":
        return True
    if flag == "0":
        return False
    return "--record-data" in sys.argv


def interactive_save_video_requested() -> bool:
    """GUI Save video tick, or the collect_data preview mp4 for CLI ``--record-data``."""
    flag = _env_flag("ROBODYNA_SAVE_VIDEO")
    if flag == "0":
        return False
    if flag == "1":
        return True
    return interactive_record_data_requested()


def interactive_export_lerobot_requested() -> bool:
    """True when the HDF5 pass should also write LeRobot. Default off.

    Experiment yaml ``export_lerobot`` sets ``ROBODYNA_EXPORT_LEROBOT``. CLI
    ``--record-data`` does not export unless that env var is 1.
    """
    if not interactive_record_data_requested():
        return False
    return _env_flag("ROBODYNA_EXPORT_LEROBOT") == "1"


def interactive_capture_requested() -> bool:
    """True when Record data and/or Save video should run after the viewer closes."""
    return interactive_record_data_requested() or interactive_save_video_requested()


def disable_rt_for_interactive_capture() -> None:
    """Use raster cameras for interactive record/video.

    Ray-traced cameras deadlock mid-episode, and a second raster Engine created
    *after* the SAPIEN viewer closes leaves static cameras on a clear-color
    buffer (blank HDF5 / mp4). Rasterize the live env from the start so the
    record pass can reuse those cameras. Set ``ROBODYNA_DISABLE_RAY_TRACING=0``
    before launch to force ray tracing anyway.
    """
    if not interactive_capture_requested():
        return
    os.environ.setdefault("ROBODYNA_DISABLE_RAY_TRACING", "1")


disable_rt_for_interactive_capture()

CAPTURE_DEMO_CAMERA_TYPE = "Demo_HD"


def capture_demo_camera_type() -> str:
    """demo_camera preset for interactive / experiment capture.

    Head stays collect_data ``D435`` (320x240). The third-person ``demo_camera``
    uses ``Demo_HD`` (1280x960, same fovy as ``Demo_D435``) so HDF5 and the
    preview mp4 are sharp. Override with ``ROBODYNA_CAPTURE_DEMO_CAMERA_TYPE``.
    """
    return os.environ.get("ROBODYNA_CAPTURE_DEMO_CAMERA_TYPE", "").strip() or CAPTURE_DEMO_CAMERA_TYPE


def use_capture_camera(config: dict) -> None:
    """Raise demo-camera resolution when this run will record data or video."""
    if not interactive_capture_requested():
        return
    demo_type = capture_demo_camera_type()
    for key in ("left_embodiment_config", "right_embodiment_config"):
        emb = config.get(key)
        if not isinstance(emb, dict):
            continue
        for cam in emb.get("static_camera_list") or []:
            if str(cam.get("name") or "") == "demo_camera":
                cam["type"] = demo_type


def _interactive_preview_cameras():
    return ["demo_camera"]


def _argv_value(flag: str, default: str | None = None) -> str | None:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _record_config_folder(env) -> str:
    """Folder name under ``data/<task>/``, matching collect_data when possible."""
    scenario = (
        os.environ.get("ROBODYNA_SCENARIO")
        or getattr(env, "interactive_scenario", None)
        or ""
    )
    scenario = str(scenario).strip()
    config_name = (
        os.environ.get("ROBODYNA_RECORD_CONFIG")
        or getattr(env, "task_config", None)
        or _argv_value("--config")
        or "demo_dynamic"
    )
    config_name = str(config_name).strip() or "demo_dynamic"
    if config_name.startswith(".interactive_gui_"):
        config_name = "demo_dynamic"
    if scenario and scenario != "default":
        return f"{config_name}_{scenario}"
    return config_name


def _next_hdf5_episode_index(save_dir: str) -> int:
    idx = 0
    data_dir = os.path.join(save_dir, "data")
    while os.path.exists(os.path.join(data_dir, f"episode{idx}.hdf5")):
        idx += 1
    return idx


def _cli_seed_for_record(env) -> int:
    seed = getattr(env, "seed", None)
    if seed is not None:
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    if "--seed" in sys.argv:
        i = sys.argv.index("--seed")
        if i + 1 < len(sys.argv):
            try:
                return int(sys.argv[i + 1])
            except ValueError:
                pass
    return 0


def _render_shape_materials(shape) -> list:
    """Materials on one render shape. Multi-part meshes expose one per part."""
    try:
        return [shape.get_material()]
    except Exception:
        pass
    out = []
    try:
        parts = list(shape.get_parts())
    except Exception:
        return out
    for part in parts:
        try:
            out.append(part.get_material())
        except Exception:
            continue
    return out


def _interactive_record_materials(env) -> list:
    """Render materials in the scene, cached once so per-frame logging is flat.

    Built at first use and never rebuilt, so indices stay aligned between the
    play-time log and the restore pass. Materials whose entity is removed later
    stay in the list (harmless); visuals created later are not tracked and are
    handled by the task's own visual-state hook.
    """
    cached = getattr(env, "_interactive_record_mat_cache", None)
    if cached is not None:
        return cached
    import sapien

    mats: list = []
    seen: set[int] = set()
    scene = getattr(env, "scene", None)
    getter = getattr(scene, "get_entities", None) if scene is not None else None
    try:
        entities = list(getter()) if callable(getter) else list(getattr(scene, "entities", []) or [])
    except Exception:
        entities = []
    for entity in entities:
        try:
            components = entity.get_components()
        except Exception:
            continue
        for comp in components:
            if not isinstance(comp, sapien.render.RenderBodyComponent):
                continue
            try:
                shapes = list(comp.render_shapes)
            except Exception:
                continue
            for shape in shapes:
                for mat in _render_shape_materials(shape):
                    if mat is None or id(mat) in seen:
                        continue
                    seen.add(id(mat))
                    mats.append(mat)
    env._interactive_record_mat_cache = mats
    return mats


def _mat_base_color(mat):
    try:
        return mat.get_base_color()
    except Exception:
        try:
            return mat.base_color
        except Exception:
            return (0.0, 0.0, 0.0, 1.0)


def _interactive_record_color_state(env):
    """Base colors as a full baseline on the first frame, then sparse deltas.

    Task code that recolors render shapes (cook doneness, whacked moles, cook
    keys, stove flame) is not replayed, so without this every saved frame would
    show the final color.
    """
    mats = _interactive_record_materials(env)
    if not mats:
        return None
    cur = np.empty((len(mats), 4), dtype=np.float32)
    for i, mat in enumerate(mats):
        cur[i] = np.asarray(_mat_base_color(mat), dtype=np.float32)[:4]
    prev = getattr(env, "_interactive_record_color_prev", None)
    env._interactive_record_color_prev = cur
    if prev is None or prev.shape != cur.shape:
        return ("full", None, cur, len(mats))
    changed = np.nonzero(np.any(cur != prev, axis=1))[0]
    if changed.size == 0:
        return None
    return ("delta", changed.astype(np.int32), cur[changed].copy(), len(mats))


def _interactive_record_restore_colors(env, entry) -> None:
    """Apply a logged color baseline / delta. Deltas accumulate in frame order."""
    if not entry:
        return
    mats = _interactive_record_materials(env)
    if not mats:
        return
    kind, idx, vals, n_mats = entry
    # A different scene (headless replay fallback) must not take these indices.
    if int(n_mats) != len(mats):
        return
    if kind == "full":
        for mat, color in zip(mats, vals):
            try:
                mat.set_base_color([float(c) for c in color])
            except Exception:
                continue
        return
    for i, color in zip(idx, vals):
        if 0 <= int(i) < len(mats):
            try:
                mats[int(i)].set_base_color([float(c) for c in color])
            except Exception:
                continue


def _interactive_record_log_state(env) -> None:
    """Cheap per-frame snapshot: robot qpos + actor/articulation poses. No cameras."""
    robot = getattr(env, "robot", None)
    scene = getattr(env, "scene", None)
    if robot is None or scene is None:
        return
    try:
        left_q = np.asarray(robot.left_entity.get_qpos(), dtype=np.float32).copy()
        right_q = np.asarray(robot.right_entity.get_qpos(), dtype=np.float32).copy()
        left_cmd = np.asarray(robot.get_left_arm_jointState(), dtype=np.float32)
        right_cmd = np.asarray(robot.get_right_arm_jointState(), dtype=np.float32)
    except Exception:
        return
    actors = []
    for actor in scene.get_all_actors():
        try:
            pose = actor.get_pose()
            actors.append(
                (np.asarray(pose.p, dtype=np.float32), np.asarray(pose.q, dtype=np.float32))
            )
        except Exception:
            actors.append(None)
    arts = []
    robot_ids = {id(robot.left_entity), id(robot.right_entity)}
    getter = getattr(scene, "get_all_articulations", None)
    if callable(getter):
        try:
            articulations = list(getter())
        except Exception:
            articulations = []
        for art in articulations:
            if id(art) in robot_ids:
                continue
            try:
                pose = art.get_root_pose() if hasattr(art, "get_root_pose") else art.get_pose()
                qpos = np.asarray(art.get_qpos(), dtype=np.float32).copy()
                arts.append(
                    (
                        np.asarray(pose.p, dtype=np.float32),
                        np.asarray(pose.q, dtype=np.float32),
                        qpos,
                    )
                )
            except Exception:
                arts.append(None)
    try:
        colors = _interactive_record_color_state(env)
    except Exception:
        colors = None
    vis = None
    getter = getattr(env, "interactive_record_visual_state", None)
    if callable(getter):
        try:
            vis = getter()
        except Exception:
            vis = None
    log = getattr(env, "_interactive_record_log", None)
    if log is None:
        env._interactive_record_log = []
        log = env._interactive_record_log
    log.append(
        {
            "lq": left_q,
            "rq": right_q,
            "lc": left_cmd,
            "rc": right_cmd,
            "lg": float(robot.get_left_gripper_val()),
            "rg": float(robot.get_right_gripper_val()),
            "actors": actors,
            "arts": arts,
            "cols": colors,
            "vis": vis,
        }
    )


def _interactive_record_restore_state(env, snap: dict) -> None:
    """Apply a logged snapshot so offscreen cameras see the live episode."""
    import sapien

    robot = env.robot
    scene = env.scene
    left_q = np.asarray(snap["lq"], dtype=np.float64)
    right_q = np.asarray(snap["rq"], dtype=np.float64)
    try:
        robot.left_entity.set_qpos(left_q)
        robot.left_entity.set_qvel(np.zeros_like(left_q))
        robot.right_entity.set_qpos(right_q)
        robot.right_entity.set_qvel(np.zeros_like(right_q))
    except Exception:
        pass
    try:
        left_active = robot.left_entity.get_active_joints()
        for joint in robot.left_arm_joints:
            idx = left_active.index(joint)
            joint.set_drive_target(float(left_q[idx]))
            joint.set_drive_velocity_target(0.0)
        right_active = robot.right_entity.get_active_joints()
        for joint in robot.right_arm_joints:
            idx = right_active.index(joint)
            joint.set_drive_target(float(right_q[idx]))
            joint.set_drive_velocity_target(0.0)
    except Exception:
        pass
    try:
        robot.left_gripper_val = float(snap["lg"])
        robot.right_gripper_val = float(snap["rg"])
        left_cmd = np.asarray(snap["lc"], dtype=np.float64)
        right_cmd = np.asarray(snap["rc"], dtype=np.float64)
        if left_cmd.size:
            robot.set_gripper(float(left_cmd[-1]), "left", gripper_eps=0)
        if right_cmd.size:
            robot.set_gripper(float(right_cmd[-1]), "right", gripper_eps=0)
        if left_cmd.size > 1:
            robot.set_arm_joints(left_cmd[:-1], np.zeros(left_cmd.size - 1), "left")
        if right_cmd.size > 1:
            robot.set_arm_joints(right_cmd[:-1], np.zeros(right_cmd.size - 1), "right")
    except Exception:
        pass

    actors = list(scene.get_all_actors())
    for actor, pose in zip(actors, snap.get("actors") or []):
        if pose is None:
            continue
        try:
            actor.set_pose(sapien.Pose(pose[0], pose[1]))
        except Exception:
            continue

    getter = getattr(scene, "get_all_articulations", None)
    if callable(getter):
        robot_ids = {id(robot.left_entity), id(robot.right_entity)}
        try:
            articulations = [art for art in getter() if id(art) not in robot_ids]
        except Exception:
            articulations = []
        for art, pose in zip(articulations, snap.get("arts") or []):
            if pose is None:
                continue
            p, q, qpos = pose
            try:
                if hasattr(art, "set_root_pose"):
                    art.set_root_pose(sapien.Pose(p, q))
                else:
                    art.set_pose(sapien.Pose(p, q))
                art.set_qpos(np.asarray(qpos, dtype=np.float64))
                try:
                    art.set_qvel(np.zeros_like(qpos, dtype=np.float64))
                except Exception:
                    pass
            except Exception:
                continue

    _interactive_record_restore_colors(env, snap.get("cols"))
    # Visuals rebuilt by the task (liquid columns, followers) come last so they
    # can key off the actor poses restored above.
    restore_vis = getattr(env, "interactive_restore_visual_state", None)
    if callable(restore_vis) and snap.get("vis") is not None:
        try:
            restore_vis(snap.get("vis"))
        except Exception:
            pass
    try:
        scene.update_render()
    except Exception:
        pass


def _interactive_record_after_step(env):
    if getattr(env, "_interactive_record_cutoff", False):
        return
    if getattr(env, "_interactive_record_finished", False):
        return
    env._interactive_record_steps = int(getattr(env, "_interactive_record_steps", 0)) + 1
    freq = int(getattr(env, "save_freq", None) or 15)
    if freq <= 0:
        freq = 15
    if env._interactive_record_steps % freq == 0:
        _interactive_record_log_state(env)


def maybe_attach_interactive_data_recorder(env) -> bool:
    """Log cheap state during play; render collect_data cameras after the viewer closes.

    Idempotent. No-ops unless Record data and/or Save video is requested.
    """
    if getattr(env, "_interactive_record_active", False):
        return True
    if not interactive_capture_requested():
        return False
    scene = getattr(env, "scene", None)
    if scene is None or not hasattr(scene, "step"):
        return False

    task_name = str(getattr(env, "task_name", None) or type(env).__name__)
    folder = _record_config_folder(env)
    save_root = str(getattr(env, "save_dir", None) or "./data")
    expected = os.path.join(task_name, folder)
    if not os.path.abspath(save_root).replace("\\", "/").endswith(expected.replace("\\", "/")):
        save_root = os.path.join(save_root, task_name, folder)
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(os.path.join(save_root, "data"), exist_ok=True)
    os.makedirs(os.path.join(save_root, "video"), exist_ok=True)

    env.save_dir = save_root
    env.save_data = False
    env.task_config = folder
    env.ep_num = _next_hdf5_episode_index(save_root)
    env.FRAME_IDX = 0
    if not isinstance(getattr(env, "data_type", None), dict):
        env.data_type = {
            "rgb": True,
            "third_view": False,
            "depth": False,
            "pointcloud": False,
            "observer": False,
            "endpose": True,
            "qpos": True,
            "mesh_segmentation": False,
            "actor_segmentation": False,
        }
    if not getattr(env, "save_freq", None):
        env.save_freq = 15

    control = (
        os.environ.get("ROBODYNA_CONTROL", "").strip()
        or _argv_value("--control")
        or "robot"
    )
    env._interactive_record_active = True
    env._interactive_record_via_step = False
    env._interactive_record_cutoff = False
    env._interactive_record_steps = 0
    env._interactive_record_finished = False
    env._interactive_record_log = []
    env._interactive_record_seed = _cli_seed_for_record(env)
    env._interactive_record_config_name = (
        _argv_value("--config")
        or os.environ.get("ROBODYNA_RECORD_CONFIG")
        or "demo_dynamic"
    )
    env._interactive_record_use_robot = str(control).strip().lower() == "robot"
    env._interactive_record_task_cls = type(env)
    env._interactive_record_write_hdf5 = interactive_record_data_requested()
    env._interactive_record_write_video = interactive_save_video_requested()
    env._interactive_record_args = {
        "task_name": task_name,
        "task_config": folder,
        "save_path": save_root,
        "save_freq": env.save_freq,
        "export_lerobot": interactive_export_lerobot_requested(),
        "lerobot_root": "./data_lerobot/domino_suite",
        "lerobot_chunks_size": 1000,
        "lerobot_task_state_dim": 32,
        "data_type": dict(env.data_type),
    }

    orig_step = scene.step

    def _step_and_record(*args, **kwargs):
        result = orig_step(*args, **kwargs)
        _interactive_record_after_step(env)
        return result

    scene.step = _step_and_record
    env._interactive_record_orig_step = orig_step

    orig_close = env.close_env

    def _close_and_finalize(*args, **kwargs):
        env.close_env = orig_close
        try:
            finish_interactive_data_recording(env, live_close=(orig_close, args, kwargs))
        except Exception as exc:
            print(f"[record-data] finalize failed: {exc}")
            try:
                orig_close(*args, **kwargs)
            except Exception:
                pass

    env.close_env = _close_and_finalize
    _interactive_record_log_state(env)
    hdf5 = os.path.join(save_root, "data", f"episode{env.ep_num}.hdf5")
    preview = os.path.join(save_root, "video", f"episode{env.ep_num}.mp4")
    bits = []
    if env._interactive_record_write_hdf5:
        bits.append(hdf5)
    if env._interactive_record_write_video:
        bits.append(preview)
    dest = " + ".join(bits) if bits else save_root
    print(
        f"[record-data] logging robot/object state for episode {env.ep_num}; "
        f"cameras (collect_data yaml) render after the viewer closes → {dest}"
    )
    return True


def _head_rgb_unique_colors(env) -> int | None:
    try:
        rgb = env.cameras.get_rgb().get("head_camera", {}).get("rgb")
        if rgb is None:
            return None
        img = np.asarray(rgb)
        if img.ndim != 3:
            return None
        return int(np.unique(img.reshape(-1, img.shape[-1]), axis=0).shape[0])
    except Exception:
        return None


def _render_logged_snapshots_on_live_env(env, snapshots: list) -> int:
    """Restore logged poses on the live env and take camera pictures.

    A second Engine after ``Viewer.close()`` leaves static cameras on a
    clear-color buffer, so this must run on the same scene the user played in.
    """
    if not snapshots:
        return 0
    if getattr(env, "cameras", None) is None or getattr(env, "scene", None) is None:
        raise RuntimeError("live env has no cameras/scene")
    env.save_data = True
    env.FRAME_IDX = 0
    env._interactive_record_via_step = False
    env._interactive_record_now = True
    env._record_write_hdf5 = bool(getattr(env, "_interactive_record_write_hdf5", True))
    env._record_write_video = bool(getattr(env, "_interactive_record_write_video", True))
    env._record_preview_cameras = _interactive_preview_cameras()
    try:
        from envs.utils.household_view import configure_standard_head_camera

        configure_standard_head_camera(env)
    except Exception:
        pass
    align_demo_camera_to_head(env)
    n = len(snapshots)
    from envs._base_task import ray_tracing_disabled

    shading = "rasterized" if ray_tracing_disabled() else "ray-traced"
    print(
        f"[record-data] rendering {n} frames on live cameras, {shading}...",
        flush=True,
    )
    for i, snap in enumerate(snapshots):
        _interactive_record_restore_state(env, snap)
        env._take_picture()
        if i == 0:
            nuniq = _head_rgb_unique_colors(env)
            preview = (_interactive_preview_cameras() or ["head_camera"])[0]
            if preview != "head_camera":
                try:
                    rgb = env.cameras.get_rgb().get(preview, {}).get("rgb")
                    img = np.asarray(rgb) if rgb is not None else None
                    if img is not None and img.ndim == 3:
                        nuniq = int(np.unique(img.reshape(-1, img.shape[-1]), axis=0).shape[0])
                except Exception:
                    pass
            if nuniq is not None and nuniq < 32:
                print(
                    f"[record-data] warning: {preview} looks empty "
                    f"({nuniq} unique colors) on the first saved frame",
                    flush=True,
                )
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(
                f"[record-data] offscreen {i + 1}/{n}",
                end="\r",
                flush=True,
            )
    print()
    return int(getattr(env, "FRAME_IDX", 0) or 0)


def _replay_interactive_cameras(meta: dict):
    """Headless setup_demo + pose restore + _take_picture for logged snapshots.

    Camera / ``data_type`` flags follow the same yaml as ``collect_data.py``
    (head + wrists + embodiment static cams such as ``demo_camera`` /
    ``front_camera``) so the HDF5 schema matches planner collection.
    The companion ``episode*.mp4`` is the interactive default ``head_camera``.
    """
    snapshots = meta.get("snapshots") or []
    if not snapshots:
        print("[record-data] no state logged; nothing to save")
        return None
    task_cls = meta["task_cls"]
    task_name = meta["task_name"]
    seed = int(meta["seed"])
    save_root = meta["save_root"]
    ep_num = int(meta["ep_num"])
    folder = meta["folder"]
    save_freq = int(meta.get("save_freq") or 15)
    data_type = meta.get("data_type") or {}
    config_name = str(meta.get("config_name") or "demo_dynamic")
    # GUI temp ymls are live-viewer only; record against the collect_data config.
    if config_name.startswith(".interactive_gui_"):
        config_name = "demo_dynamic"
    use_robot = bool(meta.get("use_robot", True))
    # Rasterize this pass: the rt camera shader can deadlock on a GPU fence
    # partway through, which would stall the HDF5 / video write forever. Set the
    # variable to 0 beforehand to record with ray tracing anyway.
    os.environ.setdefault("ROBODYNA_DISABLE_RAY_TRACING", "1")
    from envs._base_task import ray_tracing_disabled

    shading = "rasterized" if ray_tracing_disabled() else "ray-traced"
    print(
        f"[record-data] rendering {len(snapshots)} frames offscreen, {shading} "
        f"(same cameras as collect_data / {config_name})..."
    )
    config = configure_task(task_name, config_name, seed, use_robot)
    config["render_freq"] = 0
    config["save_data"] = True
    config["save_path"] = save_root
    config["now_ep_num"] = ep_num
    config["save_freq"] = save_freq
    config["task_config"] = folder
    if data_type:
        # Keep yaml data_type when unset; otherwise honor the live recorder copy
        # (same keys collect_data reads from demo_dynamic.yml).
        config["data_type"] = dict(data_type)
    replay = task_cls()
    replay.setup_demo(**config)
    replay.save_dir = save_root
    replay.save_data = True
    replay.ep_num = ep_num
    replay.FRAME_IDX = 0
    replay.task_config = folder
    replay._record_write_hdf5 = bool(meta.get("write_hdf5", True))
    replay._record_write_video = bool(meta.get("write_video", True))
    replay._record_preview_cameras = _interactive_preview_cameras()
    try:
        from envs.utils.household_view import configure_standard_head_camera

        configure_standard_head_camera(replay)
    except Exception:
        pass
    align_demo_camera_to_head(replay)
    if data_type:
        replay.data_type = dict(data_type)
    try:
        for i, snap in enumerate(snapshots):
            _interactive_record_restore_state(replay, snap)
            replay._take_picture()
            if (i + 1) % 20 == 0 or i + 1 == len(snapshots):
                print(
                    f"[record-data] offscreen {i + 1}/{len(snapshots)}",
                    end="\r",
                    flush=True,
                )
        print()
        return replay
    except Exception:
        try:
            replay.close_env()
        except Exception:
            pass
        raise


def finish_interactive_data_recording(env, live_close=None) -> str | None:
    """Close the live viewer, then render cameras from logged poses."""
    if not getattr(env, "_interactive_record_active", False):
        if live_close is not None:
            fn, args, kwargs = live_close
            fn(*args, **kwargs)
        return None
    if getattr(env, "_interactive_record_finished", False):
        if live_close is not None:
            fn, args, kwargs = live_close
            fn(*args, **kwargs)
        return getattr(env, "_interactive_record_hdf5", None)
    env._interactive_record_finished = True
    orig_step = getattr(env, "_interactive_record_orig_step", None)
    scene = getattr(env, "scene", None)
    if orig_step is not None and scene is not None:
        try:
            scene.step = orig_step
        except Exception:
            pass

    snapshots = list(getattr(env, "_interactive_record_log", None) or [])
    save_dir = str(getattr(env, "save_dir", "") or "")
    ep_num = int(getattr(env, "ep_num", 0) or 0)
    success = _LAST_TASK_RESULT
    if success is None:
        try:
            success = bool(env.check_success())
        except Exception:
            success = False
    # Capture before close_env tears the scene down. Same payload shape as
    # collect_data's ``info_db[episode_N] = play_once()`` when the task filled
    # ``env.info`` (language placeholders live under ``info``).
    live_info = None
    try:
        from copy import deepcopy

        raw = getattr(env, "info", None)
        if isinstance(raw, dict):
            live_info = deepcopy(raw)
    except Exception:
        live_info = None
    meta = {
        "snapshots": snapshots,
        "task_cls": getattr(env, "_interactive_record_task_cls", None) or type(env),
        "task_name": str(getattr(env, "task_name", None) or type(env).__name__),
        "seed": int(getattr(env, "_interactive_record_seed", 0) or 0),
        "save_root": save_dir,
        "ep_num": ep_num,
        "folder": str(getattr(env, "task_config", None) or folder_name_from_save_dir(save_dir)),
        "save_freq": int(getattr(env, "save_freq", None) or 15),
        "data_type": dict(getattr(env, "_interactive_record_args", {}) or {}).get("data_type")
        or dict(getattr(env, "data_type", None) or {}),
        "config_name": str(getattr(env, "_interactive_record_config_name", None) or "demo_dynamic"),
        "use_robot": bool(getattr(env, "_interactive_record_use_robot", True)),
        "args": dict(getattr(env, "_interactive_record_args", None) or {}),
        "success": bool(success),
        "write_hdf5": bool(getattr(env, "_interactive_record_write_hdf5", True)),
        "write_video": bool(getattr(env, "_interactive_record_write_video", True)),
        "live_info": live_info,
    }

    def _run_live_close():
        if live_close is None:
            return
        fn, args, kwargs = live_close
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"[record-data] live env close: {exc}")

    if not snapshots or not save_dir:
        print("[record-data] no frames captured; nothing to save")
        _run_live_close()
        return None

    if bool(getattr(env, "_interactive_record_write_hdf5", True)):
        _mark_interactive_recording_started()

    replay = env
    n_frames = 0
    try:
        # Keep the live renderer/cameras. Closing the SAPIEN viewer first
        # and then creating a new Engine was what produced blank frames.
        n_frames = _render_logged_snapshots_on_live_env(env, snapshots)
    except Exception as exc:
        print(f"[record-data] live camera render failed: {exc}")
        replay = None

    if replay is None or n_frames <= 0:
        _run_live_close()
        try:
            replay = _replay_interactive_cameras(meta)
        except Exception as exc:
            print(f"[record-data] camera replay failed: {exc}")
            return None
        if replay is None:
            return None
        n_frames = int(getattr(replay, "FRAME_IDX", 0) or 0)
        live_already_closed = True
    else:
        live_already_closed = False

    try:
        replay.merge_pkl_to_hdf5_video()
    except Exception as exc:
        print(f"[record-data] HDF5 merge failed: {exc}")
        if not live_already_closed:
            _run_live_close()
        else:
            try:
                replay.close_env()
            except Exception:
                pass
        return None

    args = dict(meta.get("args") or {})
    want_data = bool(meta.get("write_hdf5", True))
    want_video = bool(meta.get("write_video", True))
    if want_data and args.get("export_lerobot", False):
        try:
            from envs.utils.lerobot_export import LeRobotEpisodeExporter

            LeRobotEpisodeExporter(args).export(replay, bool(success))
        except Exception as exc:
            print(f"[record-data] LeRobot export failed: {exc}")

    try:
        replay.remove_data_cache()
    except Exception:
        pass
    if live_already_closed:
        try:
            replay.close_env()
        except Exception:
            pass
    else:
        _run_live_close()

    seed = int(meta["seed"])
    if want_data:
        seed_path = os.path.join(save_dir, "seed.txt")
        try:
            existing = []
            if os.path.exists(seed_path):
                existing = [int(x) for x in Path(seed_path).read_text(encoding="utf-8").split() if x.strip()]
            existing.append(seed)
            Path(seed_path).write_text(" ".join(str(s) for s in existing) + " ", encoding="utf-8")
        except Exception as exc:
            print(f"[record-data] seed.txt update failed: {exc}")

        info_path = os.path.join(save_dir, "scene_info.json")
        try:
            info_db = {}
            if os.path.exists(info_path):
                info_db = json.loads(Path(info_path).read_text(encoding="utf-8") or "{}")
            # Match collect_data: play_once() dict (cluttered_table_info / texture_info /
            # info placeholders) plus interactive provenance fields.
            entry = {}
            live = meta.get("live_info")
            if isinstance(live, dict):
                entry.update(live)
            entry.setdefault("cluttered_table_info", [])
            entry.setdefault(
                "texture_info",
                {"wall_texture": None, "table_texture": None},
            )
            entry.setdefault("info", {})
            entry["seed"] = seed
            entry["success"] = bool(success)
            entry["source"] = "interactive"
            option = (
                os.environ.get("ROBODYNA_SCENARIO", "").strip()
                or str(getattr(env, "interactive_scenario", None) or "").strip()
                or _resolve_catalog_option_label(env)
            )
            entry["scenario"] = option or "default"
            if option:
                # Base catalog tag (default / opt1 / opt2 / opt1+2).
                entry["option_label"] = option
            timing = _frozen_episode_timing(env)
            entry["total_time_sim_s"] = timing.get("total_time_sim_s")
            entry["wall_s"] = timing.get("wall_s")
            entry["steps"] = timing.get("steps")
            entry["frames"] = n_frames
            extras = getattr(env, "_last_result_extras", None)
            if isinstance(extras, dict) and isinstance(extras.get("metrics"), dict):
                entry["metrics"] = dict(extras["metrics"])
                fail_reason = str(
                    extras.get("fail_reason")
                    or entry["metrics"].get("fail_reason")
                    or ""
                ).strip()
                if fail_reason:
                    entry["fail_reason"] = fail_reason
            elif _LAST_TASK_RESULT is False and _LAST_TASK_DETAIL:
                entry["fail_reason"] = str(_LAST_TASK_DETAIL)
            info_db[f"episode_{ep_num}"] = entry
            Path(info_path).write_text(
                json.dumps(info_db, ensure_ascii=False, indent=4, default=_json_safe),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[record-data] scene_info.json update failed: {exc}")

    hdf5 = os.path.join(save_dir, "data", f"episode{ep_num}.hdf5")
    preview = os.path.join(save_dir, "video", f"episode{ep_num}.mp4")
    if not want_video or not os.path.exists(preview):
        preview = None
    if not want_data or not os.path.exists(hdf5):
        hdf5 = None
    env._interactive_record_hdf5 = hdf5
    extra = {
        "record_path": save_dir,
        "record_episode": ep_num,
    }
    if hdf5:
        extra["record_hdf5"] = hdf5
    if preview:
        extra["record_video"] = preview
    _persist_task_result(_LAST_TASK_RESULT, _LAST_TASK_DETAIL, extra=extra)
    if hdf5:
        print(f"[record-data] saved {hdf5} ({n_frames} frames, success={bool(success)})")
    if preview:
        print(f"[record-data] preview video {preview}")

    return hdf5

def folder_name_from_save_dir(save_dir: str) -> str:
    return os.path.basename(os.path.abspath(save_dir))


_VIEW_HELP_V = "V — cycle view: head_camera ↔ gripper(s)"


def _is_view_help_line(line: str) -> bool:
    low = line.lower()
    s = line.strip()
    return (
        "toggle view" in low
        or "cycle view" in low
        or "gripper view" in low
        or "gripper / wrist" in low
        or s.startswith("V —")
        or s.startswith("V:")
        or s.startswith("V ")
        or "V                 " in line
    )


def _normalize_view_help_lines(lines: list[str], *, include_v: bool = True) -> list[str]:
    """Keep a single V head↔gripper help line, or strip V for keyboard+mouse."""
    out: list[str] = []
    saw_v = False
    for line in lines:
        if not _is_view_help_line(line):
            out.append(line)
            continue
        if not include_v:
            continue
        # Drop legacy G-only gripper-view lines; V owns the camera cycle now.
        s = line.strip()
        if (
            s.startswith("G ")
            or s.startswith("G:")
            or s.startswith("G —")
            or s.startswith("G\t")
        ) and ("gripper view" in line.lower() or "wrist" in line.lower()):
            continue
        if saw_v:
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        # Preserve em-dash style used by some banners.
        sep = "—" if "—" in line else ("-" if " - " in line else None)
        if sep == "—":
            out.append(f"{indent}V — cycle view: head_camera ↔ gripper(s)")
        else:
            out.append(f"{indent}V                 cycle view: head_camera ↔ gripper(s)")
        saw_v = True
    if include_v and not saw_v:
        out.append(_VIEW_HELP_V)
    return out


def _ensure_view_help_lines(lines: list[str], *, include_v: bool = True) -> list[str]:
    """Normalize / append V camera help (head ↔ grippers; no top-down)."""
    return _normalize_view_help_lines(list(lines), include_v=include_v)


def print_banner(title: str, lines: list[str]):
    lines = list(lines)
    if any("Mode: robot" in line for line in lines):
        insert_at = 1 if lines else 0
        lines[insert_at:insert_at] = [
            "Arrows — move selected arm(s) in XY | E/Q — move in Z",
            "1 / 2 / 3 — select left / right / both arms (selected gripper turns green)",
            "O — return selected arm(s) to original position",
        ]
    blob = f"{title}\n" + "\n".join(lines)
    include_v = "keyboard" not in blob.lower()
    lines = _ensure_view_help_lines(lines, include_v=include_v)
    width = max(len(title), *(len(line) for line in lines), 40)
    bar = "=" * (width + 4)
    block = "\n".join([bar, f"  {title}", bar, *[f"  {line}" for line in lines], bar])
    print_instructions(block)


def edge_pressed(window, key: str, prev: dict) -> bool:
    """True on the frame a key transitions from up → down."""
    if window.key_press(key):
        prev[key] = True
        return True
    down = bool(window.key_down(key))
    was = prev.get(key, False)
    prev[key] = down
    return down and not was


def actor_entity(actor):
    """Underlying sapien entity for an Actor wrapper or raw entity."""
    if actor is None:
        return None
    return getattr(actor, "actor", actor)


def actor_scene_id(actor) -> int | None:
    entity = actor_entity(actor)
    if entity is None:
        return None
    try:
        return int(entity.per_scene_id)
    except Exception:
        return None


def click_segmentation_ids(viewer, pixel_x: int, pixel_y: int) -> tuple[int, int]:
    """Return ``(entity_id, scene_id)`` from the Segmentation buffer at a click."""
    pixel = viewer.window.get_picture_pixel("Segmentation", int(pixel_x), int(pixel_y))
    return int(pixel[1]), int(pixel[2])


def click_hits_actor(viewer, pixel_x: int, pixel_y: int, actor) -> bool:
    """True when the Segmentation pixel at ``(pixel_x, pixel_y)`` is ``actor``."""
    wanted = actor_scene_id(actor)
    if wanted is None:
        return False
    entity_id, _scene_id = click_segmentation_ids(viewer, pixel_x, pixel_y)
    return entity_id == wanted


def click_hits_actor_map(viewer, pixel_x: int, pixel_y: int, id_to_value: dict):
    """Look up a click against ``{per_scene_id: value}``; return value or None."""
    entity_id, _scene_id = click_segmentation_ids(viewer, pixel_x, pixel_y)
    return id_to_value.get(int(entity_id))


def table_xy_from_click(viewer, pixel_x: int, pixel_y: int, plane_z: float):
    """Unproject a viewer click onto the horizontal plane ``z=plane_z``.

    Prefers the renderer's ``Position`` buffer (view-space hit on visible
    geometry — usually the tabletop). Falls back to a Vulkan-correct ray /
    plane intersection when the pixel has no geometry.

    Returns ``(x, y)`` or ``None`` if the ray misses (parallel / behind).
    """
    import sapien

    window = viewer.window
    tw, th = window.get_picture_size("Segmentation")
    if tw <= 1 or th <= 1:
        return None
    px = int(np.clip(pixel_x, 0, tw - 1))
    py = int(np.clip(pixel_y, 0, th - 1))

    # SAPIEN viewer camera → GL/Vulkan view: same offset as TransformWindow.
    gl_fix = sapien.Pose([0.0, 0.0, 0.0], [-0.5, -0.5, 0.5, 0.5])
    cam_gl = window.get_camera_pose() * gl_fix
    world_from_view = np.asarray(cam_gl.to_transformation_matrix(), dtype=np.float64)
    proj = np.asarray(window.get_camera_projection_matrix(), dtype=np.float64)

    def _as_xy(world_xyz) -> tuple[float, float] | None:
        w = np.asarray(world_xyz, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(w)):
            return None
        # Project onto the table plane along world Z (table is horizontal).
        return float(w[0]), float(w[1])

    # ---- 1) Position buffer: exact surface under the cursor (table / props) ----
    for name in ("Position", "position"):
        try:
            pix = np.asarray(window.get_picture_pixel(name, px, py), dtype=np.float64)
        except Exception:
            continue
        if pix.size < 3:
            continue
        view_xyz = pix[:3]
        # Empty / sky samples are typically 0 or non-finite.
        if not np.all(np.isfinite(view_xyz)):
            continue
        if float(np.linalg.norm(view_xyz)) < 1e-5:
            continue
        world_h = world_from_view @ np.array(
            [view_xyz[0], view_xyz[1], view_xyz[2], 1.0], dtype=np.float64
        )
        if abs(float(world_h[3])) > 1e-9:
            world_h = world_h / float(world_h[3])
        hit = _as_xy(world_h[:3])
        if hit is not None:
            return hit

    # ---- 2) Ray ∩ z=plane_z (Vulkan NDC; match deferred.frag UV→NDC) ----
    try:
        inv_proj = np.linalg.inv(proj)
    except np.linalg.LinAlgError:
        return None

    u = (float(px) + 0.5) / float(tw)
    v = (float(py) + 0.5) / float(th)
    # deferred.frag: ndc.xy = inUV * 2 - 1. Picture (0,0) is top-left like mouse.
    # Try both Y orientations; keep the forward hit closest to the camera.
    candidates: list[tuple[float, float, float]] = []  # (cam_dist, x, y)
    cam_pos = np.asarray(cam_gl.p, dtype=np.float64)

    for ndc_y in (2.0 * v - 1.0, 1.0 - 2.0 * v):
        ndc_x = 2.0 * u - 1.0
        # Vulkan depth in [0, 1]; also try OpenGL-style ±1 for older matrices.
        for z_near, z_far in ((0.0, 1.0), (-1.0, 1.0)):
            def _eye(ndc_z: float) -> np.ndarray:
                clip = np.array([ndc_x, ndc_y, ndc_z, 1.0], dtype=np.float64)
                eye = inv_proj @ clip
                if abs(float(eye[3])) > 1e-12:
                    eye = eye / float(eye[3])
                return eye[:3]

            near_eye = _eye(z_near)
            far_eye = _eye(z_far)
            # View → world
            def _world(eye_xyz: np.ndarray) -> np.ndarray:
                h = world_from_view @ np.array(
                    [eye_xyz[0], eye_xyz[1], eye_xyz[2], 1.0], dtype=np.float64
                )
                if abs(float(h[3])) > 1e-12:
                    h = h / float(h[3])
                return h[:3]

            p0 = _world(near_eye)
            p1 = _world(far_eye)
            direction = p1 - p0
            denom = float(direction[2])
            if abs(denom) < 1e-9:
                continue
            t = (float(plane_z) - float(p0[2])) / denom
            if t < 0.0:
                continue
            hit = p0 + t * direction
            dist = float(np.linalg.norm(hit - cam_pos))
            candidates.append((dist, float(hit[0]), float(hit[1])))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], candidates[0][2]


def arrow_nudge_xy(window, step: float = 0.003) -> np.ndarray:
    """Map arrow keys to a world XY delta (viewer top-down friendly)."""
    dx = dy = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    return np.array([dx, dy], dtype=np.float64)


def set_actor_pose(actor, xyz, quat=None):
    import sapien

    if quat is None:
        quat = list(actor.get_pose().q)
    actor.actor.set_pose(sapien.Pose(list(xyz), list(quat)))


def hold_dynamic_at(rigid, actor, xyz, quat=None):
    """Keep a dynamic body pinned at ``xyz`` for keyboard teleop hold."""
    import sapien

    if quat is None:
        quat = list(actor.get_pose().q)
    pose = sapien.Pose(list(xyz), list(quat))
    actor.actor.set_pose(pose)
    if rigid is None:
        return
    try:
        rigid.set_disable_gravity(True)
        rigid.set_linear_velocity([0.0, 0.0, 0.0])
        rigid.set_angular_velocity([0.0, 0.0, 0.0])
        if hasattr(rigid, "set_kinematic_target"):
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                pass
    except Exception:
        pass


def release_dynamic(rigid):
    if rigid is None:
        return
    try:
        rigid.set_kinematic(False)
    except Exception:
        pass
    try:
        rigid.set_disable_gravity(False)
        rigid.set_linear_velocity([0.0, 0.0, 0.0])
        rigid.set_angular_velocity([0.0, 0.0, 0.0])
    except Exception:
        pass


# Last interactive result for GUI exit codes (household_task_gui convention):
#   0 = SUCCESS, 10 = FAILURE, 2 = closed before a result.
_LAST_TASK_RESULT: bool | None = None
_LAST_TASK_DETAIL: str | None = None
_LAST_EPISODE_CONDITION: str | None = None


def task_result_exit_code(ok: bool | None = None) -> int:
    """Map SUCCESS / FAILURE / no-result to launcher exit codes."""
    if ok is None:
        ok = _LAST_TASK_RESULT
    if ok is True:
        return 0
    if ok is False:
        return 10
    return 2


def _normalize_result_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    text = str(detail).strip()
    return text or None


def _resolve_fail_reason(env, ok: bool | None, detail: str | None) -> str | None:
    """Cause of a failed episode: caller detail, then ``env.get_fail_reason()``."""
    if ok is True:
        return None
    text = _normalize_result_detail(detail)
    if text:
        return text
    getter = getattr(env, "get_fail_reason", None)
    if callable(getter):
        try:
            return _normalize_result_detail(getter())
        except Exception:
            return None
    for attr in ("_last_fail_reason", "_fail_reason"):
        text = _normalize_result_detail(getattr(env, attr, None))
        if text:
            return text
    return None


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def experiment_session_enabled() -> bool:
    flag = os.environ.get("ROBODYNA_EXPERIMENT", "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def _resolve_catalog_option_label(env) -> str | None:
    """Base suite catalog tag: ``default`` / ``opt1`` / ``opt2`` / ``opt1+2``."""
    scenario = (
        os.environ.get("ROBODYNA_SCENARIO", "").strip()
        or str(getattr(env, "interactive_scenario", None) or "").strip()
    )
    return scenario or None


def _compute_episode_timing(env) -> dict:
    """Live timing snapshot: total_time_sim_s / wall_s / steps."""
    steps = int(getattr(env, "_exp_sim_steps", 0) or 0)
    dt = None
    try:
        dt = float(env.scene.get_timestep())
    except Exception:
        dt = None
    wall_start = getattr(env, "_exp_wall_start", None)
    wall_s = (time.perf_counter() - wall_start) if wall_start else None
    sim_s = None if dt is None else round(steps * float(dt), 6)
    return {
        "total_time_sim_s": sim_s,
        "wall_s": None if wall_s is None else round(float(wall_s), 4),
        "steps": steps,
    }


def _frozen_episode_timing(env) -> dict:
    """Timing frozen at first ``report_task_result`` (or live if not reported yet)."""
    cached = getattr(env, "_last_episode_timing", None)
    if isinstance(cached, dict) and "steps" in cached:
        return dict(cached)
    timing = _compute_episode_timing(env)
    env._last_episode_timing = dict(timing)
    return dict(timing)


def _timing_payload(timing: dict) -> dict:
    """Canonical timing keys plus legacy aliases for older log readers."""
    sim_s = timing.get("total_time_sim_s")
    wall_s = timing.get("wall_s")
    steps = timing.get("steps")
    return {
        "total_time_sim_s": sim_s,
        "wall_s": wall_s,
        "steps": steps,
        # Legacy aliases
        "wall_clock_s": wall_s,
        "simulation_s": sim_s,
        "simulation_steps": steps,
    }


def maybe_attach_episode_timing(env) -> None:
    """Count sim steps / wall time for every interactive episode."""
    if getattr(env, "_exp_timing_installed", False):
        return
    scene = getattr(env, "scene", None)
    if scene is None or not hasattr(scene, "step"):
        return
    env._exp_timing_installed = True
    env._exp_wall_start = None
    env._exp_sim_steps = 0
    orig_step = scene.step

    def _step_and_count(*args, **kwargs):
        if getattr(env, "_exp_wall_start", None) is None:
            env._exp_wall_start = time.perf_counter()
        result = orig_step(*args, **kwargs)
        env._exp_sim_steps = int(getattr(env, "_exp_sim_steps", 0) or 0) + 1
        tracker = getattr(env, "_metrics_tracker", None)
        if tracker is not None and int(env._exp_sim_steps) % 10 == 0:
            try:
                tracker.on_step()
            except Exception:
                pass
        return result

    try:
        scene.step = _step_and_count
    except Exception:
        env._exp_timing_installed = False


def maybe_attach_experiment_metrics(env) -> None:
    """Install episode timing; attach EvalMetricsTracker in experiment mode."""
    maybe_attach_episode_timing(env)
    if not experiment_session_enabled():
        return
    if getattr(env, "_metrics_tracker", None) is not None:
        return
    try:
        script_dir = str(REPO_ROOT / "script")
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from eval_metrics import EvalMetricsTracker

        args = {
            "use_dynamic": bool(getattr(env, "use_dynamic", False)),
        }
        tracker = EvalMetricsTracker(env, args)
        tracker.on_episode_start()
        env._metrics_tracker = tracker
    except Exception:
        pass


def _env_partial_score(env) -> float | None:
    """Best-effort ``get_score()`` → float in ``[0, 1]``, or ``None`` if unavailable.

    If the interactive loop latched a score at the terminal decision frame
    (``env._latched_partial_score``), prefer that so a 2s post-result hold
    cannot change PS after FAILURE/SUCCESS was already decided.
    """
    latched = getattr(env, "_latched_partial_score", None)
    if latched is not None:
        try:
            return float(latched)
        except (TypeError, ValueError):
            pass
    getter = getattr(env, "get_score", None)
    if not callable(getter):
        return None
    try:
        return float(getter())
    except Exception:
        return None


def format_ps_suffix(score: float | None) -> str:
    """Status / log suffix like `` PS 0.75`` (empty if score missing)."""
    if score is None:
        return ""
    try:
        return f" PS {float(score):.2f}"
    except (TypeError, ValueError):
        return ""


def _collect_episode_metrics(env, ok: bool | None, detail: str | None) -> dict:
    fail_reason = _resolve_fail_reason(env, ok, detail) if ok is not True else None
    metrics: dict = {
        "success": None if ok is None else bool(ok),
        "fail_reason": fail_reason or "",
        "condition": _LAST_EPISODE_CONDITION or "",
        "task": _resolve_task_name(env),
    }
    seed_raw = os.environ.get("ROBODYNA_SEED", "").strip()
    if seed_raw:
        try:
            metrics["seed"] = int(seed_raw)
        except ValueError:
            metrics["seed"] = seed_raw
    option = _resolve_catalog_option_label(env)
    if option:
        metrics["option_label"] = option
    timing = _frozen_episode_timing(env)
    metrics["total_time_sim_s"] = timing.get("total_time_sim_s")
    metrics["wall_s"] = timing.get("wall_s")
    metrics["steps"] = timing.get("steps")
    ps = _env_partial_score(env)
    if ps is not None:
        metrics["partial_score"] = float(ps)
    detail_getter = getattr(env, "get_score_detail", None)
    if callable(detail_getter):
        try:
            partial_detail = detail_getter()
            if isinstance(partial_detail, dict) and partial_detail:
                metrics["partial_score_detail"] = partial_detail
                if ok is not True and not metrics.get("fail_reason"):
                    try:
                        from envs.utils.partial_score import fail_reason_from_score_detail

                        nested = _normalize_result_detail(
                            fail_reason_from_score_detail(partial_detail)
                        )
                    except Exception:
                        nested = None
                    if nested:
                        metrics["fail_reason"] = nested
        except Exception:
            pass
    tracker = getattr(env, "_metrics_tracker", None)
    if tracker is not None:
        try:
            episode = tracker.get_episode_metrics(
                bool(ok),
                fail_reason=metrics.get("fail_reason") or detail,
                seed=metrics.get("seed"),
            )
            metrics.update(
                {
                    "manipulation_score": float(episode.manipulation_score),
                    "route_completion": float(episode.route_completion),
                    "total_penalty_factor": float(episode.total_penalty_factor),
                    "steps_taken": int(episode.steps_taken),
                    "max_steps": int(episode.max_steps),
                    "penalty_events": [
                        {
                            "event_type": p.event_type,
                            "timestep": int(p.timestep),
                            "penalty_factor": float(p.penalty_factor),
                            "details": p.details,
                        }
                        for p in (episode.penalty_events or [])
                    ],
                }
            )
        except Exception:
            pass
    compute = getattr(env, "_compute_metrics", None)
    if callable(compute):
        try:
            extra = compute()
            if isinstance(extra, dict):
                metrics["task_metrics"] = extra
                if ok is not True and not metrics.get("fail_reason"):
                    nested = _normalize_result_detail(extra.get("fail_reason"))
                    if nested:
                        metrics["fail_reason"] = nested
        except Exception:
            pass
    return metrics


def _interactive_result_extras(env, ok: bool | None, detail: str | None) -> dict:
    extras: dict = {}
    control = os.environ.get("ROBODYNA_CONTROL", "").strip()
    if control:
        extras["controller"] = control
    elif getattr(env, "_interactive_robot_mode", None) is not None:
        extras["controller"] = (
            CONTROL_ROBOT if env._interactive_robot_mode else CONTROL_KEYBOARD_MOUSE
        )
    seed_raw = os.environ.get("ROBODYNA_SEED", "").strip()
    if seed_raw:
        try:
            extras["seed"] = int(seed_raw)
        except ValueError:
            extras["seed"] = seed_raw
    option = _resolve_catalog_option_label(env)
    if option:
        extras["scenario"] = option
        extras["option_label"] = option

    # Freeze timing at metric-build / report time (not after the post-success hold).
    timing = _frozen_episode_timing(env)
    extras["time"] = _timing_payload(timing)
    extras["metrics"] = _collect_episode_metrics(env, ok, detail)
    fail_reason = extras["metrics"].get("fail_reason") or ""
    extras["fail_reason"] = fail_reason if ok is not True else ""
    ps = _env_partial_score(env)
    if ps is not None:
        extras["partial_score"] = float(ps)
    env._last_result_extras = extras
    return extras


def _persist_task_result(
    ok: bool | None,
    detail: str | None,
    *,
    condition: str | None = None,
    extra: dict | None = None,
) -> None:
    """Write the latest result for a parent GUI launcher, if requested."""
    global _LAST_EPISODE_CONDITION
    if condition is not None:
        _LAST_EPISODE_CONDITION = _normalize_result_detail(condition)
    path = os.environ.get(TASK_RESULT_ENV)
    if not path:
        return
    try:
        existing = json.loads(Path(path).read_text(encoding="utf-8") or "{}")
    except Exception:
        existing = {}
    payload = dict(existing) if isinstance(existing, dict) else {}
    payload["ok"] = ok
    payload["detail"] = detail or ""
    payload["condition"] = _LAST_EPISODE_CONDITION or ""
    if extra:
        payload.update(extra)
    if ok is not True:
        fail_reason = (
            _normalize_result_detail(payload.get("fail_reason"))
            or _normalize_result_detail(
                (payload.get("metrics") or {}).get("fail_reason")
                if isinstance(payload.get("metrics"), dict)
                else None
            )
            or payload.get("detail")
            or ""
        )
        payload["fail_reason"] = fail_reason
        if not payload.get("detail") and fail_reason:
            payload["detail"] = fail_reason
    elif "fail_reason" not in payload:
        payload["fail_reason"] = ""
    try:
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, default=_json_safe),
            encoding="utf-8",
        )
    except OSError:
        pass


def _mark_interactive_recording_started() -> None:
    """Tell the parent GUI the viewer is done and HDF5 write has begun."""
    _persist_task_result(
        _LAST_TASK_RESULT,
        _LAST_TASK_DETAIL,
        extra={"recording": True},
    )


def _resolve_task_name(env, task: str | None = None) -> str:
    if task:
        return str(task)
    name = getattr(env, "task_name", None) or getattr(env, "TASK_NAME", None)
    if name:
        return str(name)
    return type(env).__name__


def format_episode_condition(env, task: str | None = None) -> str:
    """Human-readable episode-specific goal (color, fill target, side, …)."""
    task = _resolve_task_name(env, task)
    parts: list[str] = []

    if task == "catch_marbles_trapdoors":
        names = list(getattr(env, "button_color_names", None) or getattr(env, "color_order", []) or [])
        idx = int(getattr(env, "target_button_idx", -1))
        color = names[idx] if 0 <= idx < len(names) else "?"
        parts.append(f"target marble={color}")
        if names:
            parts.append(f"doors L→R={'/'.join(str(c) for c in names)}")

    elif task == "dispense_gummy":
        color = str(getattr(env, "target_color", "?") or "?")
        parts.append(f"target gummy={color}")
        try:
            distractor = env._distractor_color()
            parts.append(f"distractor={distractor}")
        except Exception:
            pass

    elif task == "measure_ingredient":
        tgt = float(getattr(env, "target_fill", 0.0))
        tol = float(getattr(env, "fill_tol", 0.05))
        parts.append(f"target fill={tgt:.0%}±{tol:.0%}")

    elif task == "fill_coffee_jar":
        tgt = float(getattr(env, "target_fill", 0.0))
        tol = float(getattr(env, "fill_tol", 0.05))
        parts.append(f"target fill={tgt:.0%}±{tol:.0%}")
        try:
            beans = int(env._beans_needed())
            parts.append(f"~{beans} beans")
        except Exception:
            pass

    elif task == "boil_milk":
        ring = float(getattr(env, "target_ring", 100.0 * float(getattr(env, "target_level", 0.8))))
        parts.append(f"target ring={ring:.0f}%")

    elif task == "pour_beer":
        tgt = 100.0 * float(getattr(env, "target_liquid", 0.90))
        parts.append(f"need beer>{tgt:.0f}%")
        parts.append("then click finish bell")

    elif task in ("cook_food", "cook_food_timer"):
        food = str(getattr(env, "food_type", "") or "")
        if food:
            parts.append(f"food={food}")
        rng = getattr(env, "target_doneness_range", None)
        if rng is not None and len(rng) >= 2:
            parts.append(f"doneness={float(rng[0]):.0%}–{float(rng[1]):.0%}")

    elif task in ("cook_meat", "cook_meat_timer"):
        rng = getattr(env, "target_doneness_range", None)
        if rng is not None and len(rng) >= 2:
            parts.append(f"doneness={float(rng[0]):.0%}–{float(rng[1]):.0%}")

    elif task == "play_billiard":
        name = getattr(env, "_target_pocket_name", None)
        pid = getattr(env, "_target_pocket_id", None)
        if getattr(env, "specific_hole", False) and name not in (None, "any"):
            parts.append(f"target pocket={name}")
        elif pid is not None:
            parts.append(f"target pocket id={pid}")
        else:
            parts.append("target pocket=any top")

    elif task == "pick_ripe_apple":
        side = getattr(env, "apple_side", None)
        if side is not None:
            parts.append(f"ripe apple={'left' if float(side) < 0 else 'right'}")

    elif task == "load_train":
        side = getattr(env, "ball_side", None)
        if side:
            parts.append(f"ball side={side}")
        if bool(getattr(env, "target_wagon_mode", False)):
            widx = getattr(env, "target_wagon_idx", None)
            parts.append(f"target wagon={widx if widx is not None else '?'}")

    elif task == "save_goal":
        mirrored = bool(getattr(env, "mirrored", False))
        parts.append(f"keeper side={'left' if mirrored else 'right'}")
        speed = getattr(env, "ball_speed", None)
        if speed is not None:
            parts.append(f"ball speed={float(speed):.4f} m/s")

    elif task == "catch_cuboid":
        color = getattr(env, "cuboid_color", None)
        if color is not None:
            try:
                rgb = [float(c) for c in list(color)[:3]]
                parts.append(f"cuboid rgb=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})")
            except Exception:
                parts.append(f"cuboid color={color}")
        if bool(getattr(env, "catch_two_cuboids", False)):
            parts.append("catch both cuboids")

    elif task == "punch_dual_holes":
        if str(getattr(env, "missing_tile_mode", "none")) != "none":
            parts.append(
                f"missing tile={getattr(env, 'missing_tile_side', '?')}#"
                f"{getattr(env, 'missing_tile_index', '?')}"
            )

    elif task == "sort_apples_belt":
        mode = getattr(env, "color_mode", None)
        if mode:
            parts.append(f"color mode={mode}")

    elif task == "make_soup":
        veggies = getattr(env, "veggies", None)
        if veggies is not None:
            names = []
            for v in list(veggies):
                name = getattr(v, "get_name", None)
                try:
                    names.append(str(name() if callable(name) else getattr(v, "name", type(v).__name__)))
                except Exception:
                    names.append("veg")
            parts.append(f"produce={len(veggies)}" + (f" ({', '.join(names)})" if names else ""))

    elif task == "trap_bug":
        bug = getattr(env, "bug_type", None) or getattr(env, "_bug_type", None)
        if bug:
            parts.append(f"bug={bug}")

    return "; ".join(parts)


def print_episode_condition(env, task: str | None = None) -> str:
    """Print and persist the episode-specific condition; return the text (may be empty).

    Idempotent per env instance so callers may invoke both after ``setup_demo``
    and from ``run_viewer_loop`` without duplicate lines.
    """
    global _LAST_EPISODE_CONDITION
    maybe_attach_interactive_data_recorder(env)
    maybe_attach_experiment_metrics(env)
    if bool(getattr(env, "_episode_condition_printed", False)):
        return str(getattr(env, "_episode_condition_text", "") or "")
    text = format_episode_condition(env, task)
    task_name = _resolve_task_name(env, task)
    env._episode_condition_printed = True
    env._episode_condition_text = text
    if text:
        print_instructions(f"[{task_name}] condition: {text}")
        _LAST_EPISODE_CONDITION = text
        _persist_task_result(None, None, condition=text)
    return text


def report_task_result(env, detail: str | None = None, *, ok: bool | None = None) -> bool:
    """Print ``Task complete: SUCCESS|FAILURE`` from ``check_success``; return success.

    Pass ``ok=`` to override ``check_success`` (keyboard shortcuts that extract
    objects without a gripper latch still need a terminal result).

    Also stores the result for ``task_result_exit_code()`` so ``base_task_gui``
    can show SUCCESS/FAILURE like ``household_task_gui``. When ``ROBODYNA_TASK_RESULT_FILE``
    is set, persists ``ok`` + failure/success ``detail`` for the GUI status line.
    """
    global _LAST_TASK_RESULT, _LAST_TASK_DETAIL
    try:
        if ok is None:
            ok = bool(env.check_success())
        else:
            ok = bool(ok)
    except Exception as exc:
        detail = _normalize_result_detail(f"check_success error: {exc}")
        print_failure(f"Task complete: FAILURE ({detail})")
        _LAST_TASK_RESULT = False
        _LAST_TASK_DETAIL = detail
        _persist_task_result(False, detail, extra=_interactive_result_extras(env, False, detail))
        return False
    detail = _normalize_result_detail(detail)
    if not ok:
        detail = _resolve_fail_reason(env, False, detail)
    status = "SUCCESS" if ok else "FAILURE"
    ps_note = format_ps_suffix(_env_partial_score(env))
    msg = f"Task complete: {status}" + (f" ({detail})" if detail else "") + ps_note
    if ok:
        print_success(msg)
    else:
        print_failure(msg)
    _LAST_TASK_RESULT = bool(ok)
    _LAST_TASK_DETAIL = detail
    _persist_task_result(
        _LAST_TASK_RESULT,
        detail,
        extra=_interactive_result_extras(env, _LAST_TASK_RESULT, detail),
    )
    return ok


def resolve_head_camera(env=None, viewer=None):
    """Find the sapien ``head_camera`` render camera on ``env`` or ``viewer``."""
    cam = resolve_named_static_camera(env, "head_camera")
    if cam is not None:
        return cam
    if viewer is None and env is not None:
        viewer = getattr(env, "viewer", None)
    if viewer is not None:
        for cam in getattr(viewer, "cameras", []) or []:
            if getattr(cam, "name", None) == "head_camera":
                return cam
    return None


def resolve_named_static_camera(env, name: str):
    """Return the sapien static camera named ``name``, if it exists."""
    if env is None or not name:
        return None
    cams = getattr(env, "cameras", None)
    if cams is None:
        return None
    names = list(getattr(cams, "static_camera_name", []) or [])
    clist = list(getattr(cams, "static_camera_list", []) or [])
    if name in names:
        return clist[names.index(name)]
    return None


def align_demo_camera_to_head(env) -> bool:
    """Point the HD ``demo_camera`` at the ``head_camera`` view.

    Interactive / experiment capture records ``demo_camera`` at 1280x960 while
    ``head_camera`` stays 320x240 for the dataset. Both are 4:3, so copying pose
    + fovy makes the HD stream show exactly what the viewer showed. Call after
    ``configure_standard_head_camera``.
    """
    head = resolve_named_static_camera(env, "head_camera")
    demo = resolve_named_static_camera(env, "demo_camera")
    if head is None or demo is None:
        return False
    try:
        demo.entity.set_pose(head.entity.get_pose())
    except Exception:
        return False
    try:
        demo.set_fovy(float(head.fovy))
    except Exception:
        try:
            demo.fovy = float(head.fovy)
        except Exception:
            pass
    return True


def resolve_wrist_camera_link(env, side: str):
    """Return the robot wrist/camera link for ``left`` or ``right``, if any."""
    robot = getattr(env, "robot", None) if env is not None else None
    if robot is None or side not in ("left", "right"):
        return None
    return getattr(robot, f"{side}_camera", None)


def resolve_wrist_render_camera(env, side: str):
    """Return the sapien wrist render camera when ``collect_wrist_camera`` is on."""
    cams = getattr(env, "cameras", None) if env is not None else None
    if cams is None or not bool(getattr(cams, "collect_wrist_camera", False)):
        return None
    if side not in ("left", "right"):
        return None
    return getattr(cams, f"{side}_camera", None)


def active_gripper_sides(env) -> tuple[str, ...]:
    """Sides whose gripper/wrist views V may cycle through.

    Uses ``env._interactive_selected_arms`` when set (after 1/2/3). Otherwise both
    sides that have a wrist camera link. Missing links are dropped so single-arm /
    no-wrist setups degrade.
    """
    available = []
    for side in ("right", "left"):
        if resolve_wrist_camera_link(env, side) is not None:
            available.append(side)
    if not available:
        return ()
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    selected = tuple(s for s in selected if s in ("left", "right"))
    if not selected:
        # Prefer right-then-left ordering for dual-arm cycling.
        return tuple(s for s in ("right", "left") if s in available)
    ordered = []
    for side in ("right", "left"):
        if side in selected and side in available:
            ordered.append(side)
    return tuple(ordered)


# Nadir viewer: +X camera offset shifts the table left in the frame.
_TOPDOWN_VIEW_X_OFFSET = 0.08


def default_topdown_xyz(env=None) -> tuple[float, float, float]:
    """Overhead pose: table near image center, zoomed to table + dual arms."""
    bias = getattr(env, "table_xy_bias", None) if env is not None else None
    if bias is None:
        bx = by = 0.0
    else:
        bx = float(bias[0])
        by = float(bias[1])
    # Z ≈ 1.68 with ~65° fovy keeps both arm bases (x≈±embodiment_dis/2) in frame.
    return (bx + _TOPDOWN_VIEW_X_OFFSET, by, 1.68)


def gripper_width(env, side: str) -> float:
    """Normalized gripper opening in ``[0, 1]`` (1 = open, 0 = closed)."""
    robot = getattr(env, "robot", None)
    if robot is None:
        return 0.0
    if side == "left":
        return float(robot.get_left_gripper_val())
    return float(robot.get_right_gripper_val())


def toggle_selected_grippers(env, *, fallback=("left", "right"), threshold: float = 0.5) -> bool:
    """Edge-action helper: open ↔ close the highlighted gripper(s).

    Uses the current 1/2/3 selection when present; otherwise ``fallback``.
    Per selected arm: width ``> threshold`` → close (0), else open (1).
    ``trap_bug`` closes only to the outer-wall hold width (not crushed shut).
    Applies via ``robot.set_gripper`` (non-blocking) so teleop stays responsive.
    """
    robot = getattr(env, "robot", None)
    if robot is None or not hasattr(robot, "set_gripper"):
        print("No gripper available to open/close.")
        return False
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if not selected:
        if bool(getattr(env, "_interactive_universal_controls", False)):
            print("Select an arm first: 1 (left), 2 (right), or 3 (both).")
            return False
        selected = tuple(fallback)
    trap_hold = None
    if type(env).__name__ == "trap_bug":
        trap_hold = float(getattr(env, "trap_hold_gripper", 0.78))
    actions = []
    for side in selected:
        if side not in ("left", "right"):
            continue
        width = gripper_width(env, side)
        if trap_hold is not None:
            # Pinch outer walls (~0.78). Fully closed (0) tunnels the hollow box.
            # Open when already pinching / welded; otherwise close to hold width.
            if bool(getattr(env, "_trap_welded", False)) or width <= trap_hold + 0.05:
                target = 1.0
                label = "open"
            else:
                target = trap_hold
                label = "pinch"
        else:
            target = 0.0 if width > float(threshold) else 1.0
            label = "open" if target > 0.5 else "closed"
        try:
            robot.set_gripper(target, side, gripper_eps=0.0)
        except Exception as exc:
            print(f"Gripper toggle failed ({side}): {exc}")
            continue
        actions.append(f"{side}={label}")
    if not actions:
        return False
    print("Gripper: " + ", ".join(actions))
    return True


class ViewerViewToggle:
    """V cycles head_camera ↔ gripper/wrist views (no top-down).

    Head view matches GUI ``scene_snapshot`` / training ``head_camera`` RGB
    (pose + fovy). Space (edge) opens/closes the selected gripper(s) via
    ``toggle_selected_grippers``. Camera switching is V-only.

    sapien's ``focus_camera`` follow-path is disabled in this build
    (``_handle_focused_camera`` commented out), so we copy the active camera
    pose onto the free-fly viewer each frame instead.
    """

    # Kept for callers / legacy ``overhead=`` framing helpers; not used by V.
    DEFAULT_TOPDOWN_XYZ = (_TOPDOWN_VIEW_X_OFFSET, 0.0, 1.68)
    DEFAULT_TOPDOWN_RPY = (0.0, -np.pi / 2.0, -np.pi / 2.0)
    DEFAULT_TOPDOWN_FOVY = float(np.deg2rad(65.0))
    # Fallback only: shared D435 head fovy when render camera fovy is missing.
    try:
        from envs.utils.household_view import HEAD_CAMERA_FOVY as _HEAD_FOVY
    except Exception:  # pragma: no cover - import during partial bootstraps
        _HEAD_FOVY = float(np.deg2rad(37.0))
    DEFAULT_HEAD_FOVY = float(_HEAD_FOVY)
    # D435-ish wrist fallback when collect_wrist_camera is off.
    DEFAULT_GRIPPER_FOVY = float(np.deg2rad(42.0))
    _GRIPPER_MODES = ("right_gripper", "left_gripper")

    def __init__(
        self,
        viewer,
        env=None,
        head_camera=None,
        topdown_xyz=None,
        topdown_rpy=None,
        topdown_fovy=None,
        capture_current_as_topdown: bool = False,
        warn_missing_head: bool = False,
        robot_controls=None,
        allow_f_gripper: bool = False,
    ):
        # topdown_* / capture_current_as_topdown kept for API compat; ignored.
        # allow_f_gripper kept for API compat; F is never a gripper alias.
        del topdown_xyz, topdown_rpy, topdown_fovy, capture_current_as_topdown
        del allow_f_gripper
        self.viewer = viewer
        self.env = env
        self._prev_v = False
        self._prev_space = False
        self._head = head_camera
        self.robot_controls = robot_controls
        self._warned_missing_gripper = False
        if self._head is None:
            self._head = resolve_head_camera(env, viewer)
        if self._head is None and warn_missing_head:
            print("Warning: head_camera not found; V will cycle gripper views only.")

        self._disable_wasd_camera_move()
        modes = self._view_cycle_modes()
        self.mode = modes[0] if modes else "head"
        self.apply(announce=False)

    def _control_window(self):
        for plugin in getattr(self.viewer, "plugins", []) or []:
            if hasattr(plugin, "fps_camera_controller") and hasattr(plugin, "set_camera_xyz"):
                return plugin
        return getattr(self.viewer, "control_window", None)

    def _disable_wasd_camera_move(self):
        """Stop SAPIEN's FPS controller from translating the view on W/A/S/D.

        Interactive tasks own the camera (head_camera / gripper follow),
        so free-fly WASD only fights the fixed framing.
        """
        cw = self._control_window()
        if cw is not None and hasattr(cw, "move_speed"):
            try:
                cw.move_speed = 0.0
            except Exception:
                pass

    def _set_fovy(self, fovy: float):
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        try:
            near = float(getattr(window, "near", 0.1))
            far = float(getattr(window, "far", 1000.0))
            window.set_camera_parameters(near, far, float(fovy))
        except Exception:
            try:
                cw = self._control_window()
                if cw is not None:
                    cw.fovy = float(fovy)
            except Exception:
                pass

    def _camera_fovy(self, camera, default: float) -> float:
        try:
            fovy = float(camera.fovy)
            if np.isfinite(fovy) and fovy > 0.0:
                return fovy
        except Exception:
            pass
        return float(default)

    def _head_fovy(self) -> float:
        """Use the render camera's lens so head view keeps its intended framing."""
        return self._camera_fovy(self._head, self.DEFAULT_HEAD_FOVY)

    def _set_viewer_pose(self, pose):
        """Snap free-fly camera to ``pose`` and keep the FPS controller in sync."""
        try:
            self.viewer.focus_camera(None)
        except Exception:
            pass
        self.viewer.set_camera_pose(pose)
        # Mirror ControlWindow._sync_fps_camera_controller so mouse/WASD (when
        # re-enabled) start from the same orientation we just applied. Prefer
        # setXYZ/setRPY over the no-op FPSCameraController.pose setter.
        cw = self._control_window()
        if cw is None:
            return
        fps = getattr(cw, "fps_camera_controller", None)
        if fps is None:
            return
        try:
            from transforms3d.euler import quat2euler

            fps.setXYZ(*np.asarray(pose.p, dtype=np.float64))
            r, p, y = quat2euler(np.asarray(pose.q, dtype=np.float64))
            fps.setRPY(r, -p, -y)
        except Exception:
            if hasattr(cw, "_sync_fps_camera_controller"):
                try:
                    cw._sync_fps_camera_controller()
                except Exception:
                    pass

    def _head_render_pose(self):
        """Pose of the sapien head render camera (matches GUI snapshot RGB)."""
        head = self._head
        if head is None:
            return None
        try:
            return head.global_pose
        except Exception:
            pass
        try:
            return head.entity.get_pose()
        except Exception:
            return None

    def _apply_head_view(self, announce: bool = False):
        """Lock free-fly viewer to head_camera pose + fovy (GUI snapshot view)."""
        pose = self._head_render_pose()
        if pose is None:
            if announce:
                print("View: head_camera unavailable.")
            return
        self._set_fovy(self._head_fovy())
        self._set_viewer_pose(pose)
        if announce:
            print("View: head_camera")

    def _view_cycle_modes(self) -> list[str]:
        """V targets: head_camera then each active gripper/wrist view."""
        modes: list[str] = []
        if self._head is not None:
            modes.append("head")
        for side in active_gripper_sides(self.env):
            modes.append(f"{side}_gripper")
        return modes

    def _gripper_side(self) -> str | None:
        if self.mode == "right_gripper":
            return "right"
        if self.mode == "left_gripper":
            return "left"
        return None

    def _wrist_pose(self, side: str):
        """Current wrist camera pose, syncing the render camera when available."""
        link = resolve_wrist_camera_link(self.env, side)
        if link is None:
            return None
        try:
            pose = link.get_pose() if hasattr(link, "get_pose") else link.global_pose
        except Exception:
            return None
        render_cam = resolve_wrist_render_camera(self.env, side)
        if render_cam is not None:
            try:
                render_cam.entity.set_pose(pose)
            except Exception:
                pass
            try:
                return render_cam.global_pose
            except Exception:
                return pose
        return pose

    def _gripper_viewer_pose(self, pose):
        """Keep wrist look direction; roll so screen axes match world teleop.

        Arrow keys are world-fixed (right=+X, up=+Y). Roll the gripper camera
        about its look (+X) so screen-up ≈ world +Y when possible.
        """
        import sapien
        from transforms3d.quaternions import mat2quat, quat2mat

        R = np.asarray(quat2mat(pose.q), dtype=np.float64)
        look = R[:, 0].copy()
        look_n = float(np.linalg.norm(look))
        if look_n < 1e-9:
            return pose
        look /= look_n

        # Prefer world +Y as screen-up (same as arrow-up).
        ref_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = ref_up - look * float(np.dot(ref_up, look))
        if float(np.linalg.norm(up)) < 1e-6:
            ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            up = ref_up - look * float(np.dot(ref_up, look))
        up /= float(np.linalg.norm(up)) + 1e-12
        # FPS convention: left = cross(up, forward).
        left = np.cross(up, look)
        left /= float(np.linalg.norm(left)) + 1e-12
        up = np.cross(look, left)
        up /= float(np.linalg.norm(up)) + 1e-12
        R_new = np.column_stack((look, left, up))
        return sapien.Pose(p=pose.p, q=mat2quat(R_new))

    def _gripper_fovy(self, side: str) -> float:
        render_cam = resolve_wrist_render_camera(self.env, side)
        if render_cam is not None:
            return self._camera_fovy(render_cam, self.DEFAULT_GRIPPER_FOVY)
        return self.DEFAULT_GRIPPER_FOVY

    def apply(self, announce=True):
        side = self._gripper_side()
        if side is not None:
            pose = self._wrist_pose(side)
            if pose is None:
                if announce:
                    print(f"View: {side} gripper unavailable; staying put.")
                return
            self._set_fovy(self._gripper_fovy(side))
            self._set_viewer_pose(self._gripper_viewer_pose(pose))
            if announce:
                print(f"View: {side} gripper")
            return
        if self.mode != "head" or self._head is None:
            # Recover to the first available mode (head preferred).
            modes = self._view_cycle_modes()
            if not modes:
                if announce:
                    print("No head/gripper camera available.")
                return
            self.mode = modes[0]
            if self.mode != "head":
                return self.apply(announce=announce)
        self._apply_head_view(announce=announce)

    def _edge_key(self, window, key: str, prev_attr: str) -> bool:
        down = bool(window.key_down(key))
        edge = down and not getattr(self, prev_attr)
        setattr(self, prev_attr, down)
        if edge:
            return True
        try:
            return bool(window.key_press(key))
        except Exception:
            return False

    def _v_pressed(self, window) -> bool:
        return self._edge_key(window, "v", "_prev_v")

    def _space_pressed(self, window) -> bool:
        return self._edge_key(window, "space", "_prev_space")

    def _cycle_view(self):
        """Cycle head_camera ↔ active gripper/wrist view(s)."""
        modes = self._view_cycle_modes()
        if not modes:
            print("No head/gripper camera available.")
            return
        if self.mode not in modes:
            self.mode = modes[0]
            self.apply(announce=True)
            return
        if len(modes) == 1:
            label = "head_camera" if modes[0] == "head" else modes[0].replace("_", " ")
            print(f"Only one view available ({label}).")
            return
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        self.apply(announce=True)

    def update(self, window):
        if self.robot_controls is not None:
            self.robot_controls.update(window)
        elif self.env is not None:
            # Restore red failure tint when UniversalRobotControls is absent
            # (keyboard mode still uses Space gripper / task action paths).
            gripper_failure_feedback(self.env).update()
        arms_gone = bool(
            getattr(self.env, "_interactive_arms_removed", False)
        ) if self.env is not None else False
        # Space opens/closes selected gripper(s) — only when arms are present.
        # Keyboard+mouse has no grippers; stay on default head_camera (no V cycle).
        if self.env is not None and not arms_gone and self._space_pressed(window):
            toggle_selected_grippers(self.env)
        if not arms_gone and self._v_pressed(window):
            self._cycle_view()
            return
        # Keep head / gripper views locked to the moving cameras.
        # Head also re-applies fovy so the free-fly frustum stays matched to
        # head_camera RGB (GUI snapshots / training view).
        if self.mode == "head" and self._head is not None:
            self._apply_head_view(announce=False)
        else:
            side = self._gripper_side()
            if side is not None:
                pose = self._wrist_pose(side)
                if pose is not None:
                    self._set_fovy(self._gripper_fovy(side))
                    self._set_viewer_pose(self._gripper_viewer_pose(pose))


def declutter_interactive_viewer(viewer) -> None:
    """Show only the 3D scene: hide ImGui panels and camera frustum lines.

    Keeps ControlWindow camera/input logic; suppresses its (and every other
    plugin's) black side windows. Safe to call more than once.
    """
    if viewer is None:
        return
    for plugin in getattr(viewer, "plugins", []) or []:
        # Drop ImGui windows while leaving before/after_render hooks intact.
        try:
            plugin.get_ui_windows = lambda: []
        except Exception:
            pass
        try:
            if hasattr(plugin, "show_camera_linesets"):
                plugin.show_camera_linesets = False
        except Exception:
            pass
        try:
            if hasattr(plugin, "show_joint_axes"):
                plugin.show_joint_axes = False
        except Exception:
            pass
        try:
            if hasattr(plugin, "show_origin_frame"):
                plugin.show_origin_frame = False
        except Exception:
            pass
        # Contact overlay only (other plugins may reuse ``enabled``).
        try:
            if type(plugin).__name__ == "ContactWindow" and hasattr(plugin, "enabled"):
                plugin.enabled = False
        except Exception:
            pass


def make_viewer_view_toggle(
    env,
    viewer=None,
    topdown_xyz=None,
    topdown_rpy=None,
    capture_current_as_topdown: bool = False,
    **kwargs,
) -> ViewerViewToggle:
    """Build V (head ↔ gripper) view switching for an interactive env.

    Always starts on the shared suite ``head_camera`` framing (same pose/FOV as
    GUI ``scene_snapshot`` cards) for both base and household tasks. Legacy
    ``topdown_*`` / ``capture_current_as_topdown`` kwargs are accepted but ignored.
    """
    if viewer is None:
        viewer = getattr(env, "viewer", None)
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    declutter_interactive_viewer(viewer)
    # Mark + force shared head pose so base interactives match household / GUI.
    try:
        env._interactive_session = True
        from envs.utils.household_view import configure_standard_head_camera

        configure_standard_head_camera(env)
    except Exception:
        pass
    align_demo_camera_to_head(env)
    robot_controls = None
    robot_mode = bool(getattr(env, "_interactive_robot_mode", False))
    control_from_argv = None
    for index, arg in enumerate(sys.argv):
        if arg == "--control" and index + 1 < len(sys.argv):
            control_from_argv = sys.argv[index + 1]
            break
        if arg.startswith("--control="):
            control_from_argv = arg.split("=", 1)[1]
            break
    if control_from_argv is not None:
        robot_mode = is_robot_control(control_from_argv)
    elif not robot_mode:
        # Match add_robot_motion_arg default when --control is omitted.
        robot_mode = True
    # keyboard+mouse: drop arms even if a launcher forgot prepare_interactive_control.
    if not robot_mode:
        strip_interactive_arms(env)
        robot_mode = False
    if robot_mode:
        robot_controls = UniversalRobotControls(env)
    return ViewerViewToggle(
        viewer,
        env=env,
        head_camera=resolve_head_camera(env, viewer),
        topdown_xyz=topdown_xyz,
        topdown_rpy=topdown_rpy,
        capture_current_as_topdown=capture_current_as_topdown,
        robot_controls=robot_controls,
        **kwargs,
    )


class SeededArmIK:
    """Local seeded IK for one arm via SAPIEN's pinocchio CLIK solver.

    Curobo's ``solve_ik`` is unseeded: for a 2 mm nudge it happily returns an
    elbow/wrist flip several radians away, which is useless for incremental
    teleop or press animations. CLIK starts from the current configuration and
    costs ~0.05 ms, so it can run every physics step.
    """

    def __init__(self, env, side):
        robot = env.robot
        self.env = env
        self.side = side
        self.entity = robot.left_entity if side == "left" else robot.right_entity
        ee_joint = robot.left_ee if side == "left" else robot.right_ee
        arm_names = set(robot.left_arm_joints_name if side == "left"
                        else robot.right_arm_joints_name)
        self.model = self.entity.create_pinocchio_model()
        link_names = [link.get_name() for link in self.entity.get_links()]
        self.link_index = link_names.index(ee_joint.get_child_link().get_name())
        self.mask = np.zeros(self.entity.dof, dtype=np.int32)
        dofs = []
        cursor = 0
        for joint in self.entity.get_active_joints():
            width = joint.get_dof()
            if joint.get_name() in arm_names:
                self.mask[cursor:cursor + width] = 1
                dofs.extend(range(cursor, cursor + width))
            cursor += width
        self.arm_dofs = np.asarray(dofs, dtype=int)
        limits = np.asarray(self.entity.get_qlimits(), dtype=np.float64)
        self.lower = limits[self.arm_dofs, 0]
        self.upper = limits[self.arm_dofs, 1]

    def full_qpos(self):
        return np.asarray(self.entity.get_qpos(), dtype=np.float64)

    def solve(self, gripper_pose, seed=None):
        """Return ``(arm_joints, full_qpos)`` for a world gripper pose, or None."""
        world_target = self.env.robot._trans_from_gripper_to_endlink(
            list(np.asarray(gripper_pose, dtype=np.float64)), arm_tag=self.side)
        target = self.entity.get_root_pose().inv() * world_target
        seed = self.full_qpos() if seed is None else np.asarray(seed, dtype=np.float64)
        qpos, success, _ = self.model.compute_inverse_kinematics(
            self.link_index, target, initial_qpos=seed, active_qmask=self.mask,
            eps=1e-4, max_iterations=60, dt=0.2, damp=1e-3)
        if not success:
            return None
        full = np.asarray(qpos, dtype=np.float64)
        arm = np.clip(full[self.arm_dofs], self.lower, self.upper)
        full[self.arm_dofs] = arm
        return arm, full


def arm_ik(env, side):
    """Per-env cached ``SeededArmIK``; ``None`` when the model cannot be built."""
    cache = getattr(env, "_interactive_arm_ik", None)
    if cache is None:
        cache = {}
        env._interactive_arm_ik = cache
    if side not in cache:
        try:
            cache[side] = SeededArmIK(env, side)
        except Exception as exc:
            print(f"Arm IK unavailable ({exc}); robot motion is disabled.")
            cache[side] = None
    return cache[side]


class UniversalRobotControls:
    """Shared arm selection and Cartesian teleoperation for robot-mode tasks.

    Teleop integrates a commanded gripper pose and tracks it with seeded local
    IK (SAPIEN's pinocchio CLIK, ~0.05 ms per solve). Seeding on the previous
    solution keeps the arm in one kinematic branch; Curobo's ``solve_ik`` is
    unseeded and returns elbow/wrist flips that are unusable for teleop.

    F / G tip the gripper about world +Y (left / right) for pour-style motions.
    R / T yaw about world +Z (counter-clockwise / clockwise) continuously via IK.
    O returns the selected arm(s) to the pose captured when teleop started
    (task ``original`` / start pose). Space opens/closes the selected gripper
    via ``ViewerViewToggle`` so it also works when teleop is not attached.

    No arm is highlighted at start — press 1 / 2 / 3 to activate left / right /
    both. The selected gripper turns green; until then both stay gray and
    teleop / Space do nothing.
    Task scripts must not pre-seed ``_interactive_selected_arms``; any leftover
    value is cleared here so every interactive starts unselected.
    """

    # Interactive teleop rates (m/s). 20% slower than the prior snappy sandbox
    # rates; MAX_LEAD / MAX_JOINT_SPEED scale with them or the caps choke motion.
    XY_SPEED = 0.288
    Z_SPEED = 0.224
    # World-Y tip rate for F/G (rad/s) — enough to dump a board without feeling twitchy.
    ROLL_SPEED = 1.28
    # World-Z yaw rate for R/T (rad/s).
    YAW_SPEED = 0.64
    MAX_DT = 0.05
    # How far the commanded pose may run ahead of the achieved pose, so a
    # blocked or joint-limited arm cannot accumulate an unrecoverable lead.
    MAX_LEAD = 0.044
    # Near a singularity a millimetre of Cartesian travel costs radians of
    # joint travel; slew at this cap instead of whipping the arm.
    MAX_JOINT_SPEED = 4.0
    # Q/E world-Z band: ceiling = captured original EE height. Floor keeps the
    # lowest gripper finger AABB at/above the table (measured, not the 0.12
    # EE→TCP proxy which underestimates WSG finger extent).
    EE_TO_TCP_FALLBACK = 0.18
    FINGER_TABLE_CLEARANCE = 0.008

    def __init__(self, env):
        self.env = env
        # Always start with no arm selected — press 1 / 2 / 3 to activate.
        self.selected = ()
        self._previous = {key: False for key in ("1", "2", "3", "o")}
        self._last_update = None
        self._command = {}
        self._highlight_materials = {}
        self._origin_joints = {}
        self._origin_pose = {}
        env._interactive_selected_arms = ()
        env._interactive_universal_controls = True
        env._interactive_robot_controls = self
        # Ensure the shared failure feedback exists for Space/action paths.
        gripper_failure_feedback(env)
        self._capture_origin_poses()
        self._highlight_selected()

    def _capture_origin_poses(self):
        """Snapshot start arm joints / EE poses for the O reset key."""
        robot = getattr(self.env, "robot", None)
        for side in ("left", "right"):
            joints = None
            pose = None
            try:
                joints = self._drive_qpos(side).copy()
            except Exception:
                home = None
                if robot is not None:
                    home = (robot.left_homestate if side == "left"
                            else robot.right_homestate)
                if home is not None:
                    joints = np.asarray(home, dtype=np.float64).copy()
            try:
                pose = self._ee_pose(side).copy()
            except Exception:
                if robot is not None:
                    attr = ("left_original_pose" if side == "left"
                            else "right_original_pose")
                    stored = getattr(robot, attr, None)
                    if stored is not None:
                        pose = np.asarray(stored, dtype=np.float64).copy()
            if joints is not None:
                self._origin_joints[side] = joints
            if pose is not None:
                self._origin_pose[side] = pose

    def _snap_arm_to_joints(self, side, joints):
        """Instant joint reset: drive targets + articulation qpos."""
        robot = self.env.robot
        entity = robot.left_entity if side == "left" else robot.right_entity
        arm_joints = robot.left_arm_joints if side == "left" else robot.right_arm_joints
        target = np.asarray(joints, dtype=np.float64)
        active = entity.get_active_joints()
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        for joint, value in zip(arm_joints, target):
            try:
                idx = active.index(joint)
            except ValueError:
                idx = next(
                    (i for i, a in enumerate(active)
                     if a.get_name() == joint.get_name()),
                    None,
                )
            if idx is None:
                continue
            qpos[idx] = float(value)
            try:
                joint.set_drive_target(float(value))
                joint.set_drive_velocity_target(0.0)
            except Exception:
                pass
        entity.set_qpos(qpos)
        self.env.robot.set_arm_joints(target, np.zeros_like(target), side)

    def _return_selected_to_origin(self):
        """O: snap currently selected arm(s) back to the captured start pose."""
        restored = self.return_arms_to_origin(self.selected, open_grippers=False)
        if restored:
            print("Returned arm(s) to original position: " + " + ".join(restored))
        else:
            print("No original arm pose available to restore.")

    def return_arms_to_origin(self, sides=None, *, open_grippers: bool = True):
        """Snap arm(s) to the teleop start pose (both arms if ``sides`` is None).

        Used by the O key (selected only) and by tutorial stage switches (both
        arms, grippers opened) before the next prop appears.
        """
        if sides is None:
            sides = ("left", "right")
        sides = tuple(sides)
        robot = getattr(self.env, "robot", None)
        if open_grippers and robot is not None and hasattr(robot, "set_gripper"):
            for side in sides:
                try:
                    robot.set_gripper(1.0, str(side), gripper_eps=0.0)
                except Exception:
                    pass
        restored = []
        for side in sides:
            joints = self._origin_joints.get(side)
            if joints is None and robot is not None:
                home = (
                    robot.left_homestate if side == "left" else robot.right_homestate
                )
                if home is not None:
                    joints = np.asarray(home, dtype=np.float64).copy()
                    self._origin_joints[side] = joints
            if joints is None:
                continue
            self._snap_arm_to_joints(side, joints)
            self._command.pop(side, None)
            pose = self._origin_pose.get(side)
            if pose is None:
                try:
                    pose = self._ee_pose(side).copy()
                    self._origin_pose[side] = pose
                except Exception:
                    pose = None
            if pose is not None:
                cmd = getattr(self.env, "_interactive_cmd_pose", None)
                if not isinstance(cmd, dict):
                    cmd = {}
                    self.env._interactive_cmd_pose = cmd
                cmd[side] = pose.copy()
            restored.append(side)
        return restored

    def _highlight_selected(self):
        # A new selection cancels any failure tint so the highlight is readable.
        fb = getattr(self.env, "_interactive_gripper_failure", None)
        if isinstance(fb, GripperFailureFeedback) and fb._original:
            fb.restore()
        for material, color in self._highlight_materials.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass
        self._highlight_materials.clear()
        for side in self.selected:
            articulation = (self.env.robot.left_entity if side == "left"
                            else self.env.robot.right_entity)
            for link in articulation.get_links():
                if link.get_name() not in GRIPPER_LINK_NAMES:
                    continue
                for component in link.entity.get_components():
                    try:
                        import sapien
                        if not isinstance(component, sapien.render.RenderBodyComponent):
                            continue
                    except Exception:
                        continue
                    for shape in component.render_shapes:
                        material = shape.material
                        if id(material) not in self._highlight_materials:
                            self._highlight_materials[id(material)] = (
                                material, list(material.base_color))
                        try:
                            material.set_base_color_texture(None)
                            material.set_base_color(GRIPPER_SELECT_GREEN)
                            material.base_color = GRIPPER_SELECT_GREEN
                        except Exception:
                            pass

    def _edge(self, window, key):
        down = bool(window.key_down(key))
        edge = down and not self._previous[key]
        self._previous[key] = down
        return edge

    def _select(self, window):
        selected = None
        if self._edge(window, "1"):
            selected = ("left",)
        elif self._edge(window, "2"):
            selected = ("right",)
        elif self._edge(window, "3"):
            selected = ("left", "right")
        if selected is not None:
            self.selected = selected
            self.env._interactive_selected_arms = selected
            self._highlight_selected()
            print("Selected arm(s): " + " + ".join(selected))

    def _drive_qpos(self, side):
        joints = (self.env.robot.left_arm_joints if side == "left"
                  else self.env.robot.right_arm_joints)
        return np.asarray(
            [joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_pose(self, side):
        getter = (self.env.robot.get_left_ee_pose if side == "left"
                  else self.env.robot.get_right_ee_pose)
        return np.asarray(getter(), dtype=np.float64)

    def _table_top_z(self) -> float:
        """World Z of the table surface (fallback: stock 0.74 + bias)."""
        top = getattr(self.env, "table_top", None)
        if top is not None:
            return float(top)
        bias = float(getattr(self.env, "table_z_bias", 0.0) or 0.0)
        return 0.74 + bias

    def _support_surface_z(self, side, pose) -> float:
        """World Z that fingertips must stay at or above (table, or task override)."""
        support_fn = getattr(self.env, "interactive_support_z", None)
        if callable(support_fn):
            try:
                override = support_fn(side, pose)
                if override is not None:
                    return float(override)
            except Exception:
                pass
        return self._table_top_z()

    def _gripper_min_world_z(self, side: str) -> float | None:
        """Lowest world Z of this arm's finger / gripper collision AABBs."""
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return None
        entity = robot.left_entity if side == "left" else robot.right_entity
        if entity is None:
            return None
        lo = None
        for link in entity.get_links():
            name = str(link.get_name() or "")
            if name not in GRIPPER_LINK_NAMES and "finger" not in name.lower():
                continue
            try:
                aabb = link.compute_global_aabb_tight()
                z = float(aabb[0][2])
            except Exception:
                try:
                    z = float(link.get_pose().p[2])
                except Exception:
                    continue
            lo = z if lo is None else min(lo, z)
        return lo

    def _finger_below_ee(self, side: str, ee_z: float) -> float:
        """How far below the EE the lowest finger geometry currently sits."""
        finger_z = self._gripper_min_world_z(side)
        if finger_z is None:
            env_val = getattr(self.env, "EE_TO_TCP", None)
            if env_val is not None:
                return float(env_val)
            bank = getattr(self.env, "_reactive_buttons", None)
            if bank is not None and hasattr(bank, "ee_to_tcp"):
                return max(float(bank.ee_to_tcp), float(self.EE_TO_TCP_FALLBACK))
            return float(self.EE_TO_TCP_FALLBACK)
        return max(float(ee_z) - float(finger_z), 0.05)

    def _global_ee_z_band(self, side, pose) -> tuple[float, float]:
        """Absolute world-frame (z_min, z_max) for Q/E teleop.

        Max = original gripper EE height. Min keeps measured finger AABBs at or
        above the support surface (table, or task ``interactive_support_z`` such
        as a box top) plus a small clearance.
        """
        origin = self._origin_pose.get(side)
        if origin is not None:
            z_max = float(origin[2])
        else:
            z_max = float(pose[2])
        # Prefer the live EE so the finger offset tracks contact compression.
        try:
            ee_z = float(self._ee_pose(side)[2])
        except Exception:
            ee_z = float(pose[2])
        z_min = (
            self._support_surface_z(side, pose)
            + self._finger_below_ee(side, ee_z)
            + float(self.FINGER_TABLE_CLEARANCE)
        )
        floor_fn = getattr(self.env, "interactive_ee_z_floor", None)
        # Raise-only here; press-key tasks replace the band in ``_drive`` the
        # same way ``_reactive_buttons.min_ee_z_over_key`` does.
        if callable(floor_fn):
            override = floor_fn(side, pose)
            if override is not None:
                z_min = max(z_min, float(override))
        ceil_fn = getattr(self.env, "interactive_ee_z_ceiling", None)
        if callable(ceil_fn):
            override = ceil_fn(side, pose)
            if override is not None:
                z_max = float(override)
        if z_min > z_max:
            z_min = z_max
        return z_min, z_max

    def _drive(self, side, step, dt, roll: float = 0.0, yaw: float = 0.0):
        """Advance this arm's commanded pose and track it with seeded IK."""
        from transforms3d.quaternions import axangle2quat, qmult

        solver = arm_ik(self.env, side)
        if solver is None:
            return
        achieved = self._ee_pose(side)
        state = self._command.get(side)
        if state is None:
            # Anchor pose and joints on the measured state (not the drive
            # targets): the two differ by the tracking error, and mixing them
            # makes the first step look like a jump and get rate-limited away.
            measured = solver.full_qpos()
            state = {
                "pose": achieved.copy(),
                "seed": measured,
                "joints": measured[solver.arm_dofs],
            }
            self._command[side] = state

        pose = state["pose"].copy()
        prev_z = float(pose[2])
        pose[:3] += step
        freeze_wrist = False
        freeze_fn = getattr(self.env, "interactive_freeze_wrist_orientation", None)
        if callable(freeze_fn):
            try:
                freeze_wrist = bool(freeze_fn(side))
            except Exception:
                freeze_wrist = False
        # World-Y tip (F/G): rotate the commanded gripper orientation in place.
        if (not freeze_wrist) and abs(float(roll)) > 1e-9:
            dq = axangle2quat([0.0, 1.0, 0.0], float(roll))
            pose[3:7] = np.asarray(qmult(dq, pose[3:7]), dtype=np.float64)
        # World-Z yaw (R/T): table-plane spin, premultiply for fixed-axis turn.
        if (not freeze_wrist) and abs(float(yaw)) > 1e-9:
            dq = axangle2quat([0.0, 0.0, 1.0], float(yaw))
            pose[3:7] = np.asarray(qmult(dq, pose[3:7]), dtype=np.float64)
        if freeze_wrist:
            # Keep the orientation from when the grasp engaged so the wrist
            # stays rigid while the free tap hinge absorbs the pull.
            pose[3:7] = np.asarray(state["pose"][3:7], dtype=np.float64)
        # Absolute world-frame Q/E band (not relative to current EE height).
        z_min, z_max = self._global_ee_z_band(side, pose)
        # Over a cook / reactive / dispenser key: *replace* the table+finger
        # floor with the key press-depth EE floor so Q can finish the press
        # after fingers contact the keycap, then stop (no dive past full force).
        key_floor = None
        bank = getattr(self.env, "_reactive_buttons", None)
        if bank is not None:
            if hasattr(bank, "min_ee_z_over_key"):
                key_floor = bank.min_ee_z_over_key(pose[:2])
            elif hasattr(bank, "min_ee_z_over_pressed"):
                key_floor = bank.min_ee_z_over_pressed(pose[:2])
        if key_floor is None:
            floor_fn = getattr(self.env, "interactive_ee_z_floor", None)
            if callable(floor_fn):
                try:
                    override = floor_fn(side, pose)
                    if override is not None:
                        key_floor = float(override)
                except Exception:
                    key_floor = None
        if key_floor is not None:
            z_min = float(key_floor)
        # Billiard / tool-on-surface: if the held cue is already on the felt,
        # reject further -Z on that arm so it stops with the stick.
        if float(step[2]) < -1e-9:
            resting = getattr(self.env, "cue_resting_on_felt", None)
            cue_arm = str(getattr(self.env, "_cue_arm", "") or "")
            if cue_arm == side and callable(resting) and resting():
                pose[2] = prev_z
        if float(pose[2]) > z_max:
            pose[2] = z_max
        if float(pose[2]) < z_min:
            pose[2] = z_min
        # Keep the command within reach of the achieved pose: a blocked or
        # joint-limited arm must not build up a lead it later snaps through.
        lead = pose[:3] - achieved[:3]
        distance = float(np.linalg.norm(lead))
        if distance > self.MAX_LEAD:
            pose[:3] = achieved[:3] + lead * (self.MAX_LEAD / distance)

        solution = solver.solve(pose, seed=state["seed"])
        if solution is None:
            return
        target, full = solution
        delta = target - state["joints"]
        budget = self.MAX_JOINT_SPEED * dt
        peak = float(np.max(np.abs(delta)))
        if peak > budget:
            delta *= budget / peak
            target = state["joints"] + delta
            full = full.copy()
            full[solver.arm_dofs] = target
        self.env.robot.set_arm_joints(target, delta / max(dt, 1e-3), side)
        state["pose"] = pose
        state["seed"] = full
        state["joints"] = target
        # Expose the commanded EE pose so spring/contact tasks can track the
        # teleop target even when the measured link lags the drive a bit.
        cmd = getattr(self.env, "_interactive_cmd_pose", None)
        if not isinstance(cmd, dict):
            cmd = {}
            self.env._interactive_cmd_pose = cmd
        cmd[side] = pose.copy()

    def _stop(self):
        """Zero the drive velocity targets, else the arms coast after release."""
        for side, state in self._command.items():
            joints = state["joints"]
            self.env.robot.set_arm_joints(joints, np.zeros_like(joints), side)
        self._command.clear()
        if isinstance(getattr(self.env, "_interactive_cmd_pose", None), dict):
            self.env._interactive_cmd_pose.clear()

    def update(self, window):
        self._select(window)
        gripper_failure_feedback(self.env).update()
        if self._edge(window, "o"):
            self._return_selected_to_origin()
            self._last_update = time.perf_counter()
            return
        now = time.perf_counter()
        dt = 0.0 if self._last_update is None else min(now - self._last_update, self.MAX_DT)
        self._last_update = now
        if bool(getattr(self.env, "_interactive_teleop_locked", False)):
            self._stop()
            return
        x_dir = float(window.key_down("right")) - float(window.key_down("left"))
        y_dir = float(window.key_down("up")) - float(window.key_down("down"))
        z_dir = float(window.key_down("e")) - float(window.key_down("q"))
        # F tip left / G tip right about world +Y (pour axis for board tasks).
        roll_dir = float(window.key_down("g")) - float(window.key_down("f"))
        # R counter-clockwise / T clockwise about world +Z (table-plane yaw).
        yaw_dir = float(window.key_down("r")) - float(window.key_down("t"))
        if not (x_dir or y_dir or z_dir or roll_dir or yaw_dir):
            self._stop()
            return
        if dt <= 0.0:
            return
        step = np.asarray([
            x_dir * self.XY_SPEED * dt,
            y_dir * self.XY_SPEED * dt,
            z_dir * self.Z_SPEED * dt,
        ], dtype=np.float64)
        roll = float(roll_dir) * self.ROLL_SPEED * dt
        yaw = float(yaw_dir) * self.YAW_SPEED * dt
        scale_fn = getattr(self.env, "interactive_teleop_z_speed_scale", None)
        for side in self.selected:
            local_step = step
            if callable(scale_fn) and abs(float(step[2])) > 1e-12:
                try:
                    state = self._command.get(side)
                    pose = state["pose"] if state is not None else self._ee_pose(side)
                    scale = scale_fn(side, pose, float(step[2]))
                    if scale is not None:
                        local_step = step.copy()
                        local_step[2] *= float(np.clip(scale, 0.0, 1.0))
                except Exception:
                    local_step = step
            self._drive(side, local_step, dt, roll=roll, yaw=yaw)


# Keep stepping/rendering this long after a terminal SUCCESS/FAILURE so the
# result is visible before the viewer closes (wall-clock, not sim time).
TERMINAL_RESULT_HOLD_SECONDS = 2.0

# Cap physics catch-up per display frame so a hitch cannot explode into a stall.
REALTIME_MAX_SUBSTEPS = 8


class RealtimePhysicsPacer:
    """Keep sim time aligned with wall-clock across different monitor refresh rates.

    Interactive loops used to do one ``scene.step()`` per display frame. With vsync
    that makes motion ~4× slower on 60 Hz than on 240 Hz. This pacer accumulates
    wall time and returns how many fixed ``dt`` physics steps to run before the
    next render (typically ~1 at 240 Hz, ~4 at 60 Hz for dt=1/250).

    Blocking planner moves (``env.move`` / ``take_dense_action``) should set
    ``env._interactive_pacer_resync = True`` when they finish so the next frame
    does not treat the blocked wall time as catch-up debt.
    """

    def __init__(self, env, max_substeps: int = REALTIME_MAX_SUBSTEPS):
        self.env = env
        self.dt = float(env.scene.get_timestep())
        self.max_substeps = max(1, int(max_substeps))
        self._accum = 0.0
        self._prev = time.perf_counter()

    def resync(self) -> None:
        """Drop accumulated wall time after a blocking planner/dwell section."""
        self._accum = 0.0
        self._prev = time.perf_counter()

    def begin_frame(self) -> int:
        """Advance the wall-clock accumulator; return physics steps for this frame."""
        now = time.perf_counter()
        if getattr(self.env, "_interactive_pacer_resync", False):
            self.env._interactive_pacer_resync = False
            self._accum = 0.0
            self._prev = now
            return 0
        self._accum += now - self._prev
        self._prev = now
        # Drop excess after a long pause / hitch (accept temporary slow-mo).
        self._accum = min(self._accum, self.dt * self.max_substeps)
        n = 0
        while self._accum >= self.dt and n < self.max_substeps:
            self._accum -= self.dt
            n += 1
        return n


def sleep_to_timestep(env, frame_start: float) -> None:
    """Legacy single-step padder — prefer :class:`RealtimePhysicsPacer`.

    Only sleeps when a frame finished *faster* than one physics ``dt``. It cannot
    catch up on slow (e.g. 60 Hz vsync) frames, so real-time motion still drifts
    with monitor refresh if you keep one ``scene.step()`` per render.
    """
    remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
    if remaining > 0:
        time.sleep(remaining)


def terminal_hold_should_close(terminal_started_at: float | None) -> bool:
    """True once the post-result display hold has finished (wall-clock)."""
    if terminal_started_at is None:
        return False
    return time.perf_counter() - terminal_started_at >= TERMINAL_RESULT_HOLD_SECONDS


ESCAPE_QUIT_DETAIL = "gave up (Esc)"


def escape_quit_requested(
    env,
    window,
    *,
    report_result: bool = True,
    detail: str = ESCAPE_QUIT_DETAIL,
) -> bool:
    """True when Esc is pressed; first records a terminal result so quitting counts.

    Esc means the participant gave up, which is a FAILURE rather than the
    "closed before a result" exit code that experiment logs skip. A result that
    was already reported (success/failure hold) is left untouched.
    """
    if window is None or not window.key_down("escape"):
        return False
    if report_result and _LAST_TASK_RESULT is None:
        report_task_result(env, detail)
    return True


def run_viewer_loop(env, on_step, should_stop=None, max_steps: int | None = None,
                    overhead: bool = True, is_done=None, extra_plugins=None,
                    report_result: bool = True):
    """Standard interactive loop: input → physics catch-up → render.

    Physics uses a fixed timestep catch-up so wall-clock motion speed stays
    roughly constant across 60 Hz / 240 Hz displays. ``on_step`` runs once per
    display frame (first substep) so key/mouse edges are not multi-fired.

    ``is_done(step)`` may return ``True`` / ``False``, or ``(done, detail)``.
    When done, prints SUCCESS/FAILURE via ``report_task_result`` (unless
    ``report_result`` is false, e.g. tutorials), then continues
    stepping/rendering for ``TERMINAL_RESULT_HOLD_SECONDS`` wall-clock before
    closing. Returns that bool (or ``None`` if the viewer closed without a
    result). Esc reports a terminal result too (FAILURE unless the task already
    succeeded) so giving up still counts as a play. ``should_stop`` remains a
    raw break (no auto print / no hold) for backward compatibility.
    Starts on head_camera; press V to cycle head ↔ gripper/wrist view(s).
    ``overhead`` is accepted for API compat but ignored (top-down removed).
    ``extra_plugins`` are appended after the stock ImGui panels are hidden, so
    their HUD windows stay visible.
    """
    global _LAST_TASK_RESULT, _LAST_TASK_DETAIL, _LAST_EPISODE_CONDITION
    _LAST_TASK_RESULT = None
    _LAST_TASK_DETAIL = None
    _LAST_EPISODE_CONDITION = None
    del overhead  # legacy kwarg; interactive views no longer use top-down
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    print_episode_condition(env)
    views = make_viewer_view_toggle(env, viewer)
    for plugin in extra_plugins or []:
        if plugin is None:
            continue
        try:
            plugin.init(viewer)
        except Exception:
            pass
        try:
            viewer.plugins.append(plugin)
        except Exception:
            pass
    step = 0
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)
    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            # Still pump the window / Escape when a fast display frame needs 0 steps.
            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if escape_quit_requested(env, viewer.window, report_result=report_result):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            for sub in range(n_steps):
                if sub == 0 and on_step is not None:
                    on_step(viewer.window, step)
                env._update_kinematic_tasks()
                env.scene.step()
                step += 1

                if terminal_started_at is not None:
                    continue
                if is_done is not None:
                    result = is_done(step)
                    if isinstance(result, tuple):
                        done = bool(result[0])
                        detail = result[1] if len(result) > 1 else None
                    else:
                        done, detail = bool(result), None
                    if done:
                        if report_result:
                            report_task_result(env, detail)
                        terminal_started_at = time.perf_counter()
                if should_stop is not None and should_stop(step):
                    env.scene.update_render()
                    viewer.render()
                    return _LAST_TASK_RESULT
                if max_steps is not None and step >= max_steps:
                    print(f"Reached max_steps={max_steps}; evaluating.")
                    if report_result:
                        report_task_result(env, f"max_steps={max_steps}")
                    terminal_started_at = time.perf_counter()

            env.scene.update_render()
            viewer.render()
            # SAPIEN does not close its window on Escape consistently, so make
            # it an explicit launcher-level exit for every shared task loop.
            if escape_quit_requested(env, viewer.window, report_result=report_result):
                break
            if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                break
    finally:
        env.close_env()
    return _LAST_TASK_RESULT


# ---------------------------------------------------------------------------
# Shared robot key/button press (same routine as interactive_sort_apples_belt)
# ---------------------------------------------------------------------------


def add_robot_motion_arg(parser, robot_motion_default: str = "planner"):
    """Add ``--control`` + ``--robot-motion`` flags used by every interactive script."""
    parser.add_argument(
        "--control",
        choices=("keyboard", "keyboard+mouse", "robot"),
        default="robot",
        help="Interaction method (default: robot). ``keyboard`` is an alias of keyboard+mouse.",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default=robot_motion_default,
        help=(
            "Robot key-press implementation; interpolate is a faster test mode "
            f"(default: {robot_motion_default})"
        ),
    )
    add_record_data_arg(parser)
    return parser


def parse_control_arg(args) -> str:
    """Normalize ``args.control`` in place and return the canonical mode."""
    mode = normalize_control_mode(getattr(args, "control", CONTROL_ROBOT))
    try:
        args.control = mode
    except Exception:
        pass
    return mode


def _line_documents_key(lines: list[str], key: str) -> bool:
    """True when a control banner line already documents ``key`` as a binding."""
    key_u = key.strip().upper()
    for ln in lines:
        s = ln.strip()
        s_u = s.upper()
        if (
            s_u.startswith(f"{key_u} ")
            or s_u.startswith(f"{key_u}:")
            or s_u.startswith(f"{key_u}\t")
            or s_u.startswith(f"{key_u} —")
            or s_u.startswith(f"{key_u} -")
        ):
            return True
        if f"{key_u}                 " in ln.upper() or f"{key_u}: " in ln.upper():
            return True
        # Compact banners: "V: camera | Space: open/close | Escape"
        if f"{key_u}:" in s_u or f"| {key_u}:" in s_u or f"|{key_u}:" in s_u:
            return True
    return False


_GRIPPER_TOGGLE_HELP = "Space             open / close selected gripper(s)"


def _is_gripper_toggle_help_line(line: str) -> bool:
    """True for leftover F/G or Space lines that document open/close gripper."""
    s = line.strip()
    s_u = s.upper()
    # F/G now tip the wrist; don't treat the shared tilt row as a gripper toggle.
    if s_u.startswith("F / G") or s_u.startswith("F/G"):
        return False
    if not (
        s.startswith("F ")
        or s.startswith("F:")
        or s.startswith("F\t")
        or s.startswith("G ")
        or s.startswith("G:")
        or s.startswith("G\t")
        or s.startswith("G —")
        or s.startswith("Space ")
        or s.startswith("Space:")
        or s.startswith("Space\t")
        or s.startswith("Space —")
    ):
        return False
    low = line.lower()
    return "gripper" in low and ("open" in low or "close" in low or "grasp" in low)


def _is_shared_robot_teleop_help_line(line: str) -> bool:
    """True for task-banner rows that only restate universal robot teleop keys."""
    s = line.strip()
    if not s:
        return False
    s_u = s.upper()
    # Combined short rows used by several interactive scripts.
    if s_u.startswith("ARROWS / E / Q") or s_u.startswith("ARROWS/E/Q"):
        return True
    prefixes = (
        "ARROW KEYS",
        "ARROWS ",
        "ARROWS\t",
        "ARROWS:",
        "E / Q",
        "E/Q ",
        "E/Q\t",
        "E/Q:",
        "Z / X",
        "Z/X ",
        "Z/X\t",
        "Z/X:",
        "F / G",
        "F/G ",
        "F/G\t",
        "F/G:",
        "R / T",
        "R/T ",
        "R/T\t",
        "R/T:",
        "1 / 2 / 3",
        "1/2/3",
        "O ",
        "O\t",
        "O:",
        "O —",
        "O -",
    )
    return any(s_u.startswith(p) for p in prefixes)


def print_mode_controls(task_name: str, mode: str, *, keyboard: str, robot: str) -> None:
    """Print only the help block for the selected ``--control`` mode."""
    mode = normalize_control_mode(mode)
    body = (robot if mode == CONTROL_ROBOT else keyboard).strip("\n")
    if mode == CONTROL_ROBOT:
        # Drop task lines that duplicate the shared teleop block below.
        task_lines = [
            ln for ln in body.splitlines()
            if not _is_shared_robot_teleop_help_line(ln)
        ]
        while task_lines and not task_lines[0].strip():
            task_lines.pop(0)
        while task_lines and not task_lines[-1].strip():
            task_lines.pop()
        body = "\n".join(task_lines)
        # Shared teleop keys; skip Space here when the task banner already lists it.
        shared = (
            "  Arrow keys        move selected arm(s) in world XY\n"
            "  E / Q             raise / lower selected arm(s)\n"
            "  F / G             tip gripper left / right (world Y)\n"
            "  R / T             yaw gripper CCW / CW (world Z)\n"
            "  1 / 2 / 3         select left / right / both arms (selected gripper turns green)\n"
            "  O                 return selected arm(s) to original position\n"
        )
        if not _line_documents_key(task_lines, "Space"):
            shared += f"  {_GRIPPER_TOGGLE_HELP}\n"
        body = shared + (("\n" + body) if body.strip() else "")
    include_v = mode == CONTROL_ROBOT
    lines = _normalize_view_help_lines(body.splitlines(), include_v=include_v)
    # Collapse duplicate F/G/Space gripper-toggle rows; keep task-specific Space
    # wording, only rewrite stale F/G bindings to Space.
    rewritten = []
    saw_space_grip = False
    for ln in lines:
        if _is_gripper_toggle_help_line(ln):
            if saw_space_grip:
                continue
            if ln.strip().upper().startswith("SPACE"):
                rewritten.append(ln)
            else:
                indent = ln[: len(ln) - len(ln.lstrip(" "))]
                rewritten.append(f"{indent}{_GRIPPER_TOGGLE_HELP}")
            saw_space_grip = True
        else:
            rewritten.append(ln)
    lines = rewritten
    # Keyboard+mouse hides the arms — do not advertise Space as a gripper toggle.
    if mode == CONTROL_ROBOT and not _line_documents_key(lines, "Space"):
        inserted = False
        for i, ln in enumerate(lines):
            if (
                "cycle view" in ln.lower()
                or "toggle view" in ln.lower()
                or ln.strip().startswith("V ")
                or ln.strip().startswith("V:")
                or "V                 " in ln
            ):
                indent = ln[: len(ln) - len(ln.lstrip(" "))]
                lines.insert(i + 1, f"{indent}{_GRIPPER_TOGGLE_HELP}")
                inserted = True
                break
        if not inserted:
            lines.append(f"  {_GRIPPER_TOGGLE_HELP}")
    body = "\n".join(lines)
    bar = "=" * 60
    print_instructions(f"{bar}\n {task_name} — {mode} controls\n{bar}\n{body}\n{bar}")


def default_arms_for_mode(mode):
    """``left`` / ``right`` / ``dump`` → arm side names (sort_apples style)."""
    if mode == "dump":
        return ("left", "right")
    return (mode,) if mode else ()


def selected_robot_arms(env, fallback=("left",)):
    """Return the arms selected by the universal 1/2/3 robot controls.

    When universal controls are active and nothing is selected yet, returns
    ``()`` (no fallback) so grippers stay inactive until 1 / 2 / 3.
    """
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if selected:
        return selected
    if bool(getattr(env, "_interactive_universal_controls", False)):
        return ()
    return tuple(fallback)


GRIPPER_LINK_NAMES = frozenset({
    "wsg_50_base_link", "gripper_left", "gripper_right",
    "finger_left", "finger_right",
})
GRIPPER_SELECT_GREEN = [0.15, 0.82, 0.22, 1.0]
GRIPPER_FAILURE_RED = [0.92, 0.04, 0.03, 1.0]
GRIPPER_FAILURE_SECONDS = 2.0


def require_selected_arms(env, *, exactly_one: bool = False):
    """Return currently highlighted arms, or ``()`` when the action must not run.

    Unselected grippers never act. When ``exactly_one`` is set, both-arms (3)
    is also rejected so the caller can require a single highlighted gripper;
    both selected grippers flash red as the error cue.
    """
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if not selected:
        print("Select an arm first: 1 (left) or 2 (right)"
              + ("" if exactly_one else " [or 3 for both]") + ".")
        return ()
    if exactly_one and len(selected) != 1:
        action_failed(
            env,
            selected,
            detail="select exactly one arm (1 left / 2 right)",
        )
        return ()
    return selected


def resolve_action_arm(env, arm_tag_cls, *, exactly_one: bool = True):
    """Return an ``ArmTag`` for the highlighted gripper, or ``None`` to abort."""
    selected = require_selected_arms(env, exactly_one=exactly_one)
    if not selected:
        return None
    return arm_tag_cls(selected[0])


class GripperFailureFeedback:
    """Tint failed gripper mesh(es) red for a short, non-acting feedback window."""

    def __init__(self, env):
        self.env = env
        self._until = 0.0
        self._original = {}

    def active(self) -> bool:
        return bool(self._original) and time.perf_counter() < self._until

    def restore(self):
        for material, color in self._original.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass
        self._original.clear()
        self._until = 0.0

    def _tint_side(self, side: str):
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return
        articulation = getattr(robot, f"{side}_entity", None)
        if articulation is None:
            return
        try:
            import sapien
        except Exception:
            return
        for link in articulation.get_links():
            if link.get_name() not in GRIPPER_LINK_NAMES:
                continue
            for component in link.entity.get_components():
                if not isinstance(component, sapien.render.RenderBodyComponent):
                    continue
                for shape in component.render_shapes:
                    material = shape.material
                    key = id(material)
                    if key not in self._original:
                        self._original[key] = (material, list(material.base_color))
                    try:
                        material.set_base_color_texture(None)
                        material.set_base_color(GRIPPER_FAILURE_RED)
                        material.base_color = GRIPPER_FAILURE_RED
                    except Exception:
                        pass

    def flash(self, arms, message: str | None = None):
        """Show red for ``GRIPPER_FAILURE_SECONDS``; do not perform any motion."""
        sides = tuple(str(a) for a in (arms or ()) if str(a) in ("left", "right"))
        if not sides:
            return
        self.restore()
        for side in sides:
            self._tint_side(side)
        self._until = time.perf_counter() + GRIPPER_FAILURE_SECONDS
        if message:
            print(message)

    def update(self):
        if self._original and time.perf_counter() >= self._until:
            self.restore()
            controls = getattr(self.env, "_interactive_robot_controls", None)
            if controls is not None and hasattr(controls, "_highlight_selected"):
                try:
                    controls._highlight_selected()
                except Exception:
                    pass


def gripper_failure_feedback(env) -> GripperFailureFeedback:
    fb = getattr(env, "_interactive_gripper_failure", None)
    if not isinstance(fb, GripperFailureFeedback):
        fb = GripperFailureFeedback(env)
        env._interactive_gripper_failure = fb
    return fb


def flash_gripper_failure(env, arms, message: str | None = None):
    """Shared entry point: red gripper flash when a selected-arm action fails."""
    gripper_failure_feedback(env).flash(arms, message=message)


def action_failed(env, arms, detail: str = "action failed") -> bool:
    """Flash red, clear the plan-failure latch, return False for callers."""
    sides = tuple(str(a) for a in (arms or ()) if str(a))
    label = "+".join(sides) if sides else "arm"
    flash_gripper_failure(env, sides, f"{label} {detail}; no motion applied")
    try:
        env.plan_success = True
        env._last_plan_fail = None
    except Exception:
        pass
    return False


def try_interactive_grasp(env, actor, arm_tag, *, detail: str = "unreachable / grasp failed", **grasp_kwargs) -> bool:
    """Run ``grasp_actor`` + ``move`` without crashing on an unreachable arm.

    Returns True on success. On failure (no pose, plan fail, or legacy
    ``target_pose is None`` assert), flashes the arm red and returns False.
    """
    side = str(arm_tag)
    try:
        env.plan_success = True
        env.move(env.grasp_actor(actor, arm_tag=arm_tag, **grasp_kwargs))
    except AssertionError:
        env.plan_success = False
    if not env.plan_success:
        return action_failed(env, (side,), detail=detail)
    return True


class RobotButtonController:
    """Hold-to-actuate (or tap) key press: grasp → TCP-limited press → latch → clear lift.

    Matches ``interactive_sort_apples_belt.RobotButtonController``. Task scripts supply
    actor / top-z / latch adapters so the motion routine stays shared.
    """

    PRESS_HOLD_STEPS = 8
    CONTACT_CLEARANCE = 0.008
    # Must cover grasp_dis hover (~8 cm). A 4 cm cap left the arm stuck above the key.
    MAX_PRESS_DEPTH = 0.14
    PRESS_STEP = 0.02
    JOINT_RETRACT_STEPS = 25
    CLEAR_ABOVE_ACTIVE = 0.04

    def __init__(
        self,
        env,
        arm_tag,
        get_button,
        get_top_z,
        set_latch=None,
        clear_latch=None,
        arms_for_mode=None,
        on_press=None,
        hold: bool = True,
        active_dz: float | None = None,
        grasp_dis: float = 0.08,
        pre_grasp_dis: float = 0.08,
        max_press_depth: float | None = None,
        **_ignored,
    ):
        """
        Parameters
        ----------
        get_button : callable(env, side) -> actor
        get_top_z : callable(env, side) -> float
        set_latch : callable(env, mode) | None
        clear_latch : callable(env) | None
        arms_for_mode : callable(mode) -> iterable[str]
        on_press : callable(env, mode) | None  — side effect after contact lands
        hold : if False, press then immediately lift (edge tap)
        active_dz : proximity clear height; default env.PRESS_DZ_ACTIVE or 0.12
        max_press_depth : cap on -Z travel; default covers grasp_dis + margin
        """
        self.env = env
        self.arm_tag = arm_tag
        self.get_button = get_button
        self.get_top_z = get_top_z
        self.set_latch = set_latch or (lambda _e, mode: setattr(_e, "_expert_hold", mode))
        self.clear_latch = clear_latch or (lambda _e: setattr(_e, "_expert_hold", None))
        self.arms_for_mode = arms_for_mode or default_arms_for_mode
        self.on_press = on_press
        self.hold = hold
        self.active_dz = active_dz
        self.grasp_dis = grasp_dis
        self.pre_grasp_dis = pre_grasp_dis
        # Allow enough -Z to reach the key from the grasp hover pose.
        self.max_press_depth = float(
            max_press_depth if max_press_depth is not None
            else max(self.MAX_PRESS_DEPTH, float(grasp_dis) + 0.06)
        )
        self.mode = None
        self._hover_qpos = {}

    def _tcp_z(self, side):
        get_tcp = (self.env.robot.get_left_tcp_pose if side == "left"
                   else self.env.robot.get_right_tcp_pose)
        try:
            return float(get_tcp()[2])
        except Exception:
            return None

    def _drive_qpos(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _press_depth_tcp(self, side):
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return 0.0
        top_z = float(self.get_top_z(self.env, side))
        need = float(tcp_z) - top_z - self.CONTACT_CLEARANCE
        return float(np.clip(need, 0.0, self.max_press_depth))

    def _active_band(self) -> float:
        """TCP height above key top that still counts as an active press."""
        if self.active_dz is not None:
            return float(self.active_dz)
        return float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))

    def _in_key_contact(self, side, slack: float = 0.025) -> bool:
        """True when TCP is at/near the key top — stop descending further."""
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + self.CONTACT_CLEARANCE + float(slack)

    def _in_active_zone(self, side) -> bool:
        """True when TCP is inside the task proximity band (may actuate)."""
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + self._active_band()

    def _mode_in_contact(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_key_contact(side) for side in sides)

    def _mode_in_active_zone(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_active_zone(side) for side in sides)

    def _press_until_contact(self, mode) -> bool:
        """Step -Z until TCP touches the key (or budget / plan failure)."""
        sides = tuple(self.arms_for_mode(mode))
        traveled = {side: 0.0 for side in sides}
        max_steps = int(self.max_press_depth / self.PRESS_STEP) + 2
        for _ in range(max_steps):
            if self._mode_in_contact(mode):
                return True
            pending = [
                side for side in sides
                if not self._in_key_contact(side) and traveled[side] < self.max_press_depth
            ]
            if not pending:
                break
            step = min(
                self.PRESS_STEP,
                *(self._press_depth_tcp(side) or self.PRESS_STEP for side in pending),
            )
            step = max(float(step), 0.005)
            self.env.plan_success = True
            self.env.move(*[
                self.env.move_by_displacement(self.arm_tag(side), z=-step)
                for side in pending
            ])
            if not self.env.plan_success:
                print("Press step failed; stopping descent.")
                break
            for side in pending:
                traveled[side] += step
        return self._mode_in_contact(mode)

    def _interpolate_to_qpos(self, targets):
        if not targets:
            return
        starts = {side: self._drive_qpos(side) for side in targets}
        for step in range(1, self.JOINT_RETRACT_STEPS + 1):
            alpha = step / self.JOINT_RETRACT_STEPS
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            for side, target in targets.items():
                position = starts[side] + (target - starts[side]) * smooth
                velocity = (target - starts[side]) / max(self.JOINT_RETRACT_STEPS, 1)
                self.env.robot.set_arm_joints(position, velocity, side)
            self.clear_latch(self.env)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
            viewer = getattr(self.env, "viewer", None)
            if viewer is not None:
                self.env.scene.update_render()
                viewer.render()
        for side, target in targets.items():
            self.env.robot.set_arm_joints(target, np.zeros_like(target), side)
        self.env.plan_success = True
        self.env._last_plan_fail = None

    def _activate_press(self, mode) -> bool:
        """Latch + on_press when TCP is in the active press band."""
        if not self._mode_in_active_zone(mode):
            print(f"No key contact for {mode}; not actuating (door/key stays idle).")
            self.clear_latch(self.env)
            return False
        self.set_latch(self.env, mode)
        if self.on_press is not None:
            try:
                self.on_press(self.env, mode)
            except Exception as exc:
                print(f"on_press callback failed: {exc}")
        return True

    def _move_to_buttons(self, mode):
        sides = tuple(self.arms_for_mode(mode))
        if not sides:
            return
        actions = []
        for side in sides:
            button = self.get_button(self.env, side)
            actions.append(self.env.grasp_actor(
                button, arm_tag=self.arm_tag(side),
                pre_grasp_dis=self.pre_grasp_dis, grasp_dis=self.grasp_dis,
                contact_point_id=0, gripper_pos=0.0,
            ))
        self.env.move(*actions)
        if not self.env.plan_success:
            return
        self._hover_qpos = {side: self._drive_qpos(side) for side in sides}
        depths = {side: self._press_depth_tcp(side) for side in sides}
        print(
            "Press depths (TCP-limited): "
            + ", ".join(f"{side}={depths[side] * 100:.1f}cm" for side in depths)
            + f" (max {self.max_press_depth * 100:.1f}cm)"
        )
        # Step down to the keytop when possible. Actuation uses the wider
        # active band (e.g. PRESS_DZ_ACTIVE) so a fingertip touch that never
        # quite satisfies the tight stop-clearance still latches _expert_hold.
        self._press_until_contact(mode)
        if not self._mode_in_active_zone(mode):
            print("Could not reach key active zone; returning to hover (no latch).")
            self._interpolate_to_qpos(self._hover_qpos)
            self.clear_latch(self.env)
            return
        if not self._activate_press(mode):
            return
        for _ in range(self.PRESS_HOLD_STEPS):
            self.set_latch(self.env, mode)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
        print(f"Robot pressed {mode} (hold key to keep actuated)." if self.hold
              else f"Robot tapped {mode}.")

    def _clear_z_target(self, side):
        active = self.active_dz
        if active is None:
            active = float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))
        return float(self.get_top_z(self.env, side)) + float(active) + self.CLEAR_ABOVE_ACTIVE

    def _lift_clear_of_keys(self, mode):
        lifts = []
        for side in self.arms_for_mode(mode):
            tcp_z = self._tcp_z(side)
            clear_z = self._clear_z_target(side)
            if tcp_z is None:
                lifts.append((side, 0.10))
                continue
            extra = clear_z - float(tcp_z)
            if extra > 0.001:
                lifts.append((side, extra))
        if not lifts:
            return
        self.env.plan_success = True
        self.env._last_plan_fail = None
        self.env.move(*[
            self.env.move_by_displacement(self.arm_tag(side), z=dz)
            for side, dz in lifts
        ])
        if not self.env.plan_success:
            print("Clear-lift plan failed; applying joint-space +Z retreat.")
            starts = {side: self._drive_qpos(side) for side, _ in lifts}
            self.env.plan_success = True
            for side, dz in lifts:
                pose = np.asarray(
                    self.env.robot.get_left_ee_pose() if side == "left"
                    else self.env.robot.get_right_ee_pose(),
                    dtype=np.float64,
                ).copy()
                pose[2] += dz
                planner = (self.env.robot.left_plan_path if side == "left"
                           else self.env.robot.right_plan_path)
                result = planner(pose, last_qpos=np.asarray(starts[side], dtype=np.float32))
                if result is not None and result.get("status") == "Success":
                    target = np.asarray(result["position"][-1], dtype=np.float64)
                    self._interpolate_to_qpos({side: target})
            self.env.plan_success = True
            self.env._last_plan_fail = None
        self.clear_latch(self.env)
        for _ in range(12):
            self.env._update_kinematic_tasks()
            self.env.scene.step()

    def _lift_from_buttons(self, mode):
        targets = {
            side: self._hover_qpos[side]
            for side in self.arms_for_mode(mode)
            if side in self._hover_qpos
        }
        if targets:
            self._interpolate_to_qpos(targets)
        else:
            actions = [
                self.env.move_by_displacement(self.arm_tag(side), z=0.06)
                for side in self.arms_for_mode(mode)
            ]
            if actions:
                self.env.plan_success = True
                self.env.move(*actions)
        self._lift_clear_of_keys(mode)
        self._hover_qpos.clear()

    def update(self, requested_mode):
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None)
            sides = tuple(self.arms_for_mode(self.mode or requested_mode))
            action_failed(
                self.env, sides,
                detail=f"button trajectory failed ({detail or 'unknown'})",
            )
        if self.hold and requested_mode == self.mode:
            if requested_mode is not None:
                self.set_latch(self.env, requested_mode)
            return
        self.clear_latch(self.env)
        if self.mode is not None:
            self._lift_from_buttons(self.mode)
        if requested_mode is not None:
            sides = tuple(self.arms_for_mode(requested_mode))
            if not sides:
                # No highlighted arm maps to this action — do nothing.
                requested_mode = None
            else:
                self._move_to_buttons(requested_mode)
                if not self.hold:
                    # Edge tap: press then release in one update.
                    self.clear_latch(self.env)
                    self._lift_from_buttons(requested_mode)
                    requested_mode = None
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None)
            sides = tuple(self.arms_for_mode(requested_mode or self.mode))
            action_failed(
                self.env, sides,
                detail=f"button motion failed ({detail or 'unknown'})",
            )
            if self._hover_qpos:
                self._interpolate_to_qpos(self._hover_qpos)
                self._hover_qpos.clear()
            requested_mode = None
            self.clear_latch(self.env)
        self.mode = requested_mode

    def release(self):
        self.update(None)


class InterpolatedRobotButtonController:
    """Fast joint interpolation between precomputed hover/press poses (test mode)."""

    DURATION = 0.28

    def __init__(
        self,
        env,
        arm_tag,
        get_button,
        get_top_z,
        set_latch=None,
        clear_latch=None,
        arms_for_mode=None,
        on_press=None,
        hold: bool = True,
        active_dz: float | None = None,
        sides=("left", "right"),
        **_ignored,
    ):
        self.env = env
        self.arm_tag = arm_tag
        self.get_button = get_button
        self.get_top_z = get_top_z
        self.set_latch = set_latch or (lambda _e, mode: setattr(_e, "_expert_hold", mode))
        self.clear_latch = clear_latch or (lambda _e: setattr(_e, "_expert_hold", None))
        self.arms_for_mode = arms_for_mode or default_arms_for_mode
        self.on_press = on_press
        self.hold = hold
        self.active_dz = active_dz
        self.mode = None
        self._start = {}
        self._targets = {}
        self._started_at = None
        self._sides = tuple(sides)
        self.hover, self.pressed = self._build_button_targets()
        print("Fast interpolation targets prepared (collision checking is only done during setup).")

    def _plan(self, side, pose, last_qpos=None):
        planner = self.env.robot.left_plan_path if side == "left" else self.env.robot.right_plan_path
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float32)
        result = planner(pose, last_qpos=last_qpos)
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown reason")
            raise RuntimeError(f"Could not prepare the {side} button interpolation target: {reason}")
        return np.asarray(result["position"][-1], dtype=np.float32)

    def _drive_targets(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_tcp_world_dz(self, side):
        get_ee = (self.env.robot.get_left_ee_pose if side == "left"
                  else self.env.robot.get_right_ee_pose)
        get_tcp = (self.env.robot.get_left_tcp_pose if side == "left"
                   else self.env.robot.get_right_tcp_pose)
        return max(0.0, float(get_ee()[2]) - float(get_tcp()[2]))

    def _build_button_targets(self):
        hover, pressed = {}, {}
        active = self.active_dz
        if active is None:
            active = float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))
        clear_dis = float(active) + 0.04
        for side in self._sides:
            button = self.get_button(self.env, side)
            hover_pose = self.env.get_grasp_pose(
                button, self.arm_tag(side), contact_point_id=0, pre_dis=clear_dis,
            )
            hover_qpos = self._plan(side, hover_pose)
            rest_qpos = self._drive_targets(side)
            self.env.robot.set_arm_joints(hover_qpos, np.zeros_like(hover_qpos), side)
            press_pose = np.asarray(hover_pose, dtype=np.float64).copy()
            press_pose[2] = (
                float(self.get_top_z(self.env, side)) + self._ee_tcp_world_dz(side) + 0.008
            )
            # Allow a deep enough press from the clear hover to reach the key.
            press_pose[2] = max(press_pose[2], float(hover_pose[2]) - 0.16)
            self.env.robot.set_arm_joints(rest_qpos, np.zeros_like(rest_qpos), side)
            hover[side] = hover_qpos
            pressed[side] = self._plan(side, press_pose, last_qpos=hover_qpos)
        return hover, pressed

    def _begin(self, requested_mode):
        self.clear_latch(self.env)
        active = set(self.arms_for_mode(requested_mode))
        moving = set(self.arms_for_mode(self.mode)) | active
        # Only sides we precomputed.
        moving = {side for side in moving if side in self.hover}
        self._start = {side: self._drive_targets(side) for side in moving}
        self._targets = {
            side: (self.pressed[side] if side in active else self.hover[side])
            for side in moving
        }
        self._started_at = time.perf_counter()
        self.mode = requested_mode

    def _tcp_z(self, side):
        get_tcp = (self.env.robot.get_left_tcp_pose if side == "left"
                   else self.env.robot.get_right_tcp_pose)
        try:
            return float(get_tcp()[2])
        except Exception:
            return None

    def _active_band(self) -> float:
        if self.active_dz is not None:
            return float(self.active_dz)
        return float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))

    def _in_key_contact(self, side, slack: float = 0.02) -> bool:
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + 0.008 + float(slack)

    def _in_active_zone(self, side) -> bool:
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + self._active_band()

    def _mode_in_contact(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_key_contact(side) for side in sides)

    def _mode_in_active_zone(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_active_zone(side) for side in sides)

    def _activate_press(self, mode) -> bool:
        if not self._mode_in_active_zone(mode):
            print(f"No key contact for {mode}; not actuating (door/key stays idle).")
            self.clear_latch(self.env)
            return False
        self.set_latch(self.env, mode)
        if self.on_press is not None:
            try:
                self.on_press(self.env, mode)
            except Exception as exc:
                print(f"on_press callback failed: {exc}")
        return True

    def update(self, requested_mode):
        if not self.hold and requested_mode is not None and requested_mode != self.mode:
            # Tap: go to press, latch only on contact, return to hover.
            self._begin(requested_mode)
            while self._started_at is not None:
                self._advance(activate=False)
            activated = self._activate_press(requested_mode)
            if activated:
                print(f"Robot tapped {requested_mode} with fast interpolation.")
            self._begin(None)
            while self._started_at is not None:
                self._advance(activate=False)
            self.clear_latch(self.env)
            self.mode = None
            return
        if requested_mode != self.mode:
            self._begin(requested_mode)
        if self._started_at is None:
            if self.mode is not None:
                if self._mode_in_active_zone(self.mode):
                    self.set_latch(self.env, self.mode)
                else:
                    self.clear_latch(self.env)
            else:
                self.clear_latch(self.env)
            return
        self._advance(activate=True)


    def _advance(self, activate: bool = True):
        progress = min(1.0, (time.perf_counter() - self._started_at) / self.DURATION)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        for side, target in self._targets.items():
            position = self._start[side] + (target - self._start[side]) * smooth
            velocity = (target - self._start[side]) / self.DURATION if progress < 1.0 else np.zeros_like(target)
            self.env.robot.set_arm_joints(position, velocity, side)
        if progress >= 1.0:
            self._started_at = None
            if self.mode is not None:
                if activate:
                    if self._activate_press(self.mode):
                        print(f"Robot pressed {self.mode} with fast interpolation.")
            else:
                self.clear_latch(self.env)

    def release(self):
        self.update(None)


def make_button_controller(env, arm_tag, robot_motion: str = "planner", **kwargs):
    """Factory: planner (TCP press) or interpolate (joint-space) button controller."""
    if robot_motion == "interpolate":
        return InterpolatedRobotButtonController(env, arm_tag, **kwargs)
    return RobotButtonController(env, arm_tag, **kwargs)
