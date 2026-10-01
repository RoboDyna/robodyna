"""Bridge one finished DOMINO episode (its `.cache/episode{N}/*.pkl` frames) into the combined
LeRobot v2.1 suite dataset. Called inline from the collector's render pass, AFTER
merge_pkl_to_hdf5_video() and BEFORE remove_data_cache() (the cache frames still exist there).

One LeRobotEpisodeExporter is created per collection job (one task); it owns a LeRobotV21Writer that
writes directly into the shared dataset's task-{TT}/ folders. See envs/utils/lerobot_v21.py.
"""
import os
import glob
import pickle
import json
import numpy as np

from .lerobot_v21 import LeRobotV21Writer, build_features

# Stable task_index assignment for the 15-task suite (task_index == episode_chunk folder).
SUITE_TASK_INDEX = {name: i for i, name in enumerate([
    "cook_meat", "sort_apples_belt", "place_block_belt", "catch_cuboid",
    "hit_target", "punch_dual_holes", "catch_marbles_trapdoors", "control_quality",
    "drop_ball_hole", "pick_ripe_apple", "put_cup_belt",
    "catch_ramp_ball", "save_goal", "pack_fruits", "whack_moles",
])}

# sim camera name -> LeRobot video feature key
CAMERA_MAP = {
    "head_camera": "observation.images.head",
    "front_camera": "observation.images.front",
    "left_camera": "observation.images.left_wrist",
    "right_camera": "observation.images.right_wrist",
    "countertop_camera": "observation.images.countertop",
}

_STD_OBS_KEYS = {"observation", "pointcloud", "joint_action", "endpose"}


def _flatten_dynamic(block, prefix=""):
    """Flatten a task's custom get_obs block into (names, values) of numeric leaves, sorted by key."""
    names, vals = [], []
    if isinstance(block, dict):
        for k in sorted(block.keys()):
            n, v = _flatten_dynamic(block[k], f"{prefix}{k}.")
            names += n
            vals += v
    elif isinstance(block, (list, tuple, np.ndarray)):
        arr = np.asarray(block).reshape(-1)
        for i, x in enumerate(arr):
            try:
                vals.append(float(x))
                names.append(f"{prefix[:-1]}[{i}]")
            except (TypeError, ValueError):
                pass
    else:
        try:
            vals.append(float(block))
            names.append(prefix[:-1])
        except (TypeError, ValueError):
            pass
    return names, vals


def _endpose_vec(ep):
    """16-d: left_endpose(7) + left_gripper(1, normalized) + right_endpose(7) + right_gripper(1)."""
    return np.concatenate([
        np.asarray(ep["left_endpose"], dtype=np.float32).reshape(-1),
        np.atleast_1d(np.float32(ep["left_gripper"])),
        np.asarray(ep["right_endpose"], dtype=np.float32).reshape(-1),
        np.atleast_1d(np.float32(ep["right_gripper"])),
    ]).astype(np.float32)


class LeRobotEpisodeExporter:
    def __init__(self, args):
        self.task_name = args["task_name"]
        self.task_index = SUITE_TASK_INDEX.get(self.task_name)
        if self.task_index is None:  # task not in the suite map -> append deterministically
            self.task_index = len(SUITE_TASK_INDEX) + abs(hash(self.task_name)) % 1000
        self.root = args.get("lerobot_root", os.path.join(args.get("save_path_base", "data_lerobot"),
                                                          "domino_suite"))
        self.chunks_size = int(args.get("lerobot_chunks_size", 1000))
        self.task_state_dim = int(args.get("lerobot_task_state_dim", 32))
        save_freq = args.get("save_freq") or 15
        self.fps = float(args.get("lerobot_fps", 250.0 / save_freq))  # sim runs at 250 Hz; frame every save_freq steps
        self.instruction = self._canonical_instruction(args)
        self.writer = None  # lazily built on first episode (needs camera keys + rgb size from the cache)

    def _canonical_instruction(self, args):
        """One constant instruction per task (-> tasks.jsonl). Per-episode phrasing goes in the annotation."""
        path = os.path.join("description", "task_instruction", f"{self.task_name}.json")
        try:
            d = json.load(open(path))
            txt = (d.get("full_description") or "").strip()
            if not txt and d.get("seen"):
                txt = d["seen"][0]
            return txt or self.task_name.replace("_", " ")
        except Exception:
            return self.task_name.replace("_", " ")

    def _build_writer(self, cam_feature_names, rgb_hw):
        feats = build_features(cam_feature_names, rgb_hw, state_dim=14, endpose_dim=16,
                               task_state_dim=self.task_state_dim, action_dim=14, fps=self.fps)
        self.writer = LeRobotV21Writer(self.root, self.task_index, self.task_name, feats, self.fps,
                                       robot_type="aloha-agilex", chunks_size=self.chunks_size)
        self._cam_map = {sim: CAMERA_MAP[sim] for sim in self._cams}

    def export(self, task_env, success):
        cache = os.path.join(task_env.save_dir, ".cache", f"episode{task_env.ep_num}")
        pkls = sorted(glob.glob(os.path.join(cache, "*.pkl")), key=lambda p: int(os.path.basename(p)[:-4]))
        if not pkls:
            return None
        frames = [pickle.load(open(p, "rb")) for p in pkls]

        # discover cameras present (with rgb) on first episode, build the writer once
        if self.writer is None:
            self._cams = [c for c in CAMERA_MAP if c in frames[0]["observation"]
                          and "rgb" in frames[0]["observation"][c]]
            h, w = frames[0]["observation"][self._cams[0]]["rgb"].shape[:2]
            self._build_writer([CAMERA_MAP[c] for c in self._cams], (h, w))

        schema_names = None
        for fr in frames:
            state = np.asarray(fr["joint_action"]["vector"], dtype=np.float32)
            endpose = _endpose_vec(fr["endpose"])
            names, vals = [], []
            for k in sorted(set(fr.keys()) - _STD_OBS_KEYS):
                n, v = _flatten_dynamic(fr[k], f"{k}.")
                names += n
                vals += v
            schema_names = schema_names or names
            ts = np.zeros(self.task_state_dim, dtype=np.float32)
            m = min(len(vals), self.task_state_dim)
            ts[:m] = np.asarray(vals[:m], dtype=np.float32)
            images = {self._cam_map[c]: fr["observation"][c]["rgb"] for c in self._cams}
            self.writer.add_frame(
                {"observation.state": state, "observation.endpose": endpose,
                 "observation.task_state": ts, "action": state}, images)

        annotation = {
            "instruction": self.instruction,
            "primitive_annotation": self._segments(task_env, len(frames)),
            "dynamic_state": {"schema": (schema_names or [])[:self.task_state_dim],
                              "dim": self.task_state_dim},
        }
        return self.writer.save_episode(task=self.instruction, success=bool(success), annotation=annotation)

    @staticmethod
    def _segments(task_env, n):
        """Build primitive segments from any mark_phase() calls; fall back to a single segment."""
        marks = list(getattr(task_env, "_phase_marks", []))
        if not marks:
            return [{"label": "demo", "frame_duration": [0, int(n)]}]
        segs = []
        for i, mk in enumerate(marks):
            start = int(mk["frame"])
            end = int(marks[i + 1]["frame"]) if i + 1 < len(marks) else int(n)
            seg = {k: v for k, v in mk.items() if k != "frame"}
            seg["frame_duration"] = [start, end]
            segs.append(seg)
        return segs

    def close(self):
        if self.writer is not None:
            self.writer.close()
