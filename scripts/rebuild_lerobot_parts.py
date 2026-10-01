#!/usr/bin/env python
"""Reconstruct LeRobot v2.1 meta/_parts slices for a config FROM THE ON-DISK parquet+videos+annotations.

WHY THIS EXISTS
    envs/utils/lerobot_v21.py writes the parquet trajectory, the mp4 videos and the annotation JSON
    per-episode (durable), but writes the meta/_parts slices (episodes.jsonl / episodes_stats.jsonl /
    info.json) ONLY in close(). close() runs only when the collector exits cleanly (episode_num
    reached). Low-SR tasks never reach the target within PER_TRY_TIMEOUT, so the collector is killed
    by timeout/scancel and close() never runs: all the trajectory data is on disk, but _parts is
    empty, so scripts/merge_lerobot_meta.py has nothing to merge and the config has no loadable meta.

    This rebuilds _parts purely from disk, so merge can then produce a loadable v2.1 dataset -- with
    NO re-collection. It is the authoritative reconciliation of _parts to the on-disk parquet, so it
    also repairs the close()-on-resume truncation (close() rewrites _parts from only the last run's
    in-memory episodes, dropping earlier runs' episodes -- this re-adds them).

PROPERTIES
    incremental + idempotent: an episode already present (and consistent) in episodes.jsonl AND
    episodes_stats.jsonl is reused as-is; only parquet episodes missing from the slices are computed.
    info.json is always rewritten from the full on-disk set. Reads only disk, writes only _parts, so
    it is safe to run alongside live collection and safe to re-run.

USAGE
    python scripts/rebuild_lerobot_parts.py --root data_lerobot/prod_run/<cfg>
        [--full]             recompute every episode, ignore existing slices
        [--no-image-stats]   skip decoding videos for per-channel image stats (much faster)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

# reuse the writer's OWN stats/feature helpers so reconstructed slices match clean-close output exactly
sys.path.insert(0, os.getcwd())
from envs.utils.lerobot_v21 import _stats_vec, _stats_scalar, _stats_image, build_features  # noqa: E402

VEC_KEYS = ["observation.state", "observation.endpose", "observation.task_state", "action"]
SCALAR_KEYS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]


def _decode_video_frames(path):
    """Return a list of HxWx3 uint8 RGB frames, or [] if no decoder / unreadable."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    except Exception:
        pass
    try:
        import imageio.v2 as imageio
        return [np.asarray(f)[..., :3] for f in imageio.get_reader(path)]
    except Exception:
        return []


def _video_dims(path):
    """(H, W) of the first frame via cv2/imageio, or (0, 0)."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if h and w:
            return h, w
    except Exception:
        pass
    fr = _decode_video_frames(path)
    if fr:
        return fr[0].shape[0], fr[0].shape[1]
    return 0, 0


def _load_slice(path, key="episode_index"):
    out = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d[key]] = d
            except Exception:
                pass
    return out


def rebuild_root(root, full=False, image_stats=True):
    data_dirs = sorted(glob.glob(os.path.join(root, "data", "task-*")))
    if not data_dirs:
        print(f"[rebuild] no data/task-* under {root} (nothing to rebuild)")
        return 0
    grand_new = 0
    for dd in data_dirs:
        tt = os.path.basename(dd)                 # task-XXXX
        ti = int(tt.split("-")[1])
        parts = os.path.join(root, "meta", "_parts")
        os.makedirs(parts, exist_ok=True)
        ep_path = os.path.join(parts, f"{tt}.episodes.jsonl")
        st_path = os.path.join(parts, f"{tt}.episodes_stats.jsonl")

        done_ep = {} if full else _load_slice(ep_path)
        done_st = {} if full else _load_slice(st_path)

        video_root = os.path.join(root, "videos", tt)
        video_keys = sorted(os.path.basename(p) for p in glob.glob(os.path.join(video_root, "*"))) \
            if os.path.isdir(video_root) else []
        # Prefer the declared schema when available: directory discovery alone cannot notice that
        # an entire camera directory is missing.
        old_info_path = os.path.join(parts, f"{tt}.info.json")
        if os.path.isfile(old_info_path):
            try:
                old_features = json.load(open(old_info_path)).get("features", {})
                declared = [k for k, v in old_features.items() if v.get("dtype") == "video"]
                if declared:
                    video_keys = sorted(declared)
            except (OSError, ValueError):
                pass

        ep_lines, st_lines = {}, {}
        instruction = task_name = fps = None
        dims = {}
        new_here = 0
        for p in sorted(glob.glob(os.path.join(dd, "*.parquet"))):
            ee = int(os.path.basename(p).split("_")[1].split(".")[0])
            ann_path = os.path.join(root, "annotations", tt, f"episode_{ee:08d}.json")
            missing = []
            if not os.path.isfile(ann_path):
                missing.append("annotation")
            for vk in video_keys:
                vp = os.path.join(video_root, vk, f"episode_{ee:08d}.mp4")
                if not os.path.isfile(vp) or os.path.getsize(vp) == 0:
                    missing.append(vk)
            if missing:
                print(f"[rebuild] skip incomplete episode {ee}: missing {', '.join(missing)}")
                continue
            try:                                  # skip a parquet still being written by live collection
                t = pq.read_table(p)
            except Exception as e:
                print(f"[rebuild] skip unreadable {os.path.basename(p)}: {type(e).__name__}")
                continue
            n = t.num_rows
            cols = t.column_names
            vecs = {k: np.array(t.column(k).to_pylist(), dtype=np.float64) for k in VEC_KEYS if k in cols}
            for k, a in vecs.items():
                dims[k] = int(a.shape[1]) if a.ndim == 2 else 1
            if "timestamp" in cols:
                ts = np.array(t.column("timestamp").to_pylist(), dtype=np.float64)
                if fps is None and n > 1 and ts[1] > ts[0]:
                    fps = 1.0 / (ts[1] - ts[0])

            success = True
            if os.path.exists(ann_path):
                ann = json.load(open(ann_path))
                instruction = instruction or ann.get("instruction")
                task_name = task_name or ann.get("task_name")
                success = bool(ann.get("meta_data", {}).get("success", True))

            if ee in done_ep and ee in done_st:      # already reconstructed -> reuse
                ep_lines[ee] = done_ep[ee]
                st_lines[ee] = done_st[ee]
                continue

            stats = {}
            for k, a in vecs.items():
                stats[k] = _stats_vec(a)
            for k in SCALAR_KEYS:
                if k in cols:
                    stats[k] = _stats_scalar(np.array(t.column(k).to_pylist()))
            if image_stats:
                for vk in video_keys:
                    vp = os.path.join(video_root, vk, f"episode_{ee:08d}.mp4")
                    fr = _decode_video_frames(vp) if os.path.exists(vp) else []
                    if fr:
                        stats[vk] = _stats_image(fr)
            ep_lines[ee] = {"episode_index": ee, "tasks": [instruction or task_name or ""],
                            "length": int(n), "success": success}
            st_lines[ee] = {"episode_index": ee, "stats": stats}
            new_here += 1

        order = sorted(ep_lines)
        if not order:
            continue
        with open(ep_path, "w") as fh:
            for ee in order:
                fh.write(json.dumps(ep_lines[ee]) + "\n")
        with open(st_path, "w") as fh:
            for ee in order:
                fh.write(json.dumps(st_lines[ee]) + "\n")

        H = W = 0
        if video_keys:
            vs = glob.glob(os.path.join(video_root, video_keys[0], "*.mp4"))
            if vs:
                H, W = _video_dims(vs[0])
        feats = build_features(
            video_keys, (H, W),
            dims.get("observation.state", 14), dims.get("observation.endpose", 16),
            dims.get("observation.task_state", 0), dims.get("action", 14),
            fps or 16.666666666666668,
        )
        info = {
            "task_index": ti, "task_name": task_name or tt,
            "task_strings": {str(ti): instruction or task_name or ""},
            "n_episodes": len(order),
            "n_frames": int(sum(ep_lines[e]["length"] for e in order)),
            "n_videos": len(order) * len(video_keys),
            "fps": fps or 16.666666666666668, "robot_type": "aloha-agilex",
            "features": feats, "chunks_size": 1000, "video_codec": "h264",
        }
        json.dump(info, open(os.path.join(parts, f"{tt}.info.json"), "w"), indent=2)
        grand_new += new_here
        print(f"[rebuild] {tt}: {len(order)} episodes total ({new_here} newly computed) -> _parts")
    return grand_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-image-stats", action="store_true")
    a = ap.parse_args()
    rebuild_root(a.root, full=a.full, image_stats=not a.no_image_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
