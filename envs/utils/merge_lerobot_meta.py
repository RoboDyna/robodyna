"""Merge per-task LeRobot v2.1 meta slices into the global meta/ for the combined suite dataset.

Each collection job writes its task's data/videos/annotations DIRECTLY into the shared dataset's
`task-{TT:04d}/` folders plus per-task meta slices under `meta/_parts/`. This script concatenates
those slices into the standard monolithic meta files that stock lerobot loads. It is pure
concatenation + count aggregation — no data is read or rewritten, and episode_index / frame index
were pre-allocated per task so nothing needs reindexing.

Usage:  python envs/utils/merge_lerobot_meta.py --root data_lerobot/domino_suite
"""
import argparse
import glob
import json
import os

DATA_PATH = "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet"
VIDEO_PATH = "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4"
ANNOTATION_PATH = "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="combined dataset root (contains meta/_parts/)")
    args = ap.parse_args()
    parts = os.path.join(args.root, "meta", "_parts")
    info_slices = sorted(glob.glob(os.path.join(parts, "task-*.info.json")))
    if not info_slices:
        raise SystemExit(f"No meta slices found in {parts} (run collection with export_lerobot first).")

    ep_lines, stat_lines, task_strings = [], [], {}
    total_episodes = total_frames = total_videos = 0
    features = fps = robot_type = chunks_size = None

    for sl in info_slices:
        info = json.load(open(sl))
        tt = os.path.basename(sl).split(".")[0]  # task-0003
        ep_lines += [json.loads(l) for l in open(os.path.join(parts, f"{tt}.episodes.jsonl"))]
        stat_lines += [json.loads(l) for l in open(os.path.join(parts, f"{tt}.episodes_stats.jsonl"))]
        for ti, ts in info["task_strings"].items():
            task_strings[int(ti)] = ts
        total_episodes += info["n_episodes"]
        total_frames += info["n_frames"]
        total_videos += info["n_videos"]
        features = features or info["features"]
        fps = fps or info["fps"]
        robot_type = robot_type or info["robot_type"]
        chunks_size = chunks_size or info["chunks_size"]

    ep_lines.sort(key=lambda d: d["episode_index"])
    stat_lines.sort(key=lambda d: d["episode_index"])

    meta = os.path.join(args.root, "meta")
    with open(os.path.join(meta, "episodes.jsonl"), "w") as fh:
        for l in ep_lines:
            fh.write(json.dumps(l) + "\n")
    with open(os.path.join(meta, "episodes_stats.jsonl"), "w") as fh:
        for l in stat_lines:
            fh.write(json.dumps(l) + "\n")
    with open(os.path.join(meta, "tasks.jsonl"), "w") as fh:
        for ti in sorted(task_strings):
            fh.write(json.dumps({"task_index": ti, "task": task_strings[ti]}) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_strings),
        "total_videos": total_videos,
        "total_chunks": len(info_slices),
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "annotation_path": ANNOTATION_PATH,
        "features": features,
    }
    with open(os.path.join(meta, "info.json"), "w") as fh:
        json.dump(info, fh, indent=4)
    print(f"[merge] {len(info_slices)} tasks | {total_episodes} episodes | {total_frames} frames "
          f"-> {meta}/{{info.json,episodes.jsonl,episodes_stats.jsonl,tasks.jsonl}}")


if __name__ == "__main__":
    main()
