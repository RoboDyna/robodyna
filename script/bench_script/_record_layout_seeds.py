#!/usr/bin/env python3
"""Record specific seeds to head-camera demos (gallery / layout probes)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (
    build_args,
    encode_head_video,
    next_version,
    _cleanup_scratch,
)
from script.collect_data import class_decorator


def record_one(task: str, seed: int, config: str = "demo_dynamic", save_freq: int = 15) -> dict:
    """Two-pass head-only record for a fixed seed."""
    save_root = os.path.abspath(f"./tmp/tmp_{task}")
    video_dir = os.path.join(save_root, "video")
    os.makedirs(video_dir, exist_ok=True)
    ver = next_version(video_dir)
    print(f"=== {task} seed={seed} ver={ver} (head) ===", flush=True)
    _cleanup_scratch(save_root, task)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(task, config, save_root, None, [])
    args.update(
        need_plan=True,
        save_data=False,
        collect_data=False,
        check_render_success=False,
        save_failed_cases=True,
        render_freq=0,
        save_freq=int(save_freq),
        use_seed=False,
    )
    args.setdefault("camera", {})["head_camera_type"] = "D435"
    args.setdefault("data_type", {})["third_view"] = False

    env = class_decorator(task)
    shutil.rmtree(Path(save_root) / ".cache", ignore_errors=True)
    env.setup_demo(now_ep_num=0, seed=seed, **args)
    env.play_once()
    ok = bool(env.plan_success and env.check_success())
    label = "pass" if ok else "fail"
    print(f"RESULT {task} seed={seed} {label}", flush=True)
    env.save_traj_data(0)
    try:
        env.close_env()
    except Exception:
        pass

    render_args = dict(args)
    render_args.update(need_plan=False, save_data=True, render_freq=0)
    env = class_decorator(task)
    shutil.rmtree(Path(save_root) / ".cache", ignore_errors=True)
    env.setup_demo(now_ep_num=0, seed=seed, **render_args)
    env._max_episode_steps = int(os.environ.get("DEMO_MAX_STEPS", "15000"))
    env._max_episode_seconds = None
    traj = env.load_tran_data(0)
    render_args["left_joint_path"] = traj["left_joint_path"]
    render_args["right_joint_path"] = traj["right_joint_path"]
    env.set_path_lst(render_args)
    env.play_once()

    stem = f"v{ver}_seed{seed}_{label}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")
    outs = {"head": out_head, "topdown": out_head, "sidebyside": out_head}
    cache_path = f"{env.save_dir}/.cache/episode{env.ep_num}/"
    fps = 250.0 / float(env.save_freq)
    encode_head_video(cache_path, out_head, fps=fps)
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass
    _cleanup_scratch(save_root, task)
    return {"version": ver, "seed": seed, "ok": ok, **outs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--config", default="demo_dynamic")
    ns = ap.parse_args()
    for s in ns.seeds:
        record_one(ns.task, s, config=ns.config)


if __name__ == "__main__":
    main()
