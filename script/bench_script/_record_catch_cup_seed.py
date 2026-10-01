#!/usr/bin/env python3
"""Record one catch_cup seed (success or fail) to dual-view mp4s."""
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
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
    _cleanup_scratch,
)
from script.collect_data import class_decorator

TASK = "catch_cup"
CONFIG = "demo_dynamic"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--save-freq", type=int, default=40)
    ns = ap.parse_args()
    seed = int(ns.seed)

    save_root = os.path.abspath(f"./tmp/tmp_{TASK}")
    video_dir = os.path.join(save_root, "video")
    final_dir = os.path.abspath("./final_task_demos/catch_cup/seedsweep")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    ver = next_version(video_dir)
    print(f"=== record seed {seed} ver={ver} ===", flush=True)
    _cleanup_scratch(save_root, TASK)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(TASK, CONFIG, save_root, None, ["allow_fail=true"])
    args.update(
        need_plan=True,
        save_data=True,
        collect_data=False,
        check_render_success=False,
        save_failed_cases=True,
        render_freq=0,
        save_freq=int(ns.save_freq),
        use_seed=False,
    )
    args.setdefault("camera", {})["head_camera_type"] = "D435"

    task = class_decorator(TASK)
    _orig = task.setup_demo

    def _setup(**kw):
        _orig(**kw)
        configure_topdown_camera(task)

    task.setup_demo = _setup
    shutil.rmtree(Path(save_root) / ".cache", ignore_errors=True)
    task.setup_demo(now_ep_num=0, seed=seed, **args)
    task.play_once()

    plan = bool(task.plan_success)
    check = bool(task.check_success())
    caught = bool(getattr(task, "_caught_on_pillow", False))
    fell = bool(getattr(task, "_fell_on_table", False))
    state = str(getattr(task, "_cup_state", "?"))
    cup_id = int(getattr(task, "cup_id", -1))
    ok = plan and check
    label = "pass" if ok else "fail"
    print(
        f"RESULT seed={seed} label={label} cup_id={cup_id} plan={plan} "
        f"check={check} caught={caught} fell={fell} state={state}",
        flush=True,
    )

    stem = f"v{ver}_seed{seed}_{label}"
    outs = [os.path.join(video_dir, f"{stem}_{k}.mp4") for k in ("head", "topdown", "sidebyside")]
    cache_path = f"{task.save_dir}/.cache/episode{task.ep_num}/"
    if not os.path.isdir(cache_path):
        print(f"ERROR missing cache {cache_path}", flush=True)
        try:
            task.close_env(clear_cache=True)
        except Exception:
            pass
        return 2

    fps = 250.0 / float(task.save_freq)
    merge_dual_view_videos(cache_path, outs[0], outs[1], outs[2], fps=fps)
    task.close_env(clear_cache=True)
    _cleanup_scratch(save_root, TASK)

    for p in outs:
        shutil.copy2(p, os.path.join(final_dir, os.path.basename(p)))
        print("OUT", p, flush=True)
    print("DONE", stem, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
