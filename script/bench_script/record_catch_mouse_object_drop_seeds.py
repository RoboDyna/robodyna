#!/usr/bin/env python3
"""One-shot catch_mouse_object_drop recorder for fixed seeds (no infinite retry loop)."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from copy import deepcopy

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.collect_data import class_decorator
from record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
)


def record_seed(seed: int, config_name: str = "demo_dynamic") -> dict:
    task_name = "catch_mouse_object_drop"
    save_root = os.path.abspath(f"./tmp/tmp_{task_name}")
    video_dir = os.path.join(save_root, "video")
    os.makedirs(video_dir, exist_ok=True)
    ver = next_version(video_dir)
    tag = f"seed{seed}"
    stem = f"v{ver}_{tag}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")
    out_topdown = os.path.join(video_dir, f"{stem}_topdown.mp4")
    out_side = os.path.join(video_dir, f"{stem}_sidebyside.mp4")

    for junk in ("data", ".cache", "seed.txt", "scene_info.json", "_traj_data", task_name):
        p = os.path.join(save_root, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(task_name, config_name, save_root, None, [])
    args["episode_num"] = 1
    args["save_failed_cases"] = True
    args["check_render_success"] = False
    args["collect_data"] = True

    # ---- plan / traj pass ----
    task = class_decorator(task_name)
    _orig = task.setup_demo

    def _setup(**kwags):
        kwags["seed"] = int(seed)
        _orig(**kwags)
        configure_topdown_camera(task)

    task.setup_demo = _setup

    plan_args = deepcopy(args)
    plan_args["need_plan"] = True
    plan_args["save_data"] = False
    print(f"[plan] seed={seed}", flush=True)
    task.setup_demo(now_ep_num=0, **plan_args)
    task.play_once()
    ok = bool(task.plan_success and task.check_success())
    print(f"[plan] success={ok}", flush=True)
    if not ok:
        task.close_env()
        raise RuntimeError(f"plan pass failed for seed {seed}")
    task.save_traj_data(0)
    task.close_env()
    time.sleep(0.5)

    # ---- render pass (replay saved joint paths) ----
    task = class_decorator(task_name)
    _orig = task.setup_demo

    def _setup2(**kwags):
        kwags["seed"] = int(seed)
        _orig(**kwags)
        configure_topdown_camera(task)

    task.setup_demo = _setup2

    def _merge_dual(self):
        if not self.save_data:
            return
        cache_path = f"{self.save_dir}/.cache/episode{self.ep_num}/"
        fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
        merge_dual_view_videos(cache_path, out_head, out_topdown, out_side, fps=fps)

    task.merge_pkl_to_hdf5_video = _merge_dual.__get__(task, task.__class__)

    render_args = deepcopy(args)
    render_args["need_plan"] = False
    render_args["save_data"] = True
    render_args["render_freq"] = 0
    print(f"[render] seed={seed}", flush=True)
    task.setup_demo(now_ep_num=0, **render_args)
    traj_data = task.load_tran_data(0)
    render_args["left_joint_path"] = traj_data["left_joint_path"]
    render_args["right_joint_path"] = traj_data["right_joint_path"]
    task.set_path_lst(render_args)
    task.play_once()
    task.merge_pkl_to_hdf5_video()
    task.close_env()

    final_dir = os.path.abspath("./docs/final_task_demos/catch_mouse_object_drop/eval")
    os.makedirs(final_dir, exist_ok=True)
    for src, name in (
        (out_head, f"seed{seed}_head.mp4"),
        (out_topdown, f"seed{seed}_topdown.mp4"),
        (out_side, f"seed{seed}_sidebyside.mp4"),
    ):
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(final_dir, name))

    print(f"Demo seed={seed}:")
    print("  head     :", out_head)
    print("  top-down :", out_topdown)
    print("  side-by-side:", out_side)
    return {"seed": seed, "version": ver, "ok": True,
            "head": out_head, "topdown": out_topdown, "sidebyside": out_side}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,41,59")
    ns = parser.parse_args()
    seeds = [int(s.strip()) for s in ns.seeds.split(",") if s.strip()]
    for seed in seeds:
        print(f"\n######## recording seed {seed} ########", flush=True)
        try:
            record_seed(seed)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED seed {seed}: {type(exc).__name__}: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
