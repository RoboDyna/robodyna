#!/usr/bin/env python3
"""Record one successful stop_ball demo in a single plan+render pass.

Usage:
  python script/bench_script/_record_stop_ball_one.py <tag> <prefer_seed> <angle>
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
)
from script.collect_data import class_decorator

TASK = "stop_ball"
CONFIG = "demo_dynamic"


def side_for(angle: float, seed: int) -> str:
    if abs(angle) < 1e-6:
        return "left" if seed % 2 == 0 else "right"
    return "right" if angle > 0.0 else "left"


def patch_heading(env, angle: float) -> None:
    orig_load = env.load_actors

    def load_actors():
        orig_build = env._build_trajectory
        orig_feasible = env._feasible_angle

        def build():
            env._sample_roll_heading = lambda: (
                float(angle),
                np.array([np.sin(angle), -np.cos(angle)], dtype=np.float64),
            )
            env._feasible_angle = lambda ang, *a, **k: float(ang)
            try:
                return orig_build()
            finally:
                env._feasible_angle = orig_feasible

        env._build_trajectory = build
        return orig_load()

    env.load_actors = load_actors


def record_seed(tag: str, seed: int, angle: float) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}")
    video_dir = os.path.join(save_root, "video")
    ver = next_version(video_dir)
    stem = f"v{ver}_{tag}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")
    out_topdown = os.path.join(video_dir, f"{stem}_topdown.mp4")
    out_side = os.path.join(video_dir, f"{stem}_sidebyside.mp4")

    scratch = os.path.join(save_root, f".scratch_{tag}")
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch, exist_ok=True)

    side = side_for(angle, seed)
    args = build_args(
        TASK, CONFIG, scratch, option=None,
        task_arg_overrides=[f"arm_side={side}"],
    )
    # Single pass: plan while saving camera frames.
    args.update(
        need_plan=True,
        save_data=True,
        collect_data=True,
        render_freq=0,
        eval_video_log=False,
        check_render_success=True,
        use_seed=False,
        episode_num=1,
    )

    env = class_decorator(TASK)
    patch_heading(env, angle)
    orig_setup = env.setup_demo

    def setup_demo(**kwags):
        kwags["seed"] = int(seed)
        orig_setup(**kwags)
        configure_topdown_camera(env)

    env.setup_demo = setup_demo

    try:
        print(
            f"\033[93m[Record {tag} seed={seed} angle={angle:+.2f} arm={side}]\033[0m",
            flush=True,
        )
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        ok = bool(env.plan_success and env.check_success())
        if not ok:
            raise RuntimeError(
                f"fail plan={env.plan_success} check={env.check_success()} "
                f"fell={env._fell_off} state={env._ball_state} "
                f"angle={env._roll_angle:.3f} xyz={np.round(env._ball_centre(), 3).tolist()}"
            )
        fps = 250.0 / float(env.save_freq) if env.save_freq else 15.0
        cache = os.path.join(env.save_dir, ".cache", "episode0")
        if not os.path.isdir(cache):
            # Some setups nest under task/config.
            cache = os.path.join(scratch, ".cache", "episode0")
        merge_dual_view_videos(cache, out_head, out_topdown, out_side, fps=fps)
        print(f"SUCCESS tag={tag} seed={seed}", flush=True)
        print(f"  head     : {out_head}", flush=True)
        print(f"  top-down : {out_topdown}", flush=True)
        print(f"  side-by-side: {out_side}", flush=True)
        return {
            "tag": tag,
            "seed": seed,
            "angle": angle,
            "arm": side,
            "version": ver,
            "head": out_head,
            "topdown": out_topdown,
            "sidebyside": out_side,
        }
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            try:
                env.close_env()
            except Exception:
                pass
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    tag = sys.argv[1]
    prefer = int(sys.argv[2])
    angle = float(sys.argv[3])
    for seed in [prefer] + [prefer + d for d in range(1, 25)]:
        try:
            record_seed(tag, seed, angle)
            return
        except Exception as e:  # noqa: BLE001
            print(f"FAIL tag={tag} seed={seed}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    raise SystemExit(f"No successful seed for {tag}")


if __name__ == "__main__":
    main()
