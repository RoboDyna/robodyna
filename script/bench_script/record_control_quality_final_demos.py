#!/usr/bin/env python3
"""Record tagged control_quality demos into docs/final_task_demos/control_quality/.

Uses known-good seeds from the 5×4 controller suite so recording does not
retry-forever under GPU contention.
"""
from __future__ import annotations

import os
import shutil
import sys
from copy import deepcopy

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
)
from script.collect_data import class_decorator

TASK = "control_quality"
CONFIG = "demo_dynamic"
OUT = os.path.abspath(f"./docs/final_task_demos/{TASK}")
SAVE_ROOT = os.path.abspath(f"./tmp/tmp_{TASK}")
VIDEO_DIR = os.path.join(SAVE_ROOT, "video")

# Known-good seeds from test_control_quality.py (first success per condition).
CONDITIONS = {
    "default": {
        "seed": 1000,
        "args": {
            "color_mode": "alternating",
            "black_frac_max": 0.0,
            "tile_pause_s": 2.0,
            "n_tiles": 6,
        },
    },
    "opt1": {
        "seed": 2000,
        "args": {
            "color_mode": "random",
            "black_frac_max": 0.0,
            "tile_pause_s": 2.0,
            "n_tiles": 6,
        },
    },
    "opt2": {
        "seed": 3000,
        "args": {
            "color_mode": "alternating",
            "black_frac_max": 0.5,
            "tile_pause_s": 2.0,
            "n_tiles": 6,
        },
    },
    "opt1+2": {
        "seed": 4000,
        "args": {
            "color_mode": "random",
            "black_frac_max": 0.5,
            "tile_pause_s": 2.0,
            "n_tiles": 6,
        },
    },
}


def _overrides(cfg: dict) -> list[str]:
    out = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def _clean_scratch():
    for junk in ("data", ".cache", "seed.txt", "scene_info.json", "_traj_data", TASK):
        p = os.path.join(SAVE_ROOT, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def record_condition(tag: str, seed: int, cfg: dict) -> dict:
    overrides = _overrides(cfg)
    print(f"\n=== recording {tag} seed={seed} overrides={overrides} ===", flush=True)
    _clean_scratch()

    ver = next_version(VIDEO_DIR)
    stem = f"v{ver}_{tag}"
    out_head = os.path.join(VIDEO_DIR, f"{stem}_head.mp4")
    out_topdown = os.path.join(VIDEO_DIR, f"{stem}_topdown.mp4")
    out_side = os.path.join(VIDEO_DIR, f"{stem}_sidebyside.mp4")

    args = build_args(TASK, CONFIG, SAVE_ROOT, None, overrides)
    env = class_decorator(TASK)
    orig = env.setup_demo

    def setup(**kw):
        orig(**kw)
        configure_topdown_camera(env)

    env.setup_demo = setup

    def _merge(self):
        if not self.save_data:
            return
        cache = f"{self.save_dir}/.cache/episode{self.ep_num}/"
        fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
        merge_dual_view_videos(cache, out_head, out_topdown, out_side, fps=fps)

    env.merge_pkl_to_hdf5_video = _merge.__get__(env, env.__class__)

    # plan + save traj
    args2 = deepcopy(args)
    args2["need_plan"] = True
    args2["collect_data"] = False
    args2["save_data"] = True
    env.setup_demo(now_ep_num=0, seed=seed, **args2)
    env.play_once()
    assert env.plan_success and env.check_success(), (
        f"{tag}: plan/success failed on seed {seed} "
        f"(plan={env.plan_success} check={env.check_success()})"
    )
    colors = list(env.tile_colors)
    print(f"  plan ok colors={colors} black_press={env.black_press}", flush=True)
    env.save_traj_data(0)
    env.close_env()

    # render
    args2["need_plan"] = False
    args2["collect_data"] = True
    traj = env.load_tran_data(0)
    args2["left_joint_path"] = traj["left_joint_path"]
    args2["right_joint_path"] = traj["right_joint_path"]
    env.setup_demo(now_ep_num=0, seed=seed, **args2)
    env.set_path_lst(args2)
    env.play_once()
    env.close_env()
    env.merge_pkl_to_hdf5_video()

    os.makedirs(OUT, exist_ok=True)
    exports = {
        "sidebyside": os.path.join(OUT, f"{tag}_sidebyside.mp4"),
        "head": os.path.join(OUT, f"{tag}_head.mp4"),
        "topdown": os.path.join(OUT, f"{tag}_topdown.mp4"),
    }
    shutil.copy2(out_side, exports["sidebyside"])
    shutil.copy2(out_head, exports["head"])
    shutil.copy2(out_topdown, exports["topdown"])
    print(f"  wrote {exports['sidebyside']}", flush=True)
    return {"tag": tag, "seed": seed, "colors": colors, "paths": exports}


def main(only: list[str] | None = None):
    os.makedirs(OUT, exist_ok=True)
    items = list(CONDITIONS.items())
    if only:
        items = [(k, CONDITIONS[k]) for k in only]
    results = []
    for tag, spec in items:
        results.append(record_condition(tag, spec["seed"], spec["args"]))

    with open(os.path.join(OUT, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : color_mode=alternating, black_frac_max=0.0\n"
            "           red/green alternate; stamp one tile at a time under punch\n"
            "opt1     : color_mode=random,      black_frac_max=0.0\n"
            "           red/green pattern randomized\n"
            "opt2     : color_mode=alternating, black_frac_max=0.5\n"
            "           black outlier tiles must NOT be stamped\n"
            "opt1+2   : color_mode=random,      black_frac_max=0.5\n"
            "           random colors + black outliers\n\n"
            "Motion: tiles ride the belt continuously, stop under the punch\n"
            "        (tile_pause_s=2.0 max). Stamp fires when the matching key\n"
            "        is pressed.\n\n"
            "Success: every red/green tile correctly stamped;\n"
            "         every black outlier skipped (no key press).\n"
            "         Missed red/green OR stamped black → failure.\n\n"
            "Full test (5 eps × 4 conditions): see test_report.json — 20/20 success,\n"
            "all criteria_consistent=true.\n\n"
            "Files (side-by-side is the primary deliverable):\n"
            "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
            "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
            "Also: <tag>_head.mp4, <tag>_topdown.mp4\n"
        )
    print("\nDone:", [r["tag"] for r in results], flush=True)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    main(only=only)
