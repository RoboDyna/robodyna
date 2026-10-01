#!/usr/bin/env python3
"""Eval + tagged demo recordings for sort_apples_belt option matrix.

Conditions:
  default   — color_mode=alternating, rotten_prob=0
  opt1      — color_mode=random,      rotten_prob=0
  opt2      — color_mode=alternating, rotten_prob=0.3 (always ≥1 rotten; P for a 2nd)
  opt1+2    — color_mode=random,      rotten_prob=0.3

Success (must match envs/sort_apples_belt.check_success):
  - every red apple in the red basket
  - every green apple in the green basket
  - rotten (if any) in the garbage dump — never left/right baskets

Usage (repo root, robodyna env, headless Vulkan)::

  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
  python script/bench_script/run_sort_apples_belt_option_suite.py
  python script/bench_script/run_sort_apples_belt_option_suite.py --n 5 --skip-record
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import numpy as np
import yaml

from envs import CONFIGS_PATH
from script.bench_script.record_demo import record_demo
from script.collect_data import class_decorator, get_embodiment_config

TASK = "sort_apples_belt"
CONFIG = "demo_dynamic"
FINAL_DIR = os.path.join("docs/final_task_demos", TASK)

# Opt2: rotten_prob>0 always yields ≥1 rotten; P=rotten_prob may add a second.
# Demos use the suite so the garbage-bin behavior is visible.
# n_apples 4–5: expert is reliable at this length; 7–10 still flaky mid-stream.
_SHARED = {"n_apples_min": 4, "n_apples_max": 5}
CONDITIONS = [
    {
        "key": "default",
        "tag": "default",
        "label": "default (alternating, no rotten)",
        "args": {"color_mode": "alternating", "rotten_prob": 0.0, **_SHARED},
    },
    {
        "key": "opt1",
        "tag": "opt1",
        "label": "Opt 1 — random colors",
        "args": {"color_mode": "random", "rotten_prob": 0.0, **_SHARED},
    },
    {
        "key": "opt2",
        "tag": "opt2",
        "label": "Opt 2 — always ≥1 rotten (dump)",
        "args": {"color_mode": "alternating", "rotten_prob": 0.3, **_SHARED},
    },
    {
        "key": "opt1+2",
        "tag": "opt1+2",
        "label": "Opt 1+2 — random colors + always ≥1 rotten",
        "args": {"color_mode": "random", "rotten_prob": 0.3, **_SHARED},
    },
]


def _base_args() -> dict:
    with open(f"./task_config/{CONFIG}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = TASK
    args["task_config"] = CONFIG
    args["render_freq"] = 0
    args["collect_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["export_lerobot"] = False
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = False
    args["data_type"]["third_view"] = False
    args.setdefault("camera", {})
    args["camera"]["collect_wrist_camera"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)
    et = args["embodiment"]
    args["left_robot_file"] = emb[et[0]]["file_path"]
    args["right_robot_file"] = emb[et[1]]["file_path"]
    args["embodiment_dis"] = et[2]
    args["dual_arm_embodied"] = False
    args["embodiment_name"] = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def _color_name(c: int) -> str:
    return {0: "red", 1: "green", 2: "rotten"}.get(int(c), str(c))


def _landing_report(task) -> dict:
    """Per-apple destinations + rotten-in-basket flag for the summary table."""
    task._eval_landings()
    green_on_left = bool(getattr(task, "green_on_left", True))
    red_side = "right" if green_on_left else "left"
    green_side = "left" if green_on_left else "right"
    landings = []
    for i in range(int(task.n_apples)):
        color = int(task.apple_colors[i])
        delivered = task.delivered[i]
        want = task._target_side_for_apple(i)
        ok = bool(task.results[i])
        landings.append(
            {
                "idx": i,
                "color": _color_name(color),
                "want": want,
                "got": delivered,
                "ok": ok,
            }
        )
    rotten_in_basket = bool(task._rotten_in_basket())
    return {
        "n_apples": int(task.n_apples),
        "has_rotten": bool(getattr(task, "has_rotten", False)),
        "rotten_idx": getattr(task, "rotten_idx", None),
        "green_on_left": green_on_left,
        "red_bin": red_side,
        "green_bin": green_side,
        "rotten_in_basket": rotten_in_basket,
        "sorting_accuracy": float(getattr(task, "sorting_accuracy", 0.0) or 0.0),
        "landings": landings,
    }


def eval_condition(cond: dict, n: int, seed0: int = 40) -> dict:
    args = _base_args()
    args.setdefault("task_args", {}).setdefault(TASK, {})
    args["task_args"][TASK].update(cond["args"])

    rows = []
    n_plan = n_succ = 0
    for i in range(n):
        seed = seed0 + i
        task = class_decorator(TASK)
        try:
            task.setup_demo(
                now_ep_num=seed,
                seed=seed,
                is_test=True,
                **{k: v for k, v in args.items() if k != "seed"},
            )
            # Plasticbox instance set must stay fixed (task default).
            ids = tuple(getattr(task, "BASKET_INSTANCE_IDS", ()))
            if ids != (5, 7, 8, 9, 10):
                raise RuntimeError(f"unexpected BASKET_INSTANCE_IDS={ids}")
            task.play_once()
            plan_ok = bool(task.plan_success)
            check_ok = bool(task.check_success())
            succ = bool(plan_ok and check_ok)
            report = _landing_report(task)
        except Exception as e:
            plan_ok = False
            check_ok = False
            succ = False
            report = {"error": str(e)}
            rows.append(
                {
                    "seed": seed,
                    "plan": False,
                    "check_success": False,
                    "success": False,
                    "error": str(e),
                }
            )
            try:
                task.close_env()
            except Exception:
                pass
            continue
        n_plan += int(plan_ok)
        n_succ += int(succ)
        row = {
            "seed": seed,
            "plan": plan_ok,
            "check_success": check_ok,
            "success": succ,
            "color_mode": str(getattr(task, "color_mode", "")),
            "rotten_prob": float(getattr(task, "rotten_prob", 0.0)),
            **report,
        }
        rows.append(row)
        try:
            task.close_env()
        except Exception:
            pass

    return {
        "key": cond["key"],
        "label": cond["label"],
        "args": dict(cond["args"]),
        "n": n,
        "plan_ok": n_plan,
        "success": n_succ,
        "rows": rows,
    }


def record_condition(cond: dict) -> dict:
    overrides = []
    for k, v in cond["args"].items():
        if isinstance(v, float):
            overrides.append(f"{k}={v}")
        elif isinstance(v, bool):
            overrides.append(f"{k}={str(v).lower()}")
        else:
            overrides.append(f"{k}={v}")
    print(f"\n=== RECORD {cond['label']}  tag={cond['tag']} ===")
    return record_demo(
        TASK,
        config_name=CONFIG,
        task_arg_overrides=overrides,
        tag=cond["tag"],
    )


def gif_from_mp4(mp4: str, gif: str) -> None:
    os.makedirs(os.path.dirname(gif) or ".", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", mp4,
            "-vf", "fps=10,scale=480:-1:flags=lanczos",
            "-loop", "0",
            gif,
        ],
        check=False,
    )


def publish_demo(cond: dict, out: dict) -> dict:
    """Copy tagged demos into docs/final_task_demos/sort_apples_belt/."""
    os.makedirs(FINAL_DIR, exist_ok=True)
    tag = cond["tag"]
    published = {}
    for kind in ("head", "topdown", "sidebyside"):
        src = out.get(kind)
        if not src or not os.path.isfile(src):
            print(f"  WARN missing {kind}: {src}")
            continue
        dst = os.path.join(FINAL_DIR, f"{tag}_{kind}.mp4")
        shutil.copy2(src, dst)
        published[kind] = dst
        print(f"  PUBLISH {dst}")
    side = published.get("sidebyside")
    if side:
        gif = os.path.join(FINAL_DIR, f"{tag}_sidebyside.gif")
        gif_from_mp4(side, gif)
        if os.path.isfile(gif):
            published["sidebyside_gif"] = gif
            print(f"  PUBLISH {gif}")
    return published


def write_conditions_txt() -> None:
    path = os.path.join(FINAL_DIR, "CONDITIONS.txt")
    os.makedirs(FINAL_DIR, exist_ok=True)
    lines = [
        "sort_apples_belt — final expert demos + option suite",
        "",
        "default  : color_mode=alternating, rotten_prob=0",
        "opt1     : color_mode=random,      rotten_prob=0",
        "opt2     : color_mode=alternating, rotten_prob=0.3 (always ≥1; P for 2nd)",
        "opt1+2   : color_mode=random,      rotten_prob=0.3",
        "",
        "Suite/demo apple count: n_apples in [4, 5] (expert-stable length).",
        "Eval seeds: 100–104 (fixed).",
        "",
        "Success: all red→red bin, all green→green bin, rotten→garbage dump;",
        "         rotten in left/right basket is failure.",
        "Files: <tag>_sidebyside.mp4 (primary), _head, _topdown, _sidebyside.gif",
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Eval episodes per condition")
    parser.add_argument("--seed0", type=int, default=100,
                        help="First seed (default 100; 100–104 is 5/5 at n_apples 4–5)")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-record", action="store_true")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated condition keys: default,opt1,opt2,opt1+2",
    )
    ns = parser.parse_args()

    conds = CONDITIONS
    if ns.only:
        want = {x.strip() for x in ns.only.split(",")}
        conds = [c for c in CONDITIONS if c["key"] in want]

    summary = []
    published_all = {}
    for cond in conds:
        print("\n" + "=" * 72)
        print(f"CONDITION: {cond['label']}")
        print(
            f"  color_mode={cond['args']['color_mode']}  "
            f"rotten_prob={cond['args']['rotten_prob']}"
        )
        print("=" * 72)

        if not ns.skip_eval:
            res = eval_condition(cond, n=ns.n, seed0=ns.seed0)
            summary.append(res)
            print(f"EVAL  plan={res['plan_ok']}/{res['n']}  success={res['success']}/{res['n']}")
            for row in res["rows"]:
                if "error" in row and "landings" not in row:
                    print(" ", row)
                    continue
                land = ",".join(
                    f"{L['color'][0]}:{L['got'] or '?'}"
                    + ("" if L["ok"] else "!")
                    for L in row.get("landings", [])
                )
                print(
                    f"  seed={row['seed']} plan={row['plan']} succ={row['success']} "
                    f"has_rotten={row.get('has_rotten')} "
                    f"rotten_in_basket={row.get('rotten_in_basket')} "
                    f"acc={row.get('sorting_accuracy', 0):.2f} "
                    f"[{land}]"
                )

        if not ns.skip_record:
            out = record_condition(cond)
            published_all[cond["key"]] = publish_demo(cond, out)

    if summary:
        print("\n" + "=" * 72)
        print("SUMMARY (success / plan / n)")
        print("=" * 72)
        for res in summary:
            rate = f"{res['success']}/{res['n']}"
            print(
                f"  {res['key']:8s}  {rate:5s} succ,  "
                f"{res['plan_ok']}/{res['n']} plan  — {res['label']}"
            )

        report = {
            "task": TASK,
            "seed0": ns.seed0,
            "n_per_condition": ns.n,
            "summary": {
                res["key"]: {
                    "successes": res["success"],
                    "attempts": res["n"],
                    "plan_ok": res["plan_ok"],
                    "label": res["label"],
                    "args": res["args"],
                }
                for res in summary
            },
            "results": {res["key"]: res["rows"] for res in summary},
            "demos": published_all,
        }
        os.makedirs(FINAL_DIR, exist_ok=True)
        report_path = os.path.join(FINAL_DIR, "test_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {report_path}")

    if not ns.skip_record or summary:
        write_conditions_txt()


if __name__ == "__main__":
    main()
