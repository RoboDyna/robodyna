#!/usr/bin/env python3
"""Full controller test + tagged demos for catch_marbles_trapdoors.

Runs:
  - default: 5 random-color episodes + one forced episode per color (red/yellow/blue/green)
  - opt1, opt2, opt1+2: 5 episodes each

Then records one side-by-side demo per condition into
  docs/final_task_demos/catch_marbles_trapdoors/<tag>_sidebyside.mp4
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args, record_demo
from script.collect_data import class_decorator

TASK = "catch_marbles_trapdoors"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5
COLORS = ["red", "yellow", "blue", "green"]
# Expected arm follows trapdoor side (x), not color name — colors are shuffled per episode.
def expected_arm_from_door_x(door_x) -> str | None:
    if door_x is None:
        return None
    return "left" if float(door_x) < 0.0 else "right"

CONDITIONS = {
    "default": {
        "door_open_once": False,
        "enable_distractor": False,
    },
    "opt1": {
        "door_open_once": True,
        "enable_distractor": False,
    },
    "opt2": {
        "door_open_once": False,
        "enable_distractor": True,
        "distractor_collide": False,
    },
    "opt1+2": {
        "door_open_once": True,
        "enable_distractor": True,
        "distractor_collide": False,
    },
}


def _overrides(condition: str, extra: dict | None = None) -> list[str]:
    cfg = dict(CONDITIONS[condition])
    if extra:
        cfg.update(extra)
    out = []
    for k, v in cfg.items():
        if v is None:
            out.append(f"{k}=null")
        elif isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def _run_episode(task_args_overrides: list[str], seed: int, label: str) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}_test")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(TASK, CONFIG, save_root, option=None, task_arg_overrides=task_args_overrides)
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False

    env = class_decorator(TASK)
    row = {
        "label": label,
        "seed": seed,
        "success": False,
        "plan_success": False,
        "error": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan_ok = bool(env.plan_success)
        ok = bool(plan_ok and env.check_success())
        info = dict(getattr(env, "info", {}) or {})
        color = info.get("target_color", "")
        arm = info.get("operating_arm", "")
        door_x = info.get("target_door_x", None)
        row.update(
            {
                "success": ok,
                "plan_success": plan_ok,
                "target_color": color,
                "operating_arm": arm,
                "target_door_x": door_x,
                "ball_drop_door_idx": info.get("ball_drop_door_idx", -1),
                "used_matching_door": info.get("used_matching_door", False),
                "used_wrong_door": info.get("used_wrong_door", False),
                "ball_in_lower_box": info.get("ball_in_lower_box", False),
                "ball_in_box": info.get("ball_in_box", False),
                "target_door_opened": info.get("target_door_opened", False),
                "ball_still_on_top": info.get("ball_still_on_top", False),
                "distractor_through_any": info.get("distractor_through_any", False),
                "distractor_in_lower_box": info.get("distractor_in_lower_box", False),
                "door_open_once": info.get("door_open_once", None),
                "enable_distractor": info.get("enable_distractor", None),
                "metric_ok": bool(
                    info.get("ball_in_box")
                    and info.get("target_door_opened")
                    and not info.get("used_wrong_door")
                    and not info.get("distractor_through_any")
                    and not info.get("distractor_in_lower_box")
                ),
                "arm_ok": (
                    expected_arm_from_door_x(door_x) == arm
                    if door_x is not None and arm
                    else None
                ),
            }
        )
        # Success requires the metric fields, not just check_success()'s bool.
        row["success"] = bool(row["success"] and row["metric_ok"])
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        try:
            env.close_env()
        except Exception:
            pass
        if args.get("render_freq"):
            try:
                env.viewer.close()
            except Exception:
                pass
        # Free planner CUDA caches between episodes (shared GPU).
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION, extra: dict | None = None, label_prefix: str | None = None):
    overrides = _overrides(condition, extra)
    prefix = label_prefix or condition
    results = []
    seed = 0
    attempts = 0
    max_attempts = n * 8  # allow retries for transient plan fails
    print(f"\n=== {prefix}: need {n} successes | overrides={overrides} ===")
    while len([r for r in results if r["success"]]) < n and attempts < max_attempts:
        label = f"{prefix}/ep{attempts}"
        row = _run_episode(overrides, seed=seed, label=label)
        results.append(row)
        attempts += 1
        seed += 1
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} color={row.get('target_color')} "
            f"arm={row.get('operating_arm')} door_x={row.get('target_door_x')} "
            f"match={row.get('used_matching_door')} in_box={row.get('ball_in_lower_box')} "
            f"on_top={row.get('ball_still_on_top')} wrong={row.get('used_wrong_door')} "
            f"dist_fail={row.get('distractor_through_any')} dist_in={row.get('distractor_in_lower_box')} "
            f"arm_ok={row.get('arm_ok')} err={row.get('error')}"
        )
        time.sleep(0.2)
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {prefix}: {n_ok}/{n} successes in {attempts} attempts ===")
    return results


def run_color_coverage():
    """Default condition: force each of the four colors once (controller + arm side)."""
    results = []
    print("\n=== default color coverage (all 4 colors) ===")
    for i, color in enumerate(COLORS):
        overrides = _overrides("default", {"target_color": color})
        # retry a few seeds per color until success
        ok_row = None
        for trial in range(6):
            row = _run_episode(overrides, seed=1000 + i * 10 + trial, label=f"default/color_{color}")
            results.append(row)
            print(
                f"  color={color} trial={trial} [{'OK' if row['success'] else 'FAIL'}] "
                f"arm={row.get('operating_arm')} expected={expected_arm_from_door_x(row.get('target_door_x'))} "
                f"door_x={row.get('target_door_x')} arm_ok={row.get('arm_ok')} "
                f"in_box={row.get('ball_in_lower_box')}"
            )
            if row["success"] and row.get("arm_ok"):
                ok_row = row
                break
            time.sleep(0.2)
        if ok_row is None:
            print(f"  !! color {color} FAILED to produce a valid success with correct arm")
    return results


def record_and_export_demos():
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    # File tag uses opt1_opt2 (filesystem-safe); display name is opt1+2.
    demo_tags = {
        "default": "default",
        "opt1": "opt1",
        "opt2": "opt2",
        "opt1+2": "opt1_opt2",
    }
    exported = {}
    for condition, file_tag in demo_tags.items():
        print(f"\n=== recording demo: {condition} (tag={file_tag}) ===")
        info = record_demo(
            TASK,
            config_name=CONFIG,
            task_arg_overrides=_overrides(condition),
            tag=file_tag,
        )
        src = info["sidebyside"]
        dst = os.path.join(out_dir, f"{file_tag}_sidebyside.mp4")
        shutil.copy2(src, dst)
        # Also keep head/topdown for completeness.
        for key, suffix in (("head", "head"), ("topdown", "topdown")):
            s = info[key]
            d = os.path.join(out_dir, f"{file_tag}_{suffix}.mp4")
            shutil.copy2(s, d)
        exported[condition] = dst
        print(f"  copied -> {dst}")
    with open(os.path.join(out_dir, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : door_open_once=false, enable_distractor=false "
            "(≤3 opens/door, random target color)\n"
            "opt1     : door_open_once=true,  enable_distractor=false "
            "(open once only)\n"
            "opt2     : door_open_once=false, enable_distractor=true "
            "(≤3 opens/door + black distractor)\n"
            "opt1+2   : door_open_once=true,  enable_distractor=true "
            "(open once + distractor); files tagged opt1_opt2\n\n"
            "Success: target marble falls through matching-color trapdoor into lower box.\n"
            "Fail: stays on top, wrong-color door, or distractor through any door into lower box.\n"
            "Arm: left if matching trapdoor is on left half, right if on right half.\n\n"
            "Files: <tag>_sidebyside.mp4, <tag>_head.mp4, <tag>_topdown.mp4\n"
        )
    return exported


def main():
    all_results = {}
    all_results["default_colors"] = run_color_coverage()
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        all_results[cond] = run_condition(cond, n=N_PER_CONDITION)

    # Summarize
    summary = {}
    print("\n========== SUMMARY ==========")
    for key, rows in all_results.items():
        n_ok = sum(1 for r in rows if r["success"])
        n_arm = sum(1 for r in rows if r.get("arm_ok") is True)
        n_arm_checked = sum(1 for r in rows if r.get("arm_ok") is not None)
        summary[key] = {
            "successes": n_ok,
            "attempts": len(rows),
            "arm_ok": n_arm,
            "arm_checked": n_arm_checked,
        }
        print(f"  {key}: {n_ok} successes / {len(rows)} attempts | arm_ok {n_arm}/{n_arm_checked}")

    # Hard requirements
    color_ok = True
    for color in COLORS:
        rows = [r for r in all_results["default_colors"] if r.get("target_color") == color and r["success"] and r.get("arm_ok")]
        if not rows:
            color_ok = False
            print(f"  MISSING success for color={color}")
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        if summary[cond]["successes"] < N_PER_CONDITION:
            print(f"  SHORTFALL {cond}: only {summary[cond]['successes']}/{N_PER_CONDITION}")

    exported = {}
    if os.environ.get("SKIP_DEMOS", "").strip() not in ("1", "true", "True"):
        exported = record_and_export_demos()
    else:
        print("\n=== SKIP_DEMOS set; not recording demos in this test run ===")

    report_path = os.path.abspath(f"./docs/final_task_demos/{TASK}/test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": all_results, "demos": exported}, f, indent=2)
    print(f"\nWrote report: {report_path}")

    if not color_ok or any(summary[c]["successes"] < N_PER_CONDITION for c in ("default", "opt1", "opt2", "opt1+2")):
        print("\nTEST SUITE: FAILED requirements")
        sys.exit(1)
    print("\nTEST SUITE: PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
