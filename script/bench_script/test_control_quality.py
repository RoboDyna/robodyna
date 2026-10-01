#!/usr/bin/env python3
"""Full controller test + tagged demos for control_quality.

Conditions (5 episodes each):
  default : color_mode=alternating, black_frac_max=0.0
  opt1    : color_mode=random,      black_frac_max=0.0
  opt2    : color_mode=alternating, black_frac_max=0.5  (black outliers)
  opt1+2  : color_mode=random,      black_frac_max=0.5

Success: every red/green tile correctly stamped; every black outlier skipped
(no key press). Missed red/green or stamped black → failure.

Demos land in:
  docs/final_task_demos/control_quality/<tag>_sidebyside.mp4
  with tags: default, opt1, opt2, opt1+2
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

TASK = "control_quality"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "color_mode": "alternating",
        "black_frac_max": 0.0,
        "tile_pause_s": 2.0,
        "n_tiles": 6,
    },
    "opt1": {
        "color_mode": "random",
        "black_frac_max": 0.0,
        "tile_pause_s": 2.0,
        "n_tiles": 6,
    },
    "opt2": {
        "color_mode": "alternating",
        "black_frac_max": 0.5,
        "tile_pause_s": 2.0,
        "n_tiles": 6,
    },
    "opt1+2": {
        "color_mode": "random",
        "black_frac_max": 0.5,
        "tile_pause_s": 2.0,
        "n_tiles": 6,
    },
}

DEMO_FILE_TAGS = {
    "default": "default",
    "opt1": "opt1",
    "opt2": "opt2",
    "opt1+2": "opt1+2",
}


def _overrides(condition: str) -> list[str]:
    out = []
    for k, v in CONDITIONS[condition].items():
        if isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def _snapshot(env) -> dict:
    colors = list(getattr(env, "tile_colors", []))
    marked = [bool(x) for x in getattr(env, "tile_marked", [])]
    correct = [bool(x) for x in getattr(env, "tile_correct", [])]
    skipped = [bool(x) for x in getattr(env, "tile_skipped", [])]
    missed = [bool(x) for x in getattr(env, "tile_missed", [])]
    return {
        "color_mode": str(getattr(env, "color_mode", "")),
        "black_frac_max": float(getattr(env, "black_frac_max", 0.0)),
        "tile_pause_s": float(getattr(env, "tile_pause_s", 0.0)),
        "n_tiles": int(getattr(env, "n_tiles", len(colors))),
        "tile_colors": colors,
        "tile_marked": marked,
        "tile_correct": correct,
        "tile_skipped": skipped,
        "tile_missed": missed,
        "black_press": bool(getattr(env, "black_press", False)),
        "black_press_count": int(getattr(env, "black_press_count", 0)),
        "n_black": int(sum(1 for c in colors if c == "black")),
        "n_red_green": int(sum(1 for c in colors if c in ("red", "green"))),
        "n_correct_rg": int(sum(
            1 for i, c in enumerate(colors)
            if c in ("red", "green") and i < len(marked) and marked[i] and correct[i]
        )),
        "option_label": str(
            env._option_label() if hasattr(env, "_option_label") else ""
        ),
    }


def _criteria_ok(snap: dict) -> bool:
    """Independent success check matching the task contract.

    - every red/green tile marked AND correct
    - no red/green missed
    - every black skipped (not marked), and no black_press
    """
    if snap["black_press"]:
        return False
    colors = snap["tile_colors"]
    marked = snap["tile_marked"]
    correct = snap["tile_correct"]
    skipped = snap["tile_skipped"]
    missed = snap["tile_missed"]
    if not colors:
        return False
    for i, color in enumerate(colors):
        if color == "black":
            if marked[i] or not skipped[i]:
                return False
        else:
            if missed[i] or (not marked[i]) or (not correct[i]):
                return False
    return True


def _condition_shape_ok(condition: str, snap: dict) -> bool:
    """Sanity-check that the sampled episode matches the condition knobs."""
    cfg = CONDITIONS[condition]
    if snap["color_mode"] != cfg["color_mode"]:
        return False
    # black_frac_max is continuous; require zeros vs positive as configured
    if float(cfg["black_frac_max"]) <= 0.0:
        if snap["n_black"] != 0 or snap["black_frac_max"] > 0.0:
            return False
    else:
        if snap["black_frac_max"] <= 0.0:
            return False
        # With n>=2 and black_frac_max>0 the sampler guarantees ≥1 black
        if snap["n_tiles"] >= 2 and snap["n_black"] < 1:
            return False
    return True


def _run_episode(task_args_overrides: list[str], seed: int, label: str, condition: str) -> dict:
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
        "criteria_ok": False,
        "check_success": False,
        "shape_ok": False,
        "criteria_consistent": False,
        "error": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(env.check_success())
        snap = _snapshot(env)
        criteria_ok = _criteria_ok(snap)
        shape_ok = _condition_shape_ok(condition, snap)
        consistent = bool(check_ok == criteria_ok)
        ok = bool(plan_ok and check_ok and criteria_ok and shape_ok)
        row.update(
            {
                "success": ok,
                "plan_success": plan_ok,
                "check_success": check_ok,
                "criteria_ok": criteria_ok,
                "shape_ok": shape_ok,
                "criteria_consistent": consistent,
                **snap,
            }
        )
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
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION):
    overrides = _overrides(condition)
    results = []
    print(f"\n=== {condition}: {n} tests | overrides={overrides} ===", flush=True)
    for i in range(n):
        seed = 1000 * (list(CONDITIONS.keys()).index(condition) + 1) + i
        label = f"{condition}/ep{i}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"shape={row.get('shape_ok')} consistent={row.get('criteria_consistent')} "
            f"mode={row.get('color_mode')} black_frac={row.get('black_frac_max')} "
            f"colors={row.get('tile_colors')} missed={row.get('tile_missed')} "
            f"black_press={row.get('black_press')} err={row.get('error')}",
            flush=True,
        )
        time.sleep(0.05)
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {condition}: {n_ok}/{n} successes ===", flush=True)
    return results


def record_and_export_demos():
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    exported = {}
    for condition, file_tag in DEMO_FILE_TAGS.items():
        print(f"\n=== recording demo: {condition} (tag={file_tag}) ===", flush=True)
        info = record_demo(
            TASK,
            config_name=CONFIG,
            task_arg_overrides=_overrides(condition),
            tag=file_tag,
        )
        for key, suffix in (
            ("sidebyside", "sidebyside"),
            ("head", "head"),
            ("topdown", "topdown"),
        ):
            src = info[key]
            dst = os.path.join(out_dir, f"{file_tag}_{suffix}.mp4")
            shutil.copy2(src, dst)
            if key == "sidebyside":
                exported[condition] = dst
                print(f"  copied -> {dst}", flush=True)
    with open(os.path.join(out_dir, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
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
            "        (tile_pause_s=2.0 max), stamp fires when the matching key\n"
            "        is pressed.\n\n"
            "Success: every red/green tile correctly stamped;\n"
            "         every black outlier skipped (no key press).\n"
            "         Missed red/green OR stamped black → failure.\n\n"
            "Files: <tag>_sidebyside.mp4, <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    all_results = {}
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        all_results[cond] = run_condition(cond, n=N_PER_CONDITION)

    summary = {}
    print("\n========== SUMMARY ==========", flush=True)
    all_ok = True
    for key, rows in all_results.items():
        n_ok = sum(1 for r in rows if r["success"])
        n = len(rows)
        n_consistent = sum(1 for r in rows if r.get("criteria_consistent"))
        summary[key] = {
            "successes": n_ok,
            "attempts": n,
            "rate": n_ok / max(n, 1),
            "criteria_consistent": n_consistent,
            "config": CONDITIONS[key],
        }
        print(
            f"  {key}: {n_ok}/{n}  criteria_consistent={n_consistent}/{n}",
            flush=True,
        )
        if n_ok < N_PER_CONDITION:
            all_ok = False

    exported = record_and_export_demos()

    report = {
        "task": TASK,
        "n_per_condition": N_PER_CONDITION,
        "summary": summary,
        "conditions": {
            k: {
                **summary[k],
                "episodes": all_results[k],
            }
            for k in all_results
        },
        "demos": exported,
        "success_criterion": (
            "all red/green tiles correctly stamped; "
            "black outliers skipped (no press); "
            "missed red/green or stamped black → fail"
        ),
    }
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}", flush=True)
    print(f"Demos in {out_dir}", flush=True)
    if not all_ok:
        print("\n!! SOME CONDITIONS DID NOT REACH 5/5 SUCCESSES", flush=True)
        sys.exit(1)
    print("\nAll conditions 5/5 successes.", flush=True)


if __name__ == "__main__":
    main()
