#!/usr/bin/env python3
"""Full controller test + tagged demos for punch_dual_holes.

Conditions (5 episodes each):
  default : discrete stop-under-stamp, no missing tile  (both arms together)
  opt1    : missing_tile_mode=true                      (alternate 1- vs 2-arm)
  opt2    : belt_continous_motion=true                  (continuous belts)
  opt1+2  : missing_tile_mode + continuous

Success: every present tile is punched (missing slots skipped); no empty-slot press.

Demos land in:
  docs/final_task_demos/punch_dual_holes/<tag>_sidebyside.mp4
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

TASK = "punch_dual_holes"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "missing_tile_mode": False,
        "belt_continous_motion": False,
        "tile_pause_s": 2.0,
    },
    "opt1": {
        "missing_tile_mode": True,
        "belt_continous_motion": False,
        "tile_pause_s": 2.0,
    },
    "opt2": {
        "missing_tile_mode": False,
        "belt_continous_motion": True,
        "tile_pause_s": 2.0,
    },
    "opt1+2": {
        "missing_tile_mode": True,
        "belt_continous_motion": True,
        "tile_pause_s": 2.0,
    },
}

# Filesystem-safe tag; display name keeps opt1+2.
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
    def _bools(d, side):
        return [bool(x) for x in d[side]]

    return {
        "missing_tile_mode": getattr(env, "missing_tile_mode", None),
        "missing_tile_side": getattr(env, "missing_tile_side", None),
        "missing_tile_index": getattr(env, "missing_tile_index", None),
        "belt_continous_motion": bool(getattr(env, "belt_continous_motion", False)),
        "tile_pause_s": float(getattr(env, "tile_pause_s", 0.0)),
        "left_missing": _bools(env.page_missing, "left"),
        "right_missing": _bools(env.page_missing, "right"),
        "left_punched": _bools(env.page_punched, "left"),
        "right_punched": _bools(env.page_punched, "right"),
        "left_missed": _bools(env.page_missed, "left"),
        "right_missed": _bools(env.page_missed, "right"),
        "left_offsets": [
            None if o is None else float(o) for o in env.page_offset["left"]
        ],
        "right_offsets": [
            None if o is None else float(o) for o in env.page_offset["right"]
        ],
        "invalid_empty_press": bool(getattr(env, "invalid_empty_press", False)),
        "punch_score_mean": float(getattr(env, "punch_score_mean", 0.0)),
    }


def _present_all_punched(snap: dict) -> bool:
    for side in ("left", "right"):
        missing = snap[f"{side}_missing"]
        missed = snap[f"{side}_missed"]
        offsets = snap[f"{side}_offsets"]
        for k, is_missing in enumerate(missing):
            if is_missing:
                continue
            if missed[k] or offsets[k] is None:
                return False
    return not snap["invalid_empty_press"]


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
        "criteria_ok": False,
        "error": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(env.check_success())
        snap = _snapshot(env)
        criteria_ok = _present_all_punched(snap)
        ok = bool(plan_ok and check_ok and criteria_ok)
        row.update(
            {
                "success": ok,
                "plan_success": plan_ok,
                "check_success": check_ok,
                "criteria_ok": criteria_ok,
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
    print(f"\n=== {condition}: {n} tests | overrides={overrides} ===")
    for i in range(n):
        seed = i
        label = f"{condition}/ep{i}"
        row = _run_episode(overrides, seed=seed, label=label)
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"missing={row.get('missing_tile_mode')} cont={row.get('belt_continous_motion')} "
            f"miss_side={row.get('missing_tile_side')} miss_idx={row.get('missing_tile_index')} "
            f"L_missed={row.get('left_missed')} R_missed={row.get('right_missed')} "
            f"empty_press={row.get('invalid_empty_press')} err={row.get('error')}"
        )
        time.sleep(0.1)
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {condition}: {n_ok}/{n} successes ===")
    return results


def record_and_export_demos():
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    exported = {}
    for condition, file_tag in DEMO_FILE_TAGS.items():
        print(f"\n=== recording demo: {condition} (tag={file_tag}) ===")
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
                print(f"  copied -> {dst}")
    with open(os.path.join(out_dir, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : missing_tile_mode=false, belt_continous_motion=false\n"
            "           discrete stop-under-stamp; both arms punch together\n"
            "opt1     : missing_tile_mode=true,  belt_continous_motion=false\n"
            "           one random missing tile → alternate 1-arm vs 2-arm stops\n"
            "opt2     : missing_tile_mode=false, belt_continous_motion=true\n"
            "           continuous belt motion; press while tile overlaps stamp\n"
            "opt1+2   : missing_tile_mode=true,  belt_continous_motion=true\n"
            "           missing tile + continuous belts\n\n"
            "Success: every present tile is punched (missing slots skipped);\n"
            "         no empty-slot button press.\n"
            "Tile spacing: always variable.\n"
            "Discrete pause under stamp: tile_pause_s=2.0\n\n"
            "Files: <tag>_sidebyside.mp4, <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    all_results = {}
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        all_results[cond] = run_condition(cond, n=N_PER_CONDITION)

    summary = {}
    print("\n========== SUMMARY ==========")
    all_ok = True
    for key, rows in all_results.items():
        n_ok = sum(1 for r in rows if r["success"])
        n = len(rows)
        summary[key] = {"successes": n_ok, "attempts": n}
        print(f"  {key}: {n_ok}/{n}")
        if n_ok < N_PER_CONDITION:
            all_ok = False

    exported = record_and_export_demos()

    report = {
        "summary": summary,
        "results": all_results,
        "demos": exported,
        "success_criterion": (
            "every present tile punched (has offset, not missed); "
            "missing slots skipped; no invalid empty press"
        ),
    }
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")
    print(f"Demos in {out_dir}")
    if not all_ok:
        print("\n!! SOME CONDITIONS DID NOT REACH 5/5 SUCCESSES")
        sys.exit(1)
    print("\nAll conditions 5/5 successes.")


if __name__ == "__main__":
    main()
