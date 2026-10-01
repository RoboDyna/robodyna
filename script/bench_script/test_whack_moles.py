#!/usr/bin/env python3
"""Full controller test + tagged demos for whack_moles.

Conditions (5 episodes each):
  default : distractor_enabled=false, relocating_moles=false
            2 moles with randomized pop speeds from fixed holes; no rabbit
  opt1    : distractor_enabled=true,  relocating_moles=false
            + rabbit distractor (touching it fails)
  opt2    : distractor_enabled=false, relocating_moles=true
            moles reappear from a free hole each time they go down
  opt1+2  : distractor_enabled=true,  relocating_moles=true
            rabbit + relocating moles

Success (expert / check_success):
  - every mole touched from above at least once
  - no rabbit distractor hit (distractor_hit must stay false)
  - no mole exhausted its appearance budget (default 5 rise→fall cycles)

Demos land in:
  docs/final_task_demos/whack_moles/<tag>_sidebyside.mp4
  with tags: default, opt1, opt2, opt1+2
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args, record_demo
from script.collect_data import class_decorator

TASK = "whack_moles"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "distractor_enabled": False,
        "relocating_moles": False,
        "num_moles": 2,
    },
    "opt1": {
        "distractor_enabled": True,
        "relocating_moles": False,
        "num_moles": 2,
        "num_distractors": 1,
    },
    "opt2": {
        "distractor_enabled": False,
        "relocating_moles": True,
        "num_moles": 2,
    },
    "opt1+2": {
        "distractor_enabled": True,
        "relocating_moles": True,
        "num_moles": 2,
        "num_distractors": 1,
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


def _criteria_ok(env, condition: str) -> dict:
    """Independent success check: all moles touched, rabbit not hit."""
    touched = [bool(t) for t in getattr(env, "touched", []) or []]
    n = int(getattr(env, "num_moles", 0))
    distractor_hit = bool(getattr(env, "distractor_hit", False))
    expect_distractor = bool(CONDITIONS[condition]["distractor_enabled"])
    expect_reloc = bool(CONDITIONS[condition]["relocating_moles"])
    all_touched = bool(touched) and len(touched) == n and all(touched)
    setup_ok = (
        bool(getattr(env, "distractor_enabled", False)) == expect_distractor
        and bool(getattr(env, "relocating_moles", False)) == expect_reloc
        and n == int(CONDITIONS[condition]["num_moles"])
    )
    if expect_distractor:
        setup_ok = setup_ok and int(getattr(env, "num_distractors", 0)) >= 1
    else:
        setup_ok = setup_ok and int(getattr(env, "num_distractors", 0)) == 0
    criteria_success = bool(all_touched and (not distractor_hit))
    return {
        "all_touched": all_touched,
        "distractor_hit": distractor_hit,
        "criteria_success": criteria_success,
        "setup_ok": setup_ok,
        "touched": touched,
        "num_moles": n,
        "num_distractors": int(getattr(env, "num_distractors", 0)),
        "distractor_enabled": bool(getattr(env, "distractor_enabled", False)),
        "relocating_moles": bool(getattr(env, "relocating_moles", False)),
        "board_hit": bool(getattr(env, "board_hit", False)),
    }


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
    args["camera"]["collect_head_camera"] = False
    args["camera"]["collect_wrist_camera"] = False
    args["data_type"]["rgb"] = False
    args["data_type"]["third_view"] = False
    args.pop("seed", None)
    args.pop("use_seed", None)

    env = class_decorator(TASK)
    row = {
        "label": label,
        "condition": condition,
        "seed": seed,
        "success": False,
        "plan_success": False,
        "check_success": False,
        "error": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(env.check_success())
        crit = _criteria_ok(env, condition)
        # Independent criteria must agree with check_success.
        criteria_consistent = bool(crit["criteria_success"] == check_ok)
        row.update(
            {
                "success": bool(check_ok and crit["criteria_success"] and crit["setup_ok"]),
                "plan_success": plan_ok,
                "check_success": check_ok,
                "criteria_consistent": criteria_consistent,
                **{f"crit_{k}": v for k, v in crit.items()},
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
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION):
    overrides = _overrides(condition)
    results = []
    seed = int(np.random.randint(0, 10_000))
    attempts = 0
    max_attempts = n * 12
    print(f"\n=== {condition}: need {n} successes | overrides={overrides} ===", flush=True)
    while sum(1 for r in results if r["success"]) < n and attempts < max_attempts:
        label = f"{condition}/ep{attempts}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        attempts += 1
        seed += 1
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} crit={row.get('crit_criteria_success')} "
            f"setup={row.get('crit_setup_ok')} touched={row.get('crit_touched')} "
            f"rabbit_hit={row.get('crit_distractor_hit')} "
            f"consistent={row.get('criteria_consistent')} err={row.get('error')}",
            flush=True,
        )
        time.sleep(0.2)
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {condition}: {n_ok}/{n} successes in {attempts} attempts ===", flush=True)
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
        for key, suffix in (("sidebyside", "sidebyside"), ("head", "head"), ("topdown", "topdown")):
            src = info[key]
            dst = os.path.join(out_dir, f"{file_tag}_{suffix}.mp4")
            shutil.copy2(src, dst)
            if key == "sidebyside":
                exported[condition] = dst
                print(f"  copied -> {dst}", flush=True)
    with open(os.path.join(out_dir, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : distractor_enabled=false, relocating_moles=false\n"
            "           2 moles bob from fixed holes at randomized pop speeds; no rabbit\n"
            "opt1     : distractor_enabled=true,  relocating_moles=false\n"
            "           + compact rabbit distractor (touching it fails)\n"
            "opt2     : distractor_enabled=false, relocating_moles=true\n"
            "           every unhit mole may reappear from a free hole when it goes down\n"
            "opt1+2   : distractor_enabled=true,  relocating_moles=true\n"
            "           rabbit + relocating moles\n\n"
            "Success:\n"
            "  - every mole touched from above at least once (turns green), AND\n"
            "  - no rabbit touched (distractor_hit stays false).\n\n"
            "Files (side-by-side is the primary deliverable):\n"
            "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
            "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
            "Also: <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        all_results[cond] = run_condition(cond, n=N_PER_CONDITION)

    summary = {}
    print("\n========== SUMMARY ==========", flush=True)
    for key, rows in all_results.items():
        n_ok = sum(1 for r in rows if r["success"])
        n_cons = sum(1 for r in rows if r.get("criteria_consistent"))
        n_setup = sum(1 for r in rows if r.get("crit_setup_ok"))
        summary[key] = {
            "successes": n_ok,
            "attempts": len(rows),
            "criteria_consistent": n_cons,
            "setup_ok": n_setup,
        }
        print(
            f"  {key}: {n_ok}/{N_PER_CONDITION} successes "
            f"({len(rows)} attempts) | criteria_consistent {n_cons}/{len(rows)} | "
            f"setup_ok {n_setup}/{len(rows)}",
            flush=True,
        )
        if n_ok < N_PER_CONDITION:
            print(f"  SHORTFALL {key}: only {n_ok}/{N_PER_CONDITION}", flush=True)

    exported = {}
    if os.environ.get("SKIP_DEMOS", "").strip() not in ("1", "true", "True"):
        exported = record_and_export_demos()
    else:
        print("\n=== SKIP_DEMOS set; not recording demos in this test run ===", flush=True)

    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": all_results, "demos": exported}, f, indent=2)
    print(f"\nWrote report: {report_path}", flush=True)

    if any(summary[c]["successes"] < N_PER_CONDITION for c in ("default", "opt1", "opt2", "opt1+2")):
        print("\nTEST SUITE: FAILED requirements", flush=True)
        sys.exit(1)
    print("\nTEST SUITE: PASSED", flush=True)
    print("\nAll conditions 5/5 successes.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
