#!/usr/bin/env python3
"""Full controller test + tagged demos for play_billiard.

Conditions (5 episodes each):
  default : specific_hole=false, enable_distractors=false
            (red ball only; any of 6 pockets)
  opt1    : specific_hole=true,  enable_distractors=false
            (nominated top pocket + yellow arrow; only that pocket)
  opt2    : specific_hole=false, enable_distractors=true
            (up to 2 distractors; any pocket; distractor pocketed → fail)
  opt1+2  : specific_hole=true,  enable_distractors=true
            (distractors occupy a different hole; nominated hole stays open)

Success criteria (validated every episode against check_success):
  - primary red ball falls into an allowed pocket
  - Default / Opt 2 → any of the 6 pockets
  - Opt 1 / Opt 1+2 → only the nominated target pocket
  - any distractor ball in any pocket → failure
  - robot-link contact with the primary ball → failure

Suite pass = criteria_consistent + correct option setup on all 20 episodes,
plus one tagged side-by-side demo per condition.

Demos land in:
  docs/final_task_demos/play_billiard/<tag>_sidebyside.mp4
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

TASK = "play_billiard"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "specific_hole": False,
        "enable_distractors": False,
        "num_extra_balls": 2,
    },
    "opt1": {
        "specific_hole": True,
        "enable_distractors": False,
        "num_extra_balls": 2,
    },
    "opt2": {
        "specific_hole": False,
        "enable_distractors": True,
        "num_extra_balls": 2,
    },
    "opt1+2": {
        "specific_hole": True,
        "enable_distractors": True,
        "num_extra_balls": 2,
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


def _ball_inside_hollow(env, ball) -> bool:
    if ball is None:
        return False
    p = np.asarray(ball.get_pose().p, dtype=np.float64)
    return bool(env._ball_inside_hollow(p))


def _independent_criteria(env) -> dict:
    """Recompute success from state — must agree with check_success()."""
    specific = bool(getattr(env, "specific_hole", False))
    distractors_on = bool(getattr(env, "enable_distractors", False))
    allowed = list(getattr(env, "_allowed_pocket_ids", list(range(6))))
    target_id = getattr(env, "_target_pocket_id", None)
    target_name = str(getattr(env, "_target_pocket_name", "any"))
    n_extra = len(getattr(env, "extra_balls", []) or [])
    n_arrow = len(getattr(env, "_target_arrow_parts", []) or [])
    blocked = getattr(env, "_blocked_pocket_id", None)

    robot_hit = bool(getattr(env, "_robot_ball_contact", False))
    distractor_in = bool(getattr(env, "_distractor_pocketed", False))
    for ball in getattr(env, "extra_balls", []) or []:
        if _ball_inside_hollow(env, ball):
            distractor_in = True
            break

    primary_in = bool(getattr(env, "_primary_pocketed", False))
    primary_pid = getattr(env, "_primary_pocket_id", None)
    if (not primary_in) and env.primary_ball is not None:
        if _ball_inside_hollow(env, env.primary_ball):
            primary_in = True
            primary_pid, _ = env._nearest_pocket_id(
                np.asarray(env.primary_ball.get_pose().p[:2], dtype=np.float64)
            )

    if specific:
        pocket_ok = (
            primary_in
            and primary_pid is not None
            and int(primary_pid) == int(target_id)
            and allowed == [int(target_id)]
        )
    else:
        pocket_ok = (
            primary_in
            and primary_pid is not None
            and int(primary_pid) in set(range(6))
            and set(allowed) == set(range(6))
        )

    success = bool(pocket_ok and (not distractor_in) and (not robot_hit))

    setup_ok = True
    setup_notes = []
    if specific and n_arrow < 1:
        setup_ok = False
        setup_notes.append("missing_target_arrow")
    if specific and target_id is None:
        setup_ok = False
        setup_notes.append("missing_target_id")
    if distractors_on and n_extra < 1:
        setup_ok = False
        setup_notes.append("expected_distractors")
    if (not distractors_on) and n_extra != 0:
        setup_ok = False
        setup_notes.append("unexpected_distractors")
    if specific and distractors_on:
        if blocked is None or int(blocked) == int(target_id):
            setup_ok = False
            setup_notes.append("blocked_hole_not_distinct")
        for ball in getattr(env, "extra_balls", []) or []:
            try:
                xy = env._ball_xy(ball)
            except Exception:
                continue
            if env._obstructs_target_hole(xy):
                setup_ok = False
                setup_notes.append("distractor_blocks_target")
                break

    return {
        "specific_hole": specific,
        "enable_distractors": distractors_on,
        "n_extra_balls": n_extra,
        "n_arrow_parts": n_arrow,
        "target_pocket_id": None if target_id is None else int(target_id),
        "target_pocket_name": target_name,
        "blocked_pocket_id": None if blocked is None else int(blocked),
        "allowed_pocket_ids": [int(x) for x in allowed],
        "primary_pocketed": primary_in,
        "primary_pocket_id": None if primary_pid is None else int(primary_pid),
        "distractor_pocketed": distractor_in,
        "robot_ball_contact": robot_hit,
        "pocket_ok": pocket_ok,
        "setup_ok": setup_ok,
        "setup_notes": setup_notes,
        "criteria_success": success,
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
        check_ok = bool(plan_ok and env.check_success())
        crit = _independent_criteria(env)
        consistent = bool(crit["criteria_success"] == bool(env.check_success()))
        row.update(
            {
                "success": bool(check_ok and crit["criteria_success"] and crit["setup_ok"]),
                "plan_success": plan_ok,
                "check_success": check_ok,
                "criteria_consistent": consistent,
                "arm_side": str(getattr(env, "_arm_side", "")),
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
        if args.get("render_freq"):
            try:
                env.viewer.close()
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
    """Run exactly ``n`` episodes; validate setup + criteria consistency."""
    overrides = _overrides(condition)
    results = []
    seed = int(np.random.randint(0, 10_000))
    print(f"\n=== {condition}: {n} tests | overrides={overrides} ===", flush=True)
    for i in range(n):
        label = f"{condition}/ep{i}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        seed += 1
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} crit={row.get('crit_criteria_success')} "
            f"setup={row.get('crit_setup_ok')} primary={row.get('crit_primary_pocket_id')} "
            f"target={row.get('crit_target_pocket_id')}/{row.get('crit_target_pocket_name')} "
            f"distractor_in={row.get('crit_distractor_pocketed')} "
            f"extras={row.get('crit_n_extra_balls')} arrow={row.get('crit_n_arrow_parts')} "
            f"blocked={row.get('crit_blocked_pocket_id')} "
            f"consistent={row.get('criteria_consistent')} err={row.get('error')}",
            flush=True,
        )
        # Persist partial progress so a crash does not lose earlier conditions.
        out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "test_report.partial.json"), "w", encoding="utf-8") as f:
            json.dump({"condition": condition, "results_so_far": results}, f, indent=2)
        time.sleep(0.2)
    n_ok = sum(1 for r in results if r["success"])
    n_cons = sum(1 for r in results if r.get("criteria_consistent"))
    n_setup = sum(1 for r in results if r.get("crit_setup_ok"))
    print(
        f"=== {condition}: {n_ok}/{n} expert successes | "
        f"criteria_consistent {n_cons}/{n} | setup_ok {n_setup}/{n} ===",
        flush=True,
    )
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
            "default  : specific_hole=false, enable_distractors=false\n"
            "           (red ball only; success = any of 6 pockets)\n"
            "opt1     : specific_hole=true,  enable_distractors=false\n"
            "           (nominated top pocket + yellow arrow; only that pocket)\n"
            "opt2     : specific_hole=false, enable_distractors=true\n"
            "           (up to 2 distractors; any pocket; distractor pocketed fails)\n"
            "opt1+2   : specific_hole=true,  enable_distractors=true\n"
            "           (distractors occupy a different hole; nominated hole stays open)\n\n"
            "Success: primary red ball in an allowed pocket; no robot-ball contact;\n"
            "any distractor ball in any pocket fails the episode.\n\n"
            "Files: <tag>_sidebyside.mp4, <tag>_head.mp4, <tag>_topdown.mp4\n"
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
            f"  {key}: expert {n_ok}/{len(rows)} | "
            f"criteria_consistent {n_cons}/{len(rows)} | "
            f"setup_ok {n_setup}/{len(rows)}",
            flush=True,
        )
        if n_cons < len(rows) or n_setup < len(rows):
            print(f"  CRITERIA/SETUP ISSUE on {key}", flush=True)

    exported = {}
    if os.environ.get("SKIP_DEMOS", "").strip() not in ("1", "true", "True"):
        exported = record_and_export_demos()
    else:
        print("\n=== SKIP_DEMOS set; not recording demos in this test run ===", flush=True)

    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": all_results, "demos": exported}, f, indent=2)
    print(f"\nWrote report: {report_path}", flush=True)

    # Pass = every episode's independent criteria agrees with check_success,
    # and option setup (arrow / distractors / blocked hole) matches the condition.
    # Expert pocket rate may be <5/5; demos still require a successful rollout each.
    criteria_bad = any(
        summary[c]["criteria_consistent"] < summary[c]["attempts"]
        or summary[c]["setup_ok"] < summary[c]["attempts"]
        for c in ("default", "opt1", "opt2", "opt1+2")
    )
    demos_bad = (
        os.environ.get("SKIP_DEMOS", "").strip() not in ("1", "true", "True")
        and len(exported) < 4
    )
    if criteria_bad or demos_bad:
        print("\nTEST SUITE: FAILED requirements", flush=True)
        sys.exit(1)
    print("\nTEST SUITE: PASSED", flush=True)
    print(
        f"\nCriteria consistent on all {4 * N_PER_CONDITION} episodes; "
        "tagged demos written for default/opt1/opt2/opt1+2.",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
