#!/usr/bin/env python3
"""Full controller test + tagged demos for catch_ramp_ball.

Conditions (5 successful episodes each):
  default : wall_bounce_enabled=false, enable_distractor=false
            drop at ramp top (meets back wall); incline roll into cup
  opt1    : wall_bounce_enabled=true,  enable_distractor=false
            lateral heading → side-rail rebound on the way down
  opt2    : wall_bounce_enabled=false, enable_distractor=true
            blue distractor on a separate lane (catching it fails)
  opt1+2  : wall_bounce_enabled=true,  enable_distractor=true
            wall rebound + blue distractor

Success (identical spirit to catch_valley_ball):
  - red (target) ball rests in the cup, AND
  - blue distractor is NOT in the cup (when Opt 2 is on)

Demos land in:
  docs/final_task_demos/catch_ramp_ball/<tag>_sidebyside.mp4
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

TASK = "catch_ramp_ball"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "wall_bounce_enabled": False,
        "enable_distractor": False,
    },
    "opt1": {
        "wall_bounce_enabled": True,
        "enable_distractor": False,
    },
    "opt2": {
        "wall_bounce_enabled": False,
        "enable_distractor": True,
    },
    "opt1+2": {
        "wall_bounce_enabled": True,
        "enable_distractor": True,
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


def _ball_in_cup_geom(env, ball_actor) -> bool:
    """Independent geometry check matching task catch tolerance."""
    if ball_actor is None:
        return False
    bp = np.asarray(ball_actor.get_pose().p)
    cp = np.asarray(env.cup.get_pose().p)
    offset = float(np.linalg.norm(bp[:2] - cp[:2]))
    rim = float(getattr(env, "rim_radius", 0.04))
    table_top = float(getattr(env, "table_top", 0.74))
    return bool(
        offset < rim
        and (table_top - 0.01) < bp[2] < (table_top + 0.12)
    )


def _criteria_ok(env, condition: str) -> dict:
    """Independent success check: target in cup, distractor not in cup."""
    expect_wall = bool(CONDITIONS[condition]["wall_bounce_enabled"])
    expect_dist = bool(CONDITIONS[condition]["enable_distractor"])
    wall = bool(getattr(env, "wall_bounce_enabled", False))
    dist = bool(getattr(env, "enable_distractor", False))
    setup_ok = wall == expect_wall and dist == expect_dist
    if expect_dist:
        setup_ok = setup_ok and (getattr(env, "distractor", None) is not None)
    else:
        setup_ok = setup_ok and (getattr(env, "distractor", None) is None)

    red_in = _ball_in_cup_geom(env, getattr(env, "ball", None))
    blue_in = bool(
        expect_dist and _ball_in_cup_geom(env, getattr(env, "distractor", None))
    )
    # Same rule as catch_valley_ball: distractor in vessel → failure.
    criteria_success = bool(red_in and (not blue_in))
    return {
        "red_in_cup": red_in,
        "distractor_in_cup": blue_in,
        "criteria_success": criteria_success,
        "setup_ok": setup_ok,
        "wall_bounce_enabled": wall,
        "enable_distractor": dist,
        "wall_bounces": int(getattr(env, "drop_wall_bounces", 0)),
        "ball_ball_bounces": int(getattr(env, "ball_ball_bounces", 0)),
        "ball_speed": float(getattr(env, "ball_speed", 0.0)),
        "distractor_speed": float(getattr(env, "distractor_speed", 0.0)),
        "exit_separation": float(getattr(env, "exit_separation", 0.0)),
        "phase": str(getattr(env, "_ball_phase", "")),
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
        criteria_consistent = bool(crit["criteria_success"] == check_ok)
        row.update(
            {
                "success": bool(
                    plan_ok and check_ok and crit["criteria_success"] and crit["setup_ok"]
                ),
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
            f"setup={row.get('crit_setup_ok')} red={row.get('crit_red_in_cup')} "
            f"blue={row.get('crit_distractor_in_cup')} "
            f"wb={row.get('crit_wall_bounces')} bb={row.get('crit_ball_ball_bounces')} "
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
            "default  : wall_bounce_enabled=false, enable_distractor=false\n"
            "           red ball rolls straight down the ramp into the cup\n"
            "opt1     : wall_bounce_enabled=true,  enable_distractor=false\n"
            "           red ball rebounds off a side rail mid-run\n"
            "opt2     : wall_bounce_enabled=false, enable_distractor=true\n"
            "           blue distractor on a separate lane; catching blue fails\n"
            "opt1+2   : wall_bounce_enabled=true,  enable_distractor=true\n"
            "           wall rebound + blue distractor\n\n"
            "Success (same rule as catch_valley_ball):\n"
            "  - red (target) ball in the cup, AND\n"
            "  - distractor NOT in the cup.\n\n"
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
        n_blue_fail = sum(1 for r in rows if r.get("crit_distractor_in_cup"))
        summary[key] = {
            "successes": n_ok,
            "attempts": len(rows),
            "criteria_consistent": n_cons,
            "setup_ok": n_setup,
            "distractor_in_cup_count": n_blue_fail,
        }
        print(
            f"  {key}: {n_ok}/{N_PER_CONDITION} successes "
            f"({len(rows)} attempts) | criteria_consistent {n_cons}/{len(rows)} | "
            f"setup_ok {n_setup}/{len(rows)} | distractor_in_cup {n_blue_fail}",
            flush=True,
        )
        if n_ok < N_PER_CONDITION:
            print(f"  SHORTFALL {key}: only {n_ok}/{N_PER_CONDITION}", flush=True)

    with open(os.path.join(out_dir, "TEST_RESULTS.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": all_results}, f, indent=2, default=str)
    print(f"wrote {out_dir}/TEST_RESULTS.json", flush=True)

    print("\n========== DEMOS ==========", flush=True)
    exported = record_and_export_demos()
    print("DEMO_PATHS", exported, flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
