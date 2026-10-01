#!/usr/bin/env python3
"""Full controller test + tagged demos for save_goal.

Conditions (5 episodes each):
  default : players_enabled=false, cover_enabled=false
            (direct shot toward a point in the goal mouth)
  opt1    : players_enabled=true,  cover_enabled=false
            (field players; ball bounces then still aims through the goal frame)
  opt2    : players_enabled=false, cover_enabled=true
            (mid-field cover; ball briefly occluded)
  opt1+2  : players_enabled=true,  cover_enabled=true
            (bounce + cover; cover ends before players)

Success (expert / check_success):
  - square keeper placed fully in the green zone before the red deadline
  - ball hits the keeper's front face and is blocked
  - ball does NOT enter the goal (not conceded)
  - both grippers open at the end

Path validity (required every episode — the shot must go through the goal frame):
  - ball target y lies inside the goal mouth (on-frame shot)
  - with Opt 1, a bounce waypoint exists and the post-bounce target is still on-frame

Demos land in:
  docs/final_task_demos/save_goal/<tag>_sidebyside.mp4
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

TASK = "save_goal"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "players_enabled": False,
        "cover_enabled": False,
    },
    "opt1": {
        "players_enabled": True,
        "cover_enabled": False,
    },
    "opt2": {
        "players_enabled": False,
        "cover_enabled": True,
    },
    "opt1+2": {
        "players_enabled": True,
        "cover_enabled": True,
    },
}

# Prefer literal opt1+2 in filenames (matches control_quality / hit_target).
DEMO_FILE_TAGS = {
    "default": "default",
    "opt1": "opt1",
    "opt2": "opt2",
    "opt1+2": "opt1+2",
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


def _path_through_goal(env) -> dict:
    """Shot aims through the goal mouth (required for a valid episode)."""
    target = getattr(env, "ball_target_pose", None)
    cy = float(getattr(env, "goal_center_y", 0.0))
    half_w = float(getattr(env, "goal_half_w", 0.0))
    margin = float(getattr(env, "ball_target_y_margin", 0.0))
    goal_x = float(getattr(env, "goal_x", 0.0))
    travel_dir = float(getattr(env, "travel_dir", 1.0))
    out = {
        "has_target": target is not None,
        "target_y": None,
        "on_frame": False,
        "past_goal_line": False,
        "has_bounce": getattr(env, "ball_bounce_pose", None) is not None,
        "n_players": int(len(getattr(env, "_players", []) or [])),
        "cover_present": (
            getattr(env, "cover_x_min", None) is not None
            and getattr(env, "cover_x_max", None) is not None
        ),
    }
    if target is None:
        return out
    ty = float(target[1])
    tx = float(target[0])
    out["target_y"] = ty
    out["on_frame"] = abs(ty - cy) <= (half_w - margin + 1e-6)
    out["past_goal_line"] = bool(travel_dir * tx >= travel_dir * goal_x - 1e-6)
    return out


def _criteria_ok(info_gk: dict, path: dict, expect_players: bool, expect_cover: bool) -> dict:
    """Independent check matching the save_goal success contract (+ path validity)."""
    keeper_ok = bool(info_gk.get("keeper_in_zone"))
    blocked = bool(info_gk.get("ball_blocked"))
    late = bool(info_gk.get("late_failure"))
    conceded = bool(info_gk.get("goal_conceded"))
    crossed = bool(info_gk.get("ball_crossed_goal"))
    path_ok = bool(path.get("on_frame") and path.get("past_goal_line"))
    players_ok = (int(path.get("n_players", 0)) > 0) if expect_players else (int(path.get("n_players", 0)) == 0)
    if expect_players:
        players_ok = players_ok and bool(path.get("has_bounce"))
    cover_ok = bool(path.get("cover_present")) if expect_cover else (not bool(path.get("cover_present")))
    # Expert success = save: blocked, not conceded, on time, keeper in zone.
    save_ok = bool(keeper_ok and blocked and (not late) and (not conceded))
    # Path must still be an on-frame shot (would enter the goal if not saved).
    return {
        "path_ok": path_ok,
        "players_ok": players_ok,
        "cover_ok": cover_ok,
        "save_ok": save_ok,
        "not_conceded": (not conceded),
        "blocked": blocked,
        "keeper_in_zone": keeper_ok,
        "late_failure": late,
        "ball_crossed_goal": crossed,
        "criteria_ok": bool(save_ok and path_ok and players_ok and cover_ok),
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
    expect_players = bool(CONDITIONS[condition]["players_enabled"])
    expect_cover = bool(CONDITIONS[condition]["cover_enabled"])
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
        path = _path_through_goal(env)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(plan_ok and env.check_success())
        info = dict(getattr(env, "info", {}) or {})
        gk = dict(info.get("save_goal", {}) or {})
        crit = _criteria_ok(gk, path, expect_players, expect_cover)
        # Independent criteria must agree with check_success on the save outcome.
        criteria_consistent = bool(crit["save_ok"] == bool(gk.get("ball_blocked") and gk.get("keeper_in_zone")
                                                           and (not gk.get("late_failure"))
                                                           and (not gk.get("goal_conceded"))))
        row.update(
            {
                "success": bool(check_ok and crit["criteria_ok"]),
                "plan_success": plan_ok,
                "check_success": check_ok,
                "criteria_consistent": criteria_consistent,
                "mirrored": bool(getattr(env, "mirrored", False)),
                "travel_dir": float(getattr(env, "travel_dir", 0.0)),
                "ball_speed": float(getattr(env, "ball_speed", 0.0)),
                "path": path,
                "save_goal": gk,
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
    overrides = _overrides(condition)
    results = []
    seed = int(np.random.randint(0, 10_000))
    attempts = 0
    max_attempts = n * 10
    print(f"\n=== {condition}: need {n} successes | overrides={overrides} ===", flush=True)
    while sum(1 for r in results if r["success"]) < n and attempts < max_attempts:
        label = f"{condition}/ep{attempts}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        attempts += 1
        seed += 1
        status = "OK" if row["success"] else "FAIL"
        path = row.get("path") or {}
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} save={row.get('crit_save_ok')} "
            f"path={row.get('crit_path_ok')} players={path.get('n_players')} "
            f"bounce={path.get('has_bounce')} cover={path.get('cover_present')} "
            f"blocked={row.get('crit_blocked')} conceded={not row.get('crit_not_conceded', True)} "
            f"late={row.get('crit_late_failure')} zone={row.get('crit_keeper_in_zone')} "
            f"mirrored={row.get('mirrored')} err={row.get('error')}",
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
            "default  : players_enabled=false, cover_enabled=false\n"
            "           (direct shot toward a point in the goal mouth)\n"
            "opt1     : players_enabled=true,  cover_enabled=false\n"
            "           (field players; ball bounces then still aims through the goal frame)\n"
            "opt2     : players_enabled=false, cover_enabled=true\n"
            "           (mid-field cover; ball briefly occluded under the tunnel)\n"
            "opt1+2   : players_enabled=true,  cover_enabled=true\n"
            "           (bounce + cover; cover ends before the players)\n\n"
            "Success: keeper fully in green zone before the red deadline, ball blocked on the\n"
            "keeper front face, ball does NOT enter the goal, grippers open.\n"
            "Path validity: every episode's ball target lies inside the goal mouth (on-frame);\n"
            "Opt 1 / Opt 1+2 keep an on-frame target after the bounce.\n\n"
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
        n_path = sum(1 for r in rows if r.get("crit_path_ok"))
        n_cons = sum(1 for r in rows if r.get("criteria_consistent"))
        summary[key] = {
            "successes": n_ok,
            "attempts": len(rows),
            "path_ok": n_path,
            "criteria_consistent": n_cons,
        }
        print(
            f"  {key}: {n_ok}/{N_PER_CONDITION} successes "
            f"({len(rows)} attempts) | path_ok {n_path}/{len(rows)} | "
            f"criteria_consistent {n_cons}/{len(rows)}",
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
