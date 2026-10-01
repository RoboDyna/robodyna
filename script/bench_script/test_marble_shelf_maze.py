#!/usr/bin/env python3
"""Full controller test + tagged demos for marble_shelf_maze.

Conditions (5 episodes each):
  default : continuous_ball_motion=false, oscillating_bowl_enabled=false
            (marble snaps to shelf centre after each drop; static bowl)
  opt1    : continuous_ball_motion=true,  oscillating_bowl_enabled=false
            (marble keeps real dynamics after landing)
  opt2    : continuous_ball_motion=false, oscillating_bowl_enabled=true
            (bowl oscillates L↔R under the maze; last drop must be timed)
  opt1+2  : continuous_ball_motion=true,  oscillating_bowl_enabled=true

Success: the marble ends up in the bowl
  (independent XY/Z check vs live bowl pose; must agree with check_success()).

Demos land in:
  docs/final_task_demos/marble_shelf_maze/<tag>_sidebyside.mp4
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

TASK = "marble_shelf_maze"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "continuous_ball_motion": False,
        "oscillating_bowl_enabled": False,
    },
    "opt1": {
        "continuous_ball_motion": True,
        "oscillating_bowl_enabled": False,
    },
    "opt2": {
        "continuous_ball_motion": False,
        "oscillating_bowl_enabled": True,
    },
    "opt1+2": {
        "continuous_ball_motion": True,
        "oscillating_bowl_enabled": True,
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
    marble_p = (
        np.asarray(env.ball.get_pose().p, dtype=np.float64)
        if env.ball is not None
        else np.zeros(3)
    )
    bowl_xy = (
        np.asarray(env._bowl_xy(), dtype=np.float64)
        if hasattr(env, "_bowl_xy")
        else (
            np.asarray(env.bowl.get_pose().p[:2], dtype=np.float64)
            if env.bowl is not None
            else np.zeros(2)
        )
    )
    catch_r = float(getattr(env, "bowl_catch_radius", 0.032))
    ball_r = float(getattr(env, "ball_radius", 0.012))
    table_z = float(getattr(env, "table_z", 0.74))
    bowl_h = float(getattr(env, "bowl_height", 0.045))
    xy_err = float(np.linalg.norm(marble_p[:2] - bowl_xy))
    in_z = bool((table_z - 0.01) <= float(marble_p[2]) <= (table_z + bowl_h))
    # Mirror envs/marble_shelf_maze.py::_ball_in_bowl
    marble_in_bowl = bool(xy_err <= (catch_r + ball_r) and in_z)

    return {
        "continuous_ball_motion": bool(getattr(env, "continuous_ball_motion", False)),
        "oscillating_bowl_enabled": bool(getattr(env, "osc_bowl_enabled", False)),
        "option_label": str(getattr(env, "_option_label", lambda: "")()),
        "n_shelves": int(getattr(env, "n_shelves", 0)),
        "bowl_side": str(getattr(env, "bowl_side", "")),
        "active_shelf_idx": int(getattr(env, "active_shelf_idx", -1)),
        "ball_mode": str(getattr(env, "_ball_mode", "")),
        "bowl_xy": [float(x) for x in bowl_xy],
        "marble_pose": [float(x) for x in marble_p],
        "marble_bowl_xy_err": xy_err,
        "marble_in_z_band": in_z,
        "bowl_catch_radius": catch_r,
        "marble_in_bowl": marble_in_bowl,
    }


def _criteria_ok(snap: dict) -> bool:
    """Independent success: marble ends up in the bowl."""
    return bool(snap["marble_in_bowl"])


def _condition_shape_ok(condition: str, snap: dict) -> bool:
    cfg = CONDITIONS[condition]
    if bool(snap["continuous_ball_motion"]) != bool(cfg["continuous_ball_motion"]):
        return False
    if bool(snap["oscillating_bowl_enabled"]) != bool(cfg["oscillating_bowl_enabled"]):
        return False
    return True


def _cuda_cleanup():
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _run_episode(
    task_args_overrides: list[str],
    seed: int,
    label: str,
    condition: str,
    max_oom_retries: int = 4,
) -> dict:
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
    for attempt in range(max_oom_retries + 1):
        _cuda_cleanup()
        env = class_decorator(TASK)
        try:
            env.setup_demo(now_ep_num=0, seed=seed, **args)
            env.play_once()
            plan_ok = bool(env.plan_success)
            check_ok = bool(env.check_success())
            snap = _snapshot(env)
            criteria_ok = _criteria_ok(snap)
            shape_ok = _condition_shape_ok(condition, snap)
            consistent = bool(check_ok == criteria_ok)
            ok = bool(plan_ok and check_ok and criteria_ok and shape_ok and consistent)
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
            row["error"] = None
            return row
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            is_oom = "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
            if is_oom and attempt < max_oom_retries:
                print(
                    f"  [OOM retry {attempt + 1}/{max_oom_retries}] seed={seed} — waiting…",
                    flush=True,
                )
                traceback.print_exc()
                time.sleep(8.0 * (attempt + 1))
                continue
            traceback.print_exc()
            return row
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
            _cuda_cleanup()
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION, max_attempts: int = 20):
    """Run until ``n`` successes (or ``max_attempts`` tries). Retries expert/OOM fails."""
    overrides = _overrides(condition)
    results = []
    successes = []
    base = 1000 * (list(CONDITIONS.keys()).index(condition) + 1)
    print(
        f"\n=== {condition}: need {n} successes (max {max_attempts}) | overrides={overrides} ===",
        flush=True,
    )
    attempt = 0
    while len(successes) < n and attempt < max_attempts:
        seed = base + attempt
        label = f"{condition}/ep{attempt}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"shape={row.get('shape_ok')} consistent={row.get('criteria_consistent')} "
            f"cont={row.get('continuous_ball_motion')} osc={row.get('oscillating_bowl_enabled')} "
            f"in_bowl={row.get('marble_in_bowl')} xy_err={row.get('marble_bowl_xy_err')} "
            f"mode={row.get('ball_mode')} err={row.get('error')}",
            flush=True,
        )
        if row["success"]:
            successes.append(row)
        attempt += 1
        time.sleep(0.05)
    n_ok = len(successes)
    print(f"=== {condition}: {n_ok}/{n} successes ({attempt} attempts) ===", flush=True)
    return results, successes


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
            "default  : continuous_ball_motion=false, oscillating_bowl_enabled=false\n"
            "           marble snaps to shelf centre after each drop; static bowl\n"
            "opt1     : continuous_ball_motion=true,  oscillating_bowl_enabled=false\n"
            "           marble keeps real dynamics after landing (no centre snap)\n"
            "opt2     : continuous_ball_motion=false, oscillating_bowl_enabled=true\n"
            "           bowl oscillates L↔R under the maze; last drop must be timed\n"
            "opt1+2   : continuous_ball_motion=true,  oscillating_bowl_enabled=true\n"
            "           Opt 1 + Opt 2 combined\n\n"
            "Success: the marble drops into the bowl\n"
            "         (XY within bowl_catch_radius+ball_radius of bowl center;\n"
            "          Z in [table_z-0.01, table_z+bowl_height]).\n\n"
            "Files (side-by-side is the primary deliverable):\n"
            "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
            "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
            "Also: <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    all_results = {}
    all_successes = {}
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        rows, oks = run_condition(cond, n=N_PER_CONDITION)
        all_results[cond] = rows
        all_successes[cond] = oks

    summary = {}
    print("\n========== SUMMARY ==========", flush=True)
    all_ok = True
    for key, rows in all_results.items():
        n_ok = len(all_successes[key])
        n = len(rows)
        n_consistent = sum(1 for r in rows if r.get("criteria_consistent"))
        n_in_bowl = sum(1 for r in rows if r.get("marble_in_bowl"))
        summary[key] = {
            "successes": n_ok,
            "attempts": n,
            "rate": n_ok / max(n, 1),
            "criteria_consistent": n_consistent,
            "marble_in_bowl_count": n_in_bowl,
            "success_seeds": [r["seed"] for r in all_successes[key]],
            "config": dict(CONDITIONS[key]),
        }
        print(
            f"  {key}: {n_ok}/{N_PER_CONDITION} successes "
            f"({n} attempts, criteria_consistent={n_consistent}/{n}, "
            f"marble_in_bowl={n_in_bowl}/{n})",
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
            "marble ends up in the bowl "
            "(XY within bowl_catch_radius+ball_radius of bowl center; "
            "Z in [table_z-0.01, table_z+bowl_height]; "
            "must agree with check_success())"
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
        print("\nFAIL: not all conditions reached full success count", flush=True)
        sys.exit(1)
    print("\nPASS: all conditions reached target successes with marble-in-bowl criteria", flush=True)


if __name__ == "__main__":
    main()
