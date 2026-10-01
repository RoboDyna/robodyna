#!/usr/bin/env python3
"""Full controller test + tagged demos for load_train.

Conditions (5 episodes each):
  default : target_wagon_mode=false, tunnel_enabled=false
            (3 red wagons; drop into ANY open wagon)
  opt1    : target_wagon_mode=true,  tunnel_enabled=false
            (one random red target wagon; others gray; success only in red)
  opt2    : target_wagon_mode=false, tunnel_enabled=true
            (matching far+near arched tunnels; drop into ANY open wagon)
  opt1+2  : target_wagon_mode=true,  tunnel_enabled=true
            (red target wagon + both tunnels; success only in the red wagon)

Success:
  - Default / Opt 2: ball seated in any open cargo wagon
  - Opt 1 / Opt 1+2: ball seated in the nominated (red) target wagon only

Demos land in:
  docs/final_task_demos/load_train/<tag>_sidebyside.mp4
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

TASK = "load_train"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

CONDITIONS = {
    "default": {
        "target_wagon_mode": False,
        "tunnel_enabled": False,
    },
    "opt1": {
        "target_wagon_mode": True,
        "tunnel_enabled": False,
    },
    "opt2": {
        "target_wagon_mode": False,
        "tunnel_enabled": True,
    },
    "opt1+2": {
        "target_wagon_mode": True,
        "tunnel_enabled": True,
    },
}

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
        if isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def _independent_success(env) -> dict:
    """Recompute success from geometry — must match check_success()."""
    target_mode = bool(getattr(env, "target_wagon_mode", False))
    target_idx = getattr(env, "target_wagon_idx", None)
    latched = bool(getattr(env, "_ball_latched", False))
    latched_idx = getattr(env, "_latched_car_idx", None)
    cargo = list(getattr(env, "_cargo_indices", []) or [])

    in_cars = {}
    for i in cargo:
        in_cars[i] = bool(env._ball_in_car(i))

    if target_mode and target_idx is not None:
        idx = int(target_idx)
        in_target = bool(
            (latched and latched_idx == idx) or in_cars.get(idx, False)
        )
        # Wrong-wagon latch must fail under Opt 1.
        wrong_latch = bool(latched and latched_idx is not None and int(latched_idx) != idx)
        ok = bool(in_target and not wrong_latch)
        allowed = [idx]
    else:
        ok = bool(latched or any(in_cars.values()))
        allowed = list(cargo)

    return {
        "target_wagon_mode": target_mode,
        "target_wagon_idx": None if target_idx is None else int(target_idx),
        "latched": latched,
        "latched_car_idx": None if latched_idx is None else int(latched_idx),
        "in_cars": {str(k): v for k, v in in_cars.items()},
        "allowed_wagons": allowed,
        "indep_success": ok,
        "tunnel_present": bool(getattr(env, "tunnel_present", False)),
        "ball_side": str(getattr(env, "ball_side", "")),
        "n_wagons": int(getattr(env, "n_wagons", 0)),
    }


def _setup_ok(env, condition: str) -> dict:
    """Verify option toggles actually took effect for this episode."""
    expect_target = bool(CONDITIONS[condition]["target_wagon_mode"])
    expect_tunnel = bool(CONDITIONS[condition]["tunnel_enabled"])
    got_target = bool(getattr(env, "target_wagon_mode", False))
    got_tunnel = bool(getattr(env, "tunnel_present", False))
    target_idx = getattr(env, "target_wagon_idx", None)
    n_wagons = int(getattr(env, "n_wagons", 0))
    target_idx_ok = True
    if expect_target:
        target_idx_ok = (
            target_idx is not None
            and int(target_idx) in getattr(env, "_cargo_indices", [])
        )
    else:
        target_idx_ok = target_idx is None
    return {
        "expect_target": expect_target,
        "expect_tunnel": expect_tunnel,
        "got_target": got_target,
        "got_tunnel": got_tunnel,
        "target_idx_ok": target_idx_ok,
        "n_wagons_ok": n_wagons == 3,
        "setup_ok": bool(
            got_target == expect_target
            and got_tunnel == expect_tunnel
            and target_idx_ok
            and n_wagons == 3
        ),
    }


def _gpu_free_mib() -> float:
    try:
        import subprocess

        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0]
        return float(out.split()[0])
    except Exception:
        return 99999.0


def _wait_for_gpu(need_mib: float = 4800.0, poll_s: float = 25.0):
    while True:
        free = _gpu_free_mib()
        if free >= need_mib:
            return free
        print(f"  [gpu] waiting: free={free:.0f} MiB < {need_mib:.0f}", flush=True)
        time.sleep(poll_s)


def _run_episode(task_args_overrides: list[str], seed: int, label: str, condition: str) -> dict:
    _wait_for_gpu(4800.0)
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
        setup = _setup_ok(env, condition)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(plan_ok and env.check_success())
        indep = _independent_success(env)
        check_raw = bool(env.check_success())
        criteria_consistent = bool(check_raw == indep["indep_success"])
        # Under Opt 1, if latched to a non-target wagon, check_success must be False.
        opt1_strict = True
        if setup["expect_target"] and indep["latched"] and indep["target_wagon_idx"] is not None:
            if indep["latched_car_idx"] != indep["target_wagon_idx"]:
                opt1_strict = (not check_raw) and (not indep["indep_success"])
        row.update(
            {
                "success": bool(check_ok and setup["setup_ok"] and criteria_consistent and opt1_strict),
                "plan_success": plan_ok,
                "check_success": check_ok,
                "check_raw": check_raw,
                "criteria_consistent": criteria_consistent,
                "opt1_strict_ok": opt1_strict,
                **{f"setup_{k}": v for k, v in setup.items()},
                **{f"indep_{k}": v for k, v in indep.items()},
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
            f"check={row.get('check_success')} setup={row.get('setup_setup_ok')} "
            f"consistent={row.get('criteria_consistent')} "
            f"target={row.get('indep_target_wagon_idx')} "
            f"latched={row.get('indep_latched_car_idx')} "
            f"tunnel={row.get('indep_tunnel_present')} "
            f"side={row.get('indep_ball_side')} err={row.get('error')}",
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
        _wait_for_gpu(5200.0)
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
            "default  : target_wagon_mode=false, tunnel_enabled=false\n"
            "           (3 red wagons; drop into ANY open wagon)\n"
            "opt1     : target_wagon_mode=true,  tunnel_enabled=false\n"
            "           (one random red target; other wagons gray; success only in red)\n"
            "opt2     : target_wagon_mode=false, tunnel_enabled=true\n"
            "           (far-arc arched tunnel; drop into ANY open wagon)\n"
            "opt1+2   : target_wagon_mode=true,  tunnel_enabled=true\n"
            "           (red target wagon + tunnel; success only in the red wagon)\n\n"
            "Success: ball seated in an allowed wagon.\n"
            "  Default / Opt 2  → any open cargo wagon\n"
            "  Opt 1 / Opt 1+2  → the nominated (red) target wagon only\n"
            "Ball side is random each episode (left/−x or right/+x arm).\n\n"
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
        n_setup = sum(1 for r in rows if r.get("setup_setup_ok"))
        n_cons = sum(1 for r in rows if r.get("criteria_consistent"))
        summary[key] = {
            "successes": n_ok,
            "attempts": len(rows),
            "setup_ok": n_setup,
            "criteria_consistent": n_cons,
        }
        print(
            f"  {key}: {n_ok}/{N_PER_CONDITION} successes "
            f"({len(rows)} attempts) | setup_ok {n_setup}/{len(rows)} | "
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
        json.dump(
            {
                "task": TASK,
                "n_per_condition": N_PER_CONDITION,
                "summary": summary,
                "results": all_results,
                "demos": exported,
                "success_criterion": (
                    "SUCCESS iff the ball is seated in an allowed wagon: "
                    "Default/Opt2 = any open cargo wagon; "
                    "Opt1/Opt1+2 = the nominated (red) target wagon only. "
                    "Option setup (target_wagon_mode / tunnel) must match the condition."
                ),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nWrote report: {report_path}", flush=True)

    if any(summary[c]["successes"] < N_PER_CONDITION for c in ("default", "opt1", "opt2", "opt1+2")):
        print("\nTEST SUITE: FAILED requirements", flush=True)
        sys.exit(1)
    print("\nTEST SUITE: PASSED", flush=True)
    print("\nAll conditions 5/5 successes.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
