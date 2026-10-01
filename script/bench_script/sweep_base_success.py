#!/usr/bin/env python3
"""Run expert play_once for every base-suite task; report success counts.

Uses the canonical scenario matrix in ``task_config.scenario_overrides``
(default / opt1 / opt2 / opt1+2). The scripted expert path — not interactive.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "script"), os.path.join(ROOT, "script/bench_script")]

from script.bench_script.record_demo import build_args  # noqa: E402
from script.collect_data import class_decorator  # noqa: E402

from task_config.scenario_overrides import (  # noqa: E402
    BASE_SCENARIOS,
    BASE_TASKS as TASKS,
    SCENARIO_OVERRIDES,
    apply_base_scenario,
)

# Backward-compatible name for older callers.
DEFAULT_OVERRIDES = {task: SCENARIO_OVERRIDES[task]["default"] for task in TASKS}

CONFIG = "demo_dynamic"
SCENARIOS = BASE_SCENARIOS


def run_seed(task_name: str, seed: int, scenario: str = "default") -> dict:
    sweep_root = os.environ.get("ROBODYNA_SWEEP_ROOT", "./tmp")
    safe_scenario = scenario.replace("+", "plus")
    save_root = os.path.abspath(os.path.join(sweep_root, f"tmp_{task_name}_{safe_scenario}_basic_sweep"))
    os.makedirs(save_root, exist_ok=True)
    args = build_args(task_name, CONFIG, save_root, option=None, task_arg_overrides=[])
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False
    args["export_lerobot"] = False
    apply_base_scenario(args, task_name, scenario)

    env = class_decorator(task_name)
    row = {
        "task": task_name,
        "scenario": scenario,
        "seed": seed,
        "plan": False,
        "check": False,
        "ok": False,
        "err": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan = bool(getattr(env, "plan_success", True))
        check = bool(env.check_success())
        row["plan"] = plan
        row["check"] = check
        row["ok"] = bool(plan and check)
    except Exception as exc:  # noqa: BLE001
        row["err"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        try:
            env.close_env()
        except Exception:
            pass
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="seeds per task×scenario (0..n-1)")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument(
        "--scenarios",
        nargs="*",
        default=["default"],
        choices=list(SCENARIOS),
        help="Scenarios to sweep (default: default only)",
    )
    args = p.parse_args()
    seeds = list(range(int(args.start_seed), int(args.start_seed) + int(args.n)))
    scenarios = list(args.scenarios)

    summary = []
    for scenario in scenarios:
        for task in args.tasks:
            print(
                f"\n===== {task} [{scenario}] ({len(seeds)} seeds) =====",
                flush=True,
            )
            ok_seeds = []
            fail_seeds = []
            for seed in seeds:
                row = run_seed(task, seed, scenario=scenario)
                tag = "OK" if row["ok"] else "FAIL"
                if row["err"]:
                    extra = f" err={row['err']}"
                else:
                    extra = f" plan={row['plan']} check={row['check']}"
                print(f"  seed={seed} {tag}{extra}", flush=True)
                if row["ok"]:
                    ok_seeds.append(seed)
                else:
                    fail_seeds.append(seed)
            summary.append(
                (task, scenario, len(ok_seeds), len(seeds), ok_seeds, fail_seeds)
            )

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'task':26s}  {'scen':7s}  success  rate   ok_seeds", flush=True)
    for task, scenario, n_ok, n_tot, ok_seeds, fail_seeds in summary:
        print(
            f"{task:26s}  {scenario:7s}  {n_ok:2d}/{n_tot:<2d}     "
            f"{100.0 * n_ok / n_tot:5.1f}%  {ok_seeds}",
            flush=True,
        )


if __name__ == "__main__":
    main()
