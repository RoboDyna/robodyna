#!/usr/bin/env python3
"""Run expert play_once for every household task; report success counts (no data save)."""
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

TASKS = (
    "trap_bug",
    "boil_milk",
    "fill_coffee_jar",
    "pour_beer",
    "cook_food",
    "cook_food_timer",
    "measure_ingredient",
    "make_soup",
    "catch_cup",
    "catch_mouse_object_drop",
    "stop_ball",
    "clean_table",
)
CONFIG = "demo_dynamic"


def run_seed(task_name: str, seed: int) -> dict:
    sweep_root = os.environ.get("ROBODYNA_SWEEP_ROOT", "./tmp")
    save_root = os.path.abspath(os.path.join(sweep_root, f"tmp_{task_name}_hh_sweep"))
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
    args["use_dynamic"] = False

    env = class_decorator(task_name)
    row = {
        "task": task_name,
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
    p.add_argument("--n", type=int, default=10, help="seeds per task (0..n-1)")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--start-seed", type=int, default=0)
    args = p.parse_args()
    seeds = list(range(int(args.start_seed), int(args.start_seed) + int(args.n)))

    summary = []
    for task in args.tasks:
        print(f"\n===== {task} ({len(seeds)} seeds) =====", flush=True)
        ok_seeds = []
        fail_seeds = []
        for seed in seeds:
            row = run_seed(task, seed)
            tag = "OK" if row["ok"] else "FAIL"
            extra = ""
            if row["err"]:
                extra = f" err={row['err']}"
            else:
                extra = f" plan={row['plan']} check={row['check']}"
            print(f"  seed={seed} {tag}{extra}", flush=True)
            if row["ok"]:
                ok_seeds.append(seed)
            else:
                fail_seeds.append(seed)
        summary.append((task, len(ok_seeds), len(seeds), ok_seeds, fail_seeds))

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'task':22s}  success  rate   ok_seeds", flush=True)
    for task, n_ok, n_tot, ok_seeds, fail_seeds in summary:
        print(
            f"{task:22s}  {n_ok:2d}/{n_tot:<2d}     {100.0 * n_ok / n_tot:5.1f}%  {ok_seeds}",
            flush=True,
        )


if __name__ == "__main__":
    main()
