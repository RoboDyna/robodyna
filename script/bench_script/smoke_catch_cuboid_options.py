"""Smoke-test catch_cuboid expert under default / opt1 / opt2 / opt1+2 (5 seeds each)."""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import yaml

from script.collect_data import class_decorator, get_embodiment_config
from envs import CONFIGS_PATH

TASK = "catch_cuboid"
CONFIG = "demo_dynamic"
N_SEEDS = 5

CONDITIONS = [
    ("default", {"catch_two_cuboids": False, "opaque_surface": False}),
    ("opt1_catch_two_cuboids", {"catch_two_cuboids": True, "opaque_surface": False}),
    ("opt2_opaque_surface", {"catch_two_cuboids": False, "opaque_surface": True}),
    ("opt1+2", {"catch_two_cuboids": True, "opaque_surface": True}),
]


def build_base_args():
    with open(f"./task_config/{CONFIG}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = TASK
    args["task_config"] = CONFIG
    args["episode_num"] = 1
    args["save_path"] = os.path.abspath("./tmp/tmp_catch_cuboid_smoke")
    args["collect_data"] = False
    args["eval_video_log"] = False
    args["save_failed_cases"] = False
    args["use_seed"] = False
    args["check_render_success"] = False
    args["export_lerobot"] = False
    args["need_plan"] = True
    args["save_data"] = False
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = False
    args["data_type"]["third_view"] = False
    args["camera"]["collect_head_camera"] = False
    args["camera"]["collect_wrist_camera"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return emb[t]["file_path"]

    et = args["embodiment"]
    args["left_robot_file"] = emb_file(et[0])
    args["right_robot_file"] = emb_file(et[1])
    args["embodiment_dis"] = et[2]
    args["dual_arm_embodied"] = False
    args["embodiment_name"] = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def run_one(task, args, seed: int, flags: dict) -> dict:
    args = dict(args)
    args.setdefault("task_args", {}).setdefault(TASK, {})
    args["task_args"][TASK] = dict(args["task_args"].get(TASK, {}))
    args["task_args"][TASK].update(flags)
    args["left_joint_path"] = []
    args["right_joint_path"] = []
    args["need_plan"] = True
    args["save_data"] = False

    try:
        task.setup_demo(now_ep_num=0, seed=seed, **args)
        task.play_once()
        plan_ok = bool(getattr(task, "plan_success", False))
        try:
            succ = bool(task.check_success())
        except Exception as e:
            succ = False
            err = f"check_success: {e}"
        else:
            err = None
        holes = list(getattr(task, "_cuboid_holes", []))
        held = []
        try:
            held = [bool(x) for x in task._cuboids_held()]
        except Exception:
            pass
        return {
            "seed": seed,
            "plan_success": plan_ok,
            "check_success": succ,
            "ok": plan_ok and succ,
            "holes": holes,
            "cuboids_held": held,
            "error": err,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "seed": seed,
            "plan_success": False,
            "check_success": False,
            "ok": False,
            "holes": [],
            "cuboids_held": [],
            "error": str(e),
        }
    finally:
        try:
            task.close_env(clear_cache=True)
        except Exception:
            pass


def main():
    os.makedirs("./tmp/tmp_catch_cuboid_smoke", exist_ok=True)
    base = build_base_args()
    task = class_decorator(TASK)
    summary = []

    print(f"\n=== catch_cuboid smoke: {N_SEEDS} seeds × {len(CONDITIONS)} conditions ===\n")
    for name, flags in CONDITIONS:
        print(f"--- {name}: {flags} ---")
        results = []
        for seed in range(N_SEEDS):
            r = run_one(task, base, seed, flags)
            results.append(r)
            status = "OK" if r["ok"] else "FAIL"
            print(
                f"  seed={seed}: {status}  plan={r['plan_success']} "
                f"success={r['check_success']} holes={r['holes']} "
                f"held={r['cuboids_held']}"
                + (f" err={r['error']}" if r["error"] else "")
            )
        n_ok = sum(1 for r in results if r["ok"])
        summary.append((name, flags, n_ok, results))
        print(f"  => {n_ok}/{N_SEEDS} succeeded\n")

    print("=== SUMMARY ===")
    all_pass = True
    for name, flags, n_ok, _ in summary:
        mark = "PASS" if n_ok == N_SEEDS else "FAIL"
        if n_ok != N_SEEDS:
            all_pass = False
        print(f"  [{mark}] {name}: {n_ok}/{N_SEEDS}  {flags}")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
