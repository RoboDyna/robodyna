#!/usr/bin/env python3
"""Record 5 diverse successful stop_ball demos via the shared recorder."""
from __future__ import annotations

import os
import shutil
import sys
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args, record_demo
from script.collect_data import class_decorator

TASK = "stop_ball"
CONFIG = "demo_dynamic"

SCENARIOS = [
    {"tag": "front", "seed": 0, "angle": 0.00},
    {"tag": "left_mild", "seed": 3, "angle": -0.35},
    {"tag": "right_mild", "seed": 8, "angle": 0.35},
    {"tag": "left_angled", "seed": 20, "angle": -0.45},
    {"tag": "right_angled", "seed": 24, "angle": 0.45},
]


def _side_for(angle: float, seed: int) -> str:
    if abs(angle) < 1e-6:
        return "left" if seed % 2 == 0 else "right"
    return "right" if angle > 0.0 else "left"


def _patch_heading(env, angle: float):
    orig_load = env.load_actors

    def load_actors():
        orig_build = env._build_trajectory
        orig_feasible = env._feasible_angle

        def build():
            env._sample_roll_heading = lambda: (
                float(angle),
                np.array([np.sin(angle), -np.cos(angle)], dtype=np.float64),
            )
            env._feasible_angle = lambda ang, *a, **k: float(ang)
            try:
                return orig_build()
            finally:
                env._feasible_angle = orig_feasible

        env._build_trajectory = build
        return orig_load()

    env.load_actors = load_actors


def probe(seed: int, angle: float) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}_probe")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(TASK, CONFIG, save_root, option=None, task_arg_overrides=[])
    args.update(
        collect_data=False,
        save_data=False,
        eval_video_log=False,
        need_plan=True,
        render_freq=0,
        episode_num=1,
        check_render_success=False,
        use_dynamic=False,
    )
    side = _side_for(angle, seed)
    args.setdefault("task_args", {}).setdefault(TASK, {})["arm_side"] = side
    env = class_decorator(TASK)
    row = {"seed": seed, "angle": angle, "side": side, "ok": False}
    try:
        _patch_heading(env, angle)
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        row["ok"] = bool(env.plan_success and env.check_success())
        row["arm"] = str(env.arm_side)
        row["roll_angle"] = float(getattr(env, "_roll_angle", angle))
        row["exit_edge"] = str(getattr(env, "_exit_edge", "?"))
        row["xyz"] = [round(float(x), 3) for x in env._ball_centre()]
    except Exception as e:  # noqa: BLE001
        row["err"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        try:
            env.close_env()
        except Exception:
            pass
        shutil.rmtree(save_root, ignore_errors=True)
    return row


def find_seed(angle: float, prefer: int, tag: str) -> dict:
    candidates = [prefer] + [prefer + d for d in range(1, 50)]
    for seed in candidates:
        print(f"  probe {tag}: seed={seed} angle={angle:+.2f}", flush=True)
        row = probe(seed, angle)
        if row.get("ok"):
            print(
                f"  OK {tag}: seed={seed} arm={row['arm']} "
                f"angle={row['roll_angle']:+.3f} edge={row['exit_edge']} xyz={row['xyz']}",
                flush=True,
            )
            return row
        print(f"  miss {tag}: seed={seed} err={row.get('err')}", flush=True)
    raise RuntimeError(f"No successful seed for {tag}")


def record_one(seed: int, angle: float, tag: str) -> dict:
    """Call shared record_demo, forcing seed + heading + arm side."""
    import script.bench_script.record_demo as rd
    import script.collect_data as cd

    side = _side_for(angle, seed)
    overrides = [f"arm_side={side}"]

    orig_decorator = rd.class_decorator
    orig_run = rd.run

    def class_decorator_patched(name):
        env = orig_decorator(name)
        _patch_heading(env, angle)
        # Always use the chosen seed regardless of epid counter.
        orig_setup = env.setup_demo

        def setup_demo(**kwags):
            kwags["seed"] = int(seed)
            return orig_setup(**kwags)

        env.setup_demo = setup_demo
        return env

    def run_patched(TASK_ENV, args):
        # Plan only the forced seed once; then let collect phase proceed.
        print(f"\033[93m[Forced seed={seed} angle={angle:+.2f} arm={side}]\033[0m", flush=True)
        args = dict(args)
        args["need_plan"] = True
        os.makedirs(args["save_path"], exist_ok=True)
        TASK_ENV.setup_demo(now_ep_num=0, seed=seed, **args)
        TASK_ENV.play_once()
        ok = bool(TASK_ENV.plan_success and TASK_ENV.check_success())
        if not ok:
            raise RuntimeError(
                f"Forced scenario failed: tag={tag} seed={seed} "
                f"plan={TASK_ENV.plan_success} check={TASK_ENV.check_success()} "
                f"fell={TASK_ENV._fell_off} state={TASK_ENV._ball_state}"
            )
        print(f"simulate data episode 0 success! (seed = {seed})", flush=True)
        with open(os.path.join(args["save_path"], "seed.txt"), "w", encoding="utf-8") as f:
            f.write(f"{seed}\n")
        try:
            TASK_ENV.close_env(clear_cache=False)
        except TypeError:
            TASK_ENV.close_env()
        # Hand off to the standard collect/render path with saved seed + traj.
        args["use_seed"] = True
        return orig_run(TASK_ENV, args)

    rd.class_decorator = class_decorator_patched
    rd.run = run_patched
    cd.class_decorator = class_decorator_patched  # record_demo imports it at call time from itself
    try:
        return record_demo(
            TASK,
            config_name=CONFIG,
            task_arg_overrides=overrides,
            tag=tag,
        )
    finally:
        rd.class_decorator = orig_decorator
        rd.run = orig_run


def main() -> None:
    print("=== Finding 5 successful diverse stop_ball scenarios ===", flush=True)
    chosen = []
    for sc in SCENARIOS:
        row = find_seed(sc["angle"], sc["seed"], sc["tag"])
        chosen.append({**sc, "seed": row["seed"], "angle": sc["angle"]})

    print("\n=== Recording demos ===", flush=True)
    results = []
    for sc in chosen:
        print(
            f"\n--- record {sc['tag']} seed={sc['seed']} angle={sc['angle']:+.2f} ---",
            flush=True,
        )
        results.append(record_one(sc["seed"], sc["angle"], sc["tag"]))

    print("\n=== Done ===", flush=True)
    for r, sc in zip(results, chosen):
        print(
            f"  {sc['tag']:12s} v{r['version']} seed={sc['seed']} "
            f"angle={sc['angle']:+.2f}\n    {r['sidebyside']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
