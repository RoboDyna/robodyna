#!/usr/bin/env python3
"""Run catch_mouse_object_drop across several seeds and print/save a results report.

Each seed runs in a fresh subprocess so CUDA planner memory is released between
trials (same GPU is often shared with other jobs).

Usage (robodyna env, headless Vulkan)::

    python -u script/bench_script/eval_catch_mouse_object_drop.py
    python -u script/bench_script/eval_catch_mouse_object_drop.py --seeds 11,23,37,41,59
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")


def _run_worker(seed: int, out_json: Path) -> int:
    """Single-seed worker (invoked via --worker)."""
    import numpy as np
    from envs.catch_mouse_object_drop import catch_mouse_object_drop
    from record_demo import build_args

    args = build_args(
        "catch_mouse_object_drop", "demo_dynamic",
        f"/tmp/_pb_eval_seed{seed}", None, [],
    )
    args["render_freq"] = 0
    args["collect_data"] = False
    args["save_data"] = False
    args["check_render_success"] = False
    args["save_failed_cases"] = True

    row = {
        "seed": int(seed),
        "success": False,
        "error": None,
        "target_label": None,
        "target_mode": None,
        "target_x": None,
        "target_side": None,
        "mouse_end": None,
        "arm_side": None,
        "basket_placed": None,
        "caught": None,
        "fell_on_table": None,
        "obj_state": None,
        "in_basket": None,
        "touches_table": None,
        "target_final": None,
        "basket_final": None,
        "path_len": None,
        "landing": None,
    }
    task = catch_mouse_object_drop()
    try:
        task.setup_demo(now_ep_num=0, seed=int(seed), **args)
        tx = float(task.target_start[0])
        row.update({
            "target_label": str(task.target_label),
            "target_mode": str(task.target_mode),
            "target_x": round(tx, 3),
            "target_side": "right" if tx >= 0.0 else "left",
            "mouse_end": str(task.mouse_end),
            "arm_side": str(task.arm_side),
            "path_len": round(float(task._mouse_path_len), 3),
            "landing": [round(float(v), 3) for v in task._landing],
        })
        print(
            f"[seed {seed}] target={row['target_label']} ({row['target_mode']}) "
            f"side={row['target_side']} x={row['target_x']} "
            f"mouse_end={row['mouse_end']} arm={row['arm_side']}",
            flush=True,
        )
        task.play_once()
        ok = bool(task.check_success())
        tp = np.array(task.target.get_pose().p, dtype=np.float64)
        bp = np.array(task.basket.get_pose().p, dtype=np.float64)
        row.update({
            "success": ok,
            "basket_placed": bool(task._basket_placed),
            "caught": bool(task._caught),
            "fell_on_table": bool(task._fell_on_table),
            "obj_state": str(task._obj_state),
            "in_basket": bool(task._target_in_basket()),
            "touches_table": bool(task._object_touches_table()),
            "target_final": [round(float(v), 3) for v in tp],
            "basket_final": [round(float(v), 3) for v in bp],
        })
        print(
            f"  → success={ok} in_basket={row['in_basket']} "
            f"table={row['fell_on_table'] or row['touches_table']} "
            f"basket_placed={row['basket_placed']} "
            f"target_z={tp[2]:.3f}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  → ERROR {row['error']}", flush=True)
        traceback.print_exc()
    finally:
        try:
            task.close_env()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    out_json.write_text(json.dumps(row, indent=2))
    return 0 if row.get("success") else 1


def _spawn_seed(seed: int, out_dir: Path) -> dict:
    out_json = out_dir / f"seed{seed}.json"
    env = os.environ.copy()
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env.pop("DISPLAY", None)
    cmd = [
        sys.executable, "-u",
        "script/bench_script/eval_catch_mouse_object_drop.py",
        "--worker", str(seed),
        "--worker-out", str(out_json),
    ]
    print(f"\n=== spawning seed {seed} ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(Path.cwd()), env=env)
    if out_json.exists():
        return json.loads(out_json.read_text())
    return {
        "seed": int(seed),
        "success": False,
        "error": f"worker exit {proc.returncode}, no result json",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,23,37,41,59")
    parser.add_argument("--out", default="tmp/tmp_catch_mouse_object_drop/eval")
    parser.add_argument("--worker", type=int, default=None)
    parser.add_argument("--worker-out", default=None)
    ns = parser.parse_args()

    if ns.worker is not None:
        if not ns.worker_out:
            raise SystemExit("--worker-out required with --worker")
        raise SystemExit(_run_worker(ns.worker, Path(ns.worker_out)))

    seeds = [int(s.strip()) for s in ns.seeds.split(",") if s.strip()]
    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [_spawn_seed(seed, out_dir) for seed in seeds]
    n_ok = sum(1 for r in rows if r.get("success"))
    report = {
        "n": len(rows),
        "n_success": n_ok,
        "success_rate": n_ok / max(len(rows), 1),
        "seeds": rows,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n======= catch_mouse_object_drop eval =======")
    print(f"{'seed':>5}  {'ok':>5}  {'target':<12} {'side':<5} {'mouse':<5} {'arm':<5}  note")
    for r in rows:
        note = r.get("error") or (
            "table" if (r.get("fell_on_table") or r.get("touches_table"))
            else ("in_box" if r.get("in_basket") else str(r.get("obj_state")))
        )
        print(
            f"{r.get('seed'):>5}  {str(r.get('success')):>5}  "
            f"{str(r.get('target_label') or '?'):<12} "
            f"{str(r.get('target_side') or '?'):<5} "
            f"{str(r.get('mouse_end') or '?'):<5} "
            f"{str(r.get('arm_side') or '?'):<5}  {note}"
        )
    print(f"\n{n_ok}/{len(rows)} succeeded  → {report_path}")


if __name__ == "__main__":
    main()
