#!/usr/bin/env python3
"""Record tagged final demos for marble_shelf_maze (default/opt1/opt2/opt1+2).

Avoids subprocess.capture_output pipe deadlocks by recording in-process.
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import record_demo

TASK = "marble_shelf_maze"
OUT = os.path.abspath(f"./docs/final_task_demos/{TASK}")
NEED_MIB = 7000

CONDS = {
    "default": ["continuous_ball_motion=false", "oscillating_bowl_enabled=false"],
    "opt1": ["continuous_ball_motion=true", "oscillating_bowl_enabled=false"],
    "opt2": ["continuous_ball_motion=false", "oscillating_bowl_enabled=true"],
    "opt1+2": ["continuous_ball_motion=true", "oscillating_bowl_enabled=true"],
}


def free_mib() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    return float(out.split()[0])


def wait_gpu(need: float = NEED_MIB) -> None:
    while True:
        f = free_mib()
        print(f"  GPU free={f:.0f} MiB (need>={need})", flush=True)
        if f >= need:
            return
        time.sleep(20)


def main(only: list[str] | None = None) -> None:
    os.makedirs(OUT, exist_ok=True)
    tags = only or list(CONDS.keys())
    exported: dict[str, str] = {}
    for tag in tags:
        overrides = CONDS[tag]
        ok = False
        for attempt in range(1, 4):
            wait_gpu()
            print(f"\n=== recording {tag} attempt {attempt} ===", flush=True)
            try:
                info = record_demo(
                    TASK,
                    config_name="demo_dynamic",
                    task_arg_overrides=overrides,
                    tag=tag,
                )
                for key, suffix in (
                    ("sidebyside", "sidebyside"),
                    ("head", "head"),
                    ("topdown", "topdown"),
                ):
                    dst = os.path.join(OUT, f"{tag}_{suffix}.mp4")
                    shutil.copy2(info[key], dst)
                    if key == "sidebyside":
                        exported[tag] = dst
                        print(f"  copied -> {dst}", flush=True)
                ok = True
                break
            except Exception as e:
                print(f"  FAIL {tag}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(8 * attempt)
        if not ok:
            raise SystemExit(f"failed to record {tag}")
        gc.collect()
        time.sleep(2)

    with open(os.path.join(OUT, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : continuous_ball_motion=false, oscillating_bowl_enabled=false\n"
            "opt1     : continuous_ball_motion=true,  oscillating_bowl_enabled=false\n"
            "opt2     : continuous_ball_motion=false, oscillating_bowl_enabled=true\n"
            "opt1+2   : continuous_ball_motion=true,  oscillating_bowl_enabled=true\n\n"
            "Success: bowl collects the target marble.\n"
            "Physics: gap-limited tilt; table miss = fail (no shelf teleport).\n"
            "Setup: shelves/buttons 10% wider each side (shelf_length 0.18).\n"
            "Opt 1: slower roll-off + stronger damping / longer tilt.\n\n"
            "Files: <tag>_sidebyside.mp4 (+ _head / _topdown)\n"
        )

    report_path = os.path.join(OUT, "test_report.json")
    report: dict = {}
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
    # Keep prior demo paths for tags we skipped.
    prev = dict(report.get("demos") or {})
    prev.update(exported)
    report["demos"] = prev
    report["demos_rerecorded_after_physics_fix"] = True
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\nALL DEMOS OK", flush=True)
    for k, v in prev.items():
        print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    only = sys.argv[1:] or None
    main(only)
