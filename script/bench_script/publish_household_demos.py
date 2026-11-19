#!/usr/bin/env python3
"""Record and publish default expert demos for all household tasks."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, "./")

from script.bench_script.record_demo import record_demo

ROOT = os.path.abspath(".")
FINAL = os.path.join(ROOT, "docs/final_task_demos")

# Matches interactive/household_task_gui.py + README extras.
HOUSEHOLD_TASKS = (
    "trap_bug",
    "catch_cup",
    "catch_mouse_object_drop",
    "stop_ball",
    "clean_table",
    "fill_coffee_jar",
    "pour_beer",
    "boil_milk",
    "cook_food",
    "cook_food_timer",
    "make_soup",
    "measure_ingredient",
)

# Optional per-task overrides passed to record_demo (task_arg_overrides list).
TASK_ARG_OVERRIDES: dict[str, list[str]] = {}


def gif_from_mp4(mp4: str, gif: str) -> None:
    os.makedirs(os.path.dirname(gif) or ".", exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            mp4,
            "-vf",
            "fps=10,scale=960:-1:flags=lanczos",
            "-loop",
            "0",
            gif,
        ],
        check=True,
    )


def publish(task: str, out: dict) -> dict:
    dest = os.path.join(FINAL, task)
    os.makedirs(dest, exist_ok=True)
    published = {}
    for kind in ("head", "topdown", "sidebyside"):
        src = out.get(kind)
        if not src or not os.path.isfile(src):
            print(f"  WARN missing {kind}: {src}")
            continue
        dst = os.path.join(dest, f"default_{kind}.mp4")
        shutil.copy2(src, dst)
        published[kind] = dst
        print(f"  PUBLISH {dst}")
    side = published.get("sidebyside")
    if side:
        gif = os.path.join(dest, "default_sidebyside.gif")
        gif_from_mp4(side, gif)
        published["sidebyside_gif"] = gif
        print(f"  PUBLISH {gif}")
        snapshot = os.path.join(dest, "scene_snapshot.png")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                side,
                "-vf",
                "select=eq(n\\,0)",
                "-vframes",
                "1",
                snapshot,
            ],
            check=True,
        )
        published["scene_snapshot"] = snapshot
        print(f"  PUBLISH {snapshot}")
    return published


def main() -> int:
    only = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    failed = []
    for task in HOUSEHOLD_TASKS:
        if only and task not in only:
            continue
        print(f"\n=== {task} ===")
        try:
            overrides = TASK_ARG_OVERRIDES.get(task, [])
            out = record_demo(task, tag="default", task_arg_overrides=overrides or None)
            publish(task, out)
        except Exception as exc:
            print(f"FAILED {task}: {exc}")
            failed.append(task)
    if failed:
        print("\nFailed tasks:", ", ".join(failed))
        return 1
    print("\nAll household demos published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
