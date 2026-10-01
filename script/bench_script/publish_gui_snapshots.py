#!/usr/bin/env python3
"""Capture fresh head-camera scene snapshots for GUI task cards.

Suite (base interactive) tasks write four condition stills::

    docs/final_task_demos/<task>/scene_snapshot_default.png
    docs/final_task_demos/<task>/scene_snapshot_opt1.png
    docs/final_task_demos/<task>/scene_snapshot_opt2.png
    docs/final_task_demos/<task>/scene_snapshot_opt1+2.png

plus ``scene_snapshot.png`` (copy of default) for compatibility.

Keyboard+mouse GUI cards get a parallel set with arms stripped::

    docs/final_task_demos/<task>/scene_snapshot_kb_default.png
    ...
    docs/final_task_demos/<task>/scene_snapshot_kb.png   (copy of kb default)

Household GUI tasks write ``scene_snapshot.png`` (robot) and
``scene_snapshot_kb.png`` (keyboard+mouse).

Each still is a single ``head_camera`` frame (no top-down montage).

Usage (repo root, robodyna env, headless Vulkan)::

    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
    python script/bench_script/publish_gui_snapshots.py
    python script/bench_script/publish_gui_snapshots.py cook_meat catch_cuboid
    python script/bench_script/publish_gui_snapshots.py --keyboard-only
    python script/bench_script/publish_gui_snapshots.py --tutorial
    python script/bench_script/publish_gui_snapshots.py --tutorial-keyboard
    python script/bench_script/publish_gui_snapshots.py tutorial_empty
    python script/bench_script/publish_gui_snapshots.py tutorial_keyboard
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import numpy as np
from PIL import Image

from envs.utils.household_view import configure_standard_head_camera
from script.bench_script.record_demo import build_args
from script.collect_data import class_decorator
from interactive.base_task_gui import (
    SCENARIO_OVERRIDES,
    SCENARIOS,
    TASKS as SUITE_TASKS,
)
from interactive.household_task_gui import TASKS as HOUSEHOLD_GUI_TASKS

ROOT = os.path.abspath(".")
FINAL = os.path.join(ROOT, "docs/final_task_demos")
TUTORIAL_DIR = os.path.join(ROOT, "interactive", "Tutorial")
# Opening setup for each GUI card. Parts 1–2 are empty-table; 3 = cube; 4 = ball.
TUTORIAL_PART_STAGES = {1: None, 2: None, 3: "grasp", 4: "ball"}
# First stage of each keyboard+mouse tutorial card (arms stripped).
KEYBOARD_TUTORIAL_CARDS = (
    ("buttons", "num_keys"),
    ("placement", "cup_place"),
    ("base", "gummy_keys"),
    ("household", "stop_ball"),
)
# Upscale head stills for crisp GUI cards (native D435 is typically 640×480).
SNAPSHOT_MAX_WIDTH = 1280
SETTLE_STEPS = 8


def _suite_tasks() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for _label, name in SUITE_TASKS:
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _household_gui_tasks() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in HOUSEHOLD_GUI_TASKS:
        name = row[1]
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _prepare_head_frame(head: np.ndarray) -> np.ndarray:
    """Return an RGB head frame, optionally upscaled to SNAPSHOT_MAX_WIDTH."""
    frame = np.asarray(head)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Unexpected head RGB shape: {getattr(frame, 'shape', None)}")
    frame = frame[:, :, :3]
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if w < SNAPSHOT_MAX_WIDTH:
        scale = SNAPSHOT_MAX_WIDTH / float(w)
        new_w = SNAPSHOT_MAX_WIDTH - (SNAPSHOT_MAX_WIDTH % 2)
        new_h = max(1, int(round(h * scale)))
        new_h += new_h % 2
        frame = np.asarray(
            Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
        )
    return frame


def _overrides_to_cli(overrides: dict) -> list[str]:
    out: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            raw = "true" if value else "false"
        else:
            raw = str(value)
        out.append(f"{key}={raw}")
    return out


def capture_snapshot(
    task_name: str,
    *,
    seed: int = 0,
    scenario: str | None = None,
    task_arg_overrides: list[str] | None = None,
    out_name: str = "scene_snapshot.png",
    after_setup=None,
    dest_dir: str | None = None,
) -> str:
    save_root = os.path.abspath(f"./tmp/tmp_{task_name}_snapshot")
    os.makedirs(save_root, exist_ok=True)

    args = build_args(
        task_name,
        config_name="demo_dynamic",
        save_root=save_root,
        option=None,
        task_arg_overrides=task_arg_overrides,
    )
    args.update(
        collect_data=False,
        save_data=False,
        eval_video_log=False,
        need_plan=False,
        render_freq=0,
        episode_num=1,
        use_seed=True,
    )
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["third_view"] = False

    task = class_decorator(task_name)
    _orig_setup = task.setup_demo

    def _setup_demo(**kwargs):
        _orig_setup(**kwargs)
        # Same elevated head pose for base suite + household GUI cards.
        configure_standard_head_camera(task)

    task.setup_demo = _setup_demo
    task.setup_demo(now_ep_num=0, seed=int(seed), **args)
    if after_setup is not None:
        after_setup(task)

    # Let physics settle so floating/resting objects look natural.
    for _ in range(SETTLE_STEPS):
        try:
            task._update_render()
            if getattr(task, "scene", None) is not None:
                task.scene.step()
        except Exception:
            break

    obs = task.get_obs()
    head = _prepare_head_frame(obs["observation"]["head_camera"]["rgb"])

    dest_dir = os.path.abspath(dest_dir or os.path.join(FINAL, task_name))
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, out_name)
    Image.fromarray(head).save(out_path, optimize=True)
    tag = f" [{scenario}]" if scenario else ""
    print(f"  WROTE{tag} {out_path} ({head.shape[1]}x{head.shape[0]})", flush=True)

    try:
        task.close_env(clear_cache=True)
    except Exception:
        try:
            task.close_env()
        except Exception:
            pass
    return out_path


def _strip_arms_after_setup(task) -> None:
    from interactive._interactive_common import strip_interactive_arms

    strip_interactive_arms(task)


def publish_suite_task(task_name: str, seed: int = 0, *, keyboard: bool = False) -> None:
    if task_name not in SCENARIO_OVERRIDES:
        raise KeyError(f"No SCENARIO_OVERRIDES for suite task {task_name}")
    prefix = "scene_snapshot_kb_" if keyboard else "scene_snapshot_"
    after = _strip_arms_after_setup if keyboard else None
    for scenario in SCENARIOS:
        overrides = SCENARIO_OVERRIDES[task_name][scenario]
        out = capture_snapshot(
            task_name,
            seed=seed,
            scenario=scenario,
            task_arg_overrides=_overrides_to_cli(overrides),
            out_name=f"{prefix}{scenario}.png",
            after_setup=after,
        )
        if scenario == "default":
            # Keep the single-file name in sync with Default (robot or keyboard).
            legacy_name = "scene_snapshot_kb.png" if keyboard else "scene_snapshot.png"
            legacy = os.path.join(FINAL, task_name, legacy_name)
            shutil.copy2(out, legacy)
            print(f"  WROTE [legacy] {legacy}", flush=True)


def publish_household_task(task_name: str, seed: int = 0, *, keyboard: bool = False) -> None:
    out_name = "scene_snapshot_kb.png" if keyboard else "scene_snapshot.png"
    capture_snapshot(
        task_name,
        seed=seed,
        out_name=out_name,
        after_setup=_strip_arms_after_setup if keyboard else None,
    )


def publish_tutorial_snapshots(seed: int = 0, parts: list[int] | None = None) -> None:
    """Head-camera stills for Tutorial GUI cards (per-part filenames)."""
    os.makedirs(TUTORIAL_DIR, exist_ok=True)
    wanted = list(parts or TUTORIAL_PART_STAGES.keys())
    for part in wanted:
        stage = TUTORIAL_PART_STAGES.get(int(part))
        out_name = f"scene_snapshot_part{int(part)}.png"

        def _after(task, spawn_stage=stage):
            if spawn_stage:
                task.tutorial_set_stage(spawn_stage)

        print(f"\n=== tutorial part {part} (stage={stage or 'empty'}) ===", flush=True)
        path = capture_snapshot(
            "tutorial_empty",
            seed=seed,
            after_setup=_after,
            out_name=out_name,
        )
        tutorial_copy = os.path.join(TUTORIAL_DIR, out_name)
        shutil.copy2(path, tutorial_copy)
        print(f"  WROTE [tutorial] {tutorial_copy}", flush=True)


def publish_keyboard_tutorial_snapshots(
    seed: int = 0, stems: list[str] | None = None
) -> None:
    """Head-camera stills for keyboard+mouse Tutorial GUI cards."""
    from interactive._interactive_common import strip_interactive_arms

    os.makedirs(TUTORIAL_DIR, exist_ok=True)
    wanted = set(stems) if stems else {stem for stem, _stage in KEYBOARD_TUTORIAL_CARDS}
    for stem, stage in KEYBOARD_TUTORIAL_CARDS:
        if stem not in wanted:
            continue
        out_name = f"scene_snapshot_kb_{stem}.png"

        def _after(task, spawn_stage=stage):
            strip_interactive_arms(task)
            task.tutorial_set_stage(spawn_stage)

        print(f"\n=== tutorial keyboard {stem} (stage={stage}) ===", flush=True)
        path = capture_snapshot(
            "tutorial_keyboard",
            seed=seed,
            after_setup=_after,
            out_name=out_name,
        )
        tutorial_copy = os.path.join(TUTORIAL_DIR, out_name)
        shutil.copy2(path, tutorial_copy)
        print(f"  WROTE [tutorial] {tutorial_copy}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Optional task name filter (default: all selected suite/household tasks).",
    )
    parser.add_argument(
        "--suite-only",
        action="store_true",
        help="Only publish base interactive (suite) four-condition snapshots.",
    )
    parser.add_argument(
        "--household-only",
        action="store_true",
        help="Only publish household single snapshots.",
    )
    parser.add_argument(
        "--tutorial",
        action="store_true",
        help="Publish robot and keyboard Tutorial GUI stills.",
    )
    parser.add_argument(
        "--tutorial-keyboard",
        action="store_true",
        help="Publish keyboard+mouse Tutorial GUI stills only.",
    )
    parser.add_argument(
        "--tutorial-part",
        type=int,
        nargs="*",
        choices=(1, 2, 3, 4),
        help="Robot tutorial part numbers to snapshot (default: all parts).",
    )
    parser.add_argument(
        "--keyboard-only",
        action="store_true",
        help="Publish keyboard+mouse task stills only (skip robot-control stills).",
    )
    parser.add_argument(
        "--skip-keyboard",
        action="store_true",
        help="Skip keyboard+mouse task stills (robot-control stills only).",
    )
    ns = parser.parse_args()
    if ns.keyboard_only and ns.skip_keyboard:
        parser.error("use either --keyboard-only or --skip-keyboard, not both")
    only = set(ns.tasks) if ns.tasks else None
    want_tutorial = (
        bool(ns.tutorial)
        or bool(ns.tutorial_part)
        or (only is not None and "tutorial_empty" in only)
    )
    want_kb_tutorial = (
        bool(ns.tutorial)
        or bool(ns.tutorial_keyboard)
        or (only is not None and "tutorial_keyboard" in only)
    )
    do_robot = not ns.keyboard_only
    do_keyboard = not ns.skip_keyboard

    suite = [] if ns.household_only else _suite_tasks()
    household = [] if ns.suite_only else _household_gui_tasks()
    # Avoid double-publishing tasks that appear in both lists.
    household = [t for t in household if t not in set(suite)]
    tutorial_only = (
        (ns.tutorial or ns.tutorial_part or ns.tutorial_keyboard)
        and only is None
    ) or (
        only is not None and only <= {"tutorial_empty", "tutorial_keyboard"}
    )
    if tutorial_only:
        suite = []
        household = []

    failed: list[str] = []
    for task in suite:
        if only and task not in only:
            continue
        for keyboard, tag in ((False, "4 scenarios"), (True, "4 keyboard scenarios")):
            if keyboard and not do_keyboard:
                continue
            if not keyboard and not do_robot:
                continue
            print(f"\n=== {task} ({tag}) ===", flush=True)
            try:
                publish_suite_task(task, seed=0, keyboard=keyboard)
            except Exception as exc:
                print(f"FAILED {task}{' [kb]' if keyboard else ''}: {exc}", flush=True)
                failed.append(f"{task}{'[kb]' if keyboard else ''}")

    for task in household:
        if only and task not in only:
            continue
        for keyboard, tag in ((False, "robot"), (True, "keyboard")):
            if keyboard and not do_keyboard:
                continue
            if not keyboard and not do_robot:
                continue
            print(f"\n=== {task} ({tag}) ===", flush=True)
            try:
                publish_household_task(task, seed=0, keyboard=keyboard)
            except Exception as exc:
                print(f"FAILED {task}{' [kb]' if keyboard else ''}: {exc}", flush=True)
                failed.append(f"{task}{'[kb]' if keyboard else ''}")

    if want_tutorial:
        print("\n=== tutorial_empty (GUI parts) ===", flush=True)
        try:
            parts = list(ns.tutorial_part) if ns.tutorial_part else None
            publish_tutorial_snapshots(seed=0, parts=parts)
        except Exception as exc:
            print(f"FAILED tutorial_empty: {exc}", flush=True)
            failed.append("tutorial_empty")

    if want_kb_tutorial:
        print("\n=== tutorial_keyboard (GUI parts) ===", flush=True)
        try:
            publish_keyboard_tutorial_snapshots(seed=0)
        except Exception as exc:
            print(f"FAILED tutorial_keyboard: {exc}", flush=True)
            failed.append("tutorial_keyboard")

    if failed:
        print("\nFailed tasks:", ", ".join(failed), flush=True)
        return 1
    print("\nAll GUI snapshots published.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
