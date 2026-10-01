#!/usr/bin/env python3
"""20-run base-task suite (5 seeds × 4 scenarios) + record missing success demos.

For each base task / scenario:
  - Run seeds 0..n-1 (default n=5 → 20 runs/task).
  - If the scenario has a successful seed AND the gallery is missing that
    scenario's GIF (or --force-record), record a head-camera demo and publish
    GIF into repo docs/final_task_demos + task_gallery (``*_head`` stems).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "script"),
    str(ROOT / "script/bench_script"),
    str(ROOT / "tests"),
]

from sweep_base_success import (  # noqa: E402
    SCENARIO_OVERRIDES,
    SCENARIOS,
    TASKS,
    run_seed,
)
from record_demo import (  # noqa: E402
    build_args,
    encode_head_video,
    next_version,
    _cleanup_scratch,
)
from collect_data import class_decorator  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
GALLERY = Path(
    os.environ.get(
        "TASK_GALLERY_ROOT",
        str(Path(__file__).resolve().parents[2] / "docs/final_task_demos"),
    )
)
REPO_FINAL = ROOT / "docs/final_task_demos"

# Filename stems for gallery / README (head-camera demos).
SCENARIO_STEMS: dict[str, dict[str, str]] = {
    "catch_cuboid": {
        "default": "default_transparent_1cuboid_head",
        "opt1": "opt1_catch_two_cuboids_head",
        "opt2": "opt2_opaque_1cuboid_head",
        "opt1+2": "opt1+2_catch_two_cuboids_opaque_head",
    },
    "put_cup_belt": {
        "default": "default_left_head",
        "opt1": "opt1_left_head",
        "opt2": "opt2_left_head",
        "opt1+2": "opt1+2_left_head",
    },
    "place_block_belt": {
        "default": "default_head",
        "opt1": "opt1_moving_bowl_head",
        "opt2": "opt2_blocker_head",
        "opt1+2": "opt1+2_moving_bowl_blocker_head",
    },
    "drop_ball_hole": {
        "default": "cylinder_default_head",
        "opt1": "cylinder_opt1_head",
        "opt2": "cylinder_opt2_head",
        "opt1+2": "cylinder_opt1+2_head",
    },
}


def stem_for(task: str, scenario: str) -> str:
    return SCENARIO_STEMS.get(task, {}).get(scenario, f"{scenario}_head")


def gif_exists(task: str, scenario: str) -> bool:
    stem = stem_for(task, scenario)
    for root in (GALLERY / task, REPO_FINAL / task):
        if (root / f"{stem}.gif").is_file():
            return True
    return False


def gif_from_mp4(mp4: Path, gif: Path, scale: int = 480, fps: int = 10) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4),
            "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos",
            "-loop", "0", str(gif),
        ],
        check=True,
    )


def overrides_to_cli(overrides: dict) -> list[str]:
    out = []
    for k, v in overrides.items():
        if isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def record_scenario(task: str, scenario: str, seed: int, save_freq: int = 8) -> Path | None:
    """Record head-camera success demo for a fixed seed+scenario; return head mp4.

    Two-pass like collect_data (plan without save, then head-only render) at the
    policy ~30 Hz rate (save_freq=8, fps ≈ 250/8). Gallery filenames still
    publish under ``*_head`` stems for the gallery index.
    """
    save_root = os.path.abspath(f"./tmp/tmp_{task}")
    video_dir = os.path.join(save_root, "video")
    os.makedirs(video_dir, exist_ok=True)
    ver = next_version(video_dir)
    out_head = os.path.join(video_dir, f"v{ver}_{scenario}_seed{seed}_head.mp4")
    _cleanup_scratch(save_root, task)
    os.makedirs(save_root, exist_ok=True)

    overrides = SCENARIO_OVERRIDES[task][scenario]
    args = build_args(task, "demo_dynamic", save_root, None, overrides_to_cli(overrides))
    args.update(
        need_plan=True,
        save_data=False,
        collect_data=False,
        check_render_success=False,
        save_failed_cases=True,
        render_freq=0,
        save_freq=int(save_freq),
        use_seed=False,
    )
    args.setdefault("camera", {})["head_camera_type"] = "D435"
    args.setdefault("data_type", {})["third_view"] = False

    # Pass 1: plan only
    env = class_decorator(task)
    shutil.rmtree(Path(save_root) / ".cache", ignore_errors=True)
    env.setup_demo(now_ep_num=0, seed=seed, **args)
    env.play_once()
    ok = bool(env.plan_success and env.check_success())
    print(f"RESULT {task} [{scenario}] seed={seed} {'pass' if ok else 'fail'}", flush=True)
    if not ok:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        return None
    env.save_traj_data(0)
    try:
        env.close_env()
    except Exception:
        pass

    # Pass 2: head-only render replay
    render_args = dict(args)
    render_args.update(need_plan=False, save_data=True, render_freq=0)
    env = class_decorator(task)
    shutil.rmtree(Path(save_root) / ".cache", ignore_errors=True)
    env.setup_demo(now_ep_num=0, seed=seed, **render_args)
    env._max_episode_steps = int(os.environ.get("DEMO_MAX_STEPS", "15000"))
    env._max_episode_seconds = None
    traj = env.load_tran_data(0)
    render_args["left_joint_path"] = traj["left_joint_path"]
    render_args["right_joint_path"] = traj["right_joint_path"]
    env.set_path_lst(render_args)
    env.play_once()

    cache_path = f"{env.save_dir}/.cache/episode{env.ep_num}/"
    fps = 250.0 / float(env.save_freq)
    encode_head_video(cache_path, out_head, fps=fps)
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass
    _cleanup_scratch(save_root, task)
    return Path(out_head) if os.path.isfile(out_head) else None


def publish_scenario(task: str, scenario: str, head_mp4: Path) -> str:
    """Publish head-camera mp4/gif under gallery ``*_head`` stems."""
    stem = stem_for(task, scenario)
    for root in (REPO_FINAL / task, GALLERY / task):
        root.mkdir(parents=True, exist_ok=True)
        dst_mp4 = root / f"{stem}.mp4"
        dst_gif = root / f"{stem}.gif"
        shutil.copy2(head_mp4, dst_mp4)
        gif_from_mp4(head_mp4, dst_gif)
        # Also write generic scenario_head name when stem is specialized.
        generic = root / f"{scenario}_head.mp4"
        generic_gif = root / f"{scenario}_head.gif"
        if generic != dst_mp4:
            shutil.copy2(head_mp4, generic)
        if generic_gif != dst_gif:
            shutil.copy2(dst_gif, generic_gif)
        print(f"  PUBLISH {dst_gif}", flush=True)
    return f"{stem}.gif"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="seeds per scenario (total 4*n)")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--record-only", action="store_true")
    ap.add_argument("--force-record", action="store_true",
                    help="Re-record demos even if GIF already exists")
    ap.add_argument("--results", default="logs/basic_sweep_results.json")
    ns = ap.parse_args()

    results_path = Path(ns.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if ns.record_only:
        results = json.loads(results_path.read_text())
    else:
        results = {}
        for task in ns.tasks:
            results[task] = {}
            for scenario in ns.scenarios:
                print(
                    f"\n===== {task} [{scenario}] ({ns.n} seeds) =====",
                    flush=True,
                )
                ok, fail = [], []
                for seed in range(ns.n):
                    row = run_seed(task, seed, scenario=scenario)
                    tag = "OK" if row["ok"] else "FAIL"
                    extra = (
                        f" err={row['err']}"
                        if row["err"]
                        else f" plan={row['plan']} check={row['check']}"
                    )
                    print(f"  seed={seed} {tag}{extra}", flush=True)
                    (ok if row["ok"] else fail).append(seed)
                results[task][scenario] = {"ok": ok, "fail": fail}
                print(
                    f"  → {len(ok)}/{ns.n} ok={ok} fail={fail}",
                    flush=True,
                )
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {results_path}", flush=True)

    print("\n========== SUMMARY ==========", flush=True)
    for task in ns.tasks:
        for scenario in ns.scenarios:
            res = results[task][scenario]
            n_ok = len(res["ok"])
            print(
                f"{task:26s}  {scenario:7s}  {n_ok:2d}/{ns.n}  ok={res['ok']}",
                flush=True,
            )

    if ns.sweep_only:
        return 0

    recorded = []
    for task in ns.tasks:
        for scenario in ns.scenarios:
            res = results[task][scenario]
            if not res["ok"]:
                print(f"SKIP record {task}/{scenario}: no successes", flush=True)
                continue
            if gif_exists(task, scenario) and not ns.force_record:
                print(f"KEEP existing demo {task}/{scenario}", flush=True)
                continue
            seed = res["ok"][0]
            print(
                f"\n===== RECORD {task} [{scenario}] seed={seed} =====",
                flush=True,
            )
            try:
                side = record_scenario(task, scenario, seed)
                if side is None:
                    # try other ok seeds
                    for s2 in res["ok"][1:]:
                        side = record_scenario(task, scenario, s2)
                        if side is not None:
                            seed = s2
                            break
                if side is None:
                    print(f"  FAILED to record {task}/{scenario}", flush=True)
                    continue
                name = publish_scenario(task, scenario, side)
                recorded.append({"task": task, "scenario": scenario, "seed": seed, "file": name})
            except Exception as exc:  # noqa: BLE001
                print(f"  RECORD ERR {task}/{scenario}: {exc}", flush=True)
                traceback.print_exc()

    meta = {"results": results, "recorded": recorded}
    Path("/tmp/basic_gallery_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nRecorded {len(recorded)} new demos → /tmp/basic_gallery_meta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
