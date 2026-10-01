#!/usr/bin/env python3
"""Re-record all catch_marbles_trapdoors final demos with current task setup."""
from __future__ import annotations

import os
import shutil
import sys
from copy import deepcopy

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
)
from script.collect_data import class_decorator

TASK = "catch_marbles_trapdoors"
CONFIG = "demo_dynamic"
OUT = os.path.abspath(f"./docs/final_task_demos/{TASK}")
SAVE_ROOT = os.path.abspath(f"./tmp/tmp_{TASK}")
VIDEO_DIR = os.path.join(SAVE_ROOT, "video")

# file_tag -> (display_name, task_arg overrides)
CONDITIONS = {
    "default": ("default", {
        "door_open_once": False,
        "enable_distractor": False,
        "shuffle_colors": True,
    }),
    "opt1": ("opt1", {
        "door_open_once": True,
        "enable_distractor": False,
        "shuffle_colors": True,
    }),
    "opt2": ("opt2", {
        "door_open_once": False,
        "enable_distractor": True,
        "distractor_collide": False,
        "distractor_height_offset": 0.0,
        "distractor_y_offset": 0.030,
        "shuffle_colors": True,
    }),
    "opt1_opt2": ("opt1+2", {
        "door_open_once": True,
        "enable_distractor": True,
        "distractor_collide": False,
        "distractor_height_offset": 0.0,
        "distractor_y_offset": 0.030,
        "shuffle_colors": True,
    }),
}


def _overrides(cfg: dict) -> list[str]:
    out = []
    for k, v in cfg.items():
        if isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def _clean_scratch(keep_videos: bool = True):
    for junk in ("data", ".cache", "seed.txt", "scene_info.json", "_traj_data", TASK):
        p = os.path.join(SAVE_ROOT, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def _find_success_seed(overrides: list[str], start: int = 0, span: int = 40) -> tuple[int, list, str]:
    args = build_args(TASK, CONFIG, SAVE_ROOT, None, overrides)
    args["collect_data"] = False
    args["save_data"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    for s in range(start, start + span):
        env = class_decorator(TASK)
        try:
            env.setup_demo(now_ep_num=0, seed=s, **args)
            env.play_once()
            info = dict(getattr(env, "info", {}) or {})
            ok = bool(env.plan_success and env.check_success())
            # Probe must satisfy the user-facing metric fields, not just the bool.
            ok = bool(
                ok
                and info.get("ball_in_lower_box")
                and info.get("used_matching_door")
                and not info.get("used_wrong_door")
                and not info.get("ball_still_on_top")
                and not info.get("distractor_through_any")
                and not info.get("distractor_in_lower_box")
            )
            order = list(getattr(env, "color_order", env.button_color_names))
            target = env.button_color_names[env.target_button_idx] if env.target_button_idx >= 0 else ""
            print(
                f"  probe seed={s} ok={ok} order={order} target={target} "
                f"in_box={info.get('ball_in_lower_box')} match={info.get('used_matching_door')} "
                f"dist_in={info.get('distractor_in_lower_box')}",
                flush=True,
            )
            if ok:
                env.close_env()
                return s, order, target
            env.close_env()
        except Exception as e:
            print(f"  probe seed={s} ERROR {e}", flush=True)
            try:
                env.close_env()
            except Exception:
                pass
    raise RuntimeError(f"No success in seeds [{start}, {start + span})")


def record_condition(file_tag: str, display: str, cfg: dict, seed_start: int) -> dict:
    overrides = _overrides(cfg)
    print(f"\n=== {display} ({file_tag}) overrides={overrides} ===", flush=True)

    last_err = None
    for seed in range(seed_start, seed_start + 40):
        try:
            _clean_scratch()
            # Probe this seed first.
            probe_args = build_args(TASK, CONFIG, SAVE_ROOT, None, overrides)
            probe_args["collect_data"] = False
            probe_args["save_data"] = False
            probe_args["need_plan"] = True
            probe_args["render_freq"] = 0
            env = class_decorator(TASK)
            env.setup_demo(now_ep_num=0, seed=seed, **probe_args)
            env.play_once()
            plan_ok = bool(env.plan_success)
            success_ok = bool(plan_ok and env.check_success())
            info = dict(getattr(env, "info", {}) or {})
            ok = bool(
                success_ok
                and info.get("ball_in_lower_box")
                and info.get("used_matching_door")
                and not info.get("used_wrong_door")
                and not info.get("ball_still_on_top")
                and not info.get("distractor_through_any")
                and not info.get("distractor_in_lower_box")
            )
            order = list(getattr(env, "color_order", env.button_color_names))
            target = env.button_color_names[env.target_button_idx] if env.target_button_idx >= 0 else ""
            print(
                f"  probe seed={seed} ok={ok} order={order} target={target} "
                f"in_box={info.get('ball_in_lower_box')} match={info.get('used_matching_door')} "
                f"dist_in={info.get('distractor_in_lower_box')}",
                flush=True,
            )
            env.close_env()
            if not ok:
                continue

            ver = next_version(VIDEO_DIR)
            stem = f"v{ver}_{file_tag}"
            out_head = os.path.join(VIDEO_DIR, f"{stem}_head.mp4")
            out_topdown = os.path.join(VIDEO_DIR, f"{stem}_topdown.mp4")
            out_side = os.path.join(VIDEO_DIR, f"{stem}_sidebyside.mp4")

            args = build_args(TASK, CONFIG, SAVE_ROOT, None, overrides)
            env = class_decorator(TASK)
            orig = env.setup_demo

            def setup(**kw):
                orig(**kw)
                configure_topdown_camera(env)

            env.setup_demo = setup

            def _merge(self):
                if not self.save_data:
                    return
                cache = f"{self.save_dir}/.cache/episode{self.ep_num}/"
                fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
                merge_dual_view_videos(cache, out_head, out_topdown, out_side, fps=fps)

            env.merge_pkl_to_hdf5_video = _merge.__get__(env, env.__class__)

            args2 = deepcopy(args)
            args2["need_plan"] = True
            args2["collect_data"] = False
            args2["save_data"] = True
            env.setup_demo(now_ep_num=0, seed=seed, **args2)
            env.play_once()
            assert env.plan_success and env.check_success(), f"{display}: plan failed on seed {seed}"
            order = list(getattr(env, "color_order", env.button_color_names))
            target = env.button_color_names[env.target_button_idx]
            env.save_traj_data(0)
            env.close_env()

            args2["need_plan"] = False
            args2["collect_data"] = True
            traj = env.load_tran_data(0)
            args2["left_joint_path"] = traj["left_joint_path"]
            args2["right_joint_path"] = traj["right_joint_path"]
            env.setup_demo(now_ep_num=0, seed=seed, **args2)
            env.set_path_lst(args2)
            if int(traj.get("press_lead_steps", 0) or 0) > 0:
                env._press_lead_steps = int(traj["press_lead_steps"])
            env.play_once()
            render_ok = bool(env.plan_success and env.check_success())
            render_info = dict(getattr(env, "info", {}) or {})
            render_ok = bool(
                render_ok
                and render_info.get("ball_in_lower_box")
                and render_info.get("used_matching_door")
                and not render_info.get("used_wrong_door")
                and not render_info.get("ball_still_on_top")
                and not render_info.get("distractor_through_any")
                and not render_info.get("distractor_in_lower_box")
            )
            print(
                f"  render check: ok={render_ok} in_box={render_info.get('ball_in_lower_box')} "
                f"on_top={render_info.get('ball_still_on_top')} match={render_info.get('used_matching_door')} "
                f"wrong={render_info.get('used_wrong_door')} dist_fail={render_info.get('distractor_through_any')} "
                f"dist_in_box={render_info.get('distractor_in_lower_box')}",
                flush=True,
            )
            if not render_ok:
                env.close_env()
                last_err = f"render metric fail seed={seed}"
                continue
            env.close_env()
            env.merge_pkl_to_hdf5_video()

            os.makedirs(OUT, exist_ok=True)
            exports = {
                "sidebyside": os.path.join(OUT, f"{file_tag}_sidebyside.mp4"),
                "head": os.path.join(OUT, f"{file_tag}_head.mp4"),
                "topdown": os.path.join(OUT, f"{file_tag}_topdown.mp4"),
            }
            shutil.copy2(out_side, exports["sidebyside"])
            shutil.copy2(out_head, exports["head"])
            shutil.copy2(out_topdown, exports["topdown"])
            if file_tag == "opt1_opt2":
                for kind in ("sidebyside", "head", "topdown"):
                    alias = os.path.join(OUT, f"opt1+2_{kind}.mp4")
                    shutil.copy2(exports[kind], alias)
                    exports[f"alias_{kind}"] = alias
            print(f"  wrote {exports['sidebyside']}", flush=True)
            return {
                "display": display,
                "file_tag": file_tag,
                "seed": seed,
                "color_order": order,
                "target": target,
                "paths": exports,
            }
        except Exception as e:
            last_err = str(e)
            print(f"  seed={seed} ERROR {e}", flush=True)
            try:
                env.close_env()
            except Exception:
                pass
            continue
    raise RuntimeError(f"{display}: no seed with plan+render success near {seed_start}: {last_err}")


def main(only: list[str] | None = None):
    os.makedirs(OUT, exist_ok=True)
    results = []
    # Stagger seed starts so conditions don't all land on the same shuffle.
    starts = {"default": 0, "opt1": 10, "opt2": 20, "opt1_opt2": 30}
    items = CONDITIONS.items()
    if only:
        items = [(k, CONDITIONS[k]) for k in only]
    for file_tag, (display, cfg) in items:
        results.append(record_condition(file_tag, display, cfg, starts[file_tag]))

    # Keep prior default/opt1 episode notes if we only refreshed distractor demos.
    prior = {}
    cond_path = os.path.join(OUT, "CONDITIONS.txt")
    if only and os.path.isfile(cond_path):
        with open(cond_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("default:") or line.startswith("opt1:"):
                    prior[line.split(":")[0]] = line

    with open(cond_path, "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — final expert demos (current setup)\n\n"
            "Shared setup:\n"
            "  - shuffle_colors=true (doors + keys share randomized left→right color order)\n"
            "  - realistic keys: colored cap on larger/thinner black base\n"
            "  - distractor (opt2 / opt1+2): same height (Z) as target; offset on Y "
            "(distractor_y_offset) so lanes do not collide; pass-through by default "
            "(distractor_collide=false)\n\n"
            "default  : door_open_once=false, enable_distractor=false\n"
            "opt1     : door_open_once=true,  enable_distractor=false\n"
            "opt2     : door_open_once=false, enable_distractor=true\n"
            "opt1+2   : door_open_once=true,  enable_distractor=true  (files: opt1_opt2_* and opt1+2_*)\n\n"
            "Success: target marble through matching-color trapdoor into lower box.\n"
            "Fail: stays on top, wrong-color door, or distractor through any door into lower box.\n"
            "Arm: left if matching trapdoor is on left half, right if on right half.\n\n"
            "Recorded episodes:\n"
        )
        for name in ("default", "opt1"):
            if name in prior:
                f.write(f"  {prior[name]}\n")
        for r in results:
            f.write(
                f"  {r['display']}: seed={r['seed']} color_order={r['color_order']} "
                f"target={r['target']}\n"
            )
        f.write("\nFiles: <tag>_sidebyside.mp4, <tag>_head.mp4, <tag>_topdown.mp4\n")

    print("\nDEMOS UPDATED", flush=True)
    for r in results:
        print(f"  {r['display']}: seed={r['seed']} order={r['color_order']} target={r['target']}")


if __name__ == "__main__":
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "--distractor-only":
        only = ["opt2", "opt1_opt2"]
    main(only=only)
