#!/usr/bin/env python3
"""Full controller test + tagged demos for pick_ripe_apple.

Conditions (5 episodes each for pass/fail; 1 sidebyside demo each):
  default : two_apples_enabled=false, basket_move_enabled=false
            single good apple; static basket
  opt1    : two_apples_enabled=true,  basket_move_enabled=false
            good + spoiled apple; static basket
  opt2    : two_apples_enabled=false, basket_move_enabled=true
            single good apple; oscillating basket
  opt1+2  : two_apples_enabled=true,  basket_move_enabled=true
            good + spoiled; oscillating basket

Success: good apple picked inside the ripeness window AND in basket;
spoiled (if any) NOT in basket.

Demos land in:
  docs/final_task_demos/pick_ripe_apple/<tag>_sidebyside.mp4
  with tags: default, opt1, opt2, opt1+2
"""
from __future__ import annotations

import gc
import json
import os
import shutil
import sys
import time
import traceback
from copy import deepcopy

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
    record_demo,
)
from script.collect_data import class_decorator

TASK = "pick_ripe_apple"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5
OUT = os.path.abspath(f"./docs/final_task_demos/{TASK}")
SAVE_ROOT = os.path.abspath(f"./tmp/tmp_{TASK}_suite")
VIDEO_DIR = os.path.join(SAVE_ROOT, "video")

CONDITIONS = {
    "default": {
        "two_apples_enabled": False,
        "basket_move_enabled": False,
    },
    "opt1": {
        "two_apples_enabled": True,
        "basket_move_enabled": False,
    },
    "opt2": {
        "two_apples_enabled": False,
        "basket_move_enabled": True,
    },
    "opt1+2": {
        "two_apples_enabled": True,
        "basket_move_enabled": True,
    },
}

DEMO_FILE_TAGS = {
    "default": "default",
    "opt1": "opt1",
    "opt2": "opt2",
    "opt1+2": "opt1+2",
}


def _overrides(condition: str) -> list[str]:
    out = []
    for k, v in CONDITIONS[condition].items():
        if isinstance(v, bool):
            out.append(f"{k}={'true' if v else 'false'}")
        else:
            out.append(f"{k}={v}")
    return out


def _gc():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _snapshot(env) -> dict:
    info = getattr(env, "info", {}) or {}
    spoiled = getattr(env, "spoiled_apple", None)
    return {
        "two_apples_enabled": bool(getattr(env, "two_apples_enabled", False)),
        "basket_move_enabled": bool(getattr(env, "basket_move_enabled", False)),
        "n_apples": int(len(getattr(env, "apples", {}) or {})),
        "good_side": float(getattr(env, "good_side", getattr(env, "apple_side", 0.0))),
        "spoiled_side": float(getattr(env, "spoiled_side", 0.0)),
        "spoiled_present": spoiled is not None,
        "r_grasp": (
            None if getattr(env, "r_grasp", None) is None else float(env.r_grasp)
        ),
        "red_window": float(getattr(env, "red_window", 0.5)),
        "red_tolerance": float(info.get("red_tolerance", getattr(env, "red_tol", 0.12))),
        "ripeness_ok": bool(info.get("ripeness_ok", False)),
        "in_basket": bool(info.get("in_basket", False)),
        "spoiled_in_basket": bool(info.get("spoiled_in_basket", False)),
        "ripeness_score": float(info.get("ripeness_score", 0.0)),
        "basket_speed": float(getattr(env, "basket_speed", 0.0)),
    }


def _criteria_ok(snap: dict, condition: str) -> bool:
    if not snap.get("in_basket"):
        return False
    r_grasp = snap.get("r_grasp")
    if r_grasp is None:
        return False
    if abs(r_grasp - snap["red_window"]) > snap["red_tolerance"]:
        return False
    cfg = CONDITIONS[condition]
    if cfg["two_apples_enabled"]:
        if not snap.get("spoiled_present"):
            return False
        if snap.get("spoiled_in_basket"):
            return False
    else:
        if snap.get("spoiled_present"):
            return False
    return True


def _shape_ok(snap: dict, condition: str) -> bool:
    cfg = CONDITIONS[condition]
    if bool(snap["two_apples_enabled"]) != bool(cfg["two_apples_enabled"]):
        return False
    if bool(snap["basket_move_enabled"]) != bool(cfg["basket_move_enabled"]):
        return False
    expect_n = 2 if cfg["two_apples_enabled"] else 1
    if int(snap["n_apples"]) != expect_n:
        return False
    if bool(snap["spoiled_present"]) != bool(cfg["two_apples_enabled"]):
        return False
    return True


def _run_episode(overrides: list[str], seed: int, label: str, condition: str) -> dict:
    """Headless expert episode (no video)."""
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}_test")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(TASK, CONFIG, save_root, option=None, task_arg_overrides=overrides)
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False
    args["use_dynamic"] = False
    args["camera"]["collect_head_camera"] = False
    args["camera"]["collect_wrist_camera"] = False
    args["data_type"]["rgb"] = False
    args["data_type"]["third_view"] = False

    env = class_decorator(TASK)
    row = {
        "label": label,
        "seed": seed,
        "success": False,
        "plan_success": False,
        "check_success": False,
        "criteria_ok": False,
        "shape_ok": False,
        "criteria_consistent": False,
        "error": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(env.check_success())
        snap = _snapshot(env)
        criteria_ok = _criteria_ok(snap, condition)
        shape_ok = _shape_ok(snap, condition)
        consistent = bool(check_ok == criteria_ok)
        ok = bool(plan_ok and check_ok and criteria_ok and shape_ok)
        row.update(
            {
                "success": ok,
                "plan_success": plan_ok,
                "check_success": check_ok,
                "criteria_ok": criteria_ok,
                "shape_ok": shape_ok,
                "criteria_consistent": consistent,
                **snap,
            }
        )
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        try:
            env.close_env()
        except Exception:
            pass
        _gc()
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION) -> list[dict]:
    overrides = _overrides(condition)
    results = []
    base = {"default": 0, "opt1": 100, "opt2": 200, "opt1+2": 300}[condition]
    print(f"\n=== {condition}: {n} tests | overrides={overrides} ===", flush=True)
    seed = base
    attempts = 0
    max_tries = n * 12
    while len(results) < n and attempts < max_tries:
        attempts += 1
        label = f"{condition}/ep{len(results)}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        # One retry on clear exception flake.
        if (not row["success"]) and row.get("error"):
            print(f"  retry seed={seed} once after error...", flush=True)
            row2 = _run_episode(overrides, seed=seed, label=label, condition=condition)
            if row2["success"] or not row2.get("error"):
                row = row2
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"shape={row.get('shape_ok')} n_apples={row.get('n_apples')} "
            f"two={row.get('two_apples_enabled')} basket_move={row.get('basket_move_enabled')} "
            f"good_side={row.get('good_side')} in_basket={row.get('in_basket')} "
            f"spoiled_in={row.get('spoiled_in_basket')} err={row.get('error')}",
            flush=True,
        )
        seed += 1
        time.sleep(0.05)
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {condition}: {n_ok}/{n} successes ===", flush=True)
    return results


def _clean_scratch():
    for junk in ("data", ".cache", "seed.txt", "scene_info.json", "_traj_data", TASK):
        p = os.path.join(SAVE_ROOT, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)


def record_seeded_demo(condition: str, seed: int, file_tag: str) -> str | None:
    """Record one dual-view demo for a known seed; copy to docs/final_task_demos."""
    overrides = _overrides(condition)
    print(
        f"\n=== recording demo: {condition} tag={file_tag} seed={seed} ===",
        flush=True,
    )
    _clean_scratch()
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

    try:
        args_plan = deepcopy(args)
        args_plan["need_plan"] = True
        args_plan["collect_data"] = False
        args_plan["save_data"] = True
        args_plan["check_render_success"] = False
        args_plan["save_failed_cases"] = True
        env.setup_demo(now_ep_num=0, seed=seed, **args_plan)
        env.play_once()
        plan_ok = bool(env.plan_success)
        check_ok = bool(env.check_success())
        print(f"  plan={plan_ok} check={check_ok}", flush=True)
        env.save_traj_data(0)
        env.close_env()

        args_rend = deepcopy(args)
        args_rend["need_plan"] = False
        args_rend["collect_data"] = True
        args_rend["save_data"] = True
        traj = env.load_tran_data(0)
        args_rend["left_joint_path"] = traj["left_joint_path"]
        args_rend["right_joint_path"] = traj["right_joint_path"]
        env.setup_demo(now_ep_num=0, seed=seed, **args_rend)
        env.set_path_lst(args_rend)
        env.play_once()
        env.close_env()
        env.merge_pkl_to_hdf5_video()

        os.makedirs(OUT, exist_ok=True)
        dst = os.path.join(OUT, f"{file_tag}_sidebyside.mp4")
        if os.path.isfile(out_side):
            shutil.copy2(out_side, dst)
            if os.path.isfile(out_head):
                shutil.copy2(out_head, os.path.join(OUT, f"{file_tag}_head.mp4"))
            if os.path.isfile(out_topdown):
                shutil.copy2(out_topdown, os.path.join(OUT, f"{file_tag}_topdown.mp4"))
            print(f"  copied -> {dst}", flush=True)
            return dst
        print("  WARNING: sidebyside missing after render", flush=True)
        return None
    except Exception as e:
        print(f"  RECORD FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        try:
            env.close_env()
        except Exception:
            pass
        return None
    finally:
        _gc()


def record_and_export_demos(all_results: dict) -> dict:
    """One sidebyside per condition; prefer a successful seed from the suite."""
    os.makedirs(OUT, exist_ok=True)
    exported = {}
    for condition, file_tag in DEMO_FILE_TAGS.items():
        rows = all_results.get(condition) or []
        ok_seeds = [r["seed"] for r in rows if r.get("success")]
        # Fall back to first attempted seed, then record_demo (search).
        seed = ok_seeds[0] if ok_seeds else (rows[0]["seed"] if rows else 0)
        path = record_seeded_demo(condition, seed=seed, file_tag=file_tag)
        if path is None:
            print(
                f"  seeded record failed for {condition}; falling back to record_demo()",
                flush=True,
            )
            try:
                info = record_demo(
                    TASK,
                    config_name=CONFIG,
                    task_arg_overrides=_overrides(condition),
                    tag=file_tag,
                )
                dst = os.path.join(OUT, f"{file_tag}_sidebyside.mp4")
                shutil.copy2(info["sidebyside"], dst)
                shutil.copy2(info["head"], os.path.join(OUT, f"{file_tag}_head.mp4"))
                shutil.copy2(
                    info["topdown"], os.path.join(OUT, f"{file_tag}_topdown.mp4")
                )
                path = dst
                print(f"  copied -> {dst}", flush=True)
            except Exception as e:
                print(f"  fallback record_demo also failed: {e}", flush=True)
                traceback.print_exc()
        if path:
            exported[condition] = {
                "path": path,
                "seed": seed,
                "from_success": bool(ok_seeds),
            }
    return exported


def write_reports(all_results: dict, exported: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    lines = [
        f"{TASK} — expert controller demos + full condition suite\n",
        "default  : two_apples_enabled=false, basket_move_enabled=false",
        "           single good apple; static basket",
        "opt1     : two_apples_enabled=true,  basket_move_enabled=false",
        "           good (red path) + spoiled (yellow→black); static basket",
        "opt2     : two_apples_enabled=false, basket_move_enabled=true",
        "           single good apple; oscillating basket (no pause for drop)",
        "opt1+2   : two_apples_enabled=true,  basket_move_enabled=true",
        "           two apples + oscillating basket\n",
        "Success: good apple picked inside the ripeness window (|r_grasp - red_window|\n"
        "         <= red_tolerance) AND in basket; spoiled (if any) NOT in basket.\n",
        "Files (1 sidebyside demo per condition):\n"
        "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
        "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
        "Also: <tag>_head.mp4, <tag>_topdown.mp4\n",
        "Results (5 tests × 4 conditions):",
    ]
    print("\n========== SUMMARY ==========", flush=True)
    for key in ("default", "opt1", "opt2", "opt1+2"):
        rows = all_results.get(key) or []
        n_ok = sum(1 for r in rows if r.get("success"))
        n = len(rows)
        summary[key] = {
            "successes": n_ok,
            "attempts": n,
            "rate": n_ok / max(n, 1),
            "config": CONDITIONS[key],
            "success_seeds": [r["seed"] for r in rows if r.get("success")],
        }
        print(f"  {key}: {n_ok}/{n}", flush=True)
        lines.append(f"  {key}: {n_ok}/{n}")
        for r in rows:
            st = "PASS" if r.get("success") else "FAIL"
            lines.append(
                f"    seed={r.get('seed')} {st} "
                f"plan={r.get('plan_success')} check={r.get('check_success')} "
                f"n_apples={r.get('n_apples')} "
                f"basket_move={r.get('basket_move_enabled')} "
                f"err={r.get('error')}"
            )
        demo = exported.get(key) or {}
        if demo:
            lines.append(
                f"    demo: {os.path.basename(demo.get('path', ''))} "
                f"(seed={demo.get('seed')}, from_success={demo.get('from_success')})"
            )

    with open(os.path.join(OUT, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    report = {
        "task": TASK,
        "n_per_condition": N_PER_CONDITION,
        "summary": summary,
        "conditions": {
            k: {"episodes": all_results.get(k, []), **summary.get(k, {})}
            for k in CONDITIONS
        },
        "demos": exported,
        "success_criterion": (
            "good apple in basket (r_grasp latched); "
            "spoiled apple (if Opt1) not in basket"
        ),
    }
    report_path = os.path.join(OUT, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}", flush=True)
    print(f"Demos in {OUT}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    only = [a for a in sys.argv[1:] if a in CONDITIONS]
    skip_demos = os.environ.get("SKIP_DEMOS", "").strip() in ("1", "true", "True")
    conds = only or list(CONDITIONS.keys())

    all_results = {}
    for cond in conds:
        all_results[cond] = run_condition(cond, n=N_PER_CONDITION)

    # Merge prior episodes if partial CLI run.
    report_path = os.path.join(OUT, "test_report.json")
    if only and os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            prev = json.load(f)
        for k, block in (prev.get("conditions") or {}).items():
            if k not in all_results and "episodes" in block:
                all_results[k] = block["episodes"]

    exported = {}
    if skip_demos:
        print("\n=== SKIP_DEMOS set; not recording demos ===", flush=True)
    else:
        # Need all 4 condition result lists for seed selection when possible.
        full = {k: all_results.get(k, []) for k in CONDITIONS}
        if only:
            # Only re-record requested tags.
            subset = {k: all_results[k] for k in only}
            for k in only:
                full[k] = subset[k]
            exported = {}
            for cond in only:
                one = record_and_export_demos({cond: all_results[cond]})
                exported.update(one)
        else:
            exported = record_and_export_demos(full)

    write_reports(
        {k: all_results.get(k, []) for k in CONDITIONS},
        exported,
    )

    if any(c not in all_results for c in CONDITIONS):
        print("\nPartial run complete.", flush=True)
        sys.exit(0)
    if any(
        sum(1 for r in all_results[c] if r.get("success")) < N_PER_CONDITION
        for c in CONDITIONS
    ):
        print("\n!! SOME CONDITIONS DID NOT REACH 5/5 SUCCESSES", flush=True)
        sys.exit(1)
    print("\nAll conditions 5/5 successes.", flush=True)


if __name__ == "__main__":
    main()
