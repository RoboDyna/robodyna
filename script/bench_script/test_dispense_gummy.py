#!/usr/bin/env python3
"""Full controller test + tagged demos for dispense_gummy.

Conditions (5 episodes each):
  default : layout_mode=alternating, belt_continuous_motion=false
  opt1    : layout_mode=random,      belt_continuous_motion=false
  opt2    : layout_mode=alternating, belt_continuous_motion=true
  opt1+2  : layout_mode=random,      belt_continuous_motion=true

Success: bowl collects every target-colored gummy; any distractor in the bowl
fails the episode. Missed targets also fail.

Demos land in:
  docs/final_task_demos/dispense_gummy/<tag>_sidebyside.mp4
  with tags: default, opt1, opt2, opt1+2
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args, record_demo
from script.collect_data import class_decorator

TASK = "dispense_gummy"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5
BOWL_SPEED_NOMINAL = 0.15
BELT_SPEED_JITTER = 0.20

CONDITIONS = {
    "default": {
        "layout_mode": "alternating",
        "belt_continuous_motion": False,
    },
    "opt1": {
        "layout_mode": "random",
        "belt_continuous_motion": False,
    },
    "opt2": {
        "layout_mode": "alternating",
        "belt_continuous_motion": True,
    },
    "opt1+2": {
        "layout_mode": "random",
        "belt_continuous_motion": True,
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


def _snapshot(env) -> dict:
    target = str(getattr(env, "target_color", "yellow"))
    distractor = "blue" if target == "yellow" else "yellow"
    left = list(getattr(env, "_tube_stack_colors", {}).get("left", []))
    right = list(getattr(env, "_tube_stack_colors", {}).get("right", []))
    yellow_caught = int(getattr(env, "yellow_caught", 0))
    yellow_missed = int(getattr(env, "yellow_missed", 0))
    blue_caught = int(getattr(env, "blue_caught", 0))
    blue_dropped = int(getattr(env, "blue_dropped", 0))
    total_target = int(getattr(env, "total_target", 0))
    target_caught = yellow_caught if target == "yellow" else blue_caught
    target_missed = yellow_missed if target == "yellow" else blue_dropped
    distractor_caught = blue_caught if target == "yellow" else yellow_caught
    distractor_dropped = blue_dropped if target == "yellow" else yellow_missed
    expected_target = sum(1 for c in left + right if c == target)
    return {
        "target_color": target,
        "distractor_color": distractor,
        "layout_mode": str(getattr(env, "layout_mode", "")),
        "belt_continuous_motion": bool(getattr(env, "belt_continuous_motion", False)),
        "bowl_speed": float(getattr(env, "bowl_speed", 0.0)),
        "left_layout": left,
        "right_layout": right,
        "total_target": total_target,
        "expected_target": int(expected_target),
        "target_caught": int(target_caught),
        "target_missed": int(target_missed),
        "distractor_caught": int(distractor_caught),
        "distractor_dropped": int(distractor_dropped),
        "yellow_caught": yellow_caught,
        "yellow_missed": yellow_missed,
        "blue_caught": blue_caught,
        "blue_dropped": blue_dropped,
        "invalid_pattern": bool(getattr(env, "invalid_pattern", False)),
        "press_count": int(len(getattr(env, "press_history", []))),
        "option_label": str(
            env._option_label() if hasattr(env, "_option_label") else ""
        ),
    }


def _layout_is_alternating(left: list, right: list, target: str) -> bool:
    distractor = "blue" if target == "yellow" else "yellow"
    if len(left) != len(right) or not left:
        return False
    for depth, (lc, rc) in enumerate(zip(left, right)):
        if {lc, rc} != {target, distractor}:
            return False
        if depth > 0:
            # Alternating across depths within each tube.
            if lc == left[depth - 1] or rc == right[depth - 1]:
                return False
    return True


def _layout_has_one_target_per_depth(left: list, right: list, target: str) -> bool:
    if len(left) != len(right):
        return False
    for lc, rc in zip(left, right):
        if int(lc == target) + int(rc == target) > 1:
            return False
    return True


def _criteria_ok(snap: dict) -> bool:
    """Independent success check: all targets in bowl, zero distractors in bowl."""
    if snap["invalid_pattern"]:
        return False
    if snap["expected_target"] < 1:
        return False
    if snap["total_target"] != snap["expected_target"]:
        return False
    if snap["target_caught"] != snap["total_target"]:
        return False
    if snap["target_missed"] != 0:
        return False
    if snap["distractor_caught"] != 0:
        return False
    return True


def _condition_shape_ok(condition: str, snap: dict) -> bool:
    cfg = CONDITIONS[condition]
    if snap["layout_mode"] != cfg["layout_mode"]:
        return False
    if bool(snap["belt_continuous_motion"]) != bool(cfg["belt_continuous_motion"]):
        return False
    if not _layout_has_one_target_per_depth(
        snap["left_layout"], snap["right_layout"], snap["target_color"]
    ):
        return False
    if cfg["layout_mode"] == "alternating":
        if not _layout_is_alternating(
            snap["left_layout"], snap["right_layout"], snap["target_color"]
        ):
            return False
    else:
        # Random: require uneven target counts across tubes (task contract).
        left_n = sum(1 for c in snap["left_layout"] if c == snap["target_color"])
        right_n = sum(1 for c in snap["right_layout"] if c == snap["target_color"])
        if left_n < 1 or right_n < 1 or left_n == right_n:
            return False
    if cfg["belt_continuous_motion"]:
        lo = BOWL_SPEED_NOMINAL * (1.0 - BELT_SPEED_JITTER)
        hi = BOWL_SPEED_NOMINAL * (1.0 + BELT_SPEED_JITTER)
        if not (lo - 1e-9 <= snap["bowl_speed"] <= hi + 1e-9):
            return False
    return True


def _run_episode(task_args_overrides: list[str], seed: int, label: str, condition: str) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}_test")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(TASK, CONFIG, save_root, option=None, task_arg_overrides=task_args_overrides)
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False

    env = class_decorator(TASK)
    row = {
        "label": label,
        "seed": seed,
        "success": False,
        "plan_success": False,
        "criteria_ok": False,
        "check_success": False,
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
        criteria_ok = _criteria_ok(snap)
        shape_ok = _condition_shape_ok(condition, snap)
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
        if args.get("render_freq"):
            try:
                env.viewer.close()
            except Exception:
                pass
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION):
    overrides = _overrides(condition)
    results = []
    print(f"\n=== {condition}: {n} tests | overrides={overrides} ===", flush=True)
    for i in range(n):
        seed = 1000 * (list(CONDITIONS.keys()).index(condition) + 1) + i
        label = f"{condition}/ep{i}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"shape={row.get('shape_ok')} consistent={row.get('criteria_consistent')} "
            f"layout={row.get('layout_mode')} continuous={row.get('belt_continuous_motion')} "
            f"speed={row.get('bowl_speed')} "
            f"target={row.get('target_caught')}/{row.get('total_target')} "
            f"distractor_in_bowl={row.get('distractor_caught')} "
            f"L={row.get('left_layout')} R={row.get('right_layout')} "
            f"err={row.get('error')}",
            flush=True,
        )
        time.sleep(0.05)
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {condition}: {n_ok}/{n} successes ===", flush=True)
    return results


def record_and_export_demos():
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    exported = {}
    for condition, file_tag in DEMO_FILE_TAGS.items():
        print(f"\n=== recording demo: {condition} (tag={file_tag}) ===", flush=True)
        info = record_demo(
            TASK,
            config_name=CONFIG,
            task_arg_overrides=_overrides(condition),
            tag=file_tag,
        )
        for key, suffix in (
            ("sidebyside", "sidebyside"),
            ("head", "head"),
            ("topdown", "topdown"),
        ):
            src = info[key]
            dst = os.path.join(out_dir, f"{file_tag}_{suffix}.mp4")
            shutil.copy2(src, dst)
            if key == "sidebyside":
                exported[condition] = dst
                print(f"  copied -> {dst}", flush=True)
    with open(os.path.join(out_dir, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : layout_mode=alternating, belt_continuous_motion=false\n"
            "           alternating target/distractor; discrete belt station hops\n"
            "opt1     : layout_mode=random,      belt_continuous_motion=false\n"
            "           randomized stacks; ≤1 target per depth; distractors may both appear\n"
            "opt2     : layout_mode=alternating, belt_continuous_motion=true\n"
            "           hold arrow key to slide bowl; speed × U(1±0.20)\n"
            "opt1+2   : layout_mode=random,      belt_continuous_motion=true\n"
            "           random layout + continuous bowl motion\n\n"
            "Success: bowl collects every target-colored gummy;\n"
            "         any distractor in the bowl → failure;\n"
            "         any missed target → failure.\n\n"
            "Files (side-by-side is the primary deliverable):\n"
            "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
            "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
            "Also: <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    all_results = {}
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        all_results[cond] = run_condition(cond, n=N_PER_CONDITION)

    summary = {}
    print("\n========== SUMMARY ==========", flush=True)
    all_ok = True
    for key, rows in all_results.items():
        n_ok = sum(1 for r in rows if r["success"])
        n = len(rows)
        n_consistent = sum(1 for r in rows if r.get("criteria_consistent"))
        summary[key] = {
            "successes": n_ok,
            "attempts": n,
            "rate": n_ok / max(n, 1),
            "criteria_consistent": n_consistent,
            "config": CONDITIONS[key],
        }
        print(
            f"  {key}: {n_ok}/{n}  criteria_consistent={n_consistent}/{n}",
            flush=True,
        )
        if n_ok < N_PER_CONDITION:
            all_ok = False

    exported = record_and_export_demos()

    report = {
        "task": TASK,
        "n_per_condition": N_PER_CONDITION,
        "summary": summary,
        "conditions": {
            k: {
                **summary[k],
                "episodes": all_results[k],
            }
            for k in all_results
        },
        "demos": exported,
        "success_criterion": (
            "bowl collects all target gummies; "
            "any distractor in bowl → fail; "
            "any missed target → fail"
        ),
    }
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}", flush=True)
    print(f"Demos in {out_dir}", flush=True)
    if not all_ok:
        print("\n!! SOME CONDITIONS DID NOT REACH 5/5 SUCCESSES", flush=True)
        sys.exit(1)
    print("\nAll conditions 5/5 successes.", flush=True)


if __name__ == "__main__":
    main()
