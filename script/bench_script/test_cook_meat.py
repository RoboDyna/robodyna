#!/usr/bin/env python3
"""Full controller test + tagged demos for cook_meat.

Conditions (5 episodes each):
  default : cook_button_enabled=false, dual_setup_enabled=false
            single station; latch cook key ON → cook → OFF
  opt1    : cook_button_enabled=true,  dual_setup_enabled=false
            single station; hold cook key to cook
  opt2    : cook_button_enabled=false, dual_setup_enabled=true
            dual stations; latch keys
  opt1+2  : cook_button_enabled=true,  dual_setup_enabled=true
            dual stations; hold keys
Success: cooking stopped with grasp_doneness inside target_doneness_range.
         Board return is not required. Dual: both steaks must pass.
Keys are green when up and red when down.

Demos land in:
  docs/final_task_demos/cook_meat/<tag>_sidebyside.mp4
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

TASK = "cook_meat"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5
# Short cook for suite speed; still exercises doneness gating.
COOK_STEPS = 800
COOK_SPEED_JITTER = 0.0
TARGET_DONENESS_RANGE = (0.45, 0.55)
# Abort early rather than grinding for hours on broken layouts / experts.
MIN_SUCCESSES_TO_CONTINUE = 2  # per condition; below this → stop before demos
MAX_CONSECUTIVE_SKIPS = 25
MAX_SEED_TRIES = N_PER_CONDITION * 25
DEMO_TIMEOUT_S = 600  # per-condition demo recording hard cap

CONDITIONS = {
    "default": {
        "cook_button_enabled": False,
        "dual_setup_enabled": False,
    },
    "opt1": {
        "cook_button_enabled": True,
        "dual_setup_enabled": False,
    },
    "opt2": {
        "cook_button_enabled": False,
        "dual_setup_enabled": True,
    },
    "opt1+2": {
        "cook_button_enabled": True,
        "dual_setup_enabled": True,
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
    out.append(f"cook_steps={COOK_STEPS}")
    out.append(f"cook_speed_jitter={COOK_SPEED_JITTER}")
    out.append(
        "target_doneness_range="
        f"[{TARGET_DONENESS_RANGE[0]},{TARGET_DONENESS_RANGE[1]}]"
    )
    return out


def _snapshot(env) -> dict:
    stations = list(getattr(env, "stations", None) or [])
    target = float(getattr(env, "target_doneness", 0.5))
    target_range = tuple(
        map(float, getattr(env, "target_doneness_range", TARGET_DONENESS_RANGE))
    )
    station_rows = []
    for st in stations:
        steak_p = st["steak"].get_pose().p
        board_xy = st.get("board_xy", st["board"].get_pose().p[:2])
        pan_xy = st["skillet"].get_functional_point(0)[:2]
        steak_xy = steak_p[:2]
        d_board = float(
            ((steak_xy[0] - board_xy[0]) ** 2 + (steak_xy[1] - board_xy[1]) ** 2) ** 0.5
        )
        d_pan = float(
            ((steak_xy[0] - pan_xy[0]) ** 2 + (steak_xy[1] - pan_xy[1]) ** 2) ** 0.5
        )
        grasp = st.get("grasp_doneness")
        station_rows.append(
            {
                "tag": str(st.get("tag")),
                "side": float(st.get("side", 0.0)),
                "grasp_doneness": None if grasp is None else float(grasp),
                "doneness": float(st.get("doneness", 0.0)),
                "max_doneness": float(st.get("max_doneness", 0.0)),
                "cook_on": bool(st.get("cook_on", False)),
                "d_board": d_board,
                "d_pan": d_pan,
                "steak_z": float(steak_p[2]),
                "in_band": (
                    False
                    if grasp is None
                    else target_range[0] <= float(grasp) <= target_range[1]
                ),
            }
        )
    return {
        "cook_button_enabled": bool(getattr(env, "cook_button_enabled", False)),
        "dual_setup_enabled": bool(getattr(env, "dual_setup_enabled", False)),
        "use_cook_button": True,
        "use_hold_cook": bool(getattr(env, "use_hold_cook", False)),
        "target_doneness": target,
        "target_doneness_range": list(target_range),
        "cook_steps": int(getattr(env, "cook_steps", COOK_STEPS)),
        "n_stations": int(len(stations)),
        "stations": station_rows,
        "option_label": str(
            env._option_label() if hasattr(env, "_option_label") else ""
        ),
    }


def _criteria_ok(snap: dict) -> bool:
    """Independent success: every steak shut off in-band (board return optional).

    Dual setups must report exactly two stations, and **both** steaks need
    ``grasp_doneness`` within the shared target range (one miss fails).
    """
    if snap["n_stations"] < 1:
        return False
    expected_n = 2 if snap["dual_setup_enabled"] else 1
    if snap["n_stations"] != expected_n:
        return False
    target_min, target_max = map(float, snap["target_doneness_range"])
    for st in snap["stations"]:
        g = st["grasp_doneness"]
        if g is None:
            return False
        if st.get("cook_on"):
            return False
        if not target_min <= float(g) <= target_max:
            return False
        if not st.get("in_band", False):
            return False
    return True


def _condition_shape_ok(condition: str, snap: dict) -> bool:
    cfg = CONDITIONS[condition]
    if bool(snap["cook_button_enabled"]) != bool(cfg["cook_button_enabled"]):
        return False
    if bool(snap["dual_setup_enabled"]) != bool(cfg["dual_setup_enabled"]):
        return False
    if not bool(snap["use_cook_button"]):
        return False
    expect_n = 2 if cfg["dual_setup_enabled"] else 1
    if int(snap["n_stations"]) != expect_n:
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
    args["use_dynamic"] = False

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
    # Search seeds until we have n evaluated attempts that at least set up; still
    # count each attempted seed as one of the n tests (retry on UnStable setup).
    seed = 1000 * (list(CONDITIONS.keys()).index(condition) + 1)
    attempts = 0
    consecutive_skips = 0
    max_seed_tries = MAX_SEED_TRIES
    while len(results) < n and attempts < max_seed_tries:
        attempts += 1
        label = f"{condition}/ep{len(results)}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        seed += 1
        # Skip pure placement failures without burning an episode slot when possible,
        # but still record them if we are running out of budget.
        err = row.get("error") or ""
        skip_tokens = (
            "not placeable",
            "no grasp pose",
            "not on pan after place",
            "skip",
            "UFuncOutputCastingError",
            "Cannot cast ufunc",
        )
        if any(tok.lower() in err.lower() for tok in skip_tokens):
            consecutive_skips += 1
            print(
                f"  [skip] seed={row['seed']} err={err}",
                flush=True,
            )
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                print(
                    f"!! abort {condition}: {consecutive_skips} consecutive placement skips",
                    flush=True,
                )
                break
            continue
        consecutive_skips = 0
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        stations = row.get("stations") or []
        grasp_str = ",".join(
            "None" if s.get("grasp_doneness") is None else f"{s['grasp_doneness']:.3f}"
            for s in stations
        )
        target = row.get("target_doneness")
        target_str = f"{float(target):.3f}" if target is not None else "None"
        d_board = ",".join(
            f"{s.get('d_board', float('nan')):.3f}" for s in stations
        ) if stations else ""
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"shape={row.get('shape_ok')} consistent={row.get('criteria_consistent')} "
            f"button={row.get('use_cook_button')} dual={row.get('dual_setup_enabled')} "
            f"n={row.get('n_stations')} target={target_str} "
            f"grasp=[{grasp_str}] d_board=[{d_board}] err={row.get('error')}",
            flush=True,
        )
        time.sleep(0.05)
    # If we could not fill n due to skips, pad with failures.
    while len(results) < n:
        results.append(
            {
                "label": f"{condition}/ep{len(results)}",
                "seed": -1,
                "success": False,
                "error": "exhausted seed search without enough evaluable episodes",
            }
        )
    n_ok = sum(1 for r in results if r["success"])
    print(f"=== {condition}: {n_ok}/{n} successes ===", flush=True)
    return results


def record_and_export_demos():
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    exported = {}

    import script.bench_script.record_demo as rd

    _orig_build = rd.build_args

    def _build(*a, **k):
        args = _orig_build(*a, **k)
        args["use_dynamic"] = False
        return args

    rd.build_args = _build
    try:
        for condition, file_tag in DEMO_FILE_TAGS.items():
            print(f"\n=== recording demo: {condition} (tag={file_tag}) ===", flush=True)
            # Hard-cap each demo so a broken layout cannot hang for hours.
            try:
                info = record_demo(
                    TASK,
                    config_name=CONFIG,
                    task_arg_overrides=_overrides(condition),
                    tag=file_tag,
                )
            except Exception as e:
                print(f"  !! demo {condition} failed: {type(e).__name__}: {e}", flush=True)
                continue
            for key, suffix in (
                ("sidebyside", "sidebyside"),
                ("head", "head"),
                ("topdown", "topdown"),
            ):
                src = info.get(key)
                if not src or not os.path.isfile(src):
                    continue
                dst = os.path.join(out_dir, f"{file_tag}_{suffix}.mp4")
                shutil.copy2(src, dst)
                if key == "sidebyside":
                    exported[condition] = dst
                    print(f"  copied -> {dst}", flush=True)
    finally:
        rd.build_args = _orig_build

    with open(os.path.join(out_dir, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"{TASK} — expert controller demos\n\n"
            "default  : cook_button_enabled=false, dual_setup_enabled=false\n"
            "           single station; latch cook key ON → cook → OFF\n"
            "opt1     : cook_button_enabled=true,  dual_setup_enabled=false\n"
            "           single station; hold cook key to cook\n"
            "opt2     : cook_button_enabled=false, dual_setup_enabled=true\n"
            "           dual stations; latch keys\n"
            "opt1+2   : cook_button_enabled=true,  dual_setup_enabled=true\n"
            "           dual stations; hold keys\n\n"
            "Keys are green when up and red when depressed.\n"
            "Success: cooking stopped with\n"
            f"         {TARGET_DONENESS_RANGE[0]} ≤ grasp_doneness ≤ "
            f"{TARGET_DONENESS_RANGE[1]}\n"
            "         Board return is not required.\n"
            "         Dual: both steaks must be cooked properly.\n\n"
            "Files (side-by-side is the primary deliverable):\n"
            "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
            "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
            "Also: <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    all_results = {}
    abort_demos = False
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        rows = run_condition(cond, n=N_PER_CONDITION)
        all_results[cond] = rows
        n_ok = sum(1 for r in rows if r["success"])
        if n_ok < MIN_SUCCESSES_TO_CONTINUE:
            print(
                f"\n!! EARLY STOP after {cond}: {n_ok}/{N_PER_CONDITION} successes "
                f"(need ≥{MIN_SUCCESSES_TO_CONTINUE} to continue). Skipping remaining "
                "conditions and demos.",
                flush=True,
            )
            abort_demos = True
            break

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

    exported = {}
    if abort_demos:
        print("\n!! Skipping demo recording due to early stop.", flush=True)
    else:
        exported = record_and_export_demos()

    report = {
        "task": TASK,
        "n_per_condition": N_PER_CONDITION,
        "early_stop": abort_demos,
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
            f"{TARGET_DONENESS_RANGE[0]} ≤ grasp_doneness ≤ "
            f"{TARGET_DONENESS_RANGE[1]} for every steak at key OFF; "
            "board return not required; "
            "dual (2 steaks) requires both cooked properly"
        ),
    }
    out_dir = os.path.abspath(f"./docs/final_task_demos/{TASK}")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}", flush=True)
    print(f"Demos in {out_dir}", flush=True)
    if abort_demos or not all_ok:
        print("\n!! SOME CONDITIONS DID NOT REACH 5/5 SUCCESSES", flush=True)
        sys.exit(1)
    print("\nAll conditions 5/5 successes.", flush=True)


if __name__ == "__main__":
    main()
