#!/usr/bin/env python3
"""Full controller test + tagged demos for pack_fruits.

Conditions (5 episodes each):
  default : two_colors_enabled=false, distractor_enabled=false
  opt1    : two_colors_enabled=true,  distractor_enabled=false
  opt2    : two_colors_enabled=false, distractor_enabled=true (≥1 black)
  opt1+2  : two_colors_enabled=true,  distractor_enabled=true

Success: every colored apple rests in its color-matched basket
  (red/apple → left, green → right). Black distractors are ignored.

Demos land in:
  docs/final_task_demos/pack_fruits/<tag>_sidebyside.mp4
  with tags: default, opt1, opt2, opt1+2
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args, record_demo
from script.collect_data import class_decorator

TASK = "pack_fruits"
CONFIG = "demo_dynamic"
N_PER_CONDITION = 5

# Shared knobs for every condition (match demo_dynamic pack_fruits section).
_BASE = {
    "n_items_min": 3,
    "n_items_max": 5,
    "belt_speed_jitter": 0.20,
    "distractor_extra_prob": 0.3,
    "distractor_min_gap_mult": 2.0,
}

CONDITIONS = {
    "default": {
        **_BASE,
        "two_colors_enabled": False,
        "distractor_enabled": False,
    },
    "opt1": {
        **_BASE,
        "two_colors_enabled": True,
        "distractor_enabled": False,
    },
    "opt2": {
        **_BASE,
        "two_colors_enabled": False,
        "distractor_enabled": True,
    },
    "opt1+2": {
        **_BASE,
        "two_colors_enabled": True,
        "distractor_enabled": True,
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
    n = int(getattr(env, "n_items", 0))
    types = list(getattr(env, "item_types", []))
    sides = list(getattr(env, "item_sides", []))
    in_ok = []
    positions = []
    for i in range(n):
        in_ok.append(bool(env._fruit_in_basket(i)))
        positions.append([float(x) for x in env.items[i].get_pose().p])

    # Independent correct-basket check: apple/red → left, green → right.
    half_xy = getattr(env, "basket_half_xy", {})
    centers = getattr(env, "basket_centers", {})
    base_z = getattr(env, "basket_base_z", {})
    independent_ok = []
    for i in range(n):
        ftype = types[i]
        p = np.array(positions[i], dtype=np.float64)
        c = np.array(centers[ftype], dtype=np.float64)
        half_x, half_y = half_xy.get(ftype, (0.078, 0.111))
        d = np.abs(p[:2] - c)
        in_xy = bool(d[0] <= half_x and d[1] <= half_y)
        above = p[2] >= (float(base_z[ftype]) - 0.02)
        below = p[2] <= (float(base_z[ftype]) + 0.18)
        independent_ok.append(bool(in_xy and above and below))

    n_dist_slots = int(getattr(env, "n_distractor_slots", 0))
    dist_y = list(getattr(env, "_distractor_y", []))
    n_dist_active = int(sum(1 for y in dist_y if y is not None))

    return {
        "spawn_mode": str(getattr(env, "spawn_mode", "")),
        "two_colors_enabled": bool(getattr(env, "two_colors_enabled", False)),
        "distractor_enabled": bool(getattr(env, "distractor_enabled", False)),
        "distractor_color": [float(x) for x in list(getattr(env, "distractor_color", []))[:3]],
        "n_items": n,
        "n_apple": int(getattr(env, "n_apple", 0)),
        "n_green": int(getattr(env, "n_green", 0)),
        "n_orange": int(getattr(env, "n_orange", getattr(env, "n_green", 0))),
        "active_colors": list(getattr(env, "active_colors", [])),
        "item_types": types,
        "item_sides": sides,
        "item_in_correct_basket": in_ok,
        "independent_in_correct_basket": independent_ok,
        "positions": positions,
        "n_distractor_slots": n_dist_slots,
        "n_distractor_plan": int(getattr(env, "n_distractor_plan", n_dist_slots)),
        "n_distractor_active": n_dist_active,
        "n_missed": int(sum(1 for m in getattr(env, "_missed", []) if m)),
        "n_packed": int(sum(1 for p in getattr(env, "_packed", []) if p)),
    }


def _criteria_ok(snap: dict) -> bool:
    """Independent success check matching the task contract.

    - every colored apple in its color-matched basket
      (apple/red → left, green → right)
    - black distractors are ignored (not required for success)
    """
    if snap["n_items"] <= 0:
        return False
    if len(snap["independent_in_correct_basket"]) != snap["n_items"]:
        return False
    if not all(snap["independent_in_correct_basket"]):
        return False
    for t in snap["item_types"]:
        if t not in ("apple", "green"):
            return False
    return True


def _condition_shape_ok(condition: str, snap: dict) -> bool:
    """Sanity-check that the sampled episode matches the condition knobs."""
    cfg = CONDITIONS[condition]
    if bool(snap["two_colors_enabled"]) != bool(cfg["two_colors_enabled"]):
        return False
    if bool(snap["distractor_enabled"]) != bool(cfg["distractor_enabled"]):
        return False
    if snap["spawn_mode"] != "single":
        return False
    if not (3 <= snap["n_items"] <= 5):
        return False
    if cfg["two_colors_enabled"]:
        if set(snap["active_colors"]) != {"apple", "green"}:
            return False
        if snap["n_apple"] < 1 or snap["n_green"] < 1:
            return False
    else:
        if len(snap["active_colors"]) != 1:
            return False
        if snap["active_colors"][0] not in ("apple", "green"):
            return False
    if cfg["distractor_enabled"]:
        if snap["n_distractor_slots"] < 1:
            return False
        col = snap["distractor_color"]
        if len(col) < 3 or max(col) > 0.2:
            return False
    else:
        if snap["n_distractor_slots"] != 0:
            return False
    return True


def _cuda_cleanup():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _run_episode(
    task_args_overrides: list[str],
    seed: int,
    label: str,
    condition: str,
    max_oom_retries: int = 4,
) -> dict:
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
    for attempt in range(max_oom_retries + 1):
        _cuda_cleanup()
        env = class_decorator(TASK)
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
            row["error"] = None
            return row
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            is_oom = "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
            if is_oom and attempt < max_oom_retries:
                print(
                    f"  [OOM retry {attempt + 1}/{max_oom_retries}] seed={seed} — waiting…",
                    flush=True,
                )
                traceback.print_exc()
                time.sleep(8.0 * (attempt + 1))
                continue
            traceback.print_exc()
            return row
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
            _cuda_cleanup()
    return row


def run_condition(condition: str, n: int = N_PER_CONDITION, max_attempts: int = 20):
    """Run until ``n`` successes (or ``max_attempts`` tries). Retries expert/OOM fails."""
    overrides = _overrides(condition)
    results = []
    successes = []
    base = 1000 * (list(CONDITIONS.keys()).index(condition) + 1)
    print(
        f"\n=== {condition}: need {n} successes (max {max_attempts}) | overrides={overrides} ===",
        flush=True,
    )
    attempt = 0
    while len(successes) < n and attempt < max_attempts:
        seed = base + attempt
        label = f"{condition}/ep{attempt}"
        row = _run_episode(overrides, seed=seed, label=label, condition=condition)
        results.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"  [{status}] seed={row['seed']} plan={row.get('plan_success')} "
            f"check={row.get('check_success')} criteria={row.get('criteria_ok')} "
            f"shape={row.get('shape_ok')} consistent={row.get('criteria_consistent')} "
            f"mode={row.get('spawn_mode')} distractor={row.get('distractor_enabled')} "
            f"in_basket={row.get('item_in_correct_basket')} "
            f"types={row.get('item_types')} sides={row.get('item_sides')} "
            f"err={row.get('error')}",
            flush=True,
        )
        if row["success"]:
            successes.append(row)
        attempt += 1
        time.sleep(0.05)
    n_ok = len(successes)
    print(f"=== {condition}: {n_ok}/{n} successes ({attempt} attempts) ===", flush=True)
    return results, successes


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
            "default  : spawn_mode=parallel, distractor_enabled=false\n"
            "           always apple+orange pair waves; no black distractors\n"
            "opt1     : spawn_mode=random, pair_stagger_enabled=true,\n"
            "           single_wave_any_belt=true, distractor_enabled=false\n"
            "           each wave is single OR pair; singles on either belt;\n"
            "           pair Y gap ~ U(0, fruit_diameter)\n"
            "opt2     : spawn_mode=parallel, distractor_enabled=true\n"
            "           pair waves + black distractor fruit (ignored for success)\n"
            "opt1+2   : spawn_mode=random (+ stagger / any-belt) +\n"
            "           distractor_enabled=true\n\n"
            "Success: every real fruit in its color-matched basket\n"
            "         (red/apple → left, yellow/orange → right).\n"
            "         Black distractors are ignored (never packed/counted).\n\n"
            "Files (side-by-side is the primary deliverable):\n"
            "  default_sidebyside.mp4   opt1_sidebyside.mp4\n"
            "  opt2_sidebyside.mp4      opt1+2_sidebyside.mp4\n"
            "Also: <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    return exported


def main():
    all_results = {}
    all_successes = {}
    for cond in ("default", "opt1", "opt2", "opt1+2"):
        rows, oks = run_condition(cond, n=N_PER_CONDITION)
        all_results[cond] = rows
        all_successes[cond] = oks

    summary = {}
    print("\n========== SUMMARY ==========", flush=True)
    all_ok = True
    for key, rows in all_results.items():
        n_ok = len(all_successes[key])
        n = len(rows)
        n_consistent = sum(1 for r in rows if r.get("criteria_consistent"))
        summary[key] = {
            "successes": n_ok,
            "attempts": n,
            "rate": n_ok / max(n, 1),
            "criteria_consistent": n_consistent,
            "success_seeds": [r["seed"] for r in all_successes[key]],
            "config": {
                k: v for k, v in CONDITIONS[key].items() if k != "distractor_color"
            },
        }
        print(
            f"  {key}: {n_ok}/{N_PER_CONDITION} successes "
            f"({n} attempts, criteria_consistent={n_consistent}/{n})",
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
                "episodes": [
                    {
                        kk: vv
                        for kk, vv in row.items()
                        if kk != "positions"  # keep report smaller
                    }
                    for row in all_results[k]
                ],
            }
            for k in all_results
        },
        "demos": exported,
        "success_criterion": (
            "all real fruits in color-matched baskets "
            "(apple/red → left, orange/yellow → right); "
            "black distractors ignored"
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
        print("\nFAIL: not all conditions reached full success count", flush=True)
        sys.exit(1)
    print("\nPASS: all conditions reached target successes with criteria consistent", flush=True)


if __name__ == "__main__":
    main()
