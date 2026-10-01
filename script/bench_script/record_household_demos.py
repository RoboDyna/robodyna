#!/usr/bin/env python3
"""10-seed household expert sweep + success/failure dual-view demos + gallery publish.

For each household task:
  1. Run N expert seeds (plan-only).
  2. Record dual-view demos:
       - if any failure: 1 success + 1 failure
       - else: 2 successes (prefer distinct seeds)
  3. Copy side-by-side MP4s to household_demos/ and task_gallery/, make GIFs.
  4. Write /tmp/hh_gallery_meta.json for index.html / README updates.
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

from _record_layout_seeds import record_one  # noqa: E402
from sweep_household_success import TASKS, run_seed  # noqa: E402

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
HOUSEHOLD_DEMOS = ROOT / "household_demos"
GALLERY = Path(
    os.environ.get(
        "TASK_GALLERY_ROOT",
        str(Path(__file__).resolve().parents[2] / "docs/final_task_demos"),
    )
)
REPO_FINAL = ROOT / "docs/final_task_demos"

TASK_BLURBS = {
    "trap_bug": "Trap a scurrying cockroach, spider, or ant under the office bookshelf with a transparent glass box.",
    "boil_milk": "Turn the stove on so milk rises in the pot, then shut it off before the milk overflows.",
    "fill_coffee_jar": "Press the coffee dispenser's lid to fill a marked glass jar to the target fill line.",
    "pour_beer": "Hold the draft-tap button to pour beer into a mug; foam ramps while held; click the finish bell to score. Overflow fails.",
    "cook_food": "Drop meat, sausage, or onion into a pre-lit pan, then shut the stove off at the target doneness (food stays in the pan).",
    "cook_food_timer": "Same as cook_food, with a pie timer that advances while the stove is on and freezes when it is off.",
    "measure_ingredient": "Push a marked jar under an oil nozzle and fill it to the target ring; oil that misses the jar fails.",
    "make_soup": "Tip chopping-board vegetables into a pot of water on an already-lit stove without dropping any pieces.",
    "catch_cup": "Push a pillow under a tipping mug so it lands softly instead of hitting the table.",
    "catch_mouse_object_drop": "Place a pillow-lined basket under a shelf object knocked by a scurrying mouse so it does not hit the table.",
    "stop_ball": "Block a table-tennis ball that falls from the shelf and rolls toward the near table edge.",
    "clean_table": "Wipe a spreading coffee spill with a sponge before it reaches a laptop on the opposite side.",
}


def gif_from_mp4(mp4: Path, gif: Path, scale: int = 720, fps: int = 8) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4),
            "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos",
            "-loop", "0", str(gif),
        ],
        check=True,
    )


def sweep_task(task: str, n: int) -> dict:
    ok, fail = [], []
    for seed in range(n):
        row = run_seed(task, seed)
        tag = "OK" if row["ok"] else "FAIL"
        extra = f" err={row['err']}" if row["err"] else f" plan={row['plan']} check={row['check']}"
        print(f"  seed={seed} {tag}{extra}", flush=True)
        (ok if row["ok"] else fail).append(seed)
    return {"ok": ok, "fail": fail}


def pick_demo_seeds(result: dict) -> list[tuple[str, int]]:
    """Return [(label, seed), ...] — label is success/failure/success1/success2."""
    ok, fail = result["ok"], result["fail"]
    if fail and ok:
        return [("success", ok[0]), ("failure", fail[0])]
    if fail and not ok:
        # Still record a failure; second slot another failure if available else skip.
        picks = [("failure", fail[0])]
        if len(fail) > 1:
            picks.append(("failure2", fail[1]))
        return picks
    # No failures → two successes
    if len(ok) >= 2:
        return [("success1", ok[0]), ("success2", ok[1])]
    if len(ok) == 1:
        return [("success1", ok[0]), ("success2", ok[0])]
    return []


def publish_clip(task: str, label: str, head_mp4: Path, caption: str) -> dict:
    """Copy head-camera mp4 + gif into household_demos, repo final, and gallery."""
    stem = {
        "success": "success_head",
        "failure": "failure_head",
        "failure2": "failure2_head",
        "success1": "success1_head",
        "success2": "success2_head",
    }.get(label, f"{label}_head")

    # household_demos layout
    sub = "failure" if label.startswith("fail") else "success"
    hh_dir = HOUSEHOLD_DEMOS / task / sub
    hh_dir.mkdir(parents=True, exist_ok=True)
    hh_mp4 = hh_dir / f"{stem}.mp4"
    shutil.copy2(head_mp4, hh_mp4)

    # Also keep named seed copy under household_demos root-style for inspection
    seed_name = head_mp4.name
    shutil.copy2(head_mp4, HOUSEHOLD_DEMOS / task / seed_name)

    for dest_root in (REPO_FINAL / task, GALLERY / task):
        dest_root.mkdir(parents=True, exist_ok=True)
        dst_mp4 = dest_root / f"{stem}.mp4"
        dst_gif = dest_root / f"{stem}.gif"
        shutil.copy2(head_mp4, dst_mp4)
        gif_from_mp4(head_mp4, dst_gif)
        print(f"  PUBLISH {dst_gif}", flush=True)

    display_label = "Failure case" if label.startswith("fail") else "Success"
    return {
        "file": f"{stem}.gif",
        "label": display_label,
        "caption": caption,
        "mp4": str(hh_mp4),
    }


def caption_for(task: str, label: str, seed: int) -> str:
    if label.startswith("fail"):
        return f"Failure case (seed {seed})."
    if label in ("success1", "success2"):
        return f"Success rollout (seed {seed})."
    return f"Success (seed {seed})."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--record-only", action="store_true",
                    help="Skip sweep; read results JSON")
    ap.add_argument("--results", default="/tmp/hh_sweep_results.json")
    ap.add_argument("--skip-record", action="store_true")
    ns = ap.parse_args()

    results_path = Path(ns.results)
    if ns.record_only:
        results = json.loads(results_path.read_text())
    else:
        results = {}
        for task in ns.tasks:
            print(f"\n===== SWEEP {task} ({ns.n} seeds) =====", flush=True)
            results[task] = sweep_task(task, ns.n)
            print(
                f"  → {len(results[task]['ok'])}/{ns.n} ok "
                f"ok={results[task]['ok']} fail={results[task]['fail']}",
                flush=True,
            )
        results_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {results_path}", flush=True)

    if ns.sweep_only or ns.skip_record:
        return 0

    gallery_meta = []
    for task in ns.tasks:
        res = results[task]
        picks = pick_demo_seeds(res)
        print(f"\n===== RECORD {task} picks={picks} =====", flush=True)
        if not picks:
            print(f"  SKIP {task}: no runnable seeds", flush=True)
            continue
        published = []
        for label, seed in picks:
            try:
                # save_freq=15 keeps demos short enough to finish under GPU load;
                # DEMO_MAX_STEPS caps runaway kitchen experts (boil/cook).
                os.environ.setdefault("DEMO_MAX_STEPS", "1600")
                out = record_one(task, seed, config="demo_dynamic", save_freq=15)
                side = Path(out["sidebyside"])
                if not side.is_file():
                    print(f"  WARN missing sidebyside for {task} seed={seed}", flush=True)
                    continue
                # Prefer actual outcome from recording if it disagrees
                actual = "success" if out["ok"] else "failure"
                use_label = label
                if label.startswith("success") and actual == "failure":
                    use_label = "failure"
                elif label.startswith("fail") and actual == "success":
                    use_label = "success1" if not any(
                        p["label"] == "Success" for p in published
                    ) else "success2"
                meta = publish_clip(task, use_label, side, caption_for(task, use_label, seed))
                published.append(meta)
            except Exception as exc:  # noqa: BLE001
                print(f"  RECORD FAIL {task} seed={seed}: {exc}", flush=True)
                traceback.print_exc()

        # Normalize to gallery convention: success+failure OR success1+success2
        files, labels, captions = [], [], []
        has_fail = any(m["label"].startswith("Failure") for m in published)
        if has_fail:
            succs = [m for m in published if m["label"] == "Success"]
            fails = [m for m in published if m["label"].startswith("Failure")]
            ordered = (succs[:1] + fails[:1]) if succs else fails[:2]
        else:
            ordered = published[:2]
            # Rename files to success1/success2 if needed
            renamed = []
            for i, m in enumerate(ordered):
                want = f"success{i+1}_head.gif"
                if m["file"] != want:
                    for root in (GALLERY / task, REPO_FINAL / task):
                        src = root / m["file"]
                        dst = root / want
                        if src.is_file():
                            shutil.copy2(src, dst)
                            mp4_src = root / m["file"].replace(".gif", ".mp4")
                            mp4_dst = root / want.replace(".gif", ".mp4")
                            if mp4_src.is_file():
                                shutil.copy2(mp4_src, mp4_dst)
                    m = {**m, "file": want, "label": "Success"}
                renamed.append(m)
            ordered = renamed

        for m in ordered:
            files.append(m["file"])
            labels.append(m["label"])
            captions.append(m["caption"])

        gallery_meta.append({
            "task": task,
            "blurb": TASK_BLURBS.get(task, task),
            "files": files,
            "labels": labels,
            "captions": captions,
            "sweep": res,
        })

    meta_path = Path("/tmp/hh_gallery_meta.json")
    meta_path.write_text(json.dumps(gallery_meta, indent=2))
    print(f"\nWrote {meta_path}", flush=True)
    print("\n========== SWEEP SUMMARY ==========", flush=True)
    for task, res in results.items():
        n_ok = len(res["ok"])
        print(f"{task:26s}  {n_ok:2d}/{ns.n}  ok={res['ok']}  fail={res['fail']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
