#!/usr/bin/env python3
"""Search upward from a seed until a task condition has N successful evaluation seeds.

Every attempted seed is appended to JSONL immediately and successful seeds are written to a
plain text file. Re-running the same command resumes after the highest attempted seed.
No trajectory, HDF5, video, or LeRobot data is saved by the underlying sweep runners.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sweep_base_success import SCENARIOS, TASKS as CONCEPTUAL_TASKS
from sweep_base_success import run_seed as run_conceptual_seed
from sweep_household_success import TASKS as HOUSEHOLD_TASKS
from sweep_household_success import run_seed as run_household_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--family", required=True, choices=("conceptual", "household"))
    ap.add_argument("--scenario", default="base", choices=("base", *SCENARIOS))
    ap.add_argument("--start-seed", type=int, default=10000)
    ap.add_argument("--target", type=int, default=10)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    if args.family == "conceptual":
        if args.task not in CONCEPTUAL_TASKS:
            raise SystemExit(f"unknown conceptual task: {args.task}")
        scenario = "default" if args.scenario == "base" else args.scenario
    else:
        if args.task not in HOUSEHOLD_TASKS:
            raise SystemExit(f"unknown household task: {args.task}")
        scenario = "base"

    out_dir = Path(args.output_dir) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scenario.replace("+", "plus")
    attempts_path = out_dir / f"{stem}.attempts.jsonl"
    seeds_path = out_dir / f"{stem}.seeds.txt"

    attempted: set[int] = set()
    successful: list[int] = []
    if attempts_path.exists():
        for line in attempts_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seed = int(row["seed"])
            attempted.add(seed)
            if row.get("ok"):
                successful.append(seed)
    successful = list(dict.fromkeys(successful))
    next_seed = max([args.start_seed - 1, *attempted]) + 1

    def write_successes() -> None:
        tmp = seeds_path.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(map(str, successful[: args.target])) + "\n")
        os.replace(tmp, seeds_path)

    write_successes()
    print(f"[seed-search] resume attempted={len(attempted)} valid={len(successful)}/{args.target} "
          f"next_seed={next_seed}", flush=True)
    with attempts_path.open("a", buffering=1) as attempts:
        while len(successful) < args.target:
            seed = next_seed
            next_seed += 1
            if seed in attempted:
                continue
            if args.family == "conceptual":
                row = run_conceptual_seed(args.task, seed, scenario=scenario)
            else:
                row = run_household_seed(args.task, seed)
                row["scenario"] = scenario
            attempts.write(json.dumps(row, sort_keys=True) + "\n")
            attempts.flush()
            os.fsync(attempts.fileno())
            attempted.add(seed)
            if row["ok"]:
                successful.append(seed)
                write_successes()
            print(f"[seed-search] seed={seed} ok={row['ok']} valid={len(successful)}/{args.target}",
                  flush=True)

    print(f"[seed-search] COMPLETE task={args.task} scenario={scenario} seeds={successful[:args.target]}",
          flush=True)


if __name__ == "__main__":
    main()
