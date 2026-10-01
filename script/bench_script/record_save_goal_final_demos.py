#!/usr/bin/env python3
"""Record all save_goal condition demos into docs/final_task_demos/save_goal/."""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import record_demo

OUT = os.path.abspath("./docs/final_task_demos/save_goal")
CONDS = {
    "default": ["players_enabled=false", "cover_enabled=false"],
    "opt1": ["players_enabled=true", "cover_enabled=false"],
    "opt2": ["players_enabled=false", "cover_enabled=true"],
    "opt1+2": ["players_enabled=true", "cover_enabled=true"],
}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    exported = {}
    for tag, overrides in CONDS.items():
        print(f"\n=== recording {tag} ===", flush=True)
        info = record_demo(
            "save_goal",
            config_name="demo_dynamic",
            task_arg_overrides=overrides,
            tag=tag,
        )
        for key, suffix in (("sidebyside", "sidebyside"), ("head", "head"), ("topdown", "topdown")):
            dst = os.path.join(OUT, f"{tag}_{suffix}.mp4")
            shutil.copy2(info[key], dst)
            if key == "sidebyside":
                exported[tag] = dst
                print(f"copied -> {dst}", flush=True)

    with open(os.path.join(OUT, "CONDITIONS.txt"), "w", encoding="utf-8") as f:
        f.write(
            "save_goal — expert controller demos (post no-teleport fix)\n\n"
            "default  : players_enabled=false, cover_enabled=false\n"
            "opt1     : players_enabled=true,  cover_enabled=false\n"
            "opt2     : players_enabled=false, cover_enabled=true\n"
            "opt1+2   : players_enabled=true,  cover_enabled=true\n\n"
            "Success: keeper stays where dropped; front-face save; ball does not enter goal.\n"
            "Files: <tag>_sidebyside.mp4, <tag>_head.mp4, <tag>_topdown.mp4\n"
            "Tags: default | opt1 | opt2 | opt1+2\n"
        )
    print("DONE", exported, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
