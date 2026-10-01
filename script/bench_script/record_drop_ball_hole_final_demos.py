#!/usr/bin/env python3
"""Record distinct cylindrical rotating-shape-sorter demos for all conditions."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (  # noqa: E402
    _encode_mp4,
    build_args,
    configure_topdown_camera,
    load_pkl_file,
)
from script.collect_data import class_decorator  # noqa: E402


TASK = "drop_ball_hole"
CONDITIONS = {
    "default": {"stick_to_surface": False, "add_dummy_hole": False},
    "opt1": {"stick_to_surface": True, "add_dummy_hole": False},
    "opt2": {"stick_to_surface": False, "add_dummy_hole": True},
    "opt1+2": {"stick_to_surface": True, "add_dummy_hole": True},
}


def _overrides(values: dict[str, bool]) -> list[str]:
    return [f"{key}={str(value).lower()}" for key, value in values.items()]


def _encode_available_views(cache: Path, out_dir: Path, tag: str, fps: float) -> None:
    """Encode observer frames even when this task omits head-camera frames."""
    frame_paths = sorted(cache.glob("*.pkl"), key=lambda path: int(path.stem))
    top_frames = []
    for path in frame_paths:
        frame = load_pkl_file(str(path)).get("third_view_rgb")
        if frame is not None:
            top_frames.append(np.asarray(frame))
    if not top_frames:
        raise RuntimeError(f"No observer frames were saved for {tag}")
    top = np.stack(top_frames)
    # The task's rollout does not persist a head-camera stream. Duplicate the
    # recorded top-down view so gallery assets retain the side-by-side format.
    side = np.concatenate([top, top], axis=2)
    _encode_mp4(top, str(out_dir / f"{tag}_topdown.mp4"), fps)
    _encode_mp4(side, str(out_dir / f"{tag}_sidebyside.mp4"), fps)


def record_condition(tag: str, values: dict[str, bool]) -> None:
    scratch = Path("/tmp") / f"{TASK}_{tag}"
    shutil.rmtree(scratch, ignore_errors=True)
    args = build_args(TASK, "demo_dynamic", str(scratch), None, _overrides(values))
    args.update(render_freq=0, save_data=True, need_plan=True, now_ep_num=0, seed=0)

    env = class_decorator(TASK)
    original_setup = env.setup_demo

    def setup_with_topdown(**kwargs):
        original_setup(**kwargs)
        configure_topdown_camera(env)

    env.setup_demo = setup_with_topdown
    try:
        env.setup_demo(**args)
        env.play_once()
        if not (env.plan_success and env.check_success()):
            raise RuntimeError(f"{tag} did not complete successfully")
        cache = scratch / ".cache" / "episode0"
        out_dir = Path("docs/final_task_demos") / TASK
        _encode_available_views(cache, out_dir, tag, 250.0 / float(args["save_freq"]))
        print(f"Recorded {tag}: {out_dir / f'{tag}_sidebyside.mp4'}")
    finally:
        env.close_env(clear_cache=True)
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    for tag, values in CONDITIONS.items():
        record_condition(tag, values)


if __name__ == "__main__":
    main()
