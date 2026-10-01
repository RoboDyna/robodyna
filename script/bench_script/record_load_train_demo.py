"""Record a third-person demo for load_train -> ./tmp/tmp_load_train/video/vN_episode0.mp4."""
import sys
import os
import glob
import re
import shutil

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import yaml
import numpy as np

from script.collect_data import class_decorator, get_embodiment_config, run
from envs import CONFIGS_PATH
from envs.utils.pkl2hdf5 import load_pkl_file
from envs.utils.images_to_video import images_to_video

TASK_NAME = "load_train"
CONFIG_NAME = "_load_train_smoke"
SAVE_ROOT = os.path.abspath("./tmp/tmp_load_train")


def next_version(video_dir: str) -> int:
    os.makedirs(video_dir, exist_ok=True)
    nums = []
    for p in glob.glob(os.path.join(video_dir, "v*_episode0.mp4")):
        m = re.search(r"v(\d+)_", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def merge_third_view_video(cache_dir, out_mp4, fps=16.67):
    pkls = sorted(
        glob.glob(os.path.join(cache_dir, "*.pkl")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    frames = []
    for p in pkls:
        data = load_pkl_file(p)
        rgb = data.get("third_view_rgb", None)
        if rgb is None:
            continue
        frames.append(np.asarray(rgb))
    if not frames:
        raise RuntimeError(f"No third_view frames in {cache_dir}")
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    # Prefer sharper encode for demo MP4s (images_to_video defaults to CRF 23).
    stack = np.stack(frames, axis=0)
    n_frames, H, W, C = stack.shape
    import subprocess
    pixel_format = "rgb24"
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", pixel_format,
            "-video_size", f"{W}x{H}", "-framerate", str(fps),
            "-i", "-",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264",
            "-crf", "17", "-preset", "slow",
            out_mp4,
        ],
        stdin=subprocess.PIPE,
    )
    proc.stdin.write(stack.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        # Fallback to shared helper if custom ffmpeg encode fails.
        images_to_video(stack, out_path=out_mp4, fps=fps, is_rgb=True)
    else:
        print(f"Wrote {out_mp4} ({n_frames} frames at {W}×{H}, crf=17)")


def build_args():
    with open(f"./task_config/{CONFIG_NAME}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = TASK_NAME
    args["task_config"] = CONFIG_NAME
    args["episode_num"] = 1
    args["save_path"] = SAVE_ROOT
    args["collect_data"] = True
    args["eval_video_log"] = True
    args["save_failed_cases"] = False
    args["use_seed"] = False  # plan fresh so traj matches current expert
    args["check_render_success"] = True
    args["export_lerobot"] = False
    args.setdefault("data_type", {})
    args["data_type"]["third_view"] = True
    args["data_type"]["rgb"] = True

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return emb[t]["file_path"]

    et = args["embodiment"]
    args["left_robot_file"] = emb_file(et[0])
    args["right_robot_file"] = emb_file(et[1])
    args["embodiment_dis"] = et[2]
    args["dual_arm_embodied"] = False
    args["embodiment_name"] = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def main():
    args = build_args()
    video_dir = os.path.join(SAVE_ROOT, "video")
    ver = next_version(video_dir)
    out_mp4 = os.path.join(video_dir, f"v{ver}_episode0.mp4")

    # Clean prior nested scratch but keep older videos.
    for junk in ("data", ".cache", "seed.txt", "scene_info.json", TASK_NAME, "_traj_data", "seed_archive.txt"):
        p = os.path.join(SAVE_ROOT, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)

    os.makedirs(SAVE_ROOT, exist_ok=True)

    task = class_decorator(TASK_NAME)

    def _merge_third(self):
        if not self.save_data:
            return
        cache_path = f"{self.save_dir}/.cache/episode{self.ep_num}/"
        fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
        merge_third_view_video(cache_path, out_mp4, fps=fps)
        # Also leave a conventional episode0.mp4 symlink/copy for convenience.
        conv = f"{self.save_dir}/video/episode{self.ep_num}.mp4"
        os.makedirs(os.path.dirname(conv), exist_ok=True)
        if os.path.exists(out_mp4):
            shutil.copy2(out_mp4, conv)

    task.merge_pkl_to_hdf5_video = _merge_third.__get__(task, task.__class__)

    run(task, args)

    # Cleanup scratch left by run(); keep video/
    for junk in (".cache", "data", "seed.txt", "scene_info.json"):
        p = os.path.join(SAVE_ROOT, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)

    print("Demo ready:", out_mp4)


if __name__ == "__main__":
    main()
