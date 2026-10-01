# X-VLA policy adapter for RoboDyna

Drives [`RoboDyna/X-VLA-RoboDyna`](https://huggingface.co/RoboDyna/X-VLA-RoboDyna)
(X-VLA-0.9B fine-tuned on RoboDyna) through the standard `script/eval_policy.py`
harness, so it is scored identically to `policy/pi05`.

## 1. Start the inference server

Runs in its own environment — X-VLA needs `transformers`, which conflicts with
the simulator venv.

```bash
cd /path/to/X-VLA
../.venv/bin/python -m deploy \
    --model_path RoboDyna/X-VLA-RoboDyna \
    --host 0.0.0.0 --port 8010
```

`deploy.py` lives at the X-VLA repo root and imports `models.*`, so it must be
launched from that directory — pointing the venv at it from elsewhere fails.

## 2. Run the evaluation client

```bash
# bash policy/xvla/eval.sh <task> <task_config> <ckpt> <tag> <seed> <gpu_id> [scenario] [mode]

bash policy/xvla/eval.sh catch_ramp_ball demo_dynamic \
     RoboDyna/X-VLA-RoboDyna xvla30k 0 0 default frozen
bash policy/xvla/eval.sh catch_ramp_ball demo_dynamic \
     RoboDyna/X-VLA-RoboDyna xvla30k 0 0 default async
```

## Conventions — get these wrong and it fails silently

`xvla_codec.py` owns the mapping and is round-trip tested against real dataset
frames (`rotation err 1.7e-14 deg`). Three things differ from X-VLA's own
`evaluation/robotwin-2.0/client.py`, which is **not** correct for this simulator:

| | official X-VLA client | RoboDyna (this adapter) |
| --- | --- | --- |
| quaternion | xyzw (scipy default) | **wxyz** |
| gripper action | `1 - 2*(g > 0.7)` ∈ {−1, +1} | **[0, 1], 1 = open** |
| `domain_id` | 6 (RoboTwin2 slot) | **19** |

- **wxyz.** The sim takes ee actions through `_trans_from_gripper_to_endlink`
  → `t3d.quaternions.quat2mat` → `sapien.Pose`, all scalar-first. Verified
  against the dataset's own recorded wrist-camera extrinsics: `R_cam(t)ᵀ @
  R_endpose(t)` must be constant, and is — to 0.013° under wxyz versus 4.64°
  under xyzw. Feeding xyzw produces a **177°** orientation error, i.e. a
  flipped end-effector, not a subtle offset.
- **Gripper.** The sim interpolates the gripper path from `get_gripper_val()`,
  which is normalised with 1 = open; ±1 is out of range. X-VLA emits 1 = closed
  (post-sigmoid), so the adapter inverts it.
- **rot6d is interleaved** — `c1 = v[0::2]`, `c2 = v[1::2]`. Slicing `[0:3]`/
  `[3:6]` silently yields a different rotation (~110° off).

Camera order is **(head, left, right)**, matching the model's training
`CAMERA_VIEW`. `policy/pi05` uses a different order for its own model; do not
copy it across.

## Inference modes

`frozen` reproduces the historical pipeline: `scene.step()` only runs inside
`take_action`, so the world is stationary while the policy thinks and inference
latency cannot affect the score.

`async` charges that latency to the world — measured wall-clock time is
converted to physics steps at `async_sim_hz` (250 Hz) and applied before the
action lands, with the arms holding their last setpoint. **Scores therefore
depend on GPU and model speed**, so both policies must be run on the same
hardware for a comparison to mean anything.
