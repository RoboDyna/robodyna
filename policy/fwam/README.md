# FastWAM on RoboDyna

FastWAM is a world-action model: a Wan2.2-TI2V-5B video diffusion backbone with an action DiT head.
It predicts a short video of what is about to happen and reads a 14-D joint-space action chunk off the
same latents. The checkpoint used in the paper was trained on the RoboDyna LeRobot exports, so the
state/action space, camera set and image geometry match the benchmark directly.

## Requirements

- A Fast-WAM checkout (set `FASTWAM_ROOT`, default `./third_party/fastwam`) and its Python environment.
  The simulator and the model share one process and one GPU; there is no separate inference server.
- About 29 GiB of GPU memory as shipped. Skipping the T5 text encoder (precomputed prompt embeddings,
  `load_text_encoder=False`) brings this down to roughly 18 GiB.

## Running one combination

From the repository root, in the Fast-WAM environment:

```bash
python script/eval_policy.py --config policy/fwam/deploy_policy.yml \
    --overrides --task_name catch_ramp_ball --task_config demo_dynamic \
    --ckpt_setting fwam --seed 0 --scenario default \
    --async_inference False --policy_name fwam    # True for the asynchronous protocol
```

Evaluation uses the same banked seeds, step limits and episode journals as π0.5 and X-VLA.

Environment variables: `FASTWAM_ROOT`, `FASTWAM_RUN`, `FASTWAM_STEP` (checkpoint step),
`FASTWAM_REPLAN` (actions executed per inference, default 24), `FASTWAM_TRAIN_PROMPT`,
`FASTWAM_DEBUG_ACTIONS`, and `EVAL_SEEDS_OVERRIDE`.

## What the adapter does

`deploy_policy.py` is a thin shim over Fast-WAM's RoboTwin policy:

1. `ckpt_setting` stays a short label; weights are addressed by `checkpoint_id`.
2. The Hydra task group is pinned to `robodyna_uncond_3cam_384_1e-4`, so the RoboDyna normalizer and
   frame geometry are used (the RoboTwin defaults load without error but evaluate incorrectly).
3. The working directory stays at the repository root; a `checkpoints` symlink lets Fast-WAM find the
   Wan2.2 base weights.
4. Chunk inference is exposed as `get_action`, so asynchronous evaluation charges FastWAM's inference
   latency to the simulated world like the other policies.

The package is named `fwam` because Fast-WAM's own library is `fastwam`.

## Notes

- **`replan_steps`.** The predicted chunk is back-loaded: the first ~19 of 32 steps barely move the arm.
  Executing fewer than ~20 actions per inference therefore stalls the robot. The paper uses 24
  (Fast-WAM's RoboTwin default). RoboDyna runs at 16.7 Hz, so 24 steps commit to 1.44 s of open-loop
  motion.
- **Prompts.** Fast-WAM looks up text embeddings by the exact prompt string. `train_prompts.json` maps
  each task to the wording used in training; set `FASTWAM_TRAIN_PROMPT=0` to use the current
  instructions in `task_config/task_instructions.json` instead.
- **Inference cost.** About 2.5 s per chunk on an H200 with 10 denoising steps (VAE image encoding is
  the largest share).
