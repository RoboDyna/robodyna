#!/bin/bash
# One FastWAM rollout evaluation: <task> [scenario] [gpu] [mode].
# Same ten-seed suite, step limits and frozen/async accounting as run_pi05_eval.sh and
# run_xvla_eval.sh.
#
# Split deployment, like X-VLA. One FastWAM process needs about 28 GiB (5B video DiT + action
# expert + VAE + activations), so by default the model runs on its own GPU in
# policy/fwam/fwam_server.py and this script drives a simulator that talks to it over HTTP. The
# server skips the text encoder (it feeds the cached training prompt embeddings instead) and leaves
# rendering to the simulator process, which keeps it within a 16 GiB card.
#
# The server is not started here. Start it once per model GPU, in the Fast-WAM environment, and
# export FASTWAM_SERVER:
#   python policy/fwam/fwam_server.py --root "$FASTWAM_ROOT" --run "$FASTWAM_RUN" \
#       --port 8020 --step 13170
#   export FASTWAM_SERVER=<model-host>:8020
#
# With FASTWAM_SERVER unset the model is loaded in-process, which needs a card with >= 32 GiB.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

TASK=${1:?usage: run_fastwam_eval.sh <task> [scenario] [gpu] [mode]}
SCENARIO=${2:-default}
GPU=${3:-0}
MODE=${4:-${EVAL_MODE:-frozen}}
case "$MODE" in
  frozen|async) ;;
  *) echo "unknown eval mode '$MODE' (expected 'frozen' or 'async')" >&2; exit 2 ;;
esac

FASTWAM_SERVER=${FASTWAM_SERVER:-}
# Fast-WAM checkout and training run (relative to FASTWAM_ROOT). Placeholders: set both.
FASTWAM_ROOT=${FASTWAM_ROOT:-$REPO/third_party/fastwam}
FASTWAM_RUN=${FASTWAM_RUN:-runs/<fastwam_run>}
# The final checkpoint of the RoboDyna run, reported as `checkpoint_id` in the results; the paper's
# FastWAM numbers use this step.
FASTWAM_STEP=${FASTWAM_STEP:-13170}
# Actions executed per inference: 24, not the adapter yml's 8. The predicted 32-step chunk is
# back-loaded: per-step joint deltas stay at a ~0.004 rad noise floor for indices 0-18 and only
# ramp to ~0.15 rad/step after ~20. Executing only the first 8 commands the current pose, and each
# replan resets the chunk to that unchanged pose, so the arm barely moves. Upstream's RoboTwin eval
# uses 24 (`configs/sim_robotwin.yaml`). At RoboDyna's 16.67 Hz control rate, 24 steps commit to
# 1.44 s of open-loop motion. See policy/fwam/README.md.
FASTWAM_REPLAN=${FASTWAM_REPLAN:-24}

# Simulator-side interpreter. In split mode it needs sapien + mplib + curobo + scipy + requests but
# not torch, since the weights live in the server's environment. Override with SIM_PYTHON.
SIM_PYTHON=${SIM_PYTHON:-$REPO/.venv/bin/python}
[ -x "$SIM_PYTHON" ] || { echo "no simulator interpreter at $SIM_PYTHON" >&2; exit 2; }
export PATH="$(dirname "$SIM_PYTHON"):$PATH"
command -v ffmpeg >/dev/null || { echo "no ffmpeg on PATH (needed for rollout video)" >&2; exit 2; }

if [ -n "$FASTWAM_SERVER" ]; then
  export FASTWAM_SERVER
  echo "[fwam] remote inference at $FASTWAM_SERVER"
else
  echo "[fwam] WARNING: FASTWAM_SERVER unset; loading the model in-process (needs ~28 GiB)" >&2
  # FastWAM's loader resolves `checkpoints/<model_id>` against the process cwd, and the cwd must
  # stay at the repo root for ./task_config/..., so a symlink serves both.
  if [ ! -e checkpoints ]; then
    ln -s "$FASTWAM_ROOT/checkpoints" checkpoints
  fi
fi

# Same cap as pi0.5 and X-VLA: 1.2x the task's longest published expert episode.
STEP_LIMIT=$("$SIM_PYTHON" -c "
import sys, yaml
limits = yaml.safe_load(open('task_config/eval_step_limits.yml'))['step_limit']
task = sys.argv[1]
if task not in limits:
    sys.exit(f'no step_limit for {task} in task_config/eval_step_limits.yml')
print(limits[task])
" "$TASK")

# The NVIDIA Vulkan ICD file lives in different places depending on the driver package. Without it
# SAPIEN prints "failed to find a rendering device" and exits 0, which would record a run with no
# episodes, so fail here instead.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for icd in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; do
    [ -f "$icd" ] && { export VK_ICD_FILENAMES=$icd; break; }
  done
  [ -n "${VK_ICD_FILENAMES:-}" ] || { echo "no nvidia Vulkan ICD found" >&2; exit 2; }
fi
unset DISPLAY
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning
# Nothing is downloaded at inference time; offline mode also keeps a stalled Hugging Face lookup
# from looking like a slow first inference.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# See run_pi05_eval.sh: an expert failure on a validated seed does not abort the run. The expert
# pass still runs, because it generates the dynamic object's trajectory.
export EVAL_EXPERT_GATE=${EVAL_EXPERT_GATE:-0}

exec "$SIM_PYTHON" -u script/eval_policy.py \
  --config policy/fwam/deploy_policy.yml --overrides \
  --task_name "$TASK" \
  --task_config demo_dynamic \
  --model_name FastWAM-RoboDynaV2 \
  --ckpt_setting fastwam_robodynav2 \
  --checkpoint_id "$FASTWAM_STEP" \
  --fastwam_root "$FASTWAM_ROOT" \
  --fastwam_run_dir "$FASTWAM_RUN" \
  --replan_steps "$FASTWAM_REPLAN" \
  --eval_step_limit "$STEP_LIMIT" \
  --seed 0 \
  --scenario "$SCENARIO" \
  --eval_mode_name "$MODE" \
  --policy_name fwam
