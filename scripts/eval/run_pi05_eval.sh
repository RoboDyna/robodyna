#!/bin/bash
# One pi0.5 rollout evaluation: <task> [scenario] [gpu] [checkpoint_id] [mode].
# Runs the fixed ten-seed suite from task_config/eval_seeds.yml (script/evaluation_protocol.py).
# Model and simulator share one interpreter, which needs sapien + mplib + curobo alongside
# jax/openpi (by default policy/pi05/.venv).
set -euo pipefail
# The repo is derived from this script's location, so a second checkout runs its own code instead
# of importing another checkout's through PYTHONPATH.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

TASK=${1:?usage: run_pi05_eval.sh <task> [scenario] [gpu] [checkpoint_id] [mode]}
SCENARIO=${2:-default}
GPU=${3:-0}
CKPT=${4:-25000}
# frozen = the synchronous protocol: the world holds still while the policy thinks, so inference
#          latency does not affect the score.
# async  = the measured inference wall clock is charged to simulation time before the action lands,
#          as on a real robot.
MODE=${5:-${EVAL_MODE:-frozen}}
case "$MODE" in
  frozen|async) ;;
  *) echo "unknown eval mode '$MODE' (expected 'frozen' or 'async')" >&2; exit 2 ;;
esac

# Interpreter carrying sapien + curobo + jax. Override with PI05_PYTHON.
PI05_PYTHON=${PI05_PYTHON:-$REPO/policy/pi05/.venv/bin/python}
[ -x "$PI05_PYTHON" ] || { echo "no pi05 interpreter at $PI05_PYTHON" >&2; exit 2; }
# eval_policy.py calls a bare "ffmpeg" to encode rollout videos, so the interpreter's bin directory
# goes on PATH in case ffmpeg is installed there.
export PATH="$(dirname "$PI05_PYTHON"):$PATH"
command -v ffmpeg >/dev/null || { echo "no ffmpeg on PATH (needed for rollout video)" >&2; exit 2; }

# Cap a rollout at 1.2x the task's longest published expert episode rather than at
# demo_dynamic.yml's collection-side limit; see task_config/eval_step_limits.yml.
# The pi05 interpreter is used because the system python3 may not have yaml.
STEP_LIMIT=$("$PI05_PYTHON" -c "
import sys, yaml
limits = yaml.safe_load(open('task_config/eval_step_limits.yml'))['step_limit']
task = sys.argv[1]
if task not in limits:
    sys.exit(f'no step_limit for {task} in task_config/eval_step_limits.yml')
print(limits[task])
" "$TASK")

export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}
unset DISPLAY
export CUDA_VISIBLE_DEVICES=$GPU
# openpi is an editable install whose .pth may point at another checkout. .pth paths come after
# PYTHONPATH in sys.path, so naming this repo's src here keeps the whole run in one tree.
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:$REPO/policy/pi05/src:$REPO/policy/pi05/packages/openpi-client/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning

# XLA settings used for the paper runs. Triton GEMM is disabled because ptxas can crash with it on
# Ada GPUs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
# Feed the model the verbatim training prompt rather than a sampled paraphrase.
export EXACT_PROMPT=1
# A fixed seed that the expert fails to re-solve does not abort the ten-episode run. Every seed in
# task_config/eval_seeds.yml was validated with three consecutive expert solves and the expert is
# nondeterministic, so such a failure is expert noise. The expert pass still runs, because it
# generates the moving object's trajectory for dynamic tasks; only its verdict is dropped.
export EVAL_EXPERT_GATE=${EVAL_EXPERT_GATE:-0}

exec "$PI05_PYTHON" -u script/eval_policy.py \
  --config policy/pi05/deploy_policy.yml --overrides \
  --task_name "$TASK" \
  --task_config demo_dynamic \
  --train_config_name pi05_robodyna \
  --model_name baseline_bs256_30k \
  --ckpt_setting baseline_bs256_30k \
  --checkpoint_id "$CKPT" \
  --eval_step_limit "$STEP_LIMIT" \
  --seed 0 \
  --scenario "$SCENARIO" \
  --eval_mode_name "$MODE" \
  --policy_name pi05
