#!/bin/bash
# One X-VLA rollout evaluation: <task> [scenario] [gpu] [mode].
# Same ten-seed suite, step limits and frozen/async accounting as run_pi05_eval.sh. The model runs
# in a separate process (the X-VLA FastAPI server) that this script reaches over HTTP.
#
# The server is not started here. Start it once per machine and export XVLA_PORT if it differs
# from 8010:
#   cd /path/to/X-VLA && python -m deploy \
#       --model_path RoboDyna/X-VLA-RoboDyna --host 127.0.0.1 --port 8010
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"

TASK=${1:?usage: run_xvla_eval.sh <task> [scenario] [gpu] [mode]}
SCENARIO=${2:-default}
GPU=${3:-0}
MODE=${4:-${EVAL_MODE:-frozen}}
case "$MODE" in
  frozen|async) ;;
  *) echo "unknown eval mode '$MODE' (expected 'frozen' or 'async')" >&2; exit 2 ;;
esac

XVLA_HOST=${XVLA_HOST:-127.0.0.1}
XVLA_PORT=${XVLA_PORT:-8010}
# Soft-prompt slot the RoboDyna checkpoint was fine-tuned under. 6 is RoboTwin2's slot and would
# condition the model on the wrong domain embedding.
XVLA_DOMAIN_ID=${XVLA_DOMAIN_ID:-19}
XVLA_PROPRIO=${XVLA_PROPRIO:-measured}

# Simulator-side interpreter. It needs sapien + mplib + curobo + scipy + requests, not torch: the
# model lives in the server's own environment. Defaults to the pi05 interpreter, which has all of
# these; override with SIM_PYTHON.
SIM_PYTHON=${SIM_PYTHON:-${PI05_PYTHON:-$REPO/policy/pi05/.venv/bin/python}}
[ -x "$SIM_PYTHON" ] || { echo "no simulator interpreter at $SIM_PYTHON" >&2; exit 2; }
export PATH="$(dirname "$SIM_PYTHON"):$PATH"
command -v ffmpeg >/dev/null || { echo "no ffmpeg on PATH (needed for rollout video)" >&2; exit 2; }

# Same cap as pi0.5: 1.2x the task's longest published expert episode, so the policies are
# compared under the same step budget.
STEP_LIMIT=$("$SIM_PYTHON" -c "
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
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning
# See run_pi05_eval.sh: an expert failure on a validated seed does not abort the run. The expert
# pass still runs, because it generates the dynamic object's trajectory.
export EVAL_EXPERT_GATE=${EVAL_EXPERT_GATE:-0}

# Fail fast if the server is not up, instead of ten episodes of connection errors.
"$SIM_PYTHON" -c "
import sys, requests
url = 'http://%s:%s/act' % (sys.argv[1], sys.argv[2])
try:
    requests.post(url, json={}, timeout=10)
except requests.exceptions.ConnectionError as e:
    sys.exit('X-VLA server unreachable at %s (%s)' % (url, e))
except Exception:
    pass  # a 4xx/5xx on an empty body means something is listening, which is all we check here
" "$XVLA_HOST" "$XVLA_PORT"

exec "$SIM_PYTHON" -u script/eval_policy.py \
  --config policy/xvla/deploy_policy.yml --overrides \
  --task_name "$TASK" \
  --task_config demo_dynamic \
  --model_name X-VLA-RoboDyna \
  --ckpt_setting xvla_30k \
  --checkpoint_id 30000 \
  --host "$XVLA_HOST" \
  --port "$XVLA_PORT" \
  --domain_id "$XVLA_DOMAIN_ID" \
  --proprio_source "$XVLA_PROPRIO" \
  --eval_step_limit "$STEP_LIMIT" \
  --seed 0 \
  --scenario "$SCENARIO" \
  --eval_mode_name "$MODE" \
  --policy_name xvla
