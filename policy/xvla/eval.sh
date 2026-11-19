#!/bin/bash
# Forwards to scripts/eval/run_xvla_eval.sh, which applies the evaluation protocol (fixed seeds and
# the per-task step limits in task_config/eval_step_limits.yml):
#
#   bash scripts/eval/run_xvla_eval.sh <task> [scenario] [gpu] [frozen|async]
#
# This script keeps the positional order of policy/pi05/eval.sh. ckpt, tag and seed are fixed by
# the evaluation protocol and ignored:
#   bash policy/xvla/eval.sh <task> <task_config> <ckpt> <tag> <seed> <gpu_id> [scenario] [mode]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$REPO/scripts/eval/run_xvla_eval.sh" "${1:?task}" "${7:-default}" "${6:-0}" "${8:-frozen}"
