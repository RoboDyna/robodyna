#!/bin/bash
# Usage:
#   bash policy/xvla/eval.sh <task> <task_config> <ckpt> <tag> <seed> <gpu_id> [scenario] [mode]
# mode: frozen (default) | async
#
# Start the X-VLA inference server first -- see policy/xvla/README.md.

policy_name=xvla
task_name=${1}
task_config=${2}
ckpt=${3}
tag=${4}
seed=${5}
gpu_id=${6}
scenario=${7:-default}
mode=${8:-frozen}

if [ "${mode}" = "async" ]; then
    async_flag=True
elif [ "${mode}" = "frozen" ]; then
    async_flag=False
else
    echo "unknown mode '${mode}' (expected 'frozen' or 'async')" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33minference mode: ${mode}\033[0m"

cd "$(dirname "$0")/../.." # move to repo root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${tag} \
    --model_name ${ckpt} \
    --seed ${seed} \
    --scenario ${scenario} \
    --async_inference ${async_flag} \
    --policy_name ${policy_name}
