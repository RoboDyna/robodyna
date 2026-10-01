#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
scenario=${7:-default}
mode=${8:-frozen}   # frozen (default) | async

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

# source .venv/bin/activate
cd ../.. # move to root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --scenario ${scenario} \
    --async_inference ${async_flag} \
    --policy_name ${policy_name} 
