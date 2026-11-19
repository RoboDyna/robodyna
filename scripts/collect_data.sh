#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}
scenario=${4:-default}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

args=("$task_name" "$task_config" --scenario "$scenario")
if [ "$task_config" = "demo_dynamic" ]; then
    args+=(--production)
fi
PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py "${args[@]}"
rm -rf "data/${task_name}/${scenario}/.cache"
