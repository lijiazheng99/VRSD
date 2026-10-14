#!/bin/bash
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
# export RAY_DEBUG=1
ulimit -c 0

# Only pin a team entity if the caller sets WANDB_ENTITY; otherwise use the
# logged-in user's default entity. Set WANDB_MODE=offline to skip upload.
if [ -n "${WANDB_ENTITY:-}" ]; then export WANDB_ENTITY; fi
export EXPERIMENT=${1:-"experiment"}
CONFIG_NAME=${2:-"ppo_trainer"}
export TASK=${3:-"datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"}

# removes the first three arguments from the command line
if [ "$#" -ge 3 ]; then
    shift 3
else
    echo "Usage: $0 <experiment_name> <config_name> <data_path>"
    echo "Example: $0 test ppo_trainer datasets/ttcs/lasgroup_verifiable-corpus_math-ai_math500_1000"
    exit 1
fi

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task: $TASK"
echo "Arguments: $@"

python -m verl.trainer.main_ppo --config-name $CONFIG_NAME "$@"
