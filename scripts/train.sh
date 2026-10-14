#!/bin/bash
# Train one method on one task.
#
#   usage: scripts/train.sh <task> <method> [gpus]
#     task   : grading | chem | sql | python
#     method : opsd | antisd | single | multi | prompt_only
#              opsd        - on-policy self-distillation, teacher sees the question and a correct solution c
#              antisd      - AntiSD baseline (recipe in scripts/antisd_recipe.sh)
#              single      - VRSD with one teacher-only system prompt d (prompts/single/<task>.txt)
#              multi       - VRSD with K aspect prompts d_1..d_K spread over the rollouts (prompts/aspects/<task>/)
#              prompt_only - control: teacher sees the question and the aspect prompts, but no solution c
#     gpus   : comma-separated device ids (default 0,1)
#
#   env: MODEL (default Qwen/Qwen3-4B; Qwen/Qwen3-8B and allenai/OLMo-3-7B-Instruct as in the paper), SCRATCH (short path for caches and the Ray temp dir, default /tmp/vrsd),
#        SPIDER_DB_DIR (default datasets/text2sql/spider/database), WANDB_MODE (default offline),
#        VAL_DIR (where per-step validation predictions are written, default outputs/<run>),
#        GMEM (vLLM memory fraction), WANDB_PROJECT, EXTRA_ARGS (extra Hydra overrides appended last).
#   Every validation improvement writes a full FSDP checkpoint (about 47 GB for Qwen3-4B) under
#   checkpoints/<dataset dir>/<run>; re-running the same cell resumes from it (trainer.resume_mode=auto).
set -euo pipefail
ulimit -u 1000000 2>/dev/null || true
TASK=${1:?task}; METHOD=${2:?method}; GPUS=${3:-0,1}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=${MODEL:-Qwen/Qwen3-4B}; MTAG=$(basename "$MODEL")
SCRATCH=${SCRATCH:-/tmp/vrsd}
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")

# ---- task table -------------------------------------------------------------------------------------------------
P="$ROOT/prompts"
case "$TASK" in
  grading) DATAREL=datasets/ordinal_scoring/asap;         RESP=4096
           ASPECTS="$P/aspects/grading/a1_rubric.txt,$P/aspects/grading/a2_calibration.txt,$P/aspects/grading/a3_evidence.txt,$P/aspects/grading/a4_counter_v2.txt" ;;
  chem)    DATAREL=datasets/sciknoweval/chemistry_all;    RESP=2048
           ASPECTS="$P/aspects/chem/c1_eliminate.txt,$P/aspects/chem/c2_mechanism.txt,$P/aspects/chem/c3_counter_v2.txt" ;;
  sql)     DATAREL=datasets/text2sql/spider;              RESP=2048
           ASPECTS="$P/aspects/sql/s1_schema.txt,$P/aspects/sql/s2_join.txt,$P/aspects/sql/s3_shape.txt,$P/aspects/sql/s4_verify.txt" ;;
  python)  DATAREL=datasets/code/lcb_v6;                  RESP=8192
           ASPECTS="$P/aspects/python/k_contract.txt,$P/aspects/python/k_minimal.txt,$P/aspects/python/k_boundary.txt,$P/aspects/python/k_trace.txt" ;;
  *) echo "task must be grading|chem|sql|python"; exit 1 ;;
esac
SINGLE="$P/single/$TASK.txt"
IMPL_ARG=""
case "$MODEL" in *OLMo-3*|*Olmo-3*)   # no native vLLM kernel for Olmo3: use the Transformers backend
  IMPL_ARG="+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=transformers"; GMEM=${GMEM:-0.4}
  case "$TASK" in chem|sql) RESP=4096 ;; esac ;;
esac
DATA="$ROOT/$DATAREL"
[ -f "$DATA/train.parquet" ] || { echo "missing $DATA/train.parquet (see README, Data)"; exit 1; }

# ---- training recipe (Appendix D) -------------------------------------------------------------------------------
if [ "$TASK" = python ]; then
  ALPHA=1.0; LR=1e-6; WARMUP=0; ENT=0;     NORM_ADV=False; TOPK=20;  MINI=1;  MAXLEN=18944; EPOCHS=30; TEST_FREQ=5; GMEM=${GMEM:-0.55}
else
  ALPHA=0.5; LR=1e-5; WARMUP=10; ENT=0.001; NORM_ADV=True; TOPK=100; MINI=32; MAXLEN=$((RESP + 4096)); EPOCHS=4; TEST_FREQ=20; GMEM=${GMEM:-0.7}
fi
TUR=0.01; TSP=null; MT=""; PRIV=sibling; LOSS=sdpo; SD_EXTRA=""
SD_ARGS="actor_rollout_ref.actor.self_distillation.alpha=$ALPHA actor_rollout_ref.actor.self_distillation.distillation_topk=$TOPK actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
case "$METHOD" in
  opsd)        ;;
  single)      TSP="$SINGLE" ;;
  multi)       TSP="${ASPECTS%%,*}"; MT=aspects ;;
  prompt_only) TSP="${ASPECTS%%,*}"; MT=aspects; PRIV=none ;;
  antisd)      source "$ROOT/scripts/antisd_recipe.sh"
               LOSS=grpo_ca; LR=1e-6; WARMUP=0; ENT=0; TUR=1.0; NORM_ADV=False; SD_ARGS=""
               SD_EXTRA="$ANTISD_SD_EXTRA $ANTISD_CCIR actor_rollout_ref.rollout.val_kwargs.response_length=$RESP" ;;
  *) echo "method must be opsd|antisd|single|multi|prompt_only"; exit 1 ;;
esac

RUN="${MTAG}-${TASK}-${METHOD}"
VAL_DIR=${VAL_DIR:-$ROOT/outputs/$RUN}; mkdir -p "$VAL_DIR" "$SCRATCH"
C="$SCRATCH/cache_${MTAG}_${TASK}_${METHOD}"; RT="$SCRATCH/rt_${MTAG:0:6}${TASK:0:2}${METHOD:0:2}"   # Ray socket paths must stay < 107 bytes
SPIDER_DB_DIR=${SPIDER_DB_DIR:-$ROOT/datasets/text2sql/spider/database}
[ "$TASK" = sql ] && [ ! -d "$SPIDER_DB_DIR" ] && { echo "missing Spider databases at $SPIDER_DB_DIR (see README, Data)"; exit 1; }

echo "[$RUN] gpus=$GPUS alpha=$ALPHA lr=$LR epochs=$EPOCHS test_freq=$TEST_FREQ -> $VAL_DIR"
CUDA_VISIBLE_DEVICES=$GPUS MODEL_PATH=$MODEL WANDB_MODE=${WANDB_MODE:-offline} \
XDG_CACHE_HOME=$C VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton \
PYTHONPATH=$ROOT SDPO_ROOT=$ROOT SDPO_EARLY_STOP=4 SD_SKIP_TRUNCATED=1 \
PRIVILEGED_SOURCE=$PRIV REFLECT_POSITION=after_privileged MULTI_TEACHER=$MT TEACHER_PROMPT_FILES="$( [ -n "$MT" ] && echo "$ASPECTS" )" \
SPIDER_DB_DIR=$SPIDER_DB_DIR HYDRA_FULL_ERROR=1 \
bash "$ROOT/training/verl_training.sh" "$RUN" sdpo "$DATAREL" \
  data.train_files=["$DATA/train.parquet"] data.val_files=["$DATA/test.parquet"] \
  data.train_batch_size=32 data.max_response_length=$RESP data.seed=42 \
  trainer.project_name=${WANDB_PROJECT:-vrsd} trainer.experiment_name=$RUN \
  trainer.total_epochs=$EPOCHS trainer.val_before_train=True trainer.test_freq=$TEST_FREQ \
  trainer.save_freq=0 trainer.resume_mode=auto trainer.n_gpus_per_node=$NGPU \
  trainer.validation_data_dir=$VAL_DIR \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.gpu_memory_utilization=$GMEM \
  actor_rollout_ref.rollout.max_model_len=$MAXLEN \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.val_kwargs.n=16 $IMPL_ARG \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
  actor_rollout_ref.actor.optim.lr=$LR actor_rollout_ref.actor.optim.lr_warmup_steps=$WARMUP \
  actor_rollout_ref.actor.entropy_coeff=$ENT \
  actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS actor_rollout_ref.actor.use_kl_loss=False \
  $SD_ARGS \
  actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TUR $SD_EXTRA \
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
  actor_rollout_ref.actor.self_distillation.teacher_system_prompt_file=$TSP \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  algorithm.norm_adv_by_std_in_grpo=$NORM_ADV \
  algorithm.rollout_correction.rollout_is=token \
  +ray_kwargs.ray_init._temp_dir=$RT ${EXTRA_ARGS:-}

python "$ROOT/analysis/eval_from_dump.py" --dump "$VAL_DIR" --out "$VAL_DIR/metrics.json"
