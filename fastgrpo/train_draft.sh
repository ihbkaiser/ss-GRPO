#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_ENV="${CONFIG_ENV:-$SCRIPT_DIR/configs/b200.env}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$SCRIPT_DIR:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTHON="${PYTHON:-python3}"

if [[ -f "$CONFIG_ENV" ]]; then
  # shellcheck source=/dev/null
  source "$CONFIG_ENV"
fi

cd "$WORKSPACE"
if [[ "${RESUME_CHECKPOINT:-}" == "auto" ]]; then
  if [[ -d "$DRAFT_CHECKPOINT_DIR" ]]; then
    RESUME_CHECKPOINT="$(find "$DRAFT_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step*.pt' | sort -V | tail -n 1 || true)"
  else
    RESUME_CHECKPOINT=""
  fi
fi

mkdir -p "$DRAFT_LOG_DIR" "$DRAFT_SAVED_MODEL_DIR" "$DRAFT_CHECKPOINT_DIR"

"$PYTHON" "$SCRIPT_DIR/train_draft.py" \
  --model_dir "$MODEL" \
  --version_name "$DRAFT_EXP" \
  --model_type "$MODEL_TYPE" \
  --dtype "$MODEL_DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --batch_size "$DRAFT_PRETRAIN_BATCH_SIZE" \
  --num_epochs "$DRAFT_PRETRAIN_EPOCHS" \
  --lr "$DRAFT_PRETRAIN_LR" \
  --accumulation_steps "$DRAFT_PRETRAIN_ACCUMULATION_STEPS" \
  --warmup_ratio "$DRAFT_PRETRAIN_WARMUP_RATIO" \
  --sample_num "$SAMPLE_NUM" \
  --max_seq_len "$DRAFT_PRETRAIN_MAX_SEQ_LEN" \
  --num_workers "$NUM_WORKERS" \
  --persistent_workers "$PERSISTENT_WORKERS" \
  --log_dir "$DRAFT_LOG_DIR" \
  --saved_model_dir "$DRAFT_SAVED_MODEL_DIR" \
  --dataset_dir "$DRAFT_DATASET" \
  --checkpoint_dir "$DRAFT_CHECKPOINT_DIR" \
  --save_checkpoint_steps "$DRAFT_SAVE_CHECKPOINT_STEPS" \
  --keep_last_checkpoints "$KEEP_LAST_CHECKPOINTS" \
  --resume_checkpoint "$RESUME_CHECKPOINT"
