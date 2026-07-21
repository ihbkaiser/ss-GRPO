#!/usr/bin/env bash

# Sourced by pretrain_<model>.sh after model-specific defaults are defined.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-flashgrpo_b200/configs/${MODEL_KEY}/pretrain.yaml}"
PRETRAIN_DATASET="${PRETRAIN_DATASET:-sharegpt}"
DATA_PATH="${DATA_PATH:-$ROOT/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
CONVERSATION_FORMAT="${CONVERSATION_FORMAT:-sharegpt}"
PRETRAIN_DATA_FRACTION="${PRETRAIN_DATA_FRACTION:-1.0}"
PRETRAIN_SUBSET_SEED="${PRETRAIN_SUBSET_SEED:-42}"
PRETRAIN_NUM_SAMPLES="${PRETRAIN_NUM_SAMPLES:-0}"

DATASET_SLUG="$(printf '%s' "$PRETRAIN_DATASET" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
DATASET_SLUG="${DATASET_SLUG%_}"
FRACTION_TAG="${PRETRAIN_DATA_FRACTION//./p}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${EXP:-${MODEL_KEY}_pretrain_${DATASET_SLUG}_f${FRACTION_TAG}_${RUN_TAG}}}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-8}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
PRETRAIN_MAX_SEQ_LEN="${PRETRAIN_MAX_SEQ_LEN:-1024}"
PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-500}"
PRETRAIN_WARMUP_STEPS="${PRETRAIN_WARMUP_STEPS:-100}"
PRETRAIN_NUM_WORKERS="${PRETRAIN_NUM_WORKERS:-8}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-32}"
NUM_MEDUSA_HEADS="${NUM_MEDUSA_HEADS:-3}"
MODEL_DTYPE="${MODEL_DTYPE:-bf16}"
HEAD_DTYPE="${HEAD_DTYPE:-fp32}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

OUT="${OUT:-outputs/pretrain/${MODEL_KEY}}"
LOG_DIR="${LOG_DIR:-logs/flashgrpo_b200/${MODEL_KEY}/pretrain/${RUN_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cmd=(
  "$PYTHON_BIN" flashgrpo_b200/scripts/pretrain_medusa_heads.py
  --config "$CONFIG"
  --model_name_or_path "$MODEL"
  --dataset_name "$PRETRAIN_DATASET"
  --dataset_path "$DATA_PATH"
  --conversation_format "$CONVERSATION_FORMAT"
  --dataset_fraction "$PRETRAIN_DATA_FRACTION"
  --seed "$PRETRAIN_SUBSET_SEED"
  --num_samples "$PRETRAIN_NUM_SAMPLES"
  --output_dir "$OUT"
  --log_dir "$LOG_DIR"
  --num_train_epochs "$PRETRAIN_EPOCHS"
  --batch_size "$PRETRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$PRETRAIN_GRAD_ACCUM"
  --learning_rate "$PRETRAIN_LR"
  --max_seq_len "$PRETRAIN_MAX_SEQ_LEN"
  --save_steps "$PRETRAIN_SAVE_STEPS"
  --warmup_steps "$PRETRAIN_WARMUP_STEPS"
  --num_workers "$PRETRAIN_NUM_WORKERS"
  --loss_chunk_size "$LOSS_CHUNK_SIZE"
  --num_medusa_heads "$NUM_MEDUSA_HEADS"
  --dtype "$MODEL_DTYPE"
  --head_dtype "$HEAD_DTYPE"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --chain_loss_weight 0.0
)
if (($#)); then
  cmd+=("$@")
fi

printf 'Run name : %s\nModel    : %s\nDataset  : %s (fraction=%s)\nConfig   : %s\nLogs     : %s\nCheckpoint: %s\n' \
  "$RUN_NAME" "$MODEL" "$PRETRAIN_DATASET" "$PRETRAIN_DATA_FRACTION" "$CONFIG" "$LOG_DIR" "$OUT"
printf 'Command  :'; printf ' %q' "${cmd[@]}"; printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "Model config not found: $MODEL/config.json" >&2; exit 2; }
[[ -f "$DATA_PATH" ]] || { echo "Pretrain data not found: $DATA_PATH" >&2; exit 2; }
mkdir -p "$OUT" "$LOG_DIR"
"${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/console.log"

