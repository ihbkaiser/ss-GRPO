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
TRAIN_DATA_FRACTION="${TRAIN_DATA_FRACTION:-0.4}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"

cd "$WORKSPACE"

if [[ -z "${DRAFT_ADAPTER:-}" ]]; then
  if [[ -d "$DRAFT_SAVED_MODEL_DIR" ]]; then
    DRAFT_ADAPTER="$(find "$DRAFT_SAVED_MODEL_DIR" -maxdepth 1 -type f -name 'step*.pth' | sort -V | tail -n 1 || true)"
  else
    DRAFT_ADAPTER=""
  fi
fi

if [[ ! -f "$DRAFT_ADAPTER" ]]; then
  echo "Cannot find FastGRPO draft adapter: ${DRAFT_ADAPTER:-<empty>}" >&2
  echo "Run: bash fastgrpo/train_draft.sh" >&2
  echo "or pass: DRAFT_ADAPTER=/path/to/stepXXXX.pth bash fastgrpo/train_fastgrpo.sh" >&2
  exit 1
fi

mkdir -p \
  "$(dirname "$FASTGRPO_LOG_FILE")" \
  "$FASTGRPO_SAVED_MODEL_DIR" \
  "$FASTGRPO_SAVED_DRAFT_MODEL_DIR" \
  "$FASTGRPO_SAVED_STATISTICS_DIR"

"$PYTHON" "$SCRIPT_DIR/grpo_speculative.py" \
  --model_dir "$MODEL" \
  --adapter_path "$DRAFT_ADAPTER" \
  --dtype "$MODEL_DTYPE" \
  --attn_implementation "$ATTN_IMPLEMENTATION" \
  --load_lora_path "$LOAD_LORA_PATH" \
  --model_type "$MODEL_TYPE" \
  --train_option "$TRAIN_OPTION" \
  --train_data_fraction "$TRAIN_DATA_FRACTION" \
  --train_subset_seed "$TRAIN_SUBSET_SEED" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --version_name "$FASTGRPO_EXP" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --sample_num "$SAMPLE_NUM" \
  --accumulation_steps "$ACCUMULATION_STEPS" \
  --draft_accumulation_steps "$DRAFT_ACCUMULATION_STEPS" \
  --target_lr "$TARGET_LR" \
  --draft_lr "$FASTGRPO_DRAFT_LR" \
  --is_train_draft "$IS_TRAIN_DRAFT" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --max_length "$GEN_MAX_LENGTH" \
  --max_training_padding_gap "$MAX_TRAINING_PADDING_GAP" \
  --max_training_token "$MAX_TRAINING_TOKEN" \
  --grpo_iteration_num "$GRPO_ITERATION_NUM" \
  --repeated_generate_nums "$REPEATED_GENERATE_NUMS" \
  --beta "$BETA" \
  --epsilon "$EPSILON" \
  --verification_capacity "$VERIFICATION_CAPACITY" \
  --max_draft_token_length "$MAX_DRAFT_TOKEN_LENGTH" \
  --max_draft_k "$MAX_DRAFT_K" \
  --max_verification_num "$MAX_VERIFICATION_NUM" \
  --min_draft_token_length "$MIN_DRAFT_TOKEN_LENGTH" \
  --draft_token_length_c "$DRAFT_TOKEN_LENGTH_C" \
  --num_workers "$NUM_WORKERS" \
  --persistent_workers "$PERSISTENT_WORKERS" \
  --log_file "$FASTGRPO_LOG_FILE" \
  --saved_model_dir "$FASTGRPO_SAVED_MODEL_DIR" \
  --saved_draft_model_dir "$FASTGRPO_SAVED_DRAFT_MODEL_DIR" \
  --saved_statistics_dir "$FASTGRPO_SAVED_STATISTICS_DIR"
