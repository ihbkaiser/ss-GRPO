#!/usr/bin/env bash

# Sourced by train_<model>.sh after model-specific defaults are defined.
set -euo pipefail

: "${MODEL_KEY:?MODEL_KEY is required}"
: "${MODEL:?MODEL is required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

METHOD="${METHOD:-hrdcr}"
DATASET="${DATASET:-gsm8k}"
TRAIN_DATA_FRACTION="${TRAIN_DATA_FRACTION:-0.4}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"

case "${METHOD,,}" in
  hrdcr|reflex)
    METHOD="hrdcr"
    DEFAULT_CONFIG="flashgrpo_b200/configs/${MODEL_KEY}/train_hrdcr.yaml"
    ;;
  medusa_only|medusa-only|no_reflex|no-reflex)
    METHOD="medusa_only"
    DEFAULT_CONFIG="flashgrpo_b200/configs/${MODEL_KEY}/train_medusa_only.yaml"
    ;;
  *)
    echo "Unsupported METHOD=$METHOD (use hrdcr or medusa_only)" >&2
    exit 2
    ;;
esac
CONFIG="${CONFIG:-$DEFAULT_CONFIG}"

case "${DATASET,,}" in
  simplelr|simplerl|simplelr_abel|simplelr_abel_level3to5)
    TRAIN_OPTION="simplelr_abel_level3to5"
    DATASET_SLUG="simplelr_abel_l3to5"
    ;;
  simplelr_qwen|simplelr_qwen_level3to5)
    TRAIN_OPTION="simplelr_qwen_level3to5"
    DATASET_SLUG="simplelr_qwen_l3to5"
    ;;
  gsm8k)
    TRAIN_OPTION="gsm8k"
    DATASET_SLUG="gsm8k"
    ;;
  dapo|dapo-math|dapo_math)
    TRAIN_OPTION="DAPO-math"
    DATASET_SLUG="dapo_math"
    ;;
  *)
    TRAIN_OPTION="$DATASET"
    DATASET_SLUG="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_')"
    DATASET_SLUG="${DATASET_SLUG%_}"
    ;;
esac

FRACTION_TAG="${TRAIN_DATA_FRACTION//./p}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-${EXP:-${MODEL_KEY}_${METHOD}_${DATASET_SLUG}_f${FRACTION_TAG}_${RUN_TAG}}}"

HEADS="${HEADS:-outputs/pretrain/${MODEL_KEY}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SAMPLE_NUM="${SAMPLE_NUM:-100}"
TARGET_LR="${TARGET_LR:-1e-6}"
MEDUSA_LR="${MEDUSA_LR:-1e-5}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_TRAINING_TOKEN="${MAX_TRAINING_TOKEN:-8192}"
MAX_TRAINING_PADDING_GAP="${MAX_TRAINING_PADDING_GAP:-1024}"
LOGPS_CHUNK_SIZE="${LOGPS_CHUNK_SIZE:-512}"
REPEATED_GENERATE_NUMS="${REPEATED_GENERATE_NUMS:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-10}"
CPEAK_NODES="${CPEAK_NODES:-128}"
AUTO_TUNE_CPEAK="${AUTO_TUNE_CPEAK:-false}"
MAX_TREE_NODES_PER_SEQ="${MAX_TREE_NODES_PER_SEQ:-10}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
APPEND_LOG="${APPEND_LOG:-false}"

START_EPOCH="${START_EPOCH:-0}"
START_BATCH="${START_BATCH:-0}"
START_USED_ITEMS="${START_USED_ITEMS:-0}"
START_ROLLOUT_COUNT="${START_ROLLOUT_COUNT:-0}"
LOAD_LORA_PATH="${LOAD_LORA_PATH:-}"

LOG_DIR="${LOG_DIR:-logs/flashgrpo_b200/${MODEL_KEY}/train/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/${MODEL_KEY}/${RUN_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cmd=(
  "$PYTHON_BIN" flashgrpo_b200/scripts/train_flashgrpo_b200.py
  --config "$CONFIG"
  --set "run_name=$RUN_NAME"
  --set "model.model_dir=$MODEL"
  --set "model.attn_implementation=$ATTN_IMPLEMENTATION"
  --set "flashgrpo.medusa_heads_checkpoint=$HEADS"
  --set "aux_head_checkpoint=$HEADS"
  --set "flashgrpo.medusa_lr=$MEDUSA_LR"
  --set "data.train_option=$TRAIN_OPTION"
  --set "generation.temperature=$TEMPERATURE"
  --set "generation.top_p=$TOP_P"
  --set "generation.max_length=$MAX_LENGTH"
  --set "generation.max_prompt_length=$MAX_PROMPT_LENGTH"
  --set "generation.repeated_generate_nums=$REPEATED_GENERATE_NUMS"
  --set "training.batch_size=$BATCH_SIZE"
  --set "training.accumulation_steps=$ACCUMULATION_STEPS"
  --set "training.num_epochs=$NUM_EPOCHS"
  --set "training.sample_num=$SAMPLE_NUM"
  --set "training.target_lr=$TARGET_LR"
  --set "training.max_training_token=$MAX_TRAINING_TOKEN"
  --set "training.max_training_padding_gap=$MAX_TRAINING_PADDING_GAP"
  --set "training.logps_chunk_size=$LOGPS_CHUNK_SIZE"
  --set "training.train_data_fraction=$TRAIN_DATA_FRACTION"
  --set "training.train_subset_seed=$TRAIN_SUBSET_SEED"
  --set "training.max_train_samples=$MAX_TRAIN_SAMPLES"
  --set "training.num_workers=$NUM_WORKERS"
  --set "training.save_steps=$SAVE_STEPS"
  --set "training.start_epoch=$START_EPOCH"
  --set "training.start_batch=$START_BATCH"
  --set "training.start_used_items=$START_USED_ITEMS"
  --set "training.start_rollout_count=$START_ROLLOUT_COUNT"
  --set "flashgrpo.cpeak_nodes=$CPEAK_NODES"
  --set "flashgrpo.auto_tune_cpeak_enabled=$AUTO_TUNE_CPEAK"
  --set "flashgrpo.max_tree_nodes_per_seq=$MAX_TREE_NODES_PER_SEQ"
  --set "logging.log_dir=$LOG_DIR"
  --set "logging.append=$APPEND_LOG"
  --set "training.saved_model_dir=$OUTPUT_DIR/target_lora"
  --set "training.saved_medusa_dir=$OUTPUT_DIR/medusa_heads"
)

if [[ -n "$LOAD_LORA_PATH" ]]; then
  cmd+=(--set "training.load_lora_path=$LOAD_LORA_PATH")
fi
if [[ -n "${CPEAK_CANDIDATES:-}" ]]; then
  cmd+=(--set "flashgrpo.auto_tune_cpeak_candidates=$CPEAK_CANDIDATES")
fi
if (($#)); then
  cmd+=("$@")
fi

printf 'Run name : %s\nModel    : %s\nMethod   : %s\nDataset  : %s (fraction=%s)\nConfig   : %s\nHeads    : %s\nLogs     : %s\nOutputs  : %s\n' \
  "$RUN_NAME" "$MODEL" "$METHOD" "$TRAIN_OPTION" "$TRAIN_DATA_FRACTION" "$CONFIG" "$HEADS" "$LOG_DIR" "$OUTPUT_DIR"
printf 'Command  :'; printf ' %q' "${cmd[@]}"; printf '\n'

if [[ "${DRY_RUN:-false}" == "true" ]]; then
  exit 0
fi
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 2; }
[[ -f "$MODEL/config.json" ]] || { echo "Model config not found: $MODEL/config.json" >&2; exit 2; }
[[ -f "$HEADS/medusa_config.json" ]] || { echo "MEDUSA checkpoint not found: $HEADS/medusa_config.json" >&2; exit 2; }
[[ -f "$HEADS/medusa_heads.pt" ]] || { echo "MEDUSA weights not found: $HEADS/medusa_heads.pt" >&2; exit 2; }
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
"${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/console.log"
