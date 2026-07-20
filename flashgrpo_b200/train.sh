#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-$ROOT/models/Qwen2.5-7B-Instruct}"
HEADS="${HEADS:-outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b}"
RUN_NAME="${RUN_NAME:-${EXP:-reflexgrpo_b200_qwen25_7b}}"
CONFIG="${CONFIG:-flashgrpo_b200/configs/reflexgrpo_optimized_b200_qwen25_7b_simplelrabel3to5.yaml}"
DATASET="${DATASET:-simplelr_abel_level3to5}"

case "${DATASET,,}" in
  simplelr|simplerl|simplelr_abel_level3to5)
    TRAIN_OPTION="simplelr_abel_level3to5"
    ;;
  gsm8k)
    TRAIN_OPTION="gsm8k"
    ;;
  dapo|dapo-math|dapo_math)
    TRAIN_OPTION="DAPO-math"
    ;;
  *)
    # Pass through future/helper-supported dataset names unchanged.
    TRAIN_OPTION="$DATASET"
    ;;
esac

BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_TRAINING_TOKEN="${MAX_TRAINING_TOKEN:-8192}"
MAX_TRAINING_PADDING_GAP="${MAX_TRAINING_PADDING_GAP:-1024}"
LOGPS_CHUNK_SIZE="${LOGPS_CHUNK_SIZE:-512}"
CPEAK_NODES="${CPEAK_NODES:-128}"
AUTO_TUNE_CPEAK="${AUTO_TUNE_CPEAK:-true}"
MAX_TREE_NODES_PER_SEQ="${MAX_TREE_NODES_PER_SEQ:-10}"
TRAIN_DATA_FRACTION="${TRAIN_DATA_FRACTION:-0.4}"
TRAIN_SUBSET_SEED="${TRAIN_SUBSET_SEED:-42}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
LOG_DIR="${LOG_DIR:-logs/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN_NAME}"

python flashgrpo_b200/scripts/train_flashgrpo_b200.py \
  --config "$CONFIG" \
  --set run_name="$RUN_NAME" \
  --set model.model_dir="$MODEL" \
  --set flashgrpo.medusa_heads_checkpoint="$HEADS" \
  --set aux_head_checkpoint="$HEADS" \
  --set data.train_option="$TRAIN_OPTION" \
  --set generation.max_length="$MAX_LENGTH" \
  --set generation.max_prompt_length="$MAX_PROMPT_LENGTH" \
  --set training.batch_size="$BATCH_SIZE" \
  --set training.accumulation_steps="$ACCUMULATION_STEPS" \
  --set training.max_training_token="$MAX_TRAINING_TOKEN" \
  --set training.max_training_padding_gap="$MAX_TRAINING_PADDING_GAP" \
  --set training.logps_chunk_size="$LOGPS_CHUNK_SIZE" \
  --set training.train_data_fraction="$TRAIN_DATA_FRACTION" \
  --set training.train_subset_seed="$TRAIN_SUBSET_SEED" \
  --set training.max_train_samples="$MAX_TRAIN_SAMPLES" \
  --set flashgrpo.cpeak_nodes="$CPEAK_NODES" \
  --set flashgrpo.auto_tune_cpeak_enabled="$AUTO_TUNE_CPEAK" \
  --set flashgrpo.max_tree_nodes_per_seq="$MAX_TREE_NODES_PER_SEQ" \
  --set logging.log_dir="$LOG_DIR" \
  --set training.saved_model_dir="$OUTPUT_DIR/target_lora" \
  --set training.saved_medusa_dir="$OUTPUT_DIR/medusa_heads"
