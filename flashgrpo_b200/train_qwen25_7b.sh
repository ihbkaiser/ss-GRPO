#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="qwen25_7b"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-7B-Instruct}"
METHOD="${METHOD:-hrdcr}"                    # hrdcr | medusa_only
DATASET="${DATASET:-gsm8k}"                  # gsm8k | simplelr | dapo
TRAIN_DATA_FRACTION="${TRAIN_DATA_FRACTION:-0.4}"
RUN_TAG="${RUN_TAG:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_TRAINING_TOKEN="${MAX_TRAINING_TOKEN:-8192}"
MAX_TRAINING_PADDING_GAP="${MAX_TRAINING_PADDING_GAP:-1024}"
REPEATED_GENERATE_NUMS="${REPEATED_GENERATE_NUMS:-8}"
LOGPS_CHUNK_SIZE="${LOGPS_CHUNK_SIZE:-512}"
CPEAK_NODES="${CPEAK_NODES:-512}"
MAX_TREE_NODES_PER_SEQ="${MAX_TREE_NODES_PER_SEQ:-10}"
HEADS="${HEADS:-outputs/pretrain/qwen25_7b}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"

