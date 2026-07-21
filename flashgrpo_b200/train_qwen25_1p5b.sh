#!/usr/bin/env bash
set -euo pipefail

# Frequently changed experiment settings. Every value can also be overridden
# from the environment, for example: DATASET=dapo TRAIN_DATA_FRACTION=0.2 bash ...
MODEL_KEY="qwen25_1p5b"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-1.5B-Instruct}"
METHOD="${METHOD:-hrdcr}"                    # hrdcr | medusa_only
DATASET="${DATASET:-gsm8k}"                  # gsm8k | simplelr | dapo
TRAIN_DATA_FRACTION="${TRAIN_DATA_FRACTION:-0.4}"
RUN_TAG="${RUN_TAG:-}"

BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_TRAINING_TOKEN="${MAX_TRAINING_TOKEN:-8192}"
MAX_TRAINING_PADDING_GAP="${MAX_TRAINING_PADDING_GAP:-1024}"
REPEATED_GENERATE_NUMS="${REPEATED_GENERATE_NUMS:-8}"
LOGPS_CHUNK_SIZE="${LOGPS_CHUNK_SIZE:-512}"
CPEAK_NODES="${CPEAK_NODES:-256}"
MAX_TREE_NODES_PER_SEQ="${MAX_TREE_NODES_PER_SEQ:-10}"
HEADS="${HEADS:-outputs/pretrain/qwen25_1p5b}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/train_model.sh"

