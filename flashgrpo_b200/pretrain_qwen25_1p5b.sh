#!/usr/bin/env bash
set -euo pipefail

# MEDUSA-head pretraining settings. HRDCR m_t is built online and is not a
# separate pretrained module.
MODEL_KEY="qwen25_1p5b"
MODEL="${MODEL:-/workspace/storage-shared/models/Qwen2.5-1.5B-Instruct}"
PRETRAIN_DATASET="${PRETRAIN_DATASET:-sharegpt}"
DATA_PATH="${DATA_PATH:-data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
PRETRAIN_DATA_FRACTION="${PRETRAIN_DATA_FRACTION:-1.0}"
RUN_TAG="${RUN_TAG:-}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-16}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-2}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
PRETRAIN_MAX_SEQ_LEN="${PRETRAIN_MAX_SEQ_LEN:-1024}"
PRETRAIN_NUM_SAMPLES="${PRETRAIN_NUM_SAMPLES:-0}"
PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-500}"
LOSS_CHUNK_SIZE="${LOSS_CHUNK_SIZE:-128}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
OUT="${OUT:-outputs/pretrain/qwen25_1p5b}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/launch/pretrain_model.sh"
