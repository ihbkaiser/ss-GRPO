#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
log_root="${LOG_ROOT:-logs/motivation_oneclick/$run_id}"
artifact_root="${ARTIFACT_ROOT:-outputs/motivation_oneclick/$run_id}"
if [[ -n "${MODEL_DIR:-}" ]]; then
  model_dir="$MODEL_DIR"
elif [[ -f "$repo_root/models/Qwen2.5-7B-Instruct/config.json" ]]; then
  model_dir="$repo_root/models/Qwen2.5-7B-Instruct"
else
  model_dir="/workspace/storage-shared/models/Qwen2.5-7B-Instruct"
fi
sharegpt_data="${SHAREGPT_DATA:-$repo_root/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
head_dir="${MEDUSA_HEAD_DIR:-outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b}"

mkdir -p "$log_root" "$artifact_root"
exec > >(tee -a "$log_root/master.log") 2>&1

echo "[motivation] run_id=$run_id"
echo "[motivation] model=$model_dir"
echo "[motivation] logs=$log_root"

[[ -f "$model_dir/config.json" ]] || {
  echo "ERROR: model not found at $model_dir. Set MODEL_DIR." >&2
  exit 2
}
python -c 'import torch, peft, transformers, yaml, matplotlib; assert torch.cuda.is_available(), "CUDA unavailable"' || {
  echo "ERROR: activate the FlashGRPO Python environment before running this script." >&2
  exit 2
}

if [[ ! -f "$head_dir/medusa_config.json" ]]; then
  [[ -f "$sharegpt_data" ]] || {
    echo "ERROR: pretrained heads are absent and ShareGPT data was not found at $sharegpt_data. Set SHAREGPT_DATA." >&2
    exit 2
  }
  echo "[motivation] Stage 1/3: pretraining MEDUSA heads"
  MODEL="$model_dir" \
  DATA="$sharegpt_data" \
  OUT="$head_dir" \
  PRETRAIN_NUM_SAMPLES="${PRETRAIN_NUM_SAMPLES:-0}" \
  PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}" \
    bash flashgrpo_b200/pretrain.sh
else
  echo "[motivation] Stage 1/3: reusing pretrained heads at $head_dir"
fi

echo "[motivation] Stage 2/3: GRPO training with step-by-step checkpoints"
training_root="$artifact_root/grpo"
training_log="$log_root/grpo"
MODEL_DIR="$model_dir" \
MEDUSA_HEAD_DIR="$head_dir" \
RUN_ID=grpo \
RUN_ROOT="$training_root" \
LOG_DIR="$training_log" \
SAVE_STEPS=1 \
MAX_TRAIN_SAMPLES="${GRPO_PROMPTS:-256}" \
  bash flashgrpo_b200/scripts/run_qwen25_7b_gsm8k_oneclick.sh

mapfile -t target_steps < <(find "$training_root/target_lora" -mindepth 1 -maxdepth 1 -type d -name 'step*' | sort -V)
if (( ${#target_steps[@]} < 2 )); then
  echo "ERROR: fewer than two policy checkpoints were produced. Re-run with a larger GRPO_PROMPTS." >&2
  exit 3
fi

new_target="${target_steps[-1]}"
new_step_name="$(basename "$new_target")"
new_step="${new_step_name#step}"
lags="${MISMATCH_LAGS:-1,2,3}"
lags="${lags//,/ }"
completed_pairs=0

echo "[motivation] Stage 3/3: disabled/delayed/immediate replay for lags: $lags"
for lag in $lags; do
  [[ "$lag" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: invalid lag '$lag' in MISMATCH_LAGS=$lags" >&2
    exit 3
  }
  old_index=$(( ${#target_steps[@]} - 1 - lag ))
  if (( old_index < 0 )); then
    echo "[motivation] skipping lag=$lag: only ${#target_steps[@]} checkpoints are available"
    continue
  fi
  old_step_name="$(basename "${target_steps[$old_index]}")"
  old_step="${old_step_name#step}"
  old_heads="$training_root/medusa_heads/$old_step_name"
  if [[ ! -f "$old_heads/medusa_config.json" ]]; then
    echo "[motivation] skipping lag=$lag: stale heads missing at $old_heads"
    continue
  fi

  pair_name="step${old_step}_to_step${new_step}"
  echo "[motivation] replay lag=$lag: policy step$new_step + stale heads step$old_step"
  EXPERIMENT_ID="$pair_name" \
  EXPERIMENT_ROOT="$artifact_root/replay/$pair_name" \
  COMPARISON_DIR="$log_root/replay/$pair_name" \
  NEW_TARGET_LORA="$new_target" \
  OLD_HEAD_DIR="$old_heads" \
  MODEL_DIR="$model_dir" \
  MOTIVATION_PROMPTS="${MOTIVATION_PROMPTS:-64}" \
  TRACE_WINDOW_TOKENS="${TRACE_WINDOW_TOKENS:-32}" \
  CPEAK_NODES="${CPEAK_NODES:-512}" \
    bash flashgrpo_b200/scripts/run_within_rollout_motivation.sh
  completed_pairs=$((completed_pairs + 1))
done

if (( completed_pairs == 0 )); then
  echo "ERROR: no checkpoint pair could be replayed; increase GRPO_PROMPTS or adjust MISMATCH_LAGS." >&2
  exit 3
fi

archive="$artifact_root/motivation_logs_${run_id}.tar.gz"
tar -czf "$archive" -C "$log_root" .
echo "[motivation] DONE"
echo "[motivation] send this archive for analysis: $archive"
echo "[motivation] figures: $log_root/replay/step*_to_step*/immediate/within_rollout_motivation.png"
