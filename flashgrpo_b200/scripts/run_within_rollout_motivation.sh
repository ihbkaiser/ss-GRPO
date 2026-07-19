#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

new_target_lora="${NEW_TARGET_LORA:?Set NEW_TARGET_LORA to the policy checkpoint after an update}"
old_head_dir="${OLD_HEAD_DIR:?Set OLD_HEAD_DIR to the stale MEDUSA checkpoint from before that update}"
experiment_id="${EXPERIMENT_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
experiment_root="${EXPERIMENT_ROOT:-outputs/within_rollout_motivation/$experiment_id}"
comparison_dir="${COMPARISON_DIR:-logs/within_rollout_motivation/$experiment_id}"

[[ -d "$new_target_lora" ]] || { echo "Missing target LoRA checkpoint: $new_target_lora" >&2; exit 2; }
[[ -f "$old_head_dir/medusa_config.json" ]] || { echo "Missing stale MEDUSA checkpoint: $old_head_dir" >&2; exit 2; }

mkdir -p "$experiment_root" "$comparison_dir"

for mode in disabled delayed immediate; do
  ADAPTATION_MODE="$mode" \
  MEDUSA_HEAD_DIR="$old_head_dir" \
  LOAD_LORA_PATH="$new_target_lora" \
  MAX_TRAIN_SAMPLES="${MOTIVATION_PROMPTS:-32}" \
  ACCUMULATION_STEPS=1000000 \
  DO_SAMPLE=false \
  REPEATED_GENERATE_NUMS=1 \
  CPEAK_NODES="${CPEAK_NODES:-128}" \
  TRACE_WINDOW_TOKENS="${TRACE_WINDOW_TOKENS:-32}" \
  RUN_ID="$mode" \
  RUN_ROOT="$experiment_root/$mode" \
  LOG_DIR="$comparison_dir/$mode" \
    bash flashgrpo_b200/scripts/run_qwen25_7b_gsm8k_oneclick.sh
done

python flashgrpo_b200/scripts/extract_grpo_motivation.py \
  --log-dir "$comparison_dir/immediate" \
  --target-dir "$experiment_root/immediate/target_lora" \
  --head-dir "$experiment_root/immediate/medusa_heads" \
  --compare "disabled=$comparison_dir/disabled" \
  --compare "delayed=$comparison_dir/delayed" \
  --compare "immediate=$comparison_dir/immediate"

echo "Motivation figure: $comparison_dir/immediate/within_rollout_motivation.png"
