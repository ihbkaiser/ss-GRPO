#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

model_dir="${MODEL_DIR:-/workspace/storage-shared/models/Qwen2.5-7B-Instruct}"
head_dir="${MEDUSA_HEAD_DIR:-outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_root="${RUN_ROOT:-outputs/qwen25_7b_gsm8k_motivation/$run_id}"
log_dir="${LOG_DIR:-logs/qwen25_7b_gsm8k_motivation/$run_id}"
target_dir="$run_root/target_lora"
trained_head_dir="$run_root/medusa_heads"
config="flashgrpo_b200/configs/fair_fastgrpo_b200/qwen25_7b_gsm8k.yaml"
adaptation_mode="${ADAPTATION_MODE:-immediate}"
trace_window="${TRACE_WINDOW_TOKENS:-32}"

case "$adaptation_mode" in
  disabled|delayed|immediate) ;;
  *) echo "ADAPTATION_MODE must be disabled, delayed, or immediate" >&2; exit 2 ;;
esac

[[ -f "$model_dir/config.json" ]] || { echo "Missing model: $model_dir" >&2; exit 2; }
[[ -f "$head_dir/medusa_config.json" ]] || { echo "Missing MEDUSA checkpoint: $head_dir" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is unavailable" >&2; exit 2; }
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'

mkdir -p "$log_dir" "$run_root"
{
  echo "run_id=$run_id"
  echo "model_dir=$model_dir"
  echo "head_dir=$head_dir"
  echo "adaptation_mode=$adaptation_mode"
  echo "trace_window_tokens=$trace_window"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  git rev-parse HEAD
  git status --short
} > "$log_dir/run_manifest.txt"

python flashgrpo_b200/scripts/train_flashgrpo_b200.py \
  --config "$config" \
  --set "model.model_dir=$model_dir" \
  --set "flashgrpo.medusa_heads_checkpoint=$head_dir" \
  --set "aux_head_checkpoint=$head_dir" \
  --set "training.saved_model_dir=$target_dir" \
  --set "training.saved_medusa_dir=$trained_head_dir" \
  --set "training.save_steps=${SAVE_STEPS:-5}" \
  --set "training.load_lora_path=${LOAD_LORA_PATH:-}" \
  --set "training.max_train_samples=${MAX_TRAIN_SAMPLES:-0}" \
  --set "training.accumulation_steps=${ACCUMULATION_STEPS:-4}" \
  --set "generation.do_sample=${DO_SAMPLE:-true}" \
  --set "generation.repeated_generate_nums=${REPEATED_GENERATE_NUMS:-8}" \
  --set "flashgrpo.auto_tune_cpeak_enabled=false" \
  --set "flashgrpo.cpeak_nodes=${CPEAK_NODES:-128}" \
  --set "reflex.adaptation_mode=$adaptation_mode" \
  --set "reflex.feedback_stride=1" \
  --set "reflex.feedback_stride_min=1" \
  --set "reflex.motivation_trace_enabled=true" \
  --set "reflex.motivation_trace_window_tokens=$trace_window" \
  --set "logging.log_dir=$log_dir" \
  --set "logging.append=false" \
  2>&1 | tee "$log_dir/train_console.log"

python flashgrpo_b200/scripts/extract_grpo_motivation.py \
  --log-dir "$log_dir" \
  --target-dir "$target_dir" \
  --head-dir "$trained_head_dir" \
  2>&1 | tee "$log_dir/extract_console.log"

echo "Completed: $log_dir"
