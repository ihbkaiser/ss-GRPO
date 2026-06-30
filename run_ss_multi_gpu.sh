cd /mnt/hdd/nhatminh/FastGRPO-main
conda activate /mnt/hdd/nhatminh/fastgrpo_env

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

NPROC=8
MODEL=/mnt/hdd/nhatminh/FastGRPO-main/models/Qwen2.5-1.5B-Instruct
EXP=selfspec_mg_8gpu_autosearch

torchrun --standalone --nproc_per_node=$NPROC \
  self-speculative_multi_gpu/grpo_train_mg.py \
  --model_dir $MODEL \
  --train_option simplelr_abel_level3to5 \
  --version_name $EXP \
  --batch_size 8 \
  --num_epochs 1 \
  --accumulation_steps 4 \
  --target_lr 1e-6 \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_length 2048 \
  --max_training_token 3072 \
  --max_training_padding_gap 256 \
  --logps_chunk_size 256 \
  --statistical_time False \
  --repeated_generate_nums 8 \
  --auto_search_skip_layers True \
  --search_candidate_layers "" \
  --search_num_prompts 4 \
  --search_max_length 512 \
  --search_min_skip 12 \
  --search_max_skip 16 \
  --search_init_trials 6 \
  --search_bo_trials 12 \
  --search_candidate_pool 96 \
  --search_seed 13 \
  --search_json_out runs/$EXP/skip_search.json \
  --max_draft_tokens 4 \
  --confidence_threshold 0.0 \
  --beta 0.04 \
  --epsilon 0.1 \
  --log_file runs/$EXP/train.jsonl \
  --saved_model_dir runs/$EXP/checkpoints