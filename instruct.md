# Reproducing FastGRPO and Comparing Against Vanilla GRPO

This runbook describes how to reproduce the FastGRPO training setup from the paper and how to compare it with a vanilla autoregressive GRPO baseline.

Paper: FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning, arXiv:2509.21792.

## Hardware Notes

The paper reports results on H800 SXM GPUs. A faithful reproduction of Table 1 requires similar 40GB/80GB class accelerators. The current repo loads target models in normal precision and trains LoRA plus a draft model, so 7B/8B experiments are not expected to fit on a 12GB GPU.

Use small models only for smoke tests. Use the 7B/8B models below for paper-level reproduction.

## Environment

From the repository root:

```bash
cd /workspace/ss-GRPO

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional but recommended:

```bash
export HF_HOME=/workspace/ss-GRPO/.hf_cache
export CUDA_VISIBLE_DEVICES=0
```

Verify CUDA:

```bash
python - <<'PY'
import torch, transformers, datasets
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
PY
```

## Data

The paper uses:

- Draft pretraining: ShareGPT-68K.
- GRPO training/evaluation datasets: GSM8K, SimpleRL-Abel-Level3to5, DAPO-Math-17K.

Download the datasets:

```bash
huggingface-cli download openai/gsm8k \
  --repo-type dataset \
  --local-dir data/gsm8k

huggingface-cli download open-r1/DAPO-Math-17k-Processed \
  --repo-type dataset \
  --local-dir data/DAPO-Math-17k-Processed

huggingface-cli download Aeala/ShareGPT_Vicuna_unfiltered \
  ShareGPT_V4.3_unfiltered_cleaned_split.json \
  --repo-type dataset \
  --local-dir data/sharegpt_full
```

Expected local paths:

```text
data/gsm8k
data/simplelr_abel_level3to5
data/DAPO-Math-17k-Processed
data/sharegpt_full/ShareGPT_V4.3_unfiltered_cleaned_split.json
```

Check that the repo loader can see the data:

```bash
python - <<'PY'
from helper.get_QAs import get_train_QAs, get_test_QAs

for name in ["gsm8k", "simplelr_abel_level3to5", "DAPO-math"]:
    train = get_train_QAs(name)
    test = get_test_QAs(name)
    print(name, "train:", len(train), "test:", len(test))
PY
```

## Models Used In The Paper

Table 1 uses these target models:

```text
Qwen/Qwen2.5-7B-Instruct
meta-llama/Llama-3.1-8B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Qwen/Qwen2.5-Math-7B
Qwen/Qwen2.5-Math-7B-Instruct
```

Use:

```text
--model_type qwen2
```

for Qwen and DeepSeek-R1-Distill-Qwen models, and:

```text
--model_type llama
```

for Llama-3.1 models.

## Step 1: Pretrain The Draft Model On Full ShareGPT

The paper pretrains the draft model for 10 epochs on ShareGPT-68K with effective batch size 16 and learning rate 1e-4.

Example for Qwen2.5-7B-Instruct:

```bash
source .venv/bin/activate

export MODEL=Qwen/Qwen2.5-7B-Instruct
export MODEL_TYPE=qwen2
export EXP=qwen25_7b_inst_simplerl

mkdir -p outputs/$EXP/draft_pretrain logs/$EXP/draft_pretrain

python train_draft.py \
  --model_dir $MODEL \
  --version_name $EXP \
  --model_type $MODEL_TYPE \
  --draft_num_hidden_layers 1 \
  --batch_size 1 \
  --num_epochs 10 \
  --lr 1e-4 \
  --accumulation_steps 16 \
  --warmup_ratio 0.05 \
  --sample_num 100 \
  --max_length 4096 \
  --log_dir logs/$EXP/draft_pretrain \
  --saved_model_dir outputs/$EXP/draft_pretrain \
  --dataset_dir data/sharegpt_full/ShareGPT_V4.3_unfiltered_cleaned_split.json
```

The final checkpoint will look like:

```text
outputs/$EXP/draft_pretrain/stepXXXX.pth
```

Set:

```bash
export DRAFT=outputs/$EXP/draft_pretrain/stepXXXX.pth
```

Use the actual last checkpoint path.

Note: `train_draft.py` has a known smoke-test bug when `--accumulation_steps 1` because `loss1_norm` and `loss2_norm` are not defined. The paper setup uses `--accumulation_steps 16`, so this does not affect full reproduction.

## Step 2: Run FastGRPO

Paper-level FastGRPO setup:

- Prompt batch size per GPU: 4.
- Responses per prompt: 8.
- Epochs: 10.
- Target LR: 1e-6.
- Online draft LR: 5e-5.
- Sampling: temperature 1.0, top_p 0.95.
- Max generation length: 2048.
- GRPO beta: 0.04.
- GRPO epsilon: 0.1.

Run on SimpleRL-Abel-Level3to5:

```bash
mkdir -p outputs/$EXP/fastgrpo_target \
         outputs/$EXP/fastgrpo_draft \
         outputs/$EXP/fastgrpo_stats \
         logs/$EXP

python grpo_speculative.py \
  --model_dir $MODEL \
  --adapter_path $DRAFT \
  --load_lora_path "" \
  --model_type $MODEL_TYPE \
  --draft_num_hidden_layers 1 \
  --train_option simplelr_abel_level3to5 \
  --version_name $EXP \
  --batch_size 4 \
  --num_epochs 10 \
  --sample_num 100 \
  --accumulation_steps 4 \
  --draft_accumulation_steps 1 \
  --target_lr 1e-6 \
  --draft_lr 5e-5 \
  --is_train_draft True \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_length 2048 \
  --max_training_padding_gap 256 \
  --max_training_token 3072 \
  --grpo_iteration_num 1 \
  --repeated_generate_nums 8 \
  --beta 0.04 \
  --epsilon 0.1 \
  --log_file logs/$EXP/fastgrpo.jsonl \
  --saved_model_dir outputs/$EXP/fastgrpo_target \
  --saved_draft_model_dir outputs/$EXP/fastgrpo_draft \
  --saved_statistics_dir outputs/$EXP/fastgrpo_stats
```

For other paper datasets, change:

```bash
--train_option gsm8k
--train_option DAPO-math
```

Monitor logs:

```bash
tail -f logs/$EXP/fastgrpo.jsonl
```

Important fields:

```text
generate_time_cost
train_time_cost
draft_train_time_cost
average_acc_length
last_100_generate_time_cost
last_100_train_time_cost
mean_reward
```

`fastgrpo.jsonl` is appended during training, not only at the end. However, a row is written only after at least one generated group has non-zero reward variance. If all responses in a group receive identical reward, the group is skipped. Small smoke runs often produce an empty JSONL file for this reason.

## Step 3: Create A Vanilla GRPO Baseline

The paper compares FastGRPO against standard autoregressive decoding inside the same GRPO training loop.

The correct baseline should keep the same:

- target model,
- dataset,
- prompt template,
- group size,
- sampling settings,
- reward functions,
- reward filtering,
- LoRA target update,
- GRPO loss,
- optimizer settings,
- max length,
- epochs.

Only the rollout generation changes:

- FastGRPO uses `speculative_generate(...)`.
- Vanilla GRPO uses `target_model.generate(...)`.

Create a baseline file:

```bash
cp grpo_speculative.py grpo_vanilla.py
```

Edit `grpo_vanilla.py` as follows.

### 3.1 Keep The Target Model And LoRA Setup

You may leave draft loading in place if you want minimal code changes, but it is cleaner to remove online draft training from the baseline. At minimum, run with:

```bash
--is_train_draft False
```

The target LoRA training logic should remain identical to `grpo_speculative.py`.

### 3.2 Replace Speculative Rollout With Vanilla Autoregressive Rollout

Find the block in `grpo_speculative.py` that calls:

```python
outputs = speculative_generate(
    model=model,
    input_ids=input_ids,
    attention_mask=attention_mask,
    tokenizer=tokenizer,
    do_sample=True,
    max_length=max_length,
    repeated_generate_nums=repeated_generate_nums,
    temperature=temperature,
    top_p=top_p,
    return_all_draft_input=True,
    statistical_time=True,
)
```

Replace it in `grpo_vanilla.py` with vanilla generation that returns the same fields used later by the training loop:

```python
torch.cuda.synchronize()
generate_start = time.time()

repeated_input_ids = input_ids.repeat_interleave(repeated_generate_nums, dim=0)
repeated_attention_mask = attention_mask.repeat_interleave(repeated_generate_nums, dim=0)

generated = model.target_model.generate(
    input_ids=repeated_input_ids,
    attention_mask=repeated_attention_mask,
    do_sample=True,
    temperature=temperature,
    top_p=top_p,
    max_length=max_length,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

torch.cuda.synchronize()
generate_time_cost = time.time() - generate_start

prompt_length = input_ids.shape[-1]
generated_token_ids = []
sequence_lengths = []

for seq in generated:
    completion = seq[prompt_length:]
    if tokenizer.eos_token_id is not None:
        eos_positions = (completion == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            completion = completion[: eos_positions[0].item() + 1]
    generated_token_ids.append(completion.detach().cpu())
    sequence_lengths.append(len(completion))

outputs = {
    "generated_token_ids": generated_token_ids,
    "max_sequence_length": max(sequence_lengths) if sequence_lengths else 0,
    "total_time_cost": generate_time_cost,
    "total_acc_length": 0,
    "total_decoded_token_num": sum(sequence_lengths),
    "prefill_time_cost": 0,
    "target_time_cost": generate_time_cost,
    "draft_time_cost": 0,
    "check_time_cost": 0,
}
```

Then keep the existing code that decodes outputs and computes rewards:

```python
outputs["decoded_sequences"] = [
    tokenizer.decode(x, skip_special_tokens=True)
    for x in outputs["generated_token_ids"]
]
```

### 3.3 Disable Online Draft Training In Vanilla

In vanilla GRPO, do not train the draft model. The easiest safe path is to run:

```bash
--is_train_draft False
```

If you want cleaner logs, set these fields to zero in the baseline log:

```text
draft_train_time_cost = 0
average_acc_length = 0
draft_time_cost = 0
check_time_cost = 0
```

## Step 4: Run Vanilla GRPO Baseline

Use the same config as FastGRPO. Change only the script name and output paths.

```bash
mkdir -p outputs/$EXP/vanilla_target \
         outputs/$EXP/vanilla_draft_unused \
         outputs/$EXP/vanilla_stats \
         logs/$EXP

python grpo_vanilla.py \
  --model_dir $MODEL \
  --adapter_path $DRAFT \
  --load_lora_path "" \
  --model_type $MODEL_TYPE \
  --draft_num_hidden_layers 1 \
  --train_option simplelr_abel_level3to5 \
  --version_name ${EXP}_vanilla \
  --batch_size 4 \
  --num_epochs 10 \
  --sample_num 100 \
  --accumulation_steps 4 \
  --draft_accumulation_steps 1 \
  --target_lr 1e-6 \
  --draft_lr 5e-5 \
  --is_train_draft False \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_length 2048 \
  --max_training_padding_gap 256 \
  --max_training_token 3072 \
  --grpo_iteration_num 1 \
  --repeated_generate_nums 8 \
  --beta 0.04 \
  --epsilon 0.1 \
  --log_file logs/$EXP/vanilla_grpo.jsonl \
  --saved_model_dir outputs/$EXP/vanilla_target \
  --saved_draft_model_dir outputs/$EXP/vanilla_draft_unused \
  --saved_statistics_dir outputs/$EXP/vanilla_stats
```

Monitor:

```bash
tail -f logs/$EXP/vanilla_grpo.jsonl
```

## Step 5: Compute Paper Metrics

The paper reports:

```text
Gen SR = baseline generation wall-clock time / FastGRPO generation wall-clock time
E2E SR = baseline end-to-end wall-clock time / FastGRPO end-to-end wall-clock time
```

Use the cumulative fields from the last JSONL row:

```bash
python - <<'PY'
import json
from pathlib import Path

exp = "qwen25_7b_inst_simplerl"
vanilla_path = Path(f"logs/{exp}/vanilla_grpo.jsonl")
fast_path = Path(f"logs/{exp}/fastgrpo.jsonl")

def load_last(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"No log rows in {path}")
    return rows[-1]

v = load_last(vanilla_path)
f = load_last(fast_path)

v_gen = v["generate_time_cost"]
f_gen = f["generate_time_cost"]

v_e2e = v["generate_time_cost"] + v["train_time_cost"] + v.get("draft_train_time_cost", 0)
f_e2e = f["generate_time_cost"] + f["train_time_cost"] + f.get("draft_train_time_cost", 0)

print("Vanilla generation minutes:", round(v_gen, 4))
print("FastGRPO generation minutes:", round(f_gen, 4))
print("Gen SR:", round(v_gen / f_gen, 4))
print()
print("Vanilla E2E minutes:", round(v_e2e, 4))
print("FastGRPO E2E minutes:", round(f_e2e, 4))
print("E2E SR:", round(v_e2e / f_e2e, 4))
print()
print("FastGRPO average_acc_length:", f.get("average_acc_length"))
print("FastGRPO mean_reward:", f.get("mean_reward"))
print("Vanilla mean_reward:", v.get("mean_reward"))
PY
```

The repo logs these cumulative costs in minutes. Ratios are unitless, so the ratio is unchanged as long as both logs use the same unit.

## Step 6: Repeat For Table 1

For each target model and dataset pair:

1. Pretrain a draft model on ShareGPT, or reuse a model-specific pretrained draft checkpoint.
2. Run FastGRPO for 10 epochs.
3. Run vanilla GRPO with identical settings.
4. Compute Gen SR and E2E SR.

Datasets:

```text
gsm8k
simplelr_abel_level3to5
DAPO-math
```

Models:

```text
Qwen/Qwen2.5-7B-Instruct
meta-llama/Llama-3.1-8B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
Qwen/Qwen2.5-Math-7B
Qwen/Qwen2.5-Math-7B-Instruct
```

Use separate experiment names, for example:

```text
qwen25_7b_inst_gsm8k
qwen25_7b_inst_simplerl
qwen25_7b_inst_dapo
llama31_8b_inst_gsm8k
```

## Smoke Test

Use this only to verify the pipeline on a small GPU. It will not reproduce paper speedups.

```bash
export MODEL=Qwen/Qwen2.5-Math-1.5B-Instruct
export MODEL_TYPE=qwen2
export EXP=smoke_qwen25_math15b

mkdir -p outputs/$EXP/draft_pretrain logs/$EXP/draft_pretrain

python train_draft.py \
  --model_dir $MODEL \
  --version_name $EXP \
  --model_type $MODEL_TYPE \
  --draft_num_hidden_layers 1 \
  --batch_size 1 \
  --num_epochs 1 \
  --lr 1e-4 \
  --accumulation_steps 2 \
  --warmup_ratio 0.05 \
  --sample_num 2 \
  --max_length 256 \
  --log_dir logs/$EXP/draft_pretrain \
  --saved_model_dir outputs/$EXP/draft_pretrain \
  --dataset_dir data/draft_smoke4_sharegpt.json
```

Then:

```bash
export DRAFT=outputs/$EXP/draft_pretrain/step2.pth

mkdir -p outputs/$EXP/fastgrpo_target \
         outputs/$EXP/fastgrpo_draft \
         outputs/$EXP/fastgrpo_stats \
         logs/$EXP

python grpo_speculative.py \
  --model_dir $MODEL \
  --adapter_path $DRAFT \
  --load_lora_path "" \
  --model_type $MODEL_TYPE \
  --draft_num_hidden_layers 1 \
  --train_option simplelr_abel_level3to5_smoke \
  --version_name $EXP \
  --batch_size 1 \
  --num_epochs 1 \
  --sample_num 2 \
  --accumulation_steps 1 \
  --draft_accumulation_steps 1 \
  --target_lr 1e-6 \
  --draft_lr 5e-5 \
  --is_train_draft True \
  --temperature 1.0 \
  --top_p 0.95 \
  --max_length 256 \
  --max_training_padding_gap 128 \
  --max_training_token 1024 \
  --grpo_iteration_num 1 \
  --repeated_generate_nums 4 \
  --beta 0.04 \
  --epsilon 0.1 \
  --log_file logs/$EXP/fastgrpo.jsonl \
  --saved_model_dir outputs/$EXP/fastgrpo_target \
  --saved_draft_model_dir outputs/$EXP/fastgrpo_draft \
  --saved_statistics_dir outputs/$EXP/fastgrpo_stats
```

If `fastgrpo.jsonl` is empty in a smoke test, the run may still have completed successfully. It usually means every generated group had identical rewards and was skipped by the GRPO update filter.

## Generation-Only Benchmark

The repo includes:

```bash
python benchmark_qwen3_fast_vs_vanilla.py \
  --model_dir $MODEL \
  --adapter_path $DRAFT \
  --draft_num_hidden_layers 1 \
  --train_option simplelr_abel_level3to5_smoke \
  --batch_size 1 \
  --num_prompts 4 \
  --repeated_generate_nums 2 \
  --max_length 256 \
  --temperature 1.0 \
  --top_p 0.95 \
  --warmup_batches 1 \
  --json_out outputs/$EXP/bench/fast_vs_vanilla.json
```

This measures generation only. It is useful for debugging speculative decoding, but it is not the full paper metric because it does not include GRPO policy update time and online draft training time.

## Common Pitfalls

1. Empty `grpo.jsonl`

   This usually means all generated responses in each group received the same reward, so the group was skipped. Increase `repeated_generate_nums`, use a stronger model, use full datasets, or run longer.

2. No paper speedup on small GPUs

   FastGRPO is designed around high-concurrency GRPO rollouts and H800-class hardware. On small GPUs and tiny smoke runs, speculative overhead can dominate and make FastGRPO slower than vanilla generation.

3. Full 7B model does not fit

   Use A100/H100/H800 class GPUs. The repo does not currently implement memory-saving 4-bit loading for this training path.

4. Baseline must be identical

   Do not compare FastGRPO against a baseline with different batch size, group size, max length, prompts, filtering, reward, or optimizer settings. Only the rollout generator should differ.

