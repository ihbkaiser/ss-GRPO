# FlashGRPO B200 experiment launcher

## Layout

| Model | Pretrain | Train |
| --- | --- | --- |
| Qwen2.5-1.5B | `pretrain_qwen25_1p5b.sh` | `train_qwen25_1p5b.sh` |
| Qwen3-4B | `pretrain_qwen3_4b.sh` | `train_qwen3_4b.sh` |
| Qwen2.5-7B | `pretrain_qwen25_7b.sh` | `train_qwen25_7b.sh` |
| Llama-3.1-8B | `pretrain_llama31_8b.sh` | `train_llama31_8b.sh` |

Checkpoint and log paths are separated:

```text
outputs/pretrain/<model>/
outputs/train/<model>/<run_name>/{target_lora,medusa_heads}/
logs/flashgrpo_b200/<model>/pretrain/<run_name>/
logs/flashgrpo_b200/<model>/train/<run_name>/
```

The generated run name contains model, method, dataset, data fraction and a timestamp.
Set `RUN_NAME` only when a stable custom name is needed.

## Pretrain heads

```bash
bash flashgrpo_b200/pretrain_qwen25_7b.sh
```

Common overrides:

```bash
PRETRAIN_DATA_FRACTION=0.5 \
PRETRAIN_EPOCHS=1 \
PRETRAIN_BATCH_SIZE=4 \
PRETRAIN_GRAD_ACCUM=8 \
PRETRAIN_MAX_SEQ_LEN=1024 \
bash flashgrpo_b200/pretrain_qwen25_7b.sh
```

`OUT` defaults to `outputs/pretrain/<model>`. The pretrain metrics and resolved config
are written under the model-specific log tree, not mixed with checkpoint files.
Qwen2.5 pretraining requests FlashAttention-2 and safely falls back to eager attention
when `flash_attn` is unavailable. Tree verification during GRPO remains on SDPA because
it requires an arbitrary 4D ancestor mask.

## Train HRDCR

```bash
DATASET=gsm8k TRAIN_DATA_FRACTION=0.4 \
bash flashgrpo_b200/train_qwen25_7b.sh
```

Supported aliases include `gsm8k`, `simplelr`, `simplelr_qwen`, and `dapo`. A helper-
supported dataset name can also be passed directly. Frequently changed controls are at
the top of every model launcher and can be overridden from the environment:

```bash
DATASET=dapo \
TRAIN_DATA_FRACTION=0.25 \
BATCH_SIZE=8 \
ACCUMULATION_STEPS=4 \
MAX_LENGTH=2048 \
MAX_PROMPT_LENGTH=2048 \
MAX_TRAINING_TOKEN=8192 \
MAX_TRAINING_PADDING_GAP=1024 \
RUN_TAG=trial1 \
bash flashgrpo_b200/train_llama31_8b.sh
```

Use `METHOD=medusa_only` for the paired no-Reflex ablation. Use `DRY_RUN=true` to
print the fully resolved command without loading a model:

```bash
METHOD=medusa_only DATASET=gsm8k DRY_RUN=true \
bash flashgrpo_b200/train_qwen3_4b.sh
```

If the model directory name differs on the B200 host, override it without editing YAML:

```bash
MODEL=/workspace/storage-shared/models/Qwen3-4B-Instruct \
bash flashgrpo_b200/train_qwen3_4b.sh
```
