# B200 ablations

All Qwen2.5-7B ablations inherit `_base_qwen25_7b.yaml`, which in turn
inherits the controlled workload in `fair_fastgrpo_b200/_base_qwen25_7b.yaml`.
The shared workload is:

- batch size 8 and gradient accumulation 4;
- prompt/total generation limit 2048 tokens;
- training token budget 8192 and padding gap 1024;
- log-probability chunk size 512;
- eight generations per prompt and 40% of the selected training dataset.

The launcher accepts `DATASET=simplelr`, `DATASET=gsm8k`, or `DATASET=dapo`.
It also accepts the exact helper names `simplelr_abel_level3to5` and
`DAPO-math`. Artifacts are always derived from `RUN_NAME` by default:

```text
logs/<RUN_NAME>/
outputs/<RUN_NAME>/target_lora/
outputs/<RUN_NAME>/medusa_heads/
```

Example:

```bash
RUN_NAME=medusa_ce_no_reflex_gsm8k \
DATASET=gsm8k \
CONFIG=flashgrpo_b200/configs/ablations/medusa_online_ce_no_reflex_qwen25_7b.yaml \
MODEL=/workspace/storage-shared/models/Qwen2.5-7B-Instruct \
HEADS=outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b \
bash flashgrpo_b200/train.sh
```

Llama-3.1-8B uses the matching Llama ablation config and pretrained heads:

```bash
RUN_NAME=medusa_ce_no_reflex_llama31_8b_simplelr \
DATASET=simplelr \
CONFIG=flashgrpo_b200/configs/ablations/medusa_online_ce_no_reflex_llama31_8b.yaml \
MODEL=/workspace/storage-shared/models/Llama-3.1-8B-Instruct \
HEADS=outputs/flashgrpo_b200_medusa_sharegpt_llama31_8b \
bash flashgrpo_b200/train.sh
```

`heads_4.yaml` and `heads_5.yaml` require a matching pretrained checkpoint;
set `HEADS` to a checkpoint trained with the requested number of heads.
