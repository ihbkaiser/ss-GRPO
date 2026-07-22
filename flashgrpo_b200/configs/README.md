# B200 experiment configs

The canonical configs are grouped by model:

```text
configs/
  _shared/             # Common HRDCR workload and MEDUSA pretraining defaults
  qwen25_1p5b/
  qwen3_4b/
  qwen25_7b/
  llama31_8b/
```

Each model directory contains:

- `train_hrdcr.yaml`: full HRDCR (`m_t`) plus sparse post-rollout auxiliary refresh.
- `train_medusa_only.yaml`: fair no-Reflex ablation; exact verification, tree scheduling,
  pretrained heads, sparse tree and auxiliary refresh are unchanged.
- `pretrain.yaml`: standard parallel MEDUSA-head pretraining. HRDCR's hidden-space
  fast state is created online and does not require a separately pretrained module.

Files outside these directories are retained for compatibility with previous runs.
New experiments should use the per-model shell launchers at the package root.
