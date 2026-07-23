# B200 experiment configs

The canonical configs are grouped by model:

```text
configs/
  _shared/             # Common HRDCR workload and MEDUSA pretraining defaults
  qwen25_1p5b/
  qwen25_3b/
  qwen3_4b/
  qwen25_7b/
  qwen25_14b/
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

The Qwen2.5 launchers expect the local instruct checkpoints under:

```text
/workspace/storage-shared/models/Qwen2.5-1.5B-Instruct
/workspace/storage-shared/models/Qwen2.5-3B-Instruct
/workspace/storage-shared/models/Qwen2.5-7B-Instruct
/workspace/storage-shared/models/Qwen2.5-14B-Instruct
```
