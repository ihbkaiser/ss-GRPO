# FlashGRPO B200

`flashgrpo_b200` is an independent copy of the FlashGRPO code path for the
B200 environment. Its Python imports point to `flashgrpo_b200.*`, so changes
here do not affect the original `flashgrpo` package used for the 3090 setup.

Canonical B200 defaults:

- target model dtype is loaded from config and defaults to `bf16`;
- Qwen 7B uses fused PyTorch SDPA because MEDUSA tree verification requires a
  custom 4D ancestor mask;
- effective prompt-group batch is 32 by default (`16x2` for 1.5B and `8x4`
  for 4B-8B models);
- GRPO training token budget is `max_training_token: 8192`;
- the throughput-first tree budget is `cpeak_nodes: 128` with at most 10 nodes
  per sequence; the first 18 rollout batches test six hardware budgets and
  lock the one with the best median output-token throughput;
- tree masks are built in batched operations, target vocabulary projection is
  restricted to internal nodes, and accepted KV paths are compacted in place;
- exact nucleus sampling first checks a top-2048 shortlist against the full
  log-partition and falls back per row when it does not contain 95% mass;
- rollout heads use a read-only BF16 mirror, while reliability-triggered
  auxiliary updates retain FP32 master parameters and resync after commit;
- `empty_cache_after_target_train` is disabled to avoid unnecessary cache churn
  on high-memory GPUs.

The supported model launchers are Qwen2.5-1.5B, Qwen3-4B, Qwen2.5-7B and
Llama-3.1-8B. See [EXPERIMENTS.md](EXPERIMENTS.md) for the complete layout and
override list.

Default Qwen2.5-7B commands:

```bash
bash flashgrpo_b200/pretrain_qwen25_7b.sh
DATASET=gsm8k TRAIN_DATA_FRACTION=0.4 bash flashgrpo_b200/train_qwen25_7b.sh
```
