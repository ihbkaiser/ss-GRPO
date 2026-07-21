# Horizon-Resolved Delayed-Credit Reflex (HRDCR)

HRDCR keeps the sequence-local fast state `m_t` as the central mechanism. It
does not replace it with Anchor Reflex, Chain-MEDUSA, or a tree scheduler.

## Motivation

The previous implementation averaged verifier errors from all MEDUSA horizons
into one state and injected that same direction into every head. This is biased
when head 1 and head 3 have different errors, and their innovations can cancel.
It also discarded target-top tokens missing from the proposal top-k when
constructing the hidden correction.

HRDCR interprets verifier feedback as an online negative gradient. For head `k`:

```math
r_{t,k} \approx W_{LM}^{T}(p_t-q_{t,k})
              = -\nabla_z KL(p_t\|q_{t,k}).
```

This interpretation follows the online-learning view of speculative decoding:
historical verification gradients can act as predictive hints for the next
draft decision. The relevant primary references are [OnlineSpec](https://arxiv.org/abs/2603.12617),
[MEDUSA](https://arxiv.org/abs/2401.10774), and [Hydra](https://arxiv.org/abs/2402.05109).

## Method

### 1. Certified sparse innovation

For proposal top-k tokens, `q(v)` is known exactly from the cached logit and
full log-sum-exp. For a target-top token absent from proposal top-k, its unknown
proposal probability is bounded by the proposal k-th probability `q_upper`.
HRDCR uses:

```math
Delta(v) = p(v)-q(v)                    if v is in proposal top-k
Delta(v) = max(p(v)-q_upper, 0)         otherwise.
```

The second line is conservative and sign-correct. It adds a target mode only
when that mode is guaranteed to be underweighted by the proposal.

### 2. Horizon-resolved fast memories

Each MEDUSA head maintains its own memory:

```math
m_{t,k}=rho m_{t-1,k}+(1-rho) eta P_{t,k}r_{t,k}.
```

`P_{t,k}` is a mixed isotropic/diagonal variance preconditioner. The isotropic
part preserves the first innovation direction; the diagonal part adapts to
persistent coordinate anisotropy.

### 3. Consensus shrinkage

A shared state is updated from the verifier-weighted aggregate innovation.
Heads with little evidence shrink toward this state. Mature heads use the
shared state only when their state direction agrees with it. Consequently,
cross-head agreement transfers information, while disagreement cannot force
all heads into the same correction.

### 4. Delayed-credit verifier calibration

The prediction for horizon `k` matures several tokens after it is proposed.
HRDCR stores the ungated `m_{t,k}` that actually existed at proposal time and
later compares that exact vector with its verifier innovation. The old path
compared the innovation with the newer memory present at verification time,
which assigned credit to the wrong state. Negative or uncorrelated causal
hints reduce the next correction; repeatedly aligned hints receive more trust.

### 5. Bounded proposal-only correction

The effective head memory is injected only into the auxiliary proposal hidden
state. A relative-RMS trust region bounds the residual. The target policy,
target logits, target sampling, and exact verification path are unchanged.

## Experimental m_t variants

Two optional theoretical variants are implemented but disabled in the
recommended config:

- `context_rank > 0` turns `m_t` into a delta-rule fast-weight map from a
  compressed hidden context to verifier correction. The update corrects the
  map residual `r - M phi(h)` instead of accumulating outer products.
- `feedback_objective: hybrid|coverage` adds a mistake-driven candidate-set
  gradient `W[y_target] - W[y_boundary]` when the exact target sample is
  outside a head's retained top-k.

Both preserve exact decoding because they affect proposals only. Small
multi-seed Qwen/GSM8K checks found higher variance and no consistent gain over
the horizon-resolved distribution-gradient state, so the main config keeps
`context_rank: 0`, `feedback_objective: distribution`, and
`coverage_feedback_weight: 0.0` rather than claiming an unverified win.

## Auxiliary refresh

Reliability-triggered auxiliary refresh remains enabled. It updates selected
MEDUSA head parameters after persistent acceptance degradation. `m_t` itself is
gradient-free, sequence-local, reset after each completion, and never saved in
a checkpoint.

## Checkpoint and pretraining

No new Reflex pretraining is required for HRDCR. On B200/Qwen2.5-7B it directly
reuses the standard pretrained MEDUSA checkpoint:

```text
outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b
```

The Anchor-Reflex experiment remains available in its separate config, but it
is not used by HRDCR.

## Important metrics

The JSONL log includes:

```text
average_accept_length
medusa_acceptance_rate
generation_time
tokens_per_sec_generation
reflex.hint_quality_mean
reflex.hint_quality_positive_fraction
reflex.hint_trust_mean
reflex.head_fast_state_rms_mean
reflex.head_effective_updates_mean
reflex.head_hint_trust_mean
```

An improvement is not assumed from the equations alone. Compare wall-clock
tokens/s and average accepted length against the no-Reflex ablation under the
same checkpoint, prompts, random seed, tree budget, and training configuration.

The paired B200 configs are:

```text
flashgrpo_b200/configs/reflexgrpo_horizon_consensus_b200_qwen25_7b_gsm8k.yaml
flashgrpo_b200/configs/ablations/medusa_head_only_no_reflex_fair_qwen25_7b_gsm8k.yaml
```

The ablation inherits the complete HRDCR workload and overrides only the
Reflex enable/feedback/injection switches plus its run name.
