# Strict HRDCR on B200

The canonical training path is defined by:

- `configs/<model>/train_hrdcr.yaml`
- `configs/<model>/train_medusa_only.yaml`

The two resolved configs differ only in `method`, `run_name`, and
`reflex.proposal_injection_enabled`. Both variants execute the same MEDUSA tree,
exact-target verification, sparse verifier feedback, per-head fast-state update,
predictive-trust update, and sparse auxiliary-head scheduler.

## Sparse asymmetric tree

The canonical budget-10 tree allocates one target root plus `[4, 3, 2]` nodes
to MEDUSA Heads 1-3. Head-2 and Head-3 paths are selected globally by cumulative
path score instead of densely expanding every parent. A calibrated Head-3 gate
uses confidence, margin, entropy, path score, acceptance history, and candidate
regret. Rejected Head-3 slots are returned to Head 2; deterministic exploration
continues to collect Head-3 calibration data.

## Rollout feedback

Each MEDUSA horizon owns a state `m[t, k]`. Proposal records remain on GPU and
contain proposal top-L IDs, the actual tree candidates, corrected proposal
hidden, anchor hidden, and a rank-24 sketch of the proposal-time state.

At maturity, the support is:

`TopL(proposal) union TopM(target) union {actual target token}`.

Target and proposal probabilities are normalized on this same support. The
state innovation combines the negative restricted-KL gradient with a soft
candidate-coverage correction. Error TV and missing target mass determine the
EMA update severity. Every update is RMS-normalized, then the resulting state
is projected to `RMS(m) <= 1`.

Injection is eligible only after enough effective updates and proposal-time
alignment observations, with a positive alignment lower confidence bound. The
state supplies direction only. A calibrated confidence chooses a target
correction ratio in `[0.5%, 2%]` of hidden RMS, and the correction is projected
to that exact ratio. Sampled raw/effective counterfactual probes drive a
per-head safety controller; target sampling and exact verification are never
changed.

## Auxiliary update order

The target policy is updated first. The first rollout under that new policy
caches at most 512 sparse records, then performs one optimizer step on selected
MEDUSA heads using candidate regret, restricted KL, and acceptance degradation.
Each selected head receives a minimum quota, and Head 3 is force-included when
it has enough records. This update never
forwards the target backbone and never builds full-vocabulary auxiliary logits.

The logged fields include `aux_update_time`, `aux_optimizer_steps`,
`aux_records_used_by_head`, `head3_aux_records_used`,
`aux_parameter_delta_rms`, and `aux_overhead_fraction`. If
the update exceeds one percent of rolling rollout time, the scheduler reduces
the record/head budget and increases the policy-version interval.

## Fair ablation

`medusa_only` disables only proposal injection. It deliberately keeps all
feedback and auxiliary work enabled, so its runtime is a valid control for the
benefit of the HRDCR correction itself.
