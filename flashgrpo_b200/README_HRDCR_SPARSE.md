# Strict HRDCR on B200

The canonical training path is defined by:

- `configs/<model>/train_hrdcr.yaml`
- `configs/<model>/train_medusa_only.yaml`

The two resolved configs differ only in `method`, `run_name`, and
`reflex.proposal_injection_enabled`. Both variants execute the same MEDUSA tree,
exact-target verification, sparse verifier feedback, per-head fast-state update,
predictive-trust update, and sparse auxiliary-head scheduler.

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

Predictive trust is the positive alignment EMA between the saved state sketch
and the matured verifier innovation sketch, multiplied by
`n / (n + trust_n0)`. The proposal correction is capped by
`relative_rms_delta_base`, so it cannot alter target sampling or exact
verification.

## Auxiliary update order

The target policy is updated first. The first rollout under that new policy
caches at most 512 sparse records, then performs one optimizer step on at most
two MEDUSA heads with the worst target-candidate coverage. This update never
forwards the target backbone and never builds full-vocabulary auxiliary logits.

The logged fields include `aux_update_time`, `aux_optimizer_steps`,
`aux_records_used`, `aux_parameter_delta_rms`, and `aux_overhead_fraction`. If
the update exceeds one percent of rolling rollout time, the scheduler reduces
the record/head budget and increases the policy-version interval.

## Fair ablation

`medusa_only` disables only proposal injection. It deliberately keeps all
feedback and auxiliary work enabled, so its runtime is a valid control for the
benefit of the HRDCR correction itself.
