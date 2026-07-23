# Strict HRDCR on B200

Canonical configs:

- `configs/<model>/train_hrdcr.yaml`
- `configs/<model>/train_medusa_only.yaml`

Both modes use the same prompts, seed, sampling parameters, target verifier,
maximum 10-node tree, `cpeak_nodes=512`, and GRPO workload. `medusa_only` is a
lean static-head baseline: HRDCR feedback/injection and sparse auxiliary
training are disabled rather than left running as hidden baseline overhead.

## Head-specific HRDCR

Each MEDUSA horizon owns a delayed-credit state `m[t,k]`. Head 1 starts enabled
with a 1-3% hidden-RMS trust region. Heads 2 and 3 collect verifier feedback
while injection remains disabled. They are enabled only after counterfactual
candidate-mass gain, net wins, alignment confidence, and (for Head 3)
conditional path acceptance provide enough positive evidence.

The proposal-time record stores the state direction and the exact applied
correction/safety ratios. A cheap sparse boundary check compares one important
omitted target token with the current candidate boundary. It cancels a harmful
correction or rescales a helpful one once, without a second full-vocabulary
projection. Target sampling and exact-target verification are unchanged.

## Dynamic 10-node tree

The root counts toward the budget. Layouts are:

- default: `[5,4,0]`
- Head 2 unreliable: `[6,3,0]`
- Head 3 useful: `[4,4,1]`
- Head 3 strongly useful: `[4,3,2]`

Head-3 exploration reallocates the same ten nodes. It runs for 10% of eligible
warmup rounds and 3% after 1024 mature Head-3 path records. Head-3 utility is
measured as acceptance conditioned on its Head-1/2 parent path being accepted.

## Sparse auxiliary learner

Verifier records are retained independently of GRPO reward variance and policy
optimizer steps. An update requires at least 256 cached mature records, eight
rollouts since the previous update, and one head with at least 64 records.
Records are ranked before truncation; the default 256-record budget is focused
on the highest-utility eligible head.

The persistent objective is:

```text
1.00 * raw restricted KL
+ 0.25 * effective restricted KL
+ 1.00 * omitted-token boundary ranking
+ 0.01 * proximal penalty
```

The effective proposal is reconstructed from proposal-time correction
metadata, never from current trust. No target-backbone forward or
full-vocabulary auxiliary distribution is added.

Auxiliary overhead uses recent successful-update history. It needs two
consecutive expensive updates before throttling and four inexpensive updates
before recovering. Insufficient records postpone the update and remain in the
bounded reservoir.

## Metrics

Rollout logs include active and conditional correction ratios, candidate-set
change and mass gain, counterfactual net wins, conditional Head-3 acceptance,
actual tree layouts, per-head cached/used records, auxiliary loss components,
successful optimizer steps, zero-variance reward groups, and auxiliary updates
that happened without a target-policy update.

The requested 15% throughput gain is an experimental target. It must be
measured with paired runs on the same B200; metrics and verifier work are not
altered to manufacture that result.
