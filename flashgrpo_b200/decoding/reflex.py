from __future__ import annotations

import math
import hashlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from flashgrpo_b200.decoding.medusa_tree import TreePlan, dense_node_count


DEFAULT_DEPTH_BUCKET_BOUNDS: tuple[int, ...] = (128, 256, 512, 1024)


def prompt_depth_memory_key(prompt_token_ids: Iterable[int]) -> str:
    """Stable compact key for prompt-specific Reflex memory."""

    h = hashlib.blake2b(digest_size=16)
    count = 0
    for token in prompt_token_ids:
        value = max(0, int(token))
        h.update(value.to_bytes(4, byteorder="little", signed=False))
        count += 1
    h.update(count.to_bytes(4, byteorder="little", signed=False))
    return h.hexdigest()


def depth_bucket_index(depth: int, bounds: Iterable[int] = DEFAULT_DEPTH_BUCKET_BOUNDS) -> int:
    depth = max(0, int(depth))
    for idx, boundary in enumerate(bounds):
        if depth < int(boundary):
            return idx
    return len(tuple(bounds))


def _rms_clip(vector: torch.Tensor, cap: float, eps: float = 1e-6) -> torch.Tensor:
    vector = torch.nan_to_num(vector.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if float(cap) <= 0.0 or vector.numel() == 0:
        return vector
    rms = vector.square().mean(dim=-1, keepdim=True).sqrt()
    return vector * torch.clamp(float(cap) / rms.clamp_min(eps), max=1.0)


class VerificationUtilityScheduler:
    """Spend tree nodes only on horizons that pay back in verified tokens.

    The scheduler consumes outcomes already produced by exact verification. It
    never accepts a token and never changes target probabilities. Its tiny EMA
    state persists across rollout batches and is reset whenever auxiliary-head
    parameters change; periodic exploration lets a previously weak head recover.
    """

    def __init__(
        self,
        num_heads: int,
        *,
        ema_beta: float = 0.90,
        warmup_rounds: int = 8,
        min_active_heads: int = 2,
        min_depth_acceptance: float = 0.06,
        min_node_utility: float = 0.015,
        exploration_interval: int = 64,
    ):
        self.num_heads = max(0, int(num_heads))
        self.ema_beta = min(0.999, max(0.0, float(ema_beta)))
        self.warmup_rounds = max(0, int(warmup_rounds))
        self.min_active_heads = max(0, min(self.num_heads, int(min_active_heads)))
        self.min_depth_acceptance = max(0.0, float(min_depth_acceptance))
        self.min_node_utility = max(0.0, float(min_node_utility))
        self.exploration_interval = max(0, int(exploration_interval))
        self.rounds = 0
        self.observed_rounds = [0 for _ in range(self.num_heads)]
        self.ema_acceptance = [0.0 for _ in range(self.num_heads)]
        self.ema_node_cost = [0.0 for _ in range(self.num_heads)]
        self.total_trials = [0 for _ in range(self.num_heads)]
        self.total_hits = [0 for _ in range(self.num_heads)]
        self.total_nodes = [0 for _ in range(self.num_heads)]
        self.pruned_rounds = 0
        self.exploration_rounds = 0
        self.last_active_heads = 0
        self.last_topk: list[int] = []

    def _ready(self, head_idx: int) -> bool:
        return self.observed_rounds[head_idx] >= self.warmup_rounds

    def _utility(self, head_idx: int) -> float:
        return self.ema_acceptance[head_idx] / max(self.ema_node_cost[head_idx], 1e-6)

    def adapt(self, plan: TreePlan) -> tuple[TreePlan, dict]:
        topk = [max(1, int(value)) for value in plan.topk_by_depth[: plan.active_heads]]
        if not topk:
            self.last_active_heads = 0
            self.last_topk = []
            return plan, self.to_dict(compact=True)

        explore = bool(
            self.exploration_interval > 0
            and self.rounds > 0
            and self.rounds % self.exploration_interval == 0
        )
        if explore:
            self.exploration_rounds += 1
        else:
            keep = len(topk)
            for head_idx in range(self.min_active_heads, len(topk)):
                if not self._ready(head_idx):
                    continue
                acceptance = self.ema_acceptance[head_idx]
                utility = self._utility(head_idx)
                if acceptance < self.min_depth_acceptance or utility < self.min_node_utility:
                    keep = head_idx
                    break
                if utility < 2.0 * self.min_node_utility:
                    topk[head_idx] = 1
            topk = topk[:keep]
            if len(topk) < int(plan.active_heads):
                self.pruned_rounds += 1

        adapted = TreePlan(
            node_budget_per_seq=int(plan.node_budget_per_seq),
            active_heads=len(topk),
            topk_by_depth=topk,
            actual_nodes=dense_node_count(topk),
            mode=plan.mode,
            layout=plan.layout,
        )
        self.last_active_heads = len(topk)
        self.last_topk = list(topk)
        stats = {
            "enabled": True,
            "rounds": int(self.rounds),
            "pruned_rounds": int(self.pruned_rounds),
            "last_active_heads": int(self.last_active_heads),
            "last_topk": list(self.last_topk),
            "exploration": explore,
        }
        return adapted, stats

    def observe(self, accepted_per_row: list[list[int]], trees: list) -> None:
        if not accepted_per_row or not trees:
            return
        row_count = min(len(accepted_per_row), len(trees))
        if row_count <= 0:
            return
        self.rounds += 1
        beta = self.ema_beta
        # A batch uses one global dense TreePlan, so every row has the same
        # topology. Count each depth once instead of walking B_cur Python lists.
        depth_counts = [0 for _ in range(self.num_heads)]
        for depth in trees[0].depths:
            head_idx = int(depth) - 2
            if 0 <= head_idx < self.num_heads:
                depth_counts[head_idx] += 1
        for head_idx in range(self.num_heads):
            target_depth = head_idx + 2
            node_count = int(depth_counts[head_idx]) * row_count
            if node_count <= 0:
                continue
            hits = sum(1 for row in range(row_count) if len(accepted_per_row[row]) >= target_depth)
            acceptance = hits / row_count
            nodes_per_row = node_count / row_count
            if self.observed_rounds[head_idx] == 0:
                self.ema_acceptance[head_idx] = float(acceptance)
                self.ema_node_cost[head_idx] = float(nodes_per_row)
            else:
                self.ema_acceptance[head_idx] = (
                    beta * self.ema_acceptance[head_idx] + (1.0 - beta) * float(acceptance)
                )
                self.ema_node_cost[head_idx] = (
                    beta * self.ema_node_cost[head_idx] + (1.0 - beta) * float(nodes_per_row)
                )
            self.observed_rounds[head_idx] += 1
            self.total_trials[head_idx] += row_count
            self.total_hits[head_idx] += hits
            self.total_nodes[head_idx] += node_count

    def to_dict(self, *, compact: bool = False) -> dict:
        per_head = {}
        for head_idx in range(self.num_heads):
            trials = self.total_trials[head_idx]
            nodes = self.total_nodes[head_idx]
            values = {
                "observed_rounds": int(self.observed_rounds[head_idx]),
                "acceptance_ema": float(self.ema_acceptance[head_idx]),
                "node_cost_ema": float(self.ema_node_cost[head_idx]),
                "node_utility_ema": float(self._utility(head_idx)) if self.observed_rounds[head_idx] else 0.0,
            }
            if not compact:
                values.update(
                    {
                        "trials": int(trials),
                        "accepted": int(self.total_hits[head_idx]),
                        "acceptance_rate": self.total_hits[head_idx] / max(trials, 1),
                        "verified_nodes": int(nodes),
                        "accepted_per_verified_node": self.total_hits[head_idx] / max(nodes, 1),
                    }
                )
            per_head[str(head_idx + 1)] = values
        return {
            "enabled": True,
            "rounds": int(self.rounds),
            "warmup_rounds": int(self.warmup_rounds),
            "pruned_rounds": int(self.pruned_rounds),
            "exploration_rounds": int(self.exploration_rounds),
            "last_active_heads": int(self.last_active_heads),
            "last_topk": list(self.last_topk),
            "per_head": per_head,
        }


class PromptDepthFeedbackAccumulator:
    """GPU-side rollout feedback grouped by sequence and response-depth bucket."""

    def __init__(
        self,
        *,
        num_sequences: int,
        state_dim: int,
        bucket_bounds: Iterable[int] = DEFAULT_DEPTH_BUCKET_BOUNDS,
        device: torch.device,
        min_weight: float = 0.0,
    ):
        self.num_sequences = int(num_sequences)
        self.state_dim = int(state_dim)
        self.bucket_bounds = tuple(int(x) for x in bucket_bounds)
        self.bucket_count = len(self.bucket_bounds) + 1
        self.min_weight = float(min_weight)
        self.sums = torch.zeros(
            (self.num_sequences, self.bucket_count, self.state_dim),
            device=device,
            dtype=torch.float32,
        )
        self.weights = torch.zeros((self.num_sequences, self.bucket_count), device=device, dtype=torch.float32)
        self.counts = torch.zeros((self.num_sequences, self.bucket_count), device=device, dtype=torch.int32)

    @torch.no_grad()
    def add(
        self,
        *,
        sequence_ids: list[int],
        depths: list[int],
        feedback: torch.Tensor,
        has_feedback: torch.Tensor,
        weights: torch.Tensor,
    ) -> None:
        if not sequence_ids or feedback.numel() == 0:
            return
        if feedback.shape != (len(sequence_ids), self.state_dim):
            raise ValueError("Persistent-memory feedback must be [num_sequences, state_dim]")
        device = self.sums.device
        seq = torch.as_tensor(sequence_ids, dtype=torch.long, device=device)
        bucket = torch.as_tensor(
            [depth_bucket_index(depth, self.bucket_bounds) for depth in depths],
            dtype=torch.long,
            device=device,
        )
        mass = torch.nan_to_num(weights.to(device=device, dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        valid = (
            has_feedback.to(device=device, dtype=torch.bool)
            & mass.gt(float(self.min_weight))
            & seq.ge(0)
            & seq.lt(self.num_sequences)
        )
        if not bool(valid.any().item()):
            return
        flat = seq[valid] * self.bucket_count + bucket[valid].clamp(0, self.bucket_count - 1)
        flat_sums = self.sums.view(self.num_sequences * self.bucket_count, self.state_dim)
        flat_weights = self.weights.view(self.num_sequences * self.bucket_count)
        flat_counts = self.counts.view(self.num_sequences * self.bucket_count)
        cur_feedback = torch.nan_to_num(
            feedback.to(device=device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )[valid]
        cur_mass = mass[valid].clamp_min(0.0)
        flat_sums.index_add_(0, flat, cur_feedback * cur_mass.unsqueeze(-1))
        flat_weights.index_add_(0, flat, cur_mass)
        flat_counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))

    def to_batch(self) -> dict[str, torch.Tensor | list[int]]:
        if self.sums.numel() == 0 or not bool(self.weights.gt(0).any().item()):
            return {}
        return {
            "sums": self.sums.detach().cpu().to(dtype=torch.float16),
            "weights": self.weights.detach().cpu(),
            "counts": self.counts.detach().cpu(),
            "bucket_bounds": list(self.bucket_bounds),
        }


class PersistentPromptDepthMemory:
    """CPU-resident prompt/depth Reflex prior updated only after rollout consensus."""

    def __init__(
        self,
        *,
        state_dim: int,
        bucket_bounds: Iterable[int] = DEFAULT_DEPTH_BUCKET_BOUNDS,
        ema_beta: float = 0.90,
        min_valid_rollouts: int = 4,
        min_consensus: float = 0.35,
        memory_rms_cap: float = 1.0,
        max_strength: float = 0.20,
        min_feedback_weight: float = 0.0,
        age_half_life_iters: float = 512.0,
        min_observations_for_full_strength: int = 32,
        global_enabled: bool = True,
        global_strength_scale: float = 0.05,
    ):
        self.state_dim = int(state_dim)
        self.bucket_bounds = tuple(int(x) for x in bucket_bounds)
        self.bucket_count = len(self.bucket_bounds) + 1
        self.ema_beta = float(ema_beta)
        self.min_valid_rollouts = max(1, int(min_valid_rollouts))
        self.min_consensus = float(min_consensus)
        self.memory_rms_cap = float(memory_rms_cap)
        self.max_strength = float(max_strength)
        self.min_feedback_weight = float(min_feedback_weight)
        self.age_half_life_iters = max(float(age_half_life_iters), 1e-6)
        self.min_observations_for_full_strength = max(1, int(min_observations_for_full_strength))
        self.global_enabled = bool(global_enabled)
        self.global_strength_scale = float(global_strength_scale)
        self.entries: dict[str, dict[int, dict]] = {}
        self.global_entries: dict[int, dict] = {}
        self.total_updates = 0
        self.last_update_stats: dict[str, float | int] = {}
        self.last_materialize_stats: dict[str, float | int] = {}

    def _empty_vector(self) -> torch.Tensor:
        return torch.zeros((self.state_dim,), dtype=torch.float32)

    def _entry_strength(self, entry: dict, current_step: int, *, scale: float = 1.0) -> float:
        observations = int(entry.get("observations", 0) or 0)
        if observations <= 0:
            return 0.0
        consensus = max(0.0, min(1.0, float(entry.get("consensus", 0.0) or 0.0)))
        reliability = min(1.0, observations / float(self.min_observations_for_full_strength))
        age = max(0, int(current_step) - int(entry.get("updated_step", int(current_step)) or 0))
        age_gate = 2.0 ** (-float(age) / self.age_half_life_iters)
        return float(self.max_strength) * float(scale) * consensus * reliability * age_gate

    def materialize(
        self,
        prompt_keys: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        current_step: int = 0,
    ) -> dict[str, torch.Tensor | list[int] | dict]:
        batch = len(prompt_keys)
        if batch <= 0 or self.max_strength <= 0.0:
            return {}
        memory = torch.zeros((batch, self.bucket_count, self.state_dim), dtype=torch.float32)
        strength = torch.zeros((batch, self.bucket_count), dtype=torch.float32)
        prompt_entries_used = 0
        global_entries_used = 0
        for row, key in enumerate(prompt_keys):
            per_prompt = self.entries.get(str(key), {})
            for bucket_idx in range(self.bucket_count):
                vector = self._empty_vector()
                bucket_strength = 0.0
                entry = per_prompt.get(bucket_idx)
                if entry is not None:
                    vector = vector + entry["vector"].float()
                    bucket_strength = max(bucket_strength, self._entry_strength(entry, current_step))
                    prompt_entries_used += int(bucket_strength > 0.0)
                if self.global_enabled:
                    global_entry = self.global_entries.get(bucket_idx)
                    if global_entry is not None:
                        vector = vector + float(self.global_strength_scale) * global_entry["vector"].float()
                        bucket_strength = max(
                            bucket_strength,
                            self._entry_strength(
                                global_entry,
                                current_step,
                                scale=float(self.global_strength_scale),
                            ),
                        )
                        global_entries_used += int(bucket_strength > 0.0)
                if bucket_strength > 0.0:
                    memory[row, bucket_idx] = vector
                    strength[row, bucket_idx] = float(bucket_strength)
        used = int(strength.gt(0).sum().item())
        self.last_materialize_stats = {
            "persistent_memory_prompt_rows": int(batch),
            "persistent_memory_bucket_priors": used,
            "persistent_memory_prompt_entries_used": int(prompt_entries_used),
            "persistent_memory_global_entries_used": int(global_entries_used),
            "persistent_memory_strength_mean": float(strength[strength > 0].mean().item()) if used else 0.0,
            "persistent_memory_strength_max": float(strength.max().item()) if used else 0.0,
        }
        if used == 0:
            return {"stats": dict(self.last_materialize_stats)}
        return {
            "memory": memory.to(device=device, dtype=dtype, non_blocking=True),
            "strength": strength.to(device=device, dtype=torch.float32, non_blocking=True),
            "bucket_bounds": list(self.bucket_bounds),
            "stats": dict(self.last_materialize_stats),
        }

    def _update_entry(self, table: dict[int, dict], bucket_idx: int, vector: torch.Tensor, weight: float, consensus: float, step: int, valid_count: int) -> None:
        vector = _rms_clip(vector, self.memory_rms_cap).detach().cpu().float()
        if vector.numel() != self.state_dim:
            return
        scaled = float(consensus) * vector
        entry = table.get(int(bucket_idx))
        if entry is None:
            table[int(bucket_idx)] = {
                "vector": scaled,
                "observations": int(valid_count),
                "updates": 1,
                "weight": float(weight),
                "consensus": float(consensus),
                "updated_step": int(step),
            }
            return
        beta = float(self.ema_beta)
        entry["vector"] = beta * entry["vector"].float() + (1.0 - beta) * scaled
        entry["observations"] = int(entry.get("observations", 0) or 0) + int(valid_count)
        entry["updates"] = int(entry.get("updates", 0) or 0) + 1
        entry["weight"] = beta * float(entry.get("weight", 0.0) or 0.0) + (1.0 - beta) * float(weight)
        entry["consensus"] = beta * float(entry.get("consensus", 0.0) or 0.0) + (1.0 - beta) * float(consensus)
        entry["updated_step"] = int(step)

    @torch.no_grad()
    def update_from_rollout(
        self,
        *,
        prompt_keys: list[str],
        repeats: int,
        feedback_batch: dict,
        current_step: int,
    ) -> dict[str, float | int]:
        if not feedback_batch or "sums" not in feedback_batch or "weights" not in feedback_batch:
            self.last_update_stats = {
                "persistent_memory_rollout_updates": 0,
                "persistent_memory_consensus_rejects": 0,
                "persistent_memory_valid_groups": 0,
            }
            return dict(self.last_update_stats)
        sums = feedback_batch["sums"].float()
        weights = feedback_batch["weights"].float()
        if sums.dim() != 3 or weights.dim() != 2:
            return {}
        repeats = max(1, int(repeats))
        prompt_count = min(len(prompt_keys), int(sums.shape[0]) // repeats)
        updates = 0
        rejects = 0
        valid_groups = 0
        consensus_sum = 0.0
        weight_sum = 0.0
        for prompt_idx in range(prompt_count):
            key = str(prompt_keys[prompt_idx])
            rows = slice(prompt_idx * repeats, (prompt_idx + 1) * repeats)
            prompt_table = self.entries.setdefault(key, {})
            for bucket_idx in range(min(self.bucket_count, int(sums.shape[1]))):
                bucket_weights = weights[rows, bucket_idx]
                valid = bucket_weights.gt(float(self.min_feedback_weight))
                valid_count = int(valid.sum().item())
                if valid_count < self.min_valid_rollouts:
                    continue
                bucket_sums = sums[rows, bucket_idx, :][valid]
                cur_weights = bucket_weights[valid].clamp_min(1e-6)
                directions = bucket_sums / cur_weights.unsqueeze(-1)
                dir_norm = directions.norm(dim=-1).clamp_min(1e-6)
                unit = directions / dir_norm.unsqueeze(-1)
                consensus_vec = (unit * cur_weights.unsqueeze(-1)).sum(dim=0)
                total_weight = float(cur_weights.sum().item())
                consensus = float(consensus_vec.norm().item() / max(total_weight, 1e-6))
                valid_groups += 1
                if consensus < self.min_consensus:
                    rejects += 1
                    continue
                avg = (bucket_sums.sum(dim=0) / max(total_weight, 1e-6)).float()
                self._update_entry(prompt_table, bucket_idx, avg, total_weight, consensus, int(current_step), valid_count)
                if self.global_enabled:
                    self._update_entry(
                        self.global_entries,
                        bucket_idx,
                        avg,
                        total_weight,
                        consensus,
                        int(current_step),
                        valid_count,
                    )
                updates += 1
                consensus_sum += consensus
                weight_sum += total_weight
        if not any(self.entries.values()):
            self.entries = {key: value for key, value in self.entries.items() if value}
        self.total_updates += updates
        self.last_update_stats = {
            "persistent_memory_rollout_updates": int(updates),
            "persistent_memory_consensus_rejects": int(rejects),
            "persistent_memory_valid_groups": int(valid_groups),
            "persistent_memory_mean_consensus": consensus_sum / max(updates, 1),
            "persistent_memory_update_weight": float(weight_sum),
            "persistent_memory_prompt_count": int(len(self.entries)),
            "persistent_memory_entry_count": int(sum(len(v) for v in self.entries.values())),
            "persistent_memory_global_entry_count": int(len(self.global_entries)),
            "persistent_memory_total_updates": int(self.total_updates),
        }
        return dict(self.last_update_stats)

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "state_dim": self.state_dim,
            "bucket_bounds": list(self.bucket_bounds),
            "ema_beta": self.ema_beta,
            "min_valid_rollouts": self.min_valid_rollouts,
            "min_consensus": self.min_consensus,
            "memory_rms_cap": self.memory_rms_cap,
            "max_strength": self.max_strength,
            "min_feedback_weight": self.min_feedback_weight,
            "age_half_life_iters": self.age_half_life_iters,
            "min_observations_for_full_strength": self.min_observations_for_full_strength,
            "global_enabled": self.global_enabled,
            "global_strength_scale": self.global_strength_scale,
            "entries": self.entries,
            "global_entries": self.global_entries,
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("state_dim", self.state_dim)) != self.state_dim:
            raise ValueError("Persistent Reflex memory hidden size does not match this model")
        self.bucket_bounds = tuple(int(x) for x in state.get("bucket_bounds", self.bucket_bounds))
        self.bucket_count = len(self.bucket_bounds) + 1
        self.entries = state.get("entries", {}) or {}
        self.global_entries = state.get("global_entries", {}) or {}
        self.total_updates = int(state.get("total_updates", 0) or 0)

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return str(path)

    def load(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
        return True

    def stats(self) -> dict[str, float | int]:
        return {
            "persistent_memory_prompt_count": int(len(self.entries)),
            "persistent_memory_entry_count": int(sum(len(v) for v in self.entries.values())),
            "persistent_memory_global_entry_count": int(len(self.global_entries)),
            "persistent_memory_total_updates": int(self.total_updates),
            **dict(self.last_materialize_stats),
            **dict(self.last_update_stats),
        }


@dataclass(slots=True)
class PredictionRecord:
    sequence_id: int
    anchor_pos: int
    target_pos: int
    horizon: int
    top_ids: torch.Tensor
    top_logits: torch.Tensor
    logsumexp: float
    depth: int = 0
    candidate_k: int = 1
    proposal_hidden: torch.Tensor | None = None
    context_key: torch.Tensor | None = None
    fast_hint: torch.Tensor | None = None


class PredictionBuffer:
    """Detached sparse proposal summaries indexed by actual target position."""

    def __init__(self):
        self._records: dict[tuple[int, int], list[PredictionRecord]] = {}

    def add(self, record: PredictionRecord) -> None:
        key = (int(record.sequence_id), int(record.target_pos))
        self._records.setdefault(key, []).append(record)

    @torch.no_grad()
    def add_from_logits(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        logits_by_horizon: list[torch.Tensor],
        top_m: int,
        initial_lengths: torch.Tensor | None = None,
        hidden_by_horizon: list[torch.Tensor] | None = None,
        context_keys: torch.Tensor | None = None,
        fast_hints: torch.Tensor | None = None,
        candidate_topk_by_horizon: Iterable[int] | None = None,
        probabilities_required: bool = True,
    ) -> None:
        if not logits_by_horizon or top_m <= 0 or not sequence_ids:
            return
        anchor_cpu = anchor_positions.detach().to(device="cpu", dtype=torch.long).tolist()
        if initial_lengths is None:
            initial_cpu = anchor_cpu
        else:
            initial_cpu = initial_lengths.detach().to(device="cpu", dtype=torch.long).tolist()
        sequence_cpu = [int(seq_id) for seq_id in sequence_ids]
        candidate_topk = [max(1, int(value)) for value in (candidate_topk_by_horizon or [])]
        context_cpu = (
            context_keys.detach().to(device="cpu", dtype=torch.float16)
            if context_keys is not None
            else None
        )
        hints_cpu = None
        if fast_hints is not None:
            if fast_hints.dim() != 3 or int(fast_hints.shape[0]) != len(sequence_ids):
                raise ValueError("Fast hints must be [batch, num_heads, hidden_size]")
            hints_cpu = fast_hints.detach().to(device="cpu", dtype=torch.float16)

        for head_idx, logits in enumerate(logits_by_horizon):
            # The target LM head predicts t+1. MEDUSA head 1 predicts t+2.
            horizon = head_idx + 2
            if logits.numel() == 0:
                continue
            safe_logits = torch.nan_to_num(
                logits.detach().float(),
                nan=-1.0e9,
                posinf=1.0e9,
                neginf=-1.0e9,
            )
            k = min(int(top_m), int(safe_logits.shape[-1]))
            top_logits, top_ids = torch.topk(safe_logits, k=k, dim=-1)
            log_z = (
                torch.logsumexp(safe_logits, dim=-1)
                if probabilities_required
                else torch.zeros(
                    (safe_logits.shape[0],),
                    device=safe_logits.device,
                    dtype=torch.float32,
                )
            )

            # One bulk device transfer per head. The per-row loop below is CPU-only.
            top_ids_cpu = top_ids.to(device="cpu", dtype=torch.int32)
            top_logits_cpu = top_logits.clamp(min=-65504.0, max=65504.0).to(device="cpu", dtype=torch.float16)
            log_z_cpu = log_z.to(device="cpu", dtype=torch.float32).tolist()
            hidden_cpu = None
            if hidden_by_horizon is not None and head_idx < len(hidden_by_horizon):
                hidden = hidden_by_horizon[head_idx]
                if hidden.dim() == 3 and int(hidden.shape[1]) == 1:
                    hidden = hidden[:, 0]
                hidden_cpu = hidden.detach().to(device="cpu", dtype=torch.float16)
            for row, seq_id in enumerate(sequence_cpu):
                anchor = int(anchor_cpu[row])
                self.add(
                    PredictionRecord(
                        sequence_id=seq_id,
                        anchor_pos=anchor,
                        target_pos=anchor + horizon,
                        horizon=horizon,
                        top_ids=top_ids_cpu[row],
                        top_logits=top_logits_cpu[row],
                        logsumexp=float(log_z_cpu[row]),
                        depth=max(0, anchor - int(initial_cpu[row])),
                        candidate_k=(
                            candidate_topk[head_idx]
                            if head_idx < len(candidate_topk)
                            else 1
                        ),
                        proposal_hidden=(hidden_cpu[row] if hidden_cpu is not None else None),
                        context_key=(context_cpu[row] if context_cpu is not None else None),
                        fast_hint=(
                            hints_cpu[row, head_idx]
                            if hints_cpu is not None and head_idx < int(hints_cpu.shape[1])
                            else None
                        ),
                    )
                )

    def pop_mature(self, sequence_id: int, target_pos: int) -> list[PredictionRecord]:
        return self._records.pop((int(sequence_id), int(target_pos)), [])

    def clear_sequence(self, sequence_id: int) -> None:
        sequence_id = int(sequence_id)
        for key in [key for key in self._records if key[0] == sequence_id]:
            self._records.pop(key, None)

    def __len__(self) -> int:
        return sum(len(records) for records in self._records.values())


@dataclass(slots=True)
class CoveragePredictionBatch:
    """GPU-resident candidate records for the coverage feedback objective."""

    sequence_ids: torch.Tensor
    target_positions: torch.Tensor
    head_indices: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_valid: torch.Tensor
    baseline_candidate_ids: torch.Tensor
    baseline_candidate_valid: torch.Tensor
    probe_valid: torch.Tensor


@dataclass(slots=True)
class CoverageFeedbackRecords:
    """Mature coverage records, grouped by target token without Python rows."""

    group_indices: torch.Tensor
    head_indices: torch.Tensor
    candidate_ids: torch.Tensor
    candidate_valid: torch.Tensor
    baseline_candidate_ids: torch.Tensor
    baseline_candidate_valid: torch.Tensor
    probe_valid: torch.Tensor


class CoveragePredictionBuffer:
    """Tensor-only prediction buffer used by normal coverage rounds.

    Candidate ids stay on the proposal device.  In particular, this path never
    stores proposal logits, a vocabulary log-normalizer, or the full fast hint.
    One object is allocated per sampled verification round rather than per
    sequence/head pair.
    """

    def __init__(self):
        self._batches: list[CoveragePredictionBatch] = []

    @torch.no_grad()
    def add_from_logits(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        logits_by_horizon: list[torch.Tensor],
        candidate_topk_by_horizon: Iterable[int],
        baseline_logits_by_horizon: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequence_ids or not logits_by_horizon:
            device = anchor_positions.device
            zero = torch.zeros((), device=device, dtype=torch.long)
            return zero, zero

        device = logits_by_horizon[0].device
        batch_size = len(sequence_ids)
        sequence = torch.as_tensor(sequence_ids, dtype=torch.long, device=device)
        anchors = anchor_positions.detach().to(device=device, dtype=torch.long)
        if tuple(anchors.shape) != (batch_size,):
            raise ValueError("Coverage anchor positions must be [batch]")
        topk = [max(1, int(value)) for value in candidate_topk_by_horizon]
        head_count = min(len(logits_by_horizon), len(topk))
        if head_count <= 0:
            zero = torch.zeros((), device=device, dtype=torch.long)
            return zero, zero
        max_k = max(
            min(topk[head_idx], int(logits_by_horizon[head_idx].shape[-1]))
            for head_idx in range(head_count)
        )

        record_count = batch_size * head_count
        candidate_ids = torch.full(
            (record_count, max_k),
            -1,
            device=device,
            dtype=torch.long,
        )
        candidate_valid = torch.zeros_like(candidate_ids, dtype=torch.bool)
        probe_enabled = baseline_logits_by_horizon is not None
        baseline_ids = (
            torch.full_like(candidate_ids, -1)
            if probe_enabled
            else torch.empty((record_count, 0), device=device, dtype=torch.long)
        )
        baseline_valid = (
            torch.zeros_like(candidate_valid)
            if probe_enabled
            else torch.empty((record_count, 0), device=device, dtype=torch.bool)
        )
        probe_valid = torch.zeros((record_count,), device=device, dtype=torch.bool)
        head_indices = torch.arange(head_count, device=device, dtype=torch.long).repeat_interleave(
            batch_size
        )
        sequence_ids_flat = sequence.repeat(head_count)
        target_positions = torch.cat(
            [anchors + head_idx + 2 for head_idx in range(head_count)],
            dim=0,
        )

        for head_idx in range(head_count):
            start = head_idx * batch_size
            end = start + batch_size
            k = min(topk[head_idx], int(logits_by_horizon[head_idx].shape[-1]))
            ids = torch.topk(logits_by_horizon[head_idx].detach(), k=k, dim=-1).indices
            candidate_ids[start:end, :k].copy_(ids)
            candidate_valid[start:end, :k] = True
            if baseline_logits_by_horizon is not None and head_idx < len(baseline_logits_by_horizon):
                baseline = baseline_logits_by_horizon[head_idx]
                baseline_k = min(k, int(baseline.shape[-1]))
                baseline_top = torch.topk(baseline.detach(), k=baseline_k, dim=-1).indices
                baseline_ids[start:end, :baseline_k].copy_(baseline_top)
                baseline_valid[start:end, :baseline_k] = True
                probe_valid[start:end] = True

        self._batches.append(
            CoveragePredictionBatch(
                sequence_ids=sequence_ids_flat,
                target_positions=target_positions,
                head_indices=head_indices,
                candidate_ids=candidate_ids,
                candidate_valid=candidate_valid,
                baseline_candidate_ids=baseline_ids,
                baseline_candidate_valid=baseline_valid,
                probe_valid=probe_valid,
            )
        )
        if not probe_enabled:
            zero = torch.zeros((), device=device, dtype=torch.long)
            return zero, zero
        sentinel = torch.iinfo(torch.long).max
        corrected_sorted = candidate_ids.masked_fill(~candidate_valid, sentinel).sort(dim=-1).values
        baseline_sorted = baseline_ids.masked_fill(~baseline_valid, sentinel).sort(dim=-1).values
        changed = corrected_sorted.ne(baseline_sorted).any(dim=-1) & probe_valid
        return changed.sum(), probe_valid.sum()

    @torch.no_grad()
    def pop_mature_batch(
        self,
        sequence_ids: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> CoverageFeedbackRecords:
        if tuple(sequence_ids.shape) != tuple(target_positions.shape):
            raise ValueError("Coverage sequence ids and target positions must align")
        device = sequence_ids.device
        group_count = int(sequence_ids.numel())
        if group_count == 0:
            return self._empty_records(device, 0)

        groups: list[torch.Tensor] = []
        heads: list[torch.Tensor] = []
        candidates: list[torch.Tensor] = []
        candidate_masks: list[torch.Tensor] = []
        baselines: list[torch.Tensor] = []
        baseline_masks: list[torch.Tensor] = []
        probe_masks: list[torch.Tensor] = []
        retained_batches: list[CoveragePredictionBatch] = []
        max_k = max((int(batch.candidate_ids.shape[-1]) for batch in self._batches), default=0)
        baseline_max_k = max(
            (int(batch.baseline_candidate_ids.shape[-1]) for batch in self._batches),
            default=0,
        )

        for batch in self._batches:
            sequence = sequence_ids.to(device=batch.sequence_ids.device, dtype=torch.long)
            positions = target_positions.to(device=batch.target_positions.device, dtype=torch.long)
            pair_match = batch.sequence_ids.unsqueeze(1).eq(sequence.unsqueeze(0)) & batch.target_positions.unsqueeze(
                1
            ).eq(positions.unsqueeze(0))
            matched = pair_match.any(dim=1)
            matched_groups = pair_match.long().argmax(dim=1)[matched]
            groups.append(matched_groups.to(device=device))
            heads.append(batch.head_indices[matched].to(device=device))

            def padded(values: torch.Tensor, fill: int | bool, output_width: int) -> torch.Tensor:
                selected = values[matched].to(device=device)
                width = int(selected.shape[-1])
                if width >= output_width:
                    return selected
                output = torch.full(
                    (int(selected.shape[0]), output_width),
                    fill,
                    device=device,
                    dtype=selected.dtype,
                )
                output[:, :width].copy_(selected)
                return output

            candidates.append(padded(batch.candidate_ids, -1, max_k))
            candidate_masks.append(padded(batch.candidate_valid, False, max_k))
            baselines.append(padded(batch.baseline_candidate_ids, -1, baseline_max_k))
            baseline_masks.append(padded(batch.baseline_candidate_valid, False, baseline_max_k))
            probe_masks.append(batch.probe_valid[matched].to(device=device))

            keep = ~matched
            if bool(keep.any().item()):
                retained_batches.append(
                    CoveragePredictionBatch(
                        sequence_ids=batch.sequence_ids[keep],
                        target_positions=batch.target_positions[keep],
                        head_indices=batch.head_indices[keep],
                        candidate_ids=batch.candidate_ids[keep],
                        candidate_valid=batch.candidate_valid[keep],
                        baseline_candidate_ids=batch.baseline_candidate_ids[keep],
                        baseline_candidate_valid=batch.baseline_candidate_valid[keep],
                        probe_valid=batch.probe_valid[keep],
                    )
                )
        self._batches = retained_batches
        if not groups:
            return self._empty_records(device, max_k)
        return CoverageFeedbackRecords(
            group_indices=torch.cat(groups, dim=0),
            head_indices=torch.cat(heads, dim=0),
            candidate_ids=torch.cat(candidates, dim=0),
            candidate_valid=torch.cat(candidate_masks, dim=0),
            baseline_candidate_ids=torch.cat(baselines, dim=0),
            baseline_candidate_valid=torch.cat(baseline_masks, dim=0),
            probe_valid=torch.cat(probe_masks, dim=0),
        )

    @staticmethod
    def _empty_records(device: torch.device, width: int) -> CoverageFeedbackRecords:
        return CoverageFeedbackRecords(
            group_indices=torch.empty((0,), device=device, dtype=torch.long),
            head_indices=torch.empty((0,), device=device, dtype=torch.long),
            candidate_ids=torch.empty((0, width), device=device, dtype=torch.long),
            candidate_valid=torch.empty((0, width), device=device, dtype=torch.bool),
            baseline_candidate_ids=torch.empty((0, 0), device=device, dtype=torch.long),
            baseline_candidate_valid=torch.empty((0, 0), device=device, dtype=torch.bool),
            probe_valid=torch.empty((0,), device=device, dtype=torch.bool),
        )

    def clear_sequence(self, sequence_id: int) -> None:
        self.clear_sequences([sequence_id])

    def clear_sequences(self, sequence_ids: Iterable[int]) -> None:
        sequence_ids = list(sequence_ids)
        if not sequence_ids:
            return
        retained: list[CoveragePredictionBatch] = []
        for batch in self._batches:
            finished = torch.as_tensor(
                sequence_ids,
                device=batch.sequence_ids.device,
                dtype=torch.long,
            )
            keep = ~batch.sequence_ids.unsqueeze(1).eq(finished.unsqueeze(0)).any(dim=1)
            if bool(keep.any().item()):
                retained.append(
                    CoveragePredictionBatch(
                        sequence_ids=batch.sequence_ids[keep],
                        target_positions=batch.target_positions[keep],
                        head_indices=batch.head_indices[keep],
                        candidate_ids=batch.candidate_ids[keep],
                        candidate_valid=batch.candidate_valid[keep],
                        baseline_candidate_ids=batch.baseline_candidate_ids[keep],
                        baseline_candidate_valid=batch.baseline_candidate_valid[keep],
                        probe_valid=batch.probe_valid[keep],
                    )
                )
        self._batches = retained

    def __len__(self) -> int:
        return sum(int(batch.sequence_ids.numel()) for batch in self._batches)


@dataclass(slots=True)
class SparseFeedbackBatch:
    feedback: torch.Tensor
    has_feedback: torch.Tensor
    effective_mass: torch.Tensor
    head_feedback: torch.Tensor
    head_has_feedback: torch.Tensor
    head_effective_mass: torch.Tensor
    head_context_keys: torch.Tensor
    head_prediction_hint: torch.Tensor
    head_hint_observed: torch.Tensor
    feature_agreement: torch.Tensor
    feature_gate: torch.Tensor
    record_true_probs: torch.Tensor
    record_tv: torch.Tensor
    record_gates: torch.Tensor
    target_top_ids: torch.Tensor
    target_top_logits: torch.Tensor
    target_logsumexp: torch.Tensor
    coverage_head_indices: torch.Tensor | None = None
    coverage_hits: torch.Tensor | None = None
    coverage_probe_valid: torch.Tensor | None = None
    coverage_wins: torch.Tensor | None = None
    coverage_losses: torch.Tensor | None = None


class LMHeadFeedback:
    """Sparse target-distribution feedback in the LM-head hidden space."""

    def __init__(
        self,
        lm_head,
        *,
        target_topk: int = 32,
        union_cap: int = 96,
        tv_gate_low: float = 0.05,
        tv_gate_high: float = 0.20,
        horizon_weight_decay: float = 0.85,
        num_heads: int = 0,
        feature_feedback_weight: float = 0.0,
        feature_agreement_floor: float = 0.0,
        coverage_feedback_weight: float = 0.0,
        feedback_objective: str = "distribution",
        eps: float = 1e-8,
    ):
        self.lm_head = lm_head
        self.target_topk = max(1, int(target_topk))
        self.union_cap = max(self.target_topk, int(union_cap))
        self.tv_gate_low = float(tv_gate_low)
        self.tv_gate_high = max(float(tv_gate_high), self.tv_gate_low + 1e-6)
        self.horizon_weight_decay = float(horizon_weight_decay)
        self.num_heads = max(0, int(num_heads))
        self.feature_feedback_weight = max(0.0, float(feature_feedback_weight))
        self.feature_agreement_floor = min(0.99, max(-1.0, float(feature_agreement_floor)))
        self.feedback_objective = str(feedback_objective).lower()
        if self.feedback_objective not in {"distribution", "hybrid", "coverage"}:
            raise ValueError("Reflex feedback_objective must be distribution, hybrid, or coverage")
        self.coverage_feedback_weight = max(0.0, float(coverage_feedback_weight))
        if self.feedback_objective == "coverage" and self.coverage_feedback_weight == 0.0:
            self.coverage_feedback_weight = 1.0
        self.eps = float(eps)

    @staticmethod
    def _prob_map(ids: torch.Tensor, logits: torch.Tensor, log_z: float) -> dict[int, float]:
        probs = torch.exp(logits.float() - float(log_z)).clamp_(min=0.0, max=1.0)
        return {int(token): float(prob) for token, prob in zip(ids.tolist(), probs.tolist())}

    @torch.no_grad()
    def _compute_coverage_batch(
        self,
        record_groups: list[list[PredictionRecord]],
        true_tokens: list[int],
        *,
        compute_hidden_feedback: bool,
    ) -> SparseFeedbackBatch:
        """Mistake-driven m_t update aligned with exact tree membership.

        The emitted target token is a sample from the exact target policy. If
        it is outside a head's retained candidate set, W[y] - W[x_boundary]
        is the multiclass-perceptron direction that moves y across that set's
        decision boundary. No target top-k or full-vocabulary normalization is
        required for this objective.
        """

        weight = self.lm_head.weight.detach()
        device = weight.device
        group_count = len(record_groups)
        inferred_heads = max(
            (max((int(record.horizon) - 1 for record in records), default=0) for records in record_groups),
            default=0,
        )
        num_heads = max(int(self.num_heads), int(inferred_heads))
        context_rank = max(
            (
                int(record.context_key.numel())
                for records in record_groups
                for record in records
                if record.context_key is not None
            ),
            default=0,
        )
        feedback_width = int(weight.shape[-1]) if compute_hidden_feedback else 0
        shared_coeffs: list[dict[int, float]] = []
        head_coeffs: list[list[dict[int, float]]] = []
        shared_has: list[bool] = []
        shared_mass: list[float] = []
        head_has: list[list[bool]] = []
        head_mass: list[list[float]] = []
        context_values: list[list[torch.Tensor]] = []
        hint_values: list[list[torch.Tensor | None]] = []
        hint_observed: list[list[bool]] = []
        record_true_probs: list[float] = []
        record_tv: list[float] = []
        record_gates: list[float] = []

        for group_idx, records in enumerate(record_groups):
            actual = int(true_tokens[group_idx])
            group_heads = [dict() for _ in range(num_heads)]
            group_head_counts = [0 for _ in range(num_heads)]
            group_head_misses = [0 for _ in range(num_heads)]
            group_context_sum = [torch.zeros((context_rank,), dtype=torch.float32) for _ in range(num_heads)]
            group_hint_sum: list[torch.Tensor | None] = [None for _ in range(num_heads)]
            group_aux_weight = [0.0 for _ in range(num_heads)]
            group_shared: dict[int, float] = {}
            misses = 0
            for record in records:
                head_idx = int(record.horizon) - 2
                if head_idx < 0 or head_idx >= num_heads:
                    continue
                group_head_counts[head_idx] += 1
                candidate_k = min(max(1, int(record.candidate_k)), int(record.top_ids.numel()))
                candidates = [int(token) for token in record.top_ids[:candidate_k].tolist()]
                missed = bool(candidates and actual not in candidates)
                record_true_probs.append(0.0 if missed else 1.0)
                record_tv.append(1.0 if missed else 0.0)
                record_gates.append(1.0 if missed else 0.0)
                if not missed:
                    continue
                misses += 1
                group_head_misses[head_idx] += 1
                boundary = candidates[-1]
                scale = float(self.coverage_feedback_weight)
                head_map = group_heads[head_idx]
                head_map[actual] = head_map.get(actual, 0.0) + scale
                head_map[boundary] = head_map.get(boundary, 0.0) - scale
                shared_scale = scale * (float(self.horizon_weight_decay) ** head_idx)
                group_shared[actual] = group_shared.get(actual, 0.0) + shared_scale
                group_shared[boundary] = group_shared.get(boundary, 0.0) - shared_scale
                group_aux_weight[head_idx] += 1.0
                if record.context_key is not None and context_rank > 0:
                    key = record.context_key.float().reshape(-1)
                    if int(key.numel()) == context_rank:
                        group_context_sum[head_idx].add_(key)
                if record.fast_hint is not None and compute_hidden_feedback:
                    hint = record.fast_hint.float().reshape(-1)
                    if int(hint.numel()) == feedback_width:
                        if group_hint_sum[head_idx] is None:
                            group_hint_sum[head_idx] = hint.clone()
                        else:
                            group_hint_sum[head_idx].add_(hint)

            per_head_context: list[torch.Tensor] = []
            per_head_hint: list[torch.Tensor | None] = []
            per_head_hint_seen: list[bool] = []
            for head_idx in range(num_heads):
                count = max(group_head_misses[head_idx], 1)
                if group_head_misses[head_idx] > 0:
                    group_heads[head_idx] = {
                        token: coeff / count for token, coeff in group_heads[head_idx].items()
                    }
                key = group_context_sum[head_idx]
                key = key / key.norm().clamp_min(self.eps) if key.numel() else key
                per_head_context.append(key)
                hint = group_hint_sum[head_idx]
                if hint is not None:
                    hint = hint / max(group_aux_weight[head_idx], 1.0)
                per_head_hint.append(hint)
                per_head_hint_seen.append(hint is not None)
            if misses > 0:
                group_shared = {token: coeff / misses for token, coeff in group_shared.items()}
            shared_coeffs.append(group_shared)
            head_coeffs.append(group_heads)
            shared_has.append(bool(group_shared))
            shared_mass.append(misses / max(len(records), 1))
            head_has.append([bool(values) for values in group_heads])
            head_mass.append(
                [
                    group_head_misses[idx] / max(group_head_counts[idx], 1)
                    for idx in range(num_heads)
                ]
            )
            context_values.append(per_head_context)
            hint_values.append(per_head_hint)
            hint_observed.append(per_head_hint_seen)

        feedback = torch.zeros((group_count, feedback_width), device=device, dtype=torch.float32)
        head_feedback = torch.zeros(
            (group_count, num_heads, feedback_width),
            device=device,
            dtype=torch.float32,
        )
        if compute_hidden_feedback:
            flat_ids: list[int] = []
            flat_values: list[float] = []
            flat_groups: list[int] = []
            for group_idx, values in enumerate(shared_coeffs):
                for token, coefficient in values.items():
                    flat_ids.append(token)
                    flat_values.append(coefficient)
                    flat_groups.append(group_idx)
            if flat_ids:
                ids = torch.as_tensor(flat_ids, dtype=torch.long, device=device)
                coeff = torch.as_tensor(flat_values, dtype=torch.float32, device=device)
                groups = torch.as_tensor(flat_groups, dtype=torch.long, device=device)
                feedback.index_add_(0, groups, weight.index_select(0, ids).float() * coeff.unsqueeze(-1))

            flat_ids.clear()
            flat_values.clear()
            flat_groups.clear()
            for group_idx, group_heads in enumerate(head_coeffs):
                for head_idx, values in enumerate(group_heads):
                    for token, coefficient in values.items():
                        flat_ids.append(token)
                        flat_values.append(coefficient)
                        flat_groups.append(group_idx * num_heads + head_idx)
            if flat_ids:
                ids = torch.as_tensor(flat_ids, dtype=torch.long, device=device)
                coeff = torch.as_tensor(flat_values, dtype=torch.float32, device=device)
                groups = torch.as_tensor(flat_groups, dtype=torch.long, device=device)
                head_feedback.view(group_count * num_heads, feedback_width).index_add_(
                    0,
                    groups,
                    weight.index_select(0, ids).float() * coeff.unsqueeze(-1),
                )

        head_prediction_hint = torch.zeros_like(head_feedback)
        for group_idx, group in enumerate(hint_values):
            for head_idx, hint in enumerate(group):
                if hint is not None and compute_hidden_feedback:
                    head_prediction_hint[group_idx, head_idx].copy_(
                        hint.to(device=device, dtype=torch.float32)
                    )
        head_context_keys = (
            torch.stack([torch.stack(group, dim=0) for group in context_values], dim=0).to(
                device=device,
                dtype=torch.float32,
            )
            if context_rank > 0 and group_count > 0
            else weight.new_zeros((group_count, num_heads, 0), dtype=torch.float32)
        )
        empty_support = torch.empty((group_count, 0), dtype=torch.int32)
        return SparseFeedbackBatch(
            feedback=feedback,
            has_feedback=torch.as_tensor(shared_has, dtype=torch.bool, device=device),
            effective_mass=torch.as_tensor(shared_mass, dtype=torch.float32, device=device),
            head_feedback=head_feedback,
            head_has_feedback=torch.as_tensor(head_has, dtype=torch.bool, device=device),
            head_effective_mass=torch.as_tensor(head_mass, dtype=torch.float32, device=device),
            head_context_keys=head_context_keys,
            head_prediction_hint=head_prediction_hint,
            head_hint_observed=torch.as_tensor(hint_observed, dtype=torch.bool, device=device),
            feature_agreement=weight.new_zeros((group_count, num_heads), dtype=torch.float32),
            feature_gate=weight.new_zeros((group_count, num_heads), dtype=torch.float32),
            record_true_probs=torch.as_tensor(record_true_probs, dtype=torch.float32, device=device),
            record_tv=torch.as_tensor(record_tv, dtype=torch.float32, device=device),
            record_gates=torch.as_tensor(record_gates, dtype=torch.float32, device=device),
            target_top_ids=empty_support,
            target_top_logits=torch.empty((group_count, 0), dtype=torch.float16),
            target_logsumexp=torch.zeros((group_count,), dtype=torch.float32),
        )

    @torch.no_grad()
    def compute_coverage_tensors(
        self,
        records: CoverageFeedbackRecords,
        true_tokens: torch.Tensor,
        *,
        group_count: int | None = None,
        compute_hidden_feedback: bool = True,
    ) -> SparseFeedbackBatch:
        """Vectorized GPU coverage feedback for normal HRDCR rounds."""

        weight = self.lm_head.weight.detach()
        device = weight.device
        true_tokens = true_tokens.to(device=device, dtype=torch.long)
        group_count = int(true_tokens.numel()) if group_count is None else int(group_count)
        if int(true_tokens.numel()) != group_count:
            raise ValueError("Coverage true tokens must align with feedback groups")
        num_heads = max(0, int(self.num_heads))
        feedback_width = int(weight.shape[-1]) if compute_hidden_feedback else 0
        record_count = int(records.group_indices.numel())

        feedback = torch.zeros((group_count, feedback_width), device=device, dtype=torch.float32)
        head_feedback = torch.zeros(
            (group_count, num_heads, feedback_width),
            device=device,
            dtype=torch.float32,
        )
        head_has_feedback = torch.zeros((group_count, num_heads), device=device, dtype=torch.bool)
        head_effective_mass = torch.zeros((group_count, num_heads), device=device, dtype=torch.float32)
        has_feedback = torch.zeros((group_count,), device=device, dtype=torch.bool)
        effective_mass = torch.zeros((group_count,), device=device, dtype=torch.float32)
        empty_support = torch.empty((group_count, 0), dtype=torch.int32)

        if record_count == 0:
            return SparseFeedbackBatch(
                feedback=feedback,
                has_feedback=has_feedback,
                effective_mass=effective_mass,
                head_feedback=head_feedback,
                head_has_feedback=head_has_feedback,
                head_effective_mass=head_effective_mass,
                head_context_keys=weight.new_zeros((group_count, num_heads, 0), dtype=torch.float32),
                head_prediction_hint=head_feedback.clone(),
                head_hint_observed=head_has_feedback.clone(),
                feature_agreement=weight.new_zeros((group_count, num_heads), dtype=torch.float32),
                feature_gate=weight.new_zeros((group_count, num_heads), dtype=torch.float32),
                record_true_probs=weight.new_zeros((0,), dtype=torch.float32),
                record_tv=weight.new_zeros((0,), dtype=torch.float32),
                record_gates=weight.new_zeros((0,), dtype=torch.float32),
                target_top_ids=empty_support,
                target_top_logits=torch.empty((group_count, 0), dtype=torch.float16),
                target_logsumexp=torch.zeros((group_count,), dtype=torch.float32),
                coverage_head_indices=records.head_indices.to(device=device, dtype=torch.long),
                coverage_hits=torch.empty((0,), device=device, dtype=torch.bool),
                coverage_probe_valid=torch.empty((0,), device=device, dtype=torch.bool),
                coverage_wins=torch.empty((0,), device=device, dtype=torch.bool),
                coverage_losses=torch.empty((0,), device=device, dtype=torch.bool),
            )

        groups = records.group_indices.to(device=device, dtype=torch.long)
        heads = records.head_indices.to(device=device, dtype=torch.long)
        candidates = records.candidate_ids.to(device=device, dtype=torch.long)
        candidate_valid = records.candidate_valid.to(device=device, dtype=torch.bool)
        if candidates.shape != candidate_valid.shape or int(candidates.shape[0]) != record_count:
            raise ValueError("Coverage candidates and valid mask must align")

        actual = true_tokens.index_select(0, groups)
        candidate_count = candidate_valid.sum(dim=-1)
        record_valid = candidate_count.gt(0)
        hits = (candidates.eq(actual.unsqueeze(-1)) & candidate_valid).any(dim=-1) & record_valid
        misses = record_valid & ~hits
        boundary_index = (candidate_count - 1).clamp_min(0).unsqueeze(-1)
        boundary = candidates.gather(1, boundary_index).squeeze(-1)
        flat_head_groups = groups * num_heads + heads

        record_counts = torch.zeros((group_count * num_heads,), device=device, dtype=torch.float32)
        miss_counts = torch.zeros_like(record_counts)
        record_counts.index_add_(0, flat_head_groups, record_valid.float())
        miss_counts.index_add_(0, flat_head_groups, misses.float())
        head_has_feedback.copy_(miss_counts.view(group_count, num_heads).gt(0.0))
        head_effective_mass.copy_(
            (miss_counts / record_counts.clamp_min(1.0)).view(group_count, num_heads)
        )

        group_record_counts = torch.zeros((group_count,), device=device, dtype=torch.float32)
        group_miss_counts = torch.zeros_like(group_record_counts)
        group_record_counts.index_add_(0, groups, record_valid.float())
        group_miss_counts.index_add_(0, groups, misses.float())
        has_feedback.copy_(group_miss_counts.gt(0.0))
        effective_mass.copy_(group_miss_counts / group_record_counts.clamp_min(1.0))

        if compute_hidden_feedback:
            safe_boundary = boundary.clamp(0, int(weight.shape[0]) - 1)
            safe_actual = actual.clamp(0, int(weight.shape[0]) - 1)
            record_feedback = (
                weight.index_select(0, safe_actual).float()
                - weight.index_select(0, safe_boundary).float()
            )
            record_feedback.mul_(misses.unsqueeze(-1))
            record_feedback.mul_(float(self.coverage_feedback_weight))
            head_flat = head_feedback.view(group_count * num_heads, feedback_width)
            head_flat.index_add_(0, flat_head_groups, record_feedback)
            head_flat.div_(miss_counts.clamp_min(1.0).unsqueeze(-1))

            horizon_weight = torch.pow(
                torch.full((record_count,), float(self.horizon_weight_decay), device=device),
                heads.float(),
            )
            feedback.index_add_(0, groups, record_feedback * horizon_weight.unsqueeze(-1))
            feedback.div_(group_miss_counts.clamp_min(1.0).unsqueeze(-1))

        probe_valid = records.probe_valid.to(device=device, dtype=torch.bool)
        baseline_ids = records.baseline_candidate_ids.to(device=device, dtype=torch.long)
        baseline_valid = records.baseline_candidate_valid.to(device=device, dtype=torch.bool)
        baseline_hits = (
            (baseline_ids.eq(actual.unsqueeze(-1)) & baseline_valid).any(dim=-1)
            & baseline_valid.any(dim=-1)
            & probe_valid
        )
        wins = probe_valid & hits & ~baseline_hits
        losses = probe_valid & ~hits & baseline_hits
        return SparseFeedbackBatch(
            feedback=feedback,
            has_feedback=has_feedback,
            effective_mass=effective_mass,
            head_feedback=head_feedback,
            head_has_feedback=head_has_feedback,
            head_effective_mass=head_effective_mass,
            head_context_keys=weight.new_zeros((group_count, num_heads, 0), dtype=torch.float32),
            head_prediction_hint=torch.zeros_like(head_feedback),
            head_hint_observed=torch.zeros_like(head_has_feedback),
            feature_agreement=weight.new_zeros((group_count, num_heads), dtype=torch.float32),
            feature_gate=weight.new_zeros((group_count, num_heads), dtype=torch.float32),
            record_true_probs=hits.float(),
            record_tv=misses.float(),
            record_gates=misses.float(),
            target_top_ids=empty_support,
            target_top_logits=torch.empty((group_count, 0), dtype=torch.float16),
            target_logsumexp=torch.zeros((group_count,), dtype=torch.float32),
            coverage_head_indices=heads,
            coverage_hits=hits,
            coverage_probe_valid=probe_valid,
            coverage_wins=wins,
            coverage_losses=losses,
        )

    @torch.no_grad()
    def compute_batch(
        self,
        record_groups: list[list[PredictionRecord]],
        target_logits: torch.Tensor,
        true_tokens: list[int],
        *,
        compute_hidden_feedback: bool = True,
        target_hidden: torch.Tensor | None = None,
        compute_sparse_teacher: bool = True,
    ) -> SparseFeedbackBatch:
        weight = self.lm_head.weight.detach()
        device = weight.device
        group_count = len(record_groups)
        if group_count == 0:
            empty = weight.new_zeros((0,), dtype=torch.float32)
            empty_heads = max(0, int(self.num_heads))
            return SparseFeedbackBatch(
                feedback=weight.new_zeros((0, weight.shape[-1]), dtype=torch.float32),
                has_feedback=empty.bool(),
                effective_mass=empty,
                head_feedback=weight.new_zeros(
                    (0, empty_heads, weight.shape[-1] if compute_hidden_feedback else 0),
                    dtype=torch.float32,
                ),
                head_has_feedback=torch.zeros((0, empty_heads), device=device, dtype=torch.bool),
                head_effective_mass=weight.new_zeros((0, empty_heads), dtype=torch.float32),
                head_context_keys=weight.new_zeros((0, empty_heads, 0), dtype=torch.float32),
                head_prediction_hint=weight.new_zeros(
                    (0, empty_heads, weight.shape[-1] if compute_hidden_feedback else 0),
                    dtype=torch.float32,
                ),
                head_hint_observed=torch.zeros((0, empty_heads), device=device, dtype=torch.bool),
                feature_agreement=weight.new_zeros((0, empty_heads), dtype=torch.float32),
                feature_gate=weight.new_zeros((0, empty_heads), dtype=torch.float32),
                record_true_probs=empty,
                record_tv=empty,
                record_gates=empty,
                target_top_ids=torch.empty((0, 0), dtype=torch.int32),
                target_top_logits=torch.empty((0, 0), dtype=torch.float16),
                target_logsumexp=torch.empty((0,), dtype=torch.float32),
            )
        if int(target_logits.shape[0]) != group_count or len(true_tokens) != group_count:
            raise ValueError("Sparse feedback groups, target logits, and true tokens must align")
        if target_hidden is not None and int(target_hidden.shape[0]) != group_count:
            raise ValueError("Target hidden states must align with sparse feedback groups")
        if self.feedback_objective == "coverage" and not compute_sparse_teacher:
            return self._compute_coverage_batch(
                record_groups,
                true_tokens,
                compute_hidden_feedback=compute_hidden_feedback,
            )

        safe_target = torch.nan_to_num(
            target_logits.detach().float(),
            nan=-1.0e9,
            posinf=1.0e9,
            neginf=-1.0e9,
        )
        k = min(self.target_topk, int(safe_target.shape[-1]))
        _, target_seed_ids = torch.topk(safe_target, k=k, dim=-1)
        target_log_z = torch.logsumexp(safe_target, dim=-1)
        target_seed_ids_cpu = target_seed_ids.to(device="cpu", dtype=torch.int32)
        target_log_z_cpu = target_log_z.to(device="cpu", dtype=torch.float32).tolist()

        # Store a bidirectional teacher support. Target-only top-k misses a
        # confident wrong draft mode, exactly the failure that lowers
        # rejection-sampling acceptance. Keep all target seeds first, then use
        # the highest proposal probabilities to fill the remaining union cap.
        q_maps_by_group: list[list[dict[int, float]]] = []
        teacher_support: list[list[int]] = []
        support_width = min(self.union_cap, int(safe_target.shape[-1]))
        for group_idx, records in enumerate(record_groups):
            q_maps = [self._prob_map(record.top_ids, record.top_logits, record.logsumexp) for record in records]
            q_maps_by_group.append(q_maps)
            seed = [int(token) for token in target_seed_ids_cpu[group_idx].tolist()]
            seed_set = set(seed)
            proposal_scores: dict[int, float] = {}
            for q_map in q_maps:
                for token, probability in q_map.items():
                    if token not in seed_set:
                        proposal_scores[token] = max(proposal_scores.get(token, 0.0), float(probability))
            proposal_tokens = [
                token
                for token, _ in sorted(proposal_scores.items(), key=lambda item: item[1], reverse=True)
            ]
            teacher_support.append((seed + proposal_tokens)[:support_width])

        target_top_ids = torch.full((group_count, support_width), -1, dtype=torch.int32)
        for group_idx, support in enumerate(teacher_support):
            if support:
                target_top_ids[group_idx, : len(support)] = torch.as_tensor(support, dtype=torch.int32)
        support_ids_device = target_top_ids.to(device=safe_target.device, dtype=torch.long)
        valid_support = support_ids_device.ge(0)
        target_support_logits = torch.gather(safe_target, -1, support_ids_device.clamp_min(0))
        target_support_logits = target_support_logits.masked_fill(~valid_support, -65504.0)
        target_top_logits = target_support_logits.clamp(min=-65504.0, max=65504.0).to(
            device="cpu",
            dtype=torch.float16,
        )

        inferred_heads = max(
            (max((int(record.horizon) - 1 for record in records), default=0) for records in record_groups),
            default=0,
        )
        num_heads = max(int(self.num_heads), int(inferred_heads))
        context_rank = max(
            (
                int(record.context_key.numel())
                for records in record_groups
                for record in records
                if record.context_key is not None
            ),
            default=0,
        )
        support_by_group: list[list[int]] = []
        coeff_by_group: list[list[float]] = []
        head_support_by_group: list[list[list[int]]] = []
        head_coeff_by_group: list[list[list[float]]] = []
        head_has_feedback: list[list[bool]] = []
        head_effective_mass: list[list[float]] = []
        head_context_keys: list[list[torch.Tensor]] = []
        has_feedback: list[bool] = []
        effective_mass: list[float] = []
        record_true_probs: list[float] = []
        record_tv: list[float] = []
        record_gates: list[float] = []

        for group_idx, records in enumerate(record_groups):
            p_map = self._prob_map(
                target_top_ids[group_idx],
                target_top_logits[group_idx],
                float(target_log_z_cpu[group_idx]),
            )
            p_map.pop(-1, None)
            q_maps = q_maps_by_group[group_idx]
            support = set(p_map)

            aggregate = {token: 0.0 for token in support}
            total_weight = 0.0
            gate_sum = 0.0
            per_head_aggregate = [{token: 0.0 for token in support} for _ in range(num_heads)]
            per_head_weight = [0.0 for _ in range(num_heads)]
            per_head_gate_sum = [0.0 for _ in range(num_heads)]
            per_head_count = [0 for _ in range(num_heads)]
            per_head_context_sum = [torch.zeros((context_rank,), dtype=torch.float32) for _ in range(num_heads)]
            per_head_context_weight = [0.0 for _ in range(num_heads)]
            actual_token = int(true_tokens[group_idx])
            for record, q_map in zip(records, q_maps):
                # TV is exact on the partition induced by the retained draft
                # support. Unknown vocabulary entries stay together in one
                # tail bucket, avoiding the double counting caused by treating
                # an unknown cross-distribution probability as zero.
                known = [token for token in q_map if token in p_map]
                p_mass = sum(p_map[token] for token in known)
                q_mass = sum(q_map[token] for token in known)
                tv = 0.5 * (
                    sum(abs(p_map[token] - q_map[token]) for token in known)
                    + abs((1.0 - p_mass) - (1.0 - q_mass))
                )
                # For a target-top token absent from the retained proposal
                # top-k, q(v) is unknown but is upper-bounded by the proposal's
                # k-th probability.  max(p(v)-q_upper, 0) is therefore a
                # conservative, sign-correct innovation.  The old intersection
                # update omitted precisely these confidently missed target
                # modes, weakening W^T(p-q) into a one-sided correction.
                q_upper = (
                    math.exp(float(record.top_logits.float().min()) - float(record.logsumexp))
                    if record.top_logits.numel()
                    else 1.0
                )
                residual = {
                    token: (
                        p_map[token] - q_map[token]
                        if token in q_map
                        else max(0.0, p_map[token] - q_upper)
                    )
                    for token in support
                }
                candidate_k = min(
                    max(1, int(record.candidate_k)),
                    int(record.top_ids.numel()),
                )
                candidate_ids = [int(token) for token in record.top_ids[:candidate_k].tolist()]
                if (
                    self.coverage_feedback_weight > 0.0
                    and candidate_ids
                    and actual_token not in candidate_ids
                ):
                    # A multiclass-perceptron gradient for the exact MEDUSA
                    # event: insert the target-sampled token into the retained
                    # candidate set by moving it above the current boundary.
                    boundary_token = candidate_ids[-1]
                    residual[actual_token] = residual.get(actual_token, 0.0) + float(
                        self.coverage_feedback_weight
                    )
                    residual[boundary_token] = residual.get(boundary_token, 0.0) - float(
                        self.coverage_feedback_weight
                    )
                gate = min(1.0, max(0.0, (tv - self.tv_gate_low) / (self.tv_gate_high - self.tv_gate_low)))
                if self.coverage_feedback_weight > 0.0 and actual_token not in candidate_ids:
                    # A coverage miss is direct exact-verifier evidence even
                    # when sparse-TV happens to fall below its noise gate.
                    gate = max(gate, 1.0)
                head_idx = max(0, int(record.horizon) - 2)
                reliability = self.horizon_weight_decay**head_idx
                contribution_weight = gate * reliability
                if contribution_weight > 0.0:
                    for token, coefficient in residual.items():
                        aggregate[token] = aggregate.get(token, 0.0) + contribution_weight * coefficient
                    total_weight += contribution_weight
                if head_idx < num_heads:
                    per_head_count[head_idx] += 1
                    per_head_gate_sum[head_idx] += gate
                    if gate > 0.0:
                        for token, coefficient in residual.items():
                            per_head_aggregate[head_idx][token] = (
                                per_head_aggregate[head_idx].get(token, 0.0)
                                + gate * coefficient
                            )
                        per_head_weight[head_idx] += gate
                        if record.context_key is not None and context_rank > 0:
                            key = record.context_key.float().reshape(-1)
                            if int(key.numel()) == context_rank:
                                per_head_context_sum[head_idx].add_(key, alpha=gate)
                                per_head_context_weight[head_idx] += gate
                gate_sum += gate
                record_true_probs.append(max(self.eps, q_map.get(actual_token, 0.0)))
                record_tv.append(float(tv))
                record_gates.append(float(gate))

            effective_mass.append(gate_sum / max(len(records), 1))
            group_head_support: list[list[int]] = []
            group_head_coeff: list[list[float]] = []
            group_head_has: list[bool] = []
            group_head_mass: list[float] = []
            group_head_context: list[torch.Tensor] = []
            for head_idx in range(num_heads):
                group_head_mass.append(
                    per_head_gate_sum[head_idx] / max(per_head_count[head_idx], 1)
                )
                if per_head_context_weight[head_idx] > 0.0:
                    key = per_head_context_sum[head_idx] / per_head_context_weight[head_idx]
                    key = key / key.norm().clamp_min(self.eps)
                else:
                    key = torch.zeros((context_rank,), dtype=torch.float32)
                group_head_context.append(key)
                if per_head_weight[head_idx] <= 0.0:
                    group_head_support.append([])
                    group_head_coeff.append([])
                    group_head_has.append(False)
                    continue
                ranked_head = sorted(
                    (
                        (token, coeff / (per_head_weight[head_idx] + self.eps))
                        for token, coeff in per_head_aggregate[head_idx].items()
                    ),
                    key=lambda item: abs(item[1]),
                    reverse=True,
                )[: self.union_cap]
                group_head_support.append([int(token) for token, _ in ranked_head])
                group_head_coeff.append([float(coeff) for _, coeff in ranked_head])
                group_head_has.append(bool(ranked_head))
            head_support_by_group.append(group_head_support)
            head_coeff_by_group.append(group_head_coeff)
            head_has_feedback.append(group_head_has)
            head_effective_mass.append(group_head_mass)
            head_context_keys.append(group_head_context)
            if total_weight <= 0.0:
                support_by_group.append([])
                coeff_by_group.append([])
                has_feedback.append(False)
                continue

            ranked = sorted(
                ((token, coeff / (total_weight + self.eps)) for token, coeff in aggregate.items()),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[: self.union_cap]
            support_by_group.append([int(token) for token, _ in ranked])
            coeff_by_group.append([float(coeff) for _, coeff in ranked])
            has_feedback.append(bool(ranked))

        flat_ids: list[int] = []
        flat_coeff: list[float] = []
        flat_groups: list[int] = []
        for group_idx, (support, coeffs) in enumerate(zip(support_by_group, coeff_by_group)):
            flat_ids.extend(support)
            flat_coeff.extend(coeffs)
            flat_groups.extend([group_idx] * len(support))

        feedback_width = int(weight.shape[-1]) if compute_hidden_feedback else 0
        feedback = torch.zeros((group_count, feedback_width), device=device, dtype=torch.float32)
        if flat_ids and compute_hidden_feedback:
            ids = torch.as_tensor(flat_ids, dtype=torch.long, device=device)
            coeff = torch.as_tensor(flat_coeff, dtype=torch.float32, device=device)
            groups = torch.as_tensor(flat_groups, dtype=torch.long, device=device)
            rows = weight.index_select(0, ids).float()
            feedback.index_add_(0, groups, rows * coeff.unsqueeze(-1))

        head_feedback = torch.zeros(
            (group_count, num_heads, feedback_width),
            device=device,
            dtype=torch.float32,
        )
        head_flat_ids: list[int] = []
        head_flat_coeff: list[float] = []
        head_flat_groups: list[int] = []
        for group_idx, (supports, coeff_groups) in enumerate(
            zip(head_support_by_group, head_coeff_by_group)
        ):
            for head_idx, (support, coeffs) in enumerate(zip(supports, coeff_groups)):
                head_flat_ids.extend(support)
                head_flat_coeff.extend(coeffs)
                head_flat_groups.extend(
                    [group_idx * num_heads + head_idx] * len(support)
                )
        if head_flat_ids and compute_hidden_feedback:
            ids = torch.as_tensor(head_flat_ids, dtype=torch.long, device=device)
            coeff = torch.as_tensor(head_flat_coeff, dtype=torch.float32, device=device)
            groups = torch.as_tensor(head_flat_groups, dtype=torch.long, device=device)
            rows = weight.index_select(0, ids).float()
            head_feedback.view(group_count * num_heads, feedback_width).index_add_(
                0,
                groups,
                rows * coeff.unsqueeze(-1),
            )

        # Delayed verifier credit must evaluate the hint that was available at
        # proposal time, not the newer m_t that happens to exist when the
        # record matures several tokens later. Aggregate those causal hints per
        # (sequence, horizon) and move them back to the GPU in one transfer.
        head_prediction_hint = torch.zeros_like(head_feedback)
        head_hint_weight = torch.zeros(
            (group_count * num_heads,),
            device=device,
            dtype=torch.float32,
        )
        hint_vectors: list[torch.Tensor] = []
        hint_groups: list[int] = []
        hint_weights: list[float] = []
        record_idx = 0
        for group_idx, records in enumerate(record_groups):
            for record in records:
                gate = float(record_gates[record_idx])
                record_idx += 1
                head_idx = int(record.horizon) - 2
                if (
                    gate <= 0.0
                    or head_idx < 0
                    or head_idx >= num_heads
                    or record.fast_hint is None
                    or not compute_hidden_feedback
                ):
                    continue
                hint = record.fast_hint.reshape(-1)
                if int(hint.numel()) != feedback_width:
                    continue
                hint_vectors.append(hint)
                hint_groups.append(group_idx * num_heads + head_idx)
                hint_weights.append(gate)
        if hint_vectors:
            hints = torch.stack(hint_vectors, dim=0).to(device=device, dtype=torch.float32)
            groups = torch.as_tensor(hint_groups, dtype=torch.long, device=device)
            weights = torch.as_tensor(hint_weights, dtype=torch.float32, device=device)
            flat_hints = head_prediction_hint.view(group_count * num_heads, feedback_width)
            flat_hints.index_add_(0, groups, hints * weights.unsqueeze(-1))
            head_hint_weight.index_add_(0, groups, weights)
            flat_hints.div_(head_hint_weight.clamp_min(self.eps).unsqueeze(-1))
        head_hint_observed = head_hint_weight.view(group_count, num_heads).gt(0.0)

        feature_agreement = torch.zeros((group_count, num_heads), device=device, dtype=torch.float32)
        feature_gate = torch.zeros_like(feature_agreement)
        if (
            compute_hidden_feedback
            and target_hidden is not None
            and self.feature_feedback_weight > 0.0
            and feedback_width > 0
        ):
            feature_vectors: list[torch.Tensor] = []
            feature_groups: list[int] = []
            feature_heads: list[int] = []
            feature_weights: list[float] = []
            record_idx = 0
            for group_idx, records in enumerate(record_groups):
                for record in records:
                    gate = float(record_gates[record_idx])
                    record_idx += 1
                    head_idx = int(record.horizon) - 2
                    if (
                        gate <= 0.0
                        or head_idx < 0
                        or head_idx >= num_heads
                        or record.proposal_hidden is None
                    ):
                        continue
                    feature_vectors.append(record.proposal_hidden)
                    feature_groups.append(group_idx)
                    feature_heads.append(head_idx)
                    feature_weights.append(gate)

            if feature_vectors:
                proposal_hidden = torch.stack(feature_vectors, dim=0).to(
                    device=device,
                    dtype=torch.float32,
                )
                group_ids = torch.as_tensor(feature_groups, dtype=torch.long, device=device)
                head_ids = torch.as_tensor(feature_heads, dtype=torch.long, device=device)
                flat_group_ids = group_ids * num_heads + head_ids
                record_weight = torch.as_tensor(feature_weights, dtype=torch.float32, device=device)
                teacher_hidden = target_hidden.detach().to(device=device, dtype=torch.float32).index_select(
                    0,
                    group_ids,
                )
                feature_error = torch.nan_to_num(
                    teacher_hidden - proposal_hidden,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                feature_sum = torch.zeros(
                    (group_count * num_heads, feedback_width),
                    device=device,
                    dtype=torch.float32,
                )
                feature_weight_sum = torch.zeros(
                    (group_count * num_heads,),
                    device=device,
                    dtype=torch.float32,
                )
                feature_sum.index_add_(0, flat_group_ids, feature_error * record_weight.unsqueeze(-1))
                feature_weight_sum.index_add_(0, flat_group_ids, record_weight)
                feature_mean = feature_sum / feature_weight_sum.clamp_min(self.eps).unsqueeze(-1)
                distribution = head_feedback.view(group_count * num_heads, feedback_width)
                distribution_rms = distribution.square().mean(dim=-1).sqrt()
                feature_rms = feature_mean.square().mean(dim=-1).sqrt()
                valid_feature = (
                    feature_weight_sum.gt(0.0)
                    & distribution_rms.gt(self.eps)
                    & feature_rms.gt(self.eps)
                )
                cosine = (distribution * feature_mean).sum(dim=-1) / (
                    distribution.norm(dim=-1).clamp_min(self.eps)
                    * feature_mean.norm(dim=-1).clamp_min(self.eps)
                )
                agreement = torch.clamp(
                    (cosine - float(self.feature_agreement_floor))
                    / max(1.0 - float(self.feature_agreement_floor), self.eps),
                    min=0.0,
                    max=1.0,
                ).masked_fill(~valid_feature, 0.0)
                distribution_unit = distribution / distribution_rms.clamp_min(self.eps).unsqueeze(-1)
                feature_unit = feature_mean / feature_rms.clamp_min(self.eps).unsqueeze(-1)
                fused = distribution_unit + (
                    float(self.feature_feedback_weight) * agreement.unsqueeze(-1) * feature_unit
                )
                fused = fused / fused.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(self.eps)
                distribution.copy_(
                    torch.where(valid_feature.unsqueeze(-1), fused, distribution)
                )
                feature_agreement.copy_(cosine.view(group_count, num_heads).masked_fill(~valid_feature.view(group_count, num_heads), 0.0))
                feature_gate.copy_(agreement.view(group_count, num_heads))

                # The shared m_t follows the same verifier-safe fused signal as
                # the horizon memories instead of retaining the old one-sided
                # distribution aggregate.
                head_mass_tensor = torch.as_tensor(
                    head_effective_mass,
                    dtype=torch.float32,
                    device=device,
                )
                horizon_decay = torch.as_tensor(
                    [self.horizon_weight_decay**idx for idx in range(num_heads)],
                    dtype=torch.float32,
                    device=device,
                )
                shared_weights = head_mass_tensor * horizon_decay.unsqueeze(0)
                feedback = (head_feedback * shared_weights.unsqueeze(-1)).sum(dim=1) / shared_weights.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(self.eps)
                has_feedback = shared_weights.sum(dim=1).gt(0.0).tolist()

        return SparseFeedbackBatch(
            feedback=feedback,
            has_feedback=torch.as_tensor(has_feedback, dtype=torch.bool, device=device),
            effective_mass=torch.as_tensor(effective_mass, dtype=torch.float32, device=device),
            head_feedback=head_feedback,
            head_has_feedback=torch.as_tensor(head_has_feedback, dtype=torch.bool, device=device),
            head_effective_mass=torch.as_tensor(head_effective_mass, dtype=torch.float32, device=device),
            head_context_keys=(
                torch.stack(
                    [torch.stack(group, dim=0) for group in head_context_keys],
                    dim=0,
                ).to(device=device, dtype=torch.float32)
                if context_rank > 0
                else weight.new_zeros((group_count, num_heads, 0), dtype=torch.float32)
            ),
            head_prediction_hint=head_prediction_hint,
            head_hint_observed=head_hint_observed,
            feature_agreement=feature_agreement,
            feature_gate=feature_gate,
            record_true_probs=torch.as_tensor(record_true_probs, dtype=torch.float32, device=device),
            record_tv=torch.as_tensor(record_tv, dtype=torch.float32, device=device),
            record_gates=torch.as_tensor(record_gates, dtype=torch.float32, device=device),
            target_top_ids=target_top_ids,
            target_top_logits=target_top_logits,
            target_logsumexp=torch.as_tensor(target_log_z_cpu, dtype=torch.float32),
        )


class ReflexStateManager:
    """Verifier-driven fast memory, optionally resolved by prediction horizon.

    The legacy path keeps one EMA state per sequence.  Horizon-resolved Reflex
    treats each MEDUSA head as a separate online learner, then shrinks sparse
    head memories toward a shared consensus state.  A directional hint score
    suppresses history when it fails to predict the next verifier innovation.
    """

    def __init__(
        self,
        num_sequences: int,
        fast_state_dim: int,
        *,
        device: torch.device,
        half_life_tokens: float = 48.0,
        eta: float = 0.5,
        feedback_variance_beta: float = 0.99,
        feedback_rms_clip: float = 3.0,
        state_rms_clip: float = 2.0,
        numerical_reset_rms: float = 2.5,
        num_heads: int = 0,
        horizon_resolved: bool = False,
        consensus_strength: float = 0.25,
        consensus_floor: float = 0.0,
        head_shrinkage_updates: float = 8.0,
        preconditioner_mix: float = 0.25,
        hint_quality_beta: float = 0.90,
        hint_quality_floor: float = 0.0,
        hint_quality_temperature: float = 0.10,
        hint_cold_start: float = 0.25,
        context_rank: int = 0,
        context_mix: float = 0.5,
        context_min_mass: float = 1e-3,
        context_learning_rate: float = 0.5,
        eps: float = 1e-6,
    ):
        self.fast_state_dim = int(fast_state_dim)
        self.num_heads = max(0, int(num_heads))
        self.horizon_resolved = bool(horizon_resolved and self.num_heads > 0)
        self.rho = float(2.0 ** (-1.0 / max(float(half_life_tokens), 1e-6)))
        self.eta = float(eta)
        self.feedback_variance_beta = float(feedback_variance_beta)
        self.feedback_rms_clip = float(feedback_rms_clip)
        self.state_rms_clip = float(state_rms_clip)
        self.numerical_reset_rms = float(numerical_reset_rms)
        self.consensus_strength = max(0.0, float(consensus_strength))
        self.consensus_floor = min(0.99, max(-1.0, float(consensus_floor)))
        self.head_shrinkage_updates = max(float(head_shrinkage_updates), 1e-6)
        self.preconditioner_mix = min(1.0, max(0.0, float(preconditioner_mix)))
        self.hint_quality_beta = min(0.9999, max(0.0, float(hint_quality_beta)))
        self.hint_quality_floor = min(0.99, max(-1.0, float(hint_quality_floor)))
        self.hint_quality_temperature = max(float(hint_quality_temperature), 1e-4)
        self.hint_cold_start = min(1.0, max(0.0, float(hint_cold_start)))
        self.context_rank = max(0, int(context_rank)) if self.horizon_resolved else 0
        self.context_mix = min(1.0, max(0.0, float(context_mix)))
        self.context_min_mass = max(float(context_min_mass), 0.0)
        self.context_learning_rate = min(1.0, max(0.0, float(context_learning_rate)))
        self.eps = float(eps)
        count = int(num_sequences)
        if self.horizon_resolved:
            self.states = torch.zeros(
                (count, self.num_heads, self.fast_state_dim),
                device=device,
                dtype=torch.float32,
            )
            self.shared_states = torch.zeros(
                (count, self.fast_state_dim),
                device=device,
                dtype=torch.float32,
            )
            self.feedback_variance = torch.zeros_like(self.states)
            self.feedback_variance_initialized = torch.zeros(
                (count, self.num_heads),
                device=device,
                dtype=torch.bool,
            )
            self.shared_feedback_variance = torch.zeros((count,), device=device, dtype=torch.float32)
            self.shared_feedback_variance_initialized = torch.zeros((count,), device=device, dtype=torch.bool)
            self.shared_effective_updates = torch.zeros((count,), device=device, dtype=torch.float32)
            self.effective_updates = torch.zeros(
                (count, self.num_heads),
                device=device,
                dtype=torch.float32,
            )
            self.hint_quality = torch.zeros_like(self.effective_updates)
            self.hint_quality_initialized = torch.zeros_like(
                self.effective_updates,
                dtype=torch.bool,
            )
            if self.context_rank > 0:
                self.context_numerator = torch.zeros(
                    (
                        count,
                        self.num_heads,
                        self.context_rank,
                        self.fast_state_dim,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
                self.context_denominator = torch.zeros(
                    (count, self.num_heads, self.context_rank),
                    device=device,
                    dtype=torch.float32,
                )
            else:
                self.context_numerator = None
                self.context_denominator = None
        else:
            self.states = torch.zeros((count, self.fast_state_dim), device=device, dtype=torch.float32)
            self.shared_states = None
            self.feedback_variance = torch.zeros((count,), device=device, dtype=torch.float32)
            self.feedback_variance_initialized = torch.zeros((count,), device=device, dtype=torch.bool)
            self.shared_feedback_variance = None
            self.shared_feedback_variance_initialized = None
            self.shared_effective_updates = None
            self.effective_updates = torch.zeros((count,), device=device, dtype=torch.float32)
            self.hint_quality = None
            self.hint_quality_initialized = None
            self.context_numerator = None
            self.context_denominator = None
        self.numerical_reset_count = torch.zeros((), device=device, dtype=torch.long)

    def _ids(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        return torch.as_tensor(list(sequence_ids), dtype=torch.long, device=self.states.device)

    def _hint_trust(self, ids: torch.Tensor) -> torch.Tensor:
        if not self.horizon_resolved:
            return self.effective_updates.new_ones((ids.numel(),))
        quality = self.hint_quality.index_select(0, ids)
        initialized = self.hint_quality_initialized.index_select(0, ids)
        calibrated = torch.relu(
            torch.tanh(
                (quality - float(self.hint_quality_floor))
                / float(self.hint_quality_temperature)
            )
        )
        return torch.where(
            initialized,
            calibrated,
            torch.full_like(calibrated, float(self.hint_cold_start)),
        )

    def get_hint_trust(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            shape = (0, self.num_heads) if self.horizon_resolved else (0,)
            return self.effective_updates.new_zeros(shape)
        return self._hint_trust(ids)

    def _effective_horizon_state(
        self,
        ids: torch.Tensor,
        context_keys: torch.Tensor | None = None,
        *,
        apply_hint_trust: bool = True,
    ) -> torch.Tensor:
        heads = self.states.index_select(0, ids)
        shared = self.shared_states.index_select(0, ids).unsqueeze(1)
        updates = self.effective_updates.index_select(0, ids)
        evidence = 1.0 - torch.exp(-updates / float(self.head_shrinkage_updates))

        head_norm = heads.norm(dim=-1, keepdim=True)
        shared_norm = shared.norm(dim=-1, keepdim=True)
        cosine = (heads * shared).sum(dim=-1, keepdim=True) / (
            head_norm.clamp_min(self.eps) * shared_norm.clamp_min(self.eps)
        )
        agreement = torch.clamp(
            (cosine - float(self.consensus_floor))
            / max(1.0 - float(self.consensus_floor), self.eps),
            min=0.0,
            max=1.0,
        )
        agreement = torch.where(
            head_norm.gt(self.eps) & shared_norm.gt(self.eps),
            agreement,
            torch.ones_like(agreement),
        )
        specific_weight = evidence.unsqueeze(-1)
        shared_weight = (1.0 - evidence).unsqueeze(-1) + (
            float(self.consensus_strength) * evidence.unsqueeze(-1) * agreement
        )
        state = (
            specific_weight * heads + shared_weight * shared
        ) / (specific_weight + shared_weight).clamp_min(self.eps)
        if self.context_rank > 0 and context_keys is not None:
            if tuple(context_keys.shape) != (int(ids.numel()), self.context_rank):
                raise ValueError(
                    "Context query keys must be [num_sequences, context_rank]"
                )
            query = torch.nan_to_num(
                context_keys.to(device=self.states.device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            query = query / query.norm(dim=-1, keepdim=True).clamp_min(self.eps)
            numerator = self.context_numerator.index_select(0, ids)
            denominator = self.context_denominator.index_select(0, ids)
            # Delta-rule fast weights directly represent context -> verifier
            # correction. Unlike the old additive numerator/denominator
            # average, retrieval does not amplify a scarcely observed key.
            retrieved = torch.einsum("br,bkrd->bkd", query, numerator)
            retrieved_mass = torch.einsum("br,bkr->bk", query, denominator).clamp_min(0.0)
            available = retrieved_mass.gt(float(self.context_min_mass))
            kernel_gate = retrieved_mass / (
                retrieved_mass + max(float(self.context_min_mass), self.eps)
            )
            mix = float(self.context_mix) * kernel_gate
            mix = mix.masked_fill(~available, 0.0).unsqueeze(-1)
            state = (1.0 - mix) * state + mix * retrieved
        if apply_hint_trust:
            state = state * self._hint_trust(ids).unsqueeze(-1)
        return state

    def get(
        self,
        sequence_ids: Iterable[int],
        context_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            shape = (0, self.num_heads, self.fast_state_dim) if self.horizon_resolved else (0, self.fast_state_dim)
            return self.states.new_zeros(shape)
        if self.horizon_resolved:
            return self._effective_horizon_state(ids, context_keys=context_keys)
        return self.states.index_select(0, ids)

    def get_raw_horizon_state(
        self,
        sequence_ids: Iterable[int],
        context_keys: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the ungated causal m_t used to score delayed hint quality."""

        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return self.states.new_zeros((0, self.num_heads, self.fast_state_dim))
        if not self.horizon_resolved:
            raise RuntimeError("Raw horizon state requires horizon_resolved=True")
        return self._effective_horizon_state(
            ids,
            context_keys=context_keys,
            apply_hint_trust=False,
        )

    def get_effective_updates(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            shape = (0, self.num_heads) if self.horizon_resolved else (0,)
            return self.effective_updates.new_zeros(shape)
        updates = self.effective_updates.index_select(0, ids)
        if self.horizon_resolved:
            shared = self.shared_effective_updates.index_select(0, ids).unsqueeze(-1)
            updates = torch.maximum(updates, shared)
        return updates

    def get_state_and_effective_updates(
        self,
        sequence_ids: Iterable[int],
        context_keys: torch.Tensor | None = None,
        *,
        apply_hint_trust: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            state_shape = (
                (0, self.num_heads, self.fast_state_dim)
                if self.horizon_resolved
                else (0, self.fast_state_dim)
            )
            update_shape = (0, self.num_heads) if self.horizon_resolved else (0,)
            return self.states.new_zeros(state_shape), self.effective_updates.new_zeros(update_shape)
        state = (
            self._effective_horizon_state(
                ids,
                context_keys=context_keys,
                apply_hint_trust=apply_hint_trust,
            )
            if self.horizon_resolved
            else self.states.index_select(0, ids)
        )
        updates = self.effective_updates.index_select(0, ids)
        if self.horizon_resolved:
            updates = torch.maximum(
                updates,
                self.shared_effective_updates.index_select(0, ids).unsqueeze(-1),
            )
        return state, updates

    def _clip_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rms = state.square().mean(dim=-1).sqrt()
        bad = (~torch.isfinite(rms)) | rms.gt(float(self.numerical_reset_rms))
        state = state.masked_fill(bad.unsqueeze(-1), 0.0)
        rms = state.square().mean(dim=-1).sqrt()
        if self.state_rms_clip > 0.0:
            state = state * torch.clamp(
                float(self.state_rms_clip) / rms.clamp_min(self.eps),
                max=1.0,
            ).unsqueeze(-1)
        return state, bad

    def _normalize_scalar_feedback(
        self,
        ids: torch.Tensor,
        feedback: torch.Tensor,
        valid: torch.Tensor,
        variance: torch.Tensor,
        initialized_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_ms = feedback.square().mean(dim=-1)
        valid_ids = ids[valid]
        old_var = variance.index_select(0, valid_ids)
        initialized = initialized_mask.index_select(0, valid_ids)
        new_var = torch.where(
            initialized,
            float(self.feedback_variance_beta) * old_var
            + (1.0 - float(self.feedback_variance_beta)) * raw_ms[valid],
            raw_ms[valid],
        ).clamp_min_(self.eps)
        normalized = feedback[valid] / torch.sqrt(new_var).unsqueeze(-1)
        normalized_rms = normalized.square().mean(dim=-1).sqrt()
        if self.feedback_rms_clip > 0.0:
            normalized = normalized * torch.clamp(
                float(self.feedback_rms_clip) / normalized_rms.clamp_min(self.eps),
                max=1.0,
            ).unsqueeze(-1)
            normalized_rms = normalized.square().mean(dim=-1).sqrt()
        return normalized, normalized_rms, new_var

    @torch.no_grad()
    def _advance_horizon_token(
        self,
        ids: torch.Tensor,
        feedback: torch.Tensor,
        has_feedback: torch.Tensor,
        effective_mass: torch.Tensor,
        head_feedback: torch.Tensor,
        head_has_feedback: torch.Tensor,
        head_effective_mass: torch.Tensor,
        head_context_keys: torch.Tensor | None,
        head_prediction_hint: torch.Tensor | None,
        head_hint_observed: torch.Tensor | None,
        *,
        feedback_present: bool | None,
    ) -> torch.Tensor:
        batch = int(ids.numel())
        expected = (batch, self.num_heads, self.fast_state_dim)
        if tuple(head_feedback.shape) != expected:
            raise ValueError(f"Head feedback must be {expected}, got {tuple(head_feedback.shape)}")

        current_heads = self.states.index_select(0, ids)
        current_shared = self.shared_states.index_select(0, ids)
        updated_heads = float(self.rho) * current_heads
        updated_shared = float(self.rho) * current_shared
        updated_context_numerator = (
            float(self.rho) * self.context_numerator.index_select(0, ids)
            if self.context_numerator is not None
            else None
        )
        updated_context_denominator = (
            float(self.rho) * self.context_denominator.index_select(0, ids)
            if self.context_denominator is not None
            else None
        )
        if feedback_present is False:
            self.states.index_copy_(0, ids, updated_heads)
            self.shared_states.index_copy_(0, ids, updated_shared)
            if updated_context_numerator is not None:
                self.context_numerator.index_copy_(0, ids, updated_context_numerator)
                self.context_denominator.index_copy_(0, ids, updated_context_denominator)
            return self.states.new_zeros((batch,))

        head_feedback = torch.nan_to_num(
            head_feedback.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        head_has_feedback = head_has_feedback.to(device=self.states.device, dtype=torch.bool)
        head_effective_mass = torch.nan_to_num(
            head_effective_mass.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min_(0.0)
        raw_ms = head_feedback.square().mean(dim=-1)
        valid = head_has_feedback & torch.isfinite(raw_ms) & raw_ms.gt(0.0)
        feedback_rms = self.states.new_zeros((batch,))

        if bool(valid.any().item()):
            rows, heads = valid.nonzero(as_tuple=True)
            global_ids = ids.index_select(0, rows)
            raw = head_feedback[rows, heads]
            old_var = self.feedback_variance[global_ids, heads]
            initialized = self.feedback_variance_initialized[global_ids, heads]
            scalar_var = raw.square().mean(dim=-1, keepdim=True)
            new_var = torch.where(
                initialized.unsqueeze(-1),
                float(self.feedback_variance_beta) * old_var
                + (1.0 - float(self.feedback_variance_beta)) * raw.square(),
                scalar_var.expand_as(raw),
            ).clamp_min_(self.eps)
            isotropic = new_var.mean(dim=-1, keepdim=True)
            mixed_var = (
                (1.0 - float(self.preconditioner_mix)) * isotropic
                + float(self.preconditioner_mix) * new_var
            )
            normalized = raw / mixed_var.sqrt()
            normalized_rms = normalized.square().mean(dim=-1).sqrt()
            if self.feedback_rms_clip > 0.0:
                normalized = normalized * torch.clamp(
                    float(self.feedback_rms_clip) / normalized_rms.clamp_min(self.eps),
                    max=1.0,
                ).unsqueeze(-1)
                normalized_rms = normalized.square().mean(dim=-1).sqrt()

            prior = current_heads[rows, heads]
            if head_prediction_hint is not None and head_hint_observed is not None:
                if tuple(head_prediction_hint.shape) != expected:
                    raise ValueError(f"Head prediction hints must be {expected}")
                if tuple(head_hint_observed.shape) != (batch, self.num_heads):
                    raise ValueError("Head hint mask must be [batch, num_heads]")
                delayed_hint = torch.nan_to_num(
                    head_prediction_hint.to(device=self.states.device, dtype=torch.float32)[rows, heads],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                delayed_observed = head_hint_observed.to(
                    device=self.states.device,
                    dtype=torch.bool,
                )[rows, heads]
                prior = torch.where(delayed_observed.unsqueeze(-1), delayed_hint, prior)
            else:
                delayed_observed = torch.zeros_like(rows, dtype=torch.bool)
            prior_norm = prior.norm(dim=-1)
            innovation_norm = normalized.norm(dim=-1)
            quality_valid = (
                (self.effective_updates[global_ids, heads].gt(0.0) | delayed_observed)
                & prior_norm.gt(self.eps)
                & innovation_norm.gt(self.eps)
            )
            if bool(quality_valid.any().item()):
                q_rows = rows[quality_valid]
                q_heads = heads[quality_valid]
                q_ids = global_ids[quality_valid]
                cosine = (
                    prior[quality_valid] * normalized[quality_valid]
                ).sum(dim=-1) / (
                    prior_norm[quality_valid] * innovation_norm[quality_valid]
                ).clamp_min(self.eps)
                old_quality = self.hint_quality[q_ids, q_heads]
                quality_initialized = self.hint_quality_initialized[q_ids, q_heads]
                new_quality = torch.where(
                    quality_initialized,
                    float(self.hint_quality_beta) * old_quality
                    + (1.0 - float(self.hint_quality_beta)) * cosine,
                    cosine,
                ).clamp(min=-1.0, max=1.0)
                self.hint_quality[q_ids, q_heads] = new_quality
                self.hint_quality_initialized[q_ids, q_heads] = True

            updated_heads[rows, heads] += (
                (1.0 - float(self.rho)) * float(self.eta) * normalized
            )
            if updated_context_numerator is not None and head_context_keys is not None:
                if tuple(head_context_keys.shape) != (batch, self.num_heads, self.context_rank):
                    raise ValueError(
                        "Head context keys must be [batch, num_heads, context_rank]"
                    )
                keys = torch.nan_to_num(
                    head_context_keys.to(device=self.states.device, dtype=torch.float32)[rows, heads],
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                key_norm = keys.norm(dim=-1, keepdim=True)
                context_valid = key_norm.squeeze(-1).gt(self.eps)
                if bool(context_valid.any().item()):
                    c_rows = rows[context_valid]
                    c_heads = heads[context_valid]
                    keys = keys[context_valid] / key_norm[context_valid].clamp_min(self.eps)
                    values = normalized[context_valid]
                    masses = head_effective_mass[c_rows, c_heads]
                    fast_weights = updated_context_numerator[c_rows, c_heads]
                    predicted = torch.einsum("br,brd->bd", keys, fast_weights)
                    delta = values - predicted
                    key_energy = keys.square().sum(dim=-1).clamp_min(self.eps)
                    update_scale = (
                        float(self.context_learning_rate)
                        * masses.clamp(max=1.0)
                        / key_energy
                    )
                    updated_context_numerator[c_rows, c_heads] += (
                        update_scale.view(-1, 1, 1)
                        * keys.unsqueeze(-1)
                        * delta.unsqueeze(-2)
                    )
                    updated_context_denominator[c_rows, c_heads] += (
                        (1.0 - float(self.rho)) * masses.unsqueeze(-1) * keys
                    )
            self.feedback_variance[global_ids, heads] = new_var
            self.feedback_variance_initialized[global_ids, heads] = True
            self.effective_updates[global_ids, heads] += head_effective_mass[rows, heads]
            feedback_rms.index_add_(0, rows, normalized_rms)
            feedback_counts = self.states.new_zeros((batch,))
            feedback_counts.index_add_(0, rows, torch.ones_like(normalized_rms))
            feedback_rms = feedback_rms / feedback_counts.clamp_min(1.0)

        updated_heads, bad_heads = self._clip_state(updated_heads)
        if bool(bad_heads.any().item()):
            rows, heads = bad_heads.nonzero(as_tuple=True)
            global_ids = ids.index_select(0, rows)
            self.feedback_variance[global_ids, heads] = 0.0
            self.feedback_variance_initialized[global_ids, heads] = False
            self.effective_updates[global_ids, heads] = 0.0
            self.hint_quality[global_ids, heads] = 0.0
            self.hint_quality_initialized[global_ids, heads] = False
            if updated_context_numerator is not None:
                updated_context_numerator[rows, heads] = 0.0
                updated_context_denominator[rows, heads] = 0.0
            self.numerical_reset_count.add_(bad_heads.sum())
        self.states.index_copy_(0, ids, updated_heads)
        if updated_context_numerator is not None:
            self.context_numerator.index_copy_(0, ids, updated_context_numerator)
            self.context_denominator.index_copy_(0, ids, updated_context_denominator)

        feedback = torch.nan_to_num(
            feedback.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        has_feedback = has_feedback.to(device=self.states.device, dtype=torch.bool)
        shared_ms = feedback.square().mean(dim=-1)
        shared_valid = has_feedback & torch.isfinite(shared_ms) & shared_ms.gt(0.0)
        if bool(shared_valid.any().item()):
            normalized, _, new_var = self._normalize_scalar_feedback(
                ids,
                feedback,
                shared_valid,
                self.shared_feedback_variance,
                self.shared_feedback_variance_initialized,
            )
            updated_shared[shared_valid] += (
                (1.0 - float(self.rho)) * float(self.eta) * normalized
            )
            valid_ids = ids[shared_valid]
            self.shared_feedback_variance.index_copy_(0, valid_ids, new_var)
            self.shared_feedback_variance_initialized.index_fill_(0, valid_ids, True)
            shared_mass = torch.nan_to_num(
                effective_mass.to(device=self.states.device, dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min_(0.0)
            self.shared_effective_updates.index_add_(0, ids, shared_mass)
        updated_shared, bad_shared = self._clip_state(updated_shared)
        if bool(bad_shared.any().item()):
            bad_ids = ids[bad_shared]
            self.shared_feedback_variance.index_fill_(0, bad_ids, 0.0)
            self.shared_feedback_variance_initialized.index_fill_(0, bad_ids, False)
            self.shared_effective_updates.index_fill_(0, bad_ids, 0.0)
            self.numerical_reset_count.add_(bad_shared.sum())
        self.shared_states.index_copy_(0, ids, updated_shared)
        return feedback_rms

    @torch.no_grad()
    def advance_token(
        self,
        sequence_ids: list[int],
        feedback: torch.Tensor,
        has_feedback: torch.Tensor,
        effective_mass: torch.Tensor,
        *,
        head_feedback: torch.Tensor | None = None,
        head_has_feedback: torch.Tensor | None = None,
        head_effective_mass: torch.Tensor | None = None,
        head_context_keys: torch.Tensor | None = None,
        head_prediction_hint: torch.Tensor | None = None,
        head_hint_observed: torch.Tensor | None = None,
        feedback_present: bool | None = None,
    ) -> torch.Tensor:
        """Apply exactly one causal decay/update for one actual token."""

        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return self.states.new_zeros((0,))
        if feedback.shape != (ids.numel(), self.fast_state_dim):
            raise ValueError("Feedback must be [num_sequences, hidden_size]")
        if self.horizon_resolved:
            if head_feedback is None or head_has_feedback is None or head_effective_mass is None:
                if feedback_present is False:
                    head_feedback = self.states.new_zeros(
                        (ids.numel(), self.num_heads, self.fast_state_dim)
                    )
                    head_has_feedback = torch.zeros(
                        (ids.numel(), self.num_heads),
                        device=self.states.device,
                        dtype=torch.bool,
                    )
                    head_effective_mass = self.effective_updates.new_zeros(
                        (ids.numel(), self.num_heads)
                    )
                else:
                    raise ValueError("Horizon-resolved Reflex requires per-head verifier feedback")
            return self._advance_horizon_token(
                ids,
                feedback,
                has_feedback,
                effective_mass,
                head_feedback,
                head_has_feedback,
                head_effective_mass,
                head_context_keys,
                head_prediction_hint,
                head_hint_observed,
                feedback_present=feedback_present,
            )

        if feedback_present is False:
            current = self.states.index_select(0, ids)
            self.states.index_copy_(0, ids, float(self.rho) * current)
            return self.states.new_zeros((ids.numel(),))

        feedback = torch.nan_to_num(
            feedback.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        has_feedback = has_feedback.to(device=self.states.device, dtype=torch.bool)
        effective_mass = torch.nan_to_num(
            effective_mass.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min_(0.0)
        raw_ms = feedback.square().mean(dim=-1)
        valid = has_feedback & torch.isfinite(raw_ms) & raw_ms.gt(0.0)

        current = self.states.index_select(0, ids)
        updated = float(self.rho) * current
        feedback_rms = torch.zeros_like(raw_ms)
        if bool(valid.any().item()):
            normalized, normalized_rms, new_var = self._normalize_scalar_feedback(
                ids,
                feedback,
                valid,
                self.feedback_variance,
                self.feedback_variance_initialized,
            )
            updated[valid] += (1.0 - float(self.rho)) * float(self.eta) * normalized
            feedback_rms[valid] = normalized_rms
            valid_ids = ids[valid]
            self.feedback_variance.index_copy_(0, valid_ids, new_var)
            self.feedback_variance_initialized.index_fill_(0, valid_ids, True)

        updated, bad = self._clip_state(updated)
        if bool(bad.any().item()):
            bad_ids = ids[bad]
            self.feedback_variance.index_fill_(0, bad_ids, 0.0)
            self.feedback_variance_initialized.index_fill_(0, bad_ids, False)
            self.effective_updates.index_fill_(0, bad_ids, 0.0)
            effective_mass = effective_mass.masked_fill(bad, 0.0)
            self.numerical_reset_count.add_(bad.sum())
        self.states.index_copy_(0, ids, updated)
        self.effective_updates.index_add_(0, ids, effective_mass)
        return feedback_rms

    def reset(self, sequence_ids: Iterable[int]) -> None:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return
        self.states.index_fill_(0, ids, 0.0)
        self.feedback_variance.index_fill_(0, ids, 0.0)
        self.feedback_variance_initialized.index_fill_(0, ids, False)
        self.effective_updates.index_fill_(0, ids, 0.0)
        if self.horizon_resolved:
            self.shared_states.index_fill_(0, ids, 0.0)
            self.shared_feedback_variance.index_fill_(0, ids, 0.0)
            self.shared_feedback_variance_initialized.index_fill_(0, ids, False)
            self.shared_effective_updates.index_fill_(0, ids, 0.0)
            self.hint_quality.index_fill_(0, ids, 0.0)
            self.hint_quality_initialized.index_fill_(0, ids, False)
            if self.context_numerator is not None:
                self.context_numerator.index_fill_(0, ids, 0.0)
                self.context_denominator.index_fill_(0, ids, 0.0)

    def norm_stats(self) -> dict:
        if self.states.numel() == 0:
            return {
                "horizon_resolved": bool(self.horizon_resolved),
                "raw_fast_state_rms": 0.0,
                "fast_state_rms_mean": 0.0,
                "fast_state_rms_p95": 0.0,
                "hint_trust": 0.0,
                "effective_feedback_updates_mean": 0.0,
                "numerical_reset_count": int(self.numerical_reset_count.detach().cpu()),
            }
        state_rms = torch.nan_to_num(
            self.states,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).square().mean(dim=-1).sqrt()
        result = {
            "horizon_resolved": bool(self.horizon_resolved),
            "raw_fast_state_rms": float(state_rms.mean().detach().cpu()),
            "fast_state_rms_mean": float(state_rms.mean().detach().cpu()),
            "fast_state_rms_p95": float(torch.quantile(state_rms.flatten(), 0.95).detach().cpu()),
            "fast_state_norm_mean": float(state_rms.mean().detach().cpu()),
            "fast_state_norm_p95": float(torch.quantile(state_rms.flatten(), 0.95).detach().cpu()),
            "effective_feedback_updates_mean": float(self.effective_updates.mean().detach().cpu()),
            "numerical_reset_count": int(self.numerical_reset_count.detach().cpu()),
        }
        if self.horizon_resolved:
            ids = torch.arange(self.states.shape[0], device=self.states.device)
            trust = self._hint_trust(ids)
            initialized = self.hint_quality_initialized
            quality = self.hint_quality[initialized]
            shared_rms = self.shared_states.square().mean(dim=-1).sqrt()
            result.update(
                {
                    "shared_state_rms_mean": float(shared_rms.mean().detach().cpu()),
                    "hint_quality_mean": float(quality.mean().detach().cpu()) if quality.numel() else 0.0,
                    "hint_quality_positive_fraction": (
                        float(quality.gt(0.0).float().mean().detach().cpu()) if quality.numel() else 0.0
                    ),
                    "hint_trust_mean": float(trust.mean().detach().cpu()),
                    "hint_trust": float(trust.mean().detach().cpu()),
                    "head_fast_state_rms_mean": [
                        float(value)
                        for value in state_rms.mean(dim=0).detach().cpu().tolist()
                    ],
                    "head_effective_updates_mean": [
                        float(value)
                        for value in self.effective_updates.mean(dim=0).detach().cpu().tolist()
                    ],
                    "head_hint_trust_mean": [
                        float(value)
                        for value in trust.mean(dim=0).detach().cpu().tolist()
                    ],
                    "context_memory_rank": int(self.context_rank),
                    "context_memory_mass_mean": (
                        float(self.context_denominator.sum(dim=-1).mean().detach().cpu())
                        if self.context_denominator is not None
                        else 0.0
                    ),
                }
            )
        return result


@dataclass(slots=True)
class ReflexAuxiliaryAnchor:
    sequence_id: int
    anchor_pos: int
    target_pos: int
    horizon: int
    initial_len: int
    hidden: torch.Tensor
    fast_state: torch.Tensor
    reflex_scale: float


class ReflexAuxiliaryRecordBuffer:
    """Bounded detached hidden-state records used only by rare head refreshes."""

    def __init__(self, *, max_records: int = 8192, hidden_dtype: torch.dtype = torch.float16):
        self.max_records = max(0, int(max_records))
        self.hidden_dtype = hidden_dtype
        self._pending: dict[tuple[int, int], list[ReflexAuxiliaryAnchor]] = {}
        self._records: deque[dict] = deque(maxlen=self.max_records or None)

    def add_anchor_predictions(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        initial_lengths: torch.Tensor,
        hidden_states: torch.Tensor,
        fast_states: torch.Tensor | None,
        max_horizon: int,
        reflex_scale: float = 1.0,
    ) -> None:
        if self.max_records <= 0 or max_horizon < 2 or not sequence_ids:
            return
        hidden_cpu = hidden_states.detach().to(device="cpu", dtype=self.hidden_dtype)
        if fast_states is None:
            fast_cpu = torch.zeros((len(sequence_ids), 0), dtype=self.hidden_dtype)
        else:
            fast_cpu = fast_states.detach().to(device="cpu", dtype=self.hidden_dtype)
        anchor_cpu = anchor_positions.detach().to(device="cpu", dtype=torch.long).tolist()
        initial_cpu = initial_lengths.detach().to(device="cpu", dtype=torch.long).tolist()
        for row, raw_seq_id in enumerate(sequence_ids):
            seq_id = int(raw_seq_id)
            anchor = int(anchor_cpu[row])
            for horizon in range(2, int(max_horizon) + 1):
                target_pos = anchor + horizon
                self._pending.setdefault((seq_id, target_pos), []).append(
                    ReflexAuxiliaryAnchor(
                        sequence_id=seq_id,
                        anchor_pos=anchor,
                        target_pos=target_pos,
                        horizon=horizon,
                        initial_len=int(initial_cpu[row]),
                        hidden=hidden_cpu[row],
                        fast_state=fast_cpu[row],
                        reflex_scale=float(reflex_scale),
                    )
                )

    def pop_mature(
        self,
        sequence_id: int,
        target_pos: int,
        generated_tokens: list[int],
        true_token: int,
        teacher: dict | None = None,
    ) -> None:
        if self.max_records <= 0:
            return
        anchors = self._pending.pop((int(sequence_id), int(target_pos)), [])
        for anchor in anchors:
            prev_tokens: list[int] = []
            valid = True
            for abs_pos in range(anchor.anchor_pos + 1, anchor.target_pos):
                rel = abs_pos - anchor.initial_len
                if rel < 0 or rel >= len(generated_tokens):
                    valid = False
                    break
                prev_tokens.append(int(generated_tokens[rel]))
            if not valid:
                continue
            proposal = None
            if teacher:
                proposal = next(
                    (
                        record
                        for record in teacher.get("proposal_records", [])
                        if int(record.anchor_pos) == int(anchor.anchor_pos)
                        and int(record.horizon) == int(anchor.horizon)
                    ),
                    None,
                )
            self._records.append(
                {
                    "hidden": anchor.hidden,
                    "fast_state": anchor.fast_state,
                    "label": int(true_token),
                    "horizon": int(anchor.horizon),
                    "prev_tokens": prev_tokens,
                    "reflex_scale": float(anchor.reflex_scale),
                    "target_top_ids": (
                        teacher["target_top_ids"].clone()
                        if teacher and teacher.get("target_top_ids") is not None
                        else torch.empty((0,), dtype=torch.int32)
                    ),
                    "target_top_logits": (
                        teacher["target_top_logits"].clone()
                        if teacher and teacher.get("target_top_logits") is not None
                        else torch.empty((0,), dtype=torch.float16)
                    ),
                    "target_logsumexp": float(teacher.get("target_logsumexp", 0.0)) if teacher else 0.0,
                    "old_top_ids": proposal.top_ids.clone() if proposal is not None else torch.empty((0,), dtype=torch.int32),
                    "old_top_logits": (
                        proposal.top_logits.clone() if proposal is not None else torch.empty((0,), dtype=torch.float16)
                    ),
                    "old_logsumexp": float(proposal.logsumexp) if proposal is not None else 0.0,
                    "has_sparse_teacher": bool(teacher is not None and proposal is not None),
                }
            )

    def clear_sequence(self, sequence_id: int) -> None:
        sequence_id = int(sequence_id)
        for key in [key for key in self._pending if key[0] == sequence_id]:
            self._pending.pop(key, None)

    def to_batch(self) -> dict[str, torch.Tensor]:
        if not self._records:
            return {}
        max_prev = max((len(item["prev_tokens"]) for item in self._records), default=0)
        max_target_topk = max((int(item["target_top_ids"].numel()) for item in self._records), default=0)
        max_old_topk = max((int(item["old_top_ids"].numel()) for item in self._records), default=0)
        hidden = torch.stack([item["hidden"] for item in self._records], dim=0).contiguous()
        fast_state = torch.stack([item["fast_state"] for item in self._records], dim=0).contiguous()
        labels = torch.tensor([item["label"] for item in self._records], dtype=torch.long)
        horizons = torch.tensor([item["horizon"] for item in self._records], dtype=torch.long)
        reflex_scale = torch.tensor([item["reflex_scale"] for item in self._records], dtype=torch.float32)
        prev_tokens = torch.full((len(self._records), max_prev), -1, dtype=torch.long)
        prev_lens = torch.zeros((len(self._records),), dtype=torch.long)
        target_top_ids = torch.full((len(self._records), max_target_topk), -1, dtype=torch.int32)
        target_top_logits = torch.zeros((len(self._records), max_target_topk), dtype=torch.float16)
        old_top_ids = torch.full((len(self._records), max_old_topk), -1, dtype=torch.int32)
        old_top_logits = torch.zeros((len(self._records), max_old_topk), dtype=torch.float16)
        target_logsumexp = torch.tensor([item["target_logsumexp"] for item in self._records], dtype=torch.float32)
        old_logsumexp = torch.tensor([item["old_logsumexp"] for item in self._records], dtype=torch.float32)
        has_sparse_teacher = torch.tensor([item["has_sparse_teacher"] for item in self._records], dtype=torch.bool)
        for idx, item in enumerate(self._records):
            tokens = item["prev_tokens"]
            prev_lens[idx] = len(tokens)
            if tokens:
                prev_tokens[idx, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
            target_count = int(item["target_top_ids"].numel())
            if target_count:
                target_top_ids[idx, :target_count] = item["target_top_ids"]
                target_top_logits[idx, :target_count] = item["target_top_logits"]
            old_count = int(item["old_top_ids"].numel())
            if old_count:
                old_top_ids[idx, :old_count] = item["old_top_ids"]
                old_top_logits[idx, :old_count] = item["old_top_logits"]
        return {
            "hidden": hidden,
            "fast_state": fast_state,
            "labels": labels,
            "horizons": horizons,
            "reflex_scale": reflex_scale,
            "prev_tokens": prev_tokens,
            "prev_lens": prev_lens,
            "target_top_ids": target_top_ids,
            "target_top_logits": target_top_logits,
            "target_logsumexp": target_logsumexp,
            "old_top_ids": old_top_ids,
            "old_top_logits": old_top_logits,
            "old_logsumexp": old_logsumexp,
            "has_sparse_teacher": has_sparse_teacher,
        }

    def __len__(self) -> int:
        return len(self._records)


class ReflexBatchStats:
    def __init__(self, num_heads: int):
        self.num_heads = int(num_heads)
        self.mature = [0 for _ in range(self.num_heads)]
        self.accepted = [0 for _ in range(self.num_heads)]
        self.ce_sum = [0.0 for _ in range(self.num_heads)]
        self.tv_sum = [0.0 for _ in range(self.num_heads)]
        self.gated = [0 for _ in range(self.num_heads)]
        self.depth_buckets: list[dict[str, dict[str, float]]] = [dict() for _ in range(self.num_heads)]
        self.feedback_rms: list[torch.Tensor] = []
        self.feature_agreements: list[float] = []
        self.feature_gates: list[float] = []
        self._coverage_mature: torch.Tensor | None = None
        self._coverage_hits: torch.Tensor | None = None
        self._coverage_misses: torch.Tensor | None = None
        self._candidate_changed: torch.Tensor | None = None
        self._candidate_probes: torch.Tensor | None = None
        self._reflex_wins: torch.Tensor | None = None
        self._reflex_losses: torch.Tensor | None = None
        self._hrdcr_mature: torch.Tensor | None = None
        self._hrdcr_hits: torch.Tensor | None = None
        self._hrdcr_ce_sum: torch.Tensor | None = None
        self._hrdcr_tv_sum: torch.Tensor | None = None
        self._hrdcr_gated: torch.Tensor | None = None

    @staticmethod
    def _depth_bucket(depth: int) -> str:
        if depth < 128:
            return "0-128"
        if depth < 256:
            return "128-256"
        if depth < 512:
            return "256-512"
        if depth < 1024:
            return "512-1024"
        return "1024+"

    def add_records(
        self,
        records: list[PredictionRecord],
        true_probs: torch.Tensor,
        accepted_flags: list[bool],
        tv: torch.Tensor,
        gates: torch.Tensor,
    ) -> None:
        if not records:
            return
        true_probs_cpu = true_probs.detach().float().clamp_min(1e-8).cpu().tolist()
        tv_cpu = tv.detach().float().cpu().tolist()
        gates_cpu = gates.detach().float().cpu().tolist()
        for idx, record in enumerate(records):
            head_idx = int(record.horizon) - 2
            if head_idx < 0 or head_idx >= self.num_heads:
                continue
            accepted = int(bool(accepted_flags[idx]))
            ce = -math.log(max(float(true_probs_cpu[idx]), 1e-8))
            cur_tv = float(tv_cpu[idx])
            gate = float(gates_cpu[idx])
            self.mature[head_idx] += 1
            self.accepted[head_idx] += accepted
            self.ce_sum[head_idx] += ce
            self.tv_sum[head_idx] += cur_tv
            self.gated[head_idx] += int(gate > 0.0)
            bucket = self._depth_bucket(int(record.depth))
            stats = self.depth_buckets[head_idx].setdefault(
                bucket,
                {"mature": 0.0, "accepted": 0.0, "tv_sum": 0.0},
            )
            stats["mature"] += 1.0
            stats["accepted"] += float(accepted)
            stats["tv_sum"] += cur_tv

    def add_feedback_rms(self, values: torch.Tensor, has_feedback: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        self.feedback_rms.append(values[has_feedback].detach().float())

    def add_hrdcr_feedback(self, batch) -> None:
        """Accumulate strict feedback without synchronizing GPU to CPU."""

        if batch.record_head_indices.numel() == 0:
            return
        heads = batch.record_head_indices.detach().long()
        valid = heads.ge(0) & heads.lt(self.num_heads)
        heads = heads[valid]
        if heads.numel() == 0:
            return
        if self._hrdcr_mature is None:
            self._hrdcr_mature = torch.zeros(
                (self.num_heads,), device=heads.device, dtype=torch.long
            )
            self._hrdcr_hits = torch.zeros_like(self._hrdcr_mature)
            self._hrdcr_gated = torch.zeros_like(self._hrdcr_mature)
            self._hrdcr_ce_sum = torch.zeros(
                (self.num_heads,), device=heads.device, dtype=torch.float32
            )
            self._hrdcr_tv_sum = torch.zeros_like(self._hrdcr_ce_sum)
        self._hrdcr_mature.add_(torch.bincount(heads, minlength=self.num_heads)[: self.num_heads])
        hits = batch.record_candidate_hit.detach().bool()[valid]
        self._hrdcr_hits.add_(
            torch.bincount(heads[hits], minlength=self.num_heads)[: self.num_heads]
        )
        gated = batch.record_severity.detach().float()[valid].gt(0.0)
        self._hrdcr_gated.add_(
            torch.bincount(heads[gated], minlength=self.num_heads)[: self.num_heads]
        )
        self._hrdcr_ce_sum.index_add_(
            0,
            heads,
            -torch.log(batch.record_true_probs.detach().float()[valid].clamp_min(1e-8)),
        )
        self._hrdcr_tv_sum.index_add_(0, heads, batch.record_tv.detach().float()[valid])

    def add_feature_alignment(self, agreement: torch.Tensor, gate: torch.Tensor) -> None:
        if agreement.numel() == 0 or gate.numel() == 0:
            return
        active = gate.gt(0.0)
        if bool(active.any().item()):
            self.feature_agreements.extend(agreement[active].detach().float().cpu().tolist())
            self.feature_gates.extend(gate[active].detach().float().cpu().tolist())

    def add_candidate_probe(self, changed: torch.Tensor, total: torch.Tensor) -> None:
        changed = changed.detach().to(dtype=torch.long)
        total = total.detach().to(device=changed.device, dtype=torch.long)
        if self._candidate_changed is None:
            self._candidate_changed = torch.zeros((), device=changed.device, dtype=torch.long)
            self._candidate_probes = torch.zeros_like(self._candidate_changed)
        self._candidate_changed.add_(changed)
        self._candidate_probes.add_(total)

    def add_coverage_feedback(self, batch: SparseFeedbackBatch) -> None:
        if batch.coverage_head_indices is None or batch.coverage_hits is None:
            return
        heads = batch.coverage_head_indices.detach().long()
        hits = batch.coverage_hits.detach().bool()
        if heads.numel() == 0:
            return
        mature = torch.bincount(heads, minlength=self.num_heads)[: self.num_heads]
        accepted = torch.bincount(heads[hits], minlength=self.num_heads)[: self.num_heads]
        missed = torch.bincount(heads[~hits], minlength=self.num_heads)[: self.num_heads]
        if self._coverage_mature is None:
            self._coverage_mature = torch.zeros_like(mature)
            self._coverage_hits = torch.zeros_like(mature)
            self._coverage_misses = torch.zeros_like(mature)
            self._reflex_wins = torch.zeros((), device=heads.device, dtype=torch.long)
            self._reflex_losses = torch.zeros_like(self._reflex_wins)
        self._coverage_mature.add_(mature)
        self._coverage_hits.add_(accepted)
        self._coverage_misses.add_(missed)
        if batch.coverage_wins is not None:
            self._reflex_wins.add_(batch.coverage_wins.detach().long().sum())
        if batch.coverage_losses is not None:
            self._reflex_losses.add_(batch.coverage_losses.detach().long().sum())

    def to_dict(self) -> dict:
        coverage_mature = [0 for _ in range(self.num_heads)]
        coverage_hits = [0 for _ in range(self.num_heads)]
        coverage_misses = [0 for _ in range(self.num_heads)]
        if self._coverage_mature is not None:
            coverage_mature = [int(value) for value in self._coverage_mature.cpu().tolist()]
            coverage_hits = [int(value) for value in self._coverage_hits.cpu().tolist()]
            coverage_misses = [int(value) for value in self._coverage_misses.cpu().tolist()]
        hrdcr_mature = [0 for _ in range(self.num_heads)]
        hrdcr_hits = [0 for _ in range(self.num_heads)]
        hrdcr_gated = [0 for _ in range(self.num_heads)]
        hrdcr_ce_sum = [0.0 for _ in range(self.num_heads)]
        hrdcr_tv_sum = [0.0 for _ in range(self.num_heads)]
        if self._hrdcr_mature is not None:
            hrdcr_mature = [int(value) for value in self._hrdcr_mature.cpu().tolist()]
            hrdcr_hits = [int(value) for value in self._hrdcr_hits.cpu().tolist()]
            hrdcr_gated = [int(value) for value in self._hrdcr_gated.cpu().tolist()]
            hrdcr_ce_sum = [float(value) for value in self._hrdcr_ce_sum.cpu().tolist()]
            hrdcr_tv_sum = [float(value) for value in self._hrdcr_tv_sum.cpu().tolist()]
        per_head: dict[str, dict] = {}
        total_mature = 0
        total_gated = 0
        for head_idx in range(self.num_heads):
            mature = self.mature[head_idx] + coverage_mature[head_idx] + hrdcr_mature[head_idx]
            accepted = self.accepted[head_idx] + coverage_hits[head_idx] + hrdcr_hits[head_idx]
            total_mature += mature
            gated_count = self.gated[head_idx] + coverage_misses[head_idx] + hrdcr_gated[head_idx]
            total_gated += gated_count
            buckets = {}
            for name, raw in self.depth_buckets[head_idx].items():
                count = int(raw["mature"])
                buckets[name] = {
                    "mature": count,
                    "acceptance_rate": float(raw["accepted"] / max(count, 1)),
                    "sparse_tv": float(raw["tv_sum"] / max(count, 1)),
                }
            per_head[str(head_idx + 1)] = {
                "mature": mature,
                "accepted": accepted,
                "acceptance_rate": accepted / max(mature, 1),
                "rejection_rate": 1.0 - accepted / max(mature, 1) if mature else 0.0,
                "mature_ce": (self.ce_sum[head_idx] + hrdcr_ce_sum[head_idx]) / max(mature, 1),
                "sparse_tv": (self.tv_sum[head_idx] + hrdcr_tv_sum[head_idx]) / max(mature, 1),
                "nonzero_gate_fraction": gated_count / max(mature, 1),
                "depth_buckets": buckets,
            }
        feedback = torch.cat(self.feedback_rms).float().cpu() if self.feedback_rms else None
        if feedback is not None and feedback.numel() == 0:
            feedback = None
        feature_agreement = (
            torch.tensor(self.feature_agreements, dtype=torch.float32)
            if self.feature_agreements
            else None
        )
        feature_gate = torch.tensor(self.feature_gates, dtype=torch.float32) if self.feature_gates else None
        candidate_changed = int(self._candidate_changed.cpu()) if self._candidate_changed is not None else 0
        candidate_probes = int(self._candidate_probes.cpu()) if self._candidate_probes is not None else 0
        reflex_wins = int(self._reflex_wins.cpu()) if self._reflex_wins is not None else 0
        reflex_losses = int(self._reflex_losses.cpu()) if self._reflex_losses is not None else 0
        return {
            "num_reflex_updates": int(total_gated),
            "feedback_rms_mean": float(feedback.mean()) if feedback is not None else 0.0,
            "feedback_rms_p95": float(torch.quantile(feedback, 0.95)) if feedback is not None else 0.0,
            "feedback_norm_mean": float(feedback.mean()) if feedback is not None else 0.0,
            "feedback_norm_p95": float(torch.quantile(feedback, 0.95)) if feedback is not None else 0.0,
            "nonzero_gate_fraction": total_gated / max(total_mature, 1),
            "feature_agreement_mean": (
                float(feature_agreement.mean()) if feature_agreement is not None else 0.0
            ),
            "feature_gate_mean": float(feature_gate.mean()) if feature_gate is not None else 0.0,
            "feature_feedback_count": len(self.feature_gates),
            "candidate_set_changed_fraction": candidate_changed / max(candidate_probes, 1),
            "candidate_set_probe_count": candidate_probes,
            "reflex_win_count": reflex_wins,
            "reflex_loss_count": reflex_losses,
            "per_head": per_head,
        }
