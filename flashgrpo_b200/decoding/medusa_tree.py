from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class CandidateTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    scores: list[float] = field(default_factory=list)
    head3_quality: float = 0.0
    head3_gate_passed: bool = False
    head3_exploration: bool = False

    def __post_init__(self) -> None:
        self.children: dict[int, list[int]] = {}
        for idx, parent in enumerate(self.parents):
            if parent >= 0:
                self.children.setdefault(parent, []).append(idx)

    def ancestors_including_self(self, node_idx: int) -> list[int]:
        out = [node_idx]
        parent = self.parents[node_idx]
        while parent >= 0:
            out.append(parent)
            parent = self.parents[parent]
        return list(reversed(out))

    @property
    def node_count(self) -> int:
        return len(self.tokens)

    @property
    def nodes_by_head(self) -> list[int]:
        max_head = max((int(depth) - 1 for depth in self.depths), default=0)
        counts = [0 for _ in range(max_head)]
        for depth in self.depths:
            head_idx = int(depth) - 2
            if head_idx >= 0:
                counts[head_idx] += 1
        return counts


@dataclass
class TreePlan:
    node_budget_per_seq: int
    active_heads: int
    topk_by_depth: list[int]
    actual_nodes: int
    mode: str
    layout: str
    nodes_by_head: list[int] = field(default_factory=list)
    min_head3_nodes: int = 0
    head3_min_budget: int = 8
    branch_score_temperature: float = 1.0
    diversity_penalty: float = 0.0


@dataclass(slots=True)
class Head3GateResult:
    gate_mask: torch.Tensor
    exploration_mask: torch.Tensor
    quality: torch.Tensor
    expected_acceptance: torch.Tensor


class Head3QualityCalibrator:
    """Online confidence calibration for the third MEDUSA horizon.

    Statistics stay on the proposal device. Scalar conversion only happens in
    ``summary`` at rollout end, so the gate does not synchronize CUDA in the
    decoding loop.
    """

    def __init__(
        self,
        *,
        num_bins: int = 10,
        min_calibration_records: int = 1024,
        exploration_fraction: float = 0.10,
        node_cost: float = 0.35,
        ema_beta: float = 0.95,
        top1_weight: float = 0.30,
        margin_weight: float = 0.20,
        entropy_weight: float = 0.15,
        path_weight: float = 0.10,
        acceptance_weight: float = 0.20,
        regret_weight: float = 0.15,
    ):
        self.num_bins = max(2, int(num_bins))
        self.min_calibration_records = max(1, int(min_calibration_records))
        self.exploration_fraction = min(1.0, max(0.0, float(exploration_fraction)))
        self.node_cost = float(node_cost)
        self.ema_beta = min(0.9999, max(0.0, float(ema_beta)))
        self.weights = torch.tensor(
            [
                float(top1_weight),
                float(margin_weight),
                float(entropy_weight),
                float(path_weight),
                float(acceptance_weight),
                float(regret_weight),
            ],
            dtype=torch.float32,
        )
        self._device: torch.device | None = None
        self._bin_mature: torch.Tensor | None = None
        self._bin_accepted: torch.Tensor | None = None
        self._bin_mass_gain: torch.Tensor | None = None
        self._bin_probe_count: torch.Tensor | None = None
        self._acceptance_ema: torch.Tensor | None = None
        self._regret_ema: torch.Tensor | None = None
        self._counters: torch.Tensor | None = None
        self._exploration_cursor = 0

    def _ensure_device(self, device: torch.device) -> None:
        if self._device == device and self._bin_mature is not None:
            return
        self._device = device
        self.weights = self.weights.to(device=device)
        self._bin_mature = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
        self._bin_accepted = torch.zeros_like(self._bin_mature)
        self._bin_mass_gain = torch.zeros_like(self._bin_mature)
        self._bin_probe_count = torch.zeros_like(self._bin_mature)
        self._acceptance_ema = torch.full((), 0.5, device=device, dtype=torch.float32)
        self._regret_ema = torch.zeros((), device=device, dtype=torch.float32)
        # eligible, pass, reject, exploration
        self._counters = torch.zeros(4, device=device, dtype=torch.long)

    @torch.no_grad()
    def select(
        self,
        logits: torch.Tensor,
        *,
        cumulative_path_score: torch.Tensor | None = None,
        eligible: torch.Tensor | None = None,
    ) -> Head3GateResult:
        device = logits.device
        self._ensure_device(device)
        batch = int(logits.shape[0])
        if eligible is None:
            eligible = torch.ones(batch, device=device, dtype=torch.bool)
        else:
            eligible = eligible.to(device=device, dtype=torch.bool)

        values = torch.topk(logits.float(), k=min(16, int(logits.shape[-1])), dim=-1).values
        log_z = torch.logsumexp(logits.float(), dim=-1)
        top1 = torch.exp(values[:, 0] - log_z).clamp(0.0, 1.0)
        margin = values[:, 0] - values[:, min(1, int(values.shape[1]) - 1)]
        margin_score = torch.sigmoid(margin)
        shortlist_prob = torch.exp(values - log_z.unsqueeze(-1))
        tail = (1.0 - shortlist_prob.sum(dim=-1)).clamp_min(0.0)
        entropy = -(shortlist_prob * (values - log_z.unsqueeze(-1))).sum(dim=-1)
        entropy -= tail * torch.log(tail.clamp_min(1e-8))
        entropy_score = 1.0 - (entropy / math.log(max(2, int(values.shape[1]) + 1))).clamp(0.0, 1.0)
        if cumulative_path_score is None:
            path_score = top1
        else:
            path_score = torch.sigmoid(cumulative_path_score.to(device=device, dtype=torch.float32))
        recent_acceptance = self._acceptance_ema.expand_as(top1)
        regret_score = (1.0 - self._regret_ema.clamp(0.0, 1.0)).expand_as(top1)
        features = torch.stack(
            (top1, margin_score, entropy_score, path_score, recent_acceptance, regret_score),
            dim=-1,
        )
        denom = self.weights.abs().sum().clamp_min(1e-6)
        quality = (features * self.weights).sum(dim=-1).div(denom).clamp(0.0, 1.0)
        bins = torch.clamp((quality * self.num_bins).long(), max=self.num_bins - 1)
        mature = self._bin_mature.index_select(0, bins)
        empirical = self._bin_accepted.index_select(0, bins) / mature.clamp_min(1.0)
        # Smooth cold bins towards the model-derived quality score.
        blend = mature / (mature + 16.0)
        calibrated = self._bin_mature.sum().ge(float(self.min_calibration_records))
        calibrated_expected = blend * empirical + (1.0 - blend) * quality
        expected = torch.where(calibrated, calibrated_expected, quality)
        gate = eligible & expected.gt(self.node_cost)

        period = 1000
        explore_slots = max(0, min(period, int(round(self.exploration_fraction * period))))
        cursor = self._exploration_cursor
        offsets = torch.arange(batch, device=device, dtype=torch.long) + int(cursor)
        exploration = eligible & offsets.remainder(period).lt(explore_slots)
        self._exploration_cursor = (cursor + batch) % period
        gate |= exploration
        self._counters[0].add_(eligible.sum())
        self._counters[1].add_((gate & eligible).sum())
        self._counters[2].add_((~gate & eligible).sum())
        self._counters[3].add_(exploration.sum())
        return Head3GateResult(gate, exploration, quality, expected)

    @torch.no_grad()
    def observe(
        self,
        quality: torch.Tensor,
        accepted: torch.Tensor,
        candidate_regret: torch.Tensor,
        candidate_mass_gain: torch.Tensor | None = None,
    ) -> None:
        if quality.numel() == 0:
            return
        device = quality.device
        self._ensure_device(device)
        quality = quality.detach().float().clamp(0.0, 1.0)
        accepted = accepted.detach().float().to(device=device)
        regret = candidate_regret.detach().float().to(device=device).clamp(0.0, 1.0)
        bins = torch.clamp((quality * self.num_bins).long(), max=self.num_bins - 1)
        ones = torch.ones_like(quality)
        self._bin_mature.index_add_(0, bins, ones)
        self._bin_accepted.index_add_(0, bins, accepted)
        if candidate_mass_gain is not None and candidate_mass_gain.numel() == quality.numel():
            self._bin_mass_gain.index_add_(0, bins, candidate_mass_gain.detach().float().to(device))
        beta = self.ema_beta
        self._acceptance_ema.mul_(beta).add_((1.0 - beta) * accepted.mean())
        self._regret_ema.mul_(beta).add_((1.0 - beta) * regret.mean())

    @torch.no_grad()
    def observe_mass_gain(self, quality: torch.Tensor, mass_gain: torch.Tensor) -> None:
        if quality.numel() == 0:
            return
        device = quality.device
        self._ensure_device(device)
        quality = quality.detach().float().clamp(0.0, 1.0)
        bins = torch.clamp((quality * self.num_bins).long(), max=self.num_bins - 1)
        self._bin_mass_gain.index_add_(0, bins, mass_gain.detach().float().to(device))
        self._bin_probe_count.index_add_(0, bins, torch.ones_like(quality))

    def snapshot(self) -> torch.Tensor | None:
        return self._counters.detach().clone() if self._counters is not None else None

    def summary(self, since: torch.Tensor | None = None) -> dict[str, float | int]:
        if self._counters is None:
            return {
                "head3_eligible_count": 0,
                "head3_quality_gate_pass_count": 0,
                "head3_quality_gate_reject_count": 0,
                "head3_exploration_count": 0,
                "head3_calibration_records": 0,
                "head3_acceptance_ema": 0.0,
                "head3_candidate_regret_ema": 0.0,
                "head3_candidate_target_mass_gain": 0.0,
            }
        counters_tensor = self._counters
        if since is not None:
            counters_tensor = counters_tensor - since.to(device=counters_tensor.device)
        counters = counters_tensor.detach().cpu().tolist()
        return {
            "head3_eligible_count": int(counters[0]),
            "head3_quality_gate_pass_count": int(counters[1]),
            "head3_quality_gate_reject_count": int(counters[2]),
            "head3_exploration_count": int(counters[3]),
            "head3_calibration_records": int(self._bin_mature.sum().detach().cpu()),
            "head3_acceptance_ema": float(self._acceptance_ema.detach().cpu()),
            "head3_candidate_regret_ema": float(self._regret_ema.detach().cpu()),
            "head3_candidate_target_mass_gain": float(
                self._bin_mass_gain.sum().detach().cpu()
                / self._bin_probe_count.sum().clamp_min(1.0).detach().cpu()
            ),
        }


def dense_node_count(topk_by_depth: list[int]) -> int:
    total = 1
    parents = 1
    for k in topk_by_depth:
        parents *= int(k)
        total += parents
    return total


def fit_topk_to_budget(
    topk_by_depth: list[int],
    node_budget: int,
    *,
    min_topk_by_depth: list[int] | None = None,
    depth_weight_decay: float = 0.5,
) -> list[int]:
    """Fit a dense tree without needlessly dropping deeper horizons.

    Early candidate coverage is more valuable because every deeper token is
    conditional on reaching it.  The greedy score therefore discounts the
    utility of deeper branches while accounting for the number of tree nodes
    removed by each decrement.
    """

    budget = max(1, int(node_budget))
    fitted = [max(1, int(k)) for k in topk_by_depth]
    raw_min = list(min_topk_by_depth or [])
    minimum = [
        max(1, min(fitted[idx], int(raw_min[idx]) if idx < len(raw_min) else 1))
        for idx in range(len(fitted))
    ]

    while fitted and dense_node_count(fitted) > budget:
        candidates: list[tuple[float, int, int]] = []
        for depth_idx, value in enumerate(fitted):
            if value <= minimum[depth_idx]:
                continue
            trial = list(fitted)
            trial[depth_idx] -= 1
            nodes_saved = dense_node_count(fitted) - dense_node_count(trial)
            utility_loss = max(float(depth_weight_decay), 1e-3) ** depth_idx * (
                math.log1p(value) - math.log(value)
            )
            candidates.append((nodes_saved / max(utility_loss, 1e-9), depth_idx, nodes_saved))
        if candidates:
            _, depth_idx, _ = max(candidates, key=lambda item: (item[0], item[2], item[1]))
            fitted[depth_idx] -= 1
            continue

        # Even the minimum branching does not fit. Dropping the deepest
        # horizon preserves the highest-probability prefix and exactness.
        fitted.pop()
        minimum.pop()
    return fitted


def plan_tree(
    *,
    active_batch_size: int,
    num_medusa_heads: int,
    tree_mode: str,
    tree_layout: str,
    cpeak_nodes: int,
    min_tree_nodes_per_seq: int,
    max_tree_nodes_per_seq: int,
    max_tree_depth: int,
    fixed_tree_topk_by_depth: list[int],
    adaptive_tree_enabled: bool = False,
    adaptive_min_topk_by_depth: list[int] | None = None,
    sparse_nodes_by_head: list[int] | None = None,
    sparse_min_head3_nodes: int = 1,
    sparse_head3_min_budget: int = 8,
    sparse_branch_score_temperature: float = 1.0,
    sparse_diversity_penalty: float = 0.05,
) -> TreePlan:
    if tree_layout not in {"dense", "sparse", "sparse_asymmetric"}:
        raise ValueError(f"Unsupported tree_layout={tree_layout}")
    active_batch_size = max(1, int(active_batch_size))
    if tree_mode == "fixed":
        budget = int(max_tree_nodes_per_seq)
        topk = [max(1, int(k)) for k in fixed_tree_topk_by_depth[:num_medusa_heads]]
        while topk and dense_node_count(topk) > budget:
            topk[-1] -= 1
            if topk[-1] <= 0:
                topk.pop()
        active_heads = min(len(topk), max(0, int(max_tree_depth) - 1))
        topk = topk[:active_heads]
    elif tree_mode == "concurrency_aware":
        budget = math.floor(int(cpeak_nodes) / active_batch_size)
        budget = max(int(min_tree_nodes_per_seq), min(int(max_tree_nodes_per_seq), budget))
        topk = []
        max_heads = min(int(num_medusa_heads), max(0, int(max_tree_depth) - 1))
        defaults = fixed_tree_topk_by_depth or [4, 3, 2, 1, 1]
        if adaptive_tree_enabled:
            # Decide depth from the minimum adaptive tree, then let confidence
            # choose and budget-fit the actual branching. The old planner used
            # maximum branching here, which permanently discarded deep heads
            # before uncertainty could shrink the earlier branches.
            raw_min = list(adaptive_min_topk_by_depth or [])
            minimum: list[int] = []
            for depth in range(max_heads):
                min_k = max(1, int(raw_min[depth]) if depth < len(raw_min) else 1)
                if dense_node_count(minimum + [min_k]) > budget:
                    break
                minimum.append(min_k)
                topk.append(max(min_k, int(defaults[min(depth, len(defaults) - 1)])))
        else:
            parent_paths = 1
            used_nodes = 1
            for depth in range(max_heads):
                default_k = int(defaults[min(depth, len(defaults) - 1)])
                room = budget - used_nodes
                if room <= 0:
                    break
                k = min(max(1, default_k), max(1, room // parent_paths))
                if used_nodes + parent_paths * k > budget:
                    k = room // parent_paths
                if k <= 0:
                    break
                topk.append(int(k))
                parent_paths *= int(k)
                used_nodes += parent_paths
        active_heads = len(topk)
    else:
        raise ValueError(f"Unsupported tree_mode={tree_mode}")
    if tree_layout in {"sparse", "sparse_asymmetric"}:
        max_heads = min(int(num_medusa_heads), max(0, int(max_tree_depth) - 1))
        requested = list(sparse_nodes_by_head or fixed_tree_topk_by_depth or [4, 3, 2])
        allocated = [0 for _ in range(max_heads)]
        remaining = max(0, int(budget) - 1)
        for head_idx in range(max_heads):
            wanted = max(0, int(requested[head_idx] if head_idx < len(requested) else 0))
            allocated[head_idx] = min(wanted, remaining)
            remaining -= allocated[head_idx]

        # A budget that can support the third horizon must not silently starve
        # it because earlier dense branches consumed all slots.
        if max_heads >= 3 and budget >= int(sparse_head3_min_budget):
            required = max(1, int(sparse_min_head3_nodes))
            deficit = max(0, required - allocated[2])
            for donor in (1, 0):
                transferable = max(0, allocated[donor] - 1)
                moved = min(deficit, transferable)
                allocated[donor] -= moved
                allocated[2] += moved
                deficit -= moved
                if deficit == 0:
                    break
        active_heads = max((idx + 1 for idx, count in enumerate(allocated) if count > 0), default=0)
        allocated = allocated[:active_heads]
        topk = list(allocated)
        actual_nodes = 1 + sum(allocated)
        tree_layout = "sparse_asymmetric"
    else:
        allocated = []
        actual_nodes = dense_node_count(topk)
    return TreePlan(
        node_budget_per_seq=budget,
        active_heads=active_heads,
        topk_by_depth=topk,
        actual_nodes=actual_nodes,
        mode=tree_mode,
        layout=tree_layout,
        nodes_by_head=allocated,
        min_head3_nodes=max(0, int(sparse_min_head3_nodes)),
        head3_min_budget=max(1, int(sparse_head3_min_budget)),
        branch_score_temperature=max(1e-6, float(sparse_branch_score_temperature)),
        diversity_penalty=max(0.0, float(sparse_diversity_penalty)),
    )


def _unique_topk(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    if k <= 0:
        return [], []
    values, indices = torch.topk(logits, k=min(int(k) * 2, logits.shape[-1]), dim=-1)
    seen = set()
    toks: list[int] = []
    scores: list[float] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token = int(index)
        if token in seen:
            continue
        seen.add(token)
        toks.append(token)
        scores.append(float(value))
        if len(toks) >= k:
            break
    return toks, scores


def build_dense_tree(root_token: int, medusa_logits: list[torch.Tensor], plan: TreePlan) -> CandidateTree:
    tokens = [int(root_token)]
    parents = [-1]
    depths = [1]
    scores = [0.0]
    current_parents = [0]
    for depth_idx, k in enumerate(plan.topk_by_depth):
        top_tokens, top_scores = _unique_topk(medusa_logits[depth_idx], int(k))
        if not top_tokens:
            break
        next_parents = []
        for parent in current_parents:
            for token, score in zip(top_tokens, top_scores):
                tokens.append(int(token))
                parents.append(int(parent))
                depths.append(depth_idx + 2)
                scores.append(float(score))
                next_parents.append(len(tokens) - 1)
                if len(tokens) >= plan.node_budget_per_seq:
                    break
            if len(tokens) >= plan.node_budget_per_seq:
                break
        current_parents = next_parents
        if len(tokens) >= plan.node_budget_per_seq:
            break
    return CandidateTree(tokens=tokens, parents=parents, depths=depths, scores=scores)


def _select_sparse_children(
    parent_nodes: list[int],
    parent_scores: list[float],
    token_ids: list[int],
    token_scores: list[float],
    count: int,
    diversity_penalty: float,
) -> list[tuple[int, int, float]]:
    """Greedily select globally best path extensions with parent diversity."""

    if count <= 0 or not parent_nodes or not token_ids:
        return []
    pool = [
        (float(parent_scores[pidx]) + float(score), pidx, int(token))
        for pidx in range(len(parent_nodes))
        for token, score in zip(token_ids, token_scores)
    ]
    chosen: list[tuple[int, int, float]] = []
    used: set[tuple[int, int]] = set()
    parent_use = [0 for _ in parent_nodes]
    for _ in range(min(int(count), len(pool))):
        best = None
        best_value = -math.inf
        for cumulative, pidx, token in pool:
            key = (pidx, token)
            if key in used:
                continue
            value = cumulative - float(diversity_penalty) * parent_use[pidx]
            if value > best_value:
                best_value = value
                best = (pidx, token, cumulative)
        if best is None:
            break
        pidx, token, cumulative = best
        used.add((pidx, token))
        parent_use[pidx] += 1
        chosen.append((int(parent_nodes[pidx]), int(token), float(cumulative)))
    return chosen


def _build_sparse_batch_trees(
    root_tokens: torch.Tensor,
    medusa_logits: list[torch.Tensor],
    plan: TreePlan,
    *,
    head3_gate_mask: torch.Tensor | None,
    head3_exploration_mask: torch.Tensor | None,
    head3_quality: torch.Tensor | None,
) -> list[CandidateTree]:
    batch_size = int(root_tokens.shape[0])
    nodes = list(plan.nodes_by_head or plan.topk_by_depth)
    head_count = min(int(plan.active_heads), len(nodes), len(medusa_logits))
    nodes = nodes[:head_count]
    if head_count == 0:
        roots = root_tokens.detach().cpu().tolist()
        return [CandidateTree(tokens=[int(token)], parents=[-1], depths=[1], scores=[0.0]) for token in roots]

    # One top-k launch and one host transfer per horizon. The following CPU
    # work is bounded by the tiny verification budget (normally ten nodes).
    max_reallocation = nodes[2] if len(nodes) >= 3 else 0
    widths: list[int] = []
    for head_idx, count in enumerate(nodes):
        if head_idx == 0:
            width = count + max_reallocation
        elif head_idx == 1:
            width = count + max_reallocation
        else:
            width = count
        widths.append(max(1, min(int(width), int(medusa_logits[head_idx].shape[-1]))))
    packed: list[tuple[list[list[int]], list[list[float]]]] = []
    temperature = max(float(plan.branch_score_temperature), 1e-6)
    for head_idx, width in enumerate(widths):
        values, indices = torch.topk(medusa_logits[head_idx], k=width, dim=-1)
        packed.append(
            (
                indices.detach().cpu().tolist(),
                (values.detach().float() / temperature).cpu().tolist(),
            )
        )
    roots = root_tokens.detach().cpu().tolist()
    gate = (
        head3_gate_mask.detach().bool().cpu().tolist()
        if head3_gate_mask is not None
        else [True for _ in range(batch_size)]
    )
    exploration = (
        head3_exploration_mask.detach().bool().cpu().tolist()
        if head3_exploration_mask is not None
        else [False for _ in range(batch_size)]
    )
    quality = (
        head3_quality.detach().float().cpu().tolist()
        if head3_quality is not None
        else [0.0 for _ in range(batch_size)]
    )

    trees: list[CandidateTree] = []
    for row in range(batch_size):
        tokens = [int(roots[row])]
        parents = [-1]
        depths = [1]
        scores = [0.0]

        n1 = nodes[0] if nodes else 0
        h1_tokens, h1_scores = packed[0]
        for token, score in zip(h1_tokens[row][:n1], h1_scores[row][:n1]):
            tokens.append(int(token))
            parents.append(0)
            depths.append(2)
            scores.append(float(score))
        head1_nodes = list(range(1, len(tokens)))
        head1_scores = [scores[idx] for idx in head1_nodes]

        use_head3 = bool(len(nodes) >= 3 and nodes[2] > 0 and gate[row])
        n2 = (nodes[1] if len(nodes) >= 2 else 0) + (0 if use_head3 else max_reallocation)
        head2_nodes: list[int] = []
        head2_scores: list[float] = []
        if n2 > 0 and len(packed) >= 2:
            h2_tokens, h2_scores = packed[1]
            selected = _select_sparse_children(
                head1_nodes,
                head1_scores,
                h2_tokens[row],
                h2_scores[row],
                n2,
                plan.diversity_penalty,
            )
            for parent, token, score in selected:
                tokens.append(token)
                parents.append(parent)
                depths.append(3)
                scores.append(score)
                head2_nodes.append(len(tokens) - 1)
                head2_scores.append(score)

        if use_head3 and len(packed) >= 3 and head2_nodes:
            h3_tokens, h3_scores = packed[2]
            selected = _select_sparse_children(
                head2_nodes,
                head2_scores,
                h3_tokens[row],
                h3_scores[row],
                nodes[2],
                plan.diversity_penalty,
            )
            for parent, token, score in selected:
                tokens.append(token)
                parents.append(parent)
                depths.append(4)
                scores.append(score)

        # If a shallow candidate pool was unexpectedly too small, spend any
        # remaining budget on additional unique Head-1 nodes.
        seen_h1 = {tokens[idx] for idx in head1_nodes}
        for token, score in zip(h1_tokens[row][n1:], h1_scores[row][n1:]):
            if len(tokens) >= plan.node_budget_per_seq:
                break
            if int(token) in seen_h1:
                continue
            tokens.append(int(token))
            parents.append(0)
            depths.append(2)
            scores.append(float(score))
            seen_h1.add(int(token))

        trees.append(
            CandidateTree(
                tokens=tokens[: plan.node_budget_per_seq],
                parents=parents[: plan.node_budget_per_seq],
                depths=depths[: plan.node_budget_per_seq],
                scores=scores[: plan.node_budget_per_seq],
                head3_quality=float(quality[row]),
                head3_gate_passed=bool(use_head3),
                head3_exploration=bool(exploration[row]),
            )
        )
    return trees


def candidate_sets_by_head(
    trees: list[CandidateTree],
    num_heads: int,
    *,
    device: torch.device,
    widths: list[int] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pack the actual sparse-tree candidates for delayed verifier records."""

    packed_ids: list[torch.Tensor] = []
    packed_valid: list[torch.Tensor] = []
    for head_idx in range(max(0, int(num_heads))):
        rows: list[list[int]] = []
        natural_width = 0
        target_depth = head_idx + 2
        for tree in trees:
            # Candidate mass is a token set, so repeated tokens reached through
            # different parents count once here while paths remain distinct in
            # the verification tree itself.
            seen: set[int] = set()
            values: list[int] = []
            for token, depth in zip(tree.tokens, tree.depths):
                if int(depth) == target_depth and int(token) not in seen:
                    values.append(int(token))
                    seen.add(int(token))
            rows.append(values)
            natural_width = max(natural_width, len(values))
        width = natural_width
        if widths is not None and head_idx < len(widths):
            width = max(width, max(0, int(widths[head_idx])))
        width = max(1, width)
        ids = torch.full((len(trees), width), -1, dtype=torch.long)
        valid = torch.zeros((len(trees), width), dtype=torch.bool)
        for row, values in enumerate(rows):
            count = min(len(values), width)
            if count:
                ids[row, :count] = torch.as_tensor(values[:count], dtype=torch.long)
                valid[row, :count] = True
        packed_ids.append(ids.to(device=device, non_blocking=True))
        packed_valid.append(valid.to(device=device, non_blocking=True))
    return packed_ids, packed_valid


def build_batch_trees(
    root_tokens: torch.Tensor,
    medusa_logits: list[torch.Tensor],
    plan: TreePlan,
    *,
    head3_gate_mask: torch.Tensor | None = None,
    head3_exploration_mask: torch.Tensor | None = None,
    head3_quality: torch.Tensor | None = None,
) -> list[CandidateTree]:
    """Build standard MEDUSA trees with one top-k launch per depth.

    The previous implementation launched ``topk`` and synchronized CUDA once
    per sequence and depth. Standard MEDUSA heads are independent of the tree
    parent, so their top-k candidates can be extracted for the whole batch at
    once and the small tree structures can then be assembled on CPU.
    """

    if plan.layout == "sparse_asymmetric":
        return _build_sparse_batch_trees(
            root_tokens,
            medusa_logits,
            plan,
            head3_gate_mask=head3_gate_mask,
            head3_exploration_mask=head3_exploration_mask,
            head3_quality=head3_quality,
        )

    batch_size = int(root_tokens.shape[0])
    candidates_gpu: list[tuple[torch.Tensor, torch.Tensor]] = []
    for depth_idx, requested_k in enumerate(plan.topk_by_depth[: plan.active_heads]):
        if depth_idx >= len(medusa_logits):
            break
        logits = medusa_logits[depth_idx]
        k = min(max(0, int(requested_k)), int(logits.shape[-1]))
        if k <= 0:
            break
        values, indices = torch.topk(logits, k=k, dim=-1)
        candidates_gpu.append((indices, values))

    root_cpu = root_tokens.detach().cpu().tolist()
    candidates: list[tuple[list[list[int]], list[list[float]]]] = []
    for indices, values in candidates_gpu:
        candidates.append((indices.detach().cpu().tolist(), values.detach().cpu().tolist()))

    trees: list[CandidateTree] = []
    for row in range(batch_size):
        tokens = [int(root_cpu[row])]
        parents = [-1]
        depths = [1]
        scores = [0.0]
        current_parents = [0]
        for depth_idx, (token_rows, score_rows) in enumerate(candidates):
            next_parents: list[int] = []
            for parent in current_parents:
                for token, score in zip(token_rows[row], score_rows[row]):
                    tokens.append(int(token))
                    parents.append(int(parent))
                    depths.append(depth_idx + 2)
                    scores.append(float(score))
                    next_parents.append(len(tokens) - 1)
                    if len(tokens) >= plan.node_budget_per_seq:
                        break
                if len(tokens) >= plan.node_budget_per_seq:
                    break
            current_parents = next_parents
            if not current_parents or len(tokens) >= plan.node_budget_per_seq:
                break
        trees.append(CandidateTree(tokens=tokens, parents=parents, depths=depths, scores=scores))
    return trees
