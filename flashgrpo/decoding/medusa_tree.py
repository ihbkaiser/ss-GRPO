from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import torch


@dataclass
class CandidateTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    scores: list[float] = field(default_factory=list)

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


@dataclass
class TreePlan:
    node_budget_per_seq: int
    active_heads: int
    topk_by_depth: list[int]
    actual_nodes: int
    mode: str
    layout: str


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
) -> TreePlan:
    if tree_layout not in {"dense", "sparse"}:
        raise ValueError(f"Unsupported tree_layout={tree_layout}")
    if tree_layout == "sparse":
        # TODO: Replace this compatibility mapping with score-pruned sparse
        # prefix trees. Dense keeps the main exact acceptance path runnable.
        warnings.warn("flashgrpo tree_layout='sparse' is a v1 skeleton; falling back to dense.", RuntimeWarning)
        tree_layout = "dense"
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
    return TreePlan(
        node_budget_per_seq=budget,
        active_heads=active_heads,
        topk_by_depth=topk,
        actual_nodes=dense_node_count(topk),
        mode=tree_mode,
        layout=tree_layout,
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


def build_batch_trees(root_tokens: torch.Tensor, medusa_logits: list[torch.Tensor], plan: TreePlan) -> list[CandidateTree]:
    """Build standard MEDUSA trees with one top-k launch per depth.

    The previous implementation launched ``topk`` and synchronized CUDA once
    per sequence and depth. Standard MEDUSA heads are independent of the tree
    parent, so their top-k candidates can be extracted for the whole batch at
    once and the small tree structures can then be assembled on CPU.
    """

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
