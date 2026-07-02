"""Batch-aware CLaSp controller for self-speculative GRPO.

This module implements a production-oriented CLaSp variant for GRPO rollouts:
- CLaSp-style in-context layer scoring from full-model hidden states.
- A small executable skip-path codebook.
- Quantization + small-group merging to avoid low GPU utilization.

The layer score is a cheap approximation of CLaSp's DP objective: layers whose
input/output hidden states are highly cosine-similar for the last accepted token
are treated as safer to skip. This keeps runtime overhead low enough for batched
GRPO while retaining the key CLaSp property: the skip path follows the current
context rather than a fixed offline mask.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass
class ClaspRoutingStats:
    update_count: int = 0
    clasp_time_cost: float = 0.0
    codebook_size_sum: int = 0
    active_paths_sum: int = 0
    active_paths_max: int = 0
    active_paths_min: int = 10**9
    merged_rows_sum: int = 0
    routing_entropy_sum: float = 0.0
    average_skip_layers_sum: float = 0.0
    route_calls: int = 0
    default_path_rows_sum: int = 0

    def update_route(self, *, active_paths: int, codebook_size: int, merged_rows: int, counts: Sequence[int], avg_skip_layers: float, default_rows: int) -> None:
        self.route_calls += 1
        self.codebook_size_sum += int(codebook_size)
        self.active_paths_sum += int(active_paths)
        self.active_paths_max = max(self.active_paths_max, int(active_paths))
        self.active_paths_min = min(self.active_paths_min, int(active_paths))
        self.merged_rows_sum += int(merged_rows)
        self.average_skip_layers_sum += float(avg_skip_layers)
        self.default_path_rows_sum += int(default_rows)
        total = max(1, sum(int(c) for c in counts))
        entropy = 0.0
        for c in counts:
            if c <= 0:
                continue
            p = float(c) / total
            entropy -= p * torch.log2(torch.tensor(p)).item()
        self.routing_entropy_sum += entropy

    def to_dict(self) -> dict:
        calls = max(1, self.route_calls)
        return {
            "clasp_update_count": int(self.update_count),
            "clasp_time_cost": float(self.clasp_time_cost),
            "clasp_route_calls": int(self.route_calls),
            "clasp_average_codebook_size": self.codebook_size_sum / calls if self.route_calls else 0.0,
            "clasp_average_active_paths": self.active_paths_sum / calls if self.route_calls else 0.0,
            "clasp_min_active_paths": int(self.active_paths_min if self.route_calls else 0),
            "clasp_max_active_paths": int(self.active_paths_max if self.route_calls else 0),
            "clasp_average_merged_rows": self.merged_rows_sum / calls if self.route_calls else 0.0,
            "clasp_average_route_entropy": self.routing_entropy_sum / calls if self.route_calls else 0.0,
            "clasp_average_skip_layers": self.average_skip_layers_sum / calls if self.route_calls else 0.0,
            "clasp_default_path_rows": int(self.default_path_rows_sum),
        }


def parse_layer_set(value: str | Iterable[int] | None, num_layers: int) -> list[int]:
    if value is None or value == "":
        return list(range(num_layers))
    if not isinstance(value, str):
        return sorted({int(x) for x in value if 0 <= int(x) < num_layers})
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            layers.extend(range(int(left), int(right) + 1))
        else:
            layers.append(int(part))
    return sorted({x for x in layers if 0 <= x < num_layers})


def layers_to_string(layers: Iterable[int]) -> str:
    return ",".join(str(int(x)) for x in sorted(set(layers)))


def _candidate_layers(num_layers: int, candidate_layers: Sequence[int] | None, protected_first: int, protected_last: int) -> list[int]:
    if candidate_layers:
        raw = [int(x) for x in candidate_layers]
    else:
        raw = list(range(num_layers))
    lo = max(0, int(protected_first))
    hi = max(lo, num_layers - max(0, int(protected_last)))
    return [x for x in sorted(set(raw)) if lo <= x < hi]


def build_initial_codebook(
    *,
    num_layers: int,
    base_skip_layers: Iterable[int],
    codebook_size: int = 8,
    candidate_layers: Sequence[int] | None = None,
    protected_first: int = 4,
    protected_last: int = 4,
    skip_count: int | None = None,
) -> list[frozenset[int]]:
    """Create a small, ordered executable skip-path codebook.

    C0 is always the provided base mask. Later masks are deterministic medium /
    aggressive variants over candidate middle layers. The order is important:
    C0 is the default fallback path used during small-batch tails.
    """
    codebook_size = max(1, int(codebook_size))
    cand = _candidate_layers(num_layers, candidate_layers, protected_first, protected_last)
    if not cand:
        cand = list(range(num_layers))
    base = frozenset(int(x) for x in base_skip_layers if 0 <= int(x) < num_layers)
    if not base:
        k = int(skip_count or max(1, round(0.4 * len(cand))))
        base = frozenset(_even_pick(cand, min(k, len(cand))))
    base_k = len(base) if skip_count is None else max(1, min(int(skip_count), len(cand)))

    masks: list[frozenset[int]] = [frozenset(sorted(base))]
    counts = [
        max(1, min(len(cand), base_k - 2)),
        max(1, min(len(cand), base_k - 1)),
        max(1, min(len(cand), base_k)),
        max(1, min(len(cand), base_k + 1)),
        max(1, min(len(cand), base_k + 2)),
    ]
    patterns: list[list[int]] = []
    for k in counts:
        patterns.append(_even_pick(cand, k))
        patterns.append(_middle_pick(cand, k))
        patterns.append(_alternate_pick(cand, k, parity=0))
        patterns.append(_alternate_pick(cand, k, parity=1))
    for p in patterns:
        m = frozenset(p)
        if m and m not in masks:
            masks.append(m)
        if len(masks) >= codebook_size:
            break
    return masks[:codebook_size]


def _even_pick(values: Sequence[int], k: int) -> list[int]:
    if k <= 0:
        return []
    if k >= len(values):
        return list(values)
    if k == 1:
        return [values[len(values) // 2]]
    idx = torch.linspace(0, len(values) - 1, steps=k).round().long().tolist()
    out = []
    for i in idx:
        v = values[int(i)]
        if v not in out:
            out.append(v)
    pos = 0
    while len(out) < k and pos < len(values):
        if values[pos] not in out:
            out.append(values[pos])
        pos += 1
    return sorted(out[:k])


def _middle_pick(values: Sequence[int], k: int) -> list[int]:
    if k >= len(values):
        return list(values)
    center = (len(values) - 1) / 2.0
    ordered = sorted(range(len(values)), key=lambda i: abs(i - center))
    return sorted(values[i] for i in ordered[:k])


def _alternate_pick(values: Sequence[int], k: int, parity: int) -> list[int]:
    seq = list(values)[parity::2]
    if len(seq) < k:
        seq += [v for v in values if v not in seq]
    return sorted(seq[:k])


@torch.no_grad()
def propose_skip_masks_from_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    token_positions: torch.Tensor,
    *,
    candidate_layers: Sequence[int] | None = None,
    protected_first: int = 4,
    protected_last: int = 4,
    skip_count: int,
    max_rows: int | None = None,
) -> tuple[list[frozenset[int]], dict]:
    """Return one CLaSp-style ideal mask per row.

    hidden_states is the tuple returned by HF with output_hidden_states=True:
    embedding output + one tensor per decoder layer. token_positions is [B]
    selecting the last accepted token inside the current verify query.
    """
    if not hidden_states or len(hidden_states) < 2:
        return [], {"clasp_mean_layer_similarity": 0.0}
    device = hidden_states[0].device
    bsz = int(hidden_states[0].shape[0])
    num_layers = len(hidden_states) - 1
    rows = torch.arange(bsz, device=device)
    if max_rows is not None and 0 < int(max_rows) < bsz:
        # Keep deterministic spacing rather than random sampling for repeatability.
        rows = torch.linspace(0, bsz - 1, steps=int(max_rows), device=device).round().long().unique()
    positions = token_positions.to(device=device, dtype=torch.long).clamp_min(0)
    positions = positions.clamp_max(hidden_states[0].shape[1] - 1)

    cand = _candidate_layers(num_layers, candidate_layers, protected_first, protected_last)
    if not cand:
        cand = list(range(num_layers))
    k = max(1, min(int(skip_count), len(cand)))

    scores = torch.full((len(rows), num_layers), -1e9, device=device, dtype=torch.float32)
    selected_pos = positions.index_select(0, rows)
    for layer_idx in cand:
        prev_h = hidden_states[layer_idx].index_select(0, rows)[torch.arange(len(rows), device=device), selected_pos].float()
        cur_h = hidden_states[layer_idx + 1].index_select(0, rows)[torch.arange(len(rows), device=device), selected_pos].float()
        # High cosine means the layer changed the state less, hence safer to skip.
        scores[:, layer_idx] = F.cosine_similarity(prev_h, cur_h, dim=-1)

    top = torch.topk(scores, k=k, dim=-1).indices.detach().cpu().tolist()
    masks = [frozenset(sorted(int(x) for x in row)) for row in top]

    # If representative sampling was used, expand masks to all rows by nearest row index.
    if len(rows) != bsz:
        row_cpu = rows.detach().cpu().tolist()
        expanded: list[frozenset[int]] = []
        for idx in range(bsz):
            nearest = min(range(len(row_cpu)), key=lambda j: abs(row_cpu[j] - idx))
            expanded.append(masks[nearest])
        masks = expanded

    finite = scores[scores > -1e8]
    stats = {
        "clasp_mean_layer_similarity": float(finite.mean().detach().cpu().item()) if finite.numel() else 0.0,
        "clasp_skip_count": int(k),
    }
    return masks, stats


def maybe_admit_masks_to_codebook(
    codebook: list[frozenset[int]],
    ideal_masks: Sequence[frozenset[int]],
    *,
    max_codebook_size: int,
    min_frequency: int = 2,
) -> list[frozenset[int]]:
    if max_codebook_size <= len(codebook):
        return codebook
    counter = Counter(ideal_masks)
    for mask, count in counter.most_common():
        if count < min_frequency and len(codebook) > 0:
            continue
        if mask not in codebook:
            codebook.append(mask)
        if len(codebook) >= max_codebook_size:
            break
    return codebook


def quantize_and_merge(
    ideal_masks: Sequence[frozenset[int]],
    codebook: Sequence[frozenset[int]],
    *,
    max_active_paths: int,
    min_group_size: int,
    default_code_idx: int = 0,
) -> tuple[torch.Tensor, list[int], dict]:
    """Map ideal masks to codebook, then merge tiny path groups.

    Returns assignment tensor [B] in codebook-index space, active code indices,
    and metrics. Small or non-selected groups are merged to the nearest active
    code; if no active code is large enough, everything falls back to C0.
    """
    bsz = len(ideal_masks)
    if bsz == 0 or not codebook:
        return torch.zeros(0, dtype=torch.long), [0], {"clasp_merged_rows": 0, "clasp_active_path_counts": []}
    max_active_paths = max(1, int(max_active_paths))
    min_group_size = max(1, int(min_group_size))
    default_code_idx = max(0, min(int(default_code_idx), len(codebook) - 1))

    assigned = []
    for mask in ideal_masks:
        best_idx = 0
        best_dist = 10**9
        for idx, code in enumerate(codebook):
            d = len(mask.symmetric_difference(code))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        assigned.append(best_idx)

    counts = Counter(assigned)
    selected = [idx for idx, c in counts.most_common(max_active_paths) if c >= min_group_size]
    if not selected:
        selected = [default_code_idx]
    if default_code_idx not in selected and counts.get(default_code_idx, 0) >= min_group_size:
        selected = [default_code_idx] + selected[: max_active_paths - 1]
    selected = selected[:max_active_paths]

    merged = 0
    final = []
    for src_idx, mask in zip(assigned, ideal_masks):
        if src_idx in selected and counts[src_idx] >= min_group_size:
            final.append(src_idx)
            continue
        # Merge to nearest selected executable code.
        nearest = min(selected, key=lambda idx: len(mask.symmetric_difference(codebook[idx])))
        if nearest != src_idx:
            merged += 1
        final.append(nearest)

    final_counts = Counter(final)
    active = [idx for idx, _ in final_counts.most_common()]
    count_list = [final_counts[idx] for idx in active]
    avg_skip = sum(len(codebook[idx]) * final_counts[idx] for idx in active) / max(1, bsz)
    stats = {
        "clasp_merged_rows": int(merged),
        "clasp_active_path_counts": count_list,
        "clasp_active_code_indices": active,
        "clasp_default_rows": int(final_counts.get(default_code_idx, 0)),
        "clasp_average_skip_layers_this_round": float(avg_skip),
    }
    return torch.tensor(final, dtype=torch.long), active, stats
