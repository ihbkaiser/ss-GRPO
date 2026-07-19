from __future__ import annotations

import torch
import torch.nn.functional as F


def logits_to_probs(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = logits.float()
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        top_values, top_indices = torch.topk(logits, k=int(top_k), dim=-1)
        filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = filtered.scatter(-1, top_indices, top_values)
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > float(top_p)
        mask = torch.roll(mask, shifts=1, dims=-1)
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return probs


def sample_from_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = None,
    nucleus_topk_hint: int = 2048,
) -> torch.Tensor:
    if (not do_sample) or temperature == 0:
        return torch.argmax(logits, dim=-1)

    vocab_size = int(logits.shape[-1])
    output_shape = logits.shape[:-1]
    flat_logits = logits.reshape(-1, vocab_size)

    def scaled(values: torch.Tensor) -> torch.Tensor:
        values = values.float()
        if temperature is not None and temperature > 0:
            values = values / float(temperature)
        return values

    def draw(weights: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        sampled_pos = torch.multinomial(weights, num_samples=1)
        if ids is not None:
            sampled_pos = ids.gather(1, sampled_pos)
        return sampled_pos.squeeze(-1)

    def mask_nucleus_(weights: torch.Tensor) -> torch.Tensor:
        cumulative = torch.cumsum(weights, dim=-1)
        remove = torch.roll(cumulative > float(top_p), shifts=1, dims=-1)
        remove[..., 0] = False
        return weights.masked_fill_(remove, 0.0)

    if top_k is not None and 0 < int(top_k) < vocab_size:
        candidate_logits, candidate_ids = torch.topk(
            flat_logits,
            k=int(top_k),
            dim=-1,
            largest=True,
            sorted=True,
        )
        probs = F.softmax(scaled(candidate_logits), dim=-1)
        if top_p is not None and 0 < float(top_p) < 1.0:
            mask_nucleus_(probs)
        return draw(probs, candidate_ids).view(output_shape)

    if top_p is not None and 0 < float(top_p) < 1.0:
        shortlist_k = min(vocab_size, max(1, int(nucleus_topk_hint)))
        if shortlist_k < vocab_size:
            # This is an exact nucleus shortcut, not an approximation. Full
            # logsumexp gives the shortlist's true probability mass. Rows that
            # do not cover top_p fall back to a full vocabulary sort.
            flat_scaled = scaled(flat_logits)
            log_z = torch.logsumexp(flat_scaled, dim=-1, keepdim=True)
            short_logits, short_ids = torch.topk(
                flat_logits,
                k=shortlist_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            short_probs = torch.exp(scaled(short_logits) - log_z)
            covered = short_probs.sum(dim=-1) >= float(top_p)
            covered_flags = covered.detach().cpu().tolist()
            if all(covered_flags):
                mask_nucleus_(short_probs)
                return draw(short_probs, short_ids).view(output_shape)

            sampled = torch.empty(flat_logits.shape[0], dtype=torch.long, device=flat_logits.device)
            covered_rows = [idx for idx, value in enumerate(covered_flags) if value]
            fallback_rows = [idx for idx, value in enumerate(covered_flags) if not value]
            if covered_rows:
                covered_idx = torch.tensor(covered_rows, dtype=torch.long, device=flat_logits.device)
                covered_probs = short_probs.index_select(0, covered_idx)
                mask_nucleus_(covered_probs)
                sampled.index_copy_(
                    0,
                    covered_idx,
                    draw(covered_probs, short_ids.index_select(0, covered_idx)),
                )
            if fallback_rows:
                fallback_idx = torch.tensor(fallback_rows, dtype=torch.long, device=flat_logits.device)
                fallback_logits = flat_logits.index_select(0, fallback_idx)
                sorted_logits, sorted_ids = torch.sort(fallback_logits, dim=-1, descending=True)
                fallback_probs = F.softmax(scaled(sorted_logits), dim=-1)
                mask_nucleus_(fallback_probs)
                sampled.index_copy_(0, fallback_idx, draw(fallback_probs, sorted_ids))
            return sampled.view(output_shape)

        sorted_logits, sorted_ids = torch.sort(flat_logits, dim=-1, descending=True)
        probs = F.softmax(scaled(sorted_logits), dim=-1)
        mask_nucleus_(probs)
        return draw(probs, sorted_ids).view(output_shape)

    probs = F.softmax(scaled(flat_logits), dim=-1)
    return draw(probs).view(output_shape)


def exact_accept_path(
    tree,
    tree_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> tuple[list[int], list[int], int, int | None]:
    """Walk one MEDUSA tree using target samples only.

    The root token has already been sampled from target logits before tree
    construction. Future tree tokens are accepted only when a fresh target
    sample from the current parent distribution is present among that parent's
    children. A non-matching target sample is returned as a correction token;
    callers must emit it, immediately or as the forced root of the next round,
    rather than resampling from the same distribution.
    """

    accepted_tokens = [int(tree.tokens[0])]
    accepted_nodes = [0]
    parent = 0
    correction_token = None
    while True:
        children = tree.children.get(parent, [])
        if not children:
            break
        sampled = int(
            sample_from_logits(
                tree_logits[parent : parent + 1],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            ).item()
        )
        match = None
        for child in children:
            if int(tree.tokens[child]) == sampled:
                match = child
                break
        if match is None:
            correction_token = sampled
            break
        accepted_tokens.append(sampled)
        accepted_nodes.append(match)
        parent = match
    return accepted_tokens, accepted_nodes, parent, correction_token


@torch.no_grad()
def exact_accept_paths_batch(
    trees,
    tree_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    node_to_logit: torch.Tensor | None = None,
    node_to_logit_cpu: list[list[int]] | None = None,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int | None]]:
    """Batched equivalent of :func:`exact_accept_path`.

    Target samples at the same tree depth are drawn in one CUDA operation.
    Candidate lookup remains on CPU because ``CandidateTree`` is a compact
    Python structure. Non-matching samples are returned separately as target
    correction tokens. Discarding those samples and drawing again would bias
    stochastic decoding toward the proposal tree.
    """

    batch_size = len(trees)
    accepted_tokens = [[int(tree.tokens[0])] for tree in trees]
    accepted_nodes = [[0] for _ in trees]
    parent_nodes = [0 for _ in trees]
    correction_tokens: list[int | None] = [None for _ in trees]
    active_rows = [row for row, tree in enumerate(trees) if tree.children.get(0)]

    while active_rows:
        if node_to_logit is None and node_to_logit_cpu is None:
            row_index = torch.as_tensor(active_rows, dtype=torch.long, device=tree_logits.device)
            parent_index = torch.as_tensor(
                [parent_nodes[row] for row in active_rows],
                dtype=torch.long,
                device=tree_logits.device,
            )
            parent_logits = tree_logits[row_index, parent_index]
        else:
            if node_to_logit_cpu is not None:
                slots = torch.as_tensor(
                    [node_to_logit_cpu[row][parent_nodes[row]] for row in active_rows],
                    dtype=torch.long,
                    device=tree_logits.device,
                )
            else:
                row_index = torch.as_tensor(active_rows, dtype=torch.long, device=tree_logits.device)
                parent_index = torch.as_tensor(
                    [parent_nodes[row] for row in active_rows],
                    dtype=torch.long,
                    device=tree_logits.device,
                )
                slots = node_to_logit[row_index, parent_index]
            parent_logits = tree_logits.index_select(0, slots)
        sampled = sample_from_logits(
            parent_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        ).detach().cpu().tolist()

        next_active: list[int] = []
        for row, token in zip(active_rows, sampled):
            tree = trees[row]
            match = next(
                (
                    child
                    for child in tree.children.get(parent_nodes[row], [])
                    if int(tree.tokens[child]) == int(token)
                ),
                None,
            )
            if match is None:
                correction_tokens[row] = int(token)
                continue
            accepted_tokens[row].append(int(token))
            accepted_nodes[row].append(int(match))
            parent_nodes[row] = int(match)
            if tree.children.get(int(match)):
                next_active.append(row)
        active_rows = next_active

    if len(accepted_tokens) != batch_size:
        raise RuntimeError("Batched acceptance produced an invalid batch size")
    return accepted_tokens, accepted_nodes, parent_nodes, correction_tokens
