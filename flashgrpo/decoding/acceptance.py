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
) -> torch.Tensor:
    if (not do_sample) or temperature == 0:
        return torch.argmax(logits, dim=-1)
    probs = logits_to_probs(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(logits.shape[:-1])


def exact_accept_path(
    tree,
    tree_logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
) -> tuple[list[int], list[int], int]:
    """Walk one MEDUSA tree using target samples only.

    The root token has already been sampled from target logits before tree
    construction. Future tokens are accepted only when a fresh target sample
    from the current parent distribution is present among that parent node's
    children.
    """

    accepted_tokens = [int(tree.tokens[0])]
    accepted_nodes = [0]
    parent = 0
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
            break
        accepted_tokens.append(sampled)
        accepted_nodes.append(match)
        parent = match
    return accepted_tokens, accepted_nodes, parent
