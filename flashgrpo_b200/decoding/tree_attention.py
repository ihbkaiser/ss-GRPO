from __future__ import annotations

import torch


def build_tree_attention_inputs(
    trees,
    full_attention_mask: torch.Tensor,
    logical_lengths: torch.Tensor,
    *,
    pad_token_id: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten per-sequence candidate trees and build an ancestor-only mask.

    Each tree node can attend to valid prefix tokens, its ancestors, and itself.
    Siblings/cousins are masked, which makes the batched tree logits equivalent
    to forwarding each root-to-node path independently.
    """

    device = full_attention_mask.device
    batch = len(trees)
    max_nodes = max(max(tree.node_count, 1) for tree in trees)
    past_len = full_attention_mask.shape[1]
    min_dtype = torch.finfo(dtype).min
    # Build the tiny topology tensors on CPU, then issue bulk device
    # operations. The old nested loop performed several CUDA assignments and a
    # logical_lengths.item() synchronization for every node.
    token_rows: list[list[int]] = []
    depth_rows: list[list[int]] = []
    valid_rows: list[list[bool]] = []
    ancestor_rows: list[list[list[bool]]] = []
    topology_cache: dict[tuple[int, ...], list[list[bool]]] = {}
    for tree in trees:
        valid_count = tree.node_count
        token_rows.append([int(token) for token in tree.tokens] + [int(pad_token_id)] * (max_nodes - valid_count))
        depth_rows.append([int(depth) for depth in tree.depths] + [1] * (max_nodes - valid_count))
        valid_rows.append([True] * valid_count + [False] * (max_nodes - valid_count))

        topology_key = tuple(int(parent) for parent in tree.parents) + (-2, max_nodes)
        ancestor = topology_cache.get(topology_key)
        if ancestor is None:
            ancestor = [[False] * max_nodes for _ in range(max_nodes)]
            for node_idx in range(valid_count):
                for parent_idx in tree.ancestors_including_self(node_idx):
                    ancestor[node_idx][int(parent_idx)] = True
            for node_idx in range(valid_count, max_nodes):
                # Padded query rows are ignored downstream, but strict
                # attention kernels still require one valid tree position.
                ancestor[node_idx][node_idx] = True
            topology_cache[topology_key] = ancestor
        ancestor_rows.append(ancestor)

    input_ids = torch.tensor(token_rows, dtype=torch.long, device=device)
    depths = torch.tensor(depth_rows, dtype=torch.long, device=device)
    node_mask = torch.tensor(valid_rows, dtype=torch.bool, device=device)
    tree_allowed = torch.tensor(ancestor_rows, dtype=torch.bool, device=device).unsqueeze(1)
    position_ids = logical_lengths.to(device=device, dtype=torch.long).unsqueeze(1) + depths - 1
    position_ids.masked_fill_(~node_mask, 0)

    prefix_allowed = full_attention_mask.to(device=device, dtype=torch.bool)[:, None, None, :].expand(
        batch, 1, max_nodes, past_len
    )
    allowed = torch.cat((prefix_allowed, tree_allowed), dim=-1)
    attn = torch.zeros((batch, 1, max_nodes, past_len + max_nodes), dtype=dtype, device=device)
    attn.masked_fill_(~allowed, min_dtype)
    return input_ids, attn, position_ids, node_mask
