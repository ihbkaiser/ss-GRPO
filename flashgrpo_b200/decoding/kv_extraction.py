from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import DynamicCache

from flashgrpo_b200.models.qwen_flashgrpo_wrapper import (
    _cache_layer_count,
    _get_cache_layer,
    _set_cache_layer,
    cache_seq_length,
    unwrap_causal_lm,
)


@dataclass
class KvExtractionResult:
    past_key_values: object
    max_accepted_length: int
    cache_format: str


def _new_dynamic_cache(causal_lm=None):
    if causal_lm is not None:
        try:
            return DynamicCache(config=unwrap_causal_lm(causal_lm).config)
        except TypeError:
            return DynamicCache()
    return DynamicCache()


def _build_path_index(
    accepted_node_indices: Sequence[Sequence[int]],
    *,
    old_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if not accepted_node_indices:
        raise ValueError("accepted_node_indices is empty")
    max_len = max(len(path) for path in accepted_node_indices)
    if max_len <= 0:
        raise ValueError("accepted_node_indices contains an empty path")
    rows = []
    for path in accepted_node_indices:
        if not path:
            raise ValueError("accepted path cannot be empty")
        padded = list(path) + [path[-1]] * (max_len - len(path))
        rows.append([old_seq_len + int(node_idx) for node_idx in padded])
    return torch.tensor(rows, dtype=torch.long, device=device), max_len


def _gather_paths_from_layer(
    tree_tensor: torch.Tensor,
    path_positions: torch.Tensor,
) -> torch.Tensor:
    # Common HF/Qwen cache shape: [batch, num_kv_heads, seq_len, head_dim].
    if tree_tensor.dim() < 4:
        raise ValueError(f"Unsupported cache tensor shape: {tuple(tree_tensor.shape)}")
    batch, heads, _, head_dim = tree_tensor.shape[:4]
    if path_positions.shape[0] != batch:
        raise ValueError(
            f"accepted path batch {path_positions.shape[0]} != cache batch {batch}"
        )
    expanded = path_positions[:, None, :, None].expand(batch, heads, path_positions.shape[1], head_dim)
    return tree_tensor.gather(dim=2, index=expanded).contiguous()


def _compact_tree_layer_in_place(
    tree_tensor: torch.Tensor,
    path_positions: torch.Tensor,
    *,
    old_seq_len: int,
    max_len: int,
) -> torch.Tensor:
    """Move accepted tree KV into the first tail slots without copying prefix KV."""

    path = _gather_paths_from_layer(tree_tensor, path_positions)
    tree_tensor[:, :, old_seq_len : old_seq_len + max_len, :].copy_(path)
    # The view is intentionally allowed to be non-contiguous. DynamicCache
    # concatenates it with the next token block before attention, producing a
    # contiguous tensor while avoiding an O(prefix_length) copy here.
    return tree_tensor[:, :, : old_seq_len + max_len, :]


def extract_accepted_path_kv(
    old_past_key_values,
    tree_past_key_values,
    accepted_node_indices: Sequence[Sequence[int]],
    *,
    causal_lm=None,
    cache_format: str = "auto",
    compact_in_place: bool = True,
) -> KvExtractionResult:
    """Compact tree-forward KV cache to old prefix + accepted path nodes.

    Tree verification forwards all candidate nodes at positions
    ``old_seq_len .. old_seq_len + N_nodes - 1``. For each batch row we gather
    the accepted root-to-leaf path and pad shorter paths by repeating the last
    accepted node. The corresponding attention mask marks those padded cache
    slots invalid, so their key/value contents are never attended.
    """

    if old_past_key_values is None or tree_past_key_values is None:
        raise ValueError("Both old and tree past_key_values are required")
    old_seq_len = cache_seq_length(old_past_key_values)
    first_key, _ = _get_cache_layer(tree_past_key_values, 0)
    path_positions, max_len = _build_path_index(
        accepted_node_indices,
        old_seq_len=old_seq_len,
        device=first_key.device,
    )
    tree_seq_len = int(first_key.shape[2])
    max_path_position = old_seq_len + max(max(int(node_idx) for node_idx in path) for path in accepted_node_indices)
    if max_path_position >= tree_seq_len:
        raise ValueError(
            f"Accepted node points outside tree cache: max_pos={max_path_position}, "
            f"tree_seq_len={tree_seq_len}, old_seq_len={old_seq_len}"
        )

    if hasattr(tree_past_key_values, "key_cache"):
        new_cache = tree_past_key_values if compact_in_place else _new_dynamic_cache(causal_lm)
        new_keys = []
        new_values = []
        for layer_idx in range(_cache_layer_count(tree_past_key_values)):
            tree_key, tree_value = _get_cache_layer(tree_past_key_values, layer_idx)
            if compact_in_place:
                new_keys.append(
                    _compact_tree_layer_in_place(
                        tree_key,
                        path_positions,
                        old_seq_len=old_seq_len,
                        max_len=max_len,
                    )
                )
                new_values.append(
                    _compact_tree_layer_in_place(
                        tree_value,
                        path_positions,
                        old_seq_len=old_seq_len,
                        max_len=max_len,
                    )
                )
            else:
                prefix_key = tree_key[:, :, :old_seq_len, :].contiguous()
                prefix_value = tree_value[:, :, :old_seq_len, :].contiguous()
                path_key = _gather_paths_from_layer(tree_key, path_positions)
                path_value = _gather_paths_from_layer(tree_value, path_positions)
                new_keys.append(torch.cat([prefix_key, path_key], dim=2).contiguous())
                new_values.append(torch.cat([prefix_value, path_value], dim=2).contiguous())
        new_cache.key_cache = new_keys
        new_cache.value_cache = new_values
        mode = "dynamic_key_cache_in_place" if compact_in_place else "dynamic_key_cache"
        return KvExtractionResult(new_cache, max_len, mode)

    if isinstance(tree_past_key_values, (tuple, list)):
        layers = []
        for tree_key, tree_value in tree_past_key_values:
            if compact_in_place:
                key = _compact_tree_layer_in_place(
                    tree_key,
                    path_positions,
                    old_seq_len=old_seq_len,
                    max_len=max_len,
                )
                value = _compact_tree_layer_in_place(
                    tree_value,
                    path_positions,
                    old_seq_len=old_seq_len,
                    max_len=max_len,
                )
            else:
                prefix_key = tree_key[:, :, :old_seq_len, :].contiguous()
                prefix_value = tree_value[:, :, :old_seq_len, :].contiguous()
                path_key = _gather_paths_from_layer(tree_key, path_positions)
                path_value = _gather_paths_from_layer(tree_value, path_positions)
                key = torch.cat([prefix_key, path_key], dim=2).contiguous()
                value = torch.cat([prefix_value, path_value], dim=2).contiguous()
            layers.append((key, value))
        mode = "legacy_tuple_in_place" if compact_in_place else "legacy_tuple"
        return KvExtractionResult(tuple(layers), max_len, mode)

    # New Cache classes in transformers may expose layers instead of key_cache.
    if hasattr(tree_past_key_values, "layers"):
        new_cache = tree_past_key_values if compact_in_place else _new_dynamic_cache(causal_lm)
        for layer_idx in range(_cache_layer_count(tree_past_key_values)):
            tree_key, tree_value = _get_cache_layer(tree_past_key_values, layer_idx)
            if compact_in_place:
                compact_key = _compact_tree_layer_in_place(
                    tree_key,
                    path_positions,
                    old_seq_len=old_seq_len,
                    max_len=max_len,
                )
                compact_value = _compact_tree_layer_in_place(
                    tree_value,
                    path_positions,
                    old_seq_len=old_seq_len,
                    max_len=max_len,
                )
            else:
                prefix_key = tree_key[:, :, :old_seq_len, :].contiguous()
                prefix_value = tree_value[:, :, :old_seq_len, :].contiguous()
                path_key = _gather_paths_from_layer(tree_key, path_positions)
                path_value = _gather_paths_from_layer(tree_value, path_positions)
                compact_key = torch.cat([prefix_key, path_key], dim=2).contiguous()
                compact_value = torch.cat([prefix_value, path_value], dim=2).contiguous()
            _set_cache_layer(
                new_cache,
                layer_idx,
                compact_key,
                compact_value,
            )
        mode = "dynamic_layers_in_place" if compact_in_place else "dynamic_layers"
        return KvExtractionResult(new_cache, max_len, mode)

    raise TypeError(f"Unsupported cache type for KV path extraction: {type(tree_past_key_values)}")
