from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import torch

from flashgrpo.decoding.acceptance import exact_accept_path, sample_from_logits
from flashgrpo.decoding.kv_extraction import extract_accepted_path_kv
from flashgrpo.decoding.medusa_tree import TreePlan, build_batch_trees, plan_tree
from flashgrpo.decoding.tree_attention import build_tree_attention_inputs
from flashgrpo.models.qwen_flashgrpo_wrapper import (
    autocast_dtype,
    forward_tokens,
    forward_tree,
    logical_lengths as mask_logical_lengths,
    model_device,
    prefill,
    repeat_interleave_cache,
    select_cache_batch,
    unwrap_causal_lm,
)


@dataclass
class FlashMedusaConfig:
    num_medusa_heads: int = 3
    tree_mode: str = "concurrency_aware"
    tree_layout: str = "dense"
    acceptance: str = "exact_target"
    cache_update_mode: str = "extract_path"
    allow_recompute_fallback: bool = True
    cpeak_nodes: int = 64
    min_tree_nodes_per_seq: int = 1
    max_tree_nodes_per_seq: int = 16
    max_tree_depth: int = 4
    fixed_tree_topk_by_depth: tuple[int, ...] = (4, 3, 2)
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    clone_tree_cache: bool = True
    oom_shrink_factor: float = 0.5
    enable_medusa_spec_after: int = 0


class FlashMedusaDecoder:
    def __init__(self, target_model, medusa_heads, tokenizer, config: FlashMedusaConfig):
        self.target_model = target_model
        self.medusa_heads = medusa_heads
        self.tokenizer = tokenizer
        self.config = config
        if config.acceptance != "exact_target":
            raise NotImplementedError("Only exact_target acceptance is implemented in the main FlashGRPO path")
        if config.cache_update_mode not in {"extract_path", "recompute_accepted"}:
            raise ValueError(f"Unsupported cache_update_mode={config.cache_update_mode}")

    def _last_valid_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
        # The training collator left-pads prompts, so the last non-padding token
        # is at the right edge. This fallback also handles non-left-padded smoke tests.
        if bool((attention_mask[:, -1] == 1).all().item()):
            return hidden_states[:, -1, :]
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.shape[-1])
        return hidden_states.gather(1, gather_idx).squeeze(1)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        repeated_generate_nums: int | None = None,
        max_length: int = 2048,
        statistical_time: bool = False,
        generation_step: int = 0,
    ) -> dict:
        cfg = self.config
        device = model_device(self.target_model)
        repeats = max(1, int(repeated_generate_nums or 1))
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = pad_token_id

        total_start = time.time()
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.time()
        prefill_out = prefill(self.target_model, input_ids, attention_mask)
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        past_key_values = prefill_out["past_key_values"]
        prefill_hidden = prefill_out["hidden_states"]
        current_hidden = self._last_valid_hidden(prefill_hidden, attention_mask)
        del prefill_hidden, prefill_out
        with torch.amp.autocast(
            "cuda" if device.type == "cuda" else device.type,
            dtype=autocast_dtype(base),
            enabled=(device.type == "cuda"),
        ):
            current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
        if repeats > 1:
            past_key_values = repeat_interleave_cache(past_key_values, repeats, causal_lm=self.target_model)
            current_hidden = current_hidden.repeat_interleave(repeats, dim=0).contiguous()
            current_logits = current_logits.repeat_interleave(repeats, dim=0).contiguous()
            attention_mask = attention_mask.repeat_interleave(repeats, dim=0).contiguous()

        total_sequences = attention_mask.shape[0]
        generated: list[list[int]] = [[] for _ in range(total_sequences)]
        active_original_indices = list(range(total_sequences))
        full_attention_mask = attention_mask.long()
        logical_lens = mask_logical_lengths(full_attention_mask)

        total_acc_length = 0
        total_decoded_steps = 0
        total_accepted_medusa_tokens = 0
        total_proposed_medusa_tokens = 0
        total_verify_rounds = 0
        active_batch_sum = 0
        tree_node_sum = 0
        tree_sample_count = 0
        medusa_head_time = 0.0
        tree_verify_time = 0.0
        cache_update_time = 0.0
        kv_extraction_time = 0.0
        recompute_fallback_time = 0.0
        kv_extraction_success_count = 0
        kv_extraction_fallback_count = 0
        oom_count = 0
        accept_hist: dict[int, int] = {}
        accept_by_depth: dict[int, int] = {}
        proposed_by_depth: dict[int, int] = {}
        tree_plan_last = {}

        while active_original_indices:
            active_bsz = len(active_original_indices)
            remaining = max_length - logical_lens
            if not bool((remaining > 0).any().item()):
                break

            root_tokens = sample_from_logits(
                current_logits,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            use_medusa_tree = int(generation_step) >= int(cfg.enable_medusa_spec_after)
            if use_medusa_tree:
                plan = plan_tree(
                    active_batch_size=active_bsz,
                    num_medusa_heads=min(cfg.num_medusa_heads, self.medusa_heads.num_heads),
                    tree_mode=cfg.tree_mode,
                    tree_layout=cfg.tree_layout,
                    cpeak_nodes=cfg.cpeak_nodes,
                    min_tree_nodes_per_seq=cfg.min_tree_nodes_per_seq,
                    max_tree_nodes_per_seq=cfg.max_tree_nodes_per_seq,
                    max_tree_depth=cfg.max_tree_depth,
                    fixed_tree_topk_by_depth=list(cfg.fixed_tree_topk_by_depth),
                )
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                head_start = time.time()
                with torch.no_grad():
                    medusa_logits = self.medusa_heads.logits_for_last_hidden(
                        current_hidden.detach(),
                        lm_head=lm_head,
                        max_heads=plan.active_heads,
                    )
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                medusa_head_time += time.time() - head_start
            else:
                medusa_logits = []
                plan = TreePlan(
                    node_budget_per_seq=1,
                    active_heads=0,
                    topk_by_depth=[],
                    actual_nodes=1,
                    mode=cfg.tree_mode,
                    layout=cfg.tree_layout,
                )
            trees = build_batch_trees(root_tokens, medusa_logits, plan)
            tree_plan_last = {
                "B_cur": active_bsz,
                "node_budget_per_seq": plan.node_budget_per_seq,
                "active_heads": plan.active_heads,
                "topk_by_depth": plan.topk_by_depth,
                "actual_nodes": plan.actual_nodes,
            }
            active_batch_sum += active_bsz
            tree_node_sum += sum(tree.node_count for tree in trees) / max(active_bsz, 1)
            tree_sample_count += 1
            for depth, topk in enumerate(plan.topk_by_depth, start=2):
                proposed_by_depth[depth] = proposed_by_depth.get(depth, 0) + active_bsz * int(topk)

            tree_logits = None
            tree_hidden = None
            tree_past_key_values = None
            if plan.active_heads > 0 and max(tree.node_count for tree in trees) > 1:
                try:
                    tree_input_ids, tree_mask, tree_position_ids, _ = build_tree_attention_inputs(
                        trees,
                        full_attention_mask,
                        logical_lens,
                        pad_token_id=pad_token_id,
                        dtype=autocast_dtype(base),
                    )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    verify_start = time.time()
                    tree_out = forward_tree(
                        self.target_model,
                        tree_input_ids,
                        tree_mask,
                        past_key_values,
                        tree_position_ids,
                        clone_past=cfg.clone_tree_cache,
                    )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    tree_verify_time += time.time() - verify_start
                    tree_logits = tree_out["logits"].float()
                    tree_hidden = tree_out["hidden_states"]
                    tree_past_key_values = tree_out["past_key_values"]
                    del tree_out
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    oom_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tree_logits = None

            accepted_per_row: list[list[int]] = []
            accepted_nodes_per_row: list[list[int]] = []
            finished_flags: list[bool] = []
            parent_nodes: list[int] = []
            for row, tree in enumerate(trees):
                if tree_logits is None or tree.node_count == 1:
                    accepted_tokens = [int(root_tokens[row].item())]
                    accepted_nodes = [0]
                    parent = 0
                else:
                    accepted_tokens, accepted_nodes, parent = exact_accept_path(
                        tree,
                        tree_logits[row, : tree.node_count, :],
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        top_k=cfg.top_k,
                    )
                max_accept = max(1, int(remaining[row].item()))
                accepted_tokens = accepted_tokens[:max_accept]
                accepted_nodes = accepted_nodes[: len(accepted_tokens)]
                eos_seen = False
                for eos_pos, token in enumerate(accepted_tokens):
                    if token == eos_token_id:
                        accepted_tokens = accepted_tokens[: eos_pos + 1]
                        accepted_nodes = accepted_nodes[: eos_pos + 1]
                        eos_seen = True
                        break
                accepted_len = len(accepted_tokens)
                accept_hist[accepted_len] = accept_hist.get(accepted_len, 0) + 1
                for depth in range(2, accepted_len + 1):
                    accept_by_depth[depth] = accept_by_depth.get(depth, 0) + 1
                total_acc_length += accepted_len
                total_decoded_steps += 1
                total_accepted_medusa_tokens += max(accepted_len - 1, 0)
                total_proposed_medusa_tokens += max(tree.node_count - 1, 0)
                accepted_per_row.append(accepted_tokens)
                accepted_nodes_per_row.append([int(node_idx) for node_idx in accepted_nodes])
                finished_flags.append(eos_seen or int(logical_lens[row].item()) + accepted_len >= max_length)
                parent_nodes.append(parent)

            total_verify_rounds += 1
            max_acc = max(len(tokens) for tokens in accepted_per_row)
            accepted_ids = torch.full((active_bsz, max_acc), int(pad_token_id), dtype=torch.long, device=device)
            valid_ext = torch.zeros((active_bsz, max_acc), dtype=torch.long, device=device)
            position_ids = torch.zeros((active_bsz, max_acc), dtype=torch.long, device=device)
            for row, tokens in enumerate(accepted_per_row):
                accepted_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
                valid_ext[row, : len(tokens)] = 1
                position_ids[row, : len(tokens)] = logical_lens[row] + torch.arange(len(tokens), device=device)
                original_idx = active_original_indices[row]
                generated[original_idx].extend(tokens)

            new_attention_mask = torch.cat([full_attention_mask, valid_ext], dim=1)
            use_extract = (
                cfg.cache_update_mode == "extract_path"
                and tree_logits is not None
                and tree_hidden is not None
                and tree_past_key_values is not None
            )
            extracted = False
            if use_extract:
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                extract_start = time.time()
                try:
                    result = extract_accepted_path_kv(
                        past_key_values,
                        tree_past_key_values,
                        accepted_nodes_per_row,
                        causal_lm=self.target_model,
                    )
                    past_key_values = result.past_key_values
                    row_idx = torch.arange(active_bsz, device=device)
                    last_node_idx = torch.tensor(
                        [path[-1] for path in accepted_nodes_per_row],
                        dtype=torch.long,
                        device=device,
                    )
                    current_hidden = tree_hidden[row_idx, last_node_idx, :]
                    current_logits = tree_logits[row_idx, last_node_idx, :]
                    extracted = True
                    kv_extraction_success_count += 1
                    del result, row_idx, last_node_idx
                except Exception as exc:
                    kv_extraction_fallback_count += 1
                    if not cfg.allow_recompute_fallback:
                        raise RuntimeError("KV path extraction failed and allow_recompute_fallback=false") from exc
                    warnings.warn(f"KV path extraction failed; falling back to recompute_accepted: {exc}", RuntimeWarning)
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                kv_extraction_time += time.time() - extract_start
                cache_update_time += time.time() - extract_start
            elif cfg.cache_update_mode == "extract_path" and cfg.allow_recompute_fallback and plan.active_heads > 0:
                kv_extraction_fallback_count += 1

            if not extracted:
                if cfg.cache_update_mode == "extract_path" and not cfg.allow_recompute_fallback and plan.active_heads > 0:
                    raise RuntimeError("KV extraction was requested but no tree cache was available")
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                cache_start = time.time()
                cache_out = forward_tokens(
                    self.target_model,
                    accepted_ids,
                    new_attention_mask,
                    past_key_values,
                    position_ids,
                )
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - cache_start
                cache_update_time += elapsed
                if cfg.cache_update_mode == "extract_path" and plan.active_heads > 0:
                    recompute_fallback_time += elapsed
                past_key_values = cache_out["past_key_values"]
                token_hidden = cache_out["hidden_states"]
                last_indices = (valid_ext.sum(dim=-1) - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, token_hidden.shape[-1])
                current_hidden = token_hidden.gather(1, last_indices).squeeze(1)
                with torch.amp.autocast(
                    "cuda" if device.type == "cuda" else device.type,
                    dtype=autocast_dtype(base),
                    enabled=(device.type == "cuda"),
                ):
                    current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
                del cache_out, token_hidden, last_indices
            tree_logits = None
            tree_hidden = None
            tree_past_key_values = None
            full_attention_mask = new_attention_mask
            logical_lens = logical_lens + valid_ext.sum(dim=-1)

            keep_rows = [idx for idx, done in enumerate(finished_flags) if not done]
            if len(keep_rows) != active_bsz:
                if keep_rows:
                    keep = torch.tensor(keep_rows, dtype=torch.long, device=device)
                    past_key_values = select_cache_batch(past_key_values, keep, causal_lm=self.target_model)
                    current_hidden = current_hidden.index_select(0, keep)
                    current_logits = current_logits.index_select(0, keep)
                    full_attention_mask = full_attention_mask.index_select(0, keep)
                    logical_lens = logical_lens.index_select(0, keep)
                    active_original_indices = [active_original_indices[idx] for idx in keep_rows]
                else:
                    active_original_indices = []
                    break

        max_sequence_length = max((len(seq) for seq in generated), default=0)
        total_time = time.time() - total_start
        accept_rate = total_accepted_medusa_tokens / max(total_proposed_medusa_tokens, 1)
        avg_accept = total_acc_length / max(total_decoded_steps, 1)
        return {
            "generated_token_ids": generated,
            "max_sequence_length": max_sequence_length,
            "total_acc_length": int(total_acc_length),
            "average_accept_length": float(avg_accept),
            "accepted_tokens_per_medusa_step": float(avg_accept),
            "total_decoded_token_num": int(total_decoded_steps),
            "total_accepted_draft_tokens": int(total_accepted_medusa_tokens),
            "total_proposed_draft_tokens": int(total_proposed_medusa_tokens),
            "total_accepted_medusa_tokens": int(total_accepted_medusa_tokens),
            "total_proposed_medusa_tokens": int(total_proposed_medusa_tokens),
            "draft_acceptance_rate": float(accept_rate),
            "medusa_acceptance_rate": float(accept_rate),
            "total_verify_rounds": int(total_verify_rounds),
            "average_active_batch_size": active_batch_sum / max(total_verify_rounds, 1),
            "average_tree_nodes_per_seq": tree_node_sum / max(tree_sample_count, 1),
            "accept_length_histogram": accept_hist,
            "medusa_accept_by_depth": accept_by_depth,
            "medusa_proposed_by_depth": proposed_by_depth,
            "last_tree_plan": tree_plan_last,
            "cache_update_mode": cfg.cache_update_mode,
            "kv_extraction_success_count": int(kv_extraction_success_count),
            "kv_extraction_fallback_count": int(kv_extraction_fallback_count),
            "kv_extraction_time": kv_extraction_time,
            "recompute_fallback_time": recompute_fallback_time,
            "oom_count": int(oom_count),
            "total_time_cost": total_time,
            "prefill_time_cost": prefill_time,
            "target_time_cost": prefill_time + tree_verify_time + cache_update_time,
            "tree_verify_time_cost": tree_verify_time,
            "cache_update_time_cost": cache_update_time,
            "medusa_head_time_cost": medusa_head_time,
            "draft_time_cost": medusa_head_time,
            "check_time_cost": 0.0,
        }
