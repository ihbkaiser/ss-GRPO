"""Batch-aware CLaSp + concurrency-aware self-speculative generation for GRPO.

This is a drop-in improvement over self_speculative_grpo.generate:
- preserves FastGRPO-style concurrency-aware draft depth;
- preserves dynamic confidence threshold;
- adds CLaSp-style in-context skip-path routing;
- limits active skip paths and merges small groups to protect GPU utilization;
- full target verification is merged into one batch to keep rollout distribution exact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch

from self_speculative_grpo.generate import (
    SelfSpeculativeConfig,
    parse_skip_layers,
    _attention_mask_4d_from_2d,
    _clamp,
    _clone_cache,
    _concurrency_aware_draft_steps,
    _crop_cache,
    _distribution_batched,
    _empty_result,
    _forward_logits_batched,
    _model_device,
    _new_cache,
    _position_ids_from_attention_mask,
    _sample_from_probs,
    _sample_residual_batched,
    _select_cache_batch,
    _sync,
    _unwrap_causal_lm,
)
from self_speculative_claps.clasp_controller import (
    ClaspRoutingStats,
    build_initial_codebook,
    layers_to_string,
    maybe_admit_masks_to_codebook,
    parse_layer_set,
    propose_skip_masks_from_hidden_states,
    quantize_and_merge,
)


@dataclass
class ClaspGenerateConfig:
    enable_clasp: bool = True
    clasp_codebook_size: int = 8
    clasp_max_active_paths: int = 4
    clasp_min_group_size: int = 24
    clasp_update_interval: int = 4
    clasp_low_accept_trigger: float = 0.62
    clasp_protected_first: int = 4
    clasp_protected_last: int = 4
    clasp_candidate_layers: str = ""
    clasp_representative_rows: int = 0
    clasp_dynamic_codebook: bool = True
    clasp_min_code_frequency: int = 2
    clasp_disable_when_bsz_below: int = 16
    clasp_skip_count: int = 0
    # If clasp_skip_count == 0, infer skipped layers from this ratio.
    # 0.60 means skip roughly 60% of transformer blocks, clipped by protected/candidate layers.
    clasp_skip_ratio: float = 0.60
    # Running CLaSp on prefill hidden states can be memory-heavy for long prompts.
    # Default is False: the first round uses a deterministic 60% middle-layer path,
    # then CLaSp updates from cheap short verify hidden states.
    clasp_prefill_update: bool = False


def self_speculative_generate_clasp(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    *,
    skip_layers: str | Iterable[int] | None,
    max_draft_tokens: int = 4,
    confidence_threshold: float = 0.0,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.95,
    repeated_generate_nums: int | None = None,
    max_length: int = 2048,
    statistical_time: bool = True,
    verification_capacity: int = 160,
    max_verification_num: int = 160,
    min_draft_tokens: int = 1,
    draft_token_length_c: float = 0.75,
    dynamic_confidence_threshold: bool = True,
    target_accept_rate: float = 0.70,
    threshold_lr: float = 0.05,
    min_confidence_threshold: float = 0.05,
    max_confidence_threshold: float = 0.90,
    threshold_ema_beta: float = 0.90,
    # CLaSp controls
    enable_clasp: bool = True,
    clasp_codebook_size: int = 8,
    clasp_max_active_paths: int = 4,
    clasp_min_group_size: int = 24,
    clasp_update_interval: int = 4,
    clasp_low_accept_trigger: float = 0.62,
    clasp_protected_first: int = 4,
    clasp_protected_last: int = 4,
    clasp_candidate_layers: str = "",
    clasp_representative_rows: int = 0,
    clasp_dynamic_codebook: bool = True,
    clasp_min_code_frequency: int = 2,
    clasp_disable_when_bsz_below: int = 16,
    clasp_skip_count: int = 0,
    clasp_skip_ratio: float = 0.60,
    clasp_prefill_update: bool = False,
):
    base_config = SelfSpeculativeConfig(
        skip_layers=parse_skip_layers(skip_layers),
        max_draft_tokens=max(0, int(max_draft_tokens)),
        confidence_threshold=float(confidence_threshold),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        verification_capacity=max(1, int(verification_capacity)),
        max_verification_num=max(1, int(max_verification_num)),
        min_draft_tokens=max(0, int(min_draft_tokens)),
        draft_token_length_c=max(float(draft_token_length_c), 1e-6),
        dynamic_confidence_threshold=bool(dynamic_confidence_threshold),
        target_accept_rate=float(target_accept_rate),
        threshold_lr=float(threshold_lr),
        min_confidence_threshold=float(min_confidence_threshold),
        max_confidence_threshold=float(max_confidence_threshold),
        threshold_ema_beta=float(threshold_ema_beta),
    )
    clasp_config = ClaspGenerateConfig(
        enable_clasp=bool(enable_clasp),
        clasp_codebook_size=max(1, int(clasp_codebook_size)),
        clasp_max_active_paths=max(1, int(clasp_max_active_paths)),
        clasp_min_group_size=max(1, int(clasp_min_group_size)),
        clasp_update_interval=max(1, int(clasp_update_interval)),
        clasp_low_accept_trigger=float(clasp_low_accept_trigger),
        clasp_protected_first=max(0, int(clasp_protected_first)),
        clasp_protected_last=max(0, int(clasp_protected_last)),
        clasp_candidate_layers=str(clasp_candidate_layers or ""),
        clasp_representative_rows=max(0, int(clasp_representative_rows)),
        clasp_dynamic_codebook=bool(clasp_dynamic_codebook),
        clasp_min_code_frequency=max(1, int(clasp_min_code_frequency)),
        clasp_disable_when_bsz_below=max(1, int(clasp_disable_when_bsz_below)),
        clasp_skip_count=max(0, int(clasp_skip_count)),
        clasp_skip_ratio=min(max(float(clasp_skip_ratio), 0.05), 0.95),
        clasp_prefill_update=bool(clasp_prefill_update),
    )

    repeats = int(repeated_generate_nums or 1)
    if repeats < 1:
        repeats = 1

    device = _model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is required for self_speculative_generate_clasp")
    eos_token_id = int(eos_token_id)

    start_time = time.time()
    draft_time = 0.0
    verify_time = 0.0
    prefill_time = 0.0
    post_time = 0.0
    clasp_stats = ClaspRoutingStats()
    last_clasp_layer_stats: dict = {}

    if repeats > 1:
        input_ids = input_ids.repeat_interleave(repeats, dim=0)
        attention_mask = attention_mask.repeat_interleave(repeats, dim=0)

    initial_bsz = int(input_ids.shape[0])
    if initial_bsz == 0:
        out = _empty_result(start_time)
        out.update(_empty_clasp_metrics(clasp_config))
        return out

    base_model = _unwrap_causal_lm(model)
    num_layers = int(getattr(base_model.config, "num_hidden_layers", len(base_model.model.layers)))
    candidate_layers = parse_layer_set(clasp_config.clasp_candidate_layers, num_layers) if clasp_config.clasp_candidate_layers else None

    # No Bayesian/static search is required.  CLaSp needs only the number of
    # layers it is allowed to skip.  If the user does not pass an exact
    # clasp_skip_count, infer it from clasp_skip_ratio over the full depth, then
    # clip to the executable candidate region after protected layers are removed.
    protected_candidate_count = len([
        i for i in range(num_layers)
        if clasp_config.clasp_protected_first <= i < num_layers - clasp_config.clasp_protected_last
        and (candidate_layers is None or i in set(candidate_layers))
    ])
    if clasp_config.clasp_skip_count > 0:
        runtime_skip_count = min(clasp_config.clasp_skip_count, max(1, protected_candidate_count))
    else:
        runtime_skip_count = min(
            max(1, int(round(clasp_config.clasp_skip_ratio * num_layers))),
            max(1, protected_candidate_count),
        )

    # C0 is a deterministic middle-layer fallback.  It is used only before the
    # first CLaSp update and whenever Bcur is too small for multi-path routing.
    codebook = build_initial_codebook(
        num_layers=num_layers,
        base_skip_layers=base_config.skip_layers,
        codebook_size=clasp_config.clasp_codebook_size,
        candidate_layers=candidate_layers,
        protected_first=clasp_config.clasp_protected_first,
        protected_last=clasp_config.clasp_protected_last,
        skip_count=runtime_skip_count,
    )

    generated = [[] for _ in range(initial_bsz)]
    active_indices = torch.arange(initial_bsz, dtype=torch.long, device=device)
    prompt_lengths = attention_mask.sum(dim=-1).long()
    generated_lengths = torch.zeros(initial_bsz, dtype=torch.long, device=device)
    # Assignment is a codebook index for each live row. Start from C0.
    code_assignments = torch.zeros(initial_bsz, dtype=torch.long, device=device)

    target_cache = _new_cache(model)
    full_attention_mask = attention_mask.long()
    prefill_position_ids = _position_ids_from_attention_mask(full_attention_mask, input_ids.shape[1])

    if statistical_time:
        _sync(device)
        t0 = time.time()
    need_prefill_clasp_states = bool(
        clasp_config.enable_clasp
        and clasp_config.clasp_prefill_update
        and initial_bsz >= clasp_config.clasp_disable_when_bsz_below
    )
    prefill_logits, target_cache, prefill_hidden_states = _target_forward_logits_batched_clasp(
        model,
        input_ids=input_ids,
        past_key_values=target_cache,
        attention_mask=full_attention_mask,
        position_ids=prefill_position_ids,
        logits_to_keep=1,
        output_hidden_states=need_prefill_clasp_states,
    )
    if statistical_time:
        _sync(device)
        prefill_time += time.time() - t0

    first_probs = _distribution_batched(prefill_logits[:, -1, :], base_config.temperature, base_config.top_p)
    current_token = _sample_from_probs(first_probs, base_config.do_sample)

    # Optional exact CLaSp-style first update from prefill hidden states.  This is
    # disabled by default because output_hidden_states=True on long prompts can
    # materially increase VRAM.  The normal path updates after the first short
    # verify round instead.
    if need_prefill_clasp_states and prefill_hidden_states is not None:
        if statistical_time:
            _sync(device)
            t0 = time.time()
        prefill_positions = (attention_mask.sum(dim=-1) - 1).to(device=device, dtype=torch.long).clamp_min(0)
        ideal_masks, layer_stats = propose_skip_masks_from_hidden_states(
            prefill_hidden_states,
            prefill_positions,
            candidate_layers=candidate_layers,
            protected_first=clasp_config.clasp_protected_first,
            protected_last=clasp_config.clasp_protected_last,
            skip_count=runtime_skip_count,
            max_rows=clasp_config.clasp_representative_rows or None,
        )
        last_clasp_layer_stats = layer_stats
        if ideal_masks:
            if clasp_config.clasp_dynamic_codebook:
                codebook = maybe_admit_masks_to_codebook(
                    codebook,
                    ideal_masks,
                    max_codebook_size=clasp_config.clasp_codebook_size,
                    min_frequency=clasp_config.clasp_min_code_frequency,
                )
            new_assignment_cpu, _, _ = quantize_and_merge(
                ideal_masks,
                codebook,
                max_active_paths=clasp_config.clasp_max_active_paths,
                min_group_size=min(clasp_config.clasp_min_group_size, max(1, initial_bsz // max(1, clasp_config.clasp_max_active_paths))),
                default_code_idx=0,
            )
            code_assignments = new_assignment_cpu.to(device=device)
            clasp_stats.update_count += 1
        if statistical_time:
            _sync(device)
            clasp_stats.clasp_time_cost += time.time() - t0
        del prefill_hidden_states

    finished = prompt_lengths >= max_length
    for row in range(initial_bsz):
        if bool(finished[row].item()):
            continue
        tok = int(current_token[row].item())
        generated[row].append(tok)
        generated_lengths[row] += 1
        if tok == eos_token_id or prompt_lengths[row] + generated_lengths[row] >= max_length:
            finished[row] = True

    keep = (~finished).nonzero(as_tuple=False).flatten()
    if keep.numel() == 0:
        return _final_result_clasp(
            generated,
            start_time,
            total_acc_length=0,
            total_decoded_token_num=0,
            draft_time=draft_time,
            verify_time=verify_time,
            prefill_time=prefill_time,
            post_time=post_time,
            total_draft_steps=0,
            total_cache_tokens_dropped=0,
            total_proposed_draft_tokens=0,
            total_accepted_draft_tokens=0,
            total_verify_rounds=0,
            active_batch_size_sum=0,
            active_batch_size_min=0,
            active_batch_size_max=0,
            verification_num_sum=0,
            draft_steps_sum=0,
            final_confidence_threshold=float(confidence_threshold),
            average_confidence_threshold=float(confidence_threshold),
            min_confidence_threshold_seen=float(confidence_threshold),
            max_confidence_threshold_seen=float(confidence_threshold),
            dynamic_confidence_threshold=bool(dynamic_confidence_threshold),
            target_accept_rate=float(target_accept_rate),
            clasp_config=clasp_config,
            clasp_stats=clasp_stats,
            codebook=codebook,
            last_clasp_layer_stats=last_clasp_layer_stats,
        )

    input_ids = input_ids[keep]
    current_token = current_token[keep]
    prompt_lengths = prompt_lengths[keep]
    generated_lengths = generated_lengths[keep]
    full_attention_mask = full_attention_mask[keep]
    active_indices = active_indices[keep]
    code_assignments = code_assignments[keep]
    target_cache = _select_cache_batch(target_cache, keep)

    total_acc_length = 0
    total_decoded_token_num = 0
    adaptive_accept_ema = None
    total_draft_steps = 0
    total_cache_tokens_dropped = 0
    total_proposed_draft_tokens = 0
    total_accepted_draft_tokens = 0
    total_verify_rounds = 0
    active_batch_size_sum = 0
    active_batch_size_min = initial_bsz
    active_batch_size_max = initial_bsz
    verification_num_sum = 0
    draft_steps_sum = 0

    current_confidence_threshold = _clamp(
        max(base_config.confidence_threshold, base_config.min_confidence_threshold) if base_config.dynamic_confidence_threshold else base_config.confidence_threshold,
        base_config.min_confidence_threshold if base_config.dynamic_confidence_threshold else 0.0,
        base_config.max_confidence_threshold,
    )
    threshold_ema = None
    threshold_sum = 0.0
    threshold_min_seen = current_confidence_threshold
    threshold_max_seen = current_confidence_threshold
    verify_round_idx = 0

    while current_token.numel() > 0:
        cur_bsz = int(current_token.shape[0])
        remaining = (max_length - prompt_lengths - generated_lengths).clamp_min(0)
        max_remaining = int(remaining.max().item()) if cur_bsz else 0
        if max_remaining <= 0:
            break

        draft_steps, verification_num = _concurrency_aware_draft_steps(
            active_batch_size=cur_bsz,
            verification_capacity=base_config.verification_capacity,
            max_draft_tokens=base_config.max_draft_tokens,
            max_verification_num=base_config.max_verification_num,
            min_draft_tokens=base_config.min_draft_tokens,
            draft_token_length_c=base_config.draft_token_length_c,
        )
        draft_steps = min(draft_steps, max_remaining)
        total_verify_rounds += 1
        active_batch_size_sum += cur_bsz
        active_batch_size_min = min(active_batch_size_min, cur_bsz)
        active_batch_size_max = max(active_batch_size_max, cur_bsz)
        verification_num_sum += verification_num
        draft_steps_sum += draft_steps
        threshold_sum += current_confidence_threshold
        threshold_min_seen = min(threshold_min_seen, current_confidence_threshold)
        threshold_max_seen = max(threshold_max_seen, current_confidence_threshold)

        # For small tails, force one default path to avoid splitting tiny batches.
        if (not clasp_config.enable_clasp) or cur_bsz < clasp_config.clasp_disable_when_bsz_below:
            code_assignments = torch.zeros(cur_bsz, dtype=torch.long, device=device)
            active_code_indices = [0]
            route_stats = {
                "clasp_merged_rows": int(cur_bsz - int((code_assignments == 0).sum().item())),
                "clasp_active_path_counts": [cur_bsz],
                "clasp_default_rows": cur_bsz,
                "clasp_average_skip_layers_this_round": float(len(codebook[0])),
            }
        else:
            # Make sure current assignments refer to existing codebook entries.
            code_assignments = torch.clamp(code_assignments, min=0, max=len(codebook) - 1)
            active_code_indices = sorted(set(int(x) for x in code_assignments.detach().cpu().tolist()))
            # Merge if previous assignment became too fragmented.
            prev_masks = [codebook[int(x)] for x in code_assignments.detach().cpu().tolist()]
            new_assignment_cpu, active_code_indices, route_stats = quantize_and_merge(
                prev_masks,
                codebook,
                max_active_paths=clasp_config.clasp_max_active_paths,
                min_group_size=min(clasp_config.clasp_min_group_size, max(1, cur_bsz // max(1, clasp_config.clasp_max_active_paths))),
                default_code_idx=0,
            )
            code_assignments = new_assignment_cpu.to(device=device)

        clasp_stats.update_route(
            active_paths=len(active_code_indices),
            codebook_size=len(codebook),
            merged_rows=int(route_stats.get("clasp_merged_rows", 0)),
            counts=route_stats.get("clasp_active_path_counts", []),
            avg_skip_layers=float(route_stats.get("clasp_average_skip_layers_this_round", len(codebook[0]))),
            default_rows=int(route_stats.get("clasp_default_rows", 0)),
        )

        # Draft stage: split only the draft pass by active code, then merge for one target verify.
        if statistical_time:
            _sync(device)
            t0 = time.time()
        draft_token_matrix, draft_valid_matrix, draft_probs_per_step, round_proposed_draft_tokens, actual_draft_steps = _draft_with_codebook_groups(
            model=model,
            current_token=current_token,
            remaining=remaining,
            full_attention_mask=full_attention_mask,
            target_cache=target_cache,
            code_assignments=code_assignments,
            codebook=codebook,
            active_code_indices=active_code_indices,
            draft_steps=draft_steps,
            confidence_threshold=current_confidence_threshold,
            eos_token_id=eos_token_id,
            config=base_config,
        )
        if statistical_time:
            _sync(device)
            draft_time += time.time() - t0

        total_draft_steps += actual_draft_steps
        total_proposed_draft_tokens += round_proposed_draft_tokens

        verify_input = torch.cat([current_token.unsqueeze(1), draft_token_matrix], dim=1)
        verify_attention_mask = torch.cat(
            [
                full_attention_mask,
                torch.ones((cur_bsz, 1), dtype=torch.long, device=device),
                draft_valid_matrix.long(),
            ],
            dim=1,
        )
        verify_position_ids = _position_ids_from_attention_mask(verify_attention_mask, verify_input.shape[1])
        cache_past_len = full_attention_mask.shape[1]

        need_clasp_states = False
        if clasp_config.enable_clasp and cur_bsz >= clasp_config.clasp_disable_when_bsz_below:
            cadence = (verify_round_idx % clasp_config.clasp_update_interval) == 0
            low_accept = adaptive_accept_ema is not None and adaptive_accept_ema < clasp_config.clasp_low_accept_trigger
            need_clasp_states = bool(cadence or low_accept)

        if statistical_time:
            _sync(device)
            t0 = time.time()
        verify_logits, target_cache, verify_hidden_states = _target_forward_logits_batched_clasp(
            model,
            input_ids=verify_input,
            past_key_values=target_cache,
            attention_mask=verify_attention_mask,
            position_ids=verify_position_ids,
            logits_to_keep=0,
            output_hidden_states=need_clasp_states,
        )
        if statistical_time:
            _sync(device)
            verify_time += time.time() - t0

        q_len = int(verify_logits.shape[1])
        vocab_size = int(verify_logits.shape[-1])
        verify_probs = _distribution_batched(
            verify_logits.reshape(cur_bsz * q_len, vocab_size),
            base_config.temperature,
            base_config.top_p,
        ).reshape(cur_bsz, q_len, vocab_size)
        sampled_target_tokens = _sample_from_probs(
            verify_probs.reshape(cur_bsz * q_len, vocab_size),
            base_config.do_sample,
        ).reshape(cur_bsz, q_len)

        if actual_draft_steps > 0:
            draft_probs_stack = torch.stack(draft_probs_per_step, dim=1)  # [B, D, V]
            draft_targets = verify_probs[:, :actual_draft_steps, :]
            token_idx = draft_token_matrix[:, :actual_draft_steps].unsqueeze(-1)

            if base_config.do_sample:
                target_prob = torch.gather(draft_targets, dim=-1, index=token_idx).squeeze(-1).clamp_min(0.0)
                draft_prob = torch.gather(draft_probs_stack, dim=-1, index=token_idx).squeeze(-1).clamp_min(1e-12)
                accept_probs = torch.clamp(target_prob / draft_prob, max=1.0)
                accepted_matrix = torch.rand_like(accept_probs) <= accept_probs
                residual_samples = _sample_residual_batched(draft_targets, draft_probs_stack)
                replacement_matrix = torch.where(accepted_matrix, draft_token_matrix[:, :actual_draft_steps], residual_samples)
            else:
                target_argmax = sampled_target_tokens[:, :actual_draft_steps]
                accepted_matrix = draft_token_matrix[:, :actual_draft_steps] == target_argmax
                replacement_matrix = target_argmax

            accepted_matrix = accepted_matrix & draft_valid_matrix[:, :actual_draft_steps]
        else:
            accepted_matrix = torch.empty((cur_bsz, 0), dtype=torch.bool, device=device)
            replacement_matrix = torch.empty((cur_bsz, 0), dtype=torch.long, device=device)
            draft_probs_stack = None

        next_current = current_token.clone()
        finished_rows = torch.zeros(cur_bsz, dtype=torch.bool, device=device)
        cache_lengths = [1] * cur_bsz
        accepted_lengths_for_metric = []
        round_accepted_draft_tokens = 0

        if statistical_time:
            _sync(device)
            t0 = time.time()
        for row in range(cur_bsz):
            row_remaining = int(remaining[row].item())
            accepted_ids: list[int] = []
            accepted_draft_prefix = 0
            all_draft_accepted = True

            for draft_idx in range(actual_draft_steps):
                if len(accepted_ids) >= row_remaining:
                    all_draft_accepted = False
                    break
                if not bool(draft_valid_matrix[row, draft_idx].item()):
                    all_draft_accepted = False
                    break

                replacement = int(replacement_matrix[row, draft_idx].item())
                accepted = bool(accepted_matrix[row, draft_idx].item())
                accepted_ids.append(replacement)

                if accepted:
                    accepted_draft_prefix += 1
                    if replacement == eos_token_id:
                        all_draft_accepted = False
                        break
                else:
                    all_draft_accepted = False
                    break

            if all_draft_accepted and len(accepted_ids) < row_remaining:
                next_tok = int(sampled_target_tokens[row, actual_draft_steps].item())
                accepted_ids.append(next_tok)

            if not accepted_ids:
                accepted_ids = [int(sampled_target_tokens[row, 0].item())]

            accepted_ids = accepted_ids[:row_remaining]
            original_row = int(active_indices[row].item())
            generated[original_row].extend(accepted_ids)
            generated_lengths[row] += len(accepted_ids)
            total_acc_length += len(accepted_ids)
            total_decoded_token_num += 1
            accepted_lengths_for_metric.append(len(accepted_ids))

            next_current[row] = int(accepted_ids[-1])
            cache_lengths[row] = 1 + accepted_draft_prefix
            round_accepted_draft_tokens += accepted_draft_prefix

            if accepted_ids[-1] == eos_token_id or prompt_lengths[row] + generated_lengths[row] >= max_length:
                finished_rows[row] = True
        if statistical_time:
            _sync(device)
            post_time += time.time() - t0

        total_accepted_draft_tokens += round_accepted_draft_tokens
        if round_proposed_draft_tokens > 0:
            current_draft_accept_rate = round_accepted_draft_tokens / max(round_proposed_draft_tokens, 1)
            adaptive_accept_ema = (
                current_draft_accept_rate
                if adaptive_accept_ema is None
                else 0.85 * adaptive_accept_ema + 0.15 * current_draft_accept_rate
            )
            if base_config.dynamic_confidence_threshold:
                threshold_ema = (
                    current_draft_accept_rate
                    if threshold_ema is None
                    else base_config.threshold_ema_beta * threshold_ema + (1.0 - base_config.threshold_ema_beta) * current_draft_accept_rate
                )
                current_confidence_threshold = _clamp(
                    current_confidence_threshold + base_config.threshold_lr * (base_config.target_accept_rate - threshold_ema),
                    base_config.min_confidence_threshold,
                    base_config.max_confidence_threshold,
                )

        # CLaSp update for the next round, after we know the last accepted token position.
        if need_clasp_states and verify_hidden_states is not None and cur_bsz >= clasp_config.clasp_disable_when_bsz_below:
            if statistical_time:
                _sync(device)
                t0 = time.time()
            token_positions = torch.tensor([max(0, int(x) - 1) for x in cache_lengths], dtype=torch.long, device=device)
            ideal_masks, layer_stats = propose_skip_masks_from_hidden_states(
                verify_hidden_states,
                token_positions,
                candidate_layers=candidate_layers,
                protected_first=clasp_config.clasp_protected_first,
                protected_last=clasp_config.clasp_protected_last,
                skip_count=runtime_skip_count,
                max_rows=clasp_config.clasp_representative_rows or None,
            )
            last_clasp_layer_stats = layer_stats
            if ideal_masks:
                if clasp_config.clasp_dynamic_codebook:
                    codebook = maybe_admit_masks_to_codebook(
                        codebook,
                        ideal_masks,
                        max_codebook_size=clasp_config.clasp_codebook_size,
                        min_frequency=clasp_config.clasp_min_code_frequency,
                    )
                new_assignment_cpu, _, _ = quantize_and_merge(
                    ideal_masks,
                    codebook,
                    max_active_paths=clasp_config.clasp_max_active_paths,
                    min_group_size=min(clasp_config.clasp_min_group_size, max(1, cur_bsz // max(1, clasp_config.clasp_max_active_paths))),
                    default_code_idx=0,
                )
                code_assignments = new_assignment_cpu.to(device=device)
                clasp_stats.update_count += 1
            if statistical_time:
                _sync(device)
                clasp_stats.clasp_time_cost += time.time() - t0

        max_cache_extension = max(cache_lengths) if cache_lengths else 1
        target_cache = _crop_cache(target_cache, cache_past_len + max_cache_extension)
        cache_valid_extension = torch.zeros((cur_bsz, max_cache_extension), dtype=torch.long, device=device)
        for row, cache_len in enumerate(cache_lengths):
            cache_valid_extension[row, :cache_len] = 1
        total_cache_tokens_dropped += max(0, verify_input.shape[1] - max_cache_extension)
        full_attention_mask = torch.cat([full_attention_mask, cache_valid_extension], dim=1)

        keep = (~finished_rows).nonzero(as_tuple=False).flatten()
        if keep.numel() == 0:
            break

        current_token = next_current[keep]
        prompt_lengths = prompt_lengths[keep]
        generated_lengths = generated_lengths[keep]
        full_attention_mask = full_attention_mask[keep]
        active_indices = active_indices[keep]
        code_assignments = code_assignments[keep]
        target_cache = _select_cache_batch(target_cache, keep)

        verify_round_idx += 1

        del verify_logits, verify_probs, sampled_target_tokens, verify_input, verify_attention_mask, verify_position_ids
        del draft_token_matrix, draft_valid_matrix, draft_probs_per_step
        if actual_draft_steps > 0:
            del draft_probs_stack, accepted_matrix, replacement_matrix
        if verify_hidden_states is not None:
            del verify_hidden_states

    return _final_result_clasp(
        generated,
        start_time,
        total_acc_length=total_acc_length,
        total_decoded_token_num=total_decoded_token_num,
        draft_time=draft_time,
        verify_time=verify_time,
        prefill_time=prefill_time,
        post_time=post_time,
        total_draft_steps=total_draft_steps,
        total_cache_tokens_dropped=total_cache_tokens_dropped,
        total_proposed_draft_tokens=total_proposed_draft_tokens,
        total_accepted_draft_tokens=total_accepted_draft_tokens,
        total_verify_rounds=total_verify_rounds,
        active_batch_size_sum=active_batch_size_sum,
        active_batch_size_min=active_batch_size_min if total_verify_rounds else 0,
        active_batch_size_max=active_batch_size_max if total_verify_rounds else 0,
        verification_num_sum=verification_num_sum,
        draft_steps_sum=draft_steps_sum,
        final_confidence_threshold=current_confidence_threshold,
        average_confidence_threshold=threshold_sum / total_verify_rounds if total_verify_rounds else current_confidence_threshold,
        min_confidence_threshold_seen=threshold_min_seen,
        max_confidence_threshold_seen=threshold_max_seen,
        dynamic_confidence_threshold=base_config.dynamic_confidence_threshold,
        target_accept_rate=base_config.target_accept_rate,
        clasp_config=clasp_config,
        clasp_stats=clasp_stats,
        codebook=codebook,
        last_clasp_layer_stats=last_clasp_layer_stats,
    )


def _draft_with_codebook_groups(
    *,
    model,
    current_token: torch.Tensor,
    remaining: torch.Tensor,
    full_attention_mask: torch.Tensor,
    target_cache,
    code_assignments: torch.Tensor,
    codebook: list[frozenset[int]],
    active_code_indices: list[int],
    draft_steps: int,
    confidence_threshold: float,
    eos_token_id: int,
    config: SelfSpeculativeConfig,
):
    device = current_token.device
    cur_bsz = int(current_token.shape[0])
    draft_steps = max(0, int(draft_steps))
    if draft_steps <= 0:
        empty_tokens = torch.empty((cur_bsz, 0), dtype=torch.long, device=device)
        empty_valid = torch.empty((cur_bsz, 0), dtype=torch.bool, device=device)
        return empty_tokens, empty_valid, [], 0, 0

    global_tokens: list[torch.Tensor | None] = [None for _ in range(draft_steps)]
    global_valids: list[torch.Tensor | None] = [None for _ in range(draft_steps)]
    global_probs: list[torch.Tensor | None] = [None for _ in range(draft_steps)]
    actual_draft_steps = 0

    # Process largest groups first for better kernel behavior.
    groups = []
    for code_idx in active_code_indices:
        rows = (code_assignments == int(code_idx)).nonzero(as_tuple=False).flatten()
        if rows.numel() > 0:
            groups.append((int(rows.numel()), int(code_idx), rows))
    groups.sort(reverse=True, key=lambda x: x[0])

    for _, code_idx, rows in groups:
        skip_layers = codebook[code_idx]
        group_current = current_token.index_select(0, rows)
        group_remaining = remaining.index_select(0, rows)
        group_attention_mask = full_attention_mask.index_select(0, rows)
        group_cache = _clone_cache(target_cache, model)
        group_cache = _select_cache_batch(group_cache, rows)

        group_draft_active = group_remaining > 0
        group_draft_input = group_current.unsqueeze(1)
        group_draft_attention_mask = group_attention_mask

        for step_idx in range(draft_steps):
            valid_for_step = group_draft_active & (group_remaining > step_idx)
            if not bool(valid_for_step.any().item()):
                break
            query_mask = valid_for_step.long().unsqueeze(1)
            group_draft_attention_mask = torch.cat([group_draft_attention_mask, query_mask], dim=1)
            draft_position_ids = _position_ids_from_attention_mask(group_draft_attention_mask, 1)
            draft_logits, group_cache = _forward_logits_batched(
                model,
                input_ids=group_draft_input,
                skip_layers=skip_layers,
                past_key_values=group_cache,
                full_attention_mask=group_draft_attention_mask,
                position_ids=draft_position_ids,
                use_cache=True,
            )
            probs = _distribution_batched(draft_logits[:, -1, :], config.temperature, config.top_p)
            sampled = _sample_from_probs(probs, config.do_sample)
            sampled = torch.where(valid_for_step, sampled, torch.full_like(sampled, eos_token_id))

            if global_tokens[step_idx] is None:
                global_tokens[step_idx] = torch.full((cur_bsz,), eos_token_id, dtype=torch.long, device=device)
                global_valids[step_idx] = torch.zeros((cur_bsz,), dtype=torch.bool, device=device)
                global_probs[step_idx] = torch.zeros((cur_bsz, probs.shape[-1]), dtype=probs.dtype, device=device)
            global_tokens[step_idx][rows] = sampled
            global_valids[step_idx][rows] = valid_for_step
            global_probs[step_idx][rows] = probs
            actual_draft_steps = max(actual_draft_steps, step_idx + 1)

            confidence = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            group_draft_active = valid_for_step & (confidence >= confidence_threshold) & (sampled != eos_token_id)
            group_draft_input = sampled.unsqueeze(1)

    if actual_draft_steps == 0:
        empty_tokens = torch.empty((cur_bsz, 0), dtype=torch.long, device=device)
        empty_valid = torch.empty((cur_bsz, 0), dtype=torch.bool, device=device)
        return empty_tokens, empty_valid, [], 0, 0

    tokens = torch.stack([global_tokens[i] for i in range(actual_draft_steps)], dim=1)
    valid = torch.stack([global_valids[i] for i in range(actual_draft_steps)], dim=1)
    probs_per_step = [global_probs[i] for i in range(actual_draft_steps)]
    proposed = int(valid.sum().item())
    return tokens, valid, probs_per_step, proposed, actual_draft_steps


def _target_forward_logits_batched_clasp(
    model,
    input_ids,
    past_key_values,
    attention_mask,
    position_ids,
    logits_to_keep=0,
    output_hidden_states=False,
):
    input_ids = input_ids.to(_model_device(model))
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        logits_to_keep=logits_to_keep,
        output_hidden_states=bool(output_hidden_states),
    )
    return outputs.logits, outputs.past_key_values, (outputs.hidden_states if output_hidden_states else None)


def _final_result_clasp(
    generated,
    start_time,
    *,
    total_acc_length,
    total_decoded_token_num,
    draft_time,
    verify_time,
    prefill_time,
    post_time,
    total_draft_steps,
    total_cache_tokens_dropped,
    total_proposed_draft_tokens=0,
    total_accepted_draft_tokens=0,
    total_verify_rounds=0,
    active_batch_size_sum=0,
    active_batch_size_min=0,
    active_batch_size_max=0,
    verification_num_sum=0,
    draft_steps_sum=0,
    final_confidence_threshold=0.0,
    average_confidence_threshold=0.0,
    min_confidence_threshold_seen=0.0,
    max_confidence_threshold_seen=0.0,
    dynamic_confidence_threshold=False,
    target_accept_rate=0.0,
    clasp_config: ClaspGenerateConfig | None = None,
    clasp_stats: ClaspRoutingStats | None = None,
    codebook: list[frozenset[int]] | None = None,
    last_clasp_layer_stats: dict | None = None,
):
    total_time = time.time() - start_time
    generated_token_ids = [seq for seq in generated]
    max_sequence_length = max((len(item) for item in generated_token_ids), default=0)
    out = {
        "generated_token_ids": generated_token_ids,
        "max_sequence_length": max_sequence_length,
        "total_acc_length": total_acc_length,
        "total_decoded_token_num": total_decoded_token_num,
        "total_time_cost": total_time,
        "target_time_cost": verify_time,
        "draft_time_cost": draft_time,
        "check_time_cost": 0.0,
        "prefill_time_cost": prefill_time,
        "post_time_cost": post_time,
        "average_accept_length": total_acc_length / total_decoded_token_num if total_decoded_token_num else 0.0,
        "average_draft_steps": total_draft_steps / total_decoded_token_num if total_decoded_token_num else 0.0,
        "cache_tokens_dropped": total_cache_tokens_dropped,
        "total_proposed_draft_tokens": int(total_proposed_draft_tokens),
        "total_accepted_draft_tokens": int(total_accepted_draft_tokens),
        "draft_acceptance_rate": (
            total_accepted_draft_tokens / total_proposed_draft_tokens
            if total_proposed_draft_tokens else 0.0
        ),
        "total_verify_rounds": int(total_verify_rounds),
        "average_active_batch_size": active_batch_size_sum / total_verify_rounds if total_verify_rounds else 0.0,
        "min_active_batch_size": int(active_batch_size_min),
        "max_active_batch_size": int(active_batch_size_max),
        "average_verification_num": verification_num_sum / total_verify_rounds if total_verify_rounds else 0.0,
        "average_selected_draft_steps": draft_steps_sum / total_verify_rounds if total_verify_rounds else 0.0,
        "final_confidence_threshold": float(final_confidence_threshold),
        "average_confidence_threshold": float(average_confidence_threshold),
        "min_confidence_threshold_seen": float(min_confidence_threshold_seen),
        "max_confidence_threshold_seen": float(max_confidence_threshold_seen),
        "dynamic_confidence_threshold": bool(dynamic_confidence_threshold),
        "target_accept_rate": float(target_accept_rate),
    }
    if clasp_config is not None:
        out.update(_empty_clasp_metrics(clasp_config))
    if clasp_stats is not None:
        out.update(clasp_stats.to_dict())
    if codebook is not None:
        out["clasp_codebook_final"] = [layers_to_string(m) for m in codebook]
        out["clasp_codebook_final_size"] = len(codebook)
    if last_clasp_layer_stats:
        out.update(last_clasp_layer_stats)
    return out


def _empty_clasp_metrics(config: ClaspGenerateConfig) -> dict:
    return {
        "clasp_enabled": bool(config.enable_clasp),
        "clasp_codebook_size_config": int(config.clasp_codebook_size),
        "clasp_max_active_paths_config": int(config.clasp_max_active_paths),
        "clasp_min_group_size_config": int(config.clasp_min_group_size),
        "clasp_update_interval_config": int(config.clasp_update_interval),
        "clasp_disable_when_bsz_below": int(config.clasp_disable_when_bsz_below),
        "clasp_dynamic_codebook": bool(config.clasp_dynamic_codebook),
        "clasp_skip_ratio_config": float(config.clasp_skip_ratio),
        "clasp_prefill_update": bool(config.clasp_prefill_update),
    }


# Compatibility alias if existing training code imports self_speculative_generate.
self_speculative_generate = self_speculative_generate_clasp
