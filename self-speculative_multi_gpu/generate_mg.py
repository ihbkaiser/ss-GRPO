import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class SelfSpeculativeConfig:
    skip_layers: frozenset[int]
    max_draft_tokens: int = 4
    confidence_threshold: float = 0.0
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95


def parse_skip_layers(value: str | Iterable[int] | None) -> frozenset[int]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        if not value.strip():
            return frozenset()
        return frozenset(int(item.strip()) for item in value.split(",") if item.strip())
    return frozenset(int(item) for item in value)


def self_speculative_generate(
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
):
    """Optimized batched Draft-and-Verify self-speculative decoding.

    Speed optimizations relative to the previous batched version:
    1. finished sequences are physically removed from the batch and KV cache;
    2. verification distributions/acceptance tests are vectorized over [B, draft_depth];
    3. cache compaction uses the actually cached accepted prefix, not generated-token count.

    The output API and token ordering are unchanged:
    prompt0 repeat0..N, prompt1 repeat0..N, ...
    """
    config = SelfSpeculativeConfig(
        skip_layers=parse_skip_layers(skip_layers),
        max_draft_tokens=max(0, int(max_draft_tokens)),
        confidence_threshold=float(confidence_threshold),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
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
        raise ValueError("tokenizer.eos_token_id is required for self_speculative_generate")
    eos_token_id = int(eos_token_id)

    start_time = time.time()
    draft_time = 0.0
    verify_time = 0.0
    prefill_time = 0.0

    if repeats > 1:
        input_ids = input_ids.repeat_interleave(repeats, dim=0)
        attention_mask = attention_mask.repeat_interleave(repeats, dim=0)

    initial_bsz = int(input_ids.shape[0])
    if initial_bsz == 0:
        return _empty_result(start_time)

    # Output buffers are indexed by the original effective-batch row. The live
    # tensors below are compacted as rows finish.
    generated = [[] for _ in range(initial_bsz)]
    active_indices = torch.arange(initial_bsz, dtype=torch.long, device=device)
    prompt_lengths = attention_mask.sum(dim=-1).long()
    generated_lengths = torch.zeros(initial_bsz, dtype=torch.long, device=device)

    target_cache = _new_cache(model)
    full_attention_mask = attention_mask.long()
    prefill_position_ids = _position_ids_from_attention_mask(full_attention_mask, input_ids.shape[1])

    if statistical_time:
        _sync(device)
        t0 = time.time()
    prefill_logits, target_cache = _target_forward_logits_batched(
        model,
        input_ids=input_ids,
        past_key_values=target_cache,
        attention_mask=full_attention_mask,
        position_ids=prefill_position_ids,
        logits_to_keep=1,
    )
    if statistical_time:
        _sync(device)
        prefill_time += time.time() - t0

    first_probs = _distribution_batched(prefill_logits[:, -1, :], config.temperature, config.top_p)
    current_token = _sample_from_probs(first_probs, config.do_sample)

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
        return _final_result(
            generated,
            start_time,
            total_acc_length=0,
            total_decoded_token_num=0,
            draft_time=draft_time,
            verify_time=verify_time,
            prefill_time=prefill_time,
            total_draft_steps=0,
            total_draft_tokens_proposed=0,
            total_draft_tokens_accepted=0,
            total_cache_tokens_dropped=0,
        )

    input_ids = input_ids[keep]
    current_token = current_token[keep]
    prompt_lengths = prompt_lengths[keep]
    generated_lengths = generated_lengths[keep]
    full_attention_mask = full_attention_mask[keep]
    active_indices = active_indices[keep]
    target_cache = _select_cache_batch(target_cache, keep)

    total_acc_length = 0
    total_decoded_token_num = 0
    adaptive_accept_ema = None
    total_draft_steps = 0
    total_draft_tokens_proposed = 0
    total_draft_tokens_accepted = 0
    total_cache_tokens_dropped = 0

    while current_token.numel() > 0:
        cur_bsz = int(current_token.shape[0])
        remaining = (max_length - prompt_lengths - generated_lengths).clamp_min(0)
        max_remaining = int(remaining.max().item()) if cur_bsz else 0
        if max_remaining <= 0:
            break

        draft_steps = min(
            _adaptive_draft_steps(
                max_draft_tokens=config.max_draft_tokens,
                active_count=cur_bsz,
                initial_batch_size=initial_bsz,
                accept_ema=adaptive_accept_ema,
            ),
            max_remaining,
        )

        draft_tokens = []
        draft_probs_per_step = []
        draft_valid_masks = []
        draft_cache = _clone_cache(target_cache, model)
        draft_attention_mask = full_attention_mask
        draft_active = remaining > 0
        draft_input = current_token.unsqueeze(1)

        for step_idx in range(draft_steps):
            valid_for_step = draft_active & (remaining > step_idx)
            if not bool(valid_for_step.any().item()):
                break

            query_mask = valid_for_step.long().unsqueeze(1)
            draft_attention_mask = torch.cat([draft_attention_mask, query_mask], dim=1)
            draft_position_ids = _position_ids_from_attention_mask(draft_attention_mask, 1)

            if statistical_time:
                _sync(device)
                t0 = time.time()
            draft_logits, draft_cache = _forward_logits_batched(
                model,
                input_ids=draft_input,
                skip_layers=config.skip_layers,
                past_key_values=draft_cache,
                full_attention_mask=draft_attention_mask,
                position_ids=draft_position_ids,
                use_cache=True,
            )
            if statistical_time:
                _sync(device)
                draft_time += time.time() - t0

            probs = _distribution_batched(draft_logits[:, -1, :], config.temperature, config.top_p)
            sampled = _sample_from_probs(probs, config.do_sample)
            sampled = torch.where(valid_for_step, sampled, torch.full_like(sampled, eos_token_id))

            draft_tokens.append(sampled)
            draft_probs_per_step.append(probs)
            draft_valid_masks.append(valid_for_step)

            confidence = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)
            draft_active = valid_for_step & (confidence >= config.confidence_threshold) & (sampled != eos_token_id)
            draft_input = sampled.unsqueeze(1)

        actual_draft_steps = len(draft_tokens)
        total_draft_steps += actual_draft_steps

        if actual_draft_steps > 0:
            draft_token_matrix = torch.stack(draft_tokens, dim=1)  # [B, D]
            draft_valid_matrix = torch.stack(draft_valid_masks, dim=1)  # [B, D]
        else:
            draft_token_matrix = torch.empty((cur_bsz, 0), dtype=torch.long, device=device)
            draft_valid_matrix = torch.empty((cur_bsz, 0), dtype=torch.bool, device=device)

        total_draft_tokens_proposed += int(draft_valid_matrix.sum().item())

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

        if statistical_time:
            _sync(device)
            t0 = time.time()
        verify_logits, target_cache = _target_forward_logits_batched(
            model,
            input_ids=verify_input,
            past_key_values=target_cache,
            attention_mask=verify_attention_mask,
            position_ids=verify_position_ids,
            logits_to_keep=0,
        )
        if statistical_time:
            _sync(device)
            verify_time += time.time() - t0

        # Vectorize all full-vocab probability work. The previous version called
        # top-p/softmax once per row and per draft position in Python.
        q_len = int(verify_logits.shape[1])
        vocab_size = int(verify_logits.shape[-1])
        verify_probs = _distribution_batched(
            verify_logits.reshape(cur_bsz * q_len, vocab_size),
            config.temperature,
            config.top_p,
        ).reshape(cur_bsz, q_len, vocab_size)
        sampled_target_tokens = _sample_from_probs(verify_probs.reshape(cur_bsz * q_len, vocab_size), config.do_sample).reshape(cur_bsz, q_len)

        if actual_draft_steps > 0:
            draft_probs_stack = torch.stack(draft_probs_per_step, dim=1)  # [B, D, V]
            draft_targets = verify_probs[:, :actual_draft_steps, :]
            token_idx = draft_token_matrix.unsqueeze(-1)

            if config.do_sample:
                target_prob = torch.gather(draft_targets, dim=-1, index=token_idx).squeeze(-1).clamp_min(0.0)
                draft_prob = torch.gather(draft_probs_stack, dim=-1, index=token_idx).squeeze(-1).clamp_min(1e-12)
                accept_probs = torch.clamp(target_prob / draft_prob, max=1.0)
                accepted_matrix = torch.rand_like(accept_probs) <= accept_probs
                residual_samples = _sample_residual_batched(draft_targets, draft_probs_stack)
                replacement_matrix = torch.where(accepted_matrix, draft_token_matrix, residual_samples)
            else:
                target_argmax = sampled_target_tokens[:, :actual_draft_steps]
                accepted_matrix = draft_token_matrix == target_argmax
                replacement_matrix = target_argmax

            accepted_matrix = accepted_matrix & draft_valid_matrix
        else:
            accepted_matrix = torch.empty((cur_bsz, 0), dtype=torch.bool, device=device)
            replacement_matrix = torch.empty((cur_bsz, 0), dtype=torch.long, device=device)
            draft_probs_stack = None

        next_current = current_token.clone()
        finished_rows = torch.zeros(cur_bsz, dtype=torch.bool, device=device)
        cache_lengths = [1] * cur_bsz  # current_token is always cached by verify forward.
        accepted_lengths_for_metric = []

        # This loop now only handles prefix logic over at most a few draft tokens;
        # it performs no full-vocab operations.
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
                # Defensive fallback for degenerate settings.
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
            total_draft_tokens_accepted += accepted_draft_prefix

            if accepted_ids[-1] == eos_token_id or prompt_lengths[row] + generated_lengths[row] >= max_length:
                finished_rows[row] = True

        max_cache_extension = max(cache_lengths) if cache_lengths else 1
        target_cache = _crop_cache(target_cache, cache_past_len + max_cache_extension)
        cache_valid_extension = torch.zeros((cur_bsz, max_cache_extension), dtype=torch.long, device=device)
        for row, cache_len in enumerate(cache_lengths):
            cache_valid_extension[row, :cache_len] = 1
        total_cache_tokens_dropped += max(0, verify_input.shape[1] - max_cache_extension)
        full_attention_mask = torch.cat([full_attention_mask, cache_valid_extension], dim=1)

        if accepted_lengths_for_metric:
            current_step_accept = sum(accepted_lengths_for_metric) / len(accepted_lengths_for_metric)
            adaptive_accept_ema = (
                current_step_accept
                if adaptive_accept_ema is None
                else 0.85 * adaptive_accept_ema + 0.15 * current_step_accept
            )

        keep = (~finished_rows).nonzero(as_tuple=False).flatten()
        if keep.numel() == 0:
            break

        current_token = next_current[keep]
        prompt_lengths = prompt_lengths[keep]
        generated_lengths = generated_lengths[keep]
        full_attention_mask = full_attention_mask[keep]
        active_indices = active_indices[keep]
        target_cache = _select_cache_batch(target_cache, keep)

        del verify_logits, verify_probs, sampled_target_tokens, verify_input, verify_attention_mask, verify_position_ids
        del draft_token_matrix, draft_valid_matrix, draft_tokens, draft_probs_per_step, draft_valid_masks
        if actual_draft_steps > 0:
            del draft_probs_stack, accepted_matrix, replacement_matrix

    return _final_result(
        generated,
        start_time,
        total_acc_length=total_acc_length,
        total_decoded_token_num=total_decoded_token_num,
        draft_time=draft_time,
        verify_time=verify_time,
        prefill_time=prefill_time,
        total_draft_steps=total_draft_steps,
        total_draft_tokens_proposed=total_draft_tokens_proposed,
        total_draft_tokens_accepted=total_draft_tokens_accepted,
        total_cache_tokens_dropped=total_cache_tokens_dropped,
    )


def _final_result(
    generated,
    start_time,
    *,
    total_acc_length,
    total_decoded_token_num,
    draft_time,
    verify_time,
    prefill_time,
    total_draft_steps,
    total_draft_tokens_proposed,
    total_draft_tokens_accepted,
    total_cache_tokens_dropped,
):
    total_time = time.time() - start_time
    generated_token_ids = [seq for seq in generated]
    max_sequence_length = max((len(item) for item in generated_token_ids), default=0)
    return {
        "generated_token_ids": generated_token_ids,
        "max_sequence_length": max_sequence_length,
        "total_acc_length": total_acc_length,
        "total_decoded_token_num": total_decoded_token_num,
        "total_time_cost": total_time,
        "target_time_cost": verify_time,
        "draft_time_cost": draft_time,
        "check_time_cost": 0.0,
        "prefill_time_cost": prefill_time,
        "post_time_cost": 0.0,
        "average_accept_length": total_acc_length / total_decoded_token_num if total_decoded_token_num else 0.0,
        "average_draft_steps": total_draft_steps / total_decoded_token_num if total_decoded_token_num else 0.0,
        "total_draft_tokens_proposed": total_draft_tokens_proposed,
        "total_draft_tokens_accepted": total_draft_tokens_accepted,
        "draft_accept_rate": total_draft_tokens_accepted / total_draft_tokens_proposed if total_draft_tokens_proposed else 0.0,
        "cache_tokens_dropped": total_cache_tokens_dropped,
    }


def _sample_residual_batched(target_probs, draft_probs):
    """Sample from max(target_probs - draft_probs, 0) for a [B,D,V] batch."""
    bsz, steps, vocab = target_probs.shape
    flat_target = target_probs.reshape(bsz * steps, vocab)
    flat_draft = draft_probs.reshape(bsz * steps, vocab)
    residual = torch.clamp(flat_target - flat_draft, min=0.0)
    totals = residual.sum(dim=-1, keepdim=True)
    fallback = totals.squeeze(-1) <= 0
    if bool(fallback.any().item()):
        residual[fallback] = flat_target[fallback]
        totals = residual.sum(dim=-1, keepdim=True)
    return torch.multinomial(residual / totals.clamp_min(1e-12), num_samples=1).view(bsz, steps)


def _select_cache_batch(cache, indices):
    """Select live batch rows from a DynamicCache or legacy key/value cache."""
    if isinstance(indices, torch.Tensor) and indices.dtype == torch.bool:
        indices = indices.nonzero(as_tuple=False).flatten()
    indices = indices.to(dtype=torch.long)
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            layer.keys = layer.keys.index_select(0, indices).contiguous()
            layer.values = layer.values.index_select(0, indices).contiguous()
        return cache
    if hasattr(cache, "key_cache"):
        cache.key_cache = [k.index_select(0, indices).contiguous() for k in cache.key_cache]
        cache.value_cache = [v.index_select(0, indices).contiguous() for v in cache.value_cache]
        return cache
    raise TypeError(f"Unsupported cache type for batch select: {type(cache)}")

def _adaptive_draft_steps(max_draft_tokens, active_count, initial_batch_size, accept_ema):
    """Choose draft depth for batched self-speculative decoding.

    Self-spec drafting is not a separate 1-layer model; it is the same model
    with skipped layers, so draft overhead is high at large effective batch size.
    This heuristic keeps draft depth shallow while concurrency is high, then
    allows deeper drafting as active concurrency falls or observed acceptance
    improves. It preserves the user-specified max_draft_tokens as a hard cap.
    """
    max_draft_tokens = int(max_draft_tokens)
    if max_draft_tokens <= 0:
        return 0

    active_count = max(1, int(active_count))
    initial_batch_size = max(1, int(initial_batch_size))
    concurrency_ratio = active_count / initial_batch_size

    # Concurrency-aware cap: high concurrency is compute-bound, so keep draft
    # depth shallow. As sequences finish, the cap gradually relaxes.
    if concurrency_ratio >= 0.75:
        concurrency_cap = 2
    elif concurrency_ratio >= 0.50:
        concurrency_cap = 3
    elif concurrency_ratio >= 0.25:
        concurrency_cap = 4
    else:
        concurrency_cap = max_draft_tokens

    # Acceptance-aware cap: if accepted length is low, drafting far beyond the
    # observed accepted prefix wastes self-spec compute. Add one token of slack
    # so the algorithm can still discover improved acceptance.
    if accept_ema is None:
        acceptance_cap = max_draft_tokens
    else:
        acceptance_cap = max(1, int(round(float(accept_ema) + 1.0)))

    return max(1, min(max_draft_tokens, concurrency_cap, acceptance_cap))


def _crop_cache(cache, max_length):
    """Crop a DynamicCache or legacy key/value cache in-place and return it."""
    max_length = int(max_length)
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if not getattr(layer, "is_initialized", False):
                continue
            layer.keys = layer.keys[:, :, :max_length, :].contiguous()
            layer.values = layer.values[:, :, :max_length, :].contiguous()
        return cache
    if hasattr(cache, "key_cache"):
        cache.key_cache = [k[:, :, :max_length, :].contiguous() for k in cache.key_cache]
        cache.value_cache = [v[:, :, :max_length, :].contiguous() for v in cache.value_cache]
        return cache
    raise TypeError(f"Unsupported cache type for crop: {type(cache)}")


def _empty_result(start_time):
    return {
        "generated_token_ids": [],
        "max_sequence_length": 0,
        "total_acc_length": 0,
        "total_decoded_token_num": 0,
        "total_time_cost": time.time() - start_time,
        "target_time_cost": 0.0,
        "draft_time_cost": 0.0,
        "check_time_cost": 0.0,
        "prefill_time_cost": 0.0,
        "post_time_cost": 0.0,
        "average_accept_length": 0.0,
        "average_draft_steps": 0.0,
        "cache_tokens_dropped": 0,
    }


def _sample_residual(target_probs, draft_probs):
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    total = residual.sum()
    if total <= 0:
        residual = target_probs
        total = residual.sum()
    return int(torch.multinomial(residual / total.clamp_min(1e-12), 1).item())


def _sample_from_probs(probs, do_sample):
    if not do_sample:
        return torch.argmax(probs, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _distribution_1d(logits, temperature, top_p):
    logits = logits.float()
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask = torch.roll(sorted_mask, shifts=1, dims=-1)
        sorted_mask[0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return probs


def _distribution_batched(logits, temperature, top_p):
    logits = logits.float()
    if temperature and temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask = torch.roll(sorted_mask, shifts=1, dims=-1)
        sorted_mask[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return probs


def _forward_logits_batched(
    model,
    input_ids,
    skip_layers,
    past_key_values,
    full_attention_mask,
    position_ids,
    use_cache=True,
):
    device = _model_device(model)
    base_model = _unwrap_causal_lm(model)
    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(1)

    hidden_states = base_model.model.embed_tokens(input_ids)
    q_len = input_ids.shape[-1]
    past_length = full_attention_mask.shape[1] - q_len
    cache_position = torch.arange(past_length, past_length + q_len, dtype=torch.long, device=device)
    attention_mask_4d = _attention_mask_4d_from_2d(full_attention_mask, q_len, hidden_states.dtype, device)
    position_embeddings = base_model.model.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(base_model.model.layers):
        if layer_idx in skip_layers:
            continue
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

    hidden_states = base_model.model.norm(hidden_states)
    logits = base_model.lm_head(hidden_states)
    return logits, past_key_values


def _target_forward_logits_batched(
    model,
    input_ids,
    past_key_values,
    attention_mask,
    position_ids,
    logits_to_keep=0,
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
    )
    return outputs.logits, outputs.past_key_values


def _attention_mask_4d_from_2d(full_attention_mask, query_len, dtype, device):
    # full_attention_mask shape: [B, past_len + query_len]
    bsz, key_len = full_attention_mask.shape
    min_dtype = torch.finfo(dtype).min
    key_positions = torch.arange(key_len, device=device).view(1, 1, 1, key_len)
    query_positions = torch.arange(key_len - query_len, key_len, device=device).view(1, 1, query_len, 1)
    causal = key_positions <= query_positions
    valid_keys = full_attention_mask.to(device=device).bool().view(bsz, 1, 1, key_len)
    allowed = causal & valid_keys
    return torch.where(allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), min_dtype, dtype=dtype, device=device))


def _position_ids_from_attention_mask(attention_mask, query_len):
    # Left-padding and masked speculative positions are handled by cumulative
    # valid-token positions. Query positions with mask=0 receive position 0 and
    # are ignored by future attention masks.
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask.long() == 0, 0)
    return position_ids[:, -query_len:].long()


def _new_cache(model):
    base_model = _unwrap_causal_lm(model)
    return DynamicCache(config=base_model.config)


def _clone_cache(cache, model):
    base_model = _unwrap_causal_lm(model)
    cloned = DynamicCache(config=base_model.config)
    if hasattr(cache, "layers"):
        for src_layer, dst_layer in zip(cache.layers, cloned.layers):
            if not getattr(src_layer, "is_initialized", False):
                continue
            dst_layer.keys = src_layer.keys
            dst_layer.values = src_layer.values
            dst_layer.dtype = src_layer.dtype
            dst_layer.device = src_layer.device
            dst_layer.is_initialized = True
    elif hasattr(cache, "key_cache"):
        cloned.key_cache = list(cache.key_cache)
        cloned.value_cache = list(cache.value_cache)
    else:
        raise TypeError(f"Unsupported cache type: {type(cache)}")
    return cloned


def _sync(device):
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _unwrap_causal_lm(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _model_device(model):
    return next(model.parameters()).device
