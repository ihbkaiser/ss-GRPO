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
    config = SelfSpeculativeConfig(
        skip_layers=parse_skip_layers(skip_layers),
        max_draft_tokens=max_draft_tokens,
        confidence_threshold=confidence_threshold,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    repeats = repeated_generate_nums or 1
    device = _model_device(model)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    eos_token_id = tokenizer.eos_token_id

    start_time = time.time()
    draft_time = 0.0
    verify_time = 0.0
    prefill_time = 0.0
    generated_token_ids = []
    total_acc_length = 0
    total_decoded_token_num = 0

    for batch_idx in range(input_ids.shape[0]):
        prompt = input_ids[batch_idx][attention_mask[batch_idx].bool()].tolist()
        for _ in range(repeats):
            result = _generate_one(
                model,
                prompt,
                eos_token_id,
                max_length,
                config,
                statistical_time=statistical_time,
            )
            generated_token_ids.append(result["generated_tokens"])
            total_acc_length += result["total_acc_length"]
            total_decoded_token_num += result["verification_steps"]
            draft_time += result["draft_time"]
            verify_time += result["verify_time"]
            prefill_time += result["prefill_time"]

    max_sequence_length = max((len(item) for item in generated_token_ids), default=0)
    total_time = time.time() - start_time
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
    }


def _generate_one(model, prompt, eos_token_id, max_length, config, *, statistical_time):
    device = _model_device(model)
    generated_tokens = []
    total_acc_length = 0
    verification_steps = 0
    draft_time = 0.0
    verify_time = 0.0
    prefill_time = 0.0

    if len(prompt) >= max_length:
        return {
            "generated_tokens": generated_tokens,
            "total_acc_length": total_acc_length,
            "verification_steps": verification_steps,
            "draft_time": draft_time,
            "verify_time": verify_time,
            "prefill_time": prefill_time,
        }

    target_cache = _new_cache(model)
    if statistical_time:
        _sync(device)
        start = time.time()
    prefill_logits, target_cache = _target_forward_logits(
        model,
        prompt,
        past_key_values=target_cache,
        logits_to_keep=1,
    )
    if statistical_time:
        _sync(device)
        prefill_time += time.time() - start

    first_probs = _distribution(prefill_logits[0, -1, :], config.temperature, config.top_p)
    current_token = torch.argmax(first_probs).item() if not config.do_sample else torch.multinomial(first_probs, 1).item()
    generated_tokens.append(current_token)
    target_cache_len = len(prompt)
    if current_token == eos_token_id or len(prompt) + len(generated_tokens) >= max_length:
        return {
            "generated_tokens": generated_tokens,
            "total_acc_length": total_acc_length,
            "verification_steps": verification_steps,
            "draft_time": draft_time,
            "verify_time": verify_time,
            "prefill_time": prefill_time,
        }

    while len(prompt) + len(generated_tokens) < max_length:
        remaining_tokens = max_length - (len(prompt) + len(generated_tokens))
        draft_tokens = []
        draft_distributions = []
        draft_cache = _clone_cache(target_cache, model)
        draft_cache_len = target_cache_len
        draft_input = [current_token]

        for _ in range(min(config.max_draft_tokens, remaining_tokens)):
            if statistical_time:
                _sync(device)
                start = time.time()
            draft_logits, draft_cache = _forward_logits(
                model,
                draft_input,
                config.skip_layers,
                past_key_values=draft_cache,
                past_length=draft_cache_len,
                use_cache=True,
            )
            draft_cache_len += len(draft_input)
            draft_probs = _distribution(draft_logits[0, -1, :], config.temperature, config.top_p)
            if statistical_time:
                _sync(device)
                draft_time += time.time() - start

            token = torch.argmax(draft_probs).item() if not config.do_sample else torch.multinomial(draft_probs, 1).item()
            draft_tokens.append(token)
            draft_distributions.append(draft_probs)
            draft_input = [token]

            confidence = draft_probs[token].item()
            if confidence < config.confidence_threshold or token == eos_token_id:
                break

        verify_input = [current_token] + draft_tokens

        if statistical_time:
            _sync(device)
            start = time.time()
        verify_logits, target_cache = _target_forward_logits(
            model,
            verify_input,
            past_key_values=target_cache,
            logits_to_keep=0,
        )
        if statistical_time:
            _sync(device)
            verify_time += time.time() - start

        verification_steps += 1
        accepted_ids = []
        all_accepted = True

        for idx, token in enumerate(draft_tokens):
            target_logits = verify_logits[0, idx, :]
            target_probs = _distribution(target_logits, config.temperature, config.top_p)
            draft_probs = draft_distributions[idx]

            if not config.do_sample:
                target_token = torch.argmax(target_probs).item()
                accepted = token == target_token
                replacement = target_token
            else:
                target_prob = target_probs[token].clamp_min(0.0)
                draft_prob = draft_probs[token].clamp_min(1e-12)
                accepted = torch.rand((), device=target_probs.device).item() <= min(1.0, (target_prob / draft_prob).item())
                replacement = token if accepted else _sample_residual(target_probs, draft_probs)

            accepted_ids.append(replacement)

            if replacement == eos_token_id:
                accepted_ids = accepted_ids[:remaining_tokens]
                generated_tokens.extend(accepted_ids)
                target_cache_len += len(accepted_ids)
                target_cache.crop(target_cache_len)
                total_acc_length += len(accepted_ids)
                return {
                    "generated_tokens": generated_tokens,
                    "total_acc_length": total_acc_length,
                    "verification_steps": verification_steps,
                    "draft_time": draft_time,
                    "verify_time": verify_time,
                    "prefill_time": prefill_time,
                }

            if not accepted:
                all_accepted = False
                break

        if all_accepted:
            next_logits = verify_logits[0, len(draft_tokens), :]
            next_probs = _distribution(next_logits, config.temperature, config.top_p)
            next_token = torch.argmax(next_probs).item() if not config.do_sample else torch.multinomial(next_probs, 1).item()
            accepted_ids.append(next_token)

        accepted_ids = accepted_ids[:remaining_tokens]
        if not accepted_ids:
            break
        generated_tokens.extend(accepted_ids)
        current_token = accepted_ids[-1]
        target_cache_len += len(accepted_ids)
        target_cache.crop(target_cache_len)
        total_acc_length += len(accepted_ids)
        if current_token == eos_token_id:
            break

    return {
        "generated_tokens": generated_tokens,
        "total_acc_length": total_acc_length,
        "verification_steps": verification_steps,
        "draft_time": draft_time,
        "verify_time": verify_time,
        "prefill_time": prefill_time,
    }


def _single_target_token(model, sequence, config):
    logits = _forward_logits(model, sequence, frozenset())[0, -1, :]
    probs = _distribution(logits, config.temperature, config.top_p)
    token = torch.argmax(probs).item() if not config.do_sample else torch.multinomial(probs, 1).item()
    return [token], [probs]


def _sample_residual(target_probs, draft_probs):
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    total = residual.sum()
    if total <= 0:
        residual = target_probs
        total = residual.sum()
    return torch.multinomial(residual / total, 1).item()


def _distribution(logits, temperature, top_p):
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


def _forward_logits(model, token_ids, skip_layers, past_key_values=None, past_length=0, use_cache=False):
    device = _model_device(model)
    base_model = _unwrap_causal_lm(model)
    if isinstance(token_ids, torch.Tensor):
        input_ids = token_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    hidden_states = base_model.model.embed_tokens(input_ids)
    seq_len = input_ids.shape[-1]
    cache_position = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=device)
    position_ids = cache_position.unsqueeze(0)
    attention_mask = _causal_mask(seq_len, past_length + seq_len, past_length, hidden_states.dtype, device)
    position_embeddings = base_model.model.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(base_model.model.layers):
        if layer_idx in skip_layers:
            continue
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, (tuple, list)) else layer_outputs

    hidden_states = base_model.model.norm(hidden_states)
    logits = base_model.lm_head(hidden_states)
    if use_cache:
        return logits, past_key_values
    return logits


def _target_forward_logits(model, token_ids, past_key_values, logits_to_keep=0):
    device = _model_device(model)
    if isinstance(token_ids, torch.Tensor):
        input_ids = token_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
        logits_to_keep=logits_to_keep,
    )
    return outputs.logits, outputs.past_key_values


def _causal_mask(query_len, key_len, past_length, dtype, device):
    if query_len == 1:
        return None
    min_dtype = torch.finfo(dtype).min
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    query_positions = (past_length + torch.arange(query_len, device=device)).unsqueeze(1)
    mask = torch.where(key_positions <= query_positions, 0.0, min_dtype).to(dtype)
    return mask.unsqueeze(0).unsqueeze(0)


def _new_cache(model):
    base_model = _unwrap_causal_lm(model)
    return DynamicCache(config=base_model.config)


def _clone_cache(cache, model):
    base_model = _unwrap_causal_lm(model)
    cloned = DynamicCache(config=base_model.config)
    for src_layer, dst_layer in zip(cache.layers, cloned.layers):
        if not getattr(src_layer, "is_initialized", False):
            continue
        dst_layer.keys = src_layer.keys
        dst_layer.values = src_layer.values
        dst_layer.dtype = src_layer.dtype
        dst_layer.device = src_layer.device
        dst_layer.is_initialized = True
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
