import argparse
import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from helper.rewards import accuracy_reward_func, format_reward_func
from self_speculative_claps.generate_cl import self_speculative_generate_clasp as self_speculative_generate


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}m"
    return f"{seconds / 3600:.2f}h"


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-speculative GRPO using Draft & Verify layer skipping.")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--version_name", default="self_spec_grpo")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--target_lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_training_token", type=int, default=1024)
    parser.add_argument("--max_training_padding_gap", type=int, default=128)
    parser.add_argument("--logps_chunk_size", type=int, default=256, help="Sequence chunk size for token-logprob computation. Lower values reduce peak VRAM.")
    parser.add_argument("--statistical_time", type=str2bool, default=False, help="Enable exact CUDA timing. False avoids frequent cuda synchronize calls.")
    parser.add_argument("--sample_num", type=int, default=100, help="Logging window only; kept for CLI parity with FastGRPO.")
    parser.add_argument("--repeated_generate_nums", type=int, default=8)
    parser.add_argument("--skip_layers", default="", help="Deprecated fallback static path. Empty means CLaSp/heuristic initializer controls skip paths.")
    parser.add_argument("--max_draft_tokens", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.20)
    parser.add_argument("--verification_capacity", type=int, default=160, help="FastGRPO-style Cpeak/Nverify budget. Higher means deeper self-spec drafting when active batch is small.")
    parser.add_argument("--max_verification_num", type=int, default=160)
    parser.add_argument("--min_draft_tokens", "--min_draft_token_length", dest="min_draft_tokens", type=int, default=1)
    parser.add_argument("--draft_token_length_c", type=float, default=0.75)
    parser.add_argument("--dynamic_confidence_threshold", type=str2bool, default=True, help="Update confidence threshold from observed draft acceptance rate.")
    parser.add_argument("--target_accept_rate", type=float, default=0.80)
    parser.add_argument("--threshold_lr", type=float, default=0.05)
    parser.add_argument("--min_confidence_threshold", type=float, default=0.15)
    parser.add_argument("--max_confidence_threshold", type=float, default=0.95)
    parser.add_argument("--threshold_ema_beta", type=float, default=0.90)
    # Batch-aware CLaSp routing. CLaSp is used as an in-context skip-path proposer,
    # then quantized to a small codebook to avoid per-rollout path explosion.
    parser.add_argument("--enable_clasp", type=str2bool, default=True)
    parser.add_argument("--clasp_codebook_size", type=int, default=8)
    parser.add_argument("--clasp_max_active_paths", type=int, default=4)
    parser.add_argument("--clasp_min_group_size", type=int, default=24)
    parser.add_argument("--clasp_update_interval", type=int, default=4)
    parser.add_argument("--clasp_low_accept_trigger", type=float, default=0.62)
    parser.add_argument("--clasp_protected_first", type=int, default=4)
    parser.add_argument("--clasp_protected_last", type=int, default=4)
    parser.add_argument("--clasp_candidate_layers", default="", help="Comma/range list for CLaSp candidate layers, e.g. '8-27'. Empty means all non-protected layers.")
    parser.add_argument("--clasp_representative_rows", type=int, default=0, help="0 uses all rows. Set 8/16 to reduce CLaSp hidden-state scoring cost.")
    parser.add_argument("--clasp_dynamic_codebook", type=str2bool, default=True)
    parser.add_argument("--clasp_min_code_frequency", type=int, default=2)
    parser.add_argument("--clasp_disable_when_bsz_below", type=int, default=16)
    parser.add_argument("--clasp_skip_count", type=int, default=0, help="Exact number of transformer blocks to skip. 0 means infer from --clasp_skip_ratio.")
    parser.add_argument("--clasp_skip_ratio", type=float, default=0.60, help="When --clasp_skip_count=0, skip about this fraction of total transformer layers. Default: 0.60.")
    parser.add_argument("--clasp_prefill_update", type=str2bool, default=False, help="If true, run first CLaSp update from prefill hidden states. Faster to leave False for long prompts; first update then happens after round-1 verify.")
    # Deprecated Bayesian-search arguments kept for CLI compatibility only. They are intentionally ignored.
    parser.add_argument("--auto_search_skip_layers", "--auto_search_skip_path", dest="auto_search_skip_path", type=str2bool, default=False, help="Deprecated and ignored in CLaSp-no-Bayes version.")
    parser.add_argument("--search_candidate_layers", default="", help=argparse.SUPPRESS)
    parser.add_argument("--search_num_prompts", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_max_length", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_min_skip", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_max_skip", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--target_draft_layers", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--draft_layer_tolerance", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_init_trials", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_bo_trials", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_candidate_pool", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--search_seed", type=int, default=13, help=argparse.SUPPRESS)
    parser.add_argument("--search_json_out", default="", help=argparse.SUPPRESS)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--saved_model_dir", required=True)
    return parser.parse_args()


class TrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        messages = []
        answers = []
        system_prompt = "You are a math problem assistant."
        user_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}"""
        for example in batch:
            messages.append(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt.format_map({"instruction": example["question"]})},
                ]
            )
            answers.append(example["answer"])
        tokenized = self.tokenizer(
            text=self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=4096,
            padding_side="left",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "messages": messages,
            "answers": answers,
        }


def _get_base_causal_lm(causal_lm):
    """Return the underlying AutoModelForCausalLM while preserving PEFT/LoRA modules."""
    if hasattr(causal_lm, "get_base_model"):
        return causal_lm.get_base_model()
    if hasattr(causal_lm, "base_model") and hasattr(causal_lm.base_model, "model"):
        return causal_lm.base_model.model
    return causal_lm


def _autocast_dtype(model):
    dtype = getattr(model, "dtype", None)
    return torch.bfloat16 if dtype == torch.bfloat16 else torch.float16


def _token_logps_from_hidden(hidden_states, lm_head, labels, chunk_size):
    """Compute log p(labels[t+1] | tokens[:t]) without materializing full [B,T,V] logits."""
    hidden_states = hidden_states[:, :-1, :]
    labels = labels[:, 1:].to(hidden_states.device)
    seq_len = hidden_states.shape[1]
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = lm_head(hidden_states[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        selected = torch.gather(logits, dim=-1, index=cur_labels.unsqueeze(-1)).squeeze(-1)
        chunks.append(selected - torch.logsumexp(logits, dim=-1))
        del logits, selected
    return torch.cat(chunks, dim=1) if chunks else hidden_states.new_zeros((hidden_states.shape[0], 0))


def compute_model_token_logps(causal_lm, input_ids, attention_mask, chunk_size):
    """Forward backbone once, then apply LM head in chunks for memory-safe ref/old logps."""
    base_model = _get_base_causal_lm(causal_lm)
    device = input_ids.device
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(device_type, dtype=_autocast_dtype(base_model), enabled=(device.type == "cuda")):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    return _token_logps_from_hidden(hidden_states, base_model.lm_head, input_ids, chunk_size)


def compute_target_loss_and_backward(
    target_model,
    input_ids,
    attention_mask,
    mask,
    reward,
    epsilon,
    beta,
    chunk_size=256,
    loss_scale=1.0,
):
    """Memory-safe one-iteration GRPO loss for self-speculative GRPO.

    The original implementation built full `logits` and `ref_logits` and then
    called `log_softmax(-1)` on [batch, seq, vocab]. This is the same OOM
    source that was fixed in FastGRPO. Here only selected token log-probs are
    materialized, and policy LM-head/backward is chunked along sequence length.
    """
    device = input_ids.device
    token_mask = mask[:, :-1].to(device=device, dtype=torch.float32)
    denom = token_mask.sum(-1).clamp_min(1.0)
    reward = reward.to(device=device, dtype=torch.float32)
    seq_len = token_mask.shape[1]

    target_model.disable_adapter_layers()
    with torch.no_grad():
        ref_logps = compute_model_token_logps(target_model, input_ids, attention_mask, chunk_size).detach()
    target_model.enable_adapter_layers()

    base_model = _get_base_causal_lm(target_model)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(device_type, dtype=_autocast_dtype(base_model), enabled=(device.type == "cuda")):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    policy_hidden = hidden_states[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    loss_value = 0.0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = base_model.lm_head(policy_hidden[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        logps = torch.gather(logits, dim=-1, index=cur_labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        old_logps = logps.detach()
        cur_ref_logps = ref_logps[:, start:end]
        cur_mask = token_mask[:, start:end]

        coef1 = torch.exp(logps - old_logps)
        coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
        loss1 = torch.min(coef1 * reward, coef2 * reward)
        coef3 = cur_ref_logps - logps
        kl = torch.exp(coef3) - coef3 - 1
        token_loss = -(loss1 - beta * kl)
        chunk_loss = ((token_loss * cur_mask).sum(-1) / denom).sum()
        (chunk_loss * loss_scale).backward(retain_graph=(end < seq_len))
        loss_value += float(chunk_loss.detach().cpu())
        del logits, logps, old_logps, cur_ref_logps, coef1, coef2, loss1, coef3, kl, token_loss, chunk_loss

    del hidden_states, policy_hidden, ref_logps
    return loss_value


def build_training_batch(tokenizer, messages, rewards):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(text, padding=False)
    loss_mask = []
    for idx, message in enumerate(messages):
        prompt_text = tokenizer.apply_chat_template(message[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text)
        cur_mask = [0] * (len(prompt_ids) - 1) + [1] * (len(tokenized.input_ids[idx]) - len(prompt_ids) + 1)
        loss_mask.append(cur_mask)

    sorted_pairs = sorted(
        zip(tokenized.input_ids, tokenized.attention_mask, loss_mask, rewards),
        key=lambda x: len(x[0]),
    )
    max_len = max(len(item[0]) for item in sorted_pairs)
    input_ids, attention_mask, masks, sorted_rewards = [], [], [], []
    for ids, attn, mask, reward in sorted_pairs:
        pad = max_len - len(ids)
        input_ids.append(ids + [0] * pad)
        attention_mask.append(attn + [0] * pad)
        masks.append(mask + [0] * pad)
        sorted_rewards.append(reward)
    return (
        torch.tensor(input_ids, device="cuda"),
        torch.tensor(attention_mask, device="cuda"),
        torch.tensor(masks, device="cuda"),
        torch.tensor(sorted_rewards, device="cuda").unsqueeze(-1),
    )


def run_auto_skip_search(args, config, target_model, tokenizer, qas):
    search_start_time = time.time()
    rng = random.Random(args.search_seed)
    np.random.seed(args.search_seed)
    torch.manual_seed(args.search_seed)

    # If target_draft_layers is requested, search over all transformer layers by
    # default and convert the desired active draft size into a skip-count range.
    # Example for a 32-layer target with target=3,tolerance=1: active layers are
    # 3-4, so skip count is 28-29.
    if args.target_draft_layers and args.target_draft_layers > 0 and not args.search_candidate_layers:
        candidate_layers = list(range(config.num_hidden_layers))
    else:
        candidate_layers = parse_layer_set(args.search_candidate_layers, config.num_hidden_layers)
    if not candidate_layers:
        raise ValueError("No candidate layers to search.")

    if args.target_draft_layers and args.target_draft_layers > 0:
        target_active = max(1, int(args.target_draft_layers))
        tolerance = max(0, int(args.draft_layer_tolerance))
        min_active = target_active
        max_active = min(config.num_hidden_layers, target_active + tolerance)
        desired_min_skip = max(0, config.num_hidden_layers - max_active)
        desired_max_skip = max(0, config.num_hidden_layers - min_active)
        max_skip = min(desired_max_skip, len(candidate_layers))
        min_skip = min(desired_min_skip, max_skip)
    else:
        max_skip = min(args.search_max_skip, len(candidate_layers))
        min_skip = min(args.search_min_skip, max_skip)
    search_args = argparse.Namespace(
        max_draft_tokens=args.max_draft_tokens,
        confidence_threshold=args.confidence_threshold,
        max_length=args.search_max_length or args.max_length,
        verification_capacity=args.verification_capacity,
        max_verification_num=args.max_verification_num,
        min_draft_tokens=args.min_draft_tokens,
        draft_token_length_c=args.draft_token_length_c,
        dynamic_confidence_threshold=args.dynamic_confidence_threshold,
        target_accept_rate=args.target_accept_rate,
        threshold_lr=args.threshold_lr,
        min_confidence_threshold=args.min_confidence_threshold,
        max_confidence_threshold=args.max_confidence_threshold,
        threshold_ema_beta=args.threshold_ema_beta,
    )
    prompts = [build_search_prompt(tokenizer, item["question"]) for item in qas[: max(1, args.search_num_prompts)]]

    observed = []
    results = []
    initial = propose_initial_masks(candidate_layers, min_skip, max_skip, max(1, args.search_init_trials), rng)
    for trial_idx, mask in enumerate(initial):
        result = evaluate_skip_mask(target_model, tokenizer, prompts, mask, candidate_layers, search_args)
        observed.append(mask)
        results.append(result)
        print(json.dumps({"auto_skip_search_trial": trial_idx, **result}), flush=True)

    for _ in range(max(0, args.search_bo_trials)):
        best_idx = int(np.argmax([item["score"] for item in results]))
        pool = generate_candidate_pool(
            observed,
            observed[best_idx],
            len(candidate_layers),
            min_skip,
            max_skip,
            max(1, args.search_candidate_pool),
            rng,
        )
        if not pool:
            break
        x_obs = np.stack(observed).astype(np.float64)
        y_obs = np.array([item["score"] for item in results], dtype=np.float64)
        x_pool = np.stack(pool).astype(np.float64)
        acquisition = expected_improvement(x_obs, y_obs, x_pool)
        mask = pool[int(np.argmax(acquisition))]
        result = evaluate_skip_mask(target_model, tokenizer, prompts, mask, candidate_layers, search_args)
        observed.append(mask)
        results.append(result)
        print(json.dumps({"auto_skip_search_trial": len(results) - 1, **result}), flush=True)

    best_idx = int(np.argmax([item["score"] for item in results]))
    search_time_cost = time.time() - search_start_time
    summary = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "candidate_layers": candidate_layers,
        "target_draft_layers": args.target_draft_layers,
        "draft_layer_tolerance": args.draft_layer_tolerance,
        "effective_search_min_skip": min_skip,
        "effective_search_max_skip": max_skip,
        "effective_active_layers_min": config.num_hidden_layers - max_skip,
        "effective_active_layers_max": config.num_hidden_layers - min_skip,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "verification_capacity": args.verification_capacity,
        "max_verification_num": args.max_verification_num,
        "min_draft_tokens": args.min_draft_tokens,
        "draft_token_length_c": args.draft_token_length_c,
        "dynamic_confidence_threshold": args.dynamic_confidence_threshold,
        "target_accept_rate": args.target_accept_rate,
        "num_prompts": len(prompts),
        "max_length": search_args.max_length,
        "objective": "maximize self-speculative generated_tokens / wall_time_seconds before GRPO training",
        "search_time_cost": search_time_cost,
        "search_time_min": search_time_cost / 60.0,
        "best": results[best_idx],
        "trials": results,
    }
    if args.search_json_out:
        Path(args.search_json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.search_json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({
        "auto_skip_search_summary": True,
        "search_time_cost": search_time_cost,
        "search_time_min": search_time_cost / 60.0,
        "best_skip_layers": results[best_idx]["skip_layers"],
        "best_tokens_per_second": results[best_idx].get("tokens_per_second", 0.0),
        "trials": len(results),
    }), flush=True)
    torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()
    os.makedirs(args.saved_model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto", config=config).cuda().eval()
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    target_model = get_peft_model(target_model, lora_config)
    optimizer = torch.optim.AdamW(target_model.parameters(), lr=args.target_lr)

    qas = get_train_QAs(args.train_option)
    # CLaSp-no-Bayes: do not run static Bayesian skip-layer search.
    # The default path is deterministic middle-layer 60% skip, and CLaSp updates
    # routing online from full-verify hidden states. Deprecated search flags are
    # accepted only so old launch scripts do not break.
    if args.auto_search_skip_path:
        print("[CLaSp-no-Bayes] --auto_search_skip_path was passed but is ignored. Runtime CLaSp will choose skip paths online.", flush=True)
    search_summary = None
    search_time_cost = 0.0

    loader_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        qas,
        collate_fn=TrainDataCollator(tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        persistent_workers=loader_workers > 0,
        pin_memory=True,
    )

    print(datetime.now())
    print(f"AURORA-CLaSp-Q model={args.model_dir} skip_layers_fallback={args.skip_layers!r} clasp_skip_ratio={args.clasp_skip_ratio}")
    print(f"batch={args.batch_size} repeats={args.repeated_generate_nums} max_length={args.max_length}")
    with open(args.log_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "phase": "run_config",
            "time": datetime.now().isoformat(),
            "model_dir": args.model_dir,
            "train_option": args.train_option,
            "version_name": args.version_name,
            "batch_size": args.batch_size,
            "repeated_generate_nums": args.repeated_generate_nums,
            "max_length": args.max_length,
            "max_training_token": args.max_training_token,
            "max_training_padding_gap": args.max_training_padding_gap,
            "target_lr": args.target_lr,
            "beta": args.beta,
            "epsilon": args.epsilon,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "skip_layers_fallback": args.skip_layers,
            "auto_search_skip_path_ignored": args.auto_search_skip_path,
            "clasp_no_bayes": True,
            "clasp_skip_ratio": args.clasp_skip_ratio,
            "clasp_prefill_update": args.clasp_prefill_update,
            "max_draft_tokens": args.max_draft_tokens,
            "confidence_threshold": args.confidence_threshold,
            "verification_capacity": args.verification_capacity,
            "max_verification_num": args.max_verification_num,
            "min_draft_tokens": args.min_draft_tokens,
            "draft_token_length_c": args.draft_token_length_c,
            "dynamic_confidence_threshold": args.dynamic_confidence_threshold,
            "target_accept_rate": args.target_accept_rate,
            "threshold_lr": args.threshold_lr,
            "min_confidence_threshold": args.min_confidence_threshold,
            "max_confidence_threshold": args.max_confidence_threshold,
            "threshold_ema_beta": args.threshold_ema_beta,
            "enable_clasp": args.enable_clasp,
            "clasp_codebook_size": args.clasp_codebook_size,
            "clasp_max_active_paths": args.clasp_max_active_paths,
            "clasp_min_group_size": args.clasp_min_group_size,
            "clasp_update_interval": args.clasp_update_interval,
            "clasp_low_accept_trigger": args.clasp_low_accept_trigger,
            "clasp_protected_first": args.clasp_protected_first,
            "clasp_protected_last": args.clasp_protected_last,
            "clasp_candidate_layers": args.clasp_candidate_layers,
            "clasp_representative_rows": args.clasp_representative_rows,
            "clasp_dynamic_codebook": args.clasp_dynamic_codebook,
            "clasp_min_code_frequency": args.clasp_min_code_frequency,
            "clasp_disable_when_bsz_below": args.clasp_disable_when_bsz_below,
            "clasp_skip_count": args.clasp_skip_count,
            "clasp_skip_ratio": args.clasp_skip_ratio,
            "clasp_prefill_update": args.clasp_prefill_update,
        }) + "\n")
    # No offline skip-search phase in this version.


    used_items = 0
    used_items_at_last_update = 0
    start_time = time.time()
    pending_messages = []
    pending_rewards = []
    pending_std_rewards = []
    total_generate_time = 0.0
    total_train_time = 0.0
    optimizer.zero_grad(set_to_none=True)
    epoch_bar = tqdm(range(args.num_epochs), desc="Epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        skipped_long = 0
        skipped_none = 0
        batch_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        for batch_idx, batch in batch_bar:
            if batch["input_ids"].shape[-1] >= args.max_length:
                skipped_long += 1
                batch_bar.set_postfix(
                    phase="skip_max_length",
                    input_len=batch["input_ids"].shape[-1],
                    used=used_items,
                    refresh=False,
                )
                continue
            if None in batch["answers"]:
                skipped_none += 1
                batch_bar.set_postfix(phase="skip_none_answer", used=used_items, refresh=False)
                continue
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            generate_wall_start = time.time()
            with torch.inference_mode():
                outputs = self_speculative_generate(
                    target_model,
                    input_ids,
                    attention_mask,
                    tokenizer,
                    skip_layers=args.skip_layers,
                    max_draft_tokens=args.max_draft_tokens,
                    confidence_threshold=args.confidence_threshold,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repeated_generate_nums=args.repeated_generate_nums,
                    max_length=args.max_length,
                    statistical_time=args.statistical_time,
                    verification_capacity=args.verification_capacity,
                    max_verification_num=args.max_verification_num,
                    min_draft_tokens=args.min_draft_tokens,
                    draft_token_length_c=args.draft_token_length_c,
                    dynamic_confidence_threshold=args.dynamic_confidence_threshold,
                    target_accept_rate=args.target_accept_rate,
                    threshold_lr=args.threshold_lr,
                    min_confidence_threshold=args.min_confidence_threshold,
                    max_confidence_threshold=args.max_confidence_threshold,
                    threshold_ema_beta=args.threshold_ema_beta,
                    enable_clasp=args.enable_clasp,
                    clasp_codebook_size=args.clasp_codebook_size,
                    clasp_max_active_paths=args.clasp_max_active_paths,
                    clasp_min_group_size=args.clasp_min_group_size,
                    clasp_update_interval=args.clasp_update_interval,
                    clasp_low_accept_trigger=args.clasp_low_accept_trigger,
                    clasp_protected_first=args.clasp_protected_first,
                    clasp_protected_last=args.clasp_protected_last,
                    clasp_candidate_layers=args.clasp_candidate_layers,
                    clasp_representative_rows=args.clasp_representative_rows,
                    clasp_dynamic_codebook=args.clasp_dynamic_codebook,
                    clasp_min_code_frequency=args.clasp_min_code_frequency,
                    clasp_disable_when_bsz_below=args.clasp_disable_when_bsz_below,
                    clasp_skip_count=args.clasp_skip_count,
                    clasp_skip_ratio=args.clasp_skip_ratio,
                    clasp_prefill_update=args.clasp_prefill_update,
                )
            generate_wall_time = time.time() - generate_wall_start

            decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in outputs["generated_token_ids"]]
            batch_rewards = []
            for idx_batch, answer in enumerate(batch["answers"]):
                group_decoded = []
                cur_messages = []
                for idx_repeat in range(args.repeated_generate_nums):
                    seq_idx = idx_batch * args.repeated_generate_nums + idx_repeat
                    group_decoded.append(decoded[seq_idx])
                    message = deepcopy(batch["messages"][idx_batch])
                    message.append({"role": "assistant", "content": decoded[seq_idx]})
                    cur_messages.append(message)

                format_rewards = format_reward_func(group_decoded)
                answer_rewards = accuracy_reward_func(group_decoded, [answer] * args.repeated_generate_nums)
                cur_rewards = np.array([0.2 * f + a for f, a in zip(format_rewards, answer_rewards)])
                batch_rewards.extend(cur_rewards.tolist())
                if cur_rewards.std() == 0:
                    continue

                pending_messages.extend(cur_messages)
                pending_rewards.extend(cur_rewards.tolist())
                pending_std_rewards.extend(((cur_rewards - cur_rewards.mean()) / cur_rewards.std()).tolist())
                used_items += 1

            total_generate_time += generate_wall_time
            pending_used_items = used_items - used_items_at_last_update
            required_used_items = max(1, args.batch_size * args.accumulation_steps)
            mean_reward = float(np.mean(batch_rewards)) if batch_rewards else 0.0
            batch_bar.set_postfix(
                phase="generated",
                gen=format_duration(generate_wall_time),
                acc_len=f'{outputs["average_accept_length"]:.3f}',
                draft_acc=f'{outputs.get("draft_acceptance_rate", 0.0):.3f}',
                draft=f'{outputs.get("average_draft_steps", 0.0):.2f}',
                thr=f'{outputs.get("average_confidence_threshold", args.confidence_threshold):.3f}',
                active=f'{outputs.get("average_active_batch_size", 0.0):.1f}',
                paths=f'{outputs.get("clasp_average_active_paths", 0.0):.1f}',
                cupd=outputs.get("clasp_update_count", 0),
                drop=outputs.get("cache_tokens_dropped", 0),
                reward=f"{mean_reward:.3f}",
                pending=f"{pending_used_items}/{required_used_items}",
                used=used_items,
                refresh=False,
            )

            log = {
                "epoch": epoch + 1,
                "batch": batch_idx,
                "used_items": used_items,
                "pending_used_items": pending_used_items,
                "generate_time_cost": generate_wall_time,
                "generate_internal_time_cost": outputs.get("total_time_cost", generate_wall_time),
                "target_time_cost": outputs.get("target_time_cost", 0.0),
                "draft_time_cost": outputs.get("draft_time_cost", 0.0),
                "prefill_time_cost": outputs.get("prefill_time_cost", 0.0),
                "post_time_cost": outputs.get("post_time_cost", 0.0),
                "check_time_cost": outputs.get("check_time_cost", 0.0),
                "timing_accounted_time_cost": (
                    outputs.get("target_time_cost", 0.0)
                    + outputs.get("draft_time_cost", 0.0)
                    + outputs.get("prefill_time_cost", 0.0)
                    + outputs.get("post_time_cost", 0.0)
                    + outputs.get("clasp_time_cost", 0.0)
                ),
                "search_time_cost": search_time_cost,
                "total_wall_time_cost": time.time() - start_time,
                "total_generate_time_cost": total_generate_time,
                "total_train_time_cost": total_train_time,
                "average_accept_length": outputs["average_accept_length"],
                "draft_acceptance_rate": outputs.get("draft_acceptance_rate", 0.0),
                "total_accepted_draft_tokens": outputs.get("total_accepted_draft_tokens", 0),
                "total_proposed_draft_tokens": outputs.get("total_proposed_draft_tokens", 0),
                "average_draft_steps": outputs.get("average_draft_steps", 0.0),
                "average_selected_draft_steps": outputs.get("average_selected_draft_steps", 0.0),
                "average_verification_num": outputs.get("average_verification_num", 0.0),
                "average_active_batch_size": outputs.get("average_active_batch_size", 0.0),
                "min_active_batch_size": outputs.get("min_active_batch_size", 0),
                "max_active_batch_size": outputs.get("max_active_batch_size", 0),
                "average_confidence_threshold": outputs.get("average_confidence_threshold", args.confidence_threshold),
                "final_confidence_threshold": outputs.get("final_confidence_threshold", args.confidence_threshold),
                "min_confidence_threshold_seen": outputs.get("min_confidence_threshold_seen", args.confidence_threshold),
                "max_confidence_threshold_seen": outputs.get("max_confidence_threshold_seen", args.confidence_threshold),
                "target_accept_rate": outputs.get("target_accept_rate", args.target_accept_rate),
                "cache_tokens_dropped": outputs.get("cache_tokens_dropped", 0),
                "clasp_enabled": outputs.get("clasp_enabled", args.enable_clasp),
                "clasp_update_count": outputs.get("clasp_update_count", 0),
                "clasp_time_cost": outputs.get("clasp_time_cost", 0.0),
                "clasp_route_calls": outputs.get("clasp_route_calls", 0),
                "clasp_average_codebook_size": outputs.get("clasp_average_codebook_size", 0.0),
                "clasp_average_active_paths": outputs.get("clasp_average_active_paths", 0.0),
                "clasp_min_active_paths": outputs.get("clasp_min_active_paths", 0),
                "clasp_max_active_paths": outputs.get("clasp_max_active_paths", 0),
                "clasp_average_merged_rows": outputs.get("clasp_average_merged_rows", 0.0),
                "clasp_average_route_entropy": outputs.get("clasp_average_route_entropy", 0.0),
                "clasp_average_skip_layers": outputs.get("clasp_average_skip_layers", 0.0),
                "clasp_mean_layer_similarity": outputs.get("clasp_mean_layer_similarity", 0.0),
                "clasp_skip_count": outputs.get("clasp_skip_count", 0),
                "clasp_skip_ratio_config": outputs.get("clasp_skip_ratio_config", args.clasp_skip_ratio),
                "clasp_prefill_update": outputs.get("clasp_prefill_update", args.clasp_prefill_update),
                "clasp_codebook_final_size": outputs.get("clasp_codebook_final_size", 0),
                "mean_reward": mean_reward,
                "used_time_min": round((time.time() - start_time) / 60, 3),
            }

            if pending_messages and pending_used_items >= required_used_items:
                batch_bar.set_postfix(
                    phase="target_train",
                    pending=pending_used_items,
                    used=used_items,
                    refresh=False,
                )
                train_start = time.time()
                text = tokenizer.apply_chat_template(pending_messages, tokenize=False, add_generation_prompt=False)
                tokenized = tokenizer(text, padding=False)
                loss_mask = []
                for message, ids in zip(pending_messages, tokenized.input_ids):
                    prompt_text = tokenizer.apply_chat_template(message[:-1], tokenize=False, add_generation_prompt=True)
                    prompt_ids = tokenizer.encode(prompt_text)
                    loss_mask.append([0] * (len(prompt_ids) - 1) + [1] * (len(ids) - len(prompt_ids) + 1))

                sorted_pairs = sorted(
                    zip(tokenized.input_ids, tokenized.attention_mask, loss_mask, pending_std_rewards),
                    key=lambda x: len(x[0]),
                )

                total_loss = 0.0
                cur_ids, cur_attn, cur_mask, cur_rewards = [], [], [], []
                cur_max_len = 0

                def flush_microbatch():
                    nonlocal total_loss, cur_ids, cur_attn, cur_mask, cur_rewards, cur_max_len
                    if not cur_ids:
                        return
                    for idx in range(len(cur_ids)):
                        pad = cur_max_len - len(cur_ids[idx])
                        if pad > 0:
                            cur_ids[idx] = cur_ids[idx] + [0] * pad
                            cur_attn[idx] = cur_attn[idx] + [0] * pad
                            cur_mask[idx] = cur_mask[idx] + [0] * pad
                    mb_ids = torch.tensor(cur_ids, device="cuda")
                    mb_attn = torch.tensor(cur_attn, device="cuda")
                    mb_mask = torch.tensor(cur_mask, device="cuda")
                    mb_rewards = torch.tensor(cur_rewards, device="cuda").unsqueeze(-1)
                    total_loss += compute_target_loss_and_backward(
                        target_model,
                        mb_ids,
                        mb_attn,
                        mb_mask,
                        mb_rewards,
                        args.epsilon,
                        args.beta,
                        chunk_size=max(1, args.logps_chunk_size),
                        loss_scale=1.0 / max(len(pending_messages), 1),
                    )
                    del mb_ids, mb_attn, mb_mask, mb_rewards
                    cur_ids, cur_attn, cur_mask, cur_rewards, cur_max_len = [], [], [], [], 0

                for ids, attn, mask, reward in sorted_pairs:
                    next_len = max(cur_max_len, len(ids))
                    fits_tokens = next_len * (len(cur_ids) + 1) <= args.max_training_token
                    fits_padding = (len(ids) - cur_max_len) * len(cur_ids) <= args.max_training_padding_gap
                    if cur_ids and not (fits_tokens and fits_padding):
                        flush_microbatch()
                    cur_max_len = max(cur_max_len, len(ids))
                    cur_ids.append(list(ids))
                    cur_attn.append(list(attn))
                    cur_mask.append(list(mask))
                    cur_rewards.append(float(reward))
                flush_microbatch()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_elapsed = time.time() - train_start
                total_train_time += train_elapsed
                log["loss"] = total_loss
                log["train_time_cost"] = train_elapsed
                log["phase"] = "target_train"
                batch_bar.set_postfix(
                    phase="target_train",
                    gen=format_duration(generate_wall_time),
                    train=format_duration(train_elapsed),
                    loss=f"{total_loss:.4f}",
                    used=used_items,
                    refresh=False,
                )
                epoch_bar.set_postfix(
                    elapsed=format_duration(time.time() - start_time),
                    gen=format_duration(total_generate_time),
                    train=format_duration(total_train_time),
                    used=used_items,
                    refresh=False,
                )

                pending_messages.clear()
                pending_rewards.clear()
                pending_std_rewards.clear()
                used_items_at_last_update = used_items
            else:
                log["phase"] = "accumulating_rollouts" if pending_messages else "no_reward_variance"
                batch_bar.set_postfix(
                    phase=log["phase"],
                    gen=format_duration(generate_wall_time),
                    acc=f'{outputs["average_accept_length"]:.3f}',
                    pending=f"{pending_used_items}/{required_used_items}",
                    used=used_items,
                    refresh=False,
                )
                epoch_bar.set_postfix(
                    elapsed=format_duration(time.time() - start_time),
                    gen=format_duration(total_generate_time),
                    train=format_duration(total_train_time),
                    used=used_items,
                    refresh=False,
                )

            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

    total_wall_time = time.time() - start_time
    final_summary = {
        "phase": "final_summary",
        "total_wall_time_cost": total_wall_time,
        "total_wall_time_min": total_wall_time / 60.0,
        "total_generate_time_cost": total_generate_time,
        "total_train_time_cost": total_train_time,
        "search_time_cost": search_time_cost,
        "used_items": used_items,
        "skip_layers_fallback": args.skip_layers,
        "clasp_no_bayes": True,
        "clasp_skip_ratio": args.clasp_skip_ratio,
        "clasp_prefill_update": args.clasp_prefill_update,
        "enable_clasp": args.enable_clasp,
        "clasp_codebook_size": args.clasp_codebook_size,
        "clasp_max_active_paths": args.clasp_max_active_paths,
        "clasp_min_group_size": args.clasp_min_group_size,
        "clasp_update_interval": args.clasp_update_interval,
        "saved_model_dir": os.path.join(args.saved_model_dir, "step0"),
    }
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(final_summary) + "\n")
    print(json.dumps(final_summary, indent=2), flush=True)
    target_model.save_pretrained(os.path.join(args.saved_model_dir, "step0"))


if __name__ == "__main__":
    main()
