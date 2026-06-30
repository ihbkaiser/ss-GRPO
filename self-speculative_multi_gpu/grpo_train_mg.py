import argparse
import json
import os
import random
import sys
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

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

try:
    from distributed_utils import (
        setup_distributed,
        cleanup_distributed,
        barrier,
        broadcast_object,
        reduce_sum,
        reduce_max,
        rank0_print,
        average_gradients,
        make_zero_loss,
        max_memory_allocated_gb,
    )
except ImportError:
    from distributed_utils_mg import (
        setup_distributed,
        cleanup_distributed,
        barrier,
        broadcast_object,
        reduce_sum,
        reduce_max,
        rank0_print,
        average_gradients,
        make_zero_loss,
        max_memory_allocated_gb,
    )

from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from helper.rewards import accuracy_reward_func, format_reward_func
from generate_mg import self_speculative_generate
from search_skip_layers_mg import (
    build_prompt as build_search_prompt,
    evaluate_mask as evaluate_skip_mask,
    expected_improvement,
    generate_candidate_pool,
    parse_layer_set,
    propose_initial_masks,
)


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}m"
    return f"{seconds / 3600:.2f}h"


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
    parser.add_argument("--statistical_time", type=lambda x: x.lower() == "true", default=False, help="Enable exact CUDA timing. False avoids frequent cuda synchronize calls.")
    parser.add_argument("--sample_num", type=int, default=100, help="Logging window only; kept for CLI parity with FastGRPO.")
    parser.add_argument("--repeated_generate_nums", type=int, default=8)
    parser.add_argument("--skip_layers", default="24,26,28,30,32,34")
    parser.add_argument("--max_draft_tokens", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--auto_search_skip_layers", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--search_candidate_layers", default="", help="Comma/range list for auto search, e.g. '18-35'. Empty means latter half.")
    parser.add_argument("--search_num_prompts", type=int, default=1)
    parser.add_argument("--search_max_length", type=int, default=0, help="0 means reuse --max_length.")
    parser.add_argument("--search_min_skip", type=int, default=2)
    parser.add_argument("--search_max_skip", type=int, default=8)
    parser.add_argument("--search_init_trials", type=int, default=6)
    parser.add_argument("--search_bo_trials", type=int, default=12)
    parser.add_argument("--search_candidate_pool", type=int, default=96)
    parser.add_argument("--search_seed", type=int, default=13)
    parser.add_argument("--seed", type=int, default=13, help="Base random seed for distributed training.")
    parser.add_argument("--search_json_out", default="")
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



def run_auto_skip_search(args, config, target_model, tokenizer, qas, ctx):
    """Distributed skip-layer search.

    All ranks evaluate the same candidate masks on disjoint prompt shards. Metrics
    are reduced inside search_skip_layers_mg.evaluate_mask, so rank 0 sees global
    tokens/s and acceptance statistics.
    """
    search_start_time = time.time()
    rng = random.Random(args.search_seed)
    np.random.seed(args.search_seed + ctx.rank)
    torch.manual_seed(args.search_seed + ctx.rank)

    candidate_layers = parse_layer_set(args.search_candidate_layers, config.num_hidden_layers)
    if not candidate_layers:
        raise ValueError("No candidate layers to search.")
    max_skip = min(args.search_max_skip, len(candidate_layers))
    min_skip = min(args.search_min_skip, max_skip)
    search_args = argparse.Namespace(
        max_draft_tokens=args.max_draft_tokens,
        confidence_threshold=args.confidence_threshold,
        max_length=args.search_max_length or args.max_length,
    )
    prompts_all = [build_search_prompt(tokenizer, item["question"]) for item in qas[: max(1, args.search_num_prompts)]]
    prompts = prompts_all[ctx.rank::ctx.world_size]

    observed = []
    results = []
    initial = propose_initial_masks(candidate_layers, min_skip, max_skip, max(1, args.search_init_trials), rng)
    for trial_idx, mask in enumerate(initial):
        result = evaluate_skip_mask(target_model, tokenizer, prompts, mask, candidate_layers, search_args)
        observed.append(mask)
        results.append(result)
        if ctx.is_main:
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
        if ctx.is_main:
            print(json.dumps({"auto_skip_search_trial": len(results) - 1, **result}), flush=True)

    best_idx = int(np.argmax([item["score"] for item in results]))
    search_time_cost = reduce_max(time.time() - search_start_time, ctx.device)
    summary = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "candidate_layers": candidate_layers,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "num_prompts": len(prompts_all),
        "max_length": search_args.max_length,
        "objective": "maximize self-speculative generated_tokens / wall_time_seconds before GRPO training",
        "search_time_cost": search_time_cost,
        "search_time_min": search_time_cost / 60.0,
        "best": results[best_idx],
        "trials": results,
    }
    if ctx.is_main and args.search_json_out:
        Path(args.search_json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.search_json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    if ctx.is_main:
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


def _local_reward_and_messages(batch, decoded, repeated_generate_nums):
    pending_messages = []
    pending_rewards = []
    pending_std_rewards = []
    reward_sum = 0.0
    reward_count = 0
    valid_group_count = 0
    skipped_zero_variance = 0

    for idx_batch, answer in enumerate(batch["answers"]):
        group_decoded = []
        cur_messages = []
        for idx_repeat in range(repeated_generate_nums):
            seq_idx = idx_batch * repeated_generate_nums + idx_repeat
            group_decoded.append(decoded[seq_idx])
            message = deepcopy(batch["messages"][idx_batch])
            message.append({"role": "assistant", "content": decoded[seq_idx]})
            cur_messages.append(message)

        format_rewards = format_reward_func(group_decoded)
        answer_rewards = accuracy_reward_func(group_decoded, [answer] * repeated_generate_nums)
        cur_rewards = np.array([0.2 * f + a for f, a in zip(format_rewards, answer_rewards)], dtype=np.float32)
        reward_sum += float(cur_rewards.sum())
        reward_count += int(cur_rewards.size)

        if cur_rewards.std() == 0:
            skipped_zero_variance += 1
            continue

        pending_messages.extend(cur_messages)
        pending_rewards.extend(cur_rewards.tolist())
        pending_std_rewards.extend(((cur_rewards - cur_rewards.mean()) / cur_rewards.std()).tolist())
        valid_group_count += 1

    return pending_messages, pending_rewards, pending_std_rewards, reward_sum, reward_count, valid_group_count, skipped_zero_variance


def _backward_pending_batches(target_model, tokenizer, pending_messages, pending_std_rewards, args, device, global_sequence_count):
    """Backprop local pending sequences. Caller must all-reduce gradients."""
    if not pending_messages:
        zero_loss = make_zero_loss(target_model.parameters(), device)
        zero_loss.backward()
        return 0.0

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
        mb_ids = torch.tensor(cur_ids, device=device)
        mb_attn = torch.tensor(cur_attn, device=device)
        mb_mask = torch.tensor(cur_mask, device=device)
        mb_rewards = torch.tensor(cur_rewards, device=device).unsqueeze(-1)
        total_loss += compute_target_loss_and_backward(
            target_model,
            mb_ids,
            mb_attn,
            mb_mask,
            mb_rewards,
            args.epsilon,
            args.beta,
            chunk_size=max(1, args.logps_chunk_size),
            loss_scale=1.0 / max(float(global_sequence_count), 1.0),
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
    return total_loss


def main():
    args = parse_args()
    ctx = setup_distributed(seed=args.seed)
    device = ctx.device

    if ctx.is_main:
        os.makedirs(args.saved_model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        if args.search_json_out:
            os.makedirs(os.path.dirname(args.search_json_out) or ".", exist_ok=True)
    barrier()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto", config=config).to(device).eval()
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
    search_summary = None
    search_time_cost = 0.0
    if args.auto_search_skip_layers:
        search_summary = run_auto_skip_search(args, config, target_model, tokenizer, qas, ctx)
        selected_skip_layers = search_summary["best"]["skip_layers"] if ctx.is_main else None
        args.skip_layers = broadcast_object(selected_skip_layers, src=0)
        search_time_cost = float(search_summary.get("search_time_cost", 0.0))
    else:
        args.skip_layers = broadcast_object(args.skip_layers if ctx.is_main else None, src=0)

    sampler = DistributedSampler(
        qas,
        num_replicas=ctx.world_size,
        rank=ctx.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    ) if ctx.distributed else None
    loader_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        qas,
        collate_fn=TrainDataCollator(tokenizer),
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=loader_workers,
        persistent_workers=loader_workers > 0,
        pin_memory=True,
    )

    if ctx.is_main:
        print(datetime.now())
        print(f"Self-spec multi-GPU GRPO model={args.model_dir} skip_layers={args.skip_layers}")
        print(f"world_size={ctx.world_size} per_gpu_batch={args.batch_size} global_prompt_batch={ctx.world_size * args.batch_size}")
        print(f"repeats={args.repeated_generate_nums} max_length={args.max_length} accumulation_steps={args.accumulation_steps}")
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "phase": "config",
                "time": str(datetime.now()),
                "world_size": ctx.world_size,
                "per_gpu_batch_size": args.batch_size,
                "global_prompt_batch_size": ctx.world_size * args.batch_size,
                "repeated_generate_nums": args.repeated_generate_nums,
                "max_length": args.max_length,
                "skip_layers": args.skip_layers,
                "target_lr": args.target_lr,
                "accumulation_steps": args.accumulation_steps,
                "search_time_cost": search_time_cost,
            }) + "\n")
            if search_summary is not None:
                f.write(json.dumps({
                    "phase": "auto_skip_search",
                    "search_time_cost": search_time_cost,
                    "search_time_min": search_time_cost / 60.0,
                    "selected_skip_layers": args.skip_layers,
                    "best_tokens_per_second": search_summary["best"].get("tokens_per_second", 0.0),
                    "best_average_accept_length": search_summary["best"].get("average_accept_length", 0.0),
                    "search_trials": len(search_summary.get("trials", [])),
                    "search_json_out": args.search_json_out,
                }) + "\n")

    pending_messages = []
    pending_rewards = []
    pending_std_rewards = []
    used_groups_global = 0
    update_step = 0
    total_generate_time = 0.0
    total_train_time = 0.0
    total_wall_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    epoch_bar = tqdm(range(args.num_epochs), desc="Epoch", dynamic_ncols=True, disable=not ctx.is_main)
    for epoch in epoch_bar:
        if sampler is not None:
            sampler.set_epoch(epoch)
        batch_iter = enumerate(dataloader)
        batch_bar = tqdm(
            batch_iter,
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{args.num_epochs}",
            dynamic_ncols=True,
            leave=False,
            disable=not ctx.is_main,
        )
        for batch_idx, batch in batch_bar:
            iter_wall_start = time.time()
            local_generate_time = 0.0
            local_prefill_time = 0.0
            local_target_time = 0.0
            local_draft_time = 0.0
            local_generated_tokens = 0
            local_acc_len = 0
            local_decoded_steps = 0
            local_draft_proposed = 0
            local_draft_accepted = 0
            local_cache_dropped = 0
            local_reward_sum = 0.0
            local_reward_count = 0
            local_valid_groups = 0
            local_zero_variance = 0
            local_skipped_long = 0
            local_skipped_none = 0

            if batch["input_ids"].shape[-1] >= args.max_length:
                local_skipped_long = 1
            elif None in batch["answers"]:
                local_skipped_none = 1
            else:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
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
                    )

                decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in outputs["generated_token_ids"]]
                new_messages, new_rewards, new_std_rewards, reward_sum, reward_count, valid_groups, zero_variance = _local_reward_and_messages(
                    batch,
                    decoded,
                    args.repeated_generate_nums,
                )
                pending_messages.extend(new_messages)
                pending_rewards.extend(new_rewards)
                pending_std_rewards.extend(new_std_rewards)

                local_generate_time = float(outputs.get("total_time_cost", 0.0))
                local_prefill_time = float(outputs.get("prefill_time_cost", 0.0))
                local_target_time = float(outputs.get("target_time_cost", 0.0))
                local_draft_time = float(outputs.get("draft_time_cost", 0.0))
                local_generated_tokens = sum(len(item) for item in outputs.get("generated_token_ids", []))
                local_acc_len = int(outputs.get("total_acc_length", 0))
                local_decoded_steps = int(outputs.get("total_decoded_token_num", 0))
                local_draft_proposed = int(outputs.get("total_draft_tokens_proposed", 0))
                local_draft_accepted = int(outputs.get("total_draft_tokens_accepted", 0))
                local_cache_dropped = int(outputs.get("cache_tokens_dropped", 0))
                local_reward_sum = reward_sum
                local_reward_count = reward_count
                local_valid_groups = valid_groups
                local_zero_variance = zero_variance

            # Global logging statistics for this dataloader step.
            global_generate_time = reduce_max(local_generate_time, device)
            global_prefill_time = reduce_max(local_prefill_time, device)
            global_target_time = reduce_max(local_target_time, device)
            global_draft_time = reduce_max(local_draft_time, device)
            global_generated_tokens = reduce_sum(local_generated_tokens, device)
            global_acc_len = reduce_sum(local_acc_len, device)
            global_decoded_steps = reduce_sum(local_decoded_steps, device)
            global_draft_proposed = reduce_sum(local_draft_proposed, device)
            global_draft_accepted = reduce_sum(local_draft_accepted, device)
            global_cache_dropped = reduce_sum(local_cache_dropped, device)
            global_reward_sum = reduce_sum(local_reward_sum, device)
            global_reward_count = reduce_sum(local_reward_count, device)
            global_valid_groups = reduce_sum(local_valid_groups, device)
            global_zero_variance = reduce_sum(local_zero_variance, device)
            global_skipped_long = reduce_sum(local_skipped_long, device)
            global_skipped_none = reduce_sum(local_skipped_none, device)

            used_groups_global += int(global_valid_groups)
            total_generate_time += global_generate_time
            pending_global_sequences = int(reduce_sum(len(pending_messages), device))
            should_update = ((batch_idx + 1) % max(1, args.accumulation_steps) == 0) or (batch_idx + 1 == len(dataloader))

            train_time_global = 0.0
            global_loss = 0.0
            did_update = False
            if should_update:
                if pending_global_sequences > 0:
                    train_start = time.time()
                    optimizer.zero_grad(set_to_none=True)
                    local_loss = _backward_pending_batches(
                        target_model,
                        tokenizer,
                        pending_messages,
                        pending_std_rewards,
                        args,
                        device,
                        pending_global_sequences,
                    )
                    # Loss has already been scaled by global sample count. Use SUM,
                    # not average, to obtain the true global mean gradient.
                    average_gradients(target_model.parameters(), world_size=ctx.world_size, divide=False, ensure_grads=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_elapsed = time.time() - train_start
                    train_time_global = reduce_max(train_elapsed, device)
                    global_loss = reduce_sum(local_loss, device)
                    total_train_time += train_time_global
                    update_step += 1
                    did_update = True
                    pending_messages.clear()
                    pending_rewards.clear()
                    pending_std_rewards.clear()
                else:
                    # Keep reduce call count identical across ranks.
                    train_time_global = reduce_max(0.0, device)
                    global_loss = reduce_sum(0.0, device)

            iter_wall_time = reduce_max(time.time() - iter_wall_start, device)
            total_wall_time = reduce_max(time.time() - total_wall_start, device)
            avg_accept_length = global_acc_len / global_decoded_steps if global_decoded_steps else 0.0
            draft_accept_rate = global_draft_accepted / global_draft_proposed if global_draft_proposed else 0.0
            mean_reward = global_reward_sum / global_reward_count if global_reward_count else 0.0
            tokens_per_sec = global_generated_tokens / global_generate_time if global_generate_time else 0.0
            peak_vram_gb = max_memory_allocated_gb(device)

            log = {
                "phase": "target_train" if did_update else ("accumulating_rollouts" if pending_global_sequences > 0 else "rollout_only"),
                "epoch": epoch + 1,
                "batch": batch_idx,
                "rank_world_size": ctx.world_size,
                "per_gpu_batch_size": args.batch_size,
                "global_prompt_batch_size": ctx.world_size * args.batch_size,
                "global_rollout_sequences_per_step": ctx.world_size * args.batch_size * args.repeated_generate_nums,
                "used_groups_global": used_groups_global,
                "pending_sequences_global": pending_global_sequences,
                "update_step": update_step,
                "iteration_wall_time": iter_wall_time,
                "total_wall_time": total_wall_time,
                "generate_time_cost": global_generate_time,
                "prefill_time_cost": global_prefill_time,
                "target_verify_time_cost": global_target_time,
                "draft_time_cost": global_draft_time,
                "train_time_cost": train_time_global,
                "total_generate_time_cost": total_generate_time,
                "total_train_time_cost": total_train_time,
                "generated_tokens": int(global_generated_tokens),
                "tokens_per_second": tokens_per_sec,
                "total_acc_length": int(global_acc_len),
                "total_decoded_token_num": int(global_decoded_steps),
                "average_accept_length": avg_accept_length,
                "draft_tokens_proposed": int(global_draft_proposed),
                "draft_tokens_accepted": int(global_draft_accepted),
                "draft_accept_rate": draft_accept_rate,
                "cache_tokens_dropped": int(global_cache_dropped),
                "mean_reward": mean_reward,
                "valid_reward_groups": int(global_valid_groups),
                "zero_variance_groups": int(global_zero_variance),
                "skipped_long_batches": int(global_skipped_long),
                "skipped_none_answer_batches": int(global_skipped_none),
                "loss": global_loss,
                "peak_vram_gb": peak_vram_gb,
                "search_time_cost": search_time_cost,
            }

            if ctx.is_main:
                batch_bar.set_postfix(
                    phase=log["phase"],
                    gen=format_duration(global_generate_time),
                    train=format_duration(train_time_global),
                    acc=f"{avg_accept_length:.3f}",
                    ar=f"{draft_accept_rate:.3f}",
                    tokps=f"{tokens_per_sec:.1f}",
                    reward=f"{mean_reward:.3f}",
                    used=used_groups_global,
                    vram=f"{peak_vram_gb:.1f}G",
                    refresh=False,
                )
                epoch_bar.set_postfix(
                    elapsed=format_duration(total_wall_time),
                    gen=format_duration(total_generate_time),
                    train=format_duration(total_train_time),
                    used=used_groups_global,
                    refresh=False,
                )
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log) + "\n")

    final_wall_time = reduce_max(time.time() - total_wall_start, device)
    final_peak_vram = max_memory_allocated_gb(device)
    if ctx.is_main:
        final_summary = {
            "phase": "final_summary",
            "total_wall_time": final_wall_time,
            "total_wall_time_min": final_wall_time / 60.0,
            "total_generate_time_cost": total_generate_time,
            "total_train_time_cost": total_train_time,
            "search_time_cost": search_time_cost,
            "used_groups_global": used_groups_global,
            "update_steps": update_step,
            "world_size": ctx.world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_prompt_batch_size": ctx.world_size * args.batch_size,
            "peak_vram_gb": final_peak_vram,
            "saved_model_dir": args.saved_model_dir,
        }
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_summary) + "\n")
        target_model.save_pretrained(os.path.join(args.saved_model_dir, "step0"))
    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
