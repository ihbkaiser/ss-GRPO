import argparse
import json
import os
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from helper.rewards import accuracy_reward_func, format_reward_func
from self_speculative_grpo.generate import self_speculative_generate
from self_speculative_grpo.search_skip_layers import (
    build_prompt as build_search_prompt,
    evaluate_mask as evaluate_skip_mask,
    expected_improvement,
    generate_candidate_pool,
    parse_layer_set,
    propose_initial_masks,
)

try:
    from distributed_utils import (
        average_gradients,
        barrier,
        broadcast_object,
        cleanup_distributed,
        make_zero_loss,
        rank0_print,
        reduce_max,
        reduce_sum,
        setup_distributed,
    )
except ImportError:
    from self_speculative_grpo.distributed_utils import (
        average_gradients,
        barrier,
        broadcast_object,
        cleanup_distributed,
        make_zero_loss,
        rank0_print,
        reduce_max,
        reduce_sum,
        setup_distributed,
    )


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-speculative GRPO using Draft & Verify layer skipping.")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--version_name", default="self_spec_grpo")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-GPU prompt batch size under torchrun.")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--target_lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_training_token", type=int, default=1024)
    parser.add_argument("--max_training_padding_gap", type=int, default=128)
    parser.add_argument("--repeated_generate_nums", type=int, default=8)
    parser.add_argument("--skip_layers", default="24,26,28,30,32,34")
    parser.add_argument("--max_draft_tokens", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--auto_search_skip_layers", type=str2bool, default=False)
    parser.add_argument("--search_candidate_layers", default="", help="Comma/range list for auto search, e.g. '18-35'. Empty means latter half.")
    parser.add_argument("--search_num_prompts", type=int, default=1)
    parser.add_argument("--search_max_length", type=int, default=0, help="0 means reuse --max_length.")
    parser.add_argument("--search_min_skip", type=int, default=2)
    parser.add_argument("--search_max_skip", type=int, default=8)
    parser.add_argument("--search_init_trials", type=int, default=6)
    parser.add_argument("--search_bo_trials", type=int, default=12)
    parser.add_argument("--search_candidate_pool", type=int, default=96)
    parser.add_argument("--search_seed", type=int, default=13)
    parser.add_argument("--search_json_out", default="")
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--saved_model_dir", required=True)
    parser.add_argument("--checkpoint_parts_per_epoch", type=int, default=8,
                        help="Save one rolling checkpoint every 1/N epoch. Default N=8.")
    parser.add_argument("--checkpoint_root", default="",
                        help="Rolling checkpoint root. Default: outputs/<version_name>.")
    parser.add_argument("--resume_from_checkpoint", default="",
                        help='Checkpoint dir, outputs/<version_name>, or "auto" to resume from the latest rolling checkpoint.')
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


def compute_target_loss(logits, ref_logits, old_logits, labels, mask, reward, epsilon, beta):
    logits = logits[..., :-1, :].float()
    mask = mask[..., :-1]
    labels = labels[..., 1:].to(logits.device)
    logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    ref_logits = ref_logits[..., :-1, :].float()
    ref_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2).detach()
    old_logps = logps.clone().detach() if old_logits is None else old_logits

    coef1 = torch.exp(logps - old_logps)
    coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
    loss1 = torch.min(coef1 * reward, coef2 * reward)
    coef3 = ref_logps - logps
    loss2 = torch.exp(coef3) - coef3 - 1
    loss = -(loss1 - beta * loss2)
    loss = loss * mask
    denom = mask.sum(-1).clamp_min(1)
    return (loss.sum(-1) / denom).sum()


def build_training_batch(tokenizer, messages, rewards, device):
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
        torch.tensor(input_ids, device=device),
        torch.tensor(attention_mask, device=device),
        torch.tensor(masks, device=device),
        torch.tensor(sorted_rewards, device=device).unsqueeze(-1),
    )


def run_auto_skip_search(args, config, target_model, tokenizer, qas):
    rng = random.Random(args.search_seed)
    np.random.seed(args.search_seed)
    torch.manual_seed(args.search_seed)

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
    summary = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "candidate_layers": candidate_layers,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "num_prompts": len(prompts),
        "max_length": search_args.max_length,
        "objective": "maximize self-speculative generated_tokens / wall_time_seconds before GRPO training",
        "best": results[best_idx],
        "trials": results,
    }
    if args.search_json_out:
        Path(args.search_json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.search_json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    torch.cuda.empty_cache()
    return summary


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _sync_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _maybe_step_optimizer(target_model, optimizer, dist_ctx, pending_micro_steps, force=False):
    if pending_micro_steps == 0:
        return 0
    if force or pending_micro_steps >= 1:
        average_gradients(target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return 0
    return pending_micro_steps


def _resolve_resume_checkpoint(resume_from_checkpoint, root_dir, latest_filename):
    if not resume_from_checkpoint:
        return None

    candidate = root_dir if resume_from_checkpoint.lower() == "auto" else resume_from_checkpoint
    latest_path = os.path.join(candidate, latest_filename)
    if os.path.isdir(candidate):
        if os.path.exists(latest_path):
            with open(latest_path, "r", encoding="utf-8") as f:
                latest = json.load(f)
            checkpoint_dir = latest.get("checkpoint_dir") or latest.get("path")
            if checkpoint_dir and not os.path.isabs(checkpoint_dir):
                checkpoint_dir = os.path.join(candidate, checkpoint_dir)
            candidate = checkpoint_dir
        elif not os.path.exists(os.path.join(candidate, "trainer_state.pt")):
            raise FileNotFoundError(f"Cannot find {latest_filename} or trainer_state.pt in {candidate}")

    if not candidate or not os.path.isdir(candidate):
        raise FileNotFoundError(f"Cannot find resume checkpoint: {resume_from_checkpoint}")
    return candidate


def _load_peft_adapter_state(peft_model, adapter_dir):
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_state = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        adapter_state = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"Cannot find adapter weights in {adapter_dir}")

    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(peft_model, adapter_state, adapter_name="default")


def _checkpoint_part_for_batch(batch_idx, batches_per_epoch, parts_per_epoch):
    if batches_per_epoch <= 0:
        return 0
    completed_batches = batch_idx + 1
    return min(parts_per_epoch, (completed_batches * parts_per_epoch) // batches_per_epoch)


def _next_position_after_batch(epoch, batch_idx, batches_per_epoch):
    next_batch_idx = batch_idx + 1
    if next_batch_idx >= batches_per_epoch:
        return epoch + 1, 0
    return epoch, next_batch_idx


def _save_rolling_checkpoint(
    *,
    root_dir,
    prefix,
    latest_filename,
    version_name,
    checkpoint_part,
    parts_per_epoch,
    epoch,
    batch_idx,
    batches_per_epoch,
    target_model,
    tokenizer,
    optimizer,
    dist_ctx,
    trainer_state,
    rank_state,
    draft_config,
):
    checkpoint_name = f"{prefix}_epoch_{epoch + 1:04d}_part_{checkpoint_part:02d}_of_{parts_per_epoch:02d}"
    tmp_dir = os.path.join(root_dir, f".tmp_{checkpoint_name}")
    checkpoint_dir = os.path.join(root_dir, checkpoint_name)
    next_epoch, next_batch_idx = _next_position_after_batch(epoch, batch_idx, batches_per_epoch)
    state = {
        **trainer_state,
        "script": "self_speculative_grpo/grpo_train.py",
        "version_name": version_name,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "next_epoch": next_epoch,
        "next_batch_idx": next_batch_idx,
        "checkpoint_part": checkpoint_part,
        "checkpoint_parts_per_epoch": parts_per_epoch,
        "batches_per_epoch": batches_per_epoch,
        "checkpoint_name": checkpoint_name,
    }

    if dist_ctx.is_main:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        target_dir = os.path.join(tmp_dir, "target_model")
        draft_dir = os.path.join(tmp_dir, "draft_model")
        os.makedirs(draft_dir, exist_ok=True)

        target_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        with open(os.path.join(draft_dir, "self_speculative_config.json"), "w", encoding="utf-8") as f:
            json.dump(draft_config, f, indent=2)

        common_state = {
            **state,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(common_state, os.path.join(tmp_dir, "trainer_state.pt"))

        json_state = {k: v for k, v in common_state.items() if k != "optimizer"}
        with open(os.path.join(tmp_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(json_state, f, indent=2)

        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.rename(tmp_dir, checkpoint_dir)

    barrier()
    torch.save({**state, **rank_state}, os.path.join(checkpoint_dir, f"rank_{dist_ctx.rank}_state.pt"))
    barrier()

    if dist_ctx.is_main:
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if name.startswith(f"{prefix}_epoch_") and os.path.abspath(path) != os.path.abspath(checkpoint_dir):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        latest = {
            "checkpoint_dir": checkpoint_name,
            "path": checkpoint_dir,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "next_epoch": next_epoch,
            "next_batch_idx": next_batch_idx,
            "checkpoint_part": checkpoint_part,
            "checkpoint_parts_per_epoch": parts_per_epoch,
            "updated_at": datetime.now().isoformat(),
        }
        with open(os.path.join(root_dir, latest_filename), "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2)

        rank0_print(f"Saved rolling checkpoint to {checkpoint_dir}")
    barrier()


def _flush_optimizer_for_checkpoint(target_model, optimizer, dist_ctx, pending_micro_steps, optimizer_step):
    if pending_micro_steps > 0:
        average_gradients(target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        pending_micro_steps = 0
        optimizer_step += 1
    return pending_micro_steps, optimizer_step


def main():
    args = parse_args()
    dist_ctx = setup_distributed(seed=args.seed)
    device = dist_ctx.device
    checkpoint_parts_per_epoch = max(1, args.checkpoint_parts_per_epoch)
    checkpoint_root = args.checkpoint_root or os.path.join("outputs", args.version_name)
    checkpoint_prefix = "self_speculative_grpo"
    latest_checkpoint_file = f"latest_{checkpoint_prefix}_checkpoint.json"

    try:
        if dist_ctx.is_main:
            os.makedirs(args.saved_model_dir, exist_ok=True)
            os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
            os.makedirs(checkpoint_root, exist_ok=True)
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
        target_model = get_peft_model(target_model, lora_config).to(device)
        optimizer = torch.optim.AdamW(target_model.parameters(), lr=args.target_lr)

        qas = get_train_QAs(args.train_option)
        if args.auto_search_skip_layers:
            selected_skip_layers = args.skip_layers
            if dist_ctx.is_main:
                search_summary = run_auto_skip_search(args, config, target_model, tokenizer, qas)
                selected_skip_layers = search_summary["best"]["skip_layers"]
                print(f"Auto skip-layer search selected skip_layers={selected_skip_layers}", flush=True)
            args.skip_layers = broadcast_object(selected_skip_layers, src=0)
            barrier()

        sampler = None
        if dist_ctx.distributed:
            sampler = DistributedSampler(
                qas,
                num_replicas=dist_ctx.world_size,
                rank=dist_ctx.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )

        dataloader = DataLoader(
            qas,
            collate_fn=TrainDataCollator(tokenizer),
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        rank0_print(datetime.now())
        rank0_print(f"Self-spec GRPO model={args.model_dir} skip_layers={args.skip_layers}")
        rank0_print(
            f"world_size={dist_ctx.world_size} per_gpu_batch={args.batch_size} "
            f"global_batch={args.batch_size * dist_ctx.world_size} repeats={args.repeated_generate_nums} "
            f"max_length={args.max_length}"
        )
        rank0_print(f"Rolling checkpoints: every 1/{checkpoint_parts_per_epoch} epoch -> {checkpoint_root}")

        if dist_ctx.is_main and not args.resume_from_checkpoint:
            with open(args.log_file, "w", encoding="utf-8"):
                pass
        barrier()

        used_items = 0
        global_used_items = 0
        pending_micro_steps = 0
        optimizer_step = 0
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        start_epoch = 0
        resume_batch_idx = 0
        resume_checkpoint_epoch = -1
        resume_checkpoint_part = 0

        resume_checkpoint_path = _resolve_resume_checkpoint(args.resume_from_checkpoint, checkpoint_root, latest_checkpoint_file)
        if resume_checkpoint_path is not None:
            _load_peft_adapter_state(target_model, os.path.join(resume_checkpoint_path, "target_model"))
            trainer_state = torch.load(os.path.join(resume_checkpoint_path, "trainer_state.pt"), map_location=device, weights_only=False)
            optimizer.load_state_dict(trainer_state["optimizer"])

            rank_state_path = os.path.join(resume_checkpoint_path, f"rank_{dist_ctx.rank}_state.pt")
            rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=False) if os.path.exists(rank_state_path) else trainer_state

            used_items = int(rank_state.get("used_items", trainer_state.get("used_items", used_items)))
            global_used_items = int(trainer_state.get("global_used_items", global_used_items))
            pending_micro_steps = int(trainer_state.get("pending_micro_steps", 0))
            optimizer_step = int(trainer_state.get("optimizer_step", optimizer_step))
            start_epoch = int(trainer_state.get("next_epoch", 0))
            resume_batch_idx = int(trainer_state.get("next_batch_idx", 0))
            resume_checkpoint_epoch = int(trainer_state.get("epoch", -1))
            resume_checkpoint_part = int(trainer_state.get("checkpoint_part", 0))
            rank0_print(
                f"Resumed rolling checkpoint {resume_checkpoint_path} "
                f"at epoch={start_epoch + 1}, batch={resume_batch_idx}"
            )
        barrier()

        for epoch in range(start_epoch, args.num_epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            last_checkpoint_part = resume_checkpoint_part if epoch == resume_checkpoint_epoch else 0
            batches_per_epoch = len(dataloader)

            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{args.num_epochs}",
                dynamic_ncols=True,
                disable=not dist_ctx.is_main,
            )

            for batch_idx, batch in enumerate(progress_bar):
                if epoch == start_epoch and batch_idx < resume_batch_idx:
                    continue

                local_iter_start = time.time()
                local_generate_time = 0.0
                local_accept_length = 0.0
                local_decoded_steps = 0.0
                local_generated_sequences = 0
                local_mean_reward = 0.0
                local_loss_value = 0.0
                local_train_sequences = 0
                train_messages = []
                rewards = []
                std_rewards = []

                local_skip_batch = batch["input_ids"].shape[-1] >= args.max_length or any(answer is None for answer in batch["answers"])

                if not local_skip_batch:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    _sync_cuda(device)
                    generate_start = time.time()
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
                        )
                    _sync_cuda(device)
                    local_generate_time = time.time() - generate_start

                    decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in outputs["generated_token_ids"]]
                    local_generated_sequences = len(decoded)
                    local_accept_length = float(outputs.get("total_acc_length", 0.0))
                    local_decoded_steps = float(outputs.get("total_decoded_token_num", 0.0))

                    for idx_batch, answer in enumerate(batch["answers"]):
                        cur_rewards = []
                        cur_messages = []
                        for idx_repeat in range(args.repeated_generate_nums):
                            seq_idx = idx_batch * args.repeated_generate_nums + idx_repeat
                            message = deepcopy(batch["messages"][idx_batch])
                            message.append({"role": "assistant", "content": decoded[seq_idx]})
                            reward = 0.2 * format_reward_func([decoded[seq_idx]])[0] + accuracy_reward_func([decoded[seq_idx]], [answer])[0]
                            cur_rewards.append(reward)
                            cur_messages.append(message)
                        cur_rewards = np.array(cur_rewards)
                        if cur_rewards.std() == 0:
                            continue
                        train_messages.extend(cur_messages)
                        rewards.extend(cur_rewards.tolist())
                        std_rewards.extend(((cur_rewards - cur_rewards.mean()) / cur_rewards.std()).tolist())
                        used_items += 1

                    local_mean_reward = _safe_mean(rewards)

                if train_messages:
                    cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards = build_training_batch(
                        tokenizer, train_messages, std_rewards, device
                    )
                    if cur_input_ids.numel() <= args.max_training_token * max(1, len(train_messages)):
                        target_model.disable_adapter_layers()
                        with torch.no_grad():
                            ref_logits = target_model(cur_input_ids, cur_attention_mask).logits
                        target_model.enable_adapter_layers()
                        logits = target_model(cur_input_ids, cur_attention_mask).logits
                        loss = compute_target_loss(
                            logits,
                            ref_logits,
                            None,
                            cur_input_ids,
                            cur_loss_mask,
                            cur_rewards,
                            args.epsilon,
                            args.beta,
                        )
                        local_loss_value = float(loss.detach().cpu())
                        local_train_sequences = len(train_messages)
                    else:
                        loss = None
                        cur_input_ids = None
                else:
                    loss = None

                global_train_sequences = int(reduce_sum(local_train_sequences, device))
                global_used_items = int(reduce_sum(used_items, device))
                global_generated_sequences = int(reduce_sum(local_generated_sequences, device))
                global_accept_length = reduce_sum(local_accept_length, device)
                global_decoded_steps = reduce_sum(local_decoded_steps, device)
                global_generate_time = reduce_max(local_generate_time, device)
                global_iter_time = reduce_max(time.time() - local_iter_start, device)
                global_loss_sum = reduce_sum(local_loss_value, device)
                global_reward_sum = reduce_sum(local_mean_reward * max(1, local_train_sequences), device)

                if global_train_sequences > 0:
                    if loss is not None and local_train_sequences > 0:
                        # compute_target_loss returns a local sum.  Scaling by world_size/global_N
                        # and then averaging gradients across ranks gives the global mean gradient.
                        scaled_loss = loss * (dist_ctx.world_size / float(global_train_sequences))
                        (scaled_loss / args.accumulation_steps).backward()
                    else:
                        (make_zero_loss(target_model.parameters(), device) / args.accumulation_steps).backward()

                    pending_micro_steps += 1
                    if pending_micro_steps >= args.accumulation_steps:
                        average_gradients(target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        pending_micro_steps = 0
                        optimizer_step += 1

                avg_accept_length = global_accept_length / global_decoded_steps if global_decoded_steps > 0 else 0.0
                global_mean_reward = global_reward_sum / global_train_sequences if global_train_sequences > 0 else 0.0
                global_loss_mean = global_loss_sum / global_train_sequences if global_train_sequences > 0 else 0.0

                if dist_ctx.is_main:
                    log = {
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "world_size": dist_ctx.world_size,
                        "per_gpu_batch_size": args.batch_size,
                        "global_prompt_batch_size": args.batch_size * dist_ctx.world_size,
                        "used_items_global_sum": global_used_items,
                        "train_sequences_global": global_train_sequences,
                        "generated_sequences_global": global_generated_sequences,
                        "generate_time_cost_max_rank": global_generate_time,
                        "iteration_time_cost_max_rank": global_iter_time,
                        "average_accept_length_global": avg_accept_length,
                        "mean_reward_global_weighted": global_mean_reward,
                        "loss_global_mean": global_loss_mean,
                        "pending_micro_steps": pending_micro_steps,
                        "optimizer_step": optimizer_step,
                        "used_time_min": round((time.time() - start_time) / 60, 3),
                    }
                    with open(args.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log) + "\n")
                    progress_bar.set_postfix({
                        "used": global_used_items,
                        "train_seq": global_train_sequences,
                        "reward": f"{global_mean_reward:.3f}",
                        "loss": f"{global_loss_mean:.4f}",
                    })

                current_checkpoint_part = _checkpoint_part_for_batch(batch_idx, batches_per_epoch, checkpoint_parts_per_epoch)
                if current_checkpoint_part > last_checkpoint_part:
                    pending_micro_steps, optimizer_step = _flush_optimizer_for_checkpoint(
                        target_model,
                        optimizer,
                        dist_ctx,
                        pending_micro_steps,
                        optimizer_step,
                    )
                    _save_rolling_checkpoint(
                        root_dir=checkpoint_root,
                        prefix=checkpoint_prefix,
                        latest_filename=latest_checkpoint_file,
                        version_name=args.version_name,
                        checkpoint_part=current_checkpoint_part,
                        parts_per_epoch=checkpoint_parts_per_epoch,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batches_per_epoch=batches_per_epoch,
                        target_model=target_model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        dist_ctx=dist_ctx,
                        trainer_state={
                            "global_used_items": global_used_items,
                            "pending_micro_steps": pending_micro_steps,
                            "optimizer_step": optimizer_step,
                        },
                        rank_state={"used_items": used_items},
                        draft_config={
                            "type": "self_speculative_layer_skipping",
                            "skip_layers": args.skip_layers,
                            "max_draft_tokens": args.max_draft_tokens,
                            "confidence_threshold": args.confidence_threshold,
                        },
                    )
                    last_checkpoint_part = current_checkpoint_part

                torch.cuda.empty_cache()

        if pending_micro_steps > 0:
            average_gradients(target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_micro_steps = 0
            optimizer_step += 1

        barrier()
        if dist_ctx.is_main:
            save_path = os.path.join(args.saved_model_dir, "step0")
            target_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            rank0_print(f"Saved model to {save_path}")
        barrier()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
