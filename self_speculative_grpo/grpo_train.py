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
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
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
    if args.auto_search_skip_layers:
        search_summary = run_auto_skip_search(args, config, target_model, tokenizer, qas)
        args.skip_layers = search_summary["best"]["skip_layers"]
        print(f"Auto skip-layer search selected skip_layers={args.skip_layers}", flush=True)

    dataloader = DataLoader(
        qas,
        collate_fn=TrainDataCollator(tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print(datetime.now())
    print(f"Self-spec GRPO model={args.model_dir} skip_layers={args.skip_layers}")
    print(f"batch={args.batch_size} repeats={args.repeated_generate_nums} max_length={args.max_length}")
    with open(args.log_file, "w", encoding="utf-8"):
        pass

    used_items = 0
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            if batch["input_ids"].shape[-1] >= args.max_length or None in batch["answers"]:
                continue
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
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

            decoded = [tokenizer.decode(item, skip_special_tokens=True) for item in outputs["generated_token_ids"]]
            train_messages = []
            rewards = []
            std_rewards = []
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

            log = {
                "epoch": epoch + 1,
                "batch": batch_idx,
                "used_items": used_items,
                "generate_time_cost": outputs["total_time_cost"],
                "average_accept_length": outputs["average_accept_length"],
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
                "used_time_min": round((time.time() - start_time) / 60, 3),
            }

            if train_messages:
                cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards = build_training_batch(tokenizer, train_messages, std_rewards)
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
                    (loss / args.accumulation_steps).backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    log["loss"] = float(loss.detach().cpu())
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")
            torch.cuda.empty_cache()

    target_model.save_pretrained(os.path.join(args.saved_model_dir, "step0"))


if __name__ == "__main__":
    main()
