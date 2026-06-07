import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3-4B vanilla generation against Qwen3-0.6B assisted generation.")
    parser.add_argument("--target_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft_model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--repeated_generate_nums", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json_out", default="")
    return parser.parse_args()


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": f"Solve this math problem step by step: {question}"}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt").to("cuda")


def count_new_tokens(outputs, prompt_len, eos_token_id):
    total = 0
    for seq in outputs:
        for token in seq[prompt_len:].tolist():
            total += 1
            if token == eos_token_id:
                break
    return total


def generate_kwargs(args, tokenizer):
    return {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }


def time_vanilla_batched(target, tokenizer, prompts, args):
    kwargs = generate_kwargs(args, tokenizer)
    total_seconds = 0.0
    total_tokens = 0
    for inputs in prompts:
        batched = {k: v.repeat_interleave(args.repeated_generate_nums, dim=0) for k, v in inputs.items()}
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            outputs = target.generate(**batched, **kwargs)
        torch.cuda.synchronize()
        total_seconds += time.time() - start
        total_tokens += count_new_tokens(outputs, batched["input_ids"].shape[-1], tokenizer.eos_token_id)
    return summarize(total_seconds, total_tokens)


def time_vanilla_loop(target, tokenizer, prompts, args):
    kwargs = generate_kwargs(args, tokenizer)
    total_seconds = 0.0
    total_tokens = 0
    for inputs in prompts:
        for _ in range(args.repeated_generate_nums):
            torch.cuda.synchronize()
            start = time.time()
            with torch.inference_mode():
                outputs = target.generate(**inputs, **kwargs)
            torch.cuda.synchronize()
            total_seconds += time.time() - start
            total_tokens += count_new_tokens(outputs, inputs["input_ids"].shape[-1], tokenizer.eos_token_id)
    return summarize(total_seconds, total_tokens)


def time_assisted_loop(target, draft, tokenizer, prompts, args):
    kwargs = generate_kwargs(args, tokenizer)
    total_seconds = 0.0
    total_tokens = 0
    for inputs in prompts:
        for _ in range(args.repeated_generate_nums):
            torch.cuda.synchronize()
            start = time.time()
            with torch.inference_mode():
                outputs = target.generate(**inputs, assistant_model=draft, **kwargs)
            torch.cuda.synchronize()
            total_seconds += time.time() - start
            total_tokens += count_new_tokens(outputs, inputs["input_ids"].shape[-1], tokenizer.eos_token_id)
    return summarize(total_seconds, total_tokens)


def summarize(seconds, tokens):
    return {
        "seconds": seconds,
        "generated_tokens": tokens,
        "tokens_per_second": tokens / seconds if seconds else 0.0,
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    target = AutoModelForCausalLM.from_pretrained(args.target_model, torch_dtype="auto").cuda().eval()
    draft = AutoModelForCausalLM.from_pretrained(args.draft_model, torch_dtype="auto").cuda().eval()

    qas = get_train_QAs(args.train_option)[: args.num_prompts]
    prompts = [build_prompt(tokenizer, item["question"]) for item in qas]
    warmup_prompts = prompts[:1]

    if args.warmup and warmup_prompts:
        time_vanilla_batched(target, tokenizer, warmup_prompts, args)
        time_vanilla_loop(target, tokenizer, warmup_prompts, args)
        time_assisted_loop(target, draft, tokenizer, warmup_prompts, args)
        torch.cuda.empty_cache()

    vanilla_batched = time_vanilla_batched(target, tokenizer, prompts, args)
    vanilla_loop = time_vanilla_loop(target, tokenizer, prompts, args)
    assisted_loop = time_assisted_loop(target, draft, tokenizer, prompts, args)

    result = {
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "train_option": args.train_option,
        "num_prompts": args.num_prompts,
        "repeated_generate_nums": args.repeated_generate_nums,
        "max_new_tokens": args.max_new_tokens,
        "vanilla_batched": vanilla_batched,
        "vanilla_loop": vanilla_loop,
        "assisted_loop": assisted_loop,
        "assisted_vs_vanilla_batched_speedup": vanilla_batched["seconds"] / assisted_loop["seconds"]
        if assisted_loop["seconds"]
        else 0.0,
        "assisted_vs_vanilla_loop_speedup": vanilla_loop["seconds"] / assisted_loop["seconds"]
        if assisted_loop["seconds"]
        else 0.0,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
