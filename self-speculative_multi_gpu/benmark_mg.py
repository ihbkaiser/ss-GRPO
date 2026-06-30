import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from helper.get_QAs import get_train_QAs
from generate_mg import self_speculative_generate

try:
    from distributed_utils import setup_distributed, cleanup_distributed, reduce_sum, reduce_max, max_memory_allocated_gb
except ImportError:
    from distributed_utils_mg import setup_distributed, cleanup_distributed, reduce_sum, reduce_max, max_memory_allocated_gb


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU benchmark for self-speculative decoding.")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--skip_layers", default="24,26,28,30,32,34")
    parser.add_argument("--max_draft_tokens", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--num_prompts", type=int, default=16)
    parser.add_argument("--repeated_generate_nums", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--statistical_time", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--json_out", default="")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": "You are a math problem assistant."},
        {"role": "user", "content": f"Solve this math problem step by step: {question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, return_tensors="pt")


def generated_len_until_eos(tokens, eos_token_id):
    count = 0
    for token in tokens:
        count += 1
        if token == eos_token_id:
            break
    return count


def aggregate_generation_stats(local, device):
    seconds = reduce_max(local.get("seconds", 0.0), device)
    generated_tokens = reduce_sum(local.get("generated_tokens", 0), device)
    total_acc = reduce_sum(local.get("total_acc_length", 0), device)
    total_steps = reduce_sum(local.get("total_decoded_token_num", 0), device)
    draft_proposed = reduce_sum(local.get("total_draft_tokens_proposed", 0), device)
    draft_accepted = reduce_sum(local.get("total_draft_tokens_accepted", 0), device)
    return {
        "seconds": seconds,
        "generated_tokens": int(generated_tokens),
        "tokens_per_second": generated_tokens / seconds if seconds else 0.0,
        "total_acc_length": int(total_acc),
        "total_decoded_token_num": int(total_steps),
        "average_accept_length": total_acc / total_steps if total_steps else 0.0,
        "total_draft_tokens_proposed": int(draft_proposed),
        "total_draft_tokens_accepted": int(draft_accepted),
        "draft_accept_rate": draft_accepted / draft_proposed if draft_proposed else 0.0,
    }


def run_self_spec(model, tokenizer, prompts, args, ctx):
    total_seconds = 0.0
    total_tokens = 0
    total_acc = 0
    total_steps = 0
    total_draft_proposed = 0
    total_draft_accepted = 0
    iterator = tqdm(prompts, desc="self-spec local prompts", leave=False, disable=not ctx.is_main)
    for inputs in iterator:
        inputs = {k: v.to(ctx.device) for k, v in inputs.items()}
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        start = time.time()
        with torch.inference_mode():
            outputs = self_speculative_generate(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                tokenizer,
                skip_layers=args.skip_layers,
                max_draft_tokens=args.max_draft_tokens,
                confidence_threshold=args.confidence_threshold,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repeated_generate_nums=args.repeated_generate_nums,
                max_length=args.max_length,
                statistical_time=args.statistical_time,
            )
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        total_seconds += time.time() - start
        total_tokens += sum(len(seq) for seq in outputs["generated_token_ids"])
        total_acc += int(outputs.get("total_acc_length", 0))
        total_steps += int(outputs.get("total_decoded_token_num", 0))
        total_draft_proposed += int(outputs.get("total_draft_tokens_proposed", 0))
        total_draft_accepted += int(outputs.get("total_draft_tokens_accepted", 0))
    return aggregate_generation_stats({
        "seconds": total_seconds,
        "generated_tokens": total_tokens,
        "total_acc_length": total_acc,
        "total_decoded_token_num": total_steps,
        "total_draft_tokens_proposed": total_draft_proposed,
        "total_draft_tokens_accepted": total_draft_accepted,
    }, ctx.device)


def run_vanilla(model, tokenizer, prompts, args, ctx):
    total_seconds = 0.0
    total_tokens = 0
    iterator = tqdm(prompts, desc="vanilla local prompts", leave=False, disable=not ctx.is_main)
    for inputs in iterator:
        inputs = {k: v.to(ctx.device) for k, v in inputs.items()}
        repeated = {k: v.repeat_interleave(args.repeated_generate_nums, dim=0) for k, v in inputs.items()}
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        start = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **repeated,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_length,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        total_seconds += time.time() - start
        prompt_len = repeated["input_ids"].shape[-1]
        total_tokens += sum(generated_len_until_eos(seq[prompt_len:].tolist(), tokenizer.eos_token_id) for seq in outputs)
    seconds = reduce_max(total_seconds, ctx.device)
    tokens = reduce_sum(total_tokens, ctx.device)
    return {
        "seconds": seconds,
        "generated_tokens": int(tokens),
        "tokens_per_second": tokens / seconds if seconds else 0.0,
    }


def main():
    args = parse_args()
    ctx = setup_distributed(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto").to(ctx.device).eval()

    qas = get_train_QAs(args.train_option)[: args.num_prompts]
    prompts_all = [build_prompt(tokenizer, item["question"]) for item in qas]
    prompts = prompts_all[ctx.rank::ctx.world_size]

    self_spec = run_self_spec(model, tokenizer, prompts, args, ctx)
    vanilla = run_vanilla(model, tokenizer, prompts, args, ctx)
    peak_vram = max_memory_allocated_gb(ctx.device)

    result = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "world_size": ctx.world_size,
        "num_prompts_global": len(prompts_all),
        "num_prompts_local_rank0": len(prompts_all[0::ctx.world_size]),
        "skip_layers": args.skip_layers,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "repeated_generate_nums": args.repeated_generate_nums,
        "max_length": args.max_length,
        "self_speculative": self_spec,
        "vanilla": vanilla,
        "wall_time_speedup": vanilla["seconds"] / self_spec["seconds"] if self_spec["seconds"] else 0.0,
        "peak_vram_gb": peak_vram,
    }
    if ctx.is_main:
        print(json.dumps(result, indent=2), flush=True)
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
    cleanup_distributed()


if __name__ == "__main__":
    main()
