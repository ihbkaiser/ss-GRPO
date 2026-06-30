import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from self_speculative_grpo.generate import self_speculative_generate


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Draft & Verify self-speculative decoding.")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--skip_layers", default="24,26,28,30,32,34")
    parser.add_argument("--max_draft_tokens", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--repeated_generate_nums", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--json_out", default="")
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


def run_self_spec(model, tokenizer, prompts, args):
    total_seconds = 0.0
    total_tokens = 0
    total_acc = 0
    total_steps = 0
    for inputs in prompts:
        inputs = {k: v.cuda() for k, v in inputs.items()}
        torch.cuda.synchronize()
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
                statistical_time=False,
            )
        torch.cuda.synchronize()
        total_seconds += time.time() - start
        total_tokens += sum(len(seq) for seq in outputs["generated_token_ids"])
        total_acc += outputs["total_acc_length"]
        total_steps += outputs["total_decoded_token_num"]
    return {
        "seconds": total_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
        "average_accept_length": total_acc / total_steps if total_steps else 0.0,
    }


def run_vanilla(model, tokenizer, prompts, args):
    total_seconds = 0.0
    total_tokens = 0
    for inputs in prompts:
        inputs = {k: v.cuda() for k, v in inputs.items()}
        repeated = {k: v.repeat_interleave(args.repeated_generate_nums, dim=0) for k, v in inputs.items()}
        torch.cuda.synchronize()
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
        torch.cuda.synchronize()
        total_seconds += time.time() - start
        prompt_len = repeated["input_ids"].shape[-1]
        total_tokens += sum(generated_len_until_eos(seq[prompt_len:].tolist(), tokenizer.eos_token_id) for seq in outputs)
    return {
        "seconds": total_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
    }


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto").cuda().eval()
    qas = get_train_QAs(args.train_option)[: args.num_prompts]
    prompts = [build_prompt(tokenizer, item["question"]) for item in qas]

    self_spec = run_self_spec(model, tokenizer, prompts, args)
    vanilla = run_vanilla(model, tokenizer, prompts, args)
    result = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "skip_layers": args.skip_layers,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "num_prompts": args.num_prompts,
        "repeated_generate_nums": args.repeated_generate_nums,
        "max_length": args.max_length,
        "self_speculative": self_spec,
        "vanilla": vanilla,
        "wall_time_speedup": vanilla["seconds"] / self_spec["seconds"] if self_spec["seconds"] else 0.0,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
