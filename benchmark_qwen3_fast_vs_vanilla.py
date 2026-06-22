import argparse
import json
import time
from copy import deepcopy

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from helper.modeling_draft import Model
from helper.specualtive_generate import speculative_generate


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark FastGRPO generation against vanilla target generation.")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--draft_num_hidden_layers", type=int, default=1)
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=4)
    parser.add_argument("--repeated_generate_nums", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--warmup_batches", type=int, default=1)
    parser.add_argument("--json_out", default="")
    return parser.parse_args()


def build_batch(tokenizer, examples):
    system_prompt = "You are a math problem assistant."
    user_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}"""
    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format_map({"instruction": example["question"]})},
        ]
        for example in examples
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokenized = tokenizer(
        text=text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=4096,
        padding_side="left",
    )
    return tokenized["input_ids"].cuda(), tokenized["attention_mask"].cuda()


def iter_batches(qas, batch_size, num_prompts):
    qas = qas[:num_prompts]
    for start in range(0, len(qas), batch_size):
        yield qas[start : start + batch_size]


def generated_len_until_eos(tokens, eos_token_id):
    count = 0
    for token in tokens:
        count += 1
        if token == eos_token_id:
            break
    return count


def run_fast(model, tokenizer, batches, args):
    total_seconds = 0.0
    total_tokens = 0
    total_acc_length = 0
    total_decoded = 0

    for batch in batches:
        input_ids, attention_mask = build_batch(tokenizer, batch)
        if input_ids.shape[-1] >= args.max_length:
            continue
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            outputs = speculative_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                do_sample=True,
                max_length=args.max_length,
                repeated_generate_nums=args.repeated_generate_nums,
                temperature=args.temperature,
                top_p=args.top_p,
                return_all_draft_input=False,
                statistical_time=True,
            )
        torch.cuda.synchronize()
        total_seconds += time.time() - start
        total_tokens += sum(len(seq) for seq in outputs["generated_token_ids"])
        total_acc_length += outputs["total_acc_length"]
        total_decoded += outputs["total_decoded_token_num"]

    return {
        "seconds": total_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": total_tokens / total_seconds if total_seconds else 0.0,
        "average_accept_length": total_acc_length / total_decoded if total_decoded else 0.0,
    }


def run_vanilla(target_model, tokenizer, batches, args):
    total_seconds = 0.0
    total_tokens = 0
    eos_token_id = tokenizer.eos_token_id

    for batch in batches:
        input_ids, attention_mask = build_batch(tokenizer, batch)
        if input_ids.shape[-1] >= args.max_length:
            continue
        input_ids = input_ids.repeat_interleave(args.repeated_generate_nums, dim=0)
        attention_mask = attention_mask.repeat_interleave(args.repeated_generate_nums, dim=0)
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            outputs = target_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_length=args.max_length,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        torch.cuda.synchronize()
        total_seconds += time.time() - start
        prompt_len = input_ids.shape[-1]
        total_tokens += sum(generated_len_until_eos(seq[prompt_len:].tolist(), eos_token_id) for seq in outputs)

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

    target_config = AutoConfig.from_pretrained(args.model_dir)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        config=target_config,
    ).cuda().eval()

    draft_config = deepcopy(target_config)
    draft_config.rope_scaling = None
    draft_config.num_hidden_layers = args.draft_num_hidden_layers
    fast_model = Model(draft_config, target_model=target_model)
    fast_model.load_model(args.adapter_path)
    fast_model = fast_model.cuda().eval()

    qas = get_train_QAs(args.train_option)
    warmup = list(iter_batches(qas, args.batch_size, min(args.num_prompts, args.batch_size * args.warmup_batches)))
    measured = list(iter_batches(qas[args.batch_size * args.warmup_batches :], args.batch_size, args.num_prompts))
    if not measured:
        measured = list(iter_batches(qas, args.batch_size, args.num_prompts))

    with torch.inference_mode():
        if warmup:
            run_fast(fast_model, tokenizer, warmup, args)
            run_vanilla(target_model, tokenizer, warmup, args)
            torch.cuda.empty_cache()
        fast = run_fast(fast_model, tokenizer, measured, args)
        vanilla = run_vanilla(target_model, tokenizer, measured, args)

    result = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "batch_size": args.batch_size,
        "num_prompts": len(measured) * args.batch_size,
        "repeated_generate_nums": args.repeated_generate_nums,
        "draft_num_hidden_layers": args.draft_num_hidden_layers,
        "max_length": args.max_length,
        "fastgrpo": fast,
        "vanilla": vanilla,
        "wall_time_speedup": vanilla["seconds"] / fast["seconds"] if fast["seconds"] else 0.0,
        "tokens_per_second_speedup": fast["tokens_per_second"] / vanilla["tokens_per_second"]
        if vanilla["tokens_per_second"]
        else 0.0,
    }

    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
