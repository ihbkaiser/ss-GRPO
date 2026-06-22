import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from self_speculative_grpo.generate import self_speculative_generate


def parse_args():
    parser = argparse.ArgumentParser(description="Bayesian search for Draft & Verify self-speculative skip layers.")
    parser.add_argument("--model_dir", default="Qwen/Qwen3-4B")
    parser.add_argument("--train_option", default="simplelr_abel_level3to5_smoke")
    parser.add_argument("--candidate_layers", default="", help="Comma/range list, e.g. '18-35' or '20,22,24'. Empty means latter half.")
    parser.add_argument("--num_prompts", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--max_draft_tokens", type=int, default=3)
    parser.add_argument("--confidence_threshold", type=float, default=0.0)
    parser.add_argument("--min_skip", type=int, default=2)
    parser.add_argument("--max_skip", type=int, default=8)
    parser.add_argument("--init_trials", type=int, default=6)
    parser.add_argument("--bo_trials", type=int, default=12)
    parser.add_argument("--candidate_pool", type=int, default=96)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--json_out", default="runs/self_spec_qwen3_4b/skip_search.json")
    return parser.parse_args()


def parse_layer_set(value, num_layers):
    if not value:
        return list(range(num_layers // 2, num_layers))
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    return sorted({layer for layer in layers if 0 <= layer < num_layers})


def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": "You are a math problem assistant."},
        {"role": "user", "content": f"Solve this math problem step by step: {question}"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, return_tensors="pt")


def mask_to_layers(mask, candidate_layers):
    return [layer for bit, layer in zip(mask, candidate_layers) if bit]


def layers_to_string(layers):
    return ",".join(str(layer) for layer in sorted(layers))


def random_mask(dim, min_skip, max_skip, rng):
    max_skip = min(max_skip, dim)
    min_skip = min(min_skip, max_skip)
    count = rng.randint(min_skip, max_skip)
    indices = rng.sample(range(dim), count)
    mask = np.zeros(dim, dtype=np.int8)
    mask[indices] = 1
    return mask


def repair_mask(mask, min_skip, max_skip, rng):
    mask = mask.copy()
    ones = np.where(mask == 1)[0].tolist()
    zeros = np.where(mask == 0)[0].tolist()
    while len(ones) < min_skip and zeros:
        idx = rng.choice(zeros)
        zeros.remove(idx)
        ones.append(idx)
        mask[idx] = 1
    while len(ones) > max_skip:
        idx = rng.choice(ones)
        ones.remove(idx)
        mask[idx] = 0
    return mask


def propose_initial_masks(candidate_layers, min_skip, max_skip, init_trials, rng):
    dim = len(candidate_layers)
    masks = []
    patterns = [
        [idx for idx, layer in enumerate(candidate_layers) if layer % 2 == 0],
        [idx for idx, layer in enumerate(candidate_layers) if layer % 2 == 1],
        list(range(max(0, dim - max_skip), dim)),
    ]
    for pattern in patterns:
        mask = np.zeros(dim, dtype=np.int8)
        mask[pattern[:max_skip]] = 1
        mask = repair_mask(mask, min_skip, max_skip, rng)
        masks.append(mask)
    while len(masks) < init_trials:
        masks.append(random_mask(dim, min_skip, max_skip, rng))
    return unique_masks(masks)[:init_trials]


def unique_masks(masks):
    seen = set()
    unique = []
    for mask in masks:
        key = tuple(int(x) for x in mask)
        if key in seen:
            continue
        seen.add(key)
        unique.append(mask)
    return unique


def mutate_mask(mask, min_skip, max_skip, rng):
    out = mask.copy()
    flip_count = rng.randint(1, min(4, len(mask)))
    for idx in rng.sample(range(len(mask)), flip_count):
        out[idx] = 1 - out[idx]
    return repair_mask(out, min_skip, max_skip, rng)


def generate_candidate_pool(observed, best_mask, dim, min_skip, max_skip, pool_size, rng):
    observed_keys = {tuple(int(x) for x in mask) for mask in observed}
    pool = []
    attempts = 0
    while len(pool) < pool_size and attempts < pool_size * 30:
        attempts += 1
        if best_mask is not None and rng.random() < 0.65:
            mask = mutate_mask(best_mask, min_skip, max_skip, rng)
        else:
            mask = random_mask(dim, min_skip, max_skip, rng)
        key = tuple(int(x) for x in mask)
        if key not in observed_keys:
            observed_keys.add(key)
            pool.append(mask)
    return pool


def hamming_kernel(xa, xb, length_scale):
    diff = np.not_equal(xa[:, None, :], xb[None, :, :]).mean(axis=-1)
    return np.exp(-diff / max(length_scale, 1e-6))


def normal_pdf(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def normal_cdf(x):
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def expected_improvement(x_obs, y_obs, x_pool):
    if len(x_obs) < 2:
        return np.ones(len(x_pool), dtype=np.float64)
    y_mean = y_obs.mean()
    y_std = y_obs.std() or 1.0
    y = (y_obs - y_mean) / y_std
    length_scale = 0.35
    k_xx = hamming_kernel(x_obs, x_obs, length_scale) + np.eye(len(x_obs)) * 1e-5
    k_xs = hamming_kernel(x_obs, x_pool, length_scale)
    try:
        alpha = np.linalg.solve(k_xx, y)
        v = np.linalg.solve(k_xx, k_xs)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(k_xx, y, rcond=None)[0]
        v = np.linalg.lstsq(k_xx, k_xs, rcond=None)[0]
    mu = k_xs.T @ alpha
    var = np.maximum(1.0 - np.sum(k_xs * v, axis=0), 1e-9)
    sigma = np.sqrt(var)
    improvement = mu - y.max()
    z = improvement / sigma
    return improvement * normal_cdf(z) + sigma * normal_pdf(z)


def evaluate_mask(model, tokenizer, prompts, mask, candidate_layers, args):
    skip_layers = layers_to_string(mask_to_layers(mask, candidate_layers))
    total_seconds = 0.0
    total_tokens = 0
    total_accept = 0.0
    total_steps = 0
    for prompt in prompts:
        inputs = {key: value.cuda() for key, value in prompt.items()}
        torch.cuda.synchronize()
        start = time.time()
        with torch.inference_mode():
            outputs = self_speculative_generate(
                model,
                inputs["input_ids"],
                inputs["attention_mask"],
                tokenizer,
                skip_layers=skip_layers,
                max_draft_tokens=args.max_draft_tokens,
                confidence_threshold=args.confidence_threshold,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repeated_generate_nums=1,
                max_length=args.max_length,
            )
        torch.cuda.synchronize()
        total_seconds += time.time() - start
        total_tokens += sum(len(seq) for seq in outputs["generated_token_ids"])
        total_accept += outputs["total_acc_length"]
        total_steps += outputs["total_decoded_token_num"]
    tokens_per_second = total_tokens / total_seconds if total_seconds else 0.0
    return {
        "skip_layers": skip_layers,
        "seconds": total_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "average_accept_length": total_accept / total_steps if total_steps else 0.0,
        "score": tokens_per_second,
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_dir)
    candidate_layers = parse_layer_set(args.candidate_layers, config.num_hidden_layers)
    if not candidate_layers:
        raise ValueError("No candidate layers to search.")
    args.max_skip = min(args.max_skip, len(candidate_layers))
    args.min_skip = min(args.min_skip, args.max_skip)

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype="auto", config=config).cuda().eval()
    qas = get_train_QAs(args.train_option)[: args.num_prompts]
    prompts = [build_prompt(tokenizer, item["question"]) for item in qas]

    observed = []
    results = []
    initial = propose_initial_masks(candidate_layers, args.min_skip, args.max_skip, args.init_trials, rng)
    for trial_idx, mask in enumerate(initial):
        result = evaluate_mask(model, tokenizer, prompts, mask, candidate_layers, args)
        observed.append(mask)
        results.append(result)
        print(json.dumps({"trial": trial_idx, **result}), flush=True)

    for _ in range(args.bo_trials):
        best_idx = int(np.argmax([item["score"] for item in results]))
        pool = generate_candidate_pool(
            observed,
            observed[best_idx],
            len(candidate_layers),
            args.min_skip,
            args.max_skip,
            args.candidate_pool,
            rng,
        )
        if not pool:
            break
        x_obs = np.stack(observed).astype(np.float64)
        y_obs = np.array([item["score"] for item in results], dtype=np.float64)
        x_pool = np.stack(pool).astype(np.float64)
        acquisition = expected_improvement(x_obs, y_obs, x_pool)
        mask = pool[int(np.argmax(acquisition))]
        result = evaluate_mask(model, tokenizer, prompts, mask, candidate_layers, args)
        observed.append(mask)
        results.append(result)
        print(json.dumps({"trial": len(results) - 1, **result}), flush=True)

    best_idx = int(np.argmax([item["score"] for item in results]))
    summary = {
        "model_dir": args.model_dir,
        "train_option": args.train_option,
        "candidate_layers": candidate_layers,
        "max_draft_tokens": args.max_draft_tokens,
        "confidence_threshold": args.confidence_threshold,
        "num_prompts": args.num_prompts,
        "max_length": args.max_length,
        "objective": "maximize self-speculative generated_tokens / wall_time_seconds",
        "best": results[best_idx],
        "trials": results,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
