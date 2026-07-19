#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_events(log_dir: Path) -> list[dict]:
    metrics_path = log_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    return [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extract_window_rows(events: list[dict], variant: str) -> list[dict]:
    rows = []
    rollout_index = 0
    policy_step = 0
    for event in events:
        if event.get("phase") == "target_train":
            policy_step = int(event.get("step", policy_step))
        if event.get("phase") != "rollout":
            continue
        for trace in event.get("motivation_trace", []) or []:
            row = dict(trace)
            row.update({
                "variant": variant,
                "rollout_index": rollout_index,
                "policy_step_before_rollout": policy_step,
                "batch": int(event.get("batch", 0)),
            })
            rows.append(row)
        rollout_index += 1
    return rows


def plot_comparison(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"disabled": "#777777", "delayed": "#d1495b", "immediate": "#2878b5"}
    for variant in variants:
        subset = [row for row in rows if str(row["variant"]) == variant]
        windows = sorted({int(row["window_index"]) for row in subset})
        x, aal, acceptance = [], [], []
        for window in windows:
            window_rows = [row for row in subset if int(row["window_index"]) == window]
            verify = sum(int(row.get("verify_rounds", 0)) for row in window_rows)
            accepted_length = sum(int(row.get("accepted_length_sum", 0)) for row in window_rows)
            accepted = sum(int(row.get("accepted_medusa_tokens", 0)) for row in window_rows)
            proposed = sum(int(row.get("proposed_medusa_tokens", 0)) for row in window_rows)
            x.append(float(sum(int(row["token_start"]) for row in window_rows)) / len(window_rows))
            aal.append(accepted_length / max(verify, 1))
            acceptance.append(accepted / max(proposed, 1))
        color = colors.get(variant)
        axes[0].plot(x, aal, marker="o", linewidth=1.8, label=variant, color=color)
        axes[1].plot(x, acceptance, marker="o", linewidth=1.8, label=variant, color=color)
    axes[0].set_ylabel("Average accepted length")
    axes[1].set_ylabel("Draft-token acceptance")
    for axis in axes:
        axis.set_xlabel("Generated-token position within trajectory")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path / "within_rollout_motivation.png", dpi=240, bbox_inches="tight")
    fig.savefig(path / "within_rollout_motivation.svg", bbox_inches="tight")


def comparison_summary(rows: list[dict], run_events: dict[str, list[dict]]) -> dict:
    result = {"variants": {}, "generation_hashes_match": None}
    for variant in sorted({str(row["variant"]) for row in rows}):
        subset = [row for row in rows if str(row["variant"]) == variant]
        trajectories: dict[tuple[int, int], list[dict]] = {}
        for row in subset:
            key = (int(row["rollout_index"]), int(row["sequence_id"]))
            trajectories.setdefault(key, []).append(row)
        gains = []
        comparison_window = 3
        for trajectory in trajectories.values():
            by_window = {int(row["window_index"]): row for row in trajectory}
            if 0 not in by_window or comparison_window not in by_window:
                continue
            first, later = by_window[0], by_window[comparison_window]
            first_aal = float(first["accepted_length_sum"]) / max(int(first["verify_rounds"]), 1)
            later_aal = float(later["accepted_length_sum"]) / max(int(later["verify_rounds"]), 1)
            gains.append(later_aal - first_aal)
        result["variants"][variant] = {
            "trajectories": len(trajectories),
            "comparison_window": comparison_window,
            "matched_trajectories_reaching_comparison_window": len(gains),
            "mean_window3_minus_window0_aal": sum(gains) / max(len(gains), 1),
            "positive_gain_fraction": sum(gain > 0 for gain in gains) / max(len(gains), 1),
        }
    hashes = {}
    for variant, events in run_events.items():
        hashes[variant] = [
            digest
            for event in events if event.get("phase") == "rollout"
            for digest in event.get("motivation_generation_hashes", [])
        ]
    if hashes and all(hashes.values()):
        values = list(hashes.values())
        result["generation_hashes_match"] = all(value == values[0] for value in values[1:])
        result["generated_trajectories_by_variant"] = {key: len(value) for key, value in hashes.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FlashGRPO rollout and policy-update diagnostics")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--head-dir", required=True)
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="VARIANT=LOG_DIR",
        help="Add a traced run to the within-rollout comparison (repeatable)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    events = load_events(log_dir)

    rollouts: list[dict] = []
    updates: list[dict] = []
    latest_step = 0
    rollouts_since_update = 0
    previous_cumulative_tokens = 0
    for event in events:
        phase = event.get("phase")
        if phase == "target_train":
            latest_step = int(event.get("step", latest_step))
            rollouts_since_update = 0
            updates.append({
                "step": latest_step,
                "used_items": int(event.get("used_items", 0)),
                "train_time_s": float(event.get("train_time", 0.0)),
                "mean_reward": float(event.get("mean_reward", 0.0)),
                "reward_variance": float(event.get("reward_variance", 0.0)),
            })
        elif phase == "rollout":
            reflex = event.get("reflex", {}) or {}
            cumulative_tokens = int(event.get("total_rollout_tokens", 0))
            generated_tokens = max(0, cumulative_tokens - previous_cumulative_tokens)
            previous_cumulative_tokens = cumulative_tokens
            rollouts.append({
                "rollout_index": len(rollouts),
                "policy_step_before_rollout": latest_step,
                "rollouts_since_policy_update": rollouts_since_update,
                "batch": int(event.get("batch", 0)),
                "generated_tokens": generated_tokens,
                "generation_time_s": float(event.get("generation_time", 0.0)),
                "tokens_per_s": float(event.get("tokens_per_sec_generation", 0.0)),
                "average_accept_length": float(event.get("average_accept_length", 0.0)),
                "medusa_acceptance_rate": float(event.get("medusa_acceptance_rate", 0.0)),
                "accepted_medusa_tokens": int(event.get("accepted_medusa_tokens", 0)),
                "proposed_medusa_tokens": int(event.get("proposed_medusa_tokens", 0)),
                "mean_response_length": float(event.get("mean_response_length", 0.0)),
                "mean_reward": float(event.get("mean_reward", 0.0)),
                "num_reflex_updates": int(reflex.get("num_reflex_updates", 0)),
                "effective_feedback_updates_mean": float(reflex.get("effective_feedback_updates_mean", 0.0)),
                "fast_state_rms_mean": float(reflex.get("fast_state_rms_mean", 0.0)),
                "feedback_collection_fraction": float(reflex.get("feedback_collection_fraction", 0.0)),
                "proposal_injection_effective_scale": float(reflex.get("proposal_injection_effective_scale", 0.0)),
            })
            rollouts_since_update += 1

    write_csv(log_dir / "rollouts.csv", rollouts)
    write_csv(log_dir / "policy_updates.csv", updates)
    own_windows = extract_window_rows(events, "run")
    write_csv(log_dir / "within_rollout_windows.csv", own_windows)

    comparison_rows = []
    comparison_events = {}
    for item in args.compare:
        if "=" not in item:
            raise ValueError(f"--compare expects VARIANT=LOG_DIR, got {item!r}")
        variant, raw_path = item.split("=", 1)
        variant_events = load_events(Path(raw_path))
        comparison_events[variant] = variant_events
        comparison_rows.extend(extract_window_rows(variant_events, variant))
    if comparison_rows:
        write_csv(log_dir / "within_rollout_comparison.csv", comparison_rows)
        plot_comparison(log_dir, comparison_rows)
        stats = comparison_summary(comparison_rows, comparison_events)
        (log_dir / "within_rollout_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    target_steps = sorted(Path(args.target_dir).glob("step*"), key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else -1)
    head_steps = {p.name: p for p in Path(args.head_dir).glob("step*")}
    pairs = []
    for target in target_steps:
        step = int(target.name[4:]) if target.name[4:].isdigit() else None
        if step is None:
            continue
        previous = [candidate for candidate in target_steps if candidate != target and candidate.name[4:].isdigit() and int(candidate.name[4:]) < step]
        old_target = previous[-1] if previous else None
        old_head = head_steps.get(old_target.name) if old_target else None
        pairs.append({
            "old_step": int(old_target.name[4:]) if old_target else 0,
            "new_step": step,
            "old_target": str(old_target) if old_target else "base_model",
            "new_target": str(target),
            "old_heads": str(old_head or Path(args.head_dir)),
        })
    (log_dir / "checkpoint_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    if rollouts:
        x = [row["rollout_index"] for row in rollouts]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(x, [row["average_accept_length"] for row in rollouts], color="#d1495b", linewidth=1.6)
        axes[0].set_ylabel("Average accept length")
        axes[1].plot(x, [row["tokens_per_s"] for row in rollouts], color="#2878b5", linewidth=1.6)
        axes[1].set_ylabel("Generation tokens/s")
        axes[1].set_xlabel("Rollout index")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle("FlashGRPO behavior across online policy updates")
        fig.tight_layout()
        fig.savefig(log_dir / "training_diagnostics.png", dpi=200, bbox_inches="tight")
        fig.savefig(log_dir / "training_diagnostics.svg", bbox_inches="tight")

    total_tokens = sum(row["generated_tokens"] for row in rollouts)
    total_time = sum(row["generation_time_s"] for row in rollouts)
    total_accepted = sum(row["accepted_medusa_tokens"] for row in rollouts)
    total_proposed = sum(row["proposed_medusa_tokens"] for row in rollouts)
    summary = {
        "rollouts": len(rollouts),
        "policy_updates": len(updates),
        "generated_tokens": total_tokens,
        "generation_tokens_per_s": total_tokens / max(total_time, 1e-9),
        "medusa_acceptance_rate": total_accepted / max(total_proposed, 1),
        "checkpoint_pairs": len(pairs),
    }
    (log_dir / "motivation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
