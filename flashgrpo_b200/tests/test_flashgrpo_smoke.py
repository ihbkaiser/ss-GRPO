from flashgrpo_b200.decoding.medusa_tree import plan_tree
import torch

from flashgrpo_b200.training.flashgrpo_trainer import (
    CpeakRolloutTuner,
    _merge_online_ce_update_stats,
    build_medusa_update_batch,
)


def test_concurrency_tree_shrinks_with_batch():
    small_b = plan_tree(
        active_batch_size=1,
        num_medusa_heads=3,
        tree_mode="concurrency_aware",
        tree_layout="dense",
        cpeak_nodes=32,
        min_tree_nodes_per_seq=1,
        max_tree_nodes_per_seq=16,
        max_tree_depth=4,
        fixed_tree_topk_by_depth=[4, 3, 2],
    )
    large_b = plan_tree(
        active_batch_size=16,
        num_medusa_heads=3,
        tree_mode="concurrency_aware",
        tree_layout="dense",
        cpeak_nodes=32,
        min_tree_nodes_per_seq=1,
        max_tree_nodes_per_seq=16,
        max_tree_depth=4,
        fixed_tree_topk_by_depth=[4, 3, 2],
    )
    assert small_b.actual_nodes >= large_b.actual_nodes


def test_adaptive_planner_preserves_deep_heads_before_budget_fit():
    plan = plan_tree(
        active_batch_size=64,
        num_medusa_heads=3,
        tree_mode="concurrency_aware",
        tree_layout="dense",
        cpeak_nodes=256,
        min_tree_nodes_per_seq=1,
        max_tree_nodes_per_seq=12,
        max_tree_depth=4,
        fixed_tree_topk_by_depth=[4, 3, 2],
        adaptive_tree_enabled=True,
        adaptive_min_topk_by_depth=[1, 1, 1],
    )
    assert plan.node_budget_per_seq == 4
    assert plan.active_heads == 3


def test_cpeak_tuner_interleaves_candidates_and_selects_median_throughput():
    tuner = CpeakRolloutTuner(
        enabled=True,
        candidates=[128, 256],
        trials_per_candidate=2,
        start_rollout=0,
        default_cpeak=128,
    )
    observations = [(100.0, 1.0), (80.0, 1.0), (110.0, 1.0), (90.0, 1.0)]
    selected_now = False
    for rollout, (tokens, elapsed) in enumerate(observations):
        budget, tuning = tuner.budget_for(rollout)
        assert budget == [128, 256, 128, 256][rollout]
        selected_now = tuner.observe(
            budget,
            output_tokens=int(tokens),
            elapsed_s=elapsed,
            tuning=tuning,
        )
    assert selected_now
    assert tuner.selected == 128
    assert tuner.budget_for(4) == (128, False)

    resumed = CpeakRolloutTuner(
        enabled=True,
        candidates=[128, 256],
        trials_per_candidate=2,
        start_rollout=0,
        default_cpeak=192,
    )
    assert resumed.budget_for(10) == (192, False)
    assert resumed.resume_without_history


def test_rollout_ce_batch_masks_prompt_and_padding_tokens():
    prompt_ids = torch.tensor([[0, 11, 12], [21, 22, 23]])
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    generated = [[31, 32], [33], [41, 42, 43], [44, 45]]
    ids, attention, loss_mask = build_medusa_update_batch(
        prompt_ids,
        prompt_mask,
        generated,
        repeated_generate_nums=2,
        pad_token_id=0,
    )
    assert ids.shape == attention.shape == loss_mask.shape == (4, 6)
    assert loss_mask[0].tolist() == [0, 0, 1, 1, 0, 0]
    assert loss_mask[1].tolist() == [0, 0, 1, 0, 0, 0]
    assert loss_mask[2].tolist() == [0, 0, 0, 1, 1, 1]
    assert torch.equal(loss_mask.bool() & ~attention.bool(), torch.zeros_like(loss_mask, dtype=torch.bool))


def test_online_ce_stats_sum_costs_and_average_losses():
    merged = _merge_online_ce_update_stats(
        [
            {"medusa_loss": 2.0, "head_update_time": 1.0, "head_update_tokens": 100, "head_update_steps": 2},
            {"medusa_loss": 4.0, "head_update_time": 2.0, "head_update_tokens": 200, "head_update_steps": 3},
        ]
    )
    assert merged["medusa_loss"] == 3.0
    assert merged["head_update_time"] == 3.0
    assert merged["head_update_tokens"] == 300
    assert merged["head_update_steps"] == 5
    assert merged["head_update_tokens_per_sec"] == 100.0
