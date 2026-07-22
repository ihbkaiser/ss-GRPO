import torch

from flashgrpo_b200.decoding.flash_medusa_decoder import FlashMedusaConfig, FlashMedusaDecoder
from flashgrpo_b200.decoding.hrdcr import (
    HRDCRFeedback,
    HRDCRPredictionBuffer,
    HRDCRStateManager,
    merge_auxiliary_records,
)
from flashgrpo_b200.decoding.medusa_tree import (
    Head3QualityCalibrator,
    build_batch_trees,
    candidate_sets_by_head,
    plan_tree,
)
from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.training.online_medusa_trainer import OnlineMedusaConfig, OnlineMedusaTrainer


class TinyTarget(torch.nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)


def sparse_plan():
    return plan_tree(
        active_batch_size=1,
        num_medusa_heads=3,
        tree_mode="fixed",
        tree_layout="sparse_asymmetric",
        cpeak_nodes=10,
        min_tree_nodes_per_seq=1,
        max_tree_nodes_per_seq=10,
        max_tree_depth=4,
        fixed_tree_topk_by_depth=[4, 3, 2],
        sparse_nodes_by_head=[4, 3, 2],
        sparse_min_head3_nodes=1,
        sparse_head3_min_budget=8,
    )


def proposal_logits(batch=1, vocab=17):
    base = torch.arange(vocab, dtype=torch.float32).flip(0)
    return [base.repeat(batch, 1) - idx * 0.1 for idx in range(3)]


def test_sparse_tree_budget_parentage_and_head3_invariants():
    plan = sparse_plan()
    trees = build_batch_trees(
        torch.tensor([7]),
        proposal_logits(),
        plan,
        head3_gate_mask=torch.tensor([True]),
    )
    tree = trees[0]
    assert plan.active_heads == 3
    assert tree.node_count == 10
    assert tree.nodes_by_head[:3] == [4, 3, 2]
    assert len({(parent, token) for parent, token in zip(tree.parents, tree.tokens)}) == 10
    for node, (parent, depth) in enumerate(zip(tree.parents, tree.depths)):
        if node == 0:
            assert parent == -1
        else:
            assert 0 <= parent < node
            assert tree.depths[parent] == depth - 1
    assert all(tree.depths[parent] == 2 for parent, depth in zip(tree.parents, tree.depths) if depth == 3)
    assert all(tree.depths[parent] == 3 for parent, depth in zip(tree.parents, tree.depths) if depth == 4)


def test_head3_gate_reallocates_budget_and_exploration_prevents_starvation():
    plan = sparse_plan()
    rejected = build_batch_trees(
        torch.tensor([7]),
        proposal_logits(),
        plan,
        head3_gate_mask=torch.tensor([False]),
    )[0]
    assert rejected.node_count == 10
    assert rejected.nodes_by_head + [0] == [4, 5, 0]

    calibrator = Head3QualityCalibrator(
        exploration_fraction=1.0,
        min_calibration_records=1024,
        node_cost=2.0,
    )
    gate = calibrator.select(proposal_logits(batch=2)[2], eligible=torch.ones(2, dtype=torch.bool))
    assert bool(gate.exploration_mask.all())
    assert bool(gate.gate_mask.all())


def test_head3_future_position_and_actual_sparse_candidates_are_recorded():
    plan = sparse_plan()
    logits = proposal_logits()
    trees = build_batch_trees(
        torch.tensor([7]), logits, plan, head3_gate_mask=torch.tensor([True])
    )
    candidate_ids, candidate_valid = candidate_sets_by_head(
        trees, 3, device=torch.device("cpu"), widths=[4, 3, 2]
    )
    manager = HRDCRStateManager(
        1,
        3,
        17,
        device=torch.device("cpu"),
        min_effective_updates=0,
        min_alignment_count=0,
        min_state_rms=0,
    )
    state, _ = manager.get_state_and_effective_updates([0])
    buffer = HRDCRPredictionBuffer(proposal_topk=4)
    hidden = [row.clone() for row in logits]
    buffer.add_from_logits(
        sequence_ids=[0],
        anchor_positions=torch.tensor([5]),
        logits_by_horizon=logits,
        proposal_hidden_by_horizon=hidden,
        candidate_topk_by_horizon=[4, 3, 2],
        candidate_ids_by_horizon=candidate_ids,
        candidate_valid_by_horizon=candidate_valid,
        quality_by_horizon=[torch.zeros(1), torch.zeros(1), torch.ones(1)],
        anchor_hidden=torch.zeros(1, 17),
        fast_states=state,
        trust=torch.ones(1, 3),
        state_sketch=manager.sketch(state),
    )
    mature = buffer.pop_mature(torch.tensor([0]), torch.tensor([9]))
    assert mature.count == 1
    assert int(mature.head_indices[0]) == 2
    assert int(mature.candidate_valid[0].sum()) >= 1


def test_candidate_regret_uses_same_k_and_counterfactual_probe_matures():
    hidden = vocab = 8
    target = TinyTarget(hidden, vocab)
    with torch.no_grad():
        target.lm_head.weight.copy_(torch.eye(vocab))
    manager = HRDCRStateManager(1, 1, hidden, device=torch.device("cpu"), sketch_rank=4)
    state, _ = manager.get_state_and_effective_updates([0])
    corrected_hidden = torch.tensor([[4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    raw_hidden = torch.tensor([[0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    buffer = HRDCRPredictionBuffer(proposal_topk=2)
    buffer.add_from_logits(
        sequence_ids=[0],
        anchor_positions=torch.tensor([2]),
        logits_by_horizon=[corrected_hidden],
        proposal_hidden_by_horizon=[corrected_hidden],
        candidate_topk_by_horizon=[2],
        anchor_hidden=torch.zeros(1, hidden),
        fast_states=state,
        trust=torch.ones(1, 1),
        state_sketch=manager.sketch(state),
        candidate_ids_by_horizon=[torch.tensor([[0, 1]])],
        candidate_valid_by_horizon=[torch.ones(1, 2, dtype=torch.bool)],
        probe_rows=torch.tensor([0]),
        raw_proposal_hidden_by_horizon=[raw_hidden],
        raw_candidate_ids_by_horizon=[torch.tensor([[2, 3]])],
        raw_candidate_valid_by_horizon=[torch.ones(1, 2, dtype=torch.bool)],
    )
    mature = buffer.pop_mature(torch.tensor([0]), torch.tensor([4]))
    target_logits = torch.tensor([[0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0]])
    result = HRDCRFeedback(
        target.lm_head,
        num_heads=1,
        proposal_topk=2,
        target_topk=2,
        support_cap=7,
        sketch_projection=manager.sketch_projection,
    ).compute(mature, target_logits, torch.tensor([2]), collect_auxiliary=True)
    assert float(result.record_candidate_regret[0]) > 0.8
    assert int(result.probe_head_indices.numel()) == 1
    assert bool(result.probe_changed[0])
    assert bool(result.probe_losses[0])
    assert "candidate_regret" in result.auxiliary_records
    stored_support = result.auxiliary_records["support_ids"]
    stored_valid = result.auxiliary_records["support_valid"]
    assert bool(stored_support.masked_select(~stored_valid).eq(-1).all())


def test_reflex_trust_region_is_zero_when_inactive_and_exact_when_active():
    hidden = 16
    heads = MedusaHeads(hidden, 19, num_heads=1, dtype=torch.float32)
    decoder = FlashMedusaDecoder(
        object(),
        heads,
        object(),
        FlashMedusaConfig(
            reflex_enabled=True,
            reflex_state_space="hidden",
            reflex_strict_horizon_pipeline=True,
            reflex_feedback_enabled=True,
            reflex_proposal_injection_enabled=True,
            reflex_proposal_injection_scale=1.0,
            reflex_min_effective_updates=4,
            reflex_min_state_rms=0.005,
            reflex_correction_ratio_min=0.005,
            reflex_correction_ratio_max=0.020,
        ),
    )
    base = torch.randn(3, hidden)
    state = torch.randn(3, hidden)
    inactive = decoder._apply_reflex_correction(
        base, state, torch.zeros(3), 0, 0, torch.ones(3)
    )
    assert torch.equal(inactive, base)
    active = decoder._apply_reflex_correction(
        base, state, torch.full((3,), 4.0), 0, 0, torch.full((3,), 0.5)
    )
    delta = active - base
    ratio = delta.square().mean(dim=-1).sqrt() / base.square().mean(dim=-1).sqrt()
    assert torch.isfinite(active).all()
    assert torch.allclose(ratio, torch.full_like(ratio, 0.0125), atol=2e-6)
    assert bool(((ratio >= 0.005) & (ratio <= 0.020)).all())


def test_balanced_auxiliary_sampler_does_not_starve_head3():
    torch.manual_seed(3)
    hidden, vocab, per_head = 8, 13, 4
    count = per_head * 3
    target = TinyTarget(hidden, vocab)
    heads = MedusaHeads(hidden, vocab, num_heads=3, dtype=torch.float32)
    trainer = OnlineMedusaTrainer(
        target,
        heads,
        torch.optim.AdamW(heads.parameters(), lr=1e-3),
        OnlineMedusaConfig(reflex_record_microbatch_size=4),
    )
    support = torch.tensor([[0, 1, 2, 3, vocab]]).repeat(count, 1)
    support_valid = torch.ones_like(support, dtype=torch.bool)
    support_valid[:, -1] = False
    records = {
        "hidden": torch.randn(count, hidden),
        "head_indices": torch.arange(3).repeat_interleave(per_head),
        "support_ids": support.to(torch.int32),
        "support_valid": support_valid,
        "target_logits": torch.tensor([[0.0, 0.0, 5.0, 4.0, 0.0]])
        .repeat(count, 1)
        .half(),
        "proposal_logits": torch.zeros(count, 5).half(),
        "candidate_ids": torch.tensor([[0, 1]]).repeat(count, 1).to(torch.int32),
        "candidate_valid": torch.ones(count, 2, dtype=torch.bool),
        "candidate_mass": torch.full((count,), 0.2),
        "candidate_regret": torch.full((count,), 0.5),
        "restricted_kl": torch.ones(count),
        "candidate_hit": torch.zeros(count, dtype=torch.bool),
        "actual_tokens": torch.full((count,), 2),
        "fast_state": torch.randn(count, hidden),
        "trust": torch.ones(count),
    }
    stats = trainer.update_sparse_online(
        records,
        records_per_update=count,
        max_heads_per_update=3,
        optimizer_steps=1,
        min_records_per_selected_head=per_head,
        head_sampling_weights=[1.0, 1.0, 1.25],
    )
    assert 3 in stats["aux_selected_heads"]
    assert stats["head3_aux_records_used"] >= per_head
    assert stats["head3_aux_optimizer_steps"] == 1


def test_merge_auxiliary_records_pads_variable_sparse_widths():
    def records(count: int, support_width: int, candidate_width: int, head_idx: int):
        support_ids = torch.arange(support_width).repeat(count, 1).to(torch.int32)
        support_valid = torch.ones_like(support_ids, dtype=torch.bool)
        if head_idx == 1:
            support_ids[:, -1] = 1000
            support_valid[:, -1] = False
        candidate_ids = torch.arange(candidate_width).repeat(count, 1).to(torch.int32)
        return {
            "hidden": torch.randn(count, 8),
            "head_indices": torch.full((count,), head_idx, dtype=torch.long),
            "support_ids": support_ids,
            "support_valid": support_valid,
            "target_logits": torch.randn(count, support_width, dtype=torch.float16),
            "proposal_logits": torch.randn(count, support_width, dtype=torch.float16),
            "candidate_ids": candidate_ids,
            "candidate_valid": torch.ones_like(candidate_ids, dtype=torch.bool),
            "candidate_mass": torch.rand(count),
            "candidate_regret": torch.rand(count),
            "restricted_kl": torch.rand(count),
            "candidate_hit": torch.zeros(count, dtype=torch.bool),
            "actual_tokens": torch.zeros(count, dtype=torch.long),
            "fast_state": torch.randn(count, 8),
            "trust": torch.rand(count),
            "quality": torch.rand(count),
        }

    merged = merge_auxiliary_records(
        [records(2, 37, 4, 0), records(3, 36, 2, 1)],
        max_records=0,
    )

    assert merged["hidden"].shape[0] == 5
    assert merged["support_ids"].shape == (5, 37)
    assert merged["target_logits"].shape == (5, 37)
    assert merged["proposal_logits"].shape == (5, 37)
    assert merged["candidate_ids"].shape == (5, 4)
    assert bool(merged["support_ids"][2:, -1].eq(-1).all())
    assert not bool(merged["support_valid"][2:, -1].any())
    assert bool(merged["target_logits"][2:, -1].eq(0).all())
    assert bool(merged["proposal_logits"][2:, -1].eq(0).all())
    assert bool(merged["candidate_ids"][2:, 2:].eq(-1).all())
    assert not bool(merged["candidate_valid"][2:, 2:].any())
