import torch

from flashgrpo_b200.decoding.flash_medusa_decoder import FlashMedusaConfig, FlashMedusaDecoder
from flashgrpo_b200.decoding.medusa_tree import TreePlan, build_batch_trees, build_dense_tree
from flashgrpo_b200.decoding.reflex import (
    LMHeadFeedback,
    PredictionRecord,
    ReflexAuxiliaryRecordBuffer,
    ReflexStateManager,
    VerificationUtilityScheduler,
)
from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.training.flashgrpo_trainer import _merge_reflex_record_batches
from flashgrpo_b200.training.online_medusa_trainer import OnlineMedusaConfig, OnlineMedusaTrainer


def test_batched_tree_builder_matches_reference_tree():
    torch.manual_seed(7)
    batch, vocab = 5, 37
    roots = torch.randint(vocab, (batch,))
    logits = [torch.randn(batch, vocab) for _ in range(2)]
    plan = TreePlan(
        node_budget_per_seq=12,
        active_heads=2,
        topk_by_depth=[3, 2],
        actual_nodes=10,
        mode="fixed",
        layout="dense",
    )
    batched = build_batch_trees(roots, logits, plan)
    reference = [
        build_dense_tree(int(roots[row]), [head[row] for head in logits], plan)
        for row in range(batch)
    ]
    assert [(tree.tokens, tree.parents) for tree in batched] == [
        (tree.tokens, tree.parents) for tree in reference
    ]


def test_sparse_target_feedback_skips_matching_distribution():
    torch.manual_seed(11)
    vocab, hidden = 17, 8
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    logits = torch.randn(vocab)
    log_z = float(torch.logsumexp(logits, dim=-1))
    record = PredictionRecord(
        sequence_id=0,
        anchor_pos=3,
        target_pos=5,
        horizon=2,
        top_ids=torch.arange(vocab, dtype=torch.int32),
        top_logits=logits.to(torch.float16),
        logsumexp=log_z,
    )
    result = LMHeadFeedback(
        lm_head,
        target_topk=vocab,
        union_cap=vocab,
    ).compute_batch([[record]], logits.unsqueeze(0), [0])
    assert not bool(result.has_feedback.item())
    assert float(result.record_tv.item()) < 1e-3
    assert torch.count_nonzero(result.feedback) == 0


def test_sparse_teacher_can_skip_hidden_feedback_projection():
    torch.manual_seed(12)
    vocab, hidden = 19, 8
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    draft_logits = torch.randn(vocab)
    target_logits = draft_logits + 0.2 * torch.randn(vocab)
    top_values, top_ids = torch.topk(draft_logits, k=8)
    record = PredictionRecord(
        sequence_id=0,
        anchor_pos=4,
        target_pos=6,
        horizon=2,
        top_ids=top_ids.to(torch.int32),
        top_logits=top_values.to(torch.float16),
        logsumexp=float(torch.logsumexp(draft_logits, dim=-1)),
    )
    result = LMHeadFeedback(lm_head, target_topk=8, union_cap=16).compute_batch(
        [[record]],
        target_logits.unsqueeze(0),
        [int(target_logits.argmax())],
        compute_hidden_feedback=False,
    )
    assert result.feedback.shape == (1, 0)
    assert result.target_top_ids.shape == (1, 16)
    support = set(result.target_top_ids[0].tolist())
    assert set(top_ids.tolist()).issubset(support)
    assert torch.isfinite(result.record_tv).all()


def test_sparse_feedback_recovers_confident_target_mode_missing_from_draft_topk():
    vocab = hidden = 5
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(vocab))
    draft_logits = torch.tensor([5.0, 4.0, -8.0, -9.0, -10.0])
    target_logits = torch.tensor([-2.0, -2.0, 7.0, -3.0, -3.0])
    top_logits, top_ids = torch.topk(draft_logits, k=2)
    record = PredictionRecord(
        sequence_id=0,
        anchor_pos=0,
        target_pos=2,
        horizon=2,
        top_ids=top_ids.to(torch.int32),
        top_logits=top_logits.to(torch.float16),
        logsumexp=float(torch.logsumexp(draft_logits, dim=-1)),
    )
    result = LMHeadFeedback(
        lm_head,
        target_topk=2,
        union_cap=4,
        num_heads=1,
    ).compute_batch([[record]], target_logits.unsqueeze(0), [2])
    # Token 2 is absent from draft top-k. Its target probability exceeds the
    # certified q upper bound, so the hidden innovation must pull toward row 2.
    assert bool(result.head_has_feedback[0, 0])
    assert float(result.head_feedback[0, 0, 2]) > 0.0


def test_coverage_reflex_moves_target_token_above_candidate_boundary():
    vocab = hidden = 5
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(vocab))
    draft_logits = torch.tensor([5.0, 4.0, 3.0, -4.0, -5.0])
    target_logits = torch.zeros(vocab)
    top_logits, top_ids = torch.topk(draft_logits, k=3)
    record = PredictionRecord(
        sequence_id=0,
        anchor_pos=0,
        target_pos=2,
        horizon=2,
        candidate_k=2,
        top_ids=top_ids.to(torch.int32),
        top_logits=top_logits.to(torch.float16),
        logsumexp=float(torch.logsumexp(draft_logits, dim=-1)),
    )
    result = LMHeadFeedback(
        lm_head,
        target_topk=2,
        union_cap=4,
        num_heads=1,
        coverage_feedback_weight=1.0,
        feedback_objective="coverage",
    ).compute_batch(
        [[record]],
        target_logits.unsqueeze(0),
        [2],
        compute_sparse_teacher=False,
    )
    # Token 2 was outside retained top-2; token 1 was the boundary candidate.
    assert float(result.head_feedback[0, 0, 2]) > 0.0
    assert float(result.head_feedback[0, 0, 1]) < 0.0
    assert result.target_top_ids.shape[-1] == 0


def test_horizon_resolved_feedback_and_state_do_not_cancel_across_heads():
    vocab = hidden = 4
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(vocab))
    target_logits = torch.tensor([5.0, 4.0, -5.0, -5.0])
    drafts = [
        torch.tensor([1.0, 5.0, -5.0, -5.0]),
        torch.tensor([5.0, 1.0, -5.0, -5.0]),
    ]
    records = []
    for head_idx, draft in enumerate(drafts):
        values, ids = torch.topk(draft, k=vocab)
        records.append(
            PredictionRecord(
                sequence_id=0,
                anchor_pos=0,
                target_pos=2 + head_idx,
                horizon=2 + head_idx,
                top_ids=ids.to(torch.int32),
                top_logits=values.to(torch.float16),
                logsumexp=float(torch.logsumexp(draft, dim=-1)),
            )
        )
    result = LMHeadFeedback(
        lm_head,
        target_topk=vocab,
        union_cap=vocab,
        num_heads=2,
    ).compute_batch([records], target_logits.unsqueeze(0), [0])
    assert result.head_feedback.shape == (1, 2, hidden)
    assert not torch.allclose(result.head_feedback[0, 0], result.head_feedback[0, 1])

    manager = ReflexStateManager(
        1,
        hidden,
        device=torch.device("cpu"),
        num_heads=2,
        horizon_resolved=True,
        hint_cold_start=1.0,
    )
    manager.advance_token(
        [0],
        result.feedback,
        result.has_feedback,
        result.effective_mass,
        head_feedback=result.head_feedback,
        head_has_feedback=result.head_has_feedback,
        head_effective_mass=result.head_effective_mass,
    )
    state, updates = manager.get_state_and_effective_updates([0])
    assert state.shape == (1, 2, hidden)
    assert updates.shape == (1, 2)
    assert not torch.allclose(state[0, 0], state[0, 1])


def test_horizon_reflex_suppresses_a_history_hint_that_flips_direction():
    manager = ReflexStateManager(
        1,
        4,
        device=torch.device("cpu"),
        num_heads=1,
        horizon_resolved=True,
        half_life_tokens=1.0,
        hint_quality_beta=0.0,
        hint_quality_floor=0.0,
        hint_cold_start=0.25,
    )

    def update(direction: float) -> None:
        head = torch.tensor([[[direction, 0.0, 0.0, 0.0]]])
        manager.advance_token(
            [0],
            head[:, 0],
            torch.ones(1, dtype=torch.bool),
            torch.ones(1),
            head_feedback=head,
            head_has_feedback=torch.ones((1, 1), dtype=torch.bool),
            head_effective_mass=torch.ones((1, 1)),
        )

    update(1.0)
    update(1.0)
    positive_trust = float(manager._hint_trust(torch.tensor([0]))[0, 0])
    update(-1.0)
    flipped_trust = float(manager._hint_trust(torch.tensor([0]))[0, 0])
    assert positive_trust > 0.9
    assert flipped_trust == 0.0


def test_horizon_reflex_scores_the_hint_used_when_prediction_was_made():
    manager = ReflexStateManager(
        1,
        4,
        device=torch.device("cpu"),
        num_heads=1,
        horizon_resolved=True,
        half_life_tokens=1.0,
        hint_quality_beta=0.0,
        hint_quality_floor=0.0,
        hint_cold_start=1.0,
    )
    mask = torch.ones((1, 1), dtype=torch.bool)
    mass = torch.ones((1, 1))

    # The state has changed by the time delayed verification arrives.
    negative = torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]])
    manager.advance_token(
        [0],
        negative[:, 0],
        torch.ones(1, dtype=torch.bool),
        torch.ones(1),
        head_feedback=negative,
        head_has_feedback=mask,
        head_effective_mass=mass,
    )

    positive = -negative
    manager.advance_token(
        [0],
        positive[:, 0],
        torch.ones(1, dtype=torch.bool),
        torch.ones(1),
        head_feedback=positive,
        head_has_feedback=mask,
        head_effective_mass=mass,
        head_prediction_hint=positive,
        head_hint_observed=mask,
    )
    trust = float(manager._hint_trust(torch.tensor([0]))[0, 0])
    assert trust > 0.9


def test_context_addressed_reflex_retrieves_the_matching_error_memory():
    manager = ReflexStateManager(
        1,
        4,
        device=torch.device("cpu"),
        num_heads=1,
        horizon_resolved=True,
        half_life_tokens=4.0,
        context_rank=2,
        context_mix=1.0,
        context_min_mass=1e-6,
        hint_quality_floor=-1.0,
        hint_quality_temperature=0.1,
        hint_cold_start=1.0,
    )

    def update(value: torch.Tensor, key: torch.Tensor) -> None:
        head = value.view(1, 1, -1)
        manager.advance_token(
            [0],
            value.view(1, -1),
            torch.ones(1, dtype=torch.bool),
            torch.ones(1),
            head_feedback=head,
            head_has_feedback=torch.ones((1, 1), dtype=torch.bool),
            head_effective_mass=torch.ones((1, 1)),
            head_context_keys=key.view(1, 1, -1),
        )

    update(torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0]))
    update(torch.tensor([0.0, 1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0]))
    first = manager.get([0], context_keys=torch.tensor([[1.0, 0.0]]))[0, 0]
    second = manager.get([0], context_keys=torch.tensor([[0.0, 1.0]]))[0, 0]
    assert int(first.argmax()) == 0
    assert int(second.argmax()) == 1


def test_hidden_state_update_and_correction_are_rms_bounded():
    torch.manual_seed(13)
    batch, hidden, vocab = 4, 32, 41
    manager = ReflexStateManager(batch, hidden, device=torch.device("cpu"))
    feedback = torch.randn(batch, hidden)
    manager.advance_token(
        list(range(batch)),
        feedback,
        torch.ones(batch, dtype=torch.bool),
        torch.ones(batch),
    )
    assert float(manager.states.square().mean(dim=-1).sqrt().max()) <= 2.0

    heads = MedusaHeads(hidden, vocab, num_heads=2, dtype=torch.float32)
    config = FlashMedusaConfig(
        reflex_enabled=True,
        reflex_state_space="hidden",
        reflex_feedback_enabled=True,
        reflex_proposal_injection_enabled=True,
        reflex_proposal_injection_scale=1.0,
        reflex_relative_rms_delta_base=0.01,
    )
    decoder = FlashMedusaDecoder(object(), heads, object(), config)
    base_hidden = torch.randn(batch, hidden)
    corrected = decoder._apply_reflex_correction(
        base_hidden,
        manager.states,
        torch.full((batch,), 100.0),
        head_idx=0,
        generation_step=0,
    )
    correction_rms = (corrected - base_hidden).float().square().mean(dim=-1).sqrt()
    base_rms = base_hidden.float().square().mean(dim=-1).sqrt()
    assert bool((correction_rms <= 1.011 * 0.01 * base_rms).all())


def test_verification_utility_scheduler_prunes_low_value_deep_head():
    plan = TreePlan(
        node_budget_per_seq=24,
        active_heads=3,
        topk_by_depth=[3, 2, 2],
        actual_nodes=22,
        mode="fixed",
        layout="dense",
    )
    roots = torch.tensor([1, 2, 3, 4])
    logits = [torch.randn(4, 16) for _ in range(3)]
    trees = build_batch_trees(roots, logits, plan)
    scheduler = VerificationUtilityScheduler(
        3,
        warmup_rounds=2,
        min_active_heads=2,
        min_depth_acceptance=0.10,
        min_node_utility=0.02,
        exploration_interval=8,
    )
    # Head 1/2 earn accepted tokens; head 3 repeatedly verifies no useful token.
    outcomes = [[1, 5, 6], [2, 5, 6], [3, 5], [4, 5]]
    scheduler.observe(outcomes, trees)
    scheduler.observe(outcomes, trees)
    adapted, stats = scheduler.adapt(plan)
    assert adapted.active_heads == 2
    assert adapted.topk_by_depth == [3, 2]
    assert stats["last_active_heads"] == 2
    assert scheduler.to_dict()["per_head"]["3"]["node_utility_ema"] == 0.0


def test_sparse_refresh_loss_and_cache_teacher_round_trip():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    ids = torch.tensor([[0, 1]], dtype=torch.long)
    values = logits[:, :2]
    log_z = torch.logsumexp(logits, dim=-1)
    _, matching_tv = OnlineMedusaTrainer._sparse_cross_entropy_with_tail(
        logits,
        ids,
        values,
        log_z,
    )
    _, shifted_tv = OnlineMedusaTrainer._sparse_cross_entropy_with_tail(
        logits.flip(-1),
        ids,
        values,
        log_z,
    )
    assert float(matching_tv.item()) < 1e-6
    assert float(shifted_tv.item()) > float(matching_tv.item())

    buffer = ReflexAuxiliaryRecordBuffer(max_records=4)
    buffer.add_anchor_predictions(
        sequence_ids=[0],
        anchor_positions=torch.tensor([10]),
        initial_lengths=torch.tensor([5]),
        hidden_states=torch.randn(1, 8),
        fast_states=None,
        max_horizon=2,
    )
    proposal = PredictionRecord(
        sequence_id=0,
        anchor_pos=10,
        target_pos=12,
        horizon=2,
        top_ids=torch.tensor([0, 1], dtype=torch.int32),
        top_logits=torch.tensor([2.0, 1.0], dtype=torch.float16),
        logsumexp=float(log_z.item()),
    )
    buffer.pop_mature(
        0,
        12,
        generated_tokens=[3, 2, 1, 0, 1, 2, 3],
        true_token=0,
        teacher={
            "target_top_ids": torch.tensor([0, 1], dtype=torch.int32),
            "target_top_logits": torch.tensor([2.0, 1.0], dtype=torch.float16),
            "target_logsumexp": float(log_z.item()),
            "proposal_records": [proposal],
        },
    )
    batch = buffer.to_batch()
    assert bool(batch["has_sparse_teacher"].item())
    assert batch["target_top_ids"].shape == (1, 2)


def test_aux_cache_merge_pads_variable_sparse_supports():
    def records(count: int, prev_width: int, topk: int, offset: int) -> dict:
        return {
            "hidden": torch.full((count, 4), float(offset), dtype=torch.float16),
            "fast_state": torch.empty((count, 0), dtype=torch.float16),
            "labels": torch.arange(offset, offset + count),
            "horizons": torch.full((count,), 2, dtype=torch.long),
            "prev_lens": torch.full((count,), prev_width, dtype=torch.long),
            "prev_tokens": torch.full((count, prev_width), offset, dtype=torch.long),
            "target_top_ids": torch.full((count, topk), offset, dtype=torch.int32),
            "target_top_logits": torch.ones((count, topk), dtype=torch.float16),
            "target_logsumexp": torch.ones(count),
            "old_top_ids": torch.full((count, topk), offset, dtype=torch.int32),
            "old_top_logits": torch.ones((count, topk), dtype=torch.float16),
            "old_logsumexp": torch.ones(count),
            "has_sparse_teacher": torch.ones(count, dtype=torch.bool),
        }

    merged = _merge_reflex_record_batches(
        [records(2, 1, 2, 0), records(3, 2, 4, 10)],
        max_records=4,
    )
    assert merged["hidden"].shape == (4, 4)
    assert merged["prev_tokens"].shape == (4, 2)
    assert merged["target_top_ids"].shape == (4, 4)
    assert merged["old_top_ids"].shape == (4, 4)
    # The cap keeps the newest records after padding/concatenation.
    assert merged["labels"].tolist() == [1, 10, 11, 12]


def test_anchor_online_refresh_optimizes_candidate_coverage_without_target_gradients():
    torch.manual_seed(17)
    count, hidden, vocab, support_k = 6, 16, 29, 8

    class TinyTarget(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
            self.embedding = torch.nn.Embedding(vocab, hidden)

        def get_input_embeddings(self):
            return self.embedding

    target = TinyTarget()
    target.requires_grad_(False)
    heads = MedusaHeads(
        hidden,
        vocab,
        num_heads=2,
        tie_lm_head=True,
        lm_head=target.lm_head,
        anchor_conditioning_enabled=True,
        anchor_bottleneck_ratio=4,
    )
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-2)
    trainer = OnlineMedusaTrainer(
        target,
        heads,
        optimizer,
        OnlineMedusaConfig(
            reflex_record_microbatch_size=count,
            refresh_distill_weight=0.0,
            refresh_tv_weight=0.0,
            refresh_coverage_weight=1.0,
            refresh_hard_token_weight=0.0,
            refresh_proximal_weight=0.0,
            anchor_conditioning_enabled=True,
            acceptance_candidate_topk_by_head=(3, 2),
        ),
    )
    teacher = torch.randn(count, vocab)
    target_values, target_ids = torch.topk(teacher, k=support_k, dim=-1)
    records = {
        "hidden": torch.randn(count, hidden, dtype=torch.float16),
        "fast_state": torch.empty(count, 0, dtype=torch.float16),
        "labels": teacher.argmax(dim=-1),
        "horizons": torch.full((count,), 2, dtype=torch.long),
        "reflex_scale": torch.ones(count),
        "prev_tokens": torch.randint(0, vocab, (count, 1)),
        "target_top_ids": target_ids.to(torch.int32),
        "target_top_logits": target_values.to(torch.float16),
        "target_logsumexp": torch.logsumexp(teacher, dim=-1),
        "old_top_ids": torch.empty(count, 0, dtype=torch.int32),
        "old_top_logits": torch.empty(count, 0, dtype=torch.float16),
        "old_logsumexp": torch.zeros(count),
        "has_sparse_teacher": torch.ones(count, dtype=torch.bool),
    }
    before = [projection.weight.detach().clone() for projection in heads.anchor_conditioner.up]
    stats = trainer.update_reflex_records(records)
    assert stats["head_update_steps"] == 1
    assert "head_1_candidate_mass" in stats
    assert any(not torch.equal(old, projection.weight) for old, projection in zip(before, heads.anchor_conditioner.up))
    assert target.lm_head.weight.grad is None
    assert target.embedding.weight.grad is None
