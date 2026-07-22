import torch

from flashgrpo_b200.decoding.flash_medusa_decoder import FlashMedusaConfig, FlashMedusaDecoder
from flashgrpo_b200.decoding.hrdcr import (
    HRDCRFeedback,
    HRDCRPredictionBuffer,
    HRDCRStateManager,
)
from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.training.online_medusa_trainer import OnlineMedusaConfig, OnlineMedusaTrainer


class TinyTarget(torch.nn.Module):
    def __init__(self, hidden: int, vocab: int):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)


def test_sparse_support_contains_proposal_target_and_actual_token():
    vocab = hidden = 8
    target = TinyTarget(hidden, vocab)
    with torch.no_grad():
        target.lm_head.weight.copy_(torch.eye(vocab))
    manager = HRDCRStateManager(1, 1, hidden, device=torch.device("cpu"), sketch_rank=4)
    manager.states[0, 0, 2] = 1.0
    state, _ = manager.get_state_and_effective_updates([0])
    buffer = HRDCRPredictionBuffer(proposal_topk=2)
    proposal_hidden = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0, 0.0, -1.0, -2.0]])
    buffer.add_from_logits(
        sequence_ids=[0],
        anchor_positions=torch.tensor([3]),
        logits_by_horizon=[proposal_hidden],
        proposal_hidden_by_horizon=[proposal_hidden],
        candidate_topk_by_horizon=[1],
        anchor_hidden=torch.randn(1, hidden),
        fast_states=state,
        trust=manager.trust([0]),
        state_sketch=manager.sketch(state),
    )
    mature = buffer.pop_mature(torch.tensor([0]), torch.tensor([5]))
    target_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 7.0, -3.0]])
    result = HRDCRFeedback(
        target.lm_head,
        num_heads=1,
        proposal_topk=2,
        target_topk=2,
        support_cap=5,
        sketch_projection=manager.sketch_projection,
    ).compute(mature, target_logits, torch.tensor([7]), collect_auxiliary=True)
    support = set(result.auxiliary_records["support_ids"][0].tolist())
    assert {0, 1, 5, 6, 7}.issubset(support)
    assert bool(result.head_has_feedback[0, 0])
    assert float(result.record_candidate_mass[0]) < 0.5


def test_fast_state_projection_and_predictive_trust_are_exact():
    manager = HRDCRStateManager(
        1,
        1,
        4,
        device=torch.device("cpu"),
        half_life_tokens=8,
        alignment_beta=0.9,
        trust_n0=4.0,
    )
    feedback = torch.full((1, 1, 4), 100.0)
    manager.advance_token(
        [0],
        feedback,
        torch.ones((1, 1), dtype=torch.bool),
        torch.ones((1, 1)),
        torch.ones((1, 1)),
        torch.ones((1, 1), dtype=torch.bool),
    )
    state_rms = manager.states.square().mean(dim=-1).sqrt()
    assert float(state_rms.max()) <= 1.0 + 1e-6
    assert torch.allclose(manager.trust([0]), torch.tensor([[0.2]]), atol=1e-6)


def test_strict_injection_preserves_state_magnitude_and_relative_rms_cap():
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
            reflex_relative_rms_delta_base=0.02,
        ),
    )
    base = torch.randn(2, hidden)
    state = torch.full_like(base, 0.5)
    corrected = decoder._apply_reflex_correction(
        base, state, torch.ones(2), 0, 0, torch.ones(2)
    )
    ratio = (corrected - base).square().mean(dim=-1).sqrt() / base.square().mean(dim=-1).sqrt()
    assert torch.allclose(ratio, torch.full_like(ratio, 0.01), atol=2e-6)
    assert bool((ratio <= 0.02 + 1e-6).all())


def test_sparse_auxiliary_update_never_backprops_into_target():
    torch.manual_seed(4)
    hidden, vocab, count = 8, 13, 8
    target = TinyTarget(hidden, vocab)
    heads = MedusaHeads(hidden, vocab, num_heads=2, dtype=torch.float32)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=1e-3)
    trainer = OnlineMedusaTrainer(
        target,
        heads,
        optimizer,
        OnlineMedusaConfig(
            reflex_record_microbatch_size=4,
            sparse_kl_weight=0.25,
            sparse_coverage_weight=1.0,
        ),
    )
    support_ids = torch.tensor([[0, 1, 2, 3]]).repeat(count, 1)
    target_logits = torch.tensor([[0.0, 0.0, 5.0, 4.0]]).repeat(count, 1)
    records = {
        "hidden": torch.randn(count, hidden),
        "head_indices": torch.tensor([0] * 6 + [1] * 2),
        "support_ids": support_ids.to(torch.int32),
        "support_valid": torch.ones_like(support_ids, dtype=torch.bool),
        "target_logits": target_logits.to(torch.float16),
        "proposal_logits": torch.zeros(count, 4, dtype=torch.float16),
        "candidate_ids": torch.tensor([[0, 1]]).repeat(count, 1).to(torch.int32),
        "candidate_valid": torch.ones(count, 2, dtype=torch.bool),
        "candidate_mass": torch.tensor([0.1] * 6 + [0.9] * 2),
        "actual_tokens": torch.full((count,), 2, dtype=torch.long),
        "fast_state": torch.zeros(count, hidden),
        "trust": torch.zeros(count),
    }
    before_head_0 = [parameter.detach().clone() for parameter in heads.heads[0].parameters()]
    before_head_1 = [parameter.detach().clone() for parameter in heads.heads[1].parameters()]
    stats = trainer.update_sparse_online(
        records, records_per_update=8, max_heads_per_update=1, optimizer_steps=1
    )
    assert stats["aux_selected_heads"] == [1]
    assert stats["aux_optimizer_steps"] == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_head_0, heads.heads[0].parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_head_1, heads.heads[1].parameters())
    )
    assert target.lm_head.weight.grad is None
