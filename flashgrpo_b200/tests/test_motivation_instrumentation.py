from flashgrpo_b200.decoding.flash_medusa_decoder import FlashMedusaConfig, FlashMedusaDecoder
from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.training.flashgrpo_trainer import _merge_generation_outputs


def _decoder(mode: str) -> FlashMedusaDecoder:
    heads = MedusaHeads(8, 16, num_heads=2)
    return FlashMedusaDecoder(
        object(),
        heads,
        object(),
        FlashMedusaConfig(
            reflex_enabled=True,
            reflex_state_space="hidden",
            reflex_feedback_enabled=True,
            reflex_proposal_injection_enabled=True,
            reflex_proposal_injection_scale=1.0,
            reflex_adaptation_mode=mode,
        ),
    )


def test_delayed_collects_reflex_without_injecting_it() -> None:
    delayed = _decoder("delayed")
    immediate = _decoder("immediate")
    assert delayed._reflex_enabled()
    assert delayed._reflex_effective_injection_scale(0) == 0.0
    assert immediate._reflex_effective_injection_scale(0) == 1.0


def test_motivation_trace_sequence_ids_survive_microbatch_merge() -> None:
    base = {
        "generated_token_ids": [[1, 2]],
        "motivation_trace": [{"sequence_id": 0, "window_index": 0}],
        "total_time_cost": 1.0,
        "reflex_metrics": {},
    }
    merged = _merge_generation_outputs([base, base])
    assert [row["sequence_id"] for row in merged["motivation_trace"]] == [0, 1]
