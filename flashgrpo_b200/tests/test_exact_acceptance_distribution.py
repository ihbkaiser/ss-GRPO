import torch

from flashgrpo_b200.decoding.acceptance import exact_accept_paths_batch, logits_to_probs, sample_from_logits
from flashgrpo_b200.decoding.medusa_tree import CandidateTree
from flashgrpo_b200.decoding.acceptance import exact_accept_path


def test_root_only_sampling_matches_target_sampler():
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.5]])
    torch.manual_seed(123)
    vanilla = sample_from_logits(logits, do_sample=True, temperature=1.0, top_p=0.95)
    torch.manual_seed(123)
    flash_root = sample_from_logits(logits, do_sample=True, temperature=1.0, top_p=0.95)
    assert int(vanilla.item()) == int(flash_root.item())


def test_fast_nucleus_sampler_matches_reference_distribution():
    cases = [
        # The top-2 shortlist covers the nucleus.
        (torch.tensor([[5.0, 4.0, 0.0, -1.0]]), 0.85),
        # The top-2 shortlist is insufficient and exercises exact fallback.
        (torch.zeros(1, 4), 0.75),
    ]
    for case_idx, (logits, top_p) in enumerate(cases):
        expected = logits_to_probs(logits, temperature=0.8, top_p=top_p, top_k=None)[0]
        rows = logits.expand(50_000, -1)
        torch.manual_seed(321 + case_idx)
        samples = sample_from_logits(
            rows,
            do_sample=True,
            temperature=0.8,
            top_p=top_p,
            top_k=None,
            nucleus_topk_hint=2,
        )
        observed = torch.bincount(samples, minlength=logits.shape[-1]).float() / samples.numel()
        assert torch.max(torch.abs(observed - expected)) < 0.015


def test_exact_accepts_only_target_sampled_child():
    tree = CandidateTree(tokens=[5, 7, 9], parents=[-1, 0, 0], depths=[1, 2, 2])
    logits = torch.full((3, 16), -10.0)
    logits[0, 7] = 10.0
    accepted, nodes, _, correction = exact_accept_path(
        tree,
        logits,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
    )
    assert accepted == [5, 7]
    assert nodes == [0, 1]
    assert correction is None


def test_target_mismatch_is_returned_as_correction_token():
    tree = CandidateTree(tokens=[5, 7], parents=[-1, 0], depths=[1, 2])
    logits = torch.full((2, 16), -10.0)
    logits[0, 9] = 10.0
    accepted, nodes, _, correction = exact_accept_path(
        tree,
        logits,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
    )
    assert accepted == [5]
    assert nodes == [0]
    assert correction == 9


def test_stochastic_match_or_correction_preserves_target_distribution():
    sample_count = 50_000
    tree = CandidateTree(tokens=[2, 0], parents=[-1, 0], depths=[1, 2])
    trees = [tree] * sample_count
    root_logits = torch.log(torch.tensor([0.6, 0.4]))
    tree_logits = root_logits.view(1, 1, 2).expand(sample_count, 2, 2).clone()

    torch.manual_seed(456)
    accepted, _, _, corrections = exact_accept_paths_batch(
        trees,
        tree_logits,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
    )
    emitted = [tokens[1] if len(tokens) > 1 else correction for tokens, correction in zip(accepted, corrections)]
    assert all(token is not None for token in emitted)
    observed = torch.bincount(torch.tensor(emitted), minlength=2).float() / sample_count
    assert torch.max(torch.abs(observed - torch.tensor([0.6, 0.4]))) < 0.015


def test_packed_internal_logits_match_dense_acceptance():
    trees = [
        CandidateTree(tokens=[5, 7, 9, 11], parents=[-1, 0, 0, 1], depths=[1, 2, 2, 3]),
        CandidateTree(tokens=[4, 6, 8, 10], parents=[-1, 0, 0, 2], depths=[1, 2, 2, 3]),
    ]
    dense = torch.full((2, 4, 16), -10.0)
    dense[0, 0, 7] = 10.0
    dense[0, 1, 11] = 10.0
    dense[1, 0, 8] = 10.0
    dense[1, 2, 10] = 10.0
    packed = torch.stack([dense[0, 0], dense[0, 1], dense[1, 0], dense[1, 2]])
    slots = torch.tensor([[0, 1, -1, -1], [2, -1, 3, -1]])

    dense_result = exact_accept_paths_batch(
        trees,
        dense,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
    )
    packed_result = exact_accept_paths_batch(
        trees,
        packed,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        node_to_logit=slots,
    )
    assert packed_result == dense_result

    packed_cpu_result = exact_accept_paths_batch(
        trees,
        packed,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        node_to_logit_cpu=slots.tolist(),
    )
    assert packed_cpu_result == dense_result
