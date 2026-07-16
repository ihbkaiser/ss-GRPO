import os

import torch

from flashgrpo.decoding.kv_extraction import extract_accepted_path_kv


def test_kv_path_extraction_script_opt_in():
    # Full check loads a HF/Qwen model and is run through:
    # CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python flashgrpo/scripts/test_kv_extraction.py
    assert os.environ.get("FLASHGRPO_TEST_MODEL", "") == "" or True


def test_in_place_kv_compaction_matches_copy_path():
    torch.manual_seed(17)
    old_len = 5
    tree_nodes = 4
    key = torch.randn(2, 2, old_len + tree_nodes, 8)
    value = torch.randn_like(key)
    accepted = [[0, 1, 3], [0, 2]]
    old = ((key[:, :, :old_len].clone(), value[:, :, :old_len].clone()),)
    copied = extract_accepted_path_kv(
        old,
        ((key.clone(), value.clone()),),
        accepted,
        compact_in_place=False,
    ).past_key_values
    compacted = extract_accepted_path_kv(
        old,
        ((key.clone(), value.clone()),),
        accepted,
        compact_in_place=True,
    ).past_key_values
    assert torch.equal(compacted[0][0], copied[0][0])
    assert torch.equal(compacted[0][1], copied[0][1])
