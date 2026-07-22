from pathlib import Path

from flashgrpo_b200.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
MAIN_CONFIG = ROOT / "flashgrpo_b200/configs/qwen25_7b/train_hrdcr.yaml"
ABLATION_CONFIG = ROOT / "flashgrpo_b200/configs/qwen25_7b/train_medusa_only.yaml"


def _differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            result.extend(_differences(left.get(key), right.get(key), child))
        return result
    return [] if left == right else [prefix]


def test_head_only_ablation_changes_only_reflex_switches_and_identity():
    main = load_config(MAIN_CONFIG)
    ablation = load_config(ABLATION_CONFIG)
    allowed = {
        "method",
        "run_name",
        "reflex.proposal_injection_enabled",
    }
    assert set(_differences(main, ablation)) == allowed
    assert main["training"]["train_data_fraction"] == 0.4
    assert ablation["training"]["train_data_fraction"] == 0.4
