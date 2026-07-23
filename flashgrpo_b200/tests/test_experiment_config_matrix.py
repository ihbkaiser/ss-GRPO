from pathlib import Path

from flashgrpo_b200.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "flashgrpo_b200" / "configs"
MODELS = {
    "qwen25_1p5b": ("qwen2", "Qwen2.5-1.5B-Instruct"),
    "qwen25_3b": ("qwen2", "Qwen2.5-3B-Instruct"),
    "qwen3_4b": ("qwen3", "Qwen3-4B"),
    "qwen25_7b": ("qwen2", "Qwen2.5-7B-Instruct"),
    "llama31_8b": ("llama", "Llama-3.1-8B-Instruct"),
    "qwen25_14b": ("qwen2", "Qwen2.5-14B-Instruct"),
}
EXPECTED_HRDCR = {
    "qwen25_1p5b": {"batch": 16, "accumulation": 2, "half_life": 48},
    "qwen25_3b": {"batch": 8, "accumulation": 4, "half_life": 48},
    "qwen3_4b": {"batch": 8, "accumulation": 4, "half_life": 32},
    "qwen25_7b": {"batch": 8, "accumulation": 4, "half_life": 48},
    "llama31_8b": {"batch": 8, "accumulation": 4, "half_life": 32},
    "qwen25_14b": {"batch": 4, "accumulation": 8, "half_life": 64},
}
FAIR_FIELDS = (
    ("data", "train_option"),
    ("generation", "max_length"),
    ("generation", "max_prompt_length"),
    ("generation", "repeated_generate_nums"),
    ("training", "batch_size"),
    ("training", "accumulation_steps"),
    ("training", "max_training_token"),
    ("training", "max_training_padding_gap"),
    ("training", "train_data_fraction"),
    ("flashgrpo", "acceptance"),
    ("flashgrpo", "cpeak_nodes"),
    ("flashgrpo", "max_tree_nodes_per_seq"),
)


def test_model_config_matrix_and_fair_ablation():
    for model_key, (model_type, model_tail) in MODELS.items():
        main = load_config(CONFIG_ROOT / model_key / "train_hrdcr.yaml")
        normalized = load_config(CONFIG_ROOT / model_key / "train_hrdcr_normalized.yaml")
        ablation = load_config(CONFIG_ROOT / model_key / "train_medusa_only.yaml")
        pretrain = load_config(CONFIG_ROOT / model_key / "pretrain.yaml")

        expected_heads = f"outputs/pretrain/{model_key}"
        assert main["model"]["model_type"] == model_type
        assert main["model"]["model_dir"].endswith(model_tail)
        assert main["flashgrpo"]["medusa_heads_checkpoint"] == expected_heads
        assert main["aux_head_checkpoint"] == expected_heads
        assert pretrain["output_dir"] == expected_heads

        assert main["reflex"]["enabled"] is True
        assert main["reflex"]["horizon_resolved"] is True
        assert main["reflex"]["strict_horizon_pipeline"] is True
        assert normalized["reflex"]["injection_gate_mode"] == "normalized"
        assert normalized["reflex"]["feedback_objective"] == "distribution"
        assert normalized["reflex"]["feedback_stride_min"] == 1
        expected = EXPECTED_HRDCR[model_key]
        assert normalized["reflex"]["half_life_tokens"] == expected["half_life"]
        assert normalized["reflex"]["relative_rms_delta_base"] == 0.02
        assert normalized["reflex"]["warmup_effective_updates"] == 0.0
        assert main["training"]["batch_size"] == expected["batch"]
        assert main["training"]["accumulation_steps"] == expected["accumulation"]
        assert expected["batch"] * expected["accumulation"] == 32
        # The measured MEDUSA-only baseline retains feedback instrumentation
        # but never injects m_t or updates the pretrained proposal heads.
        assert ablation["reflex"]["proposal_injection_enabled"] is False
        assert ablation["reflex"]["strict_horizon_pipeline"] is True
        assert main["flashgrpo"]["medusa_update_mode"] == "sparse_online"
        assert ablation["flashgrpo"]["medusa_update_mode"] == "none"
        assert ablation["flashgrpo"]["online_medusa"] is False
        assert ablation["aux_update"]["learning_rate"] == 0.0
        assert normalized["aux_update"] == main["aux_update"]
        for section, field in FAIR_FIELDS:
            assert main[section][field] == ablation[section][field]
            assert normalized[section][field] == main[section][field]
