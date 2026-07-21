from pathlib import Path

from flashgrpo_b200.utils.config import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "flashgrpo_b200" / "configs"
MODELS = {
    "qwen25_1p5b": ("qwen2", "Qwen2.5-1.5B-Instruct"),
    "qwen3_4b": ("qwen3", "Qwen3-4B"),
    "qwen25_7b": ("qwen2", "Qwen2.5-7B-Instruct"),
    "llama31_8b": ("llama", "Llama-3.1-8B-Instruct"),
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
        assert ablation["reflex"]["enabled"] is False
        assert main["aux_update"] == ablation["aux_update"]
        for section, field in FAIR_FIELDS:
            assert main[section][field] == ablation[section][field]

