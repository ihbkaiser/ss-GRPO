
# FastGRPO

[FastGRPO](https://arxiv.org/abs/2509.21792) is an **adaptive speculative decoding framework** for Group Relative Policy Optimization (GRPO) that dynamically adjusts drafting and verification strategies based on real-time concurrency levels. The framework addresses the prohibitively slow training process of GRPO by maximizing acceleration of the generation phase while maintaining reasoning capabilities.

## 📋 Overview

The key innovations of this project include:

1. **Adaptive Speculative Decoding**: Dynamically adjusts drafting and verification strategy based on real-time concurrency levels, maximizing the acceleration of the generation process
2. **Joint Draft Model Training**: Mitigates performance degradation caused by distributional drift between the evolving target model and draft model through continuous adaptation using feedback from the target model
3. **Significant Speedup**: Achieves an end-to-end speedup of 2.35× to 2.72× compared to baseline approaches

### Core Components

1. **`train_draft.py`**: Pre-training script for the draft model
   - Used for initial draft model training
   - Can be used standalone for draft model preparation

2. **`grpo_speculative.py`**: Main training script for the GRPO speculative decoding framework
   - Jointly trains target and draft models
   - Implements speculative decoding during training
   - Accelerates GRPO training process without performance loss



## 🛠️ Prerequisites

Before starting training, please complete the following setup steps:

### Step 1: Environment Setup
Install the required dependencies via the provided `requirement.txt`:
```bash
pip install -r requirement.txt
```

### Step 2: Dataset Preparation
Download and place the training dataset under the `data/` directory.  
Ensure the data is properly formatted and accessible. Example:
```
data/
├── simplelr_abel_level3to5
├── gsm8k
└── ...
```


## 🚀 Usage

### GRPO Speculative Training (Joint Training)

Launch the joint training process for both target and draft models:

```bash
python train_draft.py \
    --model_dir <path_to_pretrained_model> \
    --version_name <your_experiment_name> \
    --model_type qwen2 \
    --batch_size 1 \
    --num_epochs 10 \
    --lr 5e-5 \
    --accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --sample_num 100 \
    --log_dir <path_to_training_log_dir> \
    --saved_model_dir <dir_to_save_model_checkpoints> \
    --dataset_dir <dir_to_dataset>
```

```bash
python grpo_speculative.py \
    --model_dir <path_to_target_model> \                                  
    --adapter_path <path_to_pretrained_draft_adapter> \                  
    --load_lora_path <path_to_resume_checkpoint_or_empty> \               
    --model_type qwen2 \                                      
    --train_option simplelr_abel_level3to5 \                          
    --version_name debug \                            
    --batch_size 4 \
    --num_epochs 10 \
    --sample_num 100 \
    --accumulation_steps 4 \
    --draft_accumulation_steps 1 \
    --target_lr 1e-6 \
    --draft_lr 1e-4 \
    --is_train_draft True \
    --temperature 1.0 \
    --top_p 0.95 \
    --max_length 2048 \
    --max_training_padding_gap 256 \
    --max_training_token 3072 \
    --grpo_iteration_num 1 \
    --repeated_generate_nums 8 \
    --beta 0.04 \
    --epsilon 0.1 \
    --log_file <path_to_save_training_log> \                                           
    --saved_model_dir <dir_to_save_target_model_checkpoints> \           
    --saved_draft_model_dir <dir_to_save_draft_model_checkpoints> \       
    --saved_statistics_dir <dir_to_save_generation_length_stats> \       
```


### Speculative Generate Function Parameters

The speculative_generate function is the core function of our project. The following is an introduction to its parameters:

#### Basic Parameters
| Parameter | Type | Description |
|----------|------|-------------|
| `input_ids` | tensor | Input token IDs for the generation process (shape: [batch_size, seq_len]) |
| `attention_mask` | tensor | Attention mask to indicate which tokens are valid (shape: [batch_size, seq_len]) |
| `tokenizer` | tokenizer | The tokenizer associated with the model for encoding/decoding tokens |

#### Sampling Parameters
| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `do_sample` | bool | `False` | Whether to use sampling (`True`) or greedy decoding (`False`) |
| `temperature` | float | `0.8` | Sampling temperature for controlling randomness |
| `top_p` | float | `0.9` | Top-p (nucleus) sampling threshold |
| `top_k` | int | `None` | Top-k sampling parameter |

#### Adaptive Control Parameters
| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `verification_capacity` | int | `160` | Maximum capacity for verification tokens |
| `max_draft_token_length` | int | `5` | Maximum length of draft tokens to generate |
| `max_draft_k` | int | `8` | Maximum branching factor for draft tree |
| `max_verification_num` | int | `160` | Maximum number of tokens to verify |
| `min_draft_token_length` | int | `3` | Minimum length of draft tokens |
| `draft_token_length_c` | float | 0.75 | A parameter that affects the tuning of the draft token length and should be set based on the capability of the draft model; the stronger the draft model, the smaller this value should be |
#### Output Control Parameters
| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `repeated_generate_nums` | int | `None` | Number of repeated generations for each input |
| `statistical_time` | bool | `True` | Whether to collect timing statistics |
| `return_all_draft_input` | bool | `False` | Whether to return all draft inputs |
| `max_length` | int | `2048` | Maximum length of generated sequences |


## Reference

```
@misc{zhang2025fastgrpoacceleratingpolicyoptimization,
      title={FastGRPO: Accelerating Policy Optimization via Concurrency-aware Speculative Decoding and Online Draft Learning}, 
      author={Yizhou Zhang and Ning Lv and Teng Wang and Jisheng Dang},
      year={2025},
      eprint={2509.21792},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21792}, 
}
```
