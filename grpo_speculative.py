import os
import pandas as pd
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,GenerationConfig
from helper.modeling_draft import Model
from helper.rewards import accuracy_reward_func , format_reward_func
from helper.get_QAs import get_test_QAs , get_train_QAs
from helper.specualtive_generate import speculative_generate
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
import signal
import sys
import torch
from copy import deepcopy
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
from datetime import datetime
import argparse 
from statistics import mean , stdev
import pickle
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings(
    "ignore",
    message="equations=True in NormalizationConfig is deprecated.*",
)


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.2f}m"
    return f"{seconds / 3600:.2f}h"

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


parser = argparse.ArgumentParser(description="Training configuration")

parser.add_argument('--model_dir',type=str)
parser.add_argument('--adapter_path',type=str)
parser.add_argument('--temperature',type=float,default=1.0)
parser.add_argument('--top_p',type=float,default=0.95)
parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps for target model')
parser.add_argument('--draft_accumulation_steps', type=int, default=1, help='Gradient accumulation steps for draft model')
parser.add_argument('--target_lr', type=float, default=1e-6, help='Learning rate for target model')
parser.add_argument('--draft_lr', type=float, default=1e-4, help='Learning rate for draft model')
parser.add_argument('--is_train_draft', type=lambda x: x.lower() == 'true', default=True, help='Whether to train the draft model (True/False)')
parser.add_argument('--model_type', type=str, default='Qwen2___5-Math-7B', help='Version name for saving checkpoints')
parser.add_argument('--draft_num_hidden_layers', type=int, default=1)
parser.add_argument('--train_option',type=str,default="simplelr_abel_level3to5")
parser.add_argument('--load_lora_path',type=str,default="")
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--version_name',type=str,default='normal')
parser.add_argument('--num_epochs',type=int,default=10)
parser.add_argument('--sample_num',type=int,default=100)
parser.add_argument('--grpo_iteration_num',type=int,default=1)
parser.add_argument('--repeated_generate_nums',type=int,default=8)
parser.add_argument('--beta',type=float,default=0.01)
parser.add_argument('--epsilon',type=float,default=0.1)
parser.add_argument('--max_length',type=int,default=2048)
parser.add_argument('--max_training_padding_gap',type=int,default=256)
parser.add_argument('--max_training_token',type=int,default=3072)
parser.add_argument('--logps_chunk_size', type=int, default=256,
                    help='Sequence chunk size for token-logprob computation. Lower values reduce peak VRAM.')
parser.add_argument('--statistical_time', type=lambda x: x.lower() == 'true', default=False,
                    help='Enable exact CUDA timing. False is faster because it avoids frequent cuda synchronize calls.')
parser.add_argument('--verification_capacity', type=int, default=160,
                    help='Concurrency-aware verification capacity. Keep 160 for speed; lower only if generation itself OOMs.')
parser.add_argument('--max_draft_token_length', type=int, default=5)
parser.add_argument('--max_draft_k', type=int, default=8)
parser.add_argument('--max_verification_num', type=int, default=160)
parser.add_argument('--min_draft_token_length', type=int, default=3)
parser.add_argument('--draft_token_length_c', type=float, default=0.75)
parser.add_argument('--log_file', type=str, required=True,
                    help="Full path to training log file, e.g., /path/to/train.log")
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help="Directory to save trained target adapter/model checkpoints")
parser.add_argument('--saved_draft_model_dir', type=str, required=True,
                    help="Directory to save trained draft model checkpoints")
parser.add_argument('--saved_statistics_dir', type=str, required=True,
                    help="Directory to save statistics of generated sequence lengths.")
args = parser.parse_args()
num_epochs=args.num_epochs
sample_num=args.sample_num
grpo_iteration_num=args.grpo_iteration_num
repeated_generate_nums=args.repeated_generate_nums
beta=args.beta
epsilon=args.epsilon
max_length=args.max_length
max_training_padding_gap=args.max_training_padding_gap
max_training_token=args.max_training_token
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
draft_accumulation_steps = args.draft_accumulation_steps
target_lr = args.target_lr
draft_lr = args.draft_lr
is_train_draft = args.is_train_draft
model_type = args.model_type
draft_num_hidden_layers = args.draft_num_hidden_layers
model_dir = args.model_dir
adapter_path = args.adapter_path
temperature = args.temperature
top_p = args.top_p
version_name = args.version_name
log_file = args.log_file
saved_model_dir = args.saved_model_dir
saved_draft_model_dir = args.saved_draft_model_dir
saved_statistics_dir = args.saved_statistics_dir
logps_chunk_size = max(1, args.logps_chunk_size)
statistical_time = args.statistical_time
verification_capacity = args.verification_capacity
max_draft_token_length = args.max_draft_token_length
max_draft_k = args.max_draft_k
max_verification_num = args.max_verification_num
min_draft_token_length = args.min_draft_token_length
draft_token_length_c = args.draft_token_length_c

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
if not os.path.exists(saved_draft_model_dir):
    os.makedirs(saved_draft_model_dir)
if not os.path.exists(saved_statistics_dir):
    os.makedirs(saved_statistics_dir)


print(datetime.now())
print(model_type,os.getenv('CUDA_VISIBLE_DEVICES'))
print("=" * 60)
print("Training & Generation Configuration")
print("=" * 60)
print(f"Model: {model_type} | Version: {version_name}")
print(f"Path: model={model_dir}, adapter={adapter_path}")
print(f"Train: epochs={num_epochs}, batch={batch_size}, "
      f"acc_steps={accumulation_steps}, draft_acc_steps={draft_accumulation_steps}")
print(f"LR: target={target_lr}, draft={draft_lr} | "
      f"Seq: max_len={max_length}, max_tokens={max_training_token}, pad_gap={max_training_padding_gap}")
print(f"Gen: temp={temperature}, top_p={top_p}"
      f"beta={beta}, epsilon={epsilon}")
print(f"Speed/VRAM: statistical_time={statistical_time}, logps_chunk_size={logps_chunk_size}, "
      f"verification_capacity={verification_capacity}, max_verification_num={max_verification_num}, "
      f"draft_len=[{min_draft_token_length},{max_draft_token_length}], max_draft_k={max_draft_k}")
print(f"Draft: train={is_train_draft}")
print(f"Draft layers: {draft_num_hidden_layers}")
print(f"Iteration: grpo_iter={grpo_iteration_num}, sample={sample_num}, "
      f"repeat_gen={repeated_generate_nums}")
print("=" * 60)


config=AutoConfig.from_pretrained(model_dir)
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype='auto',config=config).cuda()
target_model.eval()

draft_config=deepcopy(config)
draft_config.rope_scaling=None
draft_config.num_hidden_layers=draft_num_hidden_layers
model=Model(draft_config,target_model=target_model)
model.load_model(adapter_path)
print(adapter_path)
model=model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side="left")


if config.model_type == 'llama':
    tokenizer.pad_token = "<|end_of_text|>" 
    tokenizer.pad_token_id = 128001
    

QAs = get_train_QAs(args.train_option)
df = pd.DataFrame(QAs)

for param in model.draft_model.parameters():
    param.requires_grad=True

for param in model.target_model.parameters():
    param.requires_grad=False
for param in model.lm_head.parameters():
    param.requires_grad=False
for param in model.embed_tokens.parameters():
    param.requires_grad=False
    

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          
    r=64,                           
    lora_alpha=32,                
    lora_dropout=0.0,              
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model.target_model = get_peft_model(model.target_model,lora_config)
if  args.load_lora_path != "":
    model.target_model.load_adapter(args.load_lora_path,adapter_name="default")
model.target_model.print_trainable_parameters()

def _get_base_causal_lm(causal_lm):
    """Return the underlying AutoModelForCausalLM, while preserving PEFT/LoRA modules."""
    if hasattr(causal_lm, "get_base_model"):
        return causal_lm.get_base_model()
    if hasattr(causal_lm, "base_model") and hasattr(causal_lm.base_model, "model"):
        return causal_lm.base_model.model
    return causal_lm


def _autocast_dtype(model):
    dtype = getattr(model, "dtype", None)
    if dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16


def _token_logps_from_hidden(hidden_states, lm_head, labels, chunk_size):
    """
    Compute log p(labels[t+1] | tokens[:t]) without materializing [B, T, vocab]
    for the whole sequence. Returned tensor has shape [B, T-1].
    """
    hidden_states = hidden_states[:, :-1, :]
    labels = labels[:, 1:].to(hidden_states.device)
    seq_len = hidden_states.shape[1]
    logps_chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = lm_head(hidden_states[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        selected_logits = torch.gather(logits, dim=-1, index=cur_labels.unsqueeze(-1)).squeeze(-1)
        log_denominator = torch.logsumexp(logits, dim=-1)
        logps_chunks.append(selected_logits - log_denominator)
        del logits, selected_logits, log_denominator

    return torch.cat(logps_chunks, dim=1) if logps_chunks else hidden_states.new_zeros((hidden_states.shape[0], 0))


def compute_model_token_logps(causal_lm, input_ids, attention_mask, chunk_size):
    """Forward backbone once, then apply the LM head in chunks to reduce peak VRAM."""
    base_model = _get_base_causal_lm(causal_lm)
    device = input_ids.device
    device_type = "cuda" if device.type == "cuda" else device.type

    with torch.amp.autocast(device_type, dtype=_autocast_dtype(base_model), enabled=(device.type == "cuda")):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    return _token_logps_from_hidden(hidden_states, base_model.lm_head, input_ids, chunk_size)


def compute_target_loss_and_backward(
    model,
    input_ids,
    attention_mask,
    mask,
    reward,
    epsilon,
    beta,
    grpo_iteration,
    old_logps=None,
    ref_logps=None,
    chunk_size=256,
    loss_scale=1.0,
):
    """
    Memory-safe GRPO loss.

    The original code built full-vocabulary logits for policy and reference and then
    called log_softmax over [batch, seq, vocab]. That is the direct source of the
    1-2GB transient allocations in the OOM trace. This function stores only token
    log-probabilities [batch, seq] for old/ref, and backpropagates policy chunks
    so full-vocab tensors are released chunk-by-chunk.
    """
    device = input_ids.device
    token_mask = mask[:, :-1].to(device=device, dtype=torch.float32)
    denom = token_mask.sum(-1).clamp_min(1.0)
    reward = reward.to(device=device, dtype=torch.float32)
    seq_len = token_mask.shape[1]

    if grpo_iteration == 0:
        model.target_model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps_gpu = compute_model_token_logps(model.target_model, input_ids, attention_mask, chunk_size).detach()
        model.target_model.enable_adapter_layers()
        ref_logps_for_loss = ref_logps_gpu
        old_logps_for_loss = None
    else:
        if old_logps is None or ref_logps is None:
            raise ValueError("old_logps and ref_logps are required when grpo_iteration > 0")
        old_logps_for_loss = old_logps.to(device, non_blocking=True)
        ref_logps_for_loss = ref_logps.to(device, non_blocking=True)

    # Policy forward with adapters enabled. We avoid model(...).logits because it
    # materializes the full [B, T, V] tensor.
    model.target_model.enable_adapter_layers()
    base_model = _get_base_causal_lm(model.target_model)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(device_type, dtype=_autocast_dtype(base_model), enabled=(device.type == "cuda")):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    policy_hidden = hidden_states[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    old_chunks_to_store = []
    loss_value = 0.0
    abs_loss1_value = 0.0
    loss2_value = 0.0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = base_model.lm_head(policy_hidden[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        logps = torch.gather(logits, dim=-1, index=cur_labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        cur_mask = token_mask[:, start:end]

        if grpo_iteration == 0:
            cur_old_logps = logps.detach()
            old_chunks_to_store.append(cur_old_logps.detach().cpu())
        else:
            cur_old_logps = old_logps_for_loss[:, start:end]

        cur_ref_logps = ref_logps_for_loss[:, start:end]

        coef1 = torch.exp(logps - cur_old_logps)
        coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
        loss1 = torch.min(coef1 * reward, coef2 * reward)

        coef3 = cur_ref_logps - logps
        kl = torch.exp(coef3) - coef3 - 1
        token_loss = -(loss1 - beta * kl)

        # Match the original per-sequence normalization, but accumulate by chunk.
        chunk_loss = ((token_loss * cur_mask).sum(-1) / denom).sum()
        scaled_loss = chunk_loss * loss_scale
        retain = end < seq_len
        scaled_loss.backward(retain_graph=retain)

        with torch.no_grad():
            loss_value += float(chunk_loss.detach().cpu())
            abs_loss1_value += float(torch.abs((loss1 * cur_mask).sum(-1) / denom).sum().detach().cpu())
            loss2_value += float(((kl * cur_mask).sum(-1) / denom).sum().detach().cpu())

        del logits, logps, coef1, coef2, loss1, coef3, kl, token_loss, chunk_loss, scaled_loss

    if grpo_iteration == 0:
        old_logps_out = torch.cat(old_chunks_to_store, dim=1) if old_chunks_to_store else torch.empty((input_ids.shape[0], 0))
        ref_logps_out = ref_logps_for_loss.detach().cpu()
    else:
        old_logps_out = old_logps
        ref_logps_out = ref_logps

    del hidden_states, policy_hidden
    if grpo_iteration == 0:
        del ref_logps_gpu, ref_logps_for_loss
    else:
        del old_logps_for_loss, ref_logps_for_loss

    return loss_value, abs_loss1_value, loss2_value, old_logps_out, ref_logps_out


def training_draft_model(model,outputs,prompt_mask):
    

    all_draft_input_states = outputs['all_draft_input_states']
    all_draft_input_ids = outputs['all_draft_input_ids']
    all_prompt_length = [prompt_mask[idx // repeated_generate_nums].sum().item() for idx in range(len(all_draft_input_states))]
    
    prompt_mask=prompt_mask.cpu()
    device=model.target_model.device
    
    sorted_pairs = sorted(
        zip(all_draft_input_ids, all_draft_input_states, all_prompt_length),
        key=lambda x: len(x[0]),
        reverse=False  
    )

    all_draft_input_ids_sorted, all_draft_input_states_sorted, all_prompt_length_sorted = zip(*sorted_pairs)

    all_draft_input_ids = list(all_draft_input_ids_sorted)
    all_draft_input_states = list(all_draft_input_states_sorted)
    all_prompt_length = list(all_prompt_length_sorted)
    
    l1_loss=torch.nn.SmoothL1Loss(reduction='none')
    total_loss1,total_loss2=0,0
    
    draft_input_states_list=[]
    draft_input_ids_list=[]
    prompt_length_list=[]
    
    cur_max_length=0
    hidden_size=all_draft_input_states[0].shape[-1]
    
    for idx , (draft_input_states,draft_input_ids,prompt_length) in enumerate(zip(all_draft_input_states,all_draft_input_ids,all_prompt_length)):
        
        if ((draft_input_ids.shape[-1]*(len(draft_input_states_list)+1)<=max_training_token*2 and
            (draft_input_ids.shape[-1]-cur_max_length)*len(draft_input_states_list)<=max_training_padding_gap) or
            len(draft_input_states_list)==0):
            
                draft_input_states_list.append(draft_input_states)
                draft_input_ids_list.append(draft_input_ids)
                prompt_length_list.append(prompt_length)
                
                cur_max_length=max(cur_max_length, draft_input_ids.shape[-1])
            
        else:
            
            cur_batch=len(draft_input_states_list)

            loss_mask=[[] for _ in range(cur_batch)]
            attention_mask=[[] for _ in range(cur_batch)]
            
            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
                attention_mask[idx_seq]=[1]*cur_len

            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    draft_input_states_list[idx_seq]=torch.concat(
                        [draft_input_states_list[idx_seq],
                        torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                        dim=-2)
                    
                    draft_input_ids_list[idx_seq]=torch.concat(
                        [draft_input_ids_list[idx_seq],
                        torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                        dim=-1)
                    
                    loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
                    attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
            
            draft_input_states=torch.stack(draft_input_states_list,dim=0)
            draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
            loss_mask=torch.tensor(loss_mask,device=device)
            attention_mask=torch.tensor(attention_mask,device=device)

            with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
                draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                                attention_mask=attention_mask,use_cache=False)
                
            next_feature_states=draft_outputs['next_feature_states']
            draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
            draft_logits=model.lm_head(draft_hidden_states)
            
            with torch.no_grad():
                target_hidden_states=draft_input_states
                target_logits=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))
                target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
                
            loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

            loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
            loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss1=loss1.sum(-1)
            loss1=loss1*2.0
            
            draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)

            plogp=target_logits*torch.log(draft_logits)
            loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
            loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss2= - loss2.sum(-1)

            loss2=loss2*0.1
            
            loss=loss1+loss2
                
            total_loss1+=loss1.item()
            total_loss2+=loss2.item()
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                
                loss = loss.detach()
                del loss
                torch.cuda.empty_cache()
            else:

                loss=loss/len(all_draft_input_states)
                loss=loss/draft_accumulation_steps
                loss.backward()
                
            draft_input_states_list=[all_draft_input_states[idx]]
            draft_input_ids_list=[all_draft_input_ids[idx]]
            prompt_length_list=[all_prompt_length[idx]]
            cur_max_length=all_draft_input_ids[idx].shape[-1]
            
    cur_batch=len(draft_input_states_list)

    loss_mask=[[] for _ in range(cur_batch)]
    attention_mask=[[] for _ in range(cur_batch)]
    
    cur_max_length=0
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
        attention_mask[idx_seq]=[1]*cur_len
        
        cur_max_length=max(cur_max_length, cur_len)
        
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        padding_len=cur_max_length-cur_len
        
        if padding_len>0:
            draft_input_states_list[idx_seq]=torch.concat(
                [draft_input_states_list[idx_seq],
                torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                dim=-2)
            
            draft_input_ids_list[idx_seq]=torch.concat(
                [draft_input_ids_list[idx_seq],
                torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                dim=-1)
            
            loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
            attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
    
    draft_input_states=torch.stack(draft_input_states_list,dim=0)
    draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
    loss_mask=torch.tensor(loss_mask,device=device)
    attention_mask=torch.tensor(attention_mask,device=device)
    
    with torch.amp.autocast(str(model.target_model.device),
                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                        attention_mask=attention_mask,use_cache=False)
        
    next_feature_states=draft_outputs['next_feature_states']
    draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
    draft_logits=model.lm_head(draft_hidden_states)
    
    with torch.no_grad():
        target_hidden_states=draft_input_states
        target_logits=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))
        target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
        
    loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

    loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
    loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss1=loss1.sum(-1)
    loss1=loss1*2.0
    
    draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)

    plogp=target_logits*torch.log(draft_logits)
    loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
    loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss2= - loss2.sum(-1)

    loss2=loss2*0.1
    
    loss=loss1+loss2
        
    total_loss1+=loss1.item()
    total_loss2+=loss2.item()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        
        loss = loss.detach()
        del loss
        torch.cuda.empty_cache()
    else:

        loss=loss/len(all_draft_input_states)
        loss.backward()
            
        
    total_loss1/=len(all_draft_input_states)
    total_loss2/=len(all_draft_input_states)
    
    return total_loss1,total_loss2

        
optimizer_target = torch.optim.AdamW(model.target_model.parameters(), lr=target_lr)
optimizer_draft = torch.optim.AdamW(model.draft_model.parameters(), lr=draft_lr)

with open(log_file,'w',encoding='utf-8') as f:
    pass

step=0
used_items=0
used_items_at_last_update=0
draft_step=0
draft_accumulated_step=0 
batch_logs=[]
batch_data={
    'messages':[],
    'rewards':[],
    'std_rewards':[],
    'generate_time_cost':0,
    'last_generate_time_cost':[],
    'train_time_cost':0,
    'last_train_time_cost':[],
    'generate_length':0,
    'last_generate_length':[],
    'total_acc_length':0,
    'last_acc_length':[],
    'total_decoded_token_num':0,
    'last_decoded_token_num':[],
    'prefill_time_cost':0,
    'target_time_cost':0,
    'draft_time_cost':0,
    'check_time_cost':0,
    'ignore_due_correct':0,
    'ignore_due_incorrect':0,
    'mean_rewards':0,
    'last_mean_rewards':[],
    'draft_train_time_cost':0,
    'last_draft_loss1':[],
    'last_draft_loss2':[] ,
    'generate_length_list':[] 
}

optimizer_target.zero_grad(set_to_none=True)
optimizer_draft.zero_grad(set_to_none=True)
start_time=time.time()
batch=[]

class TrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch):
        system_prompt = "You are a math problem assistant." 
        user_prompt =  '''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
        messages = []
        answers = []

        for example in batch:
            messages.append([
                {"role" : "system" , "content": system_prompt} , 
                {"role" : "user" , "content": user_prompt.format_map({"instruction" : example['question']}) }
            ])
            answers.append(example['answer'])
        tokenized_inputs = self.tokenizer(
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
            return_tensors='pt',padding='longest',truncation=True,max_length=4096,padding_side='left'         
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'messages': messages,        
            'answers': answers,           
        }

dataloader=DataLoader(QAs,collate_fn=TrainDataCollator(tokenizer=tokenizer),num_workers=4,
                    persistent_workers=True,batch_size=batch_size,shuffle=True,drop_last=False)

epoch_bar = tqdm(range(num_epochs), desc="Epoch", dynamic_ncols=True)
for epoch in epoch_bar:
    
    batch_data['ignore_due_correct']=0
    batch_data['ignore_due_incorrect']=0
    batch_data['length_stdev'] = []
    batch_data['length_range'] = []
    batch_data['length_cv'] = []
    
    batch_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        dynamic_ncols=True,
        leave=False,
    )
    for i,batch in batch_bar:
        batch_time_start = time.time()
        
        if batch['input_ids'].shape[-1]>=max_length:
            batch_bar.set_postfix(
                skip="max_length",
                input_len=batch['input_ids'].shape[-1],
                used=used_items,
                refresh=False,
            )
            batch=[]
            continue
        
        if None in batch['answers']:
            batch_bar.set_postfix(skip="none_answer", used=used_items, refresh=False)
            batch=[]
            continue
        
        input_ids=batch['input_ids'].to('cuda')
        attention_mask=batch['attention_mask'].to('cuda')
        messages=batch['messages']
        answers=batch['answers']
        
        with torch.inference_mode():
            outputs=speculative_generate(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                do_sample=True,
                max_length=max_length,
                repeated_generate_nums=repeated_generate_nums,
                temperature=temperature,
                top_p=top_p,
                return_all_draft_input=True,
                statistical_time=statistical_time,
                verification_capacity=verification_capacity,
                max_draft_token_length=max_draft_token_length,
                max_draft_k=max_draft_k,
                max_verification_num=max_verification_num,
                min_draft_token_length=min_draft_token_length,
                draft_token_length_c=draft_token_length_c,
            )
        
        prompt_length=input_ids.shape[-1]
        outputs['prompt_length']=prompt_length
        
        outputs['decoded_sequences']=[tokenizer.decode(x,skip_special_tokens=True) for x in outputs['generated_token_ids']]
        token_ids_length = [len(item) for item in outputs['generated_token_ids'] ]
        length_stdev = stdev(token_ids_length)
        length_range = max(token_ids_length) - min(token_ids_length)
        length_cv = length_stdev / mean(token_ids_length) 
        length_ave = mean(token_ids_length) 
        batch_data['generate_length_list'].extend(token_ids_length)
        batch_bar.set_postfix(
            phase="generated",
            gen=format_duration(outputs['total_time_cost']),
            mean_len=f"{length_ave:.1f}",
            acc=f"{outputs['total_acc_length'] / max(outputs['total_decoded_token_num'], 1):.3f}",
            used=used_items,
            refresh=False,
        )
        
        if is_train_draft:
            if statistical_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_train_time_start=time.time()
            draft_loss1,draft_loss2=training_draft_model(model,outputs,attention_mask)
            if statistical_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            draft_train_elapsed=time.time()-draft_train_time_start
            batch_data['draft_train_time_cost']+=draft_train_elapsed
            batch_data['last_draft_loss1'].append(draft_loss1)
            batch_data['last_draft_loss2'].append(draft_loss2)
            draft_accumulated_step += 1
            if is_train_draft and draft_accumulated_step % draft_accumulation_steps == 0:
                optimizer_draft.step() 
                optimizer_draft.zero_grad(set_to_none=True)
                draft_step += 1
            batch_bar.set_postfix(
                phase="draft_train",
                gen=format_duration(outputs['total_time_cost']),
                draft=format_duration(draft_train_elapsed),
                dloss1=f"{draft_loss1:.4f}",
                dloss2=f"{draft_loss2:.4f}",
                used=used_items,
                refresh=False,
            )
    
        if draft_step % 1024 == 0 and step > 0 and is_train_draft:
            with open(f"{saved_statistics_dir}/{step}.pkl","wb") as f:
                pickle.dump(batch_data['generate_length_list'],f)
        
        generate_length=0
        for idx_batch in range(len(answers)):
            generate_length += outputs['max_sequence_length']
            decoded_sequences_for_prompt=[]
            new_messages=[]
            ground_truth=answers[idx_batch]
            for idx_k in range(repeated_generate_nums):
                idx_sequence=idx_batch*repeated_generate_nums+idx_k
                decoded_sequence=outputs['decoded_sequences'][idx_sequence]
                decoded_sequences_for_prompt.append(decoded_sequence)

                new_message=deepcopy(messages[idx_batch])
                new_message.append({
                    "role": "assistant",
                    "content": decoded_sequence
                })
                new_messages.append(new_message)

            format_rewards = format_reward_func(decoded_sequences_for_prompt)
            answer_rewards = accuracy_reward_func(decoded_sequences_for_prompt, [ground_truth] * repeated_generate_nums)
            rewards = np.array([0.2 * f + a for f, a in zip(format_rewards, answer_rewards)])
            if rewards.std()==0:
                
                if rewards[0]>=1.0:
                    batch_data['ignore_due_correct']+=1
                else:
                    batch_data['ignore_due_incorrect']+=1
                    
                continue
            
            std_rewards=(rewards-rewards.mean())/rewards.std()
            batch_data['messages']+=new_messages
            batch_data['rewards']+=rewards.tolist()
            batch_data['std_rewards']+=std_rewards.tolist()
            used_items+=1
            
        generate_length /= len(answers)
        
        batch_data['length_stdev'].append(length_stdev)
        batch_data['length_range'].append(length_range)
        batch_data['length_cv'].append(length_cv)
        batch_data['last_generate_time_cost'].append(outputs['total_time_cost'])
        batch_data['last_acc_length'].append(outputs['total_acc_length'])
        batch_data['last_decoded_token_num'].append(outputs['total_decoded_token_num'])
        batch_data['last_generate_length'].append(generate_length)
        batch_data['prefill_time_cost']+=outputs['prefill_time_cost']
        batch_data['target_time_cost']+=outputs['target_time_cost']
        batch_data['draft_time_cost']+=outputs['draft_time_cost']
        batch_data['check_time_cost']+=outputs['check_time_cost']
        
        batch_data['generate_time_cost']+=outputs['total_time_cost']
        batch_data['total_acc_length']+=outputs['total_acc_length']
        batch_data['total_decoded_token_num']+=outputs['total_decoded_token_num']
        batch_data['generate_length']+=generate_length
        batch=[]

        if len(batch_data['messages']) == 0:
            batch_bar.set_postfix(
                phase="no_reward_variance",
                gen=format_duration(outputs['total_time_cost']),
                ignored=batch_data['ignore_due_correct'] + batch_data['ignore_due_incorrect'],
                used=used_items,
                refresh=False,
            )
            continue

        pending_used_items = used_items - used_items_at_last_update
        required_used_items = max(1, batch_size * accumulation_steps)
        if pending_used_items < required_used_items:
            batch_bar.set_postfix(
                phase="accumulating_rollouts",
                pending=f"{pending_used_items}/{required_used_items}",
                gen=format_duration(outputs['total_time_cost']),
                ignored=batch_data['ignore_due_correct'] + batch_data['ignore_due_incorrect'],
                used=used_items,
                refresh=False,
            )
            continue
        
        text=tokenizer.apply_chat_template(batch_data['messages'],tokenize=False,add_generation_prompt=False)
        text=tokenizer(text,padding=False)
        loss_mask=[]
        
        for idx_message, message in enumerate(batch_data['messages']):
            prompt_text=tokenizer.apply_chat_template(message[:-1],tokenize=False,add_generation_prompt=True)
            prompt_text=tokenizer.encode(prompt_text)
            cur_loss_mask=[0]*(len(prompt_text)-1)+[1]*(len(text.input_ids[idx_message])-len(prompt_text)+1)
            loss_mask.append(cur_loss_mask)
            
        input_ids=text.input_ids
        attention_mask=text.attention_mask
        
        sorted_pairs = sorted(
            zip(input_ids, attention_mask, loss_mask),
            key=lambda x: len(x[0]),
            reverse=False   
        )

        input_ids_sorted, attention_mask_sorted, loss_mask_sorted = zip(*sorted_pairs)

        input_ids, attention_mask, loss_mask = list(input_ids_sorted), list(attention_mask_sorted), list(loss_mask_sorted)

        step = used_items // (batch_size * accumulation_steps)  
        batch_old_logps=[]
        batch_ref_logps=[]
        
        grpo_bar = tqdm(
            range(grpo_iteration_num),
            desc="GRPO",
            dynamic_ncols=True,
            leave=False,
        )
        for grpo_iteration in grpo_bar:
            if statistical_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time_start=time.time()
            
            cur_max_length=0
            device=model.target_model.device
            microbatch_index = 0
            
            cur_input_ids=[]
            cur_attention_mask=[]
            cur_loss_mask=[]
            cur_rewards=[]
            
            for j in range(len(batch_data['messages'])):
                
                if ((max(cur_max_length, len(input_ids[j])) * (len(cur_input_ids)+1)<=max_training_token and
                    (len(input_ids[j])-cur_max_length)*len(cur_input_ids)<=max_training_padding_gap) or
                    len(cur_input_ids)==0):
                    cur_max_length=max(cur_max_length, len(input_ids[j]))
                    
                    cur_input_ids.append(input_ids[j])
                    cur_attention_mask.append(attention_mask[j])
                    cur_loss_mask.append(loss_mask[j])
                    cur_rewards.append(batch_data['std_rewards'][j])
                    
                else:
                    
                    cur_batch=len(cur_input_ids)
                    for idx_seq in range(cur_batch):
                        
                        cur_len=len(cur_input_ids[idx_seq])
                        padding_len=cur_max_length-cur_len
                        
                        if padding_len>0:
                            
                            cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                            cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                            cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len
                            
                    cur_input_ids=torch.tensor(cur_input_ids, device=device)
                    cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
                    cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
                    cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

                    old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
                    ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
                    loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
                        model,
                        cur_input_ids,
                        cur_attention_mask,
                        cur_loss_mask,
                        cur_rewards,
                        epsilon,
                        beta,
                        grpo_iteration,
                        old_logps=old_logps,
                        ref_logps=ref_logps,
                        chunk_size=logps_chunk_size,
                        loss_scale=1.0 / max(len(batch_data['messages']), 1),
                    )
                        
                    if grpo_iteration==0:
                        batch_old_logps.append(old_logps)
                        batch_ref_logps.append(ref_logps)
                    microbatch_index += 1
                    
                    del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards

                    cur_input_ids=[input_ids[j]]
                    cur_attention_mask=[attention_mask[j]]
                    cur_loss_mask=[loss_mask[j]]
                    cur_rewards=[batch_data['std_rewards'][j]]
                    
                    cur_max_length=len(input_ids[j])
                    
            cur_batch=len(cur_input_ids)
            for idx_seq in range(cur_batch):
                
                cur_len=len(cur_input_ids[idx_seq])
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    
                    cur_input_ids[idx_seq]=cur_input_ids[idx_seq]+[0]*padding_len
                    cur_loss_mask[idx_seq]=cur_loss_mask[idx_seq]+[0]*padding_len
                    cur_attention_mask[idx_seq]=cur_attention_mask[idx_seq]+[0]*padding_len
                    
            cur_input_ids=torch.tensor(cur_input_ids, device=device)
            cur_attention_mask=torch.tensor(cur_attention_mask, device=device)
            cur_loss_mask=torch.tensor(cur_loss_mask, device=device)
            cur_rewards=torch.tensor(cur_rewards, device=device).unsqueeze(-1)

            old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
            ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
            loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
                model,
                cur_input_ids,
                cur_attention_mask,
                cur_loss_mask,
                cur_rewards,
                epsilon,
                beta,
                grpo_iteration,
                old_logps=old_logps,
                ref_logps=ref_logps,
                chunk_size=logps_chunk_size,
                loss_scale=1.0 / max(len(batch_data['messages']), 1),
            )
                
            if grpo_iteration==0:
                batch_old_logps.append(old_logps)
                batch_ref_logps.append(ref_logps)
            microbatch_index += 1
            del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards

                
            optimizer_target.step()
            optimizer_target.zero_grad(set_to_none=True)
            
            if statistical_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time_elapsed=time.time()-train_time_start
            batch_data['last_train_time_cost'].append(train_time_elapsed)
            batch_data['train_time_cost']+=train_time_elapsed
            batch_data['last_mean_rewards'].append(sum(batch_data['rewards'])/len(batch_data['rewards']))
            batch_data['mean_rewards']+=sum(batch_data['rewards'])/len(batch_data['rewards'])
            
            real_sample_num=sample_num*accumulation_steps
            recent_train_times = batch_data['last_train_time_cost'][-real_sample_num:]
            recent_acc_nums = batch_data['last_acc_length'][-real_sample_num:]
            recent_decoded_nums = batch_data['last_decoded_token_num'][-real_sample_num:]
            recent_rewards = batch_data['last_mean_rewards'][-real_sample_num:]
            recent_lengths = batch_data['last_generate_length'][-real_sample_num:]
            recent_draft_loss1 = batch_data['last_draft_loss1'][-real_sample_num:]
            recent_draft_loss2 = batch_data['last_draft_loss2'][-real_sample_num:]
            
            avg_logs = {
                "epoch":epoch+1,
                "step": step,
                "used_items" : used_items,
                "pending_used_items": pending_used_items,
                f"length_range" : round(mean(batch_data['length_range']),4),
                f"length_cv" : round(mean(batch_data['length_cv']),4),
                f"length_stdev" : round(mean(batch_data['length_stdev']),4),  
                "grpo_iteration":grpo_iteration+1,
                "used_time": round((time.time()-start_time)/60, 3),
                f"last_{sample_num}_generate_time_cost":round(sum(batch_data['last_generate_time_cost'][-real_sample_num:])/60,3),
                f"last_{sample_num}_train_time_cost": round(sum(recent_train_times) / 60, 3) if recent_train_times else 0,
                f"last_{sample_num}_acc_length":round(sum(recent_acc_nums) / max(sum(recent_decoded_nums), 1),4),
                f"last_{sample_num}_mean_rewards": round(sum(recent_rewards) / max(len(recent_rewards), 1), 3),
                f"last_{sample_num}_mean_length": round(sum(recent_lengths) / max(len(recent_lengths), 1), 3),
                
                "ignore_due_correct_cur_epoch":batch_data['ignore_due_correct'],
                "ignore_due_incorrect_cur_epoch":batch_data['ignore_due_incorrect'],                                
                "generate_time_cost":round(batch_data['generate_time_cost']/60,3),
                "average_acc_length":round(batch_data['total_acc_length']/max(batch_data['total_decoded_token_num'], 1),4),
                "prefill_time_cost":round(batch_data['prefill_time_cost']/60,3),
                "target_time_cost":round(batch_data['target_time_cost']/60,3),
                "draft_time_cost":round(batch_data['draft_time_cost']/60,3),
                "train_time_cost":round(batch_data['train_time_cost']/60,3),
                "check_time_cost":round(batch_data['check_time_cost']/60,3),
                "mean_reward":round(batch_data['mean_rewards']/max(used_items, 1),4),
                
                "draft_train_time_cost":round(batch_data['draft_train_time_cost']/60,3) if is_train_draft else 0, 
                f"last_{sample_num}_draft_loss1":round(sum(recent_draft_loss1)/max(len(recent_draft_loss1), 1),4) if is_train_draft and recent_draft_loss1 else 0,
                f"last_{sample_num}_draft_loss2":round(sum(recent_draft_loss2)/max(len(recent_draft_loss2), 1),4) if is_train_draft and recent_draft_loss2 else 0 
            }

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(avg_logs) + '\n')
                
            torch.cuda.empty_cache()
            grpo_bar.set_postfix(
                step=step,
                train=format_duration(train_time_elapsed),
                reward=avg_logs[f"last_{sample_num}_mean_rewards"],
                refresh=False,
            )
            batch_bar.set_postfix(
                phase="target_train",
                step=step,
                used=used_items,
                gen=format_duration(outputs['total_time_cost']),
                train=format_duration(train_time_elapsed),
                batch=format_duration(time.time() - batch_time_start),
                reward=avg_logs[f"last_{sample_num}_mean_rewards"],
                ignored=batch_data['ignore_due_correct'] + batch_data['ignore_due_incorrect'],
                refresh=False,
            )
            
        used_items_at_last_update = used_items
        batch_data['messages'].clear()
        batch_data['rewards'].clear()
        batch_data['std_rewards'].clear()
        batch_old_logps.clear()
        batch_ref_logps.clear()

        if step%500==0 and step!=0:
            model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
            model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')
    epoch_bar.set_postfix(
        step=step,
        used=used_items,
        elapsed=format_duration(time.time() - start_time),
        gen=format_duration(batch_data['generate_time_cost']),
        train=format_duration(batch_data['train_time_cost']),
        refresh=False,
    )
            

model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')   
