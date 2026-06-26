import os
import pandas as pd
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,GenerationConfig
from helper.modeling_draft import Model
from helper.rewards import accuracy_reward_func , format_reward_func
from helper.get_QAs import get_test_QAs , get_train_QAs
from helper.specualtive_generate import speculative_generate
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch import nn
import time
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
import signal
import shutil
import sys
import torch
from copy import deepcopy
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
from datetime import datetime
import argparse
from statistics import mean , stdev
import pickle
from tqdm import tqdm

try:
    from distributed_utils import (
        average_gradients,
        barrier,
        cleanup_distributed,
        make_zero_loss,
        rank0_print,
        reduce_max,
        reduce_sum,
        setup_distributed,
    )
except ImportError:
    from helper.distributed_utils import (
        average_gradients,
        barrier,
        cleanup_distributed,
        make_zero_loss,
        rank0_print,
        reduce_max,
        reduce_sum,
        setup_distributed,
    )

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
parser.add_argument('--log_file', type=str, required=True,
                    help="Full path to training log file, e.g., /path/to/train.log")
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help="Directory to save trained target adapter/model checkpoints")
parser.add_argument('--saved_draft_model_dir', type=str, required=True,
                    help="Directory to save trained draft model checkpoints")
parser.add_argument('--saved_statistics_dir', type=str, required=True,
                    help="Directory to save statistics of generated sequence lengths.")
parser.add_argument('--seed', type=int, default=13, help='Base random seed. Each distributed rank uses seed + rank.')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers per process.')
parser.add_argument('--checkpoint_parts_per_epoch', type=int, default=8,
                    help='Save one rolling checkpoint every 1/N epoch. Default N=8.')
parser.add_argument('--checkpoint_root', type=str, default='',
                    help='Rolling checkpoint root. Default: outputs/<version_name>.')
parser.add_argument('--resume_from_checkpoint', type=str, default='',
                    help='Checkpoint dir, outputs/<version_name>, or "auto" to resume from the latest rolling checkpoint.')
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
checkpoint_parts_per_epoch = max(1, args.checkpoint_parts_per_epoch)
checkpoint_root = args.checkpoint_root or os.path.join("outputs", version_name)
checkpoint_prefix = "grpo_speculative"
latest_checkpoint_file = f"latest_{checkpoint_prefix}_checkpoint.json"

dist_ctx = setup_distributed(seed=args.seed)
device = dist_ctx.device

if dist_ctx.is_main:
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(saved_draft_model_dir, exist_ok=True)
    os.makedirs(saved_statistics_dir, exist_ok=True)
    os.makedirs(checkpoint_root, exist_ok=True)
barrier()


rank0_print(datetime.now())
rank0_print(model_type, os.getenv('CUDA_VISIBLE_DEVICES'))
rank0_print("=" * 60)
rank0_print("Training & Generation Configuration")
rank0_print("=" * 60)
rank0_print(f"Model: {model_type} | Version: {version_name}")
rank0_print(f"Path: model={model_dir}, adapter={adapter_path}")
rank0_print(f"Train: epochs={num_epochs}, per_gpu_batch={batch_size}, global_batch={batch_size * dist_ctx.world_size}, "
      f"acc_steps={accumulation_steps}, draft_acc_steps={draft_accumulation_steps}")
rank0_print(f"LR: target={target_lr}, draft={draft_lr} | "
      f"Seq: max_len={max_length}, max_tokens={max_training_token}, pad_gap={max_training_padding_gap}")
rank0_print(f"Gen: temp={temperature}, top_p={top_p}"
      f"beta={beta}, epsilon={epsilon}")
rank0_print(f"Draft: train={is_train_draft}")
rank0_print(f"Draft layers: {draft_num_hidden_layers}")
rank0_print(f"Distributed: world_size={dist_ctx.world_size}, rank={dist_ctx.rank}, local_rank={dist_ctx.local_rank}, device={device}")
rank0_print(f"Rolling checkpoints: every 1/{checkpoint_parts_per_epoch} epoch -> {checkpoint_root}")
rank0_print(f"Iteration: grpo_iter={grpo_iteration_num}, sample={sample_num}, "
      f"repeat_gen={repeated_generate_nums}")
rank0_print("=" * 60)


config=AutoConfig.from_pretrained(model_dir)
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype='auto',config=config).to(device)
target_model.eval()

draft_config=deepcopy(config)
draft_config.rope_scaling=None
draft_config.num_hidden_layers=draft_num_hidden_layers
model=Model(draft_config,target_model=target_model)
model.load_model(adapter_path)
rank0_print(adapter_path)
model=model.to(device)
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

model.target_model = get_peft_model(model.target_model,lora_config).to(device)
if  args.load_lora_path != "":
    model.target_model.load_adapter(args.load_lora_path,adapter_name="default")
if dist_ctx.is_main:
    model.target_model.print_trainable_parameters()

def compute_target_loss(hidden_states, ref_hidden_states, old_logps, labels, mask, reward, epsilon, beta, grpo_iteration, lm_head):
    '''
    OOM-safe GRPO loss.
    The original implementation called log_softmax over the full [B, T, V] tensor
    which allocates hundreds of MiB per call and can OOM on 24GB GPUs.
    This version gathers token log-probs directly from hidden states in small sequence chunks.
    '''
    import os
    policy_chunk = int(os.environ.get("FASTGRPO_POLICY_CHUNK", "16"))

    mask = mask[..., :-1]
    labels = labels.to(hidden_states.device)[..., 1:]

    def gather_logps_chunked(hidden_tensor, token_labels, chunk_size, detach_result=False):
        hidden_tensor = hidden_tensor[..., :-1, :]
        parts = []
        seq_len = hidden_tensor.shape[1]

        for s in range(0, seq_len, chunk_size):
            e = min(s + chunk_size, seq_len)
            cur_hidden = hidden_tensor[:, s:e, :]
            cur_logits = lm_head(cur_hidden).float()
            cur_labels = token_labels[:, s:e].unsqueeze(-1)
            cur_logps = torch.gather(
                F.log_softmax(cur_logits, dim=-1),
                dim=2,
                index=cur_labels,
            ).squeeze(-1)
            if detach_result:
                cur_logps = cur_logps.detach()
            parts.append(cur_logps)

        return torch.cat(parts, dim=1)

    logps = gather_logps_chunked(hidden_states, labels, policy_chunk, detach_result=False)

    if grpo_iteration == 0:
        with torch.no_grad():
            ref_logps = gather_logps_chunked(ref_hidden_states, labels, policy_chunk, detach_result=True)
        old_logps = logps.detach()
    else:
        ref_logps = ref_hidden_states
        old_logps = old_logps

    coef1 = torch.exp(logps - old_logps)
    coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
    loss1 = torch.min(coef1 * reward, coef2 * reward)

    coef3 = ref_logps - logps
    loss2 = torch.exp(coef3) - coef3 - 1

    loss = -(loss1 - beta * loss2)
    loss = loss * mask
    denom = mask.sum(-1).clamp_min(1)
    loss = loss.sum(-1) / denom

    loss1_masked = loss1 * mask
    loss1_masked = loss1_masked.sum(-1) / denom
    abs_loss1 = torch.sum(torch.abs(loss1_masked))

    loss2_masked = loss2 * mask
    loss2_masked = loss2_masked.sum(-1) / denom

    return loss.sum(-1), abs_loss1, loss2_masked.sum(-1), old_logps, ref_logps


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
            loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

            loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
            loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss1=loss1.sum(-1)
            loss1=loss1*2.0

            # OOM-safe chunked CE loss for the draft model
            chunk_size = 128
            seq_len = draft_hidden_states.shape[1] - 1
            loss2_total = 0.0

            for start_idx in range(0, seq_len, chunk_size):
                end_idx = min(start_idx + chunk_size, seq_len)

                draft_h_chunk = draft_hidden_states[:, start_idx:end_idx, :]
                with torch.no_grad():
                    target_h_chunk = draft_input_states[:, start_idx+1:end_idx+1, :].to(model.target_model.dtype)
                    target_logits_chunk = model.target_model.lm_head(target_h_chunk)
                    target_probs_chunk = target_logits_chunk.float().softmax(dim=-1)

                draft_logits_chunk = model.lm_head(draft_h_chunk)
                log_probs_chunk = torch.log_softmax(draft_logits_chunk.float(), dim=-1)

                ce_chunk = -(target_probs_chunk * log_probs_chunk).sum(dim=-1)
                ce_chunk = ce_chunk * loss_mask[:, start_idx:end_idx]
                loss2_total = loss2_total + ce_chunk.sum(dim=-1)

            loss2 = loss2_total / torch.sum(loss_mask[...,:-1], dim=-1)
            loss2 = loss2.sum(-1)
            loss2 = loss2 * 0.1
            
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
    loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

    loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
    loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss1=loss1.sum(-1)
    loss1=loss1*2.0

    # OOM-safe chunked CE loss for the draft model (leftover batch)
    chunk_size = 128
    seq_len = draft_hidden_states.shape[1] - 1
    loss2_total = 0.0

    for start_idx in range(0, seq_len, chunk_size):
        end_idx = min(start_idx + chunk_size, seq_len)

        draft_h_chunk = draft_hidden_states[:, start_idx:end_idx, :]
        with torch.no_grad():
            target_h_chunk = draft_input_states[:, start_idx+1:end_idx+1, :].to(model.target_model.dtype)
            target_logits_chunk = model.target_model.lm_head(target_h_chunk)
            target_probs_chunk = target_logits_chunk.float().softmax(dim=-1)

        draft_logits_chunk = model.lm_head(draft_h_chunk)
        log_probs_chunk = torch.log_softmax(draft_logits_chunk.float(), dim=-1)

        ce_chunk = -(target_probs_chunk * log_probs_chunk).sum(dim=-1)
        ce_chunk = ce_chunk * loss_mask[:, start_idx:end_idx]
        loss2_total = loss2_total + ce_chunk.sum(dim=-1)

    loss2 = loss2_total / torch.sum(loss_mask[...,:-1], dim=-1)
    loss2 = loss2.sum(-1)
    loss2 = loss2 * 0.1
    
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

if dist_ctx.is_main and not args.resume_from_checkpoint:
    with open(log_file,'w',encoding='utf-8') as f:
        pass
barrier()

step=0
used_items=0
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


def _safe_mean(values):
    return float(mean(values)) if values else 0.0


def _safe_stdev(values):
    return float(stdev(values)) if len(values) > 1 else 0.0


def _safe_div(num, denom, default=0.0):
    return float(num) / float(denom) if denom else default


def _sync_cuda():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_resume_checkpoint(resume_from_checkpoint, root_dir, latest_filename):
    if not resume_from_checkpoint:
        return None

    candidate = root_dir if resume_from_checkpoint.lower() == "auto" else resume_from_checkpoint
    latest_path = os.path.join(candidate, latest_filename)
    if os.path.isdir(candidate):
        if os.path.exists(latest_path):
            with open(latest_path, "r", encoding="utf-8") as f:
                latest = json.load(f)
            checkpoint_dir = latest.get("checkpoint_dir") or latest.get("path")
            if checkpoint_dir and not os.path.isabs(checkpoint_dir):
                checkpoint_dir = os.path.join(candidate, checkpoint_dir)
            candidate = checkpoint_dir
        elif not os.path.exists(os.path.join(candidate, "trainer_state.pt")):
            raise FileNotFoundError(f"Cannot find {latest_filename} or trainer_state.pt in {candidate}")

    if not candidate or not os.path.isdir(candidate):
        raise FileNotFoundError(f"Cannot find resume checkpoint: {resume_from_checkpoint}")
    return candidate


def _load_peft_adapter_state(peft_model, adapter_dir):
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        adapter_state = load_file(safetensors_path, device="cpu")
    elif os.path.exists(bin_path):
        adapter_state = torch.load(bin_path, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"Cannot find adapter weights in {adapter_dir}")

    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(peft_model, adapter_state, adapter_name="default")


def _checkpoint_part_for_batch(batch_idx, batches_per_epoch, parts_per_epoch):
    if batches_per_epoch <= 0:
        return 0
    completed_batches = batch_idx + 1
    return min(parts_per_epoch, (completed_batches * parts_per_epoch) // batches_per_epoch)


def _next_position_after_batch(epoch, batch_idx, batches_per_epoch):
    next_batch_idx = batch_idx + 1
    if next_batch_idx >= batches_per_epoch:
        return epoch + 1, 0
    return epoch, next_batch_idx


def _save_rolling_checkpoint(
    *,
    root_dir,
    prefix,
    latest_filename,
    checkpoint_part,
    parts_per_epoch,
    epoch,
    batch_idx,
    batches_per_epoch,
    model,
    tokenizer,
    optimizer_target,
    optimizer_draft,
    trainer_state,
    rank_state,
):
    checkpoint_name = f"{prefix}_epoch_{epoch + 1:04d}_part_{checkpoint_part:02d}_of_{parts_per_epoch:02d}"
    tmp_dir = os.path.join(root_dir, f".tmp_{checkpoint_name}")
    checkpoint_dir = os.path.join(root_dir, checkpoint_name)

    next_epoch, next_batch_idx = _next_position_after_batch(epoch, batch_idx, batches_per_epoch)
    state = {
        **trainer_state,
        "script": "grpo_speculative.py",
        "version_name": version_name,
        "epoch": epoch,
        "batch_idx": batch_idx,
        "next_epoch": next_epoch,
        "next_batch_idx": next_batch_idx,
        "checkpoint_part": checkpoint_part,
        "checkpoint_parts_per_epoch": parts_per_epoch,
        "batches_per_epoch": batches_per_epoch,
        "checkpoint_name": checkpoint_name,
    }

    if dist_ctx.is_main:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        target_dir = os.path.join(tmp_dir, "target_model")
        draft_dir = os.path.join(tmp_dir, "draft_model")
        os.makedirs(draft_dir, exist_ok=True)

        model.target_model.save_pretrained(target_dir)
        tokenizer.save_pretrained(target_dir)
        model.save_model(os.path.join(draft_dir, "draft_model.pth"))

        common_state = {
            **state,
            "optimizer_target": optimizer_target.state_dict(),
            "optimizer_draft": optimizer_draft.state_dict(),
        }
        torch.save(common_state, os.path.join(tmp_dir, "trainer_state.pt"))

        json_state = {k: v for k, v in common_state.items() if not k.startswith("optimizer_")}
        with open(os.path.join(tmp_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(json_state, f, indent=2)

        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.rename(tmp_dir, checkpoint_dir)

    barrier()
    torch.save({**state, **rank_state}, os.path.join(checkpoint_dir, f"rank_{dist_ctx.rank}_state.pt"))
    barrier()

    if dist_ctx.is_main:
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if name.startswith(f"{prefix}_epoch_") and os.path.abspath(path) != os.path.abspath(checkpoint_dir):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        latest = {
            "checkpoint_dir": checkpoint_name,
            "path": checkpoint_dir,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "next_epoch": next_epoch,
            "next_batch_idx": next_batch_idx,
            "checkpoint_part": checkpoint_part,
            "checkpoint_parts_per_epoch": parts_per_epoch,
            "updated_at": datetime.now().isoformat(),
        }
        with open(os.path.join(root_dir, latest_filename), "w", encoding="utf-8") as f:
            json.dump(latest, f, indent=2)

        rank0_print(f"Saved rolling checkpoint to {checkpoint_dir}")
    barrier()


def _flush_optimizers_for_checkpoint(
    *,
    model,
    optimizer_target,
    optimizer_draft,
    target_micro_steps,
    draft_micro_steps,
    global_step,
    draft_step,
):
    if target_micro_steps > 0:
        average_gradients(model.target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
        optimizer_target.step()
        optimizer_target.zero_grad(set_to_none=True)
        target_micro_steps = 0
        global_step += 1

    if is_train_draft and draft_micro_steps % max(1, draft_accumulation_steps) != 0:
        average_gradients(model.draft_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
        optimizer_draft.step()
        optimizer_draft.zero_grad(set_to_none=True)
        draft_micro_steps = 0
        draft_step += 1

    return target_micro_steps, draft_micro_steps, global_step, draft_step


sampler = None
if dist_ctx.distributed:
    sampler = DistributedSampler(
        QAs,
        num_replicas=dist_ctx.world_size,
        rank=dist_ctx.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )

dataloader = DataLoader(
    QAs,
    collate_fn=TrainDataCollator(tokenizer=tokenizer),
    num_workers=args.num_workers,
    persistent_workers=args.num_workers > 0,
    batch_size=batch_size,
    shuffle=(sampler is None),
    sampler=sampler,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

global_step = 0
target_micro_steps = 0
draft_micro_steps = 0
optimizer_target.zero_grad(set_to_none=True)
optimizer_draft.zero_grad(set_to_none=True)
start_epoch = 0
resume_batch_idx = 0
resume_checkpoint_epoch = -1
resume_checkpoint_part = 0

resume_checkpoint_path = _resolve_resume_checkpoint(args.resume_from_checkpoint, checkpoint_root, latest_checkpoint_file)
if resume_checkpoint_path is not None:
    model.load_model(os.path.join(resume_checkpoint_path, "draft_model", "draft_model.pth"))
    _load_peft_adapter_state(model.target_model, os.path.join(resume_checkpoint_path, "target_model"))
    trainer_state = torch.load(os.path.join(resume_checkpoint_path, "trainer_state.pt"), map_location=device, weights_only=False)
    optimizer_target.load_state_dict(trainer_state["optimizer_target"])
    if is_train_draft and "optimizer_draft" in trainer_state:
        optimizer_draft.load_state_dict(trainer_state["optimizer_draft"])

    rank_state_path = os.path.join(resume_checkpoint_path, f"rank_{dist_ctx.rank}_state.pt")
    rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=False) if os.path.exists(rank_state_path) else trainer_state

    step = int(trainer_state.get("step", step))
    used_items = int(rank_state.get("used_items", trainer_state.get("used_items", used_items)))
    draft_step = int(trainer_state.get("draft_step", draft_step))
    global_step = int(trainer_state.get("global_step", global_step))
    target_micro_steps = int(trainer_state.get("target_micro_steps", 0))
    draft_micro_steps = int(trainer_state.get("draft_micro_steps", 0))
    start_epoch = int(trainer_state.get("next_epoch", 0))
    resume_batch_idx = int(trainer_state.get("next_batch_idx", 0))
    resume_checkpoint_epoch = int(trainer_state.get("epoch", -1))
    resume_checkpoint_part = int(trainer_state.get("checkpoint_part", 0))
    rank0_print(
        f"Resumed rolling checkpoint {resume_checkpoint_path} "
        f"at epoch={start_epoch + 1}, batch={resume_batch_idx}"
    )
barrier()

try:
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        batch_data['ignore_due_correct'] = 0
        batch_data['ignore_due_incorrect'] = 0
        batch_data['length_stdev'] = []
        batch_data['length_range'] = []
        batch_data['length_cv'] = []
        last_checkpoint_part = resume_checkpoint_part if epoch == resume_checkpoint_epoch else 0
        batches_per_epoch = len(dataloader)

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            dynamic_ncols=True,
            disable=not dist_ctx.is_main,
        )

        for i, batch in enumerate(progress_bar):
            if epoch == start_epoch and i < resume_batch_idx:
                continue

            local_iter_start = time.time()
            local_generate_time = 0.0
            local_train_time = 0.0
            local_draft_train_time = 0.0
            local_generated_sequences = 0
            local_train_messages = 0
            local_mean_reward_for_log = 0.0

            local_can_generate = (
                batch['input_ids'].shape[-1] < max_length
                and not any(answer is None for answer in batch['answers'])
            )
            global_can_generate = int(reduce_sum(1 if local_can_generate else 0, device))

            if global_can_generate == 0:
                global_used_items = int(reduce_sum(used_items, device))
                current_checkpoint_part = _checkpoint_part_for_batch(i, batches_per_epoch, checkpoint_parts_per_epoch)
                if current_checkpoint_part > last_checkpoint_part:
                    target_micro_steps, draft_micro_steps, global_step, draft_step = _flush_optimizers_for_checkpoint(
                        model=model,
                        optimizer_target=optimizer_target,
                        optimizer_draft=optimizer_draft,
                        target_micro_steps=target_micro_steps,
                        draft_micro_steps=draft_micro_steps,
                        global_step=global_step,
                        draft_step=draft_step,
                    )
                    _save_rolling_checkpoint(
                        root_dir=checkpoint_root,
                        prefix=checkpoint_prefix,
                        latest_filename=latest_checkpoint_file,
                        checkpoint_part=current_checkpoint_part,
                        parts_per_epoch=checkpoint_parts_per_epoch,
                        epoch=epoch,
                        batch_idx=i,
                        batches_per_epoch=batches_per_epoch,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer_target=optimizer_target,
                        optimizer_draft=optimizer_draft,
                        trainer_state={
                            "step": step,
                            "global_step": global_step,
                            "draft_step": draft_step,
                            "target_micro_steps": target_micro_steps,
                            "draft_micro_steps": draft_micro_steps,
                            "global_used_items": global_used_items,
                        },
                        rank_state={"used_items": used_items},
                    )
                    last_checkpoint_part = current_checkpoint_part
                continue

            if local_can_generate:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                messages = batch['messages']
                answers = batch['answers']

                _sync_cuda()
                generate_start = time.time()
                with torch.inference_mode():
                    outputs = speculative_generate(
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
                        statistical_time=True,
                    )
                _sync_cuda()
                local_generate_time = time.time() - generate_start

                prompt_length = input_ids.shape[-1]
                outputs['prompt_length'] = prompt_length
                outputs['decoded_sequences'] = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs['generated_token_ids']]
                token_ids_length = [len(item) for item in outputs['generated_token_ids']]
                length_stdev = _safe_stdev(token_ids_length)
                length_range = max(token_ids_length) - min(token_ids_length) if token_ids_length else 0
                length_ave = _safe_mean(token_ids_length)
                length_cv = _safe_div(length_stdev, length_ave, 0.0)
                batch_data['generate_length_list'].extend(token_ids_length)
                local_generated_sequences = len(outputs['decoded_sequences'])

                if is_train_draft:
                    _sync_cuda()
                    draft_train_time_start = time.time()
                    draft_loss1, draft_loss2 = training_draft_model(model, outputs, attention_mask)
                    _sync_cuda()
                    local_draft_train_time = time.time() - draft_train_time_start
                    batch_data['draft_train_time_cost'] += local_draft_train_time
                    batch_data['last_draft_loss1'].append(draft_loss1)
                    batch_data['last_draft_loss2'].append(draft_loss2)

                generate_length = 0.0
                for idx_batch in range(len(answers)):
                    generate_length += outputs['max_sequence_length']
                    rewards = []
                    new_messages = []
                    for idx_k in range(repeated_generate_nums):
                        idx_sequence = idx_batch * repeated_generate_nums + idx_k
                        decoded_sequence = outputs['decoded_sequences'][idx_sequence]
                        ground_truth = answers[idx_batch]

                        new_message = deepcopy(messages[idx_batch])
                        new_message.append({"role": "assistant", "content": decoded_sequence})

                        format_reward = format_reward_func([decoded_sequence])
                        answer_reward = accuracy_reward_func([decoded_sequence], [ground_truth])
                        reward = 0.2 * format_reward[0] + answer_reward[0]

                        rewards.append(reward)
                        new_messages.append(new_message)

                    rewards = np.array(rewards)
                    if rewards.std() == 0:
                        if rewards[0] >= 1.0:
                            batch_data['ignore_due_correct'] += 1
                        else:
                            batch_data['ignore_due_incorrect'] += 1
                        continue

                    std_rewards = (rewards - rewards.mean()) / rewards.std()
                    batch_data['messages'] += new_messages
                    batch_data['rewards'] += rewards.tolist()
                    batch_data['std_rewards'] += std_rewards.tolist()
                    used_items += 1

                generate_length = generate_length / max(1, len(answers))
                batch_data['length_stdev'].append(length_stdev)
                batch_data['length_range'].append(length_range)
                batch_data['length_cv'].append(length_cv)
                batch_data['last_generate_time_cost'].append(outputs['total_time_cost'])
                batch_data['last_acc_length'].append(outputs['total_acc_length'])
                batch_data['last_decoded_token_num'].append(outputs['total_decoded_token_num'])
                batch_data['last_generate_length'].append(generate_length)
                batch_data['prefill_time_cost'] += outputs['prefill_time_cost']
                batch_data['target_time_cost'] += outputs['target_time_cost']
                batch_data['draft_time_cost'] += outputs['draft_time_cost']
                batch_data['check_time_cost'] += outputs['check_time_cost']
                batch_data['generate_time_cost'] += outputs['total_time_cost']
                batch_data['total_acc_length'] += outputs['total_acc_length']
                batch_data['total_decoded_token_num'] += outputs['total_decoded_token_num']
                batch_data['generate_length'] += generate_length
            else:
                # Keep rank participation symmetric when another rank performs draft backward.
                if is_train_draft:
                    (make_zero_loss(model.draft_model.parameters(), device) / max(1, draft_accumulation_steps)).backward()

            if is_train_draft:
                draft_micro_steps += 1
                if draft_micro_steps % draft_accumulation_steps == 0:
                    average_gradients(model.draft_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
                    optimizer_draft.step()
                    optimizer_draft.zero_grad(set_to_none=True)
                    draft_step += 1

            if dist_ctx.is_main and draft_step % 1024 == 0 and draft_step > 0 and step > 0 and is_train_draft:
                with open(f"{saved_statistics_dir}/{step}.pkl", "wb") as f:
                    pickle.dump(batch_data['generate_length_list'], f)

            local_train_messages = len(batch_data['messages'])
            global_train_messages = int(reduce_sum(local_train_messages, device))
            global_used_items = int(reduce_sum(used_items, device))
            step = global_used_items // max(1, batch_size * accumulation_steps * dist_ctx.world_size)

            if global_train_messages == 0:
                if dist_ctx.is_main:
                    progress_bar.set_postfix({
                        'step': step,
                        'used': global_used_items,
                        'train_seq': global_train_messages,
                    })
                current_checkpoint_part = _checkpoint_part_for_batch(i, batches_per_epoch, checkpoint_parts_per_epoch)
                if current_checkpoint_part > last_checkpoint_part:
                    target_micro_steps, draft_micro_steps, global_step, draft_step = _flush_optimizers_for_checkpoint(
                        model=model,
                        optimizer_target=optimizer_target,
                        optimizer_draft=optimizer_draft,
                        target_micro_steps=target_micro_steps,
                        draft_micro_steps=draft_micro_steps,
                        global_step=global_step,
                        draft_step=draft_step,
                    )
                    _save_rolling_checkpoint(
                        root_dir=checkpoint_root,
                        prefix=checkpoint_prefix,
                        latest_filename=latest_checkpoint_file,
                        checkpoint_part=current_checkpoint_part,
                        parts_per_epoch=checkpoint_parts_per_epoch,
                        epoch=epoch,
                        batch_idx=i,
                        batches_per_epoch=batches_per_epoch,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer_target=optimizer_target,
                        optimizer_draft=optimizer_draft,
                        trainer_state={
                            "step": step,
                            "global_step": global_step,
                            "draft_step": draft_step,
                            "target_micro_steps": target_micro_steps,
                            "draft_micro_steps": draft_micro_steps,
                            "global_used_items": global_used_items,
                        },
                        rank_state={"used_items": used_items},
                    )
                    last_checkpoint_part = current_checkpoint_part
                torch.cuda.empty_cache()
                continue

            local_mean_reward_for_log = _safe_mean(batch_data['rewards'])
            batch_old_logps = []
            batch_ref_logps = []

            if local_train_messages > 0:
                text = tokenizer.apply_chat_template(batch_data['messages'], tokenize=False, add_generation_prompt=False)
                text = tokenizer(text, padding=False)
                loss_mask = []

                for idx_message, message in enumerate(batch_data['messages']):
                    prompt_text = tokenizer.apply_chat_template(message[:-1], tokenize=False, add_generation_prompt=True)
                    prompt_text = tokenizer.encode(prompt_text)
                    cur_loss_mask = [0] * (len(prompt_text) - 1) + [1] * (len(text.input_ids[idx_message]) - len(prompt_text) + 1)
                    loss_mask.append(cur_loss_mask)

                input_ids_train = text.input_ids
                attention_mask_train = text.attention_mask

                sorted_pairs = sorted(
                    zip(input_ids_train, attention_mask_train, loss_mask, batch_data['std_rewards']),
                    key=lambda x: len(x[0]),
                    reverse=False,
                )

                input_ids_train, attention_mask_train, loss_mask, sorted_rewards = map(list, zip(*sorted_pairs))
            else:
                input_ids_train, attention_mask_train, loss_mask, sorted_rewards = [], [], [], []

            for grpo_iteration in range(grpo_iteration_num):
                _sync_cuda()
                train_time_start = time.time()

                if local_train_messages > 0:
                    cur_max_length = 0
                    cur_input_ids = []
                    cur_attention_mask = []
                    cur_loss_mask = []
                    cur_rewards = []

                    def _flush_policy_chunk(final_chunk=False):
                        if len(cur_input_ids) == 0:
                            return

                        cur_batch = len(cur_input_ids)
                        for idx_seq in range(cur_batch):
                            cur_len = len(cur_input_ids[idx_seq])
                            padding_len = cur_max_length - cur_len
                            if padding_len > 0:
                                cur_input_ids[idx_seq] = cur_input_ids[idx_seq] + [0] * padding_len
                                cur_loss_mask[idx_seq] = cur_loss_mask[idx_seq] + [0] * padding_len
                                cur_attention_mask[idx_seq] = cur_attention_mask[idx_seq] + [0] * padding_len

                        tensor_input_ids = torch.tensor(cur_input_ids, device=device)
                        tensor_attention_mask = torch.tensor(cur_attention_mask, device=device)
                        tensor_loss_mask = torch.tensor(cur_loss_mask, device=device)
                        tensor_rewards = torch.tensor(cur_rewards, device=device).unsqueeze(-1)

                        chunk_cache_idx = len(batch_old_logps) if grpo_iteration == 0 else _flush_policy_chunk.cache_idx
                        if grpo_iteration == 0:
                            model.target_model.disable_adapter_layers()
                            with torch.no_grad():
                                ref_outputs = model.target_model.base_model.model.model(tensor_input_ids, attention_mask=tensor_attention_mask)
                            ref_hidden = ref_outputs[0]
                        else:
                            ref_hidden = batch_ref_logps[chunk_cache_idx]

                        model.target_model.enable_adapter_layers()
                        outputs_policy = model.target_model.base_model.model.model(tensor_input_ids, attention_mask=tensor_attention_mask)
                        hidden_states = outputs_policy[0]

                        old_logps = None if grpo_iteration == 0 else batch_old_logps[chunk_cache_idx]
                        loss, abs_loss1, loss2, old_logps, ref_logps = compute_target_loss(
                            hidden_states,
                            ref_hidden,
                            old_logps,
                            tensor_input_ids,
                            tensor_loss_mask,
                            tensor_rewards,
                            epsilon,
                            beta,
                            grpo_iteration,
                            model.target_model.base_model.model.lm_head,
                        )

                        if grpo_iteration == 0:
                            batch_old_logps.append(old_logps)
                            batch_ref_logps.append(ref_logps)
                        else:
                            _flush_policy_chunk.cache_idx += 1

                        scaled_loss = loss * (dist_ctx.world_size / float(global_train_messages))
                        (scaled_loss / max(1, accumulation_steps)).backward()

                        cur_input_ids.clear()
                        cur_attention_mask.clear()
                        cur_loss_mask.clear()
                        cur_rewards.clear()

                    _flush_policy_chunk.cache_idx = 0
                    for j in range(local_train_messages):
                        seq_len = len(input_ids_train[j])
                        if (
                            (max(cur_max_length, seq_len) * (len(cur_input_ids) + 1) <= max_training_token
                             and (seq_len - cur_max_length) * len(cur_input_ids) <= max_training_padding_gap)
                            or len(cur_input_ids) == 0
                        ):
                            cur_max_length = max(cur_max_length, seq_len)
                            cur_input_ids.append(input_ids_train[j])
                            cur_attention_mask.append(attention_mask_train[j])
                            cur_loss_mask.append(loss_mask[j])
                            cur_rewards.append(sorted_rewards[j])
                        else:
                            _flush_policy_chunk(final_chunk=False)
                            cur_max_length = seq_len
                            cur_input_ids.append(input_ids_train[j])
                            cur_attention_mask.append(attention_mask_train[j])
                            cur_loss_mask.append(loss_mask[j])
                            cur_rewards.append(sorted_rewards[j])

                    _flush_policy_chunk(final_chunk=True)
                else:
                    # No local valid GRPO samples. Missing grads will be materialized as zero during all-reduce.
                    pass

                target_micro_steps += 1
                if target_micro_steps % accumulation_steps == 0:
                    average_gradients(model.target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
                    optimizer_target.step()
                    optimizer_target.zero_grad(set_to_none=True)
                    target_micro_steps = 0
                    global_step += 1

                _sync_cuda()
                local_train_time = time.time() - train_time_start
                batch_data['last_train_time_cost'].append(local_train_time)
                batch_data['train_time_cost'] += local_train_time
                if batch_data['rewards']:
                    batch_data['last_mean_rewards'].append(_safe_mean(batch_data['rewards']))
                    batch_data['mean_rewards'] += _safe_mean(batch_data['rewards'])

                real_sample_num = sample_num * accumulation_steps
                global_generated_sequences = int(reduce_sum(local_generated_sequences, device))
                global_generate_time = reduce_max(local_generate_time, device)
                global_train_time = reduce_max(local_train_time, device)
                global_iter_time = reduce_max(time.time() - local_iter_start, device)
                global_mean_reward_sum = reduce_sum(local_mean_reward_for_log * local_train_messages, device)
                global_peak_mem = reduce_max(torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else 0.0, device)

                global_mean_reward = _safe_div(global_mean_reward_sum, global_train_messages, 0.0)
                local_last_acc_tokens = sum(batch_data['last_acc_length'][-real_sample_num:])
                local_last_decoded = sum(batch_data['last_decoded_token_num'][-real_sample_num:])
                global_last_acc_tokens = reduce_sum(local_last_acc_tokens, device)
                global_last_decoded = reduce_sum(local_last_decoded, device)

                if dist_ctx.is_main:
                    avg_logs = {
                        "epoch": epoch + 1,
                        "rank": dist_ctx.rank,
                        "world_size": dist_ctx.world_size,
                        "step": step,
                        "optimizer_step": global_step,
                        "used_items_global": global_used_items,
                        "per_gpu_batch_size": batch_size,
                        "global_prompt_batch_size": batch_size * dist_ctx.world_size,
                        "train_messages_global": global_train_messages,
                        "generated_sequences_global": global_generated_sequences,
                        "length_range": round(_safe_mean(batch_data['length_range']), 4),
                        "length_cv": round(_safe_mean(batch_data['length_cv']), 4),
                        "length_stdev": round(_safe_mean(batch_data['length_stdev']), 4),
                        "grpo_iteration": grpo_iteration + 1,
                        "used_time": round((time.time() - start_time) / 60, 3),
                        f"last_{sample_num}_generate_time_cost_local_min": round(sum(batch_data['last_generate_time_cost'][-real_sample_num:]) / 60, 3),
                        f"last_{sample_num}_train_time_cost_max_rank_min": round(global_train_time / 60, 3),
                        f"last_{sample_num}_acc_length_global": round(_safe_div(global_last_acc_tokens, global_last_decoded, 0.0), 4),
                        f"last_{sample_num}_mean_rewards_global": round(global_mean_reward, 3),
                        f"last_{sample_num}_mean_length_local": round(_safe_mean(batch_data['last_generate_length'][-real_sample_num:]), 3),
                        "ignore_due_correct_cur_epoch_local_rank0": batch_data['ignore_due_correct'],
                        "ignore_due_incorrect_cur_epoch_local_rank0": batch_data['ignore_due_incorrect'],
                        "generate_time_cost_local_min": round(batch_data['generate_time_cost'] / 60, 3),
                        "generate_time_cost_max_rank_sec": round(global_generate_time, 4),
                        "iteration_time_cost_max_rank_sec": round(global_iter_time, 4),
                        "average_acc_length_local": round(_safe_div(batch_data['total_acc_length'], batch_data['total_decoded_token_num'], 0.0), 4),
                        "prefill_time_cost_local_min": round(batch_data['prefill_time_cost'] / 60, 3),
                        "target_time_cost_local_min": round(batch_data['target_time_cost'] / 60, 3),
                        "draft_time_cost_local_min": round(batch_data['draft_time_cost'] / 60, 3),
                        "train_time_cost_local_min": round(batch_data['train_time_cost'] / 60, 3),
                        "check_time_cost_local_min": round(batch_data['check_time_cost'] / 60, 3),
                        "mean_reward_global": round(global_mean_reward, 4),
                        "peak_vram_gb_max_rank": round(global_peak_mem, 4),
                        "draft_train_time_cost_local_min": round(batch_data['draft_train_time_cost'] / 60, 3) if is_train_draft else 0,
                        f"last_{sample_num}_draft_loss1_local": round(_safe_mean(batch_data['last_draft_loss1'][-real_sample_num:]), 4) if is_train_draft else 0,
                        f"last_{sample_num}_draft_loss2_local": round(_safe_mean(batch_data['last_draft_loss2'][-real_sample_num:]), 4) if is_train_draft else 0,
                    }
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(avg_logs) + '\n')
                    progress_bar.set_postfix({
                        'step': step,
                        'opt': global_step,
                        'reward': f'{global_mean_reward:.3f}',
                        'train_seq': global_train_messages,
                        'acc': f'{_safe_div(global_last_acc_tokens, global_last_decoded, 0.0):.3f}',
                    })

                torch.cuda.empty_cache()

            batch_data['messages'].clear()
            batch_data['rewards'].clear()
            batch_data['std_rewards'].clear()
            batch_old_logps.clear()
            batch_ref_logps.clear()

            current_checkpoint_part = _checkpoint_part_for_batch(i, batches_per_epoch, checkpoint_parts_per_epoch)
            if current_checkpoint_part > last_checkpoint_part:
                target_micro_steps, draft_micro_steps, global_step, draft_step = _flush_optimizers_for_checkpoint(
                    model=model,
                    optimizer_target=optimizer_target,
                    optimizer_draft=optimizer_draft,
                    target_micro_steps=target_micro_steps,
                    draft_micro_steps=draft_micro_steps,
                    global_step=global_step,
                    draft_step=draft_step,
                )
                _save_rolling_checkpoint(
                    root_dir=checkpoint_root,
                    prefix=checkpoint_prefix,
                    latest_filename=latest_checkpoint_file,
                    checkpoint_part=current_checkpoint_part,
                    parts_per_epoch=checkpoint_parts_per_epoch,
                    epoch=epoch,
                    batch_idx=i,
                    batches_per_epoch=batches_per_epoch,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer_target=optimizer_target,
                    optimizer_draft=optimizer_draft,
                    trainer_state={
                        "step": step,
                        "global_step": global_step,
                        "draft_step": draft_step,
                        "target_micro_steps": target_micro_steps,
                        "draft_micro_steps": draft_micro_steps,
                        "global_used_items": global_used_items,
                    },
                    rank_state={"used_items": used_items},
                )
                last_checkpoint_part = current_checkpoint_part

    if target_micro_steps > 0:
        average_gradients(model.target_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
        optimizer_target.step()
        optimizer_target.zero_grad(set_to_none=True)
        target_micro_steps = 0
        global_step += 1

    if is_train_draft and draft_micro_steps % max(1, draft_accumulation_steps) != 0:
        average_gradients(model.draft_model.parameters(), dist_ctx.world_size, divide=True, ensure_grads=True)
        optimizer_draft.step()
        optimizer_draft.zero_grad(set_to_none=True)
        draft_micro_steps = 0
        draft_step += 1

    barrier()
    if dist_ctx.is_main:
        model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
        model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')
        tokenizer.save_pretrained(f'{saved_model_dir}/step{step}')
    barrier()
finally:
    cleanup_distributed()
