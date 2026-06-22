import pandas as pd
from transformers import AutoTokenizer, AutoProcessor,AutoConfig,AutoModelForCausalLM
from helper.modeling_draft import Model
import torch
import datasets
import os

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
import datasets
from transformers import get_cosine_schedule_with_warmup,get_scheduler
from transformers import DynamicCache
import json
import pandas as pd
import re
import signal
import sys
import torch
import argparse
from copy import deepcopy
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)



parser = argparse.ArgumentParser(description="Training configuration") 
parser.add_argument('--model_dir',type=str,) 
parser.add_argument('--version_name', type=str,help='Version name for saving checkpoints')
parser.add_argument('--model_type', type=str, default='qwen2',
                    choices=['qwen2', 'llama', 'deepseek'])
parser.add_argument('--draft_num_hidden_layers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--accumulation_steps', type=int, default=16)
parser.add_argument('--warmup_ratio', type=float, default=0.05)
parser.add_argument('--sample_num', type=int, default=200)
parser.add_argument('--max_length', type=int, default=4096)
parser.add_argument('--log_dir',type=str,required=True)
parser.add_argument('--saved_model_dir',type=str,required=True)
parser.add_argument('--dataset_dir',type=str,required=True)

args = parser.parse_args()
model_dir=args.model_dir
version_name=args.version_name
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
accumulation_steps = args.accumulation_steps
warmup_ratio = args.warmup_ratio
sample_num = args.sample_num
max_length = args.max_length
log_dir=args.log_dir
saved_model_dir=args.saved_model_dir
dataset_dir = args.dataset_dir

distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = 0
world_size = 1

if distributed:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

is_main_process = rank == 0
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cuda")

if is_main_process:
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

if distributed:
    dist.barrier()

def unwrap_module(module):
    return module.module if hasattr(module, "module") else module

def save_draft_model(model, save_path):
    state_dict = {
        'draft_model': unwrap_module(model.draft_model).state_dict()
    }
    torch.save(state_dict, save_path)

if is_main_process:
    print(version_name, os.getenv('CUDA_VISIBLE_DEVICES'))
    print(f"Distributed: {distributed} | world_size={world_size}")

with open(dataset_dir,'r',encoding='utf-8') as f:
    sharegpt_dataset=json.load(f)
df=pd.DataFrame(sharegpt_dataset)
dataset=datasets.Dataset.from_pandas(df)
if is_main_process:
    print(dataset)

config=AutoConfig.from_pretrained(model_dir)
model_type=args.model_type
draft_num_hidden_layers=args.draft_num_hidden_layers
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype='auto',config=config)
target_model.eval()

draft_config=deepcopy(config)
draft_config.rope_scaling=None
draft_config.num_hidden_layers=draft_num_hidden_layers
model=Model(draft_config, target_model=target_model).to(device)
if distributed:
    model.draft_model = DDP(
        model.draft_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side = "right")

count=0
for param in model.parameters():
    if param.requires_grad==True:
        if is_main_process:
            print(param.shape)
        count+=param.numel()
        
if is_main_process:
    print(count/1000/1000,'M')


class DataCollator:
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer=tokenizer
        self.max_length=max_length
        
    def __call__(self, batch):
        batch_input_ids=[]
        batch_attention_mask=[]
        batch_loss_mask=[]
        max_length=0
        
        for example in batch:
            
            input_ids=[]
            attention_mask=[]
            loss_mask=[]
            
            if model_type == 'qwen2':
                text='<|im_start|>'+'system'+'\n'+'You are a helpful assistant.'+'<|im_end|>'+'\n'
            elif model_type == 'llama':
                text= '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n'+'You are a helpful assistant.'+'<|eot_id|>'
            elif model_type == 'deepseek':
                text = "<｜begin▁of▁sentence｜>You are a helpful assistant."
            the_input_ids=self.tokenizer.encode(text,add_special_tokens=False)
            input_ids+=the_input_ids
            attention_mask+=[1]*len(the_input_ids)
            loss_mask+=[0]*len(the_input_ids)

            for idx, conversation in enumerate(example['conversations']):
                role=conversation['from']
                content=conversation['value']
                if role == 'human':
                    role = 'user'
                if role == 'gpt':
                    role = 'assistant'
                
                if model_type == 'qwen2':
                    text='<|im_start|>'+role+'\n'+content+'<|im_end|>'+'\n'
                elif model_type == 'llama':
                    text='<|start_header_id|>'+role+'<|end_header_id|>\n\n'+content+'<|eot_id|>'
                elif model_type == 'deepseek':
                    if role == 'user':
                        text = "<｜User｜>" + content
                    else:
                        text = "<｜Assistant｜>" + content + "<｜end▁of▁sentence｜>"
                the_input_ids=self.tokenizer.encode(text,add_special_tokens=False)
                input_ids+=the_input_ids
                attention_mask+=[1]*len(the_input_ids)

                if role == 'assistant' or role == 'ASSISTANT':
                    loss_mask+=[1]*len(the_input_ids)
                else:
                    loss_mask+=[0]*len(the_input_ids)


            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_loss_mask.append(loss_mask)
            max_length=max(max_length,len(input_ids))

        max_length=min(max_length,self.max_length)
        for idx in range(len(batch)):
            if len(batch_input_ids[idx])>=max_length:
                batch_input_ids[idx]=batch_input_ids[idx][:max_length]
                batch_attention_mask[idx]=batch_attention_mask[idx][:max_length]
                batch_loss_mask[idx]=batch_loss_mask[idx][:max_length]
            
            else:
                the_length=len(batch_input_ids[idx])
                batch_input_ids[idx]=batch_input_ids[idx]+[self.tokenizer.eos_token_id]*(max_length-the_length)
                batch_attention_mask[idx]=batch_attention_mask[idx]+[0]*(max_length-the_length)
                batch_loss_mask[idx]=batch_loss_mask[idx]+[0]*(max_length-the_length)
        
        return {
            'input_ids':torch.tensor(batch_input_ids),
            'attention_mask':torch.tensor(batch_attention_mask),
            'loss_mask':torch.tensor(batch_loss_mask)
        }


datacollator=DataCollator(tokenizer, max_length=max_length)
train_sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    drop_last=False,
) if distributed else None
dataloader=DataLoader(
    dataset,
    collate_fn=datacollator,
    num_workers=4,
    persistent_workers=True,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    sampler=train_sampler,
    drop_last=False,
)


def compute_acc(target_logits,draft_logits,valid_positions,k=2):

    target_indices = torch.argmax(target_logits, dim=-1)
    draft_topk_values, draft_topk_indices = torch.topk(draft_logits, k=k, dim=-1)

    top1_hit = draft_topk_indices[..., 0] == target_indices             
    topk_hit = (draft_topk_indices == target_indices.unsqueeze(-1)).any(dim=-1) 

    correct_top1 = (top1_hit & valid_positions).sum().item()
    correct_topk = (topk_hit & valid_positions).sum().item()
    total_valid_tokens = valid_positions.sum().item()
    
    return correct_top1,correct_topk,total_valid_tokens

def compute_normalized_gradient_l2_norm(model):
    gradient_l2_norm = torch.norm(
        torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    )
    num_grad_params = sum(
        param.grad.numel() for param in model.parameters() if param.grad is not None
    )
    normalized_gradient_l2_norm = gradient_l2_norm / num_grad_params
    
    return normalized_gradient_l2_norm

optimizer = torch.optim.AdamW(model.draft_model.parameters(), lr=lr)
l1_loss=nn.SmoothL1Loss(reduction='none')

num_training_steps = num_epochs * ((len(dataloader)+accumulation_steps-1)//accumulation_steps)
num_warmup_steps = min(int(warmup_ratio * num_training_steps), 500)
if is_main_process:
    print(num_training_steps)
lr_scheduler = get_scheduler(
    name="cosine_with_min_lr",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    scheduler_specific_kwargs={'min_lr_rate':0.0}, 
)

total_correct_top1=[]
total_correct_topk=[]
total_token_nums=[]

step=0
accumulated_step=0
batch_logs=[]
start_time=time.time()

for epoch in range(num_epochs):

    log_file = log_dir + f"/epoch_{epoch}.log"
    if distributed:
        train_sampler.set_epoch(epoch)

    if is_main_process:
        with open(log_file,'w',encoding='utf-8') as f:
            pass

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        dynamic_ncols=True,
        disable=not is_main_process,
    )

    for i,batch in enumerate(progress_bar):

        input_ids=batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        loss_mask=batch['loss_mask'].to(device)
        
        has_loss_token = torch.tensor(
            [1 if torch.any(loss_mask == 1) else 0],
            dtype=torch.int,
            device=device,
        )
        if distributed:
            dist.all_reduce(has_loss_token, op=dist.ReduceOp.MIN)
        if has_loss_token.item() == 0:
            continue
        
        with torch.no_grad():
            target_outputs=model.target_model.model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            output_hidden_states=False)

            last_hidden_state=target_outputs.last_hidden_state
            feature_states=last_hidden_state
            target_logits=model.target_model.lm_head(last_hidden_state)

        
        target_logits=target_logits[:,:-1,:]
        feature_states=feature_states[:,:-1,:].to(model.dtype)

        input_ids=input_ids[:,1:]
        attention_mask=attention_mask[:,:-1]
        loss_mask=loss_mask[:,:-1]

        draft_outputs=model(hidden_states=feature_states,input_ids=input_ids,attention_mask=attention_mask,use_cache=False)
        next_feature_states=draft_outputs['next_feature_states']
        draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
        draft_logits=model.lm_head(draft_hidden_states)

        loss1=l1_loss(next_feature_states[:,:-1,:].float(),feature_states[:,1:,:].float())

        loss1=torch.mean(loss1,dim=-1)*loss_mask[:,:-1] 
        loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[:,:-1], dim=-1)
        loss1=loss1.mean()
        loss1=loss1*2

        with torch.no_grad():
            target_logits=target_logits[:,1:,:].float().softmax(dim=-1).detach()
        draft_logits=draft_logits[:,:-1,:].float().softmax(dim=-1)

        plogp=target_logits*torch.log(draft_logits)
        loss2=torch.sum(plogp,dim=-1)*loss_mask[:,:-1]
        loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[:,:-1], dim=-1)
        loss2= - loss2.mean()

        loss2=loss2*0.1
        
        loss=loss1+loss2
        
        has_valid_loss = torch.tensor(
            [0 if (torch.isnan(loss).any() or torch.isinf(loss).any()) else 1],
            dtype=torch.int,
            device=device,
        )
        if distributed:
            dist.all_reduce(has_valid_loss, op=dist.ReduceOp.MIN)

        if has_valid_loss.item() == 0:
            if feature_states.grad is not None:
                feature_states.grad.zero_()
            
            loss = loss.detach()
            del loss
            del feature_states,next_feature_states,target_logits,draft_logits
            torch.cuda.empty_cache()
        
        else:
            accumulated_step+=1
            loss1_norm = torch.tensor(0.0, device=device)
            loss2_norm = torch.tensor(0.0, device=device)
            
            if (accumulated_step - 1) % accumulation_steps == 0 and not distributed: 
                
                optimizer.zero_grad(set_to_none=True)
                loss2.backward(retain_graph=True)
                loss2_norm=compute_normalized_gradient_l2_norm(model.draft_model.layers[0])
                optimizer.zero_grad(set_to_none=True)
                
                loss1.backward(retain_graph=True)
                loss1_norm=compute_normalized_gradient_l2_norm(model.draft_model.layers[0])
                optimizer.zero_grad(set_to_none=True)
            
            
            loss/=accumulation_steps
            loss.backward()

            valid_positions=loss_mask[:,:-1]
            with torch.no_grad():
                correct_top1,correct_topk,total_valid_tokens=compute_acc(target_logits,draft_logits,valid_positions,k=2)
            
            total_correct_top1.append(correct_top1)
            total_correct_topk.append(correct_topk)
            total_token_nums.append(total_valid_tokens)

            batch_logs.append({
                'loss':loss.item()*accumulation_steps,
                'loss1':loss1.item(),
                'loss2':loss2.item(),
                'loss1_norm':loss1_norm.item(),
                'loss2_norm':loss2_norm.item(),
                'correct_top1':correct_top1,
                'correct_topk':correct_topk,
                'total_valid_tokens':total_valid_tokens
            })

            if is_main_process:
                progress_bar.set_postfix({
                    'step': step,
                    'acc': f'{accumulated_step % accumulation_steps}/{accumulation_steps}',
                    'loss': f'{loss.item()*accumulation_steps:.4f}',
                    'loss1': f'{loss1.item():.4f}',
                    'loss2': f'{loss2.item():.4f}',
                })
        
            if accumulated_step%accumulation_steps==0:
                step+=1
                real_sample_num=sample_num*accumulation_steps

                avg_logs = {
                    "step": step,
                    "loss": round(sum(log["loss"] for log in batch_logs)/len(batch_logs),4),
                    "used_time": round((time.time()-start_time)/60, 3),
                    "loss1": round(sum(log["loss1"] for log in batch_logs)/len(batch_logs),4),
                    "loss2": round(sum(log["loss2"] for log in batch_logs)/len(batch_logs),4),
                    "loss1_norm": sum(log["loss1_norm"] for log in batch_logs),
                    "loss2_norm": sum(log["loss2_norm"] for log in batch_logs),
                    "top1_acc": round(sum(log['correct_top1'] for log in batch_logs)/sum(log['total_valid_tokens'] for log in batch_logs),4),
                    "topk_acc": round(sum(log['correct_topk'] for log in batch_logs)/sum(log['total_valid_tokens'] for log in batch_logs),4),
                    f"last{sample_num}_top1_acc": round(sum(total_correct_top1[-real_sample_num:])/sum(total_token_nums[-real_sample_num:]),4),
                    f"last{sample_num}_topk_acc": round(sum(total_correct_topk[-real_sample_num:])/sum(total_token_nums[-real_sample_num:]),4),
                }

                if is_main_process:
                    progress_bar.set_postfix({
                        'step': step,
                        'loss': avg_logs['loss'],
                        'top1': avg_logs['top1_acc'],
                        'topk': avg_logs['topk_acc'],
                        'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                    })
                    
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(avg_logs) + '\n')

                total_correct_top1=total_correct_top1[-real_sample_num:]
                total_correct_topk=total_correct_topk[-real_sample_num:]
                total_token_nums=total_token_nums[-real_sample_num:]
                    
                batch_logs.clear()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                if step%8000==0 and step!=0 and is_main_process:
                    save_draft_model(model, f'{saved_model_dir}/step{step}.pth')
                
                if (step*accumulation_steps)%16==0:
                    torch.cuda.empty_cache()
    

if is_main_process:
    save_draft_model(model, f'{saved_model_dir}/step{step}.pth')

if distributed:
    dist.barrier()
    dist.destroy_process_group()
