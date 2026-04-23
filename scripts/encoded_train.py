"""
Updated for VQ-VAE discrete tokens training.
"""

import math
import os
import pathlib
import time
import shutil
import yaml
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# UPDATE: Import the new classes
from encoded_dataset import EncodedEMGDataset
from encoded_model import GPTConfig, ConditionedGPT

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
args = parser.parse_args()

# load yaml 
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

print("LIBS LOADED")

device = "cuda"

# -----------------------------------------------------------------------------
# DDP initialization
ddp = int(os.environ.get('RANK', -1)) != -1  
if ddp:
    dist.init_process_group(backend=config.get('backend', 'nccl'))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  
    seed_offset = ddp_rank  
    if config['gradient_accumulation_steps'] % ddp_world_size == 0:
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        if master_process:
            print(f"WARNING: gradient_accumulation_steps not divisible by ddp_world_size")
        config['gradient_accumulation_steps'] = 1
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

model_files_base_directory = os.path.join(
    pathlib.Path(__file__).resolve().parent.__str__(), "models"
)
exp_name = config.get('exp_name', 'gesture_transformer')  
save_dir = os.path.join(model_files_base_directory, exp_name)

if master_process:
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(save_dir, os.path.basename(args.config)))

np.random.seed(1337 + seed_offset)  
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True  
device_type = "cuda" if "cuda" in device else "cpu"  

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[config.get('dtype', 'float32')]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# -----------------------------------------------------------------------------
# DATASET INITIALIZATION
# -----------------------------------------------------------------------------
split_seed = 42

# 1. Determine paths
train_data_path = config.get('train_data_path', None)
val_data_path = config.get('val_data_path', None)
encoded_data_path = config.get('encoded_data_path', "../data/encoded_df.csv")

if train_data_path and val_data_path:
    if master_process:
        print(f"Using separate train and val files:\n  Train: {train_data_path}\n  Val: {val_data_path}")
    
    train_dataset = EncodedEMGDataset(
        csv_files=[train_data_path],
        filter_class=config.get('filter_class', None)
    )
    
    test_dataset = EncodedEMGDataset(
        csv_files=[val_data_path],
        filter_class=config.get('filter_class', None)
    )
else:
    raise ValueError("Please provide separate train and val file paths in the config to avoid data leakage.")

batch_size = config.get('batch_size', 64)

# Create samplers for DDP
if ddp:
    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, seed=split_seed)
    test_sampler = DistributedSampler(test_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, seed=split_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if master_process:
    print(f"number of training samples: {len(train_dataset)}, number of test samples: {len(test_dataset)}")


train_iter = iter(train_dataloader)
test_iter = iter(test_dataloader)

def get_batch(split):
    global train_iter, test_iter
    if split == "train":
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            return next(train_iter)
    elif split == "val":
        try:
            return next(test_iter)
        except StopIteration:
            test_iter = iter(test_dataloader)
            return next(test_iter)

iter_num = 0
best_val_loss = 1e9

# -----------------------------------------------------------------------------
# MODEL INITIALIZATION
# -----------------------------------------------------------------------------
model_args = dict(
    n_layer=config.get('n_layer', 8),
    n_head=config.get('n_head', 8),
    n_embd=config.get('n_embd', 512),
    block_size=config.get('block_size', 75),  # Sequence length matches tokenized cols
    vocab_size=config.get('vocab_size', 512), # Codebook size
    num_classes=config.get('num_classes', 17), # Number of unique gestures
    dropout=config.get('dropout', 0.1),
    bias=config.get('bias', True)
)

print("Initializing a new ConditionedGPT model from scratch")
gptconf = GPTConfig(**model_args)
model = ConditionedGPT(gptconf)
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(config.get('dtype') == "float16"))

optimizer = model.configure_optimizers(
    config.get('weight_decay', 0.1), 
    config.get('learning_rate', 3e-4), 
    (config.get('beta1', 0.9), config.get('beta2', 0.99)), 
    device_type
)

if config.get('compile', False):
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model, backend="eager")  

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# -----------------------------------------------------------------------------
# EVALUATION METRICS
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out_loss = {}
    out_acc = {} # Changed from MSE to Accuracy for discrete tokens
    out_perplexity = {}
    model.eval()
    
    eval_iters = config.get('eval_iters', 50)
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        perplexity_arr = torch.zeros(eval_iters)
        for k in range(eval_iters):
            try:
                X, Y, labels = get_batch(split)
            except StopIteration:
                # Refresh iterator if we run out of batches in the eval loop
                if split == "train":
                    train_iter = iter(train_dataloader)
                    X, Y, labels = next(train_iter)
                else:
                    test_iter = iter(test_dataloader)
                    X, Y, labels = next(test_iter)
                    
            X, Y, labels = X.to(device), Y.to(device), labels.to(device)
            
            with ctx:
                logits, loss = model(X, targets=Y, labels=labels)
                predicted_idx = logits.argmax(dim=-1)  
                
                # Accuracy: ratio of exactly predicted tokens
                correct = (predicted_idx == Y).sum().item()
                total = Y.numel()
                accuracy = correct / total
                
                perplexity = torch.exp(loss)
                
            losses[k] = loss.item()
            accuracies[k] = accuracy
            perplexity_arr[k] = perplexity.item()
            
        out_loss[split] = losses.mean()
        out_acc[split] = accuracies.mean()
        out_perplexity[split] = perplexity_arr.mean()
        
    model.train()
    return out_loss, out_acc, out_perplexity

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
def get_lr(it):
    warmup_iters = config.get('warmup_iters', 2000)
    lr_decay_iters = config.get('lr_decay_iters', 100000)
    learning_rate = config.get('learning_rate', 3e-4)
    min_lr = config.get('min_lr', 1e-5)
    
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if config.get('wandb_log', False) and master_process:
    try:
        import wandb

        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        if wandb_api_key:
            print("Logging into W&B using WANDB_API_KEY environment variable...")
            wandb.login(key=wandb_api_key)
        else:
            print("Warning: WANDB_API_KEY not found. Attempting to use cached credentials...")
            try:
                wandb.login()
            except Exception as e:
                print(f"W&B login failed: {e}")
                print("Please set WANDB_API_KEY environment variable or run 'wandb login' manually")
                config['wandb_log'] = False

        if config.get('wandb_log', False):
            wandb.init(project=config.get('wandb_project_name', 'transformer'), name=exp_name, config=config)
    except Exception as e:
        print(f"W&B import/init error: {e}")
        config['wandb_log'] = False

X, Y, labels = get_batch("train")
X, Y, labels = X.to(device), Y.to(device), labels.to(device)

t0 = time.time()
local_iter_num = 0 

max_iters = config.get('max_iters', 100000)
eval_interval = config.get('eval_interval', 500)
grad_accum_steps = config.get('gradient_accumulation_steps', 1)
grad_clip = config.get('grad_clip', 1.0)
log_interval = config.get('log_interval', 10)

while True:
    lr = get_lr(iter_num) if config.get('decay_lr', True) else config.get('learning_rate', 3e-4)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses, accuracies, perplexity = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter_num}: train accuracy {accuracies['train']*100:.2f}%, val accuracy {accuracies['val']*100:.2f}%")
        print(f"step {iter_num}: train perplexity {perplexity['train']:.2f}, val perplexity {perplexity['val']:.2f}")
        
        if config.get('wandb_log', False):
            wandb.log({
                "iter": iter_num,
                "train/loss": losses["train"],
                "train/accuracy": accuracies["train"],
                "train/perplexity": perplexity["train"],
                "val/loss": losses["val"],
                "val/accuracy": accuracies["val"],
                "val/perplexity": perplexity["val"],
                "lr": lr,
            })
            
        if losses["val"] < best_val_loss or config.get('always_save_checkpoint', False):
            folder_nm = f'iter_{iter_num:05d}_train_{losses["train"]:.4f}_val_{losses["val"]:.4f}'
            best_val_loss = min(losses["val"], best_val_loss)
            
            if iter_num > 0:
                os.makedirs(os.path.join(save_dir, folder_nm), exist_ok=True)
                info = {
                    "train_loss": losses["train"].item(),
                    "val_loss": losses["val"].item(),
                    "best_val_loss": best_val_loss.item(),
                    "iter_num": iter_num,
                    "config": config,
                    "model_args": model_args,
                }
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                checkpoint.update(info)
                ckpt_save_path = os.path.join(save_dir, folder_nm, "ckpt.pt")
                print(f"saving checkpoint to {ckpt_save_path}")
                torch.save(checkpoint, ckpt_save_path)
                with open(os.path.join(save_dir, folder_nm, "info.yml"), "w") as yaml_file:
                    yaml.dump(info, yaml_file, default_flow_style=False)
                
                if config.get('wandb_log', False) and master_process:
                    try:
                        import wandb
                        wandb.save(ckpt_save_path)
                    except Exception as e:
                        print(f"W&B save error: {e}")

    if iter_num == 0 and config.get('eval_only', False):
        break

    for micro_step in range(grad_accum_steps):
        with ctx:
            logits, loss = model(X, targets=Y, labels=labels)
            loss = loss / grad_accum_steps

        # immediately prefetch next batch
        try:
            X, Y, labels = get_batch("train")
        except StopIteration:
            train_iter = iter(train_dataloader)
            X, Y, labels = next(train_iter)
            
        X, Y, labels = X.to(device), Y.to(device), labels.to(device)
        
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * grad_accum_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
        
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    dist.destroy_process_group()

if config.get('wandb_log', False) and master_process:
    try:
        import wandb
        wandb.finish()
    except Exception as e:
        print(f"W&B finish error: {e}")