"""
References:
1) nanoGPT by Karpathy:
https://github.com/karpathy/nanoGPT
"""

import glob
import math
import os
import pathlib
import pickle
import re
import socket
import shutil

# os.environ['TORCH_USE_CUDA_DSA'] = '1'  # for debugging purpose
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from chatemg_dataset import ChatEMGDataset
from model import GPTConfig, GPT_interchannel

#load yaml 
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

print("LIBS LOADED")
# -----------------------------------------------------------------------------

# # wandb logging
# wandb_log = False  # disabled by default
# wandb_project = "chatemg"  # default project name
# # data
# gradient_accumulation_steps = 1  # used to simulate larger batch sizes
# batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
# block_size = 256
# split = 0.8  # train/val split
# ckpt_path = None
# # model
# model_type = "GPT_interchannel"
# token_embedding_type = "basic_sum"
# n_layer = 12
# n_head = 8
# n_embd = 256
# dropout = 0.2
# bias = False  # do we use bias inside LayerNorm and Linear layers?
# # adamw optimizer
# learning_rate = 1e-3  # max learning rate
# max_iters = 100000
# weight_decay = 1e-1
# beta1 = 0.9
# beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
# grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# # learning rate decay settings
# decay_lr = True  # whether to decay the learning rate
# warmup_iters = 2000  # how many steps to warm up for
# lr_decay_iters = 100000  # should be ~= max_iters per Chinchilla
# min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# # DDP settings
# backend = "nccl"  # 'nccl', 'gloo', etc.
# # system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
# dtype = "float16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# compile = True  # use PyTorch 2.0 to compile the model to be faster
# # preprocessing
# median_filter_size = 9  # 1 means no median filter





# converted_data_path = "../data/"
# sensor_type = "emg"
# axis = "x" 
# #todo make decison on this 

# print("PARAMS SET")

# # -----------------------------------------------------------------------------
# config_keys = [
#     k
#     for k, v in globals().items()
#     if not k.startswith("_") and isinstance(v, (int, float, bool, str, list))
# ]
# print("1")
# exec(open("configurator.py").read())  # overrides from command line or config file
# config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# print("2")

# -----------------------------------------------------------------------------
# DDP initialization
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    dist.init_process_group(backend=config['backend'])
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    if config['gradient_accumulation_steps'] % ddp_world_size == 0:
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        if master_process:
            print(f"WARNING: gradient_accumulation_steps ({config['gradient_accumulation_steps']}) not divisible by ddp_world_size ({ddp_world_size})")
            print(f"Setting gradient_accumulation_steps to 1 per GPU. Effective total will be {ddp_world_size}")
        config['gradient_accumulation_steps'] = 1
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

model_files_base_directory = os.path.join(
    pathlib.Path(__file__).resolve().parent.__str__(), "models"
)
# timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
exp_name = config['exp_name']  #in this case experiment names should be unique otherwise rewrite # _{socket.gethostname()}_{timestr}"
save_dir = os.path.join(model_files_base_directory, exp_name)

tokens_per_iter = config['gradient_accumulation_steps'] * ddp_world_size * config['batch_size'] * config['block_size']
print(f"tokens per iteration will be: {tokens_per_iter:,}")
if master_process:
    os.makedirs(save_dir, exist_ok=True)


shutil.copy("config.yaml", os.path.join(save_dir, "config.yaml"))

np.random.seed(1337 + seed_offset)  # dataset is using numpy
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.float16,
    "float16": torch.float16,
}[config['dtype']]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

data_file_full_path = os.path.join(config['converted_data_path'], f"merged_{config['sensor_type']}.csv")
emg_data_paths = [os.path.join(config['converted_data_path'], subject_id, f"converted_emg.csv") for subject_id in config['participants_list_ids']]

# should be the converted one
sample_data_files = emg_data_paths if config['sensor_type'] == "emg" else [
    data_file_full_path,
]

split_seed = 42

# config.update({"sample_data_files": sample_data_files})
# config.update({"split_seed": split_seed})

# for random split, standardization uses mean and std of all the data for both train and test sets
dataset = ChatEMGDataset(
    csv_files=sample_data_files,
    filter_class=config['filter_class'],
    block_size=config['block_size'],
    median_filter_size=config['median_filter_size'],
    shift=config['shift'],
    flip=config['flip'],
    sensor_type=config['sensor_type'],
    which_file= "train"
)
#split the dataset
train_dataset, test_dataset = random_split(
    dataset, [config['split'], 1 - config['split']], generator=torch.Generator().manual_seed(split_seed)
)

# Create samplers for DDP
if ddp:
    train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, seed=split_seed)
    test_sampler = DistributedSampler(test_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False, seed=split_seed)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], sampler=test_sampler)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

if master_process:
    print(
        f"number of training samples: {len(train_dataset)}, number of test samples: {len(test_dataset)}"
    )


def get_batch(split):
    if split == "train":
        return next(iter(train_dataloader))
    elif split == "val":
        return next(iter(test_dataloader))


iter_num = 0
best_val_loss = 1e9
vocab_size = 1000

# model init
model_args = dict(
    n_layer=config['n_layer'],
    n_head=config['n_head'],
    n_embd=config['n_embd'],
    block_size=config['block_size'],
    bias=config['bias'],
    vocab_size=None,
    dropout=config['dropout'],
    model_type=config['model_type'],
    token_embedding_type=config['token_embedding_type'],
)
print(f"Token Embedding Type is set to {config['token_embedding_type']  }")

# init a new model from scratch
print("Initializing a new model from scratch")
model_args["vocab_size"] = vocab_size
gptconf = GPTConfig(**model_args)
model = GPT_interchannel(gptconf)

# crop down the model block size if desired, using model surgery
if config['block_size'] < model.config.block_size:
    model.crop_block_size(config['block_size'])
    model_args["block_size"] = config["block_size"]  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == "float16"))

# optimizer - create BEFORE wrapping with DDP
optimizer = model.configure_optimizers(
    config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type
)
checkpoint = None  # free up memory

# compile the model - do this BEFORE wrapping with DDP
if config['compile']:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model, backend="eager")  # requires PyTorch 2.0

# wrap model into DDP container - do this LAST
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# wrap model to access raw model (unwrap DDP/compile if needed)
raw_model = model.module if ddp else model


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out_loss = {}
    out_mse_loss = {}
    out_perplexity = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config['eval_iters'])
        mse_losses = torch.zeros(config['eval_iters'])
        perplexity_arr = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            X, Y = X.to(device), Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
                predicted_idx = logits.argmax(dim=-1)  # B, T
                mse_loss = F.mse_loss(predicted_idx, Y[:, :, 0].float())
                perplexity = torch.exp(loss)
            losses[k] = loss.item()
            mse_losses[k] = mse_loss.item()
            perplexity_arr[k] = perplexity.item()
        out_loss[split] = losses.mean()
        out_mse_loss[split] = mse_losses.mean()
        out_perplexity[split] = perplexity_arr.mean()
    model.train()
    return out_loss, out_mse_loss, out_perplexity


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])


# logging
if config['wandb_log'] and master_process:
    import wandb
    
    # Login for non-interactive environments (servers/LSF)
    # First try to get API key from environment variable
    wandb_api_key = os.environ.get('WANDB_API_KEY', None)
    if wandb_api_key:
        print("Logging into W&B using WANDB_API_KEY environment variable...")
        wandb.login(key=wandb_api_key)
    else:
        # Try to use cached credentials
        print("Warning: WANDB_API_KEY not found. Attempting to use cached credentials...")
        try:
            wandb.login()
        except Exception as e:
            print(f"W&B login failed: {e}")
            print("Please set WANDB_API_KEY environment variable or run 'wandb login' manually")
            wandb_log = False

    if config['wandb_log']:
        wandb.init(project=config['exp_name'], name=config['exp_name'], config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
X, Y = X.to(device), Y.to(device)
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if config['decay_lr'] else config['learning_rate']
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % config['eval_interval'] == 0 and master_process:
        losses, mse_losses, perplexity = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.7f}, val loss {losses['val']:.7f}"
        )
        print(
            f"step {iter_num}: train mse loss {mse_losses['train']:.7f}, val mse loss {mse_losses['val']:.7f}"
        )
        print(
            f"step {iter_num}: train perplexity {perplexity['train']:.7f}, val perplexity {perplexity['val']:.7f}"
        )
        if config['wandb_log']:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "train/mse_loss": mse_losses["train"],
                    "train/perplexity": perplexity["train"],
                    "val/loss": losses["val"],
                    "val/mse_loss": mse_losses["val"],
                    "val/perplexity": perplexity["val"],
                    "lr": lr,
                }
            )
        if losses["val"] < best_val_loss or config['always_save_checkpoint']:
            folder_nm = f'iter_{iter_num:0{len(str(config["max_iters"]))}}_train_{losses["train"]:.7f}_val_{losses["val"]:.7f}'
            best_val_loss = (
                losses["val"] if losses["val"] < best_val_loss else best_val_loss
            )
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
                print(f"saving checkpoint to {os.path.join(save_dir, folder_nm)}")
                print("---\n")
                torch.save(checkpoint, os.path.join(save_dir, folder_nm, "ckpt.pt"))
                with open(
                    os.path.join(save_dir, folder_nm, "info.yml"), "w"
                ) as yaml_file:
                    yaml.dump(info, yaml_file, default_flow_style=False)
    if iter_num == 0 and config['eval_only']:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(config['gradient_accumulation_steps']):
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / config['gradient_accumulation_steps']
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        X, Y = X.to(device), Y.to(device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if config['grad_clip'] != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % config['log_interval'] == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * config['gradient_accumulation_steps']
        print(f"iter {iter_num}: loss {lossf:.7f}, time {dt * 1000:.2f}ms")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > config['max_iters']:
        break

if ddp:
    dist.destroy_process_group()
