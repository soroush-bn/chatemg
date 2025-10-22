"""
References:
1) nanoGPT by Karpathy:
https://github.com/karpathy/nanoGPT
"""

import math
import os
import pathlib
import socket
import time
from contextlib import nullcontext

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split

from chatemg_dataset import ChatEMGDataset
from model import GPTConfig, GPT_interchannel
print("LIBS LOADED")

# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Convert config to dict for easier access
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Extract config values
    exp_name = cfg.exp_name
    filter_class = cfg.filter_class
    eval_interval = cfg.eval_interval
    log_interval = cfg.log_interval
    eval_iters = cfg.eval_iters
    eval_only = cfg.eval_only
    always_save_checkpoint = cfg.always_save_checkpoint
    init_from = cfg.init_from
    wandb_log = cfg.wandb_log
    wandb_project = cfg.get('wandb_project', 'chatemg')
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    batch_size = cfg.batch_size
    block_size = cfg.block_size
    split = cfg.split
    ckpt_path = cfg.ckpt_path
    model_type = cfg.model_type
    token_embedding_type = cfg.token_embedding_type
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    dropout = cfg.dropout
    bias = cfg.bias
    learning_rate = cfg.learning_rate
    max_iters = cfg.max_iters
    weight_decay = cfg.weight_decay
    beta1 = cfg.beta1
    beta2 = cfg.beta2
    grad_clip = cfg.grad_clip
    decay_lr = cfg.decay_lr
    warmup_iters = cfg.warmup_iters
    lr_decay_iters = cfg.lr_decay_iters
    min_lr = cfg.min_lr
    backend = cfg.backend
    device = cfg.device
    dtype = cfg.dtype
    compile_model = cfg.compile
    median_filter_size = cfg.median_filter_size
    train_csv_files = cfg.train_csv_files
    test_csv_files = cfg.test_csv_files
    sample_data_files = cfg.sample_data_files
    split_seed = cfg.split_seed
    vocab_size = cfg.vocab_size

    print("Configuration loaded:")
    print(OmegaConf.to_yaml(cfg))
    
    model_files_base_directory = os.path.join(
        pathlib.Path(__file__).resolve().parent.__str__(), "models"
    )
    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"{exp_name}_{socket.gethostname()}_{timestr}"
    save_dir = os.path.join(model_files_base_directory, exp_name)

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

np.random.seed(1337)  # dataset is using numpy
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler



participants_list_ids = ["033106b27b","bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d","ecfa481b42","e49db6578f","27f6898a3f","3f858df9cf","9780ed81f4"] #"bc4dd952fe","31afab1e30","97c6aaac2d","7037a93026","98aa5fac2d"]
data_path = "/home/sbaghernezha/data/033106b27b/"
csv_name = "finl_df.csv"
sensor_types = [ "accel"]
axis = ["x"]  

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.float16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)
# should be the converted one 
sample_data = data_path+ csv_name
sample_data_files = [
    sample_data,
]

split_seed = 42

config.update({"sample_data_files": sample_data_files})
config.update({"split_seed": split_seed})

# for random split, standardization uses mean and std of all the data for both train and test sets
dataset = ChatEMGDataset(
    csv_files=sample_data_files,
    filter_class=filter_class,
    block_size=block_size,
    median_filter_size=median_filter_size,
    shift=True,
    flip=True,
)
train_dataset, test_dataset = random_split(
    dataset, [split, 1 - split], generator=torch.Generator().manual_seed(split_seed)
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
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
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
    model_type=model_type,
    token_embedding_type=token_embedding_type,
)
print(f"Token Embedding Type is set to {token_embedding_type}")

# init a new model from scratch
print("Initializing a new model from scratch")
model_args["vocab_size"] = vocab_size
gptconf = GPTConfig(**model_args)
model = GPT_interchannel(gptconf)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args[
        "block_size"
    ] = block_size  # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model,backend="eager")  # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out_loss = {}
    out_mse_loss = {}
    out_perplexity = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        mse_losses = torch.zeros(eval_iters)
        perplexity_arr = torch.zeros(eval_iters)
        for k in range(eval_iters):
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
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=exp_name, config=config)

# training loop
X, Y = get_batch("train")  # fetch the very first batch
X, Y = X.to(device), Y.to(device)
t0 = time.time()
raw_model = model  # unwrap DDP container if needed
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
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
        if wandb_log:
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
        if losses["val"] < best_val_loss or always_save_checkpoint:
            folder_nm = f'iter_{iter_num:0{len(str(max_iters))}}_train_{losses["train"]:.7f}_val_{losses["val"]:.7f}'
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
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch("train")
        X, Y = X.to(device), Y.to(device)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.7f}, time {dt * 1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
