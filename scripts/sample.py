"""
Sample from a trained model
"""
import argparse
import os
import pathlib
import pickle
import re
from contextlib import nullcontext
from setuptools._distutils.util import strtobool


import numpy as np
import tiktoken
import torch
from torch.utils.data import random_split

import misc_utils as mu
from model import GPTConfig, GPT_interchannel
import yaml

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
args = parser.parse_args()

#load yaml 
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--num_samples", type=int, default=9)
    parser.add_argument("--nrows", type=int, default=3)
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument(
        "--sample_prompt",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--duplicate",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="ignore num of samples and generate data for the same prompt 9 times",
    )
    parser.add_argument("--independent", type=bool, default=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    # some other parameters
    # -----------------------------------------------------------------------------
    init_from = "resume"
    start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    nrows = 3
    ncols = 3
    temperature = (
       config['temperature']  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    )
    seed = args.seed
    device = config["device"]  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype = config["dtype"]  # 'float32' or 'bfloat16' or 'float16'
    compile = config["compile"]  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # do we need this in training as well?
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
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
    model_files_base_directory = os.path.join(
        pathlib.Path(__file__).resolve().parent.__str__(), "models"
    )
    save_dir = os.path.join(model_files_base_directory, config['exp_name'])

    if init_from == "resume":
        # init from a model saved in a specific directory
        # Find the folder with the highest iteration number
        iter_folders = [f for f in os.listdir(save_dir) if f.startswith('iter_')]
        if not iter_folders:
            raise ValueError(f"No iteration folders found in {save_dir}")
        
        for i in range(config["max_iters"],0,-1*config["eval_interval"]):
            max_iter_folders = [f for f in iter_folders if f.startswith(f"iter_{i}")]
            if max_iter_folders:
                max_iter_folder = max_iter_folders[0]
                break

        ckpt_folder = os.path.join(save_dir, max_iter_folder)
        print(f"Using checkpoint from: {ckpt_folder}")
        ckpt_path = os.path.join(ckpt_folder, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        filter_class = checkpoint["config"]["filter_class"]
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT_interchannel(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    if config['compile']:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)

    from chatemg_dataset import ChatEMGDataset

    #todo make decison on this 
    data_file_full_path = os.path.join(config['converted_data_path'], f"merged_{config['sensor_type']}.csv")
    emg_data_paths = [os.path.join(config['converted_data_path'], subject_id, f"converted_emg.csv") for subject_id in config['participants_list_ids']]

# should be the converted one
    sample_data_files = emg_data_paths if config['sensor_type'] == "emg" else [
    data_file_full_path,
]
    test_dataset = ChatEMGDataset(
        csv_files=sample_data_files,
        filter_class=config['filter_class'],
        block_size=config['token_len'],
        median_filter_size=config['median_filter_size']
        if config['median_filter_size'] is not None
        else checkpoint["config"]["median_filter_size"],
        sensor_type=config['sensor_type'],
        which_file= "sample",
        location= config['location']
    )
    if not args.duplicate:
        real_x = mu.sample_from_dataset(test_dataset, args.num_samples)[0]
    else:
        real_x = mu.sample_from_dataset(test_dataset, 1)[0]
        real_x = np.tile(real_x, (9, 1, 1))

    if args.sample_prompt:
        if args.independent:
            x = torch.tensor(real_x, device=device)
        else:
            x = torch.tensor(real_x[:, : config['prompt_size'], :], device=device)
    else:
        x = torch.tensor(
            [[[0, 0, 0, 0, 0, 0, 0, 0]]] * args.num_samples,
            dtype=torch.long,
            device=device,
        )

    with torch.no_grad():
        with ctx:
            num_new_tokens = config['token_len'] - config['prompt_size']
            Y = model.generate(
                x,
                num_new_tokens,
                temperature=temperature,
                top_k=config['top_k'],
                prompt_size=config['prompt_size'],
                independent=args.independent,
            )
            Y = Y.cpu().numpy()

    # use real_x and Y to compute the mse error over the whole predicted trajectory. Both are numpy array
    mse = mu.compute_mse(real_x, Y, starting_pos=150)
    rmse = np.sqrt(mse)

    mu.plot_emg_chunks_parallel(
        real_x,
        Y,
        rmse=rmse,
        nrows=nrows,
        ncols=ncols,
        vertical_location=None,
        save_fnm=f"real_vs_synthetic_{config['filter_class']}_{config['exp_name']}.png",
        save_dir=save_dir
    )
    print("done")
