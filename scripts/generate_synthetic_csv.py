"""
Sample from a trained model
"""
import argparse
import os
import pathlib

from contextlib import nullcontext

import pandas as pd
import numpy as np
import torch

import misc_utils as mu
from model import GPTConfig, GPT_interchannel
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument("--seed", type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument("--independent", type=bool, default=False, help='Independent generation mode')
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # some other parameters
    # -----------------------------------------------------------------------------
    init_from = "resume"
    temperature = config['temperature']  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
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
    # emg_data_paths if config['sensor_type'] == "emg" else 
    sample_data_files = [
        data_file_full_path,
    ]
    generate_dataset = ChatEMGDataset(
        csv_files=sample_data_files,
        filter_class=config['filter_class'],
        block_size=config['token_len'],
        vocab_size=config['vocab_size'],
        ds_factor= config['ds_factor'],
        median_filter_size=config['median_filter_size']
        if config['median_filter_size'] is not None
        else checkpoint["config"]["median_filter_size"],
        sensor_type=config['sensor_type'],
        which_file= "generate",
        location= config['location']
    )

    all_chunks = generate_dataset.get_all_chunks()
    prompts_method1 = [] 
    masking_percentage = 0.2  # mask last 10% of each rep
    generated_reps_method1 = []
    for chunk in all_chunks: 
        reps =generate_dataset.get_one_rep(chunk, None )
        for rep in reps:
            cut_off = int((1 - masking_percentage) * rep.shape[0])
            prompt = rep[:cut_off, :]
            prompts_method1.append(prompt)
            x= torch.tensor(prompt, device=device).unsqueeze(0)  # Add batch dimension: (samples, channels) -> (1, samples, channels)
            with torch.no_grad():
                with ctx:
                    num_new_tokens = int(rep.shape[0] * masking_percentage)
                    Y = model.generate(
                        x,
                        num_new_tokens,
                        temperature=temperature,
                        top_k=config['top_k'],
                        prompt_size=config['prompt_size'],
                        independent=args.independent,
                    )
                    Y = Y[0, -num_new_tokens:, :].cpu().numpy()  # (1, total_length, channels) -> (num_new_tokens, channels)
                    generated_rep = np.concatenate((prompt, Y), axis=0)
                    generated_reps_method1.append(generated_rep)
    synthetic_x = np.concatenate(generated_reps_method1, axis=0)
    save_path = os.path.join(save_dir, f"synthetic_data_{config['filter_class']}_{config['exp_name']}.csv")
    df= pd.DataFrame(synthetic_x, columns=[f"emg{i}" for i in range(synthetic_x.shape[1])])
    print("Method 1 (last 10% masking) - df.describe(): ", df.describe())
    df.to_csv(save_path, index=False)
    print(f"Saved method 1 synthetic data to: {save_path}")
    generated_reps_method2 = []
    prompts_method2 = []
    for chunk in all_chunks: 
        rep3 =generate_dataset.get_not_one_rep(chunk, 3 )
        prompt=rep3
        prompts_method2.append(prompt)
        x= torch.tensor(prompt, device=device).unsqueeze(0)  # Add batch dimension: (samples, channels) -> (1, samples, channels)
        with torch.no_grad():
            with ctx:
                num_new_tokens = int(chunk.shape[0]-rep3.shape[0] )
                Y = model.generate(
                    x,
                    num_new_tokens,
                    temperature=temperature,
                    top_k=config['top_k'],
                    prompt_size=config['prompt_size'],
                    independent=args.independent,
                )
                Y = Y[0, -num_new_tokens:, :].cpu().numpy()  # (1, total_length, channels) -> (num_new_tokens, channels)
                generated_rep = np.concatenate((prompt, Y), axis=0)
                generated_reps_method2.append(generated_rep)
    synthetic_x = np.concatenate(generated_reps_method2, axis=0)
    save_path = os.path.join(save_dir, f"synthetic_data_lastrep_{config['filter_class']}_{config['exp_name']}.csv")
    df= pd.DataFrame(synthetic_x, columns=[f"emg{i}" for i in range(synthetic_x.shape[1])])
    print("Method 2 (fourth repetition masking) - df.describe(): ", df.describe())
    df.to_csv(save_path, index=False)
    print(f"Saved method 2 synthetic data to: {save_path}")



    print("done")
