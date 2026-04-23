"""
Sample from the trained ConditionedGPT model and save discrete tokens.
"""
import argparse
import os
import pathlib
import yaml
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch

from encoded_dataset import EncodedEMGDataset
from encoded_model import GPTConfig, ConditionedGPT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument("--num_samples", type=int, default=10, help="Number of sequences to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    # 1. Load config
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    seed = args.seed
    device = config.get("device", "cuda")
    dtype = config.get("dtype", "bfloat16")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, config['exp_name'])

    # 2. Find and Load the Latest Checkpoint
    iter_folders = [f for f in os.listdir(save_dir) if f.startswith('iter_')]
    if not iter_folders:
        raise ValueError(f"No checkpoints found in {save_dir}")
    
    # Sort folders to find the one with the highest iteration number
    iter_folders.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
    ckpt_folder = os.path.join(save_dir, iter_folders[0])
    
    print(f"Loading checkpoint from: {ckpt_folder}")
    ckpt_path = os.path.join(ckpt_folder, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 3. Initialize Model
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = ConditionedGPT(gptconf)
    
    # Handle DDP / Compiled prefixes if they exist
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # 4. Load Dataset to get real prompt starting tokens
    # Use val_data_path (unseen) for sampling if available, otherwise fallback to encoded_data_path
    data_file_full_path = config.get('val_data_path', config.get('encoded_data_path', "../data/encoded_df.csv"))
    print(f"Loading source dataset for sampling from: {data_file_full_path}")
    test_dataset = EncodedEMGDataset(
        csv_files=[data_file_full_path],
        filter_class=None
    )

    # Fetch real samples to act as seed prompts
    real_x, _, real_labels = test_dataset.sample(args.num_samples)
    real_x = real_x.to(device)
    
    # We want to condition the model using random classes (or we can use the real classes of the prompts)
    # Let's generate random gesture targets between 0 and num_classes-1
    num_classes = config.get("num_classes", 17)
    target_labels = torch.randint(0, num_classes, (args.num_samples,), device=device)
    
    print(f"Targeting generation for gesture classes: {target_labels.cpu().numpy()}")

    # 5. Generate Data
    prompt_size = config.get('prompt_size', 1) # Usually 1 token is enough to start
    total_len = config.get('block_size', 74) + 1 # e.g., 75 tokens total
    num_new_tokens = total_len - prompt_size

    # Slice the real data to get just the starting tokens
    x_prompt = real_x[:, :prompt_size]

    print(f"Generating {num_new_tokens} tokens autoregressively...")
    with torch.no_grad():
        with ctx:
            # Generate shape: (Batch, Total_Sequence_Length)
            generated_tokens = model.generate(
                x_prompt, 
                max_new_tokens=num_new_tokens, 
                labels=target_labels, # Condition on our random classes
                temperature=config.get('temperature', 0.8),
                top_k=config.get('top_k', 10)
            )

    # 6. Save outputs for the VQ-VAE decoder
    gen_tokens_np = generated_tokens.cpu().numpy()
    labels_np = target_labels.cpu().numpy()

    # Create a DataFrame matching original encoded_df.csv format
    columns = ["gt"] + [f"col_{i}" for i in range(total_len)]
    
    data = np.concatenate([labels_np.reshape(-1, 1), gen_tokens_np], axis=1)
    df_generated = pd.DataFrame(data, columns=columns)

    # Save to disk
    out_file = os.path.join(save_dir, "unseen_synthetic_encoded_samples.csv")
    df_generated.to_csv(out_file, index=False)
    
    print(f"\nSuccessfully generated {args.num_samples} samples.")
    print(f"Saved discrete tokens to: {out_file}")
    print("\nFirst 3 generated samples (Label | Tokens):")
    print(df_generated.head(3))

    del model
    torch.cuda.empty_cache()