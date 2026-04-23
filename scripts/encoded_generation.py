import os
import pathlib
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from contextlib import nullcontext
import argparse

# UPDATE: Import the new classes
from encoded_model import GPTConfig, ConditionedGPT

def run_batch_generation():
    # 1. Setup & Config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config_path = args.config
    print(f"Reading config from: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = config.get("device", "cuda")
    dtype = config.get("dtype", "bfloat16")
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # 2. Load Model
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, config['exp_name'])
    
    if not os.path.exists(save_dir):
        print(f"Error: Model directory {save_dir} does not exist.")
        return

    iter_folders = sorted([f for f in os.listdir(save_dir) if f.startswith('iter_')], 
                          key=lambda x: int(x.split('_')[1]), reverse=True)
    
    if not iter_folders:
        print(f"Error: No checkpoints found in {save_dir}")
        return

    ckpt_path = os.path.join(save_dir, iter_folders[0], "ckpt.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ConditionedGPT(GPTConfig(**checkpoint["model_args"]))
    
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print("Model loaded and set to eval mode.")

    # 3. Load Data
    # Use train_data_path (seen) for generation if available, otherwise fallback to encoded_data_path
    data_file_full_path = config.get('train_data_path', config.get('encoded_data_path', "../data/encoded_df.csv"))
    print(f"Loading source dataset for generation from: {data_file_full_path}")
    full_df = pd.read_csv(data_file_full_path)
    
    all_labels = torch.tensor(full_df.iloc[:, 0].values, dtype=torch.long).to(device)
    all_tokens = torch.tensor(full_df.iloc[:, 1:].values, dtype=torch.long).to(device)
    
    num_samples = all_tokens.size(0)
    total_window_size = 75
    print(f"Total samples to process: {num_samples}")

    # 4. Ratios
    ratios = [(70, 5), (60, 15), (50, 25), (25, 50)]

    # 5. Generation Loop
    for prompt_size, gen_size in ratios:
        print(f"\n{'='*50}")
        print(f"STARTING RATIO: {prompt_size} Real | {gen_size} Synthetic")
        print(f"{'='*50}")
        
        generated_results = []
        batch_size = 64 
        
        # tqdm creates a nice visual progress bar
        pbar = tqdm(range(0, num_samples, batch_size), desc=f"Gen {gen_size} tokens")
        
        for i in pbar:
            end_idx = min(i + batch_size, num_samples)
            x_prompt = all_tokens[i:end_idx, :prompt_size]
            batch_labels = all_labels[i:end_idx]

            with torch.no_grad():
                with ctx:
                    batch_generated = model.generate(
                        idx=x_prompt,
                        max_new_tokens=gen_size,
                        labels=batch_labels,
                        temperature=config.get('temperature', 0.8),
                        top_k=config.get('top_k', 10)
                    )
            generated_results.append(batch_generated.cpu().numpy())
            
            # Optional: update progress bar with current indices
            pbar.set_postfix({"batch": f"{end_idx}/{num_samples}"})

        print(f"Merging results for {prompt_size}_{gen_size}...")
        final_tokens = np.concatenate(generated_results, axis=0)
        
        cols = ["gt"] + [f"col_{j}" for j in range(total_window_size)]
        output_data = np.concatenate([all_labels.cpu().numpy().reshape(-1, 1), final_tokens], axis=1)
        df_out = pd.DataFrame(output_data, columns=cols)
        print(f"Saving generated dataset for ratio {prompt_size}:{gen_size}...")
        
        file_name = f"seen_synthetic_df_{prompt_size}_{gen_size}.csv"
        save_path = os.path.join(save_dir, file_name)
        df_out.to_csv(save_path, index=False)
        print(f"SUCCESS: Saved to {save_path}")

    print("\nAll tasks complete.")

if __name__ == "__main__":
    run_batch_generation()
