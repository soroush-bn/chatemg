import os
import pathlib
import yaml
import torch
import numpy as np
import pandas as pd
from contextlib import nullcontext
from encoded_dataset import EncodedEMGDataset
from encoded_model import GPTConfig, ConditionedGPT

def run_batch_generation():
    # 1. Setup & Config
    config_path = './configs/encoded_config2.yaml'
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    device = config.get("device", "cuda")
    dtype = config.get("dtype", "bfloat16")
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # 2. Load Model (Latest Checkpoint)
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, config['exp_name'])
    iter_folders = sorted([f for f in os.listdir(save_dir) if f.startswith('iter_')], 
                          key=lambda x: int(x.split('_')[1]), reverse=True)
    
    ckpt_path = os.path.join(save_dir, iter_folders[0], "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ConditionedGPT(GPTConfig(**checkpoint["model_args"]))
    
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    # 3. Load ALL samples from the encoded dataset
    data_file_full_path = config.get('encoded_data_path', "../data/encoded_df.csv")
    full_df = pd.read_csv(data_file_full_path)
    
    # Extract labels and token sequences
    # Assuming columns: [gt, col_0, col_1, ..., col_74]
    all_labels = torch.tensor(full_df.iloc[:, 0].values, dtype=torch.long).to(device)
    all_tokens = torch.tensor(full_df.iloc[:, 1:].values, dtype=torch.long).to(device)
    
    num_samples = all_tokens.size(0)
    total_window_size = 75 # Total tokens per sample

    # 4. Define the 5 Ratios (Prompt_Size : Generated_Size)
    ratios = [
        (70, 5),
        (60, 15),
        (50, 25),
        (25, 50),
        (5, 70)
    ]

    # 5. Loop through ratios and generate
    for prompt_size, gen_size in ratios:
        print(f"\n--- Starting Generation: {prompt_size} Real | {gen_size} Synthetic ---")
        
        # Prepare storage for this specific ratio
        generated_results = []
        
        # We process in batches to avoid OOM (Out Of Memory)
        batch_size = 64 
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            
            x_prompt = all_tokens[i:end_idx, :prompt_size]
            batch_labels = all_labels[i:end_idx]

            with torch.no_grad():
                with ctx:
                    # Generate the remainder of the sequence
                    batch_generated = model.generate(
                        idx=x_prompt,
                        max_new_tokens=gen_size,
                        labels=batch_labels,
                        temperature=config.get('temperature', 0.8),
                        top_k=config.get('top_k', 10)
                    )
            generated_results.append(batch_generated.cpu().numpy())

        # Combine batches
        final_tokens = np.concatenate(generated_results, axis=0)
        
        # Create DataFrame
        cols = ["gt"] + [f"col_{j}" for j in range(total_window_size)]
        output_data = np.concatenate([all_labels.cpu().numpy().reshape(-1, 1), final_tokens], axis=1)
        df_out = pd.DataFrame(output_data, columns=cols)

        # Save to disk
        file_name = f"synthetic_df_{prompt_size}_{gen_size}.csv"
        save_path = os.path.join(save_dir, file_name)
        df_out.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    print("\nBatch generation complete for all ratios.")

if __name__ == "__main__":
    run_batch_generation()