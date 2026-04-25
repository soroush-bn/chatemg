import os
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
from decoder import VQVAESignalDecoder

def visualize_ratios(config_path, vq_config_path, participant_idx=0, gesture_id=0):
    # 1. Load Configs
    with open(config_path, "r") as f:
        tr_config = yaml.safe_load(f)
    with open(vq_config_path, 'r') as f:
        vq_config = yaml.safe_load(f)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    base_model_dir = os.path.join("models", exp_name)
    vq_ckpt_path = f"VQVAE/models/{vq_name}/final_model.pth"

    # 2. Initialize Decoder
    decoder = VQVAESignalDecoder(vqvae_model_path=vq_ckpt_path, vqvae_config=vq_config)
    
    # 3. Define the files we want to compare
    ratios = ["70_5", "60_15", "50_25", "25_50"]
    real_data_path = tr_config['train_data_path'] # The real encoded baseline
    
    files_to_load = [real_data_path] + [os.path.join(base_model_dir, f"seen_synthetic_df_{r}.csv") for r in ratios]
    labels = ["Real Ground Truth"] + [f"Ratio {r} (Prompt_Gen)" for r in ratios]

    # 4. Prepare Plot
    fig, axes = plt.subplots(len(files_to_load), 1, figsize=(15, 3 * len(files_to_load)), sharex=True)
    fig.suptitle(f"EMG Generation Comparison | Participant {participant_idx} | Gesture {gesture_id}", fontsize=18)

    for i, file_path in enumerate(files_to_load):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping row.")
            axes[i].text(0.5, 0.5, f"File Not Found: {os.path.basename(file_path)}", ha='center')
            continue
        
        # Load tokens
        df = pd.read_csv(file_path)
        
        # Filter for the specific gesture
        # Note: We assume the dataset order matches the training split
        # We find the first occurrence of the gesture for simplicity
        gesture_data = df[df['gt'] == gesture_id]
        if len(gesture_data) == 0:
            axes[i].text(0.5, 0.5, f"Gesture {gesture_id} not found in file", ha='center')
            continue
            
        # Take one specific sample (index 0 of the filtered set)
        sample_tokens = gesture_data.iloc[[participant_idx]] # Using participant_idx as sample index
        
        # Decode to raw signal
        # signals shape: (1, TimeSteps, Channels)
        raw_signals, _ = decoder.decode_dataset_from_df(sample_tokens)
        signal = raw_signals[0] # (TimeSteps, Channels)
        
        # Plot all 8 channels
        for c in range(signal.shape[1]):
            axes[i].plot(signal[:, c], alpha=0.7)
        
        axes[i].set_title(labels[i], fontweight='bold')
        axes[i].grid(alpha=0.3)
        axes[i].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time Steps (Samples)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_path = os.path.join(base_model_dir, f"ratio_comparison_p{participant_idx}_g{gesture_id}.png")
    plt.savefig(save_path, dpi=200)
    print(f"Comparison plot saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vqvae_config', type=str, required=True)
    parser.add_argument('--participant', type=int, default=0)
    parser.add_argument('--gesture', type=int, default=0)
    args = parser.parse_args()
    
    visualize_ratios(args.config, args.vqvae_config, args.participant, args.gesture)
