import os
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
from decoder import VQVAESignalDecoder

def compare_real_vs_synthetic(config_path, vq_config_path, participant_idx=0):
    # 1. Load Configs
    with open(config_path, "r") as f:
        tr_config = yaml.safe_load(f)
    with open(vq_config_path, 'r') as f:
        vq_config = yaml.safe_load(f)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    
    # We assume we are running from the 'scripts' directory
    base_model_dir = os.path.join("models", exp_name)
    vq_ckpt_path = f"VQVAE/models/{vq_name}/final_model.pth"

    # 2. Initialize Decoder
    decoder = VQVAESignalDecoder(vqvae_model_path=vq_ckpt_path, vqvae_config=vq_config)
    
    # 3. Define Ratios and Paths
    ratios = ["70_5", "60_15", "50_25", "25_50"]
    real_encoded_path = tr_config['train_data_path'] 
    
    print(f"Loading real encoded data from: {real_encoded_path}")
    df_real = pd.read_csv(real_encoded_path)
    
    # Create output directory for comparison plots
    save_dir = os.path.join(base_model_dir, "reconstruction_comparisons")
    os.makedirs(save_dir, exist_ok=True)

    num_classes = tr_config.get('num_classes', 17)
    
    for gesture_id in range(num_classes):
        print(f"Processing Gesture {gesture_id}...")
        
        # Filter real data for this gesture and participant
        real_gesture_data = df_real[df_real['gt'] == gesture_id]
        if len(real_gesture_data) <= participant_idx:
            print(f"  Warning: Not enough real samples for gesture {gesture_id}, participant {participant_idx}. Skipping.")
            continue
            
        real_sample = real_gesture_data.iloc[[participant_idx]]
        
        # Decode Real
        # Extract token columns
        token_cols = [c for c in real_sample.columns if c.startswith('col_')]
        real_tokens = real_sample[token_cols].values
        real_signal = decoder.decode_window(real_tokens)[0] # (Time, Channels)

        # Prepare Grid Plot
        fig, axes = plt.subplots(len(ratios), 2, figsize=(16, 4 * len(ratios)), sharex=True)
        fig.suptitle(f"Real vs Synthetic Reconstruction | Gesture {gesture_id} | Participant {participant_idx}", fontsize=18)

        for i, ratio in enumerate(ratios):
            synth_path = os.path.join(base_model_dir, f"seen_synthetic_df_{ratio}.csv")
            
            if not os.path.exists(synth_path):
                print(f"  Warning: Synth file {synth_path} not found. Skipping ratio {ratio}.")
                axes[i, 0].text(0.5, 0.5, "Real Data Plotting...", ha='center')
                axes[i, 1].text(0.5, 0.5, f"Missing: {ratio}", ha='center')
                continue
            
            # Load Synth
            df_synth = pd.read_csv(synth_path)
            synth_gesture_data = df_synth[df_synth['gt'] == gesture_id]
            
            if len(synth_gesture_data) <= participant_idx:
                print(f"  Warning: Not enough synth samples for gesture {gesture_id}, ratio {ratio}. Skipping.")
                axes[i, 1].text(0.5, 0.5, "No Synth Data", ha='center')
            else:
                synth_sample = synth_gesture_data.iloc[[participant_idx]]
                synth_tokens = synth_sample[token_cols].values
                synth_signal = decoder.decode_window(synth_tokens)[0]

                # Plot Real (Left)
                for c in range(real_signal.shape[1]):
                    axes[i, 0].plot(real_signal[:, c], alpha=0.7, label=f"Ch {c}" if i==0 else "")
                axes[i, 0].set_title(f"Real (Reconstructed) | Gesture {gesture_id}", fontweight='bold')
                axes[i, 0].set_ylabel(f"Ratio {ratio}\nAmplitude")
                axes[i, 0].grid(alpha=0.3)

                # Plot Synthetic (Right)
                for c in range(synth_signal.shape[1]):
                    axes[i, 1].plot(synth_signal[:, c], alpha=0.7)
                axes[i, 1].set_title(f"Synthetic (Ratio {ratio}) | Gesture {gesture_id}", fontweight='bold')
                axes[i, 1].grid(alpha=0.3)

        if i == 0:
            axes[0, 0].legend(loc='upper right', fontsize='small')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"compare_g{gesture_id}_p{participant_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"\nAll comparison plots saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vqvae_config', type=str, required=True)
    parser.add_argument('--participant', type=int, default=0)
    args = parser.parse_args()
    
    compare_real_vs_synthetic(args.config, args.vqvae_config, args.participant)
