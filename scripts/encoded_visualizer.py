"""
Decodes synthetic discrete tokens back to raw signals and visualizes them.
"""
import argparse
import os
import yaml
import numpy as np
import matplotlib
import pathlib
import faulthandler
faulthandler.enable()

# Use 'Agg' backend so matplotlib doesn't try to open a window on the headless server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the decoder class we just created
from decoder import VQVAESignalDecoder

def plot_synthetic_signals(signals, labels, save_path, max_plots=9):
    """
    Creates a grid plot of the generated multi-channel signals.
    signals shape: (Samples, TimeSteps, Channels)
    """
    num_samples = min(len(signals), max_plots)
    
    # Determine grid size (e.g., 3x3 for 9 samples)
    cols = 3
    rows = int(np.ceil(num_samples / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i in range(num_samples):
        ax = axes[i]
        signal = signals[i]  # Shape: (TimeSteps, Channels)
        label = labels[i]
        
        # Plot each channel
        num_channels = signal.shape[1]
        for c in range(num_channels):
            ax.plot(signal[:, c], label=f'Ch {c+1}', alpha=0.8, linewidth=1.5)
            
        ax.set_title(f"Synthetic Gesture Class: {label}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Only add legend to the first subplot to save space
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    # Hide any unused subplots
    for j in range(num_samples, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')

    parser.add_argument("--vqvae_ckpt", type=str, default='./VQVAE/models/tuned/final_model.pth', help="Path to your trained VQ-VAE .pt file")
    parser.add_argument("--vqvae_config", type=str,default='./VQVAE/models/tuned/config.yaml', help="Path to the config.yaml used to train the VQ-VAE")
    parser.add_argument("--num_plots", type=int, default=9, help="Number of generated samples to plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    with open(args.config, "r") as file:
        transformer_config = yaml.safe_load(file)
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, transformer_config['exp_name'])
    # 1. Load the VQ-VAE configuration
    with open(args.vqvae_config, 'r') as file:
        vqvae_config = yaml.safe_load(file)
        
    # 2. Initialize the decoder
    decoder = VQVAESignalDecoder(
        vqvae_model_path=args.vqvae_ckpt, 
        vqvae_config=vqvae_config
    )
    
    # 3. Decode the generated tokens back to continuous signals
    print("\n--- Starting Decoding Process ---")
    raw_signals, labels = decoder.decode_dataset(
        csv_path=f'{save_dir}/synthetic_encoded_samples.csv',
        save_dir=save_dir
    )
    
    # 4. Visualize and save the plots
    print("\n--- Starting Visualization ---")
    plot_filename = os.path.join(save_dir, "synthetic_signals_grid.png")
    plot_synthetic_signals(raw_signals, labels, save_path=plot_filename, max_plots=args.num_plots)