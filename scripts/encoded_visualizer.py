"""
Decodes synthetic discrete tokens back to raw signals and visualizes them.
"""
import argparse
import os
import yaml
from decoder import VQVAESignalDecoder
import numpy as np
import matplotlib
import pathlib
import faulthandler
faulthandler.enable()

# Use 'Agg' backend so matplotlib doesn't try to open a window on the headless server
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        if i >= num_samples: break
        ax = axes[i]
        signal = signals[i]  # Shape: (TimeSteps, Channels)
        label = labels[i]
        
        # Plot each channel
        num_channels = signal.shape[1]
        for c in range(num_channels):
            ax.plot(signal[:, c], label=f'Ch {c+1}', alpha=0.8, linewidth=1.5)
            
        ax.set_title(f"Synthetic Gesture Class: {label}")
        # ax.set_xlabel("Time Steps")
        # ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    for j in range(num_samples, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {save_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to Transformer config (replicate_small.yaml)')
    parser.add_argument("--vqvae_config", type=str, default=None, help="Path to VQ-VAE config")
    parser.add_argument("--vqvae_ckpt", type=str, default=None, help="Path to VQ-VAE weights")
    parser.add_argument("--num_plots", type=int, default=9, help="Number of generated samples to plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    with open(args.config, "r") as file:
        transformer_config = yaml.safe_load(file)
    
    exp_name = transformer_config['exp_name']
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, exp_name)

    # Resolve VQ-VAE paths
    vq_config_path = args.vqvae_config
    vq_ckpt_path = args.vqvae_ckpt

    if vq_config_path is None or vq_ckpt_path is None:
        # If not provided, try to derive from VQ_CONFIG env or default to tuned2
        vq_config_path = vq_config_path or "./VQVAE/tuned_config2.yaml"
        with open(vq_config_path, 'r') as file:
            vq_cfg = yaml.safe_load(file)
        vq_name = vq_cfg.get('name', 'tuned2')
        vq_ckpt_path = vq_ckpt_path or f"./VQVAE/models/{vq_name}/final_model.pth"

    with open(vq_config_path, 'r') as file:
        vqvae_config = yaml.safe_load(file)
        
    # 2. Initialize the decoder
    decoder = VQVAESignalDecoder(
        vqvae_model_path=vq_ckpt_path, 
        vqvae_config=vqvae_config
    )
    
    # 3. Decode the generated tokens back to continuous signals
    print(f"\n--- Starting Decoding Process for {exp_name} ---")
    
    # Prioritize 'unseen_synthetic' files
    sample_file = os.path.join(save_dir, "unseen_synthetic_df_5_70.csv") 
    if not os.path.exists(sample_file):
        # Try generic unseen sample file
        sample_file = os.path.join(save_dir, "unseen_synthetic_encoded_samples.csv")
        
    if not os.path.exists(sample_file):
        # Fallback to any unseen synthetic file
        synth_files = sorted([f for f in os.listdir(save_dir) if f.startswith("unseen_synthetic_")], reverse=True)
        if synth_files:
            sample_file = os.path.join(save_dir, synth_files[0])
            
    if not os.path.exists(sample_file):
        # Last resort fallback to old naming
        synth_files = sorted([f for f in os.listdir(save_dir) if f.startswith("synthetic_df_")], reverse=True)
        if synth_files:
            sample_file = os.path.join(save_dir, synth_files[0])

    if os.path.exists(sample_file):
        print(f"Processing tokens from: {sample_file}")
        raw_signals, labels = decoder.decode_dataset(
            csv_path=sample_file,
            save_dir=save_dir
        )
        
        # 4. Visualize and save the plots
        print("\n--- Starting Visualization ---")
        plot_filename = os.path.join(save_dir, "synthetic_signals_grid.png")
        plot_synthetic_signals(raw_signals, labels, save_path=plot_filename, max_plots=args.num_plots)
    else:
        print(f"No synthetic data found to visualize in {save_dir}")
