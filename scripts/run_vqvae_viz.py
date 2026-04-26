import os
import yaml
import torch
import argparse
from VQVAE.model import SDformerVQVAE
from VQVAE.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae_config', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    # 1. Load Config
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize Model
    model = SDformerVQVAE(vq_config).to(device)
    
    # 3. Load Weights
    vq_name = vq_config['name']
    ckpt_path = f"VQVAE/models/{vq_name}/final_model.pth"
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded VQ-VAE weights from {ckpt_path}")
    else:
        print(f"Error: VQ-VAE checkpoint not found at {ckpt_path}")
        return

    # 4. Initialize Visualizer
    viz = Visualizer(model, device, vq_config)
    
    # Override save_dir if provided
    if args.save_dir:
        viz.save_dir = args.save_dir
        os.makedirs(viz.save_dir, exist_ok=True)

    # 5. Run VQ-VAE specific plots
    print("Generating Codebook t-SNE and Atomic Patterns...")
    viz.visualize_codebook()

    print(f"VQ-VAE visualizations completed. Results in: {viz.save_dir}")

if __name__ == "__main__":
    main()
