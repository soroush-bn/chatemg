import os
import yaml 
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse

# Import your modules
from dataset import EMGDataset
from evaluation import evaluate_model
from model import SDformerVQVAE
from train import train_vqvae
from visualizer import Visualizer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run VQVAE pipeline')
parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
args = parser.parse_args()

# --- 1. Load Config & Setup Directories ---
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Create unique folder for this run
save_dir = f"./models/{config['name']}/"
os.makedirs(save_dir, exist_ok=True)

# Save a copy of the config for reproducibility
with open(os.path.join(save_dir, "config.yaml"), "w") as file:
    yaml.dump(config, file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline initialized on {device}. Saving results to: {save_dir}")

# --- 2. Initialize Full Dataset ---
# This uses the new EMGDataset that exposes self.df
full_dataset = EMGDataset(window_size=config['window_size'], stride=config['stride'])

# --- 3. Train/Validation Split ---
# 80% Train, 20% Val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Data Split: {len(train_dataset)} Training | {len(val_dataset)} Validation")

# Create DataLoaders
# Drop_last=True is important for batch norm stability in VQ-VAE
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

# Sanity Check
first_batch = next(iter(train_loader))
assert first_batch.shape == (config['batch_size'], 8, config['window_size']), \
    f"Unexpected batch shape: {first_batch.shape}"

# --- 4. Initialize Model ---
model = SDformerVQVAE(config).to(device)
learning_rate = float(config['learning_rate'])
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- 5. Train Model ---
print("\n--- Starting Training ---")
model = train_vqvae(model, train_loader, device, optimizer, config)

# --- 6. Save Final Model ---
model_path = os.path.join(save_dir, "final_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# --- 7. Evaluate Model ---
print("\n--- Starting Evaluation ---")
evaluate_model(model, val_loader, device, config)

# --- 8. Visualization Suite ---
print("\n--- Generating Diagnostic Plots ---")
viz = Visualizer(model, device, config)

print("1/4. Visualizing Codebook (t-SNE & Patterns)...")
viz.visualize_codebook()

print("2/4. Checking Data Distribution (Real vs Recon t-SNE)...")
viz.plot_data_distribution(val_loader, num_samples=2000)

print("3/4. Plotting Random Sample Reconstructions...")
viz.plot_single_reconstruction(val_loader, sample_index=0)
viz.plot_single_reconstruction(val_loader, sample_index=10)
viz.plot_single_reconstruction(val_loader, sample_index=20)

print("4/4. Tracing Full Gesture Pipeline...")

label_map = {
    "Thumb Extension":0, "index Extension":1, "Middle Extension":2, "Ring Extension":3,
    "Pinky Extension":4, "Thumbs Up":5, "Right Angle":6, "Peace":7, "OK":8, "Horn":9, 
    "Hang Loose":10, "Power Grip":11, "Hand Open":12, "Wrist Extension":13, 
    "Wrist Flexion":14, "Ulnar deviation":15, "Radial Deviation":16    
}

try:
    viz.plot_gesture_pipeline(
        full_dataset.df, 
        label_name="Power Grip", 
        label_map=label_map, 
        duration_sec=2.0
    )
except Exception as e:
    print(f"Could not plot specific gesture pipeline: {e}")
    print("Trying fallback: High Energy Burst...")
    # Fallback: Just plot the loudest 2 seconds
    viz.plot_gesture_pipeline(full_dataset.df, label_name="Loudest Burst", duration_sec=2.0)

print(f"\nPipeline Completed Successfully! All results in: {save_dir}")