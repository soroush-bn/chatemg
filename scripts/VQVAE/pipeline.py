import os
import yaml 
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse

from dataset import EMGDataset
from evaluation import evaluate_model
from model import SDformerVQVAE
from train import train_vqvae
from visualizer import Visualizer

parser = argparse.ArgumentParser(description='Run VQVAE pipeline')
parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

save_dir = f"./models/{config['name']}/"
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "config.yaml"), "w") as file:
    yaml.dump(config, file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Pipeline initialized on {device}. Saving results to: {save_dir}")
# --- 2. Load Dataset ---
print("\n--- Loading Datasets ---")
train_dataset = EMGDataset(window_size=config['window_size'], stride=config['stride'], split='train')
unseen_dataset = EMGDataset(window_size=config['window_size'], stride=config['stride'], split='unseen')

train_dataset.save_df(os.path.join(save_dir, "train_data_preprocessed.csv")) 
unseen_dataset.save_df(os.path.join(save_dir, "unseen_data_preprocessed.csv")) 

print(f"Train dataset: {len(train_dataset)} samples")
print(f"Unseen dataset: {len(unseen_dataset)} samples")
print("$" * 50)

# --- 3. Train/Validation Split ---
# Train on 75% (train split), Test on 25% (unseen split)
val_dataset = unseen_dataset

print(f"Data Split: {len(train_dataset)} Training | {len(val_dataset)} Validation (Unseen)")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)


# Sanity Check
first_batch = next(iter(train_loader))
assert first_batch.shape == (config['batch_size'], 8, config['window_size']), \
    f"Unexpected batch shape: {first_batch.shape}"

# --- 4. Initialize Model ---
model = SDformerVQVAE(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['learning_rate']))

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

print("1/4. Visualizing Codebook...")
viz.visualize_codebook()

print("2/4. Checking Data Distribution...")
viz.plot_data_distribution(val_loader, num_samples=2000)

print("3/4. Plotting Random Sample Reconstructions...")
viz.plot_single_reconstruction(val_loader, sample_index=0)
viz.plot_single_reconstruction(val_loader, sample_index=10)

print("4/4. Tracing Full Gesture Pipeline...")
# Re-load full dataset for visualization and encoding
full_dataset = EMGDataset(window_size=config['window_size'], stride=config['stride'], split='all')

gestures = [11, 8, 17] # Power Grip, OK, Rest
for label_id in gestures:
    for rep in [0, 4]: # Participant 1 & 2
        try:
            viz.plot_gesture_pipeline(full_dataset.df, label_id=label_id, repetition_index=rep)
        except: pass

# --- 9. Generate Encoded Dataset (Codebooks Only) ---
print("\n--- Generating Encoded Dataset (Tokens) ---")
encoded_save_path = os.path.join(save_dir, "encoded_df.csv")

# Create a loader for the ENTIRE dataset
full_loader = DataLoader(full_dataset, batch_size=128, shuffle=False, drop_last=False)

model.eval()
all_codes = []
all_labels = []

print("Mapping labels to windows...")
all_gt_values = full_dataset.df['gt'].values
total_windows = len(full_dataset)
window_centers = [i * full_dataset.stride + full_dataset.window_size // 2 for i in range(total_windows)]

print(f"Processing {total_windows} windows...")

with torch.no_grad():
    batch_start_idx = 0
    for i, batch in enumerate(full_loader):
        batch = batch.to(device)
        current_batch_size = batch.size(0)
        
        # Pass through model
        _,_, _, indices = model(batch)
        
        # --- FIX: Reshape flattened indices ---
        # If indices are [Batch * Time], reshape to [Batch, Time]
        if indices.dim() == 1:
            indices = indices.view(current_batch_size, -1)
            
        # Now shape is [Batch, 75]
        batch_codes = indices.cpu().numpy()
        all_codes.append(batch_codes)
        
        # Get corresponding labels
        batch_indices = range(batch_start_idx, batch_start_idx + current_batch_size)
        
        # Safely grab center indices (clamping to max length of df)
        center_indices = [min(window_centers[idx], len(all_gt_values)-1) for idx in batch_indices]
        batch_labels = all_gt_values[center_indices]
        all_labels.append(batch_labels)
        
        batch_start_idx += current_batch_size
        
        if i % 100 == 0:
            print(f"  Encoded {batch_start_idx} / {total_windows} samples...")

# Concatenate
final_codes = np.concatenate(all_codes, axis=0) # Should now be [Total_Samples, 75]
final_labels = np.concatenate(all_labels, axis=0)

# Debug Print
print(f"Final Codes Shape: {final_codes.shape}") 

# Create DataFrame
token_cols = [f"col_{i}" for i in range(final_codes.shape[1])]
df_encoded = pd.DataFrame(final_codes, columns=token_cols)
df_encoded.insert(0, "gt", final_labels)

# Save
df_encoded.to_csv(encoded_save_path, index=False)
print(f"Encoded dataset saved successfully!")
print(f"File: {encoded_save_path}")
print(f"\nPipeline Completed Successfully! All results in: {save_dir}")