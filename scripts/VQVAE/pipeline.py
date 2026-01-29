import os
import yaml 
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

# Import your modules
from dataset import EMGDataset
from evaluation import evaluate_model
from model import SDformerVQVAE
from train import train_vqvae

# 1. Load Config
with open("vqvae_config.yaml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Assignment ready on {device}.")

# 2. Initialize Full Dataset
full_dataset = EMGDataset(window_size=config['window_size'], stride=config['stride'])

# --- IMPROVEMENT: Train/Validation Split ---
# We use 80% for training and 20% for evaluation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Data Split: {len(train_dataset)} Training samples | {len(val_dataset)} Validation samples")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

# 3. Sanity Check (Efficiently)
first_batch = next(iter(train_loader))
print(f"Batch Shape: {first_batch.shape}")
assert first_batch.shape == (config['batch_size'], 8, config['window_size']), \
    f"Unexpected batch shape: {first_batch.shape}"

# 4. Create Model
model = SDformerVQVAE(config).to(device)
learning_rate = float(config['learning_rate'])
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. Train Model (Pass the TRAIN loader)
# Note: We capture the returned model, though it updates in place anyway
model = train_vqvae(model, train_loader, device, optimizer, config)

# 6. Save Final Model
os.makedirs("./models/", exist_ok=True)
save_path = f"./models/{config['name']}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# 7. Evaluate Model (Pass the VALIDATION loader)
print("\n--- Starting Evaluation on Unseen Data ---")
evaluate_model(model, val_loader, device, config)

print("Pipeline completed successfully.")