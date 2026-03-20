import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import os

from VQVAEmodel import SDformerVQVAE

# --- 1. CNN Architecture ---
class CNNClassifier(nn.Module):
    def __init__(self, code_dim=32, num_classes=17):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=code_dim, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.squeeze(-1)   
        return self.classifier(x)

# --- 2. Helper: Load and Map Data ---
def load_and_embed_data(csv_path, vqvae, device):
    """Loads a CSV and converts discrete tokens to continuous latent embeddings."""
    df = pd.read_csv(csv_path)
    
    gt_raw = df['gt'].values
    labels = gt_raw - gt_raw.min() 
    y = torch.tensor(labels, dtype=torch.long)
    
    indices = torch.tensor(df.drop(columns=['gt']).values, dtype=torch.long).to(device)
    num_samples, seq_len = indices.shape

    with torch.no_grad():
        flat_indices = indices.reshape(-1)
        embeddings = vqvae.quantizer.embedding[flat_indices]
        embeddings = embeddings.reshape(num_samples, seq_len, -1)
        X = embeddings.cpu()
        
    return X, y

# --- 3. Helper: Train Model ---
def train_model(X_train, y_train, device, code_dim=32, epochs=20, lr=0.001):
    """Trains a CNN on the provided dataset and returns the trained model."""
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = CNNClassifier(code_dim=code_dim, num_classes=17).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    return model

# --- 4. Helper: Evaluate Model ---
def evaluate_model(model, X_test, y_test, device):
    """Evaluates a trained model on a test set and returns metrics."""
    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return acc, prec, rec, f1

# --- 5. Main Cross-Validation Pipeline ---
def cross_domain_validation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Configs & Paths ---
    CONFIG_PATH = "./VQVAE/models/tuned/config.yaml" 
    VQVAE_WEIGHTS = "./VQVAE/models/tuned/final_model.pth"
    REAL_DATA_PATH = "/home/sbaghernezha/projects/chatemg/chatemg/data/encoded_df.csv"
    
    SYNTH_DATASETS = {
        "70% Real / 5% Synth": "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/encoded_test_small/synthetic_df_70_5.csv",
        "60% Real / 15% Synth": "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/encoded_test_small/synthetic_df_60_15.csv",
        "50% Real / 25% Synth": "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/encoded_test_small/synthetic_df_50_25.csv",
        "25% Real / 50% Synth": "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/encoded_test_small/synthetic_df_25_50.csv",
        "5% Real / 70% Synth": "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/encoded_test_small/synthetic_df_5_70.csv"
    }

    # --- Initialization ---
    print("Loading VQ-VAE codebook...")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    vqvae = SDformerVQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=device))
    vqvae.eval()

    # --- Load Real Data ---
    print(f"Loading Real Baseline Data: {REAL_DATA_PATH}")
    X_real, y_real = load_and_embed_data(REAL_DATA_PATH, vqvae, device)
    code_dim = X_real.shape[-1]
    
    print("\n" + "="*60)
    print("PHASE 1: Train on REAL -> Test on SYNTHETIC")
    print("="*60)
    
    # Train ONE model on the entire Real dataset
    print("Training baseline model on 100% of Real Data... (Please wait)")
    model_real = train_model(X_real, y_real, device, code_dim=code_dim, epochs=20)
    print("Baseline model training complete.\n")
    
    # Evaluate on all synthetic datasets
    for name, path in SYNTH_DATASETS.items():
        if not os.path.exists(path):
            print(f"Skipping {name} - File not found.")
            continue
            
        X_synth, y_synth = load_and_embed_data(path, vqvae, device)
        acc, prec, rec, f1 = evaluate_model(model_real, X_synth, y_synth, device)
        print(f"[Test on {name}] -> Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")


    print("\n" + "="*60)
    print("PHASE 2: Train on SYNTHETIC -> Test on REAL")
    print("="*60)
    
    # Train a new model for EACH synthetic dataset and test on Real data
    for name, path in SYNTH_DATASETS.items():
        if not os.path.exists(path):
            continue
            
        print(f"\nTraining model on 100% of {name}...")
        X_synth, y_synth = load_and_embed_data(path, vqvae, device)
        
        model_synth = train_model(X_synth, y_synth, device, code_dim=code_dim, epochs=20)
        
        acc, prec, rec, f1 = evaluate_model(model_synth, X_real, y_real, device)
        print(f"[Train on {name} -> Test on Real Baseline] -> Acc: {acc*100:.2f}% | F1: {f1*100:.2f}%")

if __name__ == "__main__":
    cross_domain_validation()