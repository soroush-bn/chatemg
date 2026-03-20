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

# --- 2. Helper: Load, Embed, and Chunk Data ---
def load_and_embed_data(csv_path, vqvae, device):
    """Loads a CSV, maps tokens to embeddings, and detects participant boundaries."""
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
        
    # Detect participant boundaries based on GT drops (e.g., 16 -> 0)
    participant_ids = np.zeros(num_samples, dtype=int)
    current_p = 0
    for i in range(1, num_samples):
        if gt_raw[i] < gt_raw[i-1]:
            current_p += 1
        participant_ids[i] = current_p
        
    return X, y, participant_ids, current_p + 1

# --- 3. Helper: Train Model ---
def train_model(X_train, y_train, device, code_dim=32, epochs=20, lr=0.001):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = CNNClassifier(code_dim=code_dim, num_classes=17).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

# --- 4. Helper: Evaluate Model ---
def evaluate_model(model, X_test, y_test, device):
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
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return acc, f1

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

    # --- Load Real Data Baseline ---
    print(f"Loading Real Baseline Data: {REAL_DATA_PATH}")
    X_real, y_real, p_ids_real, num_participants = load_and_embed_data(REAL_DATA_PATH, vqvae, device)
    code_dim = X_real.shape[-1]
    
    # --- PRE-TRAIN BASELINE REAL MODELS ---
    print("\nPre-training Baseline Real Models (This only happens once)...")
    
    # Global Model (Scenario 1)
    print("  -> Training Global Real Model (All Subjects)...")
    global_real_model = train_model(X_real, y_real, device, code_dim=code_dim, epochs=20)
    
    # Per-Participant Models (Scenario 2)
    per_participant_real_models = []
    print(f"  -> Training {num_participants} Individual Real Models (Within-Subject)...")
    for p_id in range(num_participants):
        p_mask = (p_ids_real == p_id)
        model_p = train_model(X_real[p_mask], y_real[p_mask], device, code_dim=code_dim, epochs=20)
        per_participant_real_models.append(model_p)
        
    print("Baseline training complete.\n")

    # --- LOOP OVER SYNTHETIC DATASETS ---
    for synth_name, synth_path in SYNTH_DATASETS.items():
        if not os.path.exists(synth_path):
            print(f"Skipping {synth_name} - File not found at {synth_path}")
            continue
            
        print("="*70)
        print(f"EVALUATING SYNTHETIC DATASET: {synth_name}")
        print("="*70)
        
        # Load this specific synthetic dataset
        X_synth, y_synth, p_ids_synth, _ = load_and_embed_data(synth_path, vqvae, device)
        
        # -----------------------------------------------------------------
        # SCENARIO 1: BETWEEN-SUBJECTS (Global)
        # -----------------------------------------------------------------
        print("\n--- SCENARIO 1: BETWEEN-SUBJECTS (Global Classification) ---")
        
        # Phase 1A: Train Real -> Test Synth
        acc_1A, f1_1A = evaluate_model(global_real_model, X_synth, y_synth, device)
        print(f"[Train: REAL -> Test: SYNTH] Accuracy: {acc_1A*100:.2f}% | F1: {f1_1A*100:.2f}%")
        
        # Phase 1B: Train Synth -> Test Real
        # We must train a new global model on this synthetic dataset
        global_synth_model = train_model(X_synth, y_synth, device, code_dim=code_dim, epochs=20)
        acc_1B, f1_1B = evaluate_model(global_synth_model, X_real, y_real, device)
        print(f"[Train: SYNTH -> Test: REAL] Accuracy: {acc_1B*100:.2f}% | F1: {f1_1B*100:.2f}%")
        
        # -----------------------------------------------------------------
        # SCENARIO 2: WITHIN-SUBJECT (Per-Person)
        # -----------------------------------------------------------------
        print("\n--- SCENARIO 2: WITHIN-SUBJECT (Per-Participant Averaged) ---")
        
        s2_metrics_real_to_synth = {'acc': [], 'f1': []}
        s2_metrics_synth_to_real = {'acc': [], 'f1': []}
        
        for p_id in range(num_participants):
            # Masks for participant p
            mask_real = (p_ids_real == p_id)
            mask_synth = (p_ids_synth == p_id)
            
            # Phase 2A: Train Real (Pre-trained) -> Test Synth
            acc_2A, f1_2A = evaluate_model(per_participant_real_models[p_id], X_synth[mask_synth], y_synth[mask_synth], device)
            s2_metrics_real_to_synth['acc'].append(acc_2A)
            s2_metrics_real_to_synth['f1'].append(f1_2A)
            
            # Phase 2B: Train Synth -> Test Real
            synth_model_p = train_model(X_synth[mask_synth], y_synth[mask_synth], device, code_dim=code_dim, epochs=20)
            acc_2B, f1_2B = evaluate_model(synth_model_p, X_real[mask_real], y_real[mask_real], device)
            s2_metrics_synth_to_real['acc'].append(acc_2B)
            s2_metrics_synth_to_real['f1'].append(f1_2B)
            
        # Calculate Averages for Scenario 2
        avg_acc_2A = np.mean(s2_metrics_real_to_synth['acc']) * 100
        avg_f1_2A = np.mean(s2_metrics_real_to_synth['f1']) * 100
        
        avg_acc_2B = np.mean(s2_metrics_synth_to_real['acc']) * 100
        avg_f1_2B = np.mean(s2_metrics_synth_to_real['f1']) * 100
        
        print(f"[Train: REAL -> Test: SYNTH] Avg Accuracy: {avg_acc_2A:.2f}% | Avg F1: {avg_f1_2A:.2f}%")
        print(f"[Train: SYNTH -> Test: REAL] Avg Accuracy: {avg_acc_2B:.2f}% | Avg F1: {avg_f1_2B:.2f}%")
        print("\n")

if __name__ == "__main__":
    cross_domain_validation()