import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import yaml
import os
import argparse

from VQVAE.model import SDformerVQVAE

# --- 1. MLP Classifier for Latent Space ---
class LatentMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=17):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes)
        )

    def forward(self, x):
        # x is [Batch, seq_len, code_dim] -> we pool over time
        x = x.mean(dim=1)
        return self.network(x)

# --- 2. Data Loading & Embedding ---
def load_and_embed_data(csv_path, vqvae, device):
    df = pd.read_csv(csv_path)
    gt_raw = df['gt'].values
    labels = gt_raw - gt_raw.min()
    y = torch.tensor(labels, dtype=torch.long)
    
    indices = torch.tensor(df.drop(columns=['gt']).values, dtype=torch.long).to(device)
    
    with torch.no_grad():
        flat_indices = indices.reshape(-1)
        embeddings = vqvae.quantizer.embedding[flat_indices]
        X = embeddings.reshape(indices.shape[0], indices.shape[1], -1).cpu()

    # Participant boundaries based on GT order
    p_ids = np.zeros(len(X), dtype=int)
    current_p = 0
    for i in range(1, len(X)):
        if gt_raw[i] < gt_raw[i-1]: current_p += 1
        p_ids[i] = current_p
        
    return X, y, p_ids, current_p + 1

# --- 3. Training Helper ---
def train_model(X, y, device, code_dim, epochs=20):
    model = LatentMLP(input_size=code_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    
    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    return model

# --- 4. Evaluation Helper ---
def evaluate_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds = preds.cpu().numpy()
        all_targets = y.numpy()
        
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return acc, f1

# --- 5. Main Cross-Validation Pipeline ---
def cross_domain_validation(config_path, vqvae_weights, real_data_path, synth_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # --- Initialization ---
    print("Loading VQ-VAE codebook...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    vqvae = SDformerVQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_weights, map_location=device))
    vqvae.eval()

    # --- Load Real Data Baseline ---
    print(f"Loading Real Baseline Data: {real_data_path}")
    X_real, y_real, p_ids_real, num_participants = load_and_embed_data(real_data_path, vqvae, device)
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
    for synth_name, synth_path in synth_datasets.items():
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config (replicate_small.yaml)")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config (tuned_config2.yaml)")
    args = parser.parse_args()

    # --- Configs & Paths ---
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    
    # Derive paths
    VQVAE_WEIGHTS = f"./VQVAE/models/{vq_name}/final_model.pth"
    REAL_DATA_PATH = "/home/sbaghernezha/projects/chatemg/chatemg/data/encoded_df.csv"
    
    base_model_dir = f"/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/{exp_name}"
    
    SYNTH_DATASETS = {
        "70% Real / 5% Synth": f"{base_model_dir}/synthetic_df_70_5.csv",
        "60% Real / 15% Synth": f"{base_model_dir}/synthetic_df_60_15.csv",
        "50% Real / 25% Synth": f"{base_model_dir}/synthetic_df_50_25.csv",
        "25% Real / 50% Synth": f"{base_model_dir}/synthetic_df_25_50.csv",
        "5% Real / 70% Synth": f"{base_model_dir}/synthetic_df_5_70.csv"
    }

    print(f"--- Starting Cross-Domain Validation for Experiment: {exp_name} ---")
    cross_domain_validation(args.vqvae_config, VQVAE_WEIGHTS, REAL_DATA_PATH, SYNTH_DATASETS)
