import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import os
import pathlib
import argparse

from VQVAE.model import SDformerVQVAE

class CNNClassifier(nn.Module):
    def __init__(self, code_dim=32, num_classes=17):
        super().__init__()
        # Input shape: [Batch, Time, Channels] -> [Batch, 75, code_dim]
        # PyTorch Conv1d expects: [Batch, Channels, Time]
        
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
        # x: [Batch, Time, Channels] -> [Batch, Channels, Time]
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.squeeze(-1)
        return self.classifier(x)

def load_and_map_data(csv_path, vqvae, device):
    """
    Loads CSV tokens and maps them to their VQ-VAE embedding vectors.
    Also identifies participant IDs based on gesture resets.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: File not found {csv_path}")
        return None, None, None, 0
        
    df = pd.read_csv(csv_path)
    gt_raw = df['gt'].values
    y = torch.tensor(gt_raw, dtype=torch.long)
    
    token_cols = [c for c in df.columns if c != 'gt']
    indices = torch.tensor(df[token_cols].values, dtype=torch.long).to(device)
    num_samples, seq_len = indices.shape

    # Map token indices to VQ-VAE codebook vectors
    with torch.no_grad():
        flat_indices = indices.reshape(-1)
        embeddings = vqvae.quantizer.embedding[flat_indices]
        X = embeddings.reshape(num_samples, seq_len, -1).cpu()
    
    # Identify participant IDs (current_p increments when gesture ID resets)
    p_ids = np.zeros(num_samples, dtype=int)
    current_p = 0
    for i in range(1, num_samples):
        if gt_raw[i] < gt_raw[i-1]:
            current_p += 1
        p_ids[i] = current_p
        
    return X, y, p_ids, current_p + 1

def run_experiment(X_train, y_train, X_test, y_test, device, code_dim, num_classes, name, epochs=30):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=128, shuffle=False)
    
    model = CNNClassifier(code_dim=code_dim, num_classes=num_classes).to(device)
    model = train_model(model, train_loader, device, epochs=epochs)
    acc, f1 = evaluate_model(model, test_loader, device, name=name)
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config")
    args = parser.parse_args()

    # 1. Load Configs
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    
    # 2. Paths
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, exp_name)
    
    TRAIN_PATH = tr_config.get('train_data_path', f"./VQVAE/models/{vq_name}/train_encoded_df.csv")
    VAL_PATH = tr_config.get('val_data_path', f"./VQVAE/models/{vq_name}/unseen_encoded_df.csv")
    VQVAE_WEIGHTS = f"./VQVAE/models/{vq_name}/final_model.pth"

    # 3. Load VQ-VAE
    print(f"\n--- Initializing Classification Comparison Pipeline [{exp_name}] ---")
    vqvae = SDformerVQVAE(vq_config).to(device)
    if os.path.exists(VQVAE_WEIGHTS):
        vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=device))
        print(f"VQ-VAE weights loaded from {VQVAE_WEIGHTS}")
    vqvae.eval()

    # 4. Load & Map Original Datasets
    print("\n[1/3] Loading Real SEEN and UNSEEN datasets...")
    X_seen, y_seen, p_ids_seen, num_p_seen = load_and_map_data(TRAIN_PATH, vqvae, device)
    X_unseen, y_unseen, p_ids_unseen, num_p_unseen = load_and_map_data(VAL_PATH, vqvae, device)
    
    if X_seen is None or X_unseen is None:
        print("Error: Required real datasets (seen/unseen) not found. Exiting.")
        return

    num_participants = min(num_p_seen, num_p_unseen)
    num_classes = tr_config.get('num_classes', 17)
    code_dim = X_seen.shape[-1]
    ratios = ["70_5", "60_15", "50_25", "25_50", "5_70"]

    results = []

    # -------------------------------------------------------------------------
    # SCENARIO 1: GLOBAL (BETWEEN-SUBJECTS)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'SCENARIO 1: GLOBAL (BETWEEN-SUBJECTS)':^60}")
    print("="*60)

    # Global Baseline
    print("\nRunning Global Baseline (Seen reps 1-3 -> Unseen rep 4)...")
    acc, f1 = run_experiment(X_seen, y_seen, X_unseen, y_unseen, device, code_dim, num_classes, "Global Baseline")
    results.append({"Scenario": "Global", "Setup": "Baseline", "Accuracy": acc, "F1": f1})

    # Global Augmented
    for r in ratios:
        synth_path = os.path.join(save_dir, f"seen_synthetic_df_{r}.csv")
        if not os.path.exists(synth_path): continue
        
        print(f"\nRunning Global Augmented ({r})...")
        X_synth, y_synth, _, _ = load_and_map_data(synth_path, vqvae, device)
        if X_synth is not None:
            X_comb = torch.cat([X_seen, X_synth], dim=0)
            y_comb = torch.cat([y_seen, y_synth], dim=0)
            acc, f1 = run_experiment(X_comb, y_comb, X_unseen, y_unseen, device, code_dim, num_classes, f"Global Aug ({r})")
            results.append({"Scenario": "Global", "Setup": f"Augmented ({r})", "Accuracy": acc, "F1": f1})

    # -------------------------------------------------------------------------
    # SCENARIO 2: WITHIN-SUBJECT (PER-PARTICIPANT)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"{'SCENARIO 2: WITHIN-SUBJECT (PER-PARTICIPANT)':^60}")
    print("="*60)

    # Within-Subject Baseline
    print("\nRunning Within-Subject Baseline (Averaged)...")
    ws_b_accs, ws_b_f1s = [], []
    for p_id in range(num_participants):
        mask_s = (p_ids_seen == p_id)
        mask_u = (p_ids_unseen == p_id)
        if not mask_s.any() or not mask_u.any(): continue
        
        acc, f1 = run_experiment(X_seen[mask_s], y_seen[mask_s], X_unseen[mask_u], y_unseen[mask_u], 
                                 device, code_dim, num_classes, f"WS Baseline P{p_id}", epochs=20)
        ws_b_accs.append(acc); ws_b_f1s.append(f1)
    
    results.append({"Scenario": "Within-Subject", "Setup": "Baseline", "Accuracy": np.mean(ws_b_accs), "F1": np.mean(ws_b_f1s)})

    # Within-Subject Augmented
    for r in ratios:
        synth_path = os.path.join(save_dir, f"seen_synthetic_df_{r}.csv")
        if not os.path.exists(synth_path): continue
        
        X_synth, y_synth, p_ids_synth, _ = load_and_map_data(synth_path, vqvae, device)
        if X_synth is None: continue
        
        print(f"\nRunning Within-Subject Augmented ({r})...")
        ws_a_accs, ws_a_f1s = [], []
        for p_id in range(num_participants):
            mask_s = (p_ids_seen == p_id)
            mask_u = (p_ids_unseen == p_id)
            mask_syn = (p_ids_synth == p_id)
            if not mask_s.any() or not mask_u.any() or not mask_syn.any(): continue
            
            X_comb = torch.cat([X_seen[mask_s], X_synth[mask_syn]], dim=0)
            y_comb = torch.cat([y_seen[mask_s], y_synth[mask_syn]], dim=0)
            
            acc, f1 = run_experiment(X_comb, y_comb, X_unseen[mask_u], y_unseen[mask_u], 
                                     device, code_dim, num_classes, f"WS Aug ({r}) P{p_id}", epochs=20)
            ws_a_accs.append(acc); ws_a_f1s.append(f1)
        
        results.append({"Scenario": "Within-Subject", "Setup": f"Augmented ({r})", "Accuracy": np.mean(ws_a_accs), "F1": np.mean(ws_a_f1s)})

    # 7. Print Final Comparison Table
    print("\n" + "="*75)
    print(f"{'FINAL CLASSIFICATION COMPARISON SUMMARY':^75}")
    print("="*75)
    print(f"{'Scenario':<18} | {'Setup':<22} | {'Accuracy':<12} | {'F1-Score':<12}")
    print("-" * 75)
    for res in results:
        print(f"{res['Scenario']:<18} | {res['Setup']:<22} | {res['Accuracy']*100:>10.2f}% | {res['F1']*100:>10.2f}%")
    print("="*75)

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(save_dir, "augmentation_comparison_results.csv"), index=False)
    print(f"Full results saved to: {os.path.join(save_dir, 'augmentation_comparison_results.csv')}")

if __name__ == "__main__":
    main()
