import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
import yaml
import os
import pathlib
import argparse
from tqdm import tqdm

from VQVAE.model import SDformerVQVAE
from classifier_model import LatentMLP

# --- 2. Data Loading & Embedding ---
def load_and_embed_data(csv_path, vqvae, device):
    if not os.path.exists(csv_path):
        return None, None, None, 0
        
    df = pd.read_csv(csv_path)
    gt_raw = df['gt'].values
    labels = gt_raw - gt_raw.min()
    y = torch.tensor(labels, dtype=torch.long)
    
    token_cols = [c for c in df.columns if c != 'gt']
    indices = torch.tensor(df[token_cols].values, dtype=torch.long).to(device)
    
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
def train_model(X, y, device, code_dim, epochs=50, verbose=False):
    model = LatentMLP(input_size=code_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
    
    # Add a scheduler for better convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, 
        steps_per_epoch=len(loader), 
        epochs=epochs
    )
    
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training")

    for _ in iterator:
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
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

def run_classification_workflow(X_train, y_train, p_ids_train, X_test, y_test, p_ids_test, device, code_dim, num_participants):
    assert num_participants == 11, "Number of participants must be equal to 11"
    
    # Between-subjects (Global)
    model_global = train_model(X_train, y_train, device, code_dim=code_dim)
    acc_global, f1_global = evaluate_model(model_global, X_test, y_test, device)
    
    # Within-subjects (Per-participant)
    ws_accs, ws_f1s = [], []
    for p_id in range(num_participants):
        mask_train = (p_ids_train == p_id)
        mask_test = (p_ids_test == p_id)
        
        if not mask_train.any() or not mask_test.any():
            continue
            
        model_p = train_model(X_train[mask_train], y_train[mask_train], device, code_dim=code_dim)
        acc_p, f1_p = evaluate_model(model_p, X_test[mask_test], y_test[mask_test], device)
        ws_accs.append(acc_p)
        ws_f1s.append(f1_p)
        
    return acc_global, f1_global, np.mean(ws_accs) if ws_accs else 0, np.mean(ws_f1s) if ws_f1s else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    save_dir = os.path.join(model_files_base_directory, exp_name)
    
    TRAIN_PATH = tr_config.get('train_data_path', f"./VQVAE/models/{vq_name}/train_encoded_df.csv")
    VAL_PATH = tr_config.get('val_data_path', f"./VQVAE/models/{vq_name}/unseen_encoded_df.csv")
    VQVAE_WEIGHTS = f"./VQVAE/models/{vq_name}/final_model.pth"

    print(f"\n--- Classification Experiments Workflow [{exp_name}] ---")
    vqvae = SDformerVQVAE(vq_config).to(device)
    if os.path.exists(VQVAE_WEIGHTS):
        vqvae.load_state_dict(torch.load(VQVAE_WEIGHTS, map_location=device))
        print(f"VQ-VAE weights loaded.")
    vqvae.eval()

    print("\nLoading Real SEEN and UNSEEN datasets...")
    X_seen, y_seen, p_ids_seen, num_p_seen = load_and_embed_data(TRAIN_PATH, vqvae, device)
    X_unseen, y_unseen, p_ids_unseen, num_p_unseen = load_and_embed_data(VAL_PATH, vqvae, device)
    
    if X_seen is None or X_unseen is None:
        print("Error: Required real datasets (seen/unseen) not found.")
        return

    num_participants = min(num_p_seen, num_p_unseen)
    code_dim = X_seen.shape[-1]
    ratios = ["70_5", "60_15", "50_25", "25_50"]
    
    results = []

    # --- EXPERIMENT 1: Train on Encoded SEEN, Test on UNSEEN ---
    print("\n[EXP 1] Training on SEEN, Testing on UNSEEN...")
    acc_g, f1_g, acc_w, f1_w = run_classification_workflow(
        X_seen, y_seen, p_ids_seen, 
        X_unseen, y_unseen, p_ids_unseen, 
        device, code_dim, num_participants
    )
    results.append({
        "Experiment": "Exp 1: Seen Real",
        "Ratio": "N/A",
        "Between-Subj Acc": acc_g,
        "Between-Subj F1": f1_g,
        "Within-Subj Acc": acc_w,
        "Within-Subj F1": f1_w
    })

    # --- EXPERIMENT 2 & 3 ---
    for r in ratios:
        synth_path = os.path.join(save_dir, f"seen_synthetic_df_{r}.csv")
        if not os.path.exists(synth_path):
            print(f"Skipping ratio {r}: synthetic data not found at {synth_path}")
            continue
            
        print(f"\nProcessing Ratio: {r}")
        X_synth, y_synth, p_ids_synth, _ = load_and_embed_data(synth_path, vqvae, device)
        
        # EXPERIMENT 2: Train on Synthetic, Test on UNSEEN
        print(f" [EXP 2] Training on Synthetic ({r}), Testing on UNSEEN...")
        acc_g, f1_g, acc_w, f1_w = run_classification_workflow(
            X_synth, y_synth, p_ids_synth, 
            X_unseen, y_unseen, p_ids_unseen, 
            device, code_dim, num_participants
        )
        results.append({
            "Experiment": "Exp 2: Synthetic Only",
            "Ratio": r,
            "Between-Subj Acc": acc_g,
            "Between-Subj F1": f1_g,
            "Within-Subj Acc": acc_w,
            "Within-Subj F1": f1_w
        })

        # EXPERIMENT 3: Train on Augmented (Seen + Synthetic), Test on UNSEEN
        print(f" [EXP 3] Training on Augmented ({r}), Testing on UNSEEN...")
        X_aug = torch.cat([X_seen, X_synth], dim=0)
        y_aug = torch.cat([y_seen, y_synth], dim=0)
        p_ids_aug = np.concatenate([p_ids_seen, p_ids_synth], axis=0)
        
        acc_g, f1_g, acc_w, f1_w = run_classification_workflow(
            X_aug, y_aug, p_ids_aug, 
            X_unseen, y_unseen, p_ids_unseen, 
            device, code_dim, num_participants
        )
        results.append({
            "Experiment": "Exp 3: Augmented",
            "Ratio": r,
            "Between-Subj Acc": acc_g,
            "Between-Subj F1": f1_g,
            "Within-Subj Acc": acc_w,
            "Within-Subj F1": f1_w
        })

    # --- Print Summary ---
    print("\n" + "="*100)
    print(f"{'CLASSIFICATION EXPERIMENTS SUMMARY':^100}")
    print("="*100)
    header = f"{'Experiment':<25} | {'Ratio':<10} | {'Between-Subj Acc':<18} | {'Within-Subj Acc':<18}"
    print(header)
    print("-" * 100)
    for res in results:
        line = f"{res['Experiment']:<25} | {res['Ratio']:<10} | {res['Between-Subj Acc']*100:>16.2f}% | {res['Within-Subj Acc']*100:>16.2f}%"
        print(line)
    print("="*100)

    res_df = pd.DataFrame(results)
    save_path = os.path.join(save_dir, "classification_experiments_results.csv")
    res_df.to_csv(save_path, index=False)
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()
