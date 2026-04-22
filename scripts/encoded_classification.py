import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import os

from VQVAE.model import SDformerVQVAE

class CNNClassifier(nn.Module):
    def __init__(self, code_dim=32, num_classes=17):
        super().__init__()
        # PyTorch Conv1d expects input shape: [Batch, Channels, Time]
        # Our input will be reshaped from [Batch, 75, 32] to [Batch, 32, 75]
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=code_dim, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Halves the time dimension
            
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
        # x comes in as [Batch, 75, 32]
        # Permute to [Batch, Channels, Time] -> [Batch, 32, 75]
        x = x.permute(0, 2, 1)
        
        x = self.features(x) # Output: [Batch, 128, 1]
        x = x.squeeze(-1)    # Output: [Batch, 128]
        
        return self.classifier(x)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=17):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.4), 
            
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_sizes[2], num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(model, train_loader, test_loader, device, epochs=20, lr=0.001, verbose=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return acc, prec, rec, f1
def classification_pipeline(encoded_df_path, config_path, vqvae_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup & Data Loading ---
    print("Loading VQ-VAE to access codebook...")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    vqvae = SDformerVQVAE(config).to(device)
    vqvae.load_state_dict(torch.load(vqvae_weights_path, map_location=device))
    vqvae.eval()

    print(f"Loading synthetic data from {encoded_df_path}...")
    df = pd.read_csv(encoded_df_path)
    
    # Ensure labels are 0-indexed.
    gt_raw = df['gt'].values
    labels = gt_raw - gt_raw.min() 
    y = torch.tensor(labels, dtype=torch.long)
    
    indices = torch.tensor(df.drop(columns=['gt']).values, dtype=torch.long).to(device)
    num_samples, seq_len = indices.shape

    # Map Indexes to Codebook Vectors
    print("Mapping token indices to continuous latent vectors...")
    with torch.no_grad():
        flat_indices = indices.reshape(-1)
        embeddings = vqvae.quantizer.embedding[flat_indices]
        embeddings = embeddings.reshape(num_samples, seq_len, -1)
        X = embeddings.cpu()
    
    code_dim = X.shape[-1]
    print(f"Data mapped successfully. Shape: {X.shape}")

    print("Extracting Participant boundaries based on GT order...")
    participant_ids = np.zeros(num_samples, dtype=int)
    current_p = 0
    for i in range(1, num_samples):
        # A drop in GT value (e.g., 16 -> 0) means the start of a new participant
        if gt_raw[i] < gt_raw[i-1]:
            current_p += 1
        participant_ids[i] = current_p
    
    num_participants = current_p + 1
    print(f"Detected {num_participants} participants in the dataset.")

    # SCENARIO 1: Between-Subjects 
    print(f"\n{'='*50}")
    print("SCENARIO 1: Classification (All Participants)")
    print(f"{'='*50}")
    
    X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    train_loader_s1 = DataLoader(TensorDataset(X_train_s1, y_train_s1), batch_size=128, shuffle=True)
    test_loader_s1 = DataLoader(TensorDataset(X_test_s1, y_test_s1), batch_size=128, shuffle=False)
    
    # Use the new CNN
    model_s1 = CNNClassifier(code_dim=code_dim, num_classes=17).to(device)
    acc_s1, prec_s1, rec_s1, f1_s1 = train_and_evaluate(
        model_s1, train_loader_s1, test_loader_s1, device, epochs=20, verbose=True
    )
    
    print(f"\n[Scenario 1 Results] (20%)")
    print(f"Accuracy:  {acc_s1*100:.2f}% | Precision: {prec_s1*100:.2f}% | Recall: {rec_s1*100:.2f}% | F1: {f1_s1*100:.2f}%")

    # SCENARIO 2: Within-Subject (Per-Person Models)
    print(f"\n{'='*50}")
    print("SCENARIO 2: Within-Subject Classification (Per Participant)")
    print(f"{'='*50}")
    
    scenario2_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}
    
    for p_id in range(num_participants):
        p_mask = (participant_ids == p_id)
        X_p = X[p_mask]
        y_p = y[p_mask]
        
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_p, y_p, test_size=0.20, random_state=42, stratify=y_p
        )
        
        train_loader_p = DataLoader(TensorDataset(X_train_p, y_train_p), batch_size=128, shuffle=True)
        test_loader_p = DataLoader(TensorDataset(X_test_p, y_test_p), batch_size=128, shuffle=False)
        
        # Initialize a fresh CNN model for this participant
        model_p = CNNClassifier(code_dim=code_dim, num_classes=17).to(device)
        
        acc_p, prec_p, rec_p, f1_p = train_and_evaluate(
            model_p, train_loader_p, test_loader_p, device, epochs=20, verbose=False
        )
        
        scenario2_metrics['acc'].append(acc_p)
        scenario2_metrics['prec'].append(prec_p)
        scenario2_metrics['rec'].append(rec_p)
        scenario2_metrics['f1'].append(f1_p)
        
        print(f"Participant {p_id} -> Acc: {acc_p*100:.2f}% | F1: {f1_p*100:.2f}%")

    avg_acc = np.mean(scenario2_metrics['acc']) * 100
    avg_prec = np.mean(scenario2_metrics['prec']) * 100
    avg_rec = np.mean(scenario2_metrics['rec']) * 100
    avg_f1 = np.mean(scenario2_metrics['f1']) * 100

    print(f"\n[Scenario 2 Results] Average Across All {num_participants} Participants")
    print(f"Average Accuracy:  {avg_acc:.2f}%")
    print(f"Average Precision: {avg_prec:.2f}%")
    print(f"Average Recall:    {avg_rec:.2f}%")
    print(f"Average F1-Score:  {avg_f1:.2f}%")
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config (replicate_small.yaml)")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config (tuned_config2.yaml)")
    args = parser.parse_args()

    # Load configs to derive paths
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name'] # e.g., "tuned2"

    # Derive weights path (assuming standard structure)
    VQVAE_WEIGHTS = f"./VQVAE/models/{vq_name}/final_model.pth"
    
    # Base data directory
    base_model_dir = f"/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/{exp_name}"

    # Real data path
    ENCODED_DF_PATH = "/home/sbaghernezha/projects/chatemg/chatemg/data/encoded_df.csv"
    
    print(f"--- Starting Classification Pipeline for Experiment: {exp_name} ---")
    classification_pipeline(ENCODED_DF_PATH, args.vqvae_config, VQVAE_WEIGHTS)

    # List of synthetic datasets to evaluate
    synth_files = [
        "synthetic_df_70_5.csv",
        "synthetic_df_60_15.csv",
        "synthetic_df_50_25.csv",
        "synthetic_df_25_50.csv",
        "synthetic_df_5_70.csv"
    ]

    for s_file in synth_files:
        full_path = os.path.join(base_model_dir, s_file)
        if os.path.exists(full_path):
            print(f"\n\nEvaluating on synthetic dataset: {s_file}")
            classification_pipeline(full_path, args.vqvae_config, VQVAE_WEIGHTS)
        else:
            print(f"Skipping {s_file} - Not found at {full_path}")