import torch
import torch.nn.functional as F
import pandas as pd
import os
import yaml
import numpy as np
from torch.utils.data import DataLoader, random_split

from model import SDformerVQVAE
from dataset import EMGDataset

def load_model_and_config(model_name, base_dir="./models/", device="cpu"):
    """
    Loads a specific model and its corresponding config file.
    """
    model_folder = os.path.join(base_dir, model_name)
    config_path = os.path.join(model_folder, "config.yaml")
    weights_path = os.path.join(model_folder, "final_model.pth")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"Skipping {model_name}: Missing config or model file.")
        return None, None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = SDformerVQVAE(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    return model, config

def compare_models(model_names, device):
    """
    Compares multiple models based on MSE (Accuracy) and Codebook Usage (Health).
    Returns a DataFrame with the scorecard.
    """
    print(f"--- Comparing Models: {model_names} ---")
    
    # (Assuming all models used the same data preprocessing parameters)
    first_model_dir = os.path.join("./models/", model_names[0])
    with open(os.path.join(first_model_dir, "config.yaml"), "r") as f:
        temp_config = yaml.safe_load(f)
        
    full_dataset = EMGDataset(window_size=temp_config['window_size'], stride=temp_config['stride'])
    
    # Use the same validation split as training (Last 20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    print(f"Validation Set: {len(val_dataset)} samples")

    results = []

    # 2. Evaluate Each Model
    for name in model_names:
        print(f"Evaluating {name}...", end=" ")
        model, config = load_model_and_config(name, device=device)
        
        if model is None:
            continue

        total_mse = 0
        all_indices = []
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                
                # Forward Pass
                x_recon, _, indices = model(x)
                
                # 1. Metric: Reconstruction Error (MSE)
                mse = F.mse_loss(x_recon, x)
                total_mse += mse.item()
                
                # Collect indices for usage calculation
                if indices.dim() > 1:
                    indices = indices.flatten()
                all_indices.append(indices.cpu())

        # Aggregate Metrics
        avg_mse = total_mse / len(val_loader)
        
        # 2. Metric: Codebook Usage
        # Check how many unique codes were actually used in the validation set
        flat_indices = torch.cat(all_indices)
        unique_tokens = len(torch.unique(flat_indices))
        total_possible = config['codebook_size']
        usage_pct = (unique_tokens / total_possible) * 100
        
        results.append({
            "Model": name,
            "MSE": avg_mse,
            "Usage %": usage_pct,
            "Unique Codes": f"{unique_tokens}/{total_possible}",
            "Latent Dim": config['hidden_dim']
        })
        print(f"Done. (MSE: {avg_mse:.4f}, Usage: {usage_pct:.1f}%)")

    # 3. Create Scorecard
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No models evaluated.")
        return


    mse_score = (df['MSE'].max() - df['MSE']) / (df['MSE'].max() - df['MSE'].min() + 1e-6)
    
    usage_score = df['Usage %'] / 100.0
    
    df['Score'] = (0.6 * mse_score) + (0.4 * usage_score)
    
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print("FINAL COMPARISON SCORECARD")
    print("="*50)
    print(df[['Model', 'MSE', 'Usage %', 'Unique Codes', 'Score']].to_markdown())
    
    winner = df.iloc[0]
    print(f"\n🏆 The Best Model is: {winner['Model']}")
    print(f"   Reason: Best balance of low error ({winner['MSE']:.4f}) and high codebook usage ({winner['Usage %']:.1f}%).")
    
    df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved full results to 'model_comparison_results.csv'")

if __name__ == "__main__":
    models_to_compare = [
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run1_512_512_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run2_1024_512_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run3_1024_1024_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run4_2048_512_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run5_2048_2048_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run6_512_512_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run7_512_512_100epoch",
        "/home/sbaghernezha/projects/chatemg/chatemg/scripts/VQVAE/models/run8_512_512_100epoch"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter only existing directories
    existing_models = [m for m in models_to_compare if os.path.exists(os.path.join("./models/", m))]
    
    if len(existing_models) > 0:
        compare_models(existing_models, device)
    else:
        print("No valid model directories found. Check 'models_to_compare' list.")