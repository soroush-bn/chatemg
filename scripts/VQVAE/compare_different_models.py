import torch
import torch.nn.functional as F
import pandas as pd
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# Import your custom modules
from model import SDformerVQVAE
from dataset import EMGDataset
from visualizer import Visualizer  # <--- NEW IMPORT

# --- CONFIGURATION ---
# Base directory for all comparison outputs
COMPARISON_BASE_DIR = "./comparisons_all/"
os.makedirs(COMPARISON_BASE_DIR, exist_ok=True)

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

    try:
        model = SDformerVQVAE(config).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model, config
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def compare_models(model_names, device):
    print(f"--- Comparing {len(model_names)} Models ---")
    print(f"--- Saving Results to: {COMPARISON_BASE_DIR} ---")
    
    # 1. Setup Data (Use first model's config for windowing parameters)
    first_model_path = os.path.join("./models/", model_names[0])
    if not os.path.exists(os.path.join(first_model_path, "config.yaml")):
        print(f"Error: Config not found for {model_names[0]}")
        return

    with open(os.path.join(first_model_path, "config.yaml"), "r") as f:
        temp_config = yaml.safe_load(f)
        
    # Initialize Dataset (Load RAM once)
    print("Loading Dataset (this may take a moment)...")
    full_dataset = EMGDataset(window_size=temp_config['window_size'], stride=temp_config['stride'])
    
    # Create Validation Loader for Metrics
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    
    results = []

    # 2. Loop Through All Models
    for name in model_names:
        print(f"\nProcessing Model: {name}")
        
        # Load
        model, config = load_model_and_config(name, base_dir="./models/", device=device)
        if model is None: continue

        # --- A. Calculate Metrics (MSE & Usage) ---
        total_mse = 0
        all_indices = []
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_recon, _,_, indices = model(x)
                
                mse = F.mse_loss(x_recon, x)
                total_mse += mse.item()
                
                if indices.dim() > 1: indices = indices.flatten()
                all_indices.append(indices.cpu())

        avg_mse = total_mse / len(val_loader)
        flat_indices = torch.cat(all_indices)
        unique_tokens = len(torch.unique(flat_indices))
        usage_pct = (unique_tokens / config['codebook_size']) * 100
        
        print(f"  > MSE: {avg_mse:.5f} | Usage: {usage_pct:.1f}%")
        
        results.append({
            "Model": name,
            "MSE": avg_mse,
            "Usage %": usage_pct,
            "Unique Codes": f"{unique_tokens}/{config['codebook_size']}",
            "Latent Dim": config['hidden_dim']
        })

        # --- B. Visualize Gesture Pipeline ---
        # We override the visualizer's save directory to put this model's plots 
        # into its own folder inside ./comparisons/
        
        model_save_dir = os.path.join(COMPARISON_BASE_DIR, name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        # Init Visualizer
        viz = Visualizer(model, device, config)
        viz.save_dir = model_save_dir # <--- OVERRIDE PATH
        
        # Plot "Power Grip" (Label 11), Repetition 0
        try:
            print(f"  > Plotting Power Grip (Rep 0)...")
            viz.plot_gesture_pipeline(
                full_dataset.df, 
                label_id=11,     # Power Grip
                repetition_index=0
            )
        except Exception as e:
            print(f"    ! Could not plot pipeline: {e}")

    # 3. Create & Save Scorecard
    df = pd.DataFrame(results)
    if df.empty: return

    # Calculate Composite Score
    mse_score = (df['MSE'].max() - df['MSE']) / (df['MSE'].max() - df['MSE'].min() + 1e-6)
    usage_score = df['Usage %'] / 100.0
    df['Score'] = (0.6 * mse_score) + (0.4 * usage_score)
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON SCORECARD")
    print("="*60)
    print(df[['Model', 'MSE', 'Usage %', 'Unique Codes', 'Score']].to_markdown())
    
    winner = df.iloc[0]
    print(f"\n🏆 The Best Model is: {winner['Model']}")
    
    # Save CSV
    csv_path = os.path.join(COMPARISON_BASE_DIR, "model_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")
    print(f"Gesture plots saved in subfolders at: {COMPARISON_BASE_DIR}")

if __name__ == "__main__":
    # LIST RELATIVE FOLDER NAMES (must be inside ./models/)

    """
            "run1_512_512_100epoch",
        "run2_1024_512_100epoch",
        "run3_1024_1024_100epoch",
        "run4_2048_512_100epoch",
        "run5_2048_2048_100epoch",
        "run6_512_512_100epoch",
        "run7_512_512_100epoch",
        "run8_512_512_100epoch"
    """
    models_to_compare = [
        "lambda_0.01",
        "lambda_0.1",
        "lambda_0.25",
        "lambda_1",
        "window_300",
        "window_600",
        "stride_10",
        "stride_25",
        "stride_50",
        "codebook_256",
        "codebook_512",
        "codebook_1024",
        "codebook_2048",
        "code_dim_64",
        "code_dim_128",
        "code_dim_512",
        "code_dim_1024",


    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter existing
    existing_models = [m for m in models_to_compare if os.path.exists(os.path.join("./models/", m))]
    
    if len(existing_models) > 0:
        compare_models(existing_models, device)
    else:
        print("No valid model directories found in ./models/. Check your list.")