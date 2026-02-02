import torch
import torch.nn.functional as F
import pandas as pd
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from model import SDformerVQVAE
from dataset import EMGDataset

# --- CONFIGURATION (Fixed: Relative Path) ---
# This will create a folder named 'comparisons' right next to this script
COMPARISON_SAVE_DIR = "./comparisons/"
os.makedirs(COMPARISON_SAVE_DIR, exist_ok=True)

def load_model_and_config(model_name, base_dir="./models/", device="cpu"):
    """
    Loads a specific model and its corresponding config file.
    """
    # model_name can be a simple folder name if base_dir is set, 
    # or a full relative path if base_dir is empty.
    model_folder = os.path.join(base_dir, model_name)
    config_path = os.path.join(model_folder, "config.yaml")
    weights_path = os.path.join(model_folder, "final_model.pth")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"Skipping {model_name}: Missing config or model file.")
        return None, None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize Model
    try:
        model = SDformerVQVAE(config).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        return model, config
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def plot_and_save_reconstruction(model, x_input, model_name, sample_id, device):
    """
    Plots a single sample reconstruction and saves it to the comparison dir.
    """
    x_batch = x_input.to(device).unsqueeze(0) # [1, 8, Window]
    
    with torch.no_grad():
        x_recon, _, _ = model(x_batch)
    
    orig = x_batch[0].cpu().numpy()
    recon = x_recon[0].cpu().numpy()
    mse = np.mean((orig - recon)**2)
    
    fig, axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
    time_steps = range(orig.shape[1])
    
    for ch in range(8):
        axes[ch].plot(time_steps, orig[ch], 'k', alpha=0.6, label='Original')
        axes[ch].plot(time_steps, recon[ch], 'r--', label='Recon')
        axes[ch].set_ylabel(f'Ch {ch+1}')
        axes[ch].grid(True, alpha=0.2)
        axes[ch].spines['top'].set_visible(False)
        axes[ch].spines['right'].set_visible(False)
        if ch == 0: axes[ch].legend(loc='upper right')

    clean_name = os.path.basename(model_name)
    plt.suptitle(f"Model: {clean_name}\nSample: {sample_id} | MSE: {mse:.5f}", y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(COMPARISON_SAVE_DIR, f"sample_{sample_id}_{clean_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def compare_models(model_names, device):
    print(f"--- Comparing Models: {len(model_names)} models found ---")
    print(f"--- Saving Results to: {COMPARISON_SAVE_DIR} ---")
    
    # 1. Setup Data (Use first model's config to define the input window)
    # We look for the first model inside ./models/ to get the window size
    first_model_path = os.path.join("./models/", model_names[0])

    if not os.path.exists(os.path.join(first_model_path, "config.yaml")):
        print(f"Error: Could not find config for {model_names[0]}")
        return

    with open(os.path.join(first_model_path, "config.yaml"), "r") as f:
        temp_config = yaml.safe_load(f)
        
    # Initialize dataset ONCE -> All models see the EXACT same input windows
    full_dataset = EMGDataset(window_size=temp_config['window_size'], stride=temp_config['stride'])
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    print(f"Validation Set: {len(val_dataset)} samples")

    # Pick fixed samples for visual comparison
    vis_indices = [0, 20, 40]
    vis_indices = [i for i in vis_indices if i < len(val_dataset)]
    fixed_samples = {i: val_dataset[i] for i in vis_indices}

    results = []

    # 2. Evaluate Each Model
    for name in model_names:
        print(f"Evaluating {name}...", end=" ")
        
        # We explicitly assume models are in ./models/
        model, config = load_model_and_config(name, base_dir="./models/", device=device)
        
        if model is None: continue

        total_mse = 0
        all_indices = []
        
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_recon, _, indices = model(x)
                
                mse = F.mse_loss(x_recon, x)
                total_mse += mse.item()
                
                if indices.dim() > 1: indices = indices.flatten()
                all_indices.append(indices.cpu())

        # Visualize Fixed Samples
        for idx, sample in fixed_samples.items():
            plot_and_save_reconstruction(model, sample, name, idx, device)

        # Aggregate Metrics
        avg_mse = total_mse / len(val_loader)
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
        print(f"Done. (MSE: {avg_mse:.4f})")

    # 3. Create Scorecard
    df = pd.DataFrame(results)
    if df.empty: return

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
    
    save_path = os.path.join(COMPARISON_SAVE_DIR, "model_comparison_results.csv")
    df.to_csv(save_path, index=False)
    print(f"\nSaved full results to {save_path}")

if __name__ == "__main__":
    # LIST RELATIVE FOLDER NAMES (Assuming they are inside ./models/)
    models_to_compare = [
        "run1_512_512_100epoch",
        "run2_1024_512_100epoch",
        "run3_1024_1024_100epoch",
        "run4_2048_512_100epoch",
        "run5_2048_2048_100epoch",
        "run6_512_512_100epoch",
        "run7_512_512_100epoch",
        "run8_512_512_100epoch"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter only existing directories inside ./models/
    existing_models = [m for m in models_to_compare if os.path.exists(os.path.join("./models/", m))]
    
    if len(existing_models) > 0:
        compare_models(existing_models, device)
    else:
        print("No valid model directories found in ./models/. Check your list.")