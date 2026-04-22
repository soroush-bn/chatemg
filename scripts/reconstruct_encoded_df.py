import os
import yaml
import torch
import pandas as pd
import numpy as np
import argparse

from VQVAE.model import SDformerVQVAE

def reconstruct_pipeline(original_data_path, encoded_data_path, vqvae, device):
    """
    Reconstructs the original signals from the encoded tokens using the VQ-VAE decoder.
    """
    print(f"Loading original data from {original_data_path} for column info...")
    df_orig = pd.read_csv(original_data_path)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    
    print(f"Loading encoded tokens from {encoded_data_path}...")
    df_encoded = pd.read_csv(encoded_data_path)
    
    # Extract only the token columns
    token_cols = [c for c in df_encoded.columns if c != 'gt']
    indices = torch.tensor(df_encoded[token_cols].values, dtype=torch.long).to(device)
    
    # Get ground truth labels
    gt_labels = df_encoded['gt'].values

    print(f"Indices shape: {indices.shape}")
    
    # --- DECODING PHASE ---
    print("Decoding tokens back to continuous signal...")
    with torch.no_grad():
        # VQ-VAE decode: tokens -> latent -> signal
        # Map indices to embeddings
        z_q = vqvae.quantizer.embedding[indices.reshape(-1)]
        z_q = z_q.reshape(indices.shape[0], indices.shape[1], -1)
        
        # Reshape to [Batch, Channels, SeqLen] if VQ-VAE expects it
        z_q = z_q.permute(0, 2, 1)
        
        # Decode
        reconstructed_signal = vqvae.decoder(z_q)
        
    print(f"Reconstructed signal shape: {reconstructed_signal.shape}")
    
    # Flatten or handle overlapping windows (assuming stride=window_size for simple reconstruction)
    # If using stride < window, this needs more complex overlap-add logic
    # For now, we assume sequences are non-overlapping or treated as independent samples
    
    # Convert to numpy and reshape for CSV
    # Result should be [TotalTime, NumSensors]
    recon_np = reconstructed_signal.permute(0, 2, 1).cpu().numpy()
    final_signal = recon_np.reshape(-1, len(feature_cols))
    
    # Create DataFrame from reconstructed signal
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    
    # Add back the GT labels (repeat GT for each time step in the window)
    # seq_len_per_sample = final_signal.shape[0] // len(gt_labels)
    # df_reconstructed['gt'] = np.repeat(gt_labels, seq_len_per_sample)

    # --- SHAPE FIX: Sync lengths between original and reconstructed ---
    common_len = min(len(df_orig), len(df_reconstructed))

    df_reconstructed_final = df_reconstructed.iloc[:common_len]
    df_orig_final = df_orig[feature_cols].iloc[:common_len]
    
    # Add GT to recon
    df_reconstructed_final = df_reconstructed_final.copy()
    df_reconstructed_final['gt'] = df_orig['gt'].iloc[:common_len].values

    print(f"Sync complete. Final Shape: {df_reconstructed_final.shape}")
    return df_reconstructed_final, df_orig_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config (replicate_small.yaml)")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config (tuned_config2.yaml)")
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']

    # --- Path Configuration ---
    CONFIG_PATH = args.vqvae_config
    ORIGINAL_DATA_PATH = f"./VQVAE/models/{vq_name}/original_data_after_preprocessing.csv"
    
    # Defaulting to 5_70 for reconstruction as per previous script state
    base_model_dir = f"/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/{exp_name}"
    ENCODED_DATA_PATH = f"{base_model_dir}/synthetic_df_5_70.csv"
    
    MODEL_WEIGHTS_PATH = f"./VQVAE/models/{vq_name}/final_model.pth" 
    SAVE_OUTPUT_PATH = f"{base_model_dir}/reconstructed_final.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        return

    # Initialize and Load Model
    model = SDformerVQVAE(vq_config).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"Weights loaded from {MODEL_WEIGHTS_PATH}. Using device: {device}")
    else:
        print(f"Error: Weights not found at {MODEL_WEIGHTS_PATH}")
        return

    try:
        recon_df, _ = reconstruct_pipeline(
            ORIGINAL_DATA_PATH, 
            ENCODED_DATA_PATH, 
            model, 
            device
        )
        
        # Save results
        os.makedirs(os.path.dirname(SAVE_OUTPUT_PATH), exist_ok=True)
        recon_df.to_csv(SAVE_OUTPUT_PATH, index=False)
        print(f"Successfully saved reconstructed data to: {SAVE_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")

if __name__ == "__main__":
    main()
