import os
import pathlib
import yaml
import torch
import pandas as pd
import numpy as np
import argparse

from VQVAE.model import SDformerVQVAE

def reconstruct_pipeline(original_data_path, encoded_data_path, vqvae, device, window_size=300, stride=150):
    """
    Reconstructs the original signals from the encoded tokens using the VQ-VAE decoder.
    Handles overlapping windows by taking the center part of each decoded window.
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
    all_reconstructed_chunks = []
    
    with torch.no_grad():
        # Process in batches to avoid OOM
        batch_size = 128
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            
            # Map indices to embeddings
            z_q = vqvae.quantizer.embedding[batch_indices.reshape(-1)]
            z_q = z_q.reshape(batch_indices.shape[0], batch_indices.shape[1], -1)
            
            # Reshape to [Batch, Channels, SeqLen]
            z_q = z_q.permute(0, 2, 1)
            
            # Decode
            reconstructed_batch = vqvae.decoder(z_q) # [Batch, Channels, WindowSize]
            
            # --- OVERLAP-ADD / STRIDE LOGIC ---
            # If stride < window_size, we need to pick which part of the window to keep.
            # A common approach is to take the middle 'stride' samples.
            
            margin = (window_size - stride) // 2
            # recon_batch: [B, C, W] -> take [B, C, margin : margin + stride]
            chunk = reconstructed_batch[:, :, margin : margin + stride]
            all_reconstructed_chunks.append(chunk.permute(0, 2, 1).cpu().numpy())
        
    # Concatenate all chunks: [TotalChunks, Stride, Channels]
    recon_np = np.concatenate(all_reconstructed_chunks, axis=0)
    # Reshape to continuous: [TotalTime, Channels]
    final_signal = recon_np.reshape(-1, len(feature_cols))
    
    # Create DataFrame from reconstructed signal
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    
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
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    vqvae_models_dir = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "VQVAE", "models")
    CONFIG_PATH = args.vqvae_config
    # Point to the UNSEEN preprocessed data from the VQ-VAE phase
    ORIGINAL_DATA_PATH = os.path.join(vqvae_models_dir, vq_name, "unseen_data_preprocessed.csv")
    
    # Priority for encoded data:
    # 1. New 'unseen_synthetic' files
    # 2. Generic 'unseen_synthetic_encoded_samples.csv'
    # 3. Fallback to old naming 'synthetic_df_...'
    
    base_model_dir = os.path.join(model_files_base_directory, exp_name)
    ENCODED_DATA_PATH = os.path.join(base_model_dir, "unseen_synthetic_df_5_70.csv")
    
    if not os.path.exists(ENCODED_DATA_PATH):
        ENCODED_DATA_PATH = os.path.join(base_model_dir, "unseen_synthetic_encoded_samples.csv")
        
    if not os.path.exists(ENCODED_DATA_PATH):
        synth_files = sorted([f for f in os.listdir(base_model_dir) if f.startswith("unseen_synthetic_")], reverse=True)
        if synth_files:
            ENCODED_DATA_PATH = os.path.join(base_model_dir, synth_files[0])
            
    if not os.path.exists(ENCODED_DATA_PATH):
        synth_files = sorted([f for f in os.listdir(base_model_dir) if f.startswith("synthetic_df_")], reverse=True)
        if synth_files:
            ENCODED_DATA_PATH = os.path.join(base_model_dir, synth_files[0])
    
    MODEL_WEIGHTS_PATH = os.path.join(vqvae_models_dir, vq_name, "final_model.pth") 
    SAVE_OUTPUT_PATH = os.path.join(base_model_dir, "unseen_reconstructed_final.csv")

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
            device,
            window_size=vq_config.get('window_size', 300),
            stride=vq_config.get('stride', 150)
        )
        
        # Save results
        os.makedirs(os.path.dirname(SAVE_OUTPUT_PATH), exist_ok=True)
        recon_df.to_csv(SAVE_OUTPUT_PATH, index=False)
        print(f"Successfully saved reconstructed data to: {SAVE_OUTPUT_PATH}")
        
    except Exception as e:
        print(f"Reconstruction failed: {e}")

if __name__ == "__main__":
    main()
