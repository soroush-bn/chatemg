import pandas as pd
import numpy as np
import torch
import os
import yaml
from VQVAEmodel import SDformerVQVAE

def reconstruct_pipeline(original_csv_path, encoded_csv_path, model, device, window_size=300, stride=30):
    """
    Decodes tokens from encoded_df and stitches them back together using Overlap-Add.
    """
    print("Reading CSV files...")
    df_orig = pd.read_csv(original_csv_path)
    df_enc = pd.read_csv(encoded_csv_path)
    
    print(f"Original data shape: {df_orig.shape}")
    print(f"Encoded data shape: {df_enc.shape}")
    
    # Identify feature columns (the 9 sensors)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    
    # --- FIX 1: Use reshape instead of view to avoid stride/contiguous errors ---
    # Shape of indices after dropping 'gt': [108903, 75]
    indices = torch.tensor(df_enc.drop(columns=['gt']).values, dtype=torch.long).to(device)
    
    print(f"Decoding {indices.shape[0]} windows...")
    model.eval()
    with torch.no_grad():
        # Map indices back to the normalized codebook vectors
        # Flattening with reshape handles non-contiguous memory automatically
        flat_indices = indices.reshape(-1) 
        z_q = model.quantizer.embedding[flat_indices]
        
        # Reshape to [Batch, Time, Dim] -> [108903, 75, code_dim]
        z_q = z_q.reshape(indices.shape[0], indices.shape[1], -1)
        
        # Prepare for Decoder: [Batch, Channels, Length]
        z_q = z_q.permute(0, 2, 1).contiguous()
        
        # Run through Decoder -> Output: [108903, 9, 300]
        reconstructed_windows = model.decoder(z_q) 
        reconstructed_np = reconstructed_windows.cpu().numpy()

    print("Stitching signal using Overlap-Add averaging...")
    num_windows, num_channels, win_len = reconstructed_np.shape
    total_recon_len = (num_windows - 1) * stride + win_len
    
    # --- FIX 2: Use float32 for canvas to save memory on 3.2M rows ---
    canvas = np.zeros((total_recon_len, num_channels), dtype=np.float32)
    counts = np.zeros((total_recon_len, num_channels), dtype=np.float32)
    
    for i in range(num_windows):
        start = i * stride
        end = start + win_len
        
        # Transpose window from [9, 300] to [300, 9]
        canvas[start:end, :] += reconstructed_np[i].T
        counts[start:end, :] += 1
        
        if i % 25000 == 0:
            print(f"  Processed {i}/{num_windows} windows...")
    
    # Final averaging
    final_signal = canvas / np.maximum(counts, 1)
    
    # Convert to DataFrame and crop to original length (3267373)
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    df_reconstructed = df_reconstructed.iloc[:len(df_orig)]
    
    print(f"Final reconstructed shape: {df_reconstructed.shape}")
    return df_reconstructed, df_orig[feature_cols]

def main():
    # --- Path Configuration ---
    CONFIG_PATH = "./VQVAE/tuned_config.yaml"
    ORIGINAL_DATA_PATH = "./VQVAE/models/tuned/original_data_after_preprocessing.csv"
    ENCODED_DATA_PATH = "./VQVAE/models/tuned/encoded_df.csv"
    MODEL_WEIGHTS_PATH = "./VQVAE/models/tuned/final_model.pth" 
    SAVE_OUTPUT_PATH = "./VQVAE/models/tuned/reconstructed_final.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Config ---
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Config loaded. Input Dim: {config.get('input_dim')}")

    # --- Initialize and Load Model ---
    model = SDformerVQVAE(config).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        # Load state dict with map_location to handle CPU/GPU transfers
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"Weights loaded from {MODEL_WEIGHTS_PATH}")
    else:
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        return

    # --- Run Reconstruction ---
    try:
        recon_df, orig_features = reconstruct_pipeline(
            ORIGINAL_DATA_PATH, 
            ENCODED_DATA_PATH, 
            model, 
            device,
            window_size=300,
            stride=30
        )

        # Calculate Accuracy
        # Ensure we only compare the overlapping indices
        mse = np.mean((recon_df.values - orig_features.values) ** 2)
        print(f"\nReconstruction MSE: {mse:.8f}")

        # Save result
        recon_df.to_csv(SAVE_OUTPUT_PATH, index=False)
        print(f"Reconstructed data saved to: {SAVE_OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred during reconstruction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()