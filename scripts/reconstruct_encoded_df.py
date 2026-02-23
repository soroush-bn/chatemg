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
    print(f"Original data shape: {df_orig.shape}")
    print(f"Original columns: {df_orig.columns.tolist()}")
    print(f"Original data sample:\n{df_orig.head()}\n")
    print(f"Encoded data shape: {pd.read_csv(encoded_csv_path).shape}")
    print(f"Encoded data columns: {pd.read_csv(encoded_csv_path).columns.tolist()}")
    print(f"Encoded data sample:\n{pd.read_csv(encoded_csv_path).head()}\n")
    df_enc = pd.read_csv(encoded_csv_path)
    
    # Identify feature columns (everything except the label 'gt')
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    
    # Extract indices: Shape [Total_Windows, 75]
    # Dropping 'gt' column which is at index 0
    indices = torch.tensor(df_enc.drop(columns=['gt']).values, dtype=torch.long).to(device)
    
    print(f"Decoding {indices.shape[0]} windows...")
    model.eval()
    with torch.no_grad():
        # 1. Map indices back to the normalized codebook vectors
        flat_indices = indices.view(-1)
        z_q = model.quantizer.embedding[flat_indices]
        
        # 2. Reshape to [Batch, Time, Dim] -> [Batch, 75, code_dim]
        z_q = z_q.view(indices.shape[0], indices.shape[1], -1)
        
        # 3. Prepare for Decoder: [Batch, Channels, Length]
        z_q = z_q.permute(0, 2, 1).contiguous()
        
        # 4. Run through Decoder
        reconstructed_windows = model.decoder(z_q) 
        reconstructed_np = reconstructed_windows.cpu().numpy()

    print("Stitching signal using Overlap-Add averaging...")
    num_windows, num_channels, _ = reconstructed_np.shape
    total_recon_len = (num_windows - 1) * stride + window_size
    
    canvas = np.zeros((total_recon_len, num_channels))
    counts = np.zeros((total_recon_len, num_channels))
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        # window_data is [Channels, 300] -> Transpose to [300, Channels]
        canvas[start:end, :] += reconstructed_np[i].T
        counts[start:end, :] += 1
    
    final_signal = canvas / np.maximum(counts, 1)
    
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    df_reconstructed = df_reconstructed.iloc[:len(df_orig)]
    print(f"reconstructed data shape: {df_reconstructed.shape}")
    print(f"reconstructed data columns: {df_reconstructed.columns.tolist()}")
    print(f"reconstructed data sample:\n{df_reconstructed.head()}\n")
    return df_reconstructed, df_orig[feature_cols]

def main():
    # --- 1. Path Configuration ---
    CONFIG_PATH = "./VQVAE/tuned_config.yaml"
    ORIGINAL_DATA_PATH = "./VQVAE/models/tuned/original_data_after_preprocessing.csv"
    ENCODED_DATA_PATH = "./VQVAE/models/tuned/encoded_df.csv"
    MODEL_WEIGHTS_PATH = "./VQVAE/models/final_model.pth" 
    SAVE_OUTPUT_PATH = "./VQVAE/models/reconstructed_final.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Load Config from YAML ---
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config file not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Config loaded. Input Dim: {config.get('input_dim')}")

    # --- 3. Initialize and Load Model ---
    model = SDformerVQVAE(config).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"Weights loaded from {MODEL_WEIGHTS_PATH}")
    else:
        print("Error: Model weights not found.")
        return

    # --- 4. Run Reconstruction ---
    try:
        recon_df, orig_features = reconstruct_pipeline(
            ORIGINAL_DATA_PATH, 
            ENCODED_DATA_PATH, 
            model, 
            device,
            window_size=300,
            stride=30
        )

        mse = np.mean((recon_df.values - orig_features.values) ** 2)
        print(f"\nReconstruction MSE: {mse:.8f}")

        # Save result
        recon_df.to_csv(SAVE_OUTPUT_PATH, index=False)
        print(f"Reconstructed data saved to: {SAVE_OUTPUT_PATH}")



    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()