import pandas as pd
import numpy as np
import torch
import os
import yaml
from VQVAE.model import SDformerVQVAE

def reconstruct_pipeline(original_csv_path, encoded_csv_path, model, device, window_size=300, stride=30, batch_size=512):
    """
    Decodes tokens in batches and stitches them using Overlap-Add.
    Handles memory by offloading to CPU and syncs lengths to avoid shape errors.
    """
    print("Reading CSV files...")
    df_orig = pd.read_csv(original_csv_path)
    df_enc = pd.read_csv(encoded_csv_path)
    
    # Identify feature columns (exclude ground truth)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    num_channels = len(feature_cols)
    
    # Prepare indices from encoded data
    indices_np = df_enc.drop(columns=['gt']).values
    num_windows = indices_np.shape[0]
    
    print(f"Original rows: {len(df_orig)} | Encoded windows: {num_windows}")
    print(f"Decoding in batches of {batch_size}...")

    # Initialize canvas on CPU (RAM) to avoid GPU OOM
    total_recon_len = (num_windows - 1) * stride + window_size
    canvas = np.zeros((total_recon_len, num_channels), dtype=np.float32)
    counts = np.zeros((total_recon_len, num_channels), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            # Batch slicing
            end_idx = min(i + batch_size, num_windows)
            batch_indices = torch.tensor(indices_np[i:end_idx], dtype=torch.long).to(device)
            
            # Map to codebook: [Batch, Time, Dim]
            # Flattening handles non-contiguous memory
            z_q = model.quantizer.embedding[batch_indices.reshape(-1)]
            z_q = z_q.reshape(batch_indices.shape[0], batch_indices.shape[1], -1)
            
            # Prepare for Decoder: [Batch, Channels, Length]
            z_q = z_q.permute(0, 2, 1).contiguous()
            
            # Decode and move to CPU immediately
            batch_recon = model.decoder(z_q).cpu().numpy() 
            
            # Overlap-Add stitching
            for j in range(batch_recon.shape[0]):
                win_idx = i + j
                start = win_idx * stride
                end = start + window_size
                # batch_recon shape is [Batch, Channels, Window_Size]
                # We transpose to [Window_Size, Channels] for the canvas
                canvas[start:end, :] += batch_recon[j].T
                counts[start:end, :] += 1

            if i % (batch_size * 5) == 0:
                print(f"  Processed {i}/{num_windows} windows...")

    print("Finalizing signal averaging...")
    # Avoid division by zero
    final_signal = canvas / np.maximum(counts, 1)
    
    # Create DataFrame from reconstructed signal
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    
    # --- SHAPE FIX: Sync lengths between original and reconstructed ---
    # The stride-based reconstruction often leaves a small tail of original data unaddressed
    common_len = min(len(df_orig), len(df_reconstructed))
    
    df_reconstructed_final = df_reconstructed.iloc[:common_len]
    df_orig_final = df_orig[feature_cols].iloc[:common_len]
    
    print(f"Sync complete. Final Shape: {df_reconstructed_final.shape}")
    return df_reconstructed_final, df_orig_final

def main():
    # --- Path Configuration ---
    CONFIG_PATH = "./VQVAE/tuned_config2.yaml"
    ORIGINAL_DATA_PATH = "./VQVAE/models/tuned2/original_data_after_preprocessing.csv"
    ENCODED_DATA_PATH = "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/replicate_small/synthetic_df_5_70.csv"
    MODEL_WEIGHTS_PATH = "./VQVAE/models/tuned2/final_model.pth" 
    SAVE_OUTPUT_PATH = "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/replicate_small/reconstructed_final.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CONFIG_PATH):
        print(f"Error: Config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize and Load Model
    model = SDformerVQVAE(config).to(device)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"Weights loaded. Using device: {device}")
    else:
        print(f"Error: Weights not found at {MODEL_WEIGHTS_PATH}")
        return

    try:
        recon_df, orig_features = reconstruct_pipeline(
            ORIGINAL_DATA_PATH, 
            ENCODED_DATA_PATH, 
            model, 
            device,
            window_size=300,
            stride=30,
            batch_size=256 # Adjusted for safety; increase if VRAM allows
        )

        # Calculate MSE on synced lengths
        mse = np.mean((recon_df.values - orig_features.values) ** 2)
        print(f"\nReconstruction MSE: {mse:.8f}")

        # Save result
        recon_df.to_csv(SAVE_OUTPUT_PATH, index=False)
        print(f"Reconstructed data saved to: {SAVE_OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()