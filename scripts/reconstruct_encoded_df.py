import pandas as pd
import numpy as np
import torch
import os
import yaml
from VQVAEmodel import SDformerVQVAE
def reconstruct_pipeline(original_csv_path, encoded_csv_path, model, device, window_size=300, stride=30, batch_size=256):
    print("Reading CSV files...")
    df_orig = pd.read_csv(original_csv_path)
    df_enc = pd.read_csv(encoded_csv_path)
    
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    indices_np = df_enc.drop(columns=['gt']).values
    num_windows = indices_np.shape[0]
    
    print(f"Decoding {num_windows} windows in batches of {batch_size}...")
    
    # Initialize the "canvas" on CPU to save GPU memory
    total_recon_len = (num_windows - 1) * stride + window_size
    canvas = np.zeros((total_recon_len, len(feature_cols)), dtype=np.float32)
    counts = np.zeros((total_recon_len, len(feature_cols)), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            # 1. Slice the current batch
            batch_indices = torch.tensor(indices_np[i : i + batch_size], dtype=torch.long).to(device)
            
            # 2. Map to codebook and reshape
            # [Batch, Time, Dim]
            z_q = model.quantizer.embedding[batch_indices.reshape(-1)]
            z_q = z_q.reshape(batch_indices.shape[0], batch_indices.shape[1], -1)
            z_q = z_q.permute(0, 2, 1).contiguous() 
            
            # 3. Decode batch
            batch_recon = model.decoder(z_q) # Result: [Batch, 9, 300]
            
            # 4. Move to CPU and add to canvas (Overlap-Add)
            batch_recon_np = batch_recon.cpu().numpy()
            
            for j in range(batch_recon_np.shape[0]):
                global_idx = i + j
                start = global_idx * stride
                end = start + window_size
                
                # Transpose [9, 300] to [300, 9] for the canvas
                canvas[start:end, :] += batch_recon_np[j].T
                counts[start:end, :] += 1

            if i % (batch_size * 10) == 0:
                print(f"  Processed {i}/{num_windows} windows...")

    print("Finalizing signal averaging...")
    final_signal = canvas / np.maximum(counts, 1)
    
    df_reconstructed = pd.DataFrame(final_signal, columns=feature_cols)
    df_reconstructed = df_reconstructed.iloc[:len(df_orig)]
    
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