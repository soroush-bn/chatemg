import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml

def visualize_gesture_reconstruction(original_path, reconstructed_path, save_dir="./gesture_plots"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Loading data from {original_path} and {reconstructed_path}...")
    if not os.path.exists(original_path) or not os.path.exists(reconstructed_path):
        print("Error: One or more data files not found.")
        return

    df_orig = pd.read_csv(original_path)
    df_recon = pd.read_csv(reconstructed_path)

    # Ensure we are only looking at sensor columns (drop gt for plotting)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    # We will plot the first sensor for clarity, but you can change this index
    sensor_to_plot = feature_cols[0] 

    print("Identifying gesture segments...")
    # Find indices where the gesture ID changes
    df_orig['change'] = df_orig['gt'].diff().ne(0).astype(int)
    df_orig['block_id'] = df_orig['change'].cumsum()

    group = df_orig.groupby('block_id')
    block_indices = group.apply(lambda x: (x.index[0], x.index[-1], x['gt'].iloc[0])).values
    
    gesture_map = {}
    for start, end, g_id in block_indices:
        if g_id not in gesture_map:
            gesture_map[g_id] = []
        gesture_map[g_id].append((start, end))

    print(f"Detected {len(gesture_map)} unique gestures.")

    for g_id in sorted(gesture_map.keys()):
        all_reps = gesture_map[g_id]
        print(f"Processing Gesture {g_id}: Found {len(all_reps)} total segments.")

        # Determine how many participants we can plot (assuming segments are grouped by participant)
        # For unseen data, we might have fewer reps per person.
        
        # We'll group reps by participant more safely
        # Assuming each participant has some number of reps
        num_participants = 11
        for p_idx in range(num_participants):
            # In a clean dataset, reps are usually sequential for participants
            # But let's just take a slice and plot what we have
            # If your dataset has a different number of reps, this logic helps visualize whatever is there
            
            # Estimate which reps belong to this participant
            # If we don't know for sure, we at least won't crash
            reps_per_p = max(1, len(all_reps) // num_participants)
            participant_reps = all_reps[p_idx * reps_per_p : (p_idx + 1) * reps_per_p]

            if len(participant_reps) == 0:
                continue

            # Create a dynamic grid based on available reps
            n_rows = len(participant_reps)
            fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3 * n_rows), squeeze=False)
            fig.suptitle(f"Participant {p_idx} | Gesture {g_id}\n(Left: Original, Right: Reconstructed)", fontsize=16)

            for r_idx, (start, end) in enumerate(participant_reps):
                # Ensure we don't go out of bounds of the reconstructed data
                if start >= len(df_recon):
                    print(f"  Warning: Segment {start}:{end} is beyond reconstructed data length ({len(df_recon)}). Skipping.")
                    continue
                
                # Adjust end if it exceeds recon length
                actual_end = min(end, len(df_recon))
                
                orig_segment = df_orig.iloc[start:actual_end][sensor_to_plot].values
                recon_segment = df_recon.iloc[start:actual_end][sensor_to_plot].values

                axes[r_idx, 0].plot(orig_segment, color='steelblue', lw=1.5)
                axes[r_idx, 0].set_ylabel(f"Seg {r_idx + 1}", fontweight='bold')
                if r_idx == 0:
                    axes[r_idx, 0].set_title("Original Data", fontsize=14)
                
                if len(recon_segment) > 0:
                    axes[r_idx, 1].plot(recon_segment, color='indianred', lw=1.5)
                else:
                    axes[r_idx, 1].text(0.5, 0.5, "No Reconstructed Data", ha='center')
                    
                if r_idx == 0:
                    axes[r_idx, 1].set_title("Reconstructed Data", fontsize=14)
                
                axes[r_idx, 0].grid(alpha=0.3)
                axes[r_idx, 1].grid(alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            save_name = f"participant{p_idx}_gt{int(g_id)}.png"
            plt.savefig(os.path.join(save_dir, save_name), dpi=100)
            plt.close()

    print(f"Done! All plots saved in '{save_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to Transformer config (replicate_small.yaml)")
    parser.add_argument('--vqvae_config', type=str, required=True, help="Path to VQ-VAE config (tuned_config2.yaml)")
    parser.add_argument('--save_dir', type=str, default=None, help="Custom save directory for plots")
    args = parser.parse_args()

    # Load configs
    with open(args.config, 'r') as f:
        tr_config = yaml.safe_load(f)
    with open(args.vqvae_config, 'r') as f:
        vq_config = yaml.safe_load(f)

    exp_name = tr_config['exp_name']
    vq_name = vq_config['name']
    model_files_base_directory = os.path.join(pathlib.Path(__file__).resolve().parent.__str__(), "models")
    base_model_dir = os.path.join(model_files_base_directory, exp_name)
    # Use UNSEEN data paths
    ORIG_PATH = f"./VQVAE/models/{vq_name}/unseen_data_preprocessed.csv"
    RECON_PATH = f"{base_model_dir}/unseen_reconstructed_final.csv"
    
    # Priority for SAVE_DIR: Command line arg > Default derived path
    SAVE_DIR = args.save_dir if args.save_dir else f"{base_model_dir}/unseen_gesture_plots"
    
    # Adjust visualization for single (unseen) repetition if needed
    # (The plotting function assumes 4 reps, but for unseen data we might only have 1 per participant per gesture)
    visualize_gesture_reconstruction(ORIG_PATH, RECON_PATH, save_dir=SAVE_DIR)
