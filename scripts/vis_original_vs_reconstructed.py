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
        print(f"Processing Gesture {g_id}: Found {len(all_reps)} repetitions.")

        for p_idx in range(11):
            start_idx_in_map = p_idx * 4
            participant_reps = all_reps[start_idx_in_map : start_idx_in_map + 4]

            if len(participant_reps) < 4:
                continue

            fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=False)
            fig.suptitle(f"Participant {p_idx} | Gesture {g_id}\n(Left: Original, Right: Reconstructed)", fontsize=16)

            for r_idx, (start, end) in enumerate(participant_reps):
                orig_segment = df_orig.iloc[start:end][sensor_to_plot].values
                recon_segment = df_recon.iloc[start:end][sensor_to_plot].values

                axes[r_idx, 0].plot(orig_segment, color='steelblue', lw=1.5)
                axes[r_idx, 0].set_ylabel(f"Rep {r_idx + 1}", fontweight='bold')
                if r_idx == 0:
                    axes[r_idx, 0].set_title("Original Data", fontsize=14)
                
                axes[r_idx, 1].plot(recon_segment, color='indianred', lw=1.5)
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
    SAVE_DIR = f"{base_model_dir}/unseen_gesture_plots"
    
    # Adjust visualization for single (unseen) repetition if needed
    # (The plotting function assumes 4 reps, but for unseen data we might only have 1 per participant per gesture)
    visualize_gesture_reconstruction(ORIG_PATH, RECON_PATH, save_dir=SAVE_DIR)
