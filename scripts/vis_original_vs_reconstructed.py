import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_gesture_reconstruction(original_path, reconstructed_path, save_dir="./gesture_plots"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading data...")
    df_orig = pd.read_csv(original_path)
    df_recon = pd.read_csv(reconstructed_path)

    # Ensure we are only looking at sensor columns (drop gt for plotting)
    feature_cols = [c for c in df_orig.columns if c != 'gt']
    # We will plot the first sensor for clarity, but you can change this index
    sensor_to_plot = feature_cols[0] 

    print("Identifying gesture segments...")
    # Find indices where the gesture ID changes
    # This creates a 'block' for every repetition
    df_orig['change'] = df_orig['gt'].diff().ne(0).astype(int)
    df_orig['block_id'] = df_orig['change'].cumsum()

    # Get the start and end indices for every block
    # A block here represents one 2-second repetition
    group = df_orig.groupby('block_id')
    block_indices = group.apply(lambda x: (x.index[0], x.index[-1], x['gt'].iloc[0])).values
    
    # Organize data: dict[gesture_id] = [list of (start, end) tuples]
    gesture_map = {}
    for start, end, g_id in block_indices:
        if g_id not in gesture_map:
            gesture_map[g_id] = []
        gesture_map[g_id].append((start, end))

    print(f"Detected {len(gesture_map)} unique gestures.")

    # Iterate through each unique gesture (1 to 17)
    for g_id in sorted(gesture_map.keys()):
        all_reps = gesture_map[g_id]
        
        # total_reps should be 44 (11 participants * 4 reps)
        print(f"Processing Gesture {g_id}: Found {len(all_reps)} repetitions.")

        # Iterate through each participant (0 to 10)
        for p_idx in range(11):
            # Each participant has a chunk of 4 repetitions
            start_idx_in_map = p_idx * 4
            participant_reps = all_reps[start_idx_in_map : start_idx_in_map + 4]

            if len(participant_reps) < 4:
                continue

            # Create the 4x2 subplot grid
            fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=False)
            fig.suptitle(f"Participant {p_idx} | Gesture {g_id}\n(Left: Original, Right: Reconstructed)", fontsize=16)

            for r_idx, (start, end) in enumerate(participant_reps):
                # Extract the segments
                orig_segment = df_orig.iloc[start:end][sensor_to_plot].values
                recon_segment = df_recon.iloc[start:end][sensor_to_plot].values

                # Column 0: Original
                axes[r_idx, 0].plot(orig_segment, color='steelblue', lw=1.5)
                axes[r_idx, 0].set_ylabel(f"Rep {r_idx + 1}", fontweight='bold')
                if r_idx == 0:
                    axes[r_idx, 0].set_title("Original Data", fontsize=14)
                
                # Column 1: Reconstructed
                axes[r_idx, 1].plot(recon_segment, color='indianred', lw=1.5)
                if r_idx == 0:
                    axes[r_idx, 1].set_title("Reconstructed Data", fontsize=14)
                
                # Optional: Add grid for better comparison
                axes[r_idx, 0].grid(alpha=0.3)
                axes[r_idx, 1].grid(alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the figure
            save_name = f"participant{p_idx}_gt{int(g_id)}.png"
            plt.savefig(os.path.join(save_dir, save_name), dpi=100)
            plt.close()

    print(f"Done! All plots saved in '{save_dir}'")

if __name__ == "__main__":
    # Update these paths to your actual file locations
    ORIG_PATH = "./VQVAE/models/tuned2/original_data_after_preprocessing.csv"
    RECON_PATH = "/home/sbaghernezha/projects/chatemg/chatemg/scripts/models/replicate_small/reconstructed_final.csv"
    
    visualize_gesture_reconstruction(ORIG_PATH, RECON_PATH)