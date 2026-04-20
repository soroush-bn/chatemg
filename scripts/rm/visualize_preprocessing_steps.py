import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import misc_utils as mu

def visualize_preprocessing_steps(csv_file, sensor_type="emg", location="both", 
                                clip_min=0, clip_max=999, median_filter_size=9,
                                gesture_class=15, num_samples=2560):
    """
    Visualize how preprocessing steps affect the original EMG/IMU data
    
    Args:
        csv_file: Path to the CSV file
        sensor_type: "emg" or "imu" 
        location: "both", "forearm", or "wrist"
        clip_min, clip_max: Clipping values
        median_filter_size: Size of median filter
        gesture_class: Which gesture class to visualize (default 15)
        num_samples: How many samples to show from the gesture
    """
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)
    
    # Filter for specific gesture class
    gesture_df = df[df['gt'] == gesture_class].copy()
    if len(gesture_df) == 0:
        print(f"No data found for gesture class {gesture_class}")
        available_classes = sorted(df['gt'].unique())
        print(f"Available classes: {available_classes}")
        return
    
    # Take a subset for visualization
    if len(gesture_df) > num_samples:
        gesture_df = gesture_df.iloc[:num_samples].copy()
    
    print(f"Visualizing {len(gesture_df)} samples of gesture class {gesture_class}")
    
    # Step 1: Original data (relevant columns only)
    if location == "both":
        if sensor_type == "emg":
            original_cols = [f"emg{i}" for i in range(8)]
        else:  # imu
            original_cols = [f"{sensor_type}_{i}" for i in range(1, 9)]
    elif location == "forearm":
        if sensor_type == "emg":
            original_cols = [f"emg{i}" for i in range(4)]
        else:
            original_cols = [f"{sensor_type}_{i}" for i in range(1, 5)]
    elif location == "wrist":
        if sensor_type == "emg":
            original_cols = [f"emg{i}" for i in range(4, 8)]
        else:
            original_cols = [f"{sensor_type}_{i}" for i in range(5, 9)]
    
    # Check which columns actually exist
    available_cols = [col for col in original_cols if col in gesture_df.columns]
    if not available_cols:
        print(f"No columns found for sensor_type='{sensor_type}', location='{location}'")
        print(f"Available columns: {list(gesture_df.columns)}")
        return
    
    original_data = gesture_df[available_cols].values
    
    # Step 2: After clean_dataframe (normalization)
    X_after_clean, y = mu.clean_dataframe(gesture_df, sensor_type, location)
    
    # Step 3: After clipping
    X_after_clip = np.clip(X_after_clean, a_min=clip_min, a_max=clip_max)
    
    # Step 4: After median filtering
    X_after_median = X_after_clip.copy()
    if median_filter_size != 1:
        X_after_median = medfilt(X_after_median, kernel_size=[median_filter_size, 1])
    
    # Create visualization
    num_channels = original_data.shape[1]
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    fig.suptitle(f'Preprocessing Steps for Gesture Class {gesture_class} ({sensor_type}, {location})', fontsize=16)
    
    time_axis = np.arange(len(original_data))
    colors = plt.cm.tab10(np.linspace(0, 1, num_channels))
    
    # Plot 1: Original data
    ax1 = axes[0]
    for i in range(num_channels):
        ax1.plot(time_axis, original_data[:, i], color=colors[i], alpha=0.7, 
                label=f'Channel {i}')
    ax1.set_title('1. Original Data')
    ax1.set_ylabel('Raw Values')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: After clean_dataframe (normalized to 0-999)
    ax2 = axes[1]
    for i in range(num_channels):
        ax2.plot(time_axis, X_after_clean[:, i], color=colors[i], alpha=0.7,
                label=f'Channel {i}')
    ax2.set_title('2. After clean_dataframe() - Normalized to 0-999')
    ax2.set_ylabel('Normalized Values')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: After clipping
    ax3 = axes[2]
    for i in range(num_channels):
        ax3.plot(time_axis, X_after_clip[:, i], color=colors[i], alpha=0.7,
                label=f'Channel {i}')
    ax3.set_title(f'3. After np.clip(min={clip_min}, max={clip_max})')
    ax3.set_ylabel('Clipped Values')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: After median filtering
    ax4 = axes[3]
    for i in range(num_channels):
        ax4.plot(time_axis, X_after_median[:, i], color=colors[i], alpha=0.7,
                label=f'Channel {i}')
    if median_filter_size != 1:
        ax4.set_title(f'4. After Median Filter (kernel_size={median_filter_size})')
    else:
        ax4.set_title('4. No Median Filter Applied')
    ax4.set_ylabel('Filtered Values')
    ax4.set_xlabel('Sample Index')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics for each step
    print("\n=== Data Statistics ===")
    print(f"Original data shape: {original_data.shape}")
    print(f"Original data range: [{original_data.min():.2f}, {original_data.max():.2f}]")
    print(f"Original data mean: {original_data.mean():.2f}, std: {original_data.std():.2f}")
    
    print(f"\nAfter clean_dataframe:")
    print(f"Range: [{X_after_clean.min():.2f}, {X_after_clean.max():.2f}]")
    print(f"Mean: {X_after_clean.mean():.2f}, std: {X_after_clean.std():.2f}")
    
    print(f"\nAfter clipping:")
    print(f"Range: [{X_after_clip.min():.2f}, {X_after_clip.max():.2f}]")
    print(f"Mean: {X_after_clip.mean():.2f}, std: {X_after_clip.std():.2f}")
    
    print(f"\nAfter median filtering:")
    print(f"Range: [{X_after_median.min():.2f}, {X_after_median.max():.2f}]")
    print(f"Mean: {X_after_median.mean():.2f}, std: {X_after_median.std():.2f}")
    
    # Show the difference between steps
    print("\n=== Changes Between Steps ===")
    diff_clean = np.abs(X_after_clean - original_data).mean()
    print(f"Average absolute change after normalization: {diff_clean:.2f}")
    
    diff_clip = np.abs(X_after_clip - X_after_clean).mean()
    print(f"Average absolute change after clipping: {diff_clip:.2f}")
    
    if median_filter_size != 1:
        diff_median = np.abs(X_after_median - X_after_clip).mean()
        print(f"Average absolute change after median filtering: {diff_median:.2f}")
    
    return original_data, X_after_clean, X_after_clip, X_after_median


if __name__ == "__main__":
    # Example usage - modify these parameters as needed
    csv_file = "E:\\projects\\chatemgserver\\chatemg\\data\\final_df.csv"
    
    # Try with EMG data first
    print("=== Visualizing EMG Data ===")
    try:
        visualize_preprocessing_steps(
            csv_file=csv_file,
            sensor_type="emg",
            location="both", 
            clip_min=0,
            clip_max=999,
            median_filter_size=9,
            gesture_class=15,  # You can change this to see different gestures
            num_samples=1000   # Show first 1000 samples
        )
    except Exception as e:
        print(f"Error with EMG visualization: {e}")
    
    # If you have IMU data, you can also try:
    # print("\n=== Visualizing IMU Data ===")
    # try:
    #     visualize_preprocessing_steps(
    #         csv_file=csv_file,
    #         sensor_type="imu",
    #         location="both",
    #         clip_min=0,
    #         clip_max=999,
    #         median_filter_size=9,
    #         gesture_class=15,
    #         num_samples=1000
    #     )
    # except Exception as e:
    #     print(f"Error with IMU visualization: {e}")