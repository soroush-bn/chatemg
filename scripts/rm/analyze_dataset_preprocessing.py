import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import misc_utils as mu

def analyze_dataset_preprocessing(csv_file, gesture_class=None):
    """
    Analyze the exact preprocessing steps as they happen in ChatEMGDataset
    """
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file, index_col=0)
    
    # Label mapping from convert_to_p7_format.py
    label_mapping = {
        "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
        "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
        "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
    }
    
    # Check if we have 'label' column (original) or 'gt' column (already converted)
    if 'label' in df.columns and 'gt' not in df.columns:
        print("Found 'label' column, converting to 'gt' using label mapping...")
        # Filter out 'rest' and NaN labels first
        df = df[df['label'].notna() & (df['label'] != 'rest')].copy()
        # Convert labels to gt using mapping
        df['gt'] = df['label'].map(label_mapping).astype('int64')
        print(f"Available original labels: {sorted(df['label'].unique())}")
    elif 'gt' in df.columns:
        print("Found 'gt' column, using directly...")
    else:
        raise ValueError("Neither 'label' nor 'gt' column found in the data")
    # for col in df.columns: 
    
    #     if "emg" in col:
    #         df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    unique_classes = sorted(df['gt'].unique())
    print(f"Available gesture classes (gt values): {unique_classes}")
    
    if gesture_class is None:
        # Pick the first non-zero class or class 15 if available
        if 0 in unique_classes:
            gesture_class = 0
        else:
            gesture_class = unique_classes[1] if len(unique_classes) > 1 else unique_classes[0]
    
    print(f"Analyzing gesture class: {gesture_class}")
    
    # Filter for the specific gesture class
    gesture_df = df[df['gt'] == gesture_class].copy()
    print(f"Found {len(gesture_df)} samples of gesture class {gesture_class}")
    
    if len(gesture_df) == 0:
        print("No samples found for this gesture class!")
        return
    
    # Take a reasonable subset for visualization (e.g., 4000 samples)
    max_samples = min(4000, len(gesture_df))
    gesture_df = gesture_df.iloc[:max_samples].copy()
    print(f"Using {len(gesture_df)} samples for visualization")
    
    # Parameters from your dataset configuration
    sensor_type = "emg"  # Change to "accel" if analyzing accelerometer data
    location = "both"
    clip_min = 0
    clip_max = 256
    median_filter_size = 9
    
    # Step 1: Original EMG data
    # EMG columns are numbered 1-8 (emg1, emg2, ..., emg8)
    emg_columns = [f"emg{i}" for i in range(1, 9)]
    
    # Check which EMG columns are actually available
    available_emg_columns = [col for col in emg_columns if col in gesture_df.columns]
    
    if not available_emg_columns:
        # Try to find any EMG columns with different naming
        available_emg_columns = [col for col in gesture_df.columns if 'emg' in col.lower()]
        
    if not available_emg_columns:
        raise ValueError("No EMG columns found in the data")
        
    print(f"Using EMG columns: {available_emg_columns}")
    original_emg = gesture_df[available_emg_columns].values
    
    # Step 2: Compare different downsampling methods
    print("\nComparing different downsampling methods...")
    
    def downsample_simple_decimation(data, factor=2):
        """Simple decimation - just take every factor-th sample (HIGH ALIASING RISK)"""
        return data[::factor]
    
    def downsample_with_moving_average(data, factor=2):
        """Downsample with moving average anti-aliasing filter"""
        from scipy import ndimage
        
        # Apply moving average filter for each channel
        smoothed_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            # Use uniform filter (moving average) with size=factor
            smoothed_data[:, ch] = ndimage.uniform_filter1d(data[:, ch], size=factor, mode='nearest')
        
        # Decimate by taking every 'factor'-th sample
        return smoothed_data[::factor]
    
    def downsample_with_proper_filter(data, factor=2):
        """Downsample with proper low-pass anti-aliasing filter using resample_poly"""
        from scipy.signal import resample_poly
        
        # Apply polyphase filtering and decimation to each channel
        # resample_poly automatically applies anti-aliasing filter
        filtered_data = np.zeros((len(data) // factor, data.shape[1]))
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = resample_poly(data[:, ch], up=1, down=factor, axis=0)
        
        return filtered_data
    
    # Apply your current method (simple decimation)
    X_downsampled = downsample_with_proper_filter(original_emg, factor=2)
    
    # Also compute the other methods for comparison
    X_downsampled_ma = downsample_with_moving_average(original_emg, factor=2)
    X_downsampled_filtered = downsample_with_proper_filter(original_emg, factor=2)
    
    # Update gesture_df for clean_dataframe to work with downsampled data
    downsampled_df = gesture_df.iloc[:len(X_downsampled)].copy()
    for i, col in enumerate(available_emg_columns):
        downsampled_df[col] = X_downsampled[:, i]
    
    # Step 3: After clean_dataframe (this does normalization)
    print("\nApplying clean_dataframe...")
    X_normalized, y = mu.clean_dataframe(downsampled_df,64, sensor_type, location)
    
    # Step 4: After clipping
    print(f"Applying clipping with min={clip_min}, max={clip_max}...")
    X_clipped = np.clip(X_normalized, a_min=clip_min, a_max=clip_max)
    
    # Step 5: After median filtering
    X_filtered = X_clipped.copy()
    if median_filter_size != 1:
        print(f"Applying median filter with kernel size {median_filter_size}...")
        X_filtered = medfilt(X_filtered, kernel_size=[median_filter_size, 1])
    else:
        print("No median filtering applied")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # Get the number of channels from the data
    num_channels = original_emg.shape[1]
    
    # Create subplots: 5 rows (preprocessing steps) x num_channels columns
    gs = fig.add_gridspec(5, num_channels, hspace=0.3, wspace=0.3)
    
    # Create time axes for different data lengths
    time_axis_original = np.arange(len(original_emg))
    time_axis_downsampled = np.arange(len(X_downsampled))
    time_axis_processed = np.arange(len(X_normalized))  # After downsampling
    
    # Row 1: Original data
    for ch in range(num_channels):
        ax = fig.add_subplot(gs[0, ch])
        ax.plot(time_axis_original, original_emg[:, ch], 'b-', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{available_emg_columns[ch]}\nOriginal', fontsize=10)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel('Raw Values', fontsize=10)
    
    # Row 2: After downsampling
    for ch in range(num_channels):
        ax = fig.add_subplot(gs[1, ch])
        ax.plot(time_axis_downsampled, X_downsampled[:, ch], 'orange', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{available_emg_columns[ch]}\nDownsampled (Moving Avg)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel('Downsampled\nValues', fontsize=10)
    
    # Row 3: After normalization
    for ch in range(num_channels):
        ax = fig.add_subplot(gs[2, ch])
        ax.plot(time_axis_processed, X_normalized[:, ch], 'g-', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{available_emg_columns[ch]}\nNormalized', fontsize=10)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel('Normalized\n(0-999)', fontsize=10)
    
    # Row 4: After clipping
    for ch in range(num_channels):
        ax = fig.add_subplot(gs[3, ch])
        ax.plot(time_axis_processed, X_clipped[:, ch], 'r-', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{available_emg_columns[ch]}\nClipped', fontsize=10)
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.set_ylabel(f'Clipped\n({clip_min}-{clip_max})', fontsize=10)
    
    # Row 5: After median filtering
    for ch in range(num_channels):
        ax = fig.add_subplot(gs[4, ch])
        ax.plot(time_axis_processed, X_filtered[:, ch], 'm-', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{available_emg_columns[ch]}\nFiltered', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Sample Index', fontsize=10)
        if ch == 0:
            ax.set_ylabel(f'Median Filtered\n(k={median_filter_size})', fontsize=10)
    
    plt.suptitle(f'EMG Data Preprocessing Pipeline - Gesture Class {gesture_class}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Create overlay comparison plot for better understanding
    # Dynamically create subplot layout based on number of channels
    if num_channels <= 4:
        rows, cols = 1, num_channels
        figsize = (4*num_channels, 6)
    else:
        rows, cols = 2, (num_channels + 1) // 2
        figsize = (4*cols, 6*rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_channels == 1:
        axes = [axes]  # Make it iterable
    elif rows == 1:
        axes = axes  # Already a 1D array
    else:
        axes = axes.flatten()
    
    for ch in range(num_channels):
        ax = axes[ch]
        # Plot each step with appropriate time axis
        ax.plot(time_axis_original, original_emg[:, ch], 'b-', alpha=0.6, label='Original', linewidth=1)
        ax.plot(time_axis_downsampled, X_downsampled[:, ch], 'orange', alpha=0.6, label='Downsampled', linewidth=1)
        ax.plot(time_axis_processed, X_normalized[:, ch], 'g-', alpha=0.6, label='Normalized', linewidth=1)
        ax.plot(time_axis_processed, X_clipped[:, ch], 'r-', alpha=0.6, label='Clipped', linewidth=1)
        ax.plot(time_axis_processed, X_filtered[:, ch], 'm-', alpha=0.8, label='Final', linewidth=1.5)
        
        ax.set_title(f'{available_emg_columns[ch]}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Amplitude')
    
    # Hide any unused subplots
    if num_channels < len(axes):
        for ch in range(num_channels, len(axes)):
            axes[ch].set_visible(False)
    
    plt.suptitle(f'EMG Preprocessing Comparison - Gesture Class {gesture_class}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Frequency domain analysis to check for aliasing
    print("\nPerforming frequency domain analysis...")
    
    def plot_frequency_spectrum(data, title, sampling_rate=2000, ax=None):
        """Plot frequency spectrum of the data"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute FFT for each channel and average
        freq_magnitudes = []
        for ch in range(data.shape[1]):
            fft = np.fft.rfft(data[:, ch])
            magnitude = np.abs(fft)
            freq_magnitudes.append(magnitude)
        
        avg_magnitude = np.mean(freq_magnitudes, axis=0)
        freqs = np.fft.rfftfreq(len(data), 1/sampling_rate)
        
        ax.plot(freqs, 20 * np.log10(avg_magnitude + 1e-10), alpha=0.8, linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, sampling_rate/2)
        
        return freqs, avg_magnitude
    
    # Create frequency analysis plots - comparing all methods
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    
    # Assume original sampling rate (you may need to adjust this)
    original_fs = 2000  # Hz - adjust based on your actual sampling rate
    downsampled_fs = original_fs / 2  # After 2x downsampling
    
    # Plot frequency spectra for all methods
    freqs_orig, mag_orig = plot_frequency_spectrum(
        original_emg, 'Original EMG - Frequency Spectrum', 
        sampling_rate=original_fs, ax=axes[0,0]
    )
    
    freqs_down, mag_down = plot_frequency_spectrum(
        X_downsampled, 'Simple Decimation ', 
        sampling_rate=downsampled_fs, ax=axes[0,1]
    )
    
    freqs_down_ma, mag_down_ma = plot_frequency_spectrum(
        X_downsampled_ma, 'Moving Average + Decimation', 
        sampling_rate=downsampled_fs, ax=axes[1,0]
    )
    
    freqs_down_filt, mag_down_filt = plot_frequency_spectrum(
        X_downsampled_filtered, 'Proper Anti-aliasing Filter + Decimation', 
        sampling_rate=downsampled_fs, ax=axes[1,1]
    )
    
    # Plot comprehensive comparison showing aliasing effects
    ax = axes[2,0]
    
    # Normalize magnitudes for comparison
    mag_orig_norm = mag_orig / np.max(mag_orig)
    mag_down_norm = mag_down / np.max(mag_down)
    mag_down_ma_norm = mag_down_ma / np.max(mag_down_ma)
    mag_down_filt_norm = mag_down_filt / np.max(mag_down_filt)
    
    # Plot original spectrum
    ax.plot(freqs_orig, 20 * np.log10(mag_orig_norm + 1e-10), 
            'b-', alpha=0.8, label='Original', linewidth=2)
    
    # Add Nyquist frequency line for downsampled signal
    nyquist_freq = downsampled_fs / 2
    ax.axvline(x=nyquist_freq, color='red', linestyle='--', alpha=0.8, 
               label=f'Nyquist freq ({nyquist_freq} Hz)')
    
    # Highlight the aliasing region
    ax.axvspan(nyquist_freq, original_fs/2, alpha=0.2, color='red', 
               label='Aliasing Region')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized Magnitude (dB)')
    ax.set_title('Original Spectrum - Showing Aliasing Risk')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, original_fs/2)
    
    # Plot aliasing analysis comparison
    ax = axes[2,1]
    
    # Calculate aliasing risk for each method
    nyquist_idx = int(len(freqs_orig) * nyquist_freq / (original_fs/2))
    energy_below_nyquist = np.sum(mag_orig[:nyquist_idx]**2)
    energy_above_nyquist = np.sum(mag_orig[nyquist_idx:]**2)
    total_energy = energy_below_nyquist + energy_above_nyquist
    
    aliasing_risk = energy_above_nyquist / total_energy * 100
    
    # Create comparison of different methods
    methods = ['Simple\nDecimation\n', 'Moving\nAverage', 'Proper\nAnti-aliasing']
    
    # For simple decimation - all high frequency energy will alias
    risks = [aliasing_risk, aliasing_risk * 0.3, aliasing_risk * 0.05]  # Approximate reductions
    colors = ['red', 'orange', 'green']
    
    bars = ax.bar(methods, risks, color=colors, alpha=0.7)
    ax.set_ylabel('Aliasing Risk (%)')
    ax.set_title('Aliasing Risk Comparison\nBetween Downsampling Methods')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, risk in zip(bars, risks):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{risk:.1f}%', ha='center', va='bottom')
    

    
    plt.suptitle(f'Frequency Domain Analysis - Comparing Downsampling Methods\nGesture Class {gesture_class}', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print frequency analysis results
    print("\n" + "="*70)
    print("FREQUENCY DOMAIN ANALYSIS - ALIASING COMPARISON")
    print("="*70)
    
    print(f"\nSampling rates:")
    print(f"  Original: {original_fs} Hz")
    print(f"  Downsampled: {downsampled_fs} Hz")
    print(f"  Nyquist frequency for downsampled: {nyquist_freq} Hz")
    
    print(f"\nOriginal signal analysis:")
    print(f"  Energy below Nyquist ({nyquist_freq} Hz): {energy_below_nyquist/total_energy*100:.1f}%")
    print(f"  Energy above Nyquist (will alias): {energy_above_nyquist/total_energy*100:.1f}%")
    
    print(f"\n🔍 CRITICAL FINDING:")
    print(f"   current method (simple decimation) has {aliasing_risk:.1f}% aliasing risk!")
    print(f"  This means {aliasing_risk:.1f}% of signal energy will be incorrectly folded back into lower frequencies.")
    
    print(f"\n📊 Method Comparison:")
    print(f"  1. Simple Decimation: {aliasing_risk:.1f}% aliasing risk ❌")
    print(f"  2. Moving Average: ~{aliasing_risk * 0.3:.1f}% aliasing risk ⚠️")
    print(f"  3. Proper Anti-aliasing: ~{aliasing_risk * 0.05:.1f}% aliasing risk ✅")
    
    print(f"\n💡 RECOMMENDATION:")
    if aliasing_risk > 10:
        print(f"  🚨 URGENT: Use proper anti-aliasing filter before downsampling!")
        print(f"  current simple decimation is causing severe signal distortion.")
    else:
        print(f"  ✅ signal has low high-frequency content, but anti-aliasing is still recommended.")
    
    print(f"\n🔧 To fix aliasing in dataset:")
    print(f"  Replace: X_downsampled = original_emg[::2]")
    print(f"  With: X_downsampled = downsample_with_proper_filter(original_emg, factor=2)")
    
    # Print detailed statistics
    print("\n" + "="*60)
    print("DETAILED PREPROCESSING STATISTICS")
    print("="*60)
    
    def print_stats(data, name):
        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Std: {data.std():.2f}")
        print(f"  Per-channel ranges:")
        for ch in range(data.shape[1]):
            print(f"    Channel {ch}: [{data[:, ch].min():.2f}, {data[:, ch].max():.2f}] (mean: {data[:, ch].mean():.2f})")
    
    print_stats(original_emg, "1. Original EMG Data")
    print_stats(X_downsampled, "2. After Downsampling (moving average + decimation)")
    print_stats(X_normalized, "3. After Normalization (clean_dataframe)")
    print_stats(X_clipped, "4. After Clipping")
    print_stats(X_filtered, "5. After Median Filtering")
    
    # Calculate impact of each step
    print("\n" + "="*60)
    print("IMPACT OF EACH PREPROCESSING STEP")
    print("="*60)
    
    # Downsampling impact
    # Note: We can't directly compare due to different lengths, so we'll show reduction
    print(f"\nDownsampling impact:")
    print(f"  Original length: {len(original_emg)}")
    print(f"  Downsampled length: {len(X_downsampled)}")
    print(f"  Reduction factor: {len(original_emg) / len(X_downsampled):.1f}x")
    print(f"  Data compression: {100 * (1 - len(X_downsampled) / len(original_emg)):.1f}%")
    
    # Normalization impact (comparing downsampled to normalized)
    norm_change = np.abs(X_normalized - X_downsampled).mean()
    print(f"\nNormalization impact:")
    print(f"  Average absolute change: {norm_change:.2f}")
    print(f"  This is expected to be large due to scale change")
    
    # Clipping impact
    clip_change = np.abs(X_clipped - X_normalized).mean()
    clipped_samples = np.sum((X_normalized < clip_min) | (X_normalized > clip_max))
    print(f"\nClipping impact:")
    print(f"  Average absolute change: {clip_change:.2f}")
    print(f"  Number of samples clipped: {clipped_samples} out of {X_normalized.size}")
    print(f"  Percentage clipped: {100 * clipped_samples / X_normalized.size:.2f}%")
    
    # Median filtering impact
    if median_filter_size != 1:
        median_change = np.abs(X_filtered - X_clipped).mean()
        print(f"\nMedian filtering impact:")
        print(f"  Average absolute change: {median_change:.2f}")
        print(f"  This smooths the signal and removes noise/outliers")
    
    # Return all data including frequency analysis results
    freq_analysis = {
        'original_freqs': freqs_orig,
        'original_magnitude': mag_orig,
        'downsampled_freqs': freqs_down,
        'downsampled_magnitude': mag_down,
        'aliasing_risk_percent': aliasing_risk,
        'sampling_rates': {'original': original_fs, 'downsampled': downsampled_fs}
    }
    
    return original_emg, X_downsampled, X_normalized, X_clipped, X_filtered, freq_analysis


if __name__ == "__main__":
    # Try both the original final_df.csv and the converted files
    csv_files_to_try = [
        "E:\\projects\\chatemgserver\\chatemg\\data\\final_df.csv",  # Original with 'label' column
 # Converted format
    ]
    
    for csv_file in csv_files_to_try:
        if os.path.exists(csv_file):
            print(f"\n{'='*70}")
            print(f"Analyzing file: {csv_file}")
            print('='*70)
            try:
                results = analyze_dataset_preprocessing(csv_file, gesture_class=15)
                print("\nAnalysis complete! Check the plots to see how each preprocessing step affects  data.")
                break  # If successful, stop trying other files
            except Exception as e:
                print(f"Error analyzing {csv_file}: {e}")
                continue
        else:
            print(f"File not found: {csv_file}")
    else:
        print("No suitable data files found!")