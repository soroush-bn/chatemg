
import matplotlib.pyplot as plt
import numpy as np


def plot_gesture_emg(num_timesteps, class_label, subject_number, dataframe=df_converted_emg):
    """
    Visualizes EMG data for a specific gesture, subject (identified by index offset), and number of time steps.

    Args:
        num_timesteps (int): The number of time steps to plot.
        class_label (int): The class label (gesture) to filter by.
        subject_number (int): The subject number (1-indexed) to select from the filtered data.
        dataframe (pd.DataFrame): The DataFrame containing the EMG data.
    """
    # Based on the user's dataset description:
    # 2 seconds of gesture data per repetition * 2000 Hz = 4000 samples per repetition
    # 4 repetitions per subject per gesture = 4 * 4000 = 16000 samples per subject per gesture.
    SAMPLES_PER_SUBJECT_PER_GESTURE = 16000

    # Filter by class label first
    filtered_df = dataframe[dataframe['gt'] == float(class_label)]

    if filtered_df.empty:
        print(f"No data found for class label: {class_label}")
        return

    # Calculate the starting and ending index for the specified subject within the filtered data
    # We assume subjects are ordered sequentially within the filtered data for each gesture.
    start_idx_in_filtered = (subject_number - 1) * SAMPLES_PER_SUBJECT_PER_GESTURE
    end_idx_in_filtered = start_idx_in_filtered + num_timesteps

    # Check if the calculated range is valid
    if start_idx_in_filtered >= len(filtered_df):
        print(f"Error: Subject {subject_number} (starting index {start_idx_in_filtered}) is out of bounds for class label {class_label} which has {len(filtered_df)} entries.")
        return

    # Select the specified segment of time steps for the subject
    # Use .iloc to select by integer position
    data_to_plot = filtered_df.iloc[start_idx_in_filtered : end_idx_in_filtered]

    if data_to_plot.empty:
        print(f"No data found for subject {subject_number}, class label {class_label} in the specified range. It might be that the calculated range is invalid or there's not enough data.")
        return

    if len(data_to_plot) < num_timesteps:
        print(f"Warning: Only {len(data_to_plot)} time steps available for gesture {class_label}, subject {subject_number} starting from index {start_idx_in_filtered}. Plotting all available.")

    # Select EMG columns
    emg_cols = [c for c in data_to_plot.columns if 'emg' in c.lower()]
    emg_data = data_to_plot[emg_cols]

    if emg_data.empty:
        print("No EMG data columns found to plot.")
        return

    # Plot the data
    plt.figure(figsize=(15, 8))
    # Plot against a continuous range (0 to len(emg_data)-1) for the x-axis
    # This resolves the 'gaps' caused by plotting against the original, potentially sparse, DataFrame indices.
    x_values = range(len(emg_data))
    for col in emg_data.columns:
        plt.plot(x_values, emg_data[col], label=col)

    plt.title(f'EMG Data for Gesture {class_label}, Subject {subject_number} (First {len(emg_data)} Timesteps)')
    plt.xlabel('Time Step within Segment') # Update x-axis label
    plt.ylabel('EMG Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def plot_processed_emg_window(emg_tensor_window, title="Processed EMG Window from Dataset"):
    """
    Visualizes a single pre-processed EMG window (tensor) as obtained from EMGDataset.

    Args:
        emg_tensor_window (torch.Tensor): A tensor representing a single EMG window,
                                          expected shape [Channels, Window_Size].
        title (str): Title for the plot.
    """
    # Move to CPU and convert to NumPy array
    emg_np = emg_tensor_window.cpu().numpy()

    # Ensure it's [Window_Size, Channels] for easier plotting
    if emg_np.shape[0] < emg_np.shape[1]: # If Channels < Window_Size, it's likely [Channels, Window_Size]
        emg_np = emg_np.T # Transpose to [Window_Size, Channels]

    # Create a DataFrame for plotting, assuming 'emg1' to 'emg8'
    emg_df = pd.DataFrame(emg_np, columns=[f'emg{i+1}' for i in range(emg_np.shape[1])])

    plt.figure(figsize=(15, 8))
    for col in emg_df.columns:
        plt.plot(emg_df.index, emg_df[col], label=col)

    plt.title(title)
    plt.xlabel('Time Step within Window')
    plt.ylabel('Normalized EMG Value')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
