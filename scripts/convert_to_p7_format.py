import pandas as pd
import datetime
import numpy as np
import os 
saving_dir = "./"   
label_mapping = {
        "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
                "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
                "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
    }


def stratified_split(df, train_ratio=0.9, random_state=42):
    """Split dataframe into train and test sets with stratification by 'gt' label.
    
    Args:
        df: DataFrame with 'gt' column containing labels
        train_ratio: Ratio of data to use for training (default 0.9 = 90%)
        random_state: Random seed for reproducibility
    
    Returns:
        train_df, test_df: Two dataframes with stratified split
    """
    train_list = []
    test_list = []
    
    # Group by label and split each group
    for label in df['gt'].unique():
        label_df = df[df['gt'] == label].copy()
        # Shuffle within each label group
        label_df = label_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        # Split into train and test
        split_idx = int(len(label_df) * train_ratio)
        train_list.append(label_df.iloc[:split_idx])
        test_list.append(label_df.iloc[split_idx:])
    
    # Concatenate all splits
    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    # Shuffle the final dataframes
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return train_df, test_df


def convert_raw_values(df, normalize=True):
    for col in df.columns: 
        if "accel" in col : 
            df[col] = df[col] /2048 # converting to g
            if normalize:
                df[col] = (df[col] / 2048.0 + 16) / 32.0
        if "gyro" in col :
            df[col] = df[col] /16.4 # degree per second
            if normalize: 
                # df[col] = (df[col] / 16.4 + 2000) / 4000.0
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
        if "mag" in col: 
            df[col] = df[col] *0.6 # micro tesla
            if normalize:
                df[col] = (df[col] * 0.6 + 4800) / 9600.0
        if "emg" in col:
            # assume values are in µV
            if normalize:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df 


def get_emg_df(df,saving_dir=saving_dir): #final df
    
    # Filter FIRST - remove the rows where label == 'rest' or is NaN
    df = df[df['label'].notna() & (df['label'] != 'rest')].copy()
    
    new_df = pd.DataFrame()
    # Copy EMG data (emg1 to emg8), fill NaN with 0 and convert to int64
    for i in range(1, 9):
        new_df[f'emg{i}'] = df[f'emg{i}'].fillna(0).astype('int64')

    # Set labels as 'gt' with numeric mapping
    new_df['gt'] = df['label'].map(label_mapping).astype('int64')

    # Print unique labels to debug
    print("\nUnique labels in the data:")
    print(new_df['gt'].unique())

    # Generate timestamps
    current_time = datetime.datetime.now()
    base_timestamp = current_time.strftime('%Y%m%d%H%M%S%f')[:17]  # Format: YYYYMMDDHHMMSSmmm

    # Generate time_elapsed (2 seconds - relative time)
    num_rows = len(df)
    total_time = 2.0  # 2 seconds
    # Add timestamps and time_elapsed, convert to appropriate types
    new_df['current_time'] = df["time"]
    new_df['current_task'] = 'None'
    new_df = new_df[['gt', 'current_time', 'current_task', 
                    'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']]

    print("Data types of columns:")
    print(new_df.dtypes)
    
    # Create stratified train/test splits
    train_df, test_df = stratified_split(new_df, train_ratio=0.9, random_state=42)
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Train labels: {sorted(train_df['gt'].unique())}")
    print(f"Test set: {len(test_df)} samples")
    print(f"Test labels: {sorted(test_df['gt'].unique())}")
    
    # Save both splits
    train_name = os.path.join(saving_dir,'converted_emg_train.csv')
    test_name = os.path.join(saving_dir,'converted_emg_test.csv')
    train_df.to_csv(train_name, index=True)
    test_df.to_csv(test_name, index=True)
    
    return train_df, test_df


def get_IMU_df(df, type, axis,saving_dir=saving_dir):
    """Create a wide-form dataframe for IMU readings similar to get_emg_df.

    The output has columns: gt, current_time, current_task, sensor1_{type}_{axis}, ..., sensor8_{type}_{axis}.
    """
    # Downsample by taking every 10th row
    df = df.iloc[::10].copy()
    
    # Filter FIRST - remove the rows where label == 'rest' or is NaN
    df = df[df['label'].notna() & (df['label'] != 'rest')].copy()
    
    new_df = pd.DataFrame()

    # Copy sensor data (sensor1 to sensor8), fill NaN with 0 and convert to float64
    for i in range(1, 9):
        col = f'sensor{i}_{type}_{axis}'
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")
        new_df[f'sensor{i}_{type}_{axis}'] = df[col].fillna(0).astype('float64')

    # Set labels as 'gt' with numeric mapping
    new_df['gt'] = df['label'].map(label_mapping).astype('int64')

    # Print unique labels to debug
    print("\nUnique labels in the data:")
    print(df['label'].unique())

    # Add timestamps and task
    new_df['current_time'] = df['time']
    new_df['current_task'] = 'None'

    # reorder columns to match get_emg_df style
    new_df = new_df[['gt', 'current_time', 'current_task'] + 
                    [f'sensor{i}_{type}_{axis}' for i in range(1, 9)]]

    print("Data types of columns:")
    print(new_df.dtypes)
    
    # Create stratified train/test splits
    train_df, test_df = stratified_split(new_df, train_ratio=0.9, random_state=42)
    
    print(f"\nTrain set: {len(train_df)} samples")
    print(f"Train labels: {sorted(train_df['gt'].unique())}")
    print(f"Test set: {len(test_df)} samples")
    print(f"Test labels: {sorted(test_df['gt'].unique())}")
    
    # Save both splits
    train_name = os.path.join(saving_dir,f'converted_{type}_{axis}_train.csv')
    test_name = os.path.join(saving_dir,f'converted_{type}_{axis}_test.csv')
    train_df.to_csv(train_name, index=True)
    test_df.to_csv(test_name, index=True)
    
    return train_df, test_df

if __name__ == "__main__":
    
    df = pd.read_csv('final_df.csv')
    df = convert_raw_values(df, normalize=False)
    imu_train, imu_test = get_IMU_df(df,'accel','x')
    emg_train, emg_test = get_emg_df(df)
    print("\nConversion completed. Train and test files saved with stratified splits.")
