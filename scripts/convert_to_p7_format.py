import pandas as pd
import datetime
import numpy as np

        
label_mapping = {
        "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
                "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
                "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
    }


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


def get_emg_df(df): #final df
    
    new_df = pd.DataFrame()

    # Copy EMG data (emg1 to emg8), fill NaN with 0 and convert to int64
    for i in range(1, 9):
        new_df[f'emg{i}'] = df[f'emg{i}'].fillna(0).astype('int64')

    # Print any columns with NaN values to debug
    print("\nColumns with NaN values:")
    print(df.isna().sum())

    # Set labels as 'gt' with numeric mapping
    new_df['gt'] = df['label'].fillna('None').map(label_mapping).fillna(0).astype('int64')

    # Print unique labels to debug
    print("\nUnique labels in the data:")
    print(df['label'].unique())

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

    new_df.to_csv('converted_emg.csv', index=True)
    return new_df


def get_IMU_df(df, type, axis):
    """Create a wide-form dataframe for IMU readings similar to get_emg_df.

    The output has columns: gt, current_time, current_task, sensor1_{type}_{axis}, ..., sensor8_{type}_{axis}.
    """
    new_df = pd.DataFrame()

    # Copy sensor data (sensor1 to sensor8), fill NaN with 0 and convert to float64
    for i in range(1, 9):
        col = f'sensor{i}_{type}_{axis}'
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in dataframe")
        new_df[f'sensor{i}_{type}_{axis}'] = df[col].fillna(0).astype('float64')

    # Print any columns with NaN values to debug
    print("\nColumns with NaN values:")
    print(df.isna().sum())

    # Set labels as 'gt' with numeric mapping
    new_df['gt'] = df['label'].fillna('None').map(label_mapping).fillna(0).astype('int64')

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

    new_df.to_csv(f'converted_{type}_{axis}.csv', index=True)
    return new_df

if __name__ == "__main__":
    
    df = pd.read_csv('final_df.csv')
    df = convert_raw_values(df, normalize=False)
    imu_df = get_IMU_df(df,'accel','x')
    emg_df=  get_emg_df(df)
    print("\nConversion completed. File saved as 'converted_final_df.csv'")
