from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import yaml 
import os 
import torch


with open("vqvae_config.yaml", "r") as file:
    config = yaml.safe_load(file)

class TorchStandardScaler:
    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x):
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True, unbiased=False)
        return self

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class EMGDataset(Dataset):
    def __init__(self, window_size=300, stride=1):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.window_size = window_size
        self.stride = stride
        self.saving_dir = "./"  
        self.label_mapping = {
        "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
        "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
        "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
    }   
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        raw_merged_data = self.read_data()
        assert np.array_equal(np.sort(raw_merged_data['gt'].unique()), np.arange(17, dtype=float)), "Unique values in 'gt' do not match expected range 0-16"
        print(len(raw_merged_data)==3537806, f"Dataframe length {len(raw_merged_data)} does not match expected 3537806")

        emg_cols = [c for c in raw_merged_data.columns if 'emg' in c.lower()]
        data = raw_merged_data[emg_cols].values
        # 2. Handle NaNs (Linear interpolation is usually best for short gaps in time series)
        # If the gap is at the start/end, we fill with 0
        df_temp = pd.DataFrame(data)
        df_temp = df_temp.interpolate(method='linear', limit_direction='both').fillna(0)
        data = df_temp.values

        # 3. Normalize (Zero mean, Unit Variance)
        # self.scaler = StandardScaler()
        # self.data = self.scaler.fit_transform(data)

        # Convert to tensor (float32)
        data = torch.tensor(data, dtype=torch.float32)
        # in nabayad local baseh? per person ya per gesture? 
        self.scaler = TorchStandardScaler()
        self.data = self.scaler.fit_transform(data)

    def __len__(self):
        max_start = len(self.data) - self.window_size
        if max_start < 0:
            return 0
        return max_start // self.stride + 1

    def __getitem__(self, idx):
        # Shape: [Window_Size, Channels] -> Transpose to [Channels, Window_Size] for Conv1D
        start = idx * self.stride
        window = self.data[start : start + self.window_size]
        return window.transpose(0, 1)
    

    def __merge_subjects__(self, type="emg"):
        dfs = []
        saving_csv_name = f"converted_{config['sensor_type']}_{config['axis']}.csv" if config['sensor_type'] != "emg" else "converted_emg.csv"

        for subject in config['participants_list_ids']:
            subject_folder = os.path.join(config["converted_data_path"], subject)
            merged_df = pd.DataFrame()
            printed_subject = False

            if type == "emg":
                csv_path = os.path.join(subject_folder, saving_csv_name)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    self._print_subject_distribution(subject, df)
                    merged_df = pd.concat([merged_df, df], axis=1)
            else:
                for ax in config['axis']:
                    csv_path = os.path.join(subject_folder, f"converted_{config['sensor_type']}_{ax}.csv")
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if not printed_subject:
                            self._print_subject_distribution(subject, df)
                            printed_subject = True
                        merged_df = pd.concat([merged_df, df], axis=1)
            dfs.append(merged_df)
        
        print(len(dfs))

        df= pd.concat(dfs, axis=0)
        name_save = f'VQ_VAE_merged_{type}.csv'
        save_path = os.path.join(config["converted_data_path"], name_save)
        df.to_csv(save_path, index=True)
        print(len(df))
        print(df.head())
        print(df.describe())
        print(f"Merged dataframe saved to {save_path}")
        print("--"*20)
        return df

    def convert_raw_values(self,df, normalize=False):
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



        
    def get_emg_df(self, df, saving_dir=None): #final df
        if saving_dir is None:
            saving_dir = self.saving_dir
        
        # Filter FIRST - remove the rows where label == 'rest' or is NaN
        df = df[df['label'].notna() & (df['label'] != 'rest')].copy()
        
        new_df = pd.DataFrame()
        # Copy EMG data (emg1 to emg8), fill NaN with 0 and convert to int64
        for i in range(1, 9):
            new_df[f'emg{i}'] = df[f'emg{i}'].fillna(0)

        # Set labels as 'gt' with numeric mapping
        new_df['gt'] = df['label'].map(self.label_mapping)

        # Print unique labels to debug
        print("\nUnique labels in the data:")
        print(new_df['gt'].unique())


        # Add timestamps and time_elapsed, convert to appropriate types
        new_df = new_df[['gt',  
                        'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8']]

        print("Data types of columns:")
        print(new_df.dtypes)
        save_name = os.path.join(saving_dir,'converted_emg.csv')
        new_df.to_csv(save_name, index=True)
        return new_df


    def get_IMU_df(self, df, type, axis, saving_dir=None):
        """Create a wide-form dataframe for IMU readings similar to get_emg_df.

        The output has columns: gt, current_time, current_task, sensor1_{type}_{axis}, ..., sensor8_{type}_{axis}.
        """
        if saving_dir is None:
            saving_dir = self.saving_dir
        
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
        new_df['gt'] = df['label'].map(self.label_mapping)

        # Print unique labels to debug
        print("\nUnique labels in the data:")
        print(df['label'].unique())

        # reorder columns to match get_emg_df style
        new_df = new_df[['gt'] + 
                        [f'sensor{i}_{type}_{axis}' for i in range(1, 9)]]

        print("Data types of columns:")
        print(new_df.dtypes)
        save_name = os.path.join(saving_dir,f'converted_{type}_{axis}.csv')
        new_df.to_csv(save_name, index=True)
        return new_df




    def read_data(self):
        
        saving_csv_name = f"converted_{config['sensor_type']}_{config['axis']}.csv" if config['sensor_type'] != "emg" else "converted_emg.csv"

        for participant_id in config['participants_list_ids']:
            participant_folder = os.path.join(config["converted_data_path"], participant_id)
            # check if already converted file exists 
            if os.path.exists(os.path.join(participant_folder, saving_csv_name)):
                print(f"Converted file already exists for participant: {participant_id}. Skipping conversion.")
                continue
            if os.path.exists(participant_folder):
                pass
            else:
                os.makedirs(participant_folder, exist_ok=True)
            print(f"Processing participant: {participant_id}")
            csv_path1 = os.path.join(os.path.join(config["raw_data_path"], participant_id), config["df_raw_name"])
            print(f"Reading data from: {csv_path1}")
            df1 = pd.read_csv(csv_path1)
            df1 = self.convert_raw_values(df1)
            print(f"Converted raw values for participant: {participant_id}.")
            if config['sensor_type'] == "emg":
                emg_df = self.get_emg_df(df1,saving_dir=participant_folder)
            else:
                imu_df = self.get_IMU_df(df1, config['sensor_type'], config['axis'],saving_dir=participant_folder)
                print(f"Converted {config['sensor_type']} data along {config['axis']} for participant: {participant_id}.")
            print(f"Conversion completed for participant: {participant_id}.")

        merged_df= self.__merge_subjects__(config['sensor_type'])
        return merged_df
    

    def _print_subject_distribution(self, subject_id, df):
        if 'gt' not in df.columns:
            print(f"Subject {subject_id}: no 'gt' column found, skipping gesture summary.")
            return

        counts = df['gt'].dropna().astype(int).value_counts().sort_index()
        if counts.empty:
            print(f"Subject {subject_id}: no labeled samples found.")
            return

        print(f"\nSubject {subject_id} gesture breakdown:")
        for gesture_id, sample_count in counts.items():
            label = self.id_to_label.get(gesture_id, f"Unknown({gesture_id})")
            approx_windows = 0
            if sample_count >= self.window_size:
                approx_windows = (sample_count - self.window_size) // self.stride + 1
            print(f"    {sample_count:,} samples (~{approx_windows:,} windows) - {label}")

    def display_data_structure(self):
        pass

#todo add merge function per person, for all participants.
#todo adding df payin functionality