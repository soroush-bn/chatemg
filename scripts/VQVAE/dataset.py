import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import yaml 
import os 
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, iirnotch
label_mapping = {
    "Thumb Extension":0,"index Extension":1,"Middle Extension":2,"Ring Extension":3,
    "Pinky Extension":4,"Thumbs Up":5,"Right Angle":6,"Peace":7,"OK":8,"Horn":9,"Hang Loose":10,
    "Power Grip":11,"Hand Open":12,"Wrist Extension":13,"Wrist Flexion":14,"Ulnar deviation":15,"Radial Deviation":16    
}
class EMGDataset(Dataset):
    def __init__(self, config, window_size=300, stride=1, fs=2000, split='all'):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.config = config
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.split = split
        self.df = None 

        print(f"Initializing Dataset (split={self.split})...")
        self.data = self._load_and_normalize_all()
        
        print(f"Dataset Ready. Total Samples: {len(self.data)}")
        if len(self.data) > 0:
            print(f"Tensor Stats -> Mean: {self.data.mean():.4f}, Std: {self.data.std():.4f}")

    def _apply_filters(self, data):
        """
        Apply Bandpass (20-450Hz) and Notch (50Hz) filters.
        """
        nyq = 0.5 * self.fs
        
        # 1. Bandpass Filter (20-450Hz)
        low = 20 / nyq
        high = 450 / nyq
        b, a = butter(4, [low, high], btype='band')
        data = filtfilt(b, a, data, axis=0)
        
        # 2. Notch Filter (50Hz)
        # Quality factor Q = 30 is standard for notch filters
        b_notch, a_notch = iirnotch(50 / nyq, 30)
        data = filtfilt(b_notch, a_notch, data, axis=0)
        
        return data

    def _load_and_normalize_all(self):
        """
        Loads raw data for all subjects, splits it into train/unseen,
        and applies Global Standardization based on the train split.
        """
        all_train_dfs = []
        all_unseen_dfs = []
        
        for subject_id in self.config['participants_list_ids']:
            raw_path = os.path.join(self.config["raw_data_path"], subject_id, self.config["df_raw_name"])
            
            if not os.path.exists(raw_path):
                print(f"WARNING: File not found for {subject_id} at {raw_path}. Skipping.")
                continue
                
            print(f"Loading raw data for: {subject_id}")
            df = pd.read_csv(raw_path)
            df = self._convert_units(df)
            
            sensor_cols = [c for c in df.columns if any(k in c.lower() for k in ['emg', 'accel', 'gyro'])]
            df[sensor_cols] = df[sensor_cols].interpolate(method='linear', limit_direction='both').fillna(0)
            
            if self.config['sensor_type'] == 'emg':
                if 'label' in df.columns:
                    # Filter out rest and NaNs early
                    df = df[df['label'].notna() & (df['label'] != 'rest')].copy()
                    df['gt'] = df['label'].map(label_mapping).astype('int64') 

                emg_cols = [c for c in df.columns if 'emg' in c.lower()]
                
                # Apply Frequency Filtering
                df[emg_cols] = self._apply_filters(df[emg_cols].values)
                
                cols_to_keep = emg_cols + (['gt'] if 'gt' in df.columns else [])
                df_filtered = df[cols_to_keep].copy()
                
                # Split per gesture for this participant
                train_parts = []
                unseen_parts = []
                
                # Identify contiguous segments of the same gesture
                df_filtered['segment'] = (df_filtered['gt'] != df_filtered['gt'].shift()).cumsum()
                
                for _, segment_df in df_filtered.groupby('segment'):
                    if len(segment_df) < 1: continue
                    
                    # Split: last 25% (one repetition out of four)
                    n = len(segment_df)
                    # Use 4000 if it's close to a multiple of 4000, otherwise 75%
                    if n >= 4000:
                        split_idx = n - 4000
                    else:
                        split_idx = int(n * 0.75)
                    
                    train_parts.append(segment_df.iloc[:split_idx])
                    unseen_parts.append(segment_df.iloc[split_idx:])
                
                if train_parts:
                    all_train_dfs.append(pd.concat(train_parts).drop(columns=['segment']))
                if unseen_parts:
                    all_unseen_dfs.append(pd.concat(unseen_parts).drop(columns=['segment']))

        if not all_train_dfs:
            raise RuntimeError("No data loaded! Check your paths and participant IDs.")
            
        train_df = pd.concat(all_train_dfs, axis=0, ignore_index=True)
        unseen_df = pd.concat(all_unseen_dfs, axis=0, ignore_index=True) if all_unseen_dfs else pd.DataFrame()
        
        print(f"Split Summary -> Train Rows: {len(train_df)}, Unseen Rows: {len(unseen_df)}")
        
        # Fit scaler ONLY on train data
        print("Fitting Global StandardScaler on Train Split...")
        scaler = StandardScaler()
        emg_cols = [c for c in train_df.columns if 'emg' in c.lower()]
        scaler.fit(train_df[emg_cols].values)
        
        # Select target df
        if self.split == 'train':
            target_df = train_df
        elif self.split == 'unseen':
            target_df = unseen_df
        else:
            target_df = pd.concat([train_df, unseen_df], axis=0, ignore_index=True)
            
        self.df = target_df
        if len(target_df) == 0:
            return torch.tensor([], dtype=torch.float32)
            
        np_data = scaler.transform(target_df[emg_cols].values)
        return torch.tensor(np_data, dtype=torch.float32)


    def _convert_units(self, df):
        """Simple unit conversion helper"""
        for col in df.columns:
            if "accel" in col: df[col] = df[col] / 2048
            if "gyro" in col:  df[col] = df[col] / 16.4
            if "mag" in col:   df[col] = df[col] * 0.6
        return df

    def save_df(self, save_path):
        if self.df is not None:
            self.df.to_csv(save_path, index=False)
            print(f"Saved merged DataFrame to: {save_path}")
        else:
            print("No DataFrame to save!")
    def __len__(self):
        max_start = len(self.data) - self.window_size
        if max_start < 0: return 0
        return max_start // self.stride + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        window = self.data[start : start + self.window_size]
        return window.transpose(0, 1)