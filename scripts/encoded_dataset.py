import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EncodedEMGDataset(Dataset):
    def __init__(
        self,
        csv_files,
        filter_class=None
    ):
        """
        Dataset loader for VQ-VAE encoded discrete tokens.
        Loads data from provided CSV files without internal splitting.
        """
        self.csv_files = csv_files if isinstance(csv_files, list) else [csv_files]
        self.filter_class = filter_class

        print(f"[SHAPE TRACK] ===== INITIALIZING ENCODED DATASET =====")
        df_list = []
        for f in self.csv_files:
            if not os.path.exists(f):
                print(f"WARNING: File {f} not found!")
                continue
            df = pd.read_csv(f)
            print(f"[SHAPE TRACK] Loaded CSV from {os.path.basename(f)}: {df.shape}")
            df_list.append(df)
            
        if not df_list:
            raise FileNotFoundError("No valid CSV files found for dataset initialization.")

        df_all = pd.concat(df_list, ignore_index=True)

        if self.filter_class is not None:
            print(f"[SHAPE TRACK] Filtering for class {self.filter_class}...")
            df_all = df_all[df_all['gt'] == self.filter_class]

        # Store labels
        self.labels = df_all['gt'].values
        
        # Extract only the token columns (col_0 to col_74)
        token_cols = [c for c in df_all.columns if c != 'gt']
        self.tokens = df_all[token_cols].values

        print(f"[SHAPE TRACK] Total sequences available: {len(self.tokens)}")
        print(f"[SHAPE TRACK] Token sequence length: {self.tokens.shape[1]}")

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        seq = self.tokens[idx]
        label = self.labels[idx]
        
        # For autoregressive training (predicting the next token):
        x = seq[:-1]
        y = seq[1:]
        
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return x, y, label

    def sample(self, num):
        idx = np.random.choice(range(self.__len__()), num, replace=False)
        X, Y, L = [], [], []
        for i in idx:
            x, y, l = self.__getitem__(i)
            X.append(x)
            Y.append(y)
            L.append(l)
        return torch.stack(X), torch.stack(Y), torch.stack(L)

if __name__ == "__main__":
    # Test the dataset class
    # Path relative to scripts/
    path = "../VQVAE/models/test_run8_1epoch/train_encoded_df.csv"
    if os.path.exists(path):
        dataset = EncodedEMGDataset(csv_files=[path])
        x, y, label = dataset[0]
        print(f"\nx shape: {x.shape}, y shape: {y.shape}")
        print(f"Gesture Label: {label}")
    else:
        print(f"Test file not found at {path}")
