import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class EncodedEMGDataset(Dataset):
    def __init__(
        self,
        csv_files,
        filter_class=None,
        which_file="train",
        split_ratio=0.7
    ):
        """
        Dataset loader specifically for VQ-VAE encoded discrete tokens.
        """
        self.csv_files = csv_files if isinstance(csv_files, list) else [csv_files]
        self.filter_class = filter_class
        self.which_file = which_file

        print(f"[SHAPE TRACK] ===== INITIALIZING ENCODED DATASET =====")
        df_list = []
        for f in self.csv_files:
            df = pd.read_csv(f)
            print(f"[SHAPE TRACK] Loaded CSV from {os.path.basename(f)}: {df.shape}")
            df_list.append(df)
            
        df_all = pd.concat(df_list, ignore_index=True)

        if self.filter_class is not None:
            print(f"[SHAPE TRACK] Filtering for class {self.filter_class}...")
            df_all = df_all[df_all['gt'] == self.filter_class]

        # Stratified split for training vs sampling
        if which_file in ["train", "sample"]:
            train_list = []
            test_list = []
            for label in df_all['gt'].unique():
                label_df = df_all[df_all['gt'] == label]
                split_index = int(split_ratio * len(label_df))
                train_list.append(label_df.iloc[:split_index])
                test_list.append(label_df.iloc[split_index:])
                
            if which_file == "train":
                df_all = pd.concat(train_list, ignore_index=True)
                print(f"[SHAPE TRACK] After train split ({split_ratio*100}%): {df_all.shape}")
            elif which_file == "sample":
                df_all = pd.concat(test_list, ignore_index=True)
                print(f"[SHAPE TRACK] After sample split ({(1-split_ratio)*100}%): {df_all.shape}")

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
        # x is the sequence from the start to the second-to-last token
        # y is the sequence from the second token to the end
        x = seq[:-1]
        y = seq[1:]
        
        # VQ-VAE tokens are discrete indices, so they MUST be LongTensors
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        
        # Keep the label as a LongTensor to condition the transformer
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
    # Test the new dataset class
    dataset = EncodedEMGDataset(
        csv_files=[".\data\encoded_df.csv"],
        filter_class=None, 
        which_file="train"
    )
    
    x, y, label = dataset[0]
    print(f"\nx shape: {x.shape}, y shape: {y.shape}")
    print(f"Gesture Ground Truth Label: {label}")
    print(f"Input tokens (x): {x[:10]}...")
    print(f"Target tokens (y): {y[:10]}...")