import os
import pathlib

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from torch.utils.data import Dataset

import misc_utils as mu


def reconstruct_dataframe_from_filtered_data(filtered_data_list, label_list=None, filter_class=None, sensor_type="emg", location="both"):
    """
    Reconstructs the original dataframe from filtered_data_list and label_list.
    
    Args:
        filtered_data_list: List of numpy arrays, each with shape (timesteps, channels)
        label_list: Optional list of numpy arrays containing ground truth labels. 
                   If None and filter_class is provided, all labels will be set to filter_class.
        filter_class: The class label used for filtering (if applicable)
        sensor_type: Type of sensor data ('emg' or 'imu')
        location: Location of sensors ('both', 'forearm', or 'wrist')
    
    Returns:
        pd.DataFrame: Reconstructed dataframe with sensor columns and 'gt' column
    """
    # Concatenate all chunks
    all_data = np.concatenate(filtered_data_list, axis=0)
    
    # Handle labels
    if label_list is not None:
        if len(filtered_data_list) != len(label_list):
            raise ValueError(f"Mismatch: {len(filtered_data_list)} data arrays but {len(label_list)} label arrays")
        all_labels = np.concatenate(label_list, axis=0)
    elif filter_class is not None:
        # If no label_list but filter_class is provided, assume all data belongs to that class
        all_labels = np.full(len(all_data), filter_class)
    else:
        raise ValueError("Either label_list or filter_class must be provided")
    
    # Determine column names based on sensor type and location
    if location == "both":
        num_channels = 8
        if sensor_type == "emg":
            col_names = [str(i) for i in range(1, 9)]
        else:
            col_names = [f"{sensor_type}_{i}" for i in range(1, 9)]
    elif location == "forearm":
        num_channels = 4
        if sensor_type == "emg":
            col_names = ["1", "2", "3", "4"]
        else:
            col_names = [f"{sensor_type}_{i}" for i in range(1, 5)]
    elif location == "wrist":
        num_channels = 4
        if sensor_type == "emg":
            col_names = ["5", "6", "7", "8"]
        else:
            col_names = [f"{sensor_type}_{i}" for i in range(5, 9)]
    else:
        raise ValueError(f"Invalid location: {location}")
    
    if all_data.shape[1] != num_channels:
        raise ValueError(f"Expected {num_channels} channels for location '{location}', got {all_data.shape[1]}")
    
    # Create dataframe
    df = pd.DataFrame(all_data, columns=col_names)
    df['gt'] = all_labels
    
    return df


class ChatEMGDataset(Dataset):
    #chera constructor enghadr functionality dare????? :||||
    def __init__(
        self,
        csv_files,
        filter_class,
        block_size,
        clip_min=0,
        clip_max=999,
        vocab_size=128,
        ds_factor = 2 ,
        median_filter_size=1,
        shift=False,
        flip=False,
        sensor_type="accel",
        axis="x",
        which_file= "train",
        location= "both"
    ):
        # assert block_size < mu.SAMPLING_FREQ * mu.GESTURE_DURATION_SEC / ds_factor
        self.csv_files = csv_files
        self.filter_class = filter_class
        self.block_size = block_size
        self.shift = shift
        self.flip = flip
        self.sensor_type = sensor_type
        self.location = location
        self.vocab_size = vocab_size
        self.axis = axis
        self.which_file = which_file
        data_list = []
        label_list = []
        print(type(median_filter_size))
        if not isinstance(median_filter_size, int):
            raise Exception("Manual Exception: median_filter_size is not of type int")
        if median_filter_size != 1:
            print(f"Using median filter with size {median_filter_size}")
        df_all_subjects = pd.DataFrame()
        for f in self.csv_files:
            data_path = f
            df = pd.read_csv(data_path, index_col=0)
            print(f"\n[SHAPE TRACK] Loaded CSV from {os.path.basename(data_path)}: {df.shape}")
            
            # Stratified split - get 90%/10% of each label class
            if which_file == "train":
                # Get first three rep
                train_list = []
                for label in df['gt'].unique():
                    label_df = df[df['gt'] == label]
                    split_index = int(0.7 * len(label_df))
                    train_list.append(label_df.iloc[:split_index])
                df = pd.concat(train_list, ignore_index=True)
                print(f"[SHAPE TRACK] After train split (70%): {df.shape}")
            elif which_file == "sample":
                # Get last rep
                test_list = []
                for label in df['gt'].unique():
                    label_df = df[df['gt'] == label]
                    split_index = int(0.7 * len(label_df))
                    test_list.append(label_df.iloc[split_index:])
                df = pd.concat(test_list, ignore_index=True)
                print(f"[SHAPE TRACK] After sample split (30%): {df.shape}")
            elif which_file == "generate":
                pass
                # whole data
            df_all_subjects = pd.concat([df_all_subjects, df], ignore_index=True)
            print(f"[SHAPE TRACK] After concatenating subject data: {df_all_subjects.shape}")

            print("df before downsampling:", df_all_subjects.shape)
            print(df_all_subjects["gt"].unique())
            print(df_all_subjects.describe())
            if ds_factor> 1 :
                df_all_subjects =mu.downsample_with_proper_filter(df_all_subjects, factor= ds_factor)
            print("df after downsampling:", df_all_subjects.shape)
            print(df_all_subjects["gt"].unique())
            print(df_all_subjects.describe())
            X, y = mu.clean_dataframe(df_all_subjects,vocab_size,sensor_type,location)
            print(f"[SHAPE TRACK] After clean_dataframe - X: {X.shape}, y: {y.shape}")
            X = np.clip(X, a_min=clip_min, a_max=clip_max)
            print(f"[SHAPE TRACK] After clipping (min={clip_min}, max={clip_max}) - X: {X.shape}")
            if median_filter_size != 1:
                X = medfilt(X, kernel_size=[median_filter_size, 1])
                print(f"[SHAPE TRACK] After median filter (size={median_filter_size}) - X: {X.shape}")
            
            data_list.append(X)
            label_list.append(y)
            print(f"[SHAPE TRACK] Added to data_list - X: {X.shape}, y: {y.shape}")
        print(f"Number of loaded files: {len(data_list)}")
        print(f"[SHAPE TRACK] Total data_list shapes: {[d.shape for d in data_list]}")
        print(label_list)
        # filtering data based on class labels
        self.filtered_data_list = data_list
        print("filter class is" , self.filter_class)
        print("block size is" , self.block_size)
        print(f"Chunk shapes: {[d.shape for d in self.filtered_data_list]}")

        if self.filter_class is not None:
            print(f"[SHAPE TRACK] Filtering for class {self.filter_class}...")
            self.filtered_data_list = []
            for d, l in zip(data_list, label_list):

                filtered_d = []
                for i in range(len(d)):
                    if l[i] == self.filter_class:
                        filtered_d.append(d[i])
                        if i + 1 == len(d) or l[i + 1] != self.filter_class:
                            self.filtered_data_list.append(np.array(filtered_d))
                            filtered_d = []
            print(f"[SHAPE TRACK] After filtering for class {self.filter_class}: {len(self.filtered_data_list)} chunks")
        # I assume that each chunk in filtered_data_list is one repertition of a gesture
        print(f"Chunk shapes: {[d.shape for d in self.filtered_data_list]}")


        # now I am removing chunks shorter than block size + 1, because we need to consider y as well
        print(f"[SHAPE TRACK] Before removing short chunks (<{self.block_size + 1}): {len(self.filtered_data_list)} chunks")
        self.filtered_data_list = [
            d for d in self.filtered_data_list if len(d) >= (self.block_size + 1)
        ]
        print("after removing short chunks, number of chunks is" , len(self.filtered_data_list))
        print(f"[SHAPE TRACK] After removing short chunks: {[d.shape for d in self.filtered_data_list]}")
        # Data augmentation for inter-channel setup
        if self.shift:
            print(f"[SHAPE TRACK] Before shift augmentation: {len(self.filtered_data_list)} chunks")
            augment_list = []
            for d in self.filtered_data_list:
                for i in range(1, 8 if self.location == "both" else 4  ):  # shift 7 times
                    d_shifted = np.roll(d, i, axis=-1)
                    augment_list.append(d_shifted)
            for ad in augment_list:
                self.filtered_data_list.append(ad)
            print(f"[SHAPE TRACK] After shift augmentation: {len(self.filtered_data_list)} chunks (added {len(augment_list)} shifted)")

        if self.flip:
            print(f"[SHAPE TRACK] Before flip augmentation: {len(self.filtered_data_list)} chunks")
            augment_list = []
            for d in self.filtered_data_list:
                d_flipped = np.flip(d, axis=-1).copy()
                augment_list.append(d_flipped)
            for ad in augment_list:
                self.filtered_data_list.append(ad)
            print(f"[SHAPE TRACK] After flip augmentation: {len(self.filtered_data_list)} chunks (added {len(augment_list)} flipped)")

        # compute mean and std
        concatenated_data = np.concatenate(self.filtered_data_list)
        print(f"[SHAPE TRACK] Concatenated all chunks for statistics: {concatenated_data.shape}")
        self.mean = np.mean(concatenated_data, axis=0)
        self.std = np.std(concatenated_data, axis=0)
        print(f"[SHAPE TRACK] Computed mean shape: {self.mean.shape}, std shape: {self.std.shape}")
        self.filtered_data_lens = [
            len(d) - self.block_size for d in self.filtered_data_list
        ]

        #      [chunk_idx, position]
        #[0] → [0, 0]        # sample 0 from chunk 0, position 0
        #[1] → [0, 1]        # sample 1 from chunk 0, position 1
        #...
        #[243] → [0, 243]    # sample 243 from chunk 0, position 243
        #[244] → [1, 0]      # sample 244 from chunk 1, position 0
        #[245] → [1, 1]      # sample 245 from chunk 1, position 1
        # first element is which sublist, second element is which position in the sublist
        self.table = np.zeros((sum(self.filtered_data_lens), 2), dtype=int)
        s = 0
        for i, l in enumerate(self.filtered_data_lens):
            self.table[s : s + l, 0] = i
            self.table[s : s + l, 1] = range(l)
            s = s + l
        print(f"total number of 8-channel samples: {sum(self.filtered_data_lens)}")
        print(f"[SHAPE TRACK] Final lookup table shape: {self.table.shape}")
        print(f"[SHAPE TRACK] ===== DATASET INITIALIZATION COMPLETE =====")
        print(f"[SHAPE TRACK] Total samples available: {sum(self.filtered_data_lens)}")
        print(f"[SHAPE TRACK] Number of chunks: {len(self.filtered_data_list)}")
        print(f"[SHAPE TRACK] Sample shapes (first 5 chunks): {[d.shape for d in self.filtered_data_list[:5]]}")

    def get_len(self):
        return sum(self.filtered_data_lens)

    def __len__(self):
        return self.get_len()

    def __getitem__(self, item):
        return self.get_sample(item)

    def get_chunk(self, item):
        assert item < len(self.filtered_data_list)
        return self.filtered_data_list[item]

    def get_all_chunks(self):
        return self.filtered_data_list
    
    def get_one_rep(self, chunk,which_rep=3):
        if self.which_file == "generate":
            
            d =chunk
            rep_length = len(d) // mu.NUMBER_OF_REPEATS
            if which_rep ==None :
                #return chunk devided to 4 repetitions
                return [d[i*rep_length:(i+1)*rep_length] for i in range(mu.NUMBER_OF_REPEATS)]
            else:
                start_idx = which_rep * rep_length
                end_idx = start_idx + rep_length
                return d[start_idx:end_idx]
        else:
            raise Exception("get_one_rep is only for which_file='generate'")
    def get_not_one_rep(self, chunk,which_rep=3):
        if self.which_file == "generate":
            d =chunk
            rep_length = len(d) // mu.NUMBER_OF_REPEATS 
            start_idx = which_rep * rep_length
            end_idx = start_idx + rep_length
            return np.concatenate([d[:start_idx], d[end_idx:]], axis=0)
        else:
            raise Exception("get_not_one_rep is only for which_file='generate'")
        
    def get_sample(self, item):
        assert item < self.__len__()
        # return (256, 8)
        a, b = self.table[item]
        x = self.filtered_data_list[a][b : b + self.block_size]
        # in y label nist, chiziye ke mikhaym predict konim
        y = self.filtered_data_list[a][b + 1 : b + 1 + self.block_size]
        return x, y

    def sample(self, num):
        idx = np.random.choice(range(self.__len__()), num, replace=False)
        X = []
        Y = []
        for i in idx:
            x, y = self.__getitem__(i)
            X.append(x)
            Y.append(y)
        return np.stack(X), np.stack(Y)


if __name__ == "__main__":
    # Configuration matching sample.py
    config = {
        'converted_data_path': 'E:\\projects\\chatemgserver\\chatemg\\data',
        'sensor_type': 'emg',
        'participants_list_ids': ['p1', 'p3', 'p4', 'p7', 'p8'],
        'filter_class': 15,
        'token_len': 256,
        'vocab_size': 128,
        'ds_factor': 2,
        'median_filter_size': 1,
        'location': 'both'
    }
    
    # Replicate dataset creation from sample.py
    emg_data_paths = [
        os.path.join(config['converted_data_path'], subject_id, "converted_emg.csv") 
        for subject_id in config['participants_list_ids']
    ]
    
    data_file_full_path = os.path.join(
        config['converted_data_path'], 
        f"merged_{config['sensor_type']}.csv"
    )
    
    sample_data_files = emg_data_paths if config['sensor_type'] == "emg" else [
        data_file_full_path,
    ]
    
    dataset = ChatEMGDataset(
        csv_files=sample_data_files,
        filter_class=config['filter_class'],
        block_size=config['token_len'],
        vocab_size=config['vocab_size'],
        ds_factor=config['ds_factor'],
        median_filter_size=config['median_filter_size'],
        sensor_type=config['sensor_type'],
        which_file="sample",
        location=config['location']
    )
    
    print(f"num samples: {len(dataset)}")
    x, y = dataset[0]
    print(f"x shape: {x.shape}, y shape: {y.shape}")
    print(f"Number of chunks in filtered_data_list: {len(dataset.filtered_data_list)}")
    print(f"Chunk shapes: {[d.shape for d in dataset.filtered_data_list]}")
    
    # Test the reconstruction function
    print("\n--- Testing reconstruction function ---")
    reconstructed_df = reconstruct_dataframe_from_filtered_data(
        filtered_data_list=dataset.filtered_data_list,
        filter_class=config['filter_class'],
        sensor_type=config['sensor_type'],
        location=config['location']
    )
    print(f"Reconstructed dataframe shape: {reconstructed_df.shape}")
    print(f"Reconstructed dataframe columns: {list(reconstructed_df.columns)}")
    print(f"Unique labels in reconstructed df: {reconstructed_df['gt'].unique()}")
    print(f"First few rows:\n{reconstructed_df.head()}")
