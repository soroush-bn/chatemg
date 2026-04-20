import os
import argparse

import torch
import numpy as np
import pandas as pd
import yaml

# Link to the main VQ-VAE model implementation
from VQVAE.model import SDformerVQVAE

class VQVAESignalDecoder:
    def __init__(self, vqvae_model_path, vqvae_config, device="cuda"):
        """
        Initializes the decoder and loads the pretrained SDformerVQVAE weights.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading VQ-VAE model from {vqvae_model_path} onto {self.device}...")
        
        # Instantiate your specific VQ-VAE model
        self.model = SDformerVQVAE(vqvae_config)
        
        # Handle potential DDP "module." prefix if the VQ-VAE was trained on multiple GPUs
        state_dict = torch.load(vqvae_model_path, map_location=self.device)
        unwanted_prefix = "module."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)
        
        # Based on your Encoder architecture (two stride=2 layers), 
        # 1 token = 4 raw time steps.
        self.downsample_factor = 4 

    def decode_window(self, window_indices):
        """
        Takes a subset of tokens representing a specific time window 
        and decodes them into raw continuous EMG/IMU signals.
        """
        # Ensure indices are properly shaped: (batch_size, sequence_length)
        if isinstance(window_indices, np.ndarray):
            window_indices = torch.tensor(window_indices, dtype=torch.long)
        
        if window_indices.dim() == 1:
            window_indices = window_indices.unsqueeze(0)
            
        window_indices = window_indices.to(self.device)
        
        with torch.no_grad():
            # 1. Map discrete indices to continuous vectors using the Quantizer's embedding table
            # Resulting shape: (Batch, Sequence_Length, Embedding_Dim)
            quantized_vectors = self.model.quantizer.embedding[window_indices]
            
            # 2. Your Decoder expects (Batch, Dim, Time), so we permute
            quantized_vectors = quantized_vectors.permute(0, 2, 1)
            
            # 3. Pass through the ResNet1D Decoder
            # Output shape: (Batch, Raw_Channels, Raw_Time_Steps)
            raw_signals = self.model.decoder(quantized_vectors)
            
            # 4. Permute back to standard time-series format: (Batch, Time, Channels)
            raw_signals = raw_signals.permute(0, 2, 1)

        return raw_signals.cpu().numpy()

    def decode_gesture(self, gesture_indices):
        """
        Decodes a full sequence of tokens representing a complete gesture.
        """
        raw_gesture = self.decode_window(gesture_indices)
        
        # Remove batch dimension if processing a single gesture
        if raw_gesture.shape[0] == 1:
            raw_gesture = raw_gesture.squeeze(0)
            
        return raw_gesture

    def decode_dataset(self, csv_path, save_dir):
        """
        Reads the generated dataset (tokens), decodes every row back into 
        raw continuous signals, and saves the output as a compressed Numpy archive.
        """
        print(f"Loading generated tokens from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        labels = df['gt'].values
        
        # Extract only the token columns
        token_cols = [c for c in df.columns if c.startswith('col_')]
        tokens = df[token_cols].values
        
        print(f"Decoding {len(tokens)} synthetic gestures...")
        
        # Decode the entire dataset in batches to avoid VRAM overflow if dataset is huge
        batch_size = 32
        raw_signals_list = []
        
        for i in range(0, len(tokens), batch_size):
            batch_tokens = tokens[i:i+batch_size]
            batch_signals = self.decode_window(batch_tokens)
            raw_signals_list.append(batch_signals)
            
        raw_signals = np.concatenate(raw_signals_list, axis=0)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "decoded_synthetic_signals.npz")
        
        # Save as a compressed .npz file containing both the signals and ground truth
        np.savez_compressed(
            save_path, 
            signals=raw_signals, 
            labels=labels
        )
        
        print(f"Successfully saved decoded signals to: {save_path}")
        print(f"Final Tensor Shape: {raw_signals.shape} (Samples, Time Steps, Channels)")
        
        return raw_signals, labels

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", type=str, required=True, help="Path to your trained VQ-VAE .pt file")
    parser.add_argument("--vqvae_config", type=str, required=True, help="Path to the config.yaml used to train the VQ-VAE")
    parser.add_argument("--generated_csv", type=str, required=True, help="Path to synthetic_encoded_samples.csv")
    parser.add_argument("--save_dir", type=str, default="./decoded_output", help="Where to save the raw signals")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Load the VQ-VAE configuration
    with open(args.vqvae_config, 'r') as file:
        vqvae_config = yaml.safe_load(file)
        
    # Initialize the decoder pipeline
    decoder = VQVAESignalDecoder(
        vqvae_model_path=args.vqvae_ckpt, 
        vqvae_config=vqvae_config
    )
    
    # Decode the entire generated CSV
    raw_dataset, dataset_labels = decoder.decode_dataset(
        csv_path=args.generated_csv, 
        save_dir=args.save_dir
    )