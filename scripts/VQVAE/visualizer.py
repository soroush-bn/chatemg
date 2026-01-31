import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

class Visualizer:
    def __init__(self, model, device, config):
        """
        Args:
            model: The trained VQ-VAE model.
            device: torch.device ('cuda' or 'cpu').
            config: The configuration dictionary containing 'name'.
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Create the specific directory for this run's figures
        self.save_dir = f"./models/{config['name']}/figs/"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[Visualizer] Saving all figures to: {self.save_dir}")

    def visualize_codebook(self, perplexity=30):
        """
        Saves two plots:
        1. 'codebook_tsne.png': t-SNE of codebook weights colored by usage.
        2. 'atomic_patterns.png': Decoded waveforms of the top 3 used codes.
        """
        self.model.eval()
        
        # --- Data Gathering ---
        embeddings = self.model.quantizer.embedding.detach().cpu().numpy()
        usage_counts = self.model.quantizer.ema_cluster_size.detach().cpu().numpy()
        top_indices = np.argsort(usage_counts)[::-1]
        
        print(f"[Visualizer] Top 5 most used codes: {top_indices[:5]}")

        # --- Plot 1: t-SNE ---
        print("[Visualizer] Running t-SNE on codebook...")
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            embeddings_2d = tsne.fit_transform(embeddings)
            
            fig1 = plt.figure(figsize=(10, 8))
            sc = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=np.log1p(usage_counts), cmap='viridis', alpha=0.7, s=30)
            plt.colorbar(sc, label='Log Usage Count')
            plt.title("t-SNE of Codebook Vectors (Color = Usage)")
            plt.xlabel("Dim 1")
            plt.ylabel("Dim 2")
            plt.grid(True, alpha=0.3)
            
            save_path1 = os.path.join(self.save_dir, "codebook_tsne.png")
            plt.savefig(save_path1, bbox_inches='tight')
            plt.close(fig1)
            print(f"[Visualizer] Saved {save_path1}")
        except Exception as e:
            print(f"[Visualizer] t-SNE failed: {e}")

        # --- Plot 2: Atomic Patterns ---
        print("[Visualizer] Decoding top atomic patterns...")
        num_codes = 3
        fig2, axes = plt.subplots(num_codes, 1, figsize=(12, 4 * num_codes), sharex=True)
        if num_codes == 1: axes = [axes]
        
        with torch.no_grad():
            for i, code_idx in enumerate(top_indices[:num_codes]):
                code_vec = self.model.quantizer.embedding[code_idx]
                # Repeat to make a sequence
                fake_latent = code_vec.view(1, -1, 1).repeat(1, 1, 20).to(self.device)
                sig = self.model.decoder(fake_latent)[0].cpu().numpy()
                
                for ch in range(sig.shape[0]):
                    axes[i].plot(sig[ch], label=f'Ch {ch+1}' if i==0 else "")
                
                axes[i].set_title(f"Pattern for Code #{code_idx} (Usage: {int(usage_counts[code_idx])})")
                axes[i].set_ylabel("Amplitude")
                axes[i].grid(True, alpha=0.3)
                if i == 0: axes[i].legend(loc='upper right')

        plt.xlabel("Time Steps")
        plt.tight_layout()
        save_path2 = os.path.join(self.save_dir, "atomic_patterns.png")
        plt.savefig(save_path2, bbox_inches='tight')
        plt.close(fig2)
        print(f"[Visualizer] Saved {save_path2}")

    def plot_data_distribution(self, dataloader, num_samples=1000):
        """
        Saves 'data_distribution_tsne.png': 
        Overlap of Real (Blue) vs Reconstructed (Orange) data distributions.
        """
        self.model.eval()
        real_list, recon_list = [], []
        
        print(f"[Visualizer] Collecting {num_samples} samples for distribution check...")
        with torch.no_grad():
            collected = 0
            for x in dataloader:
                x = x.to(self.device)
                x_recon, _, _ = self.model(x)
                
                # Flatten windows to vectors
                real_list.append(x.view(x.size(0), -1).cpu().numpy())
                recon_list.append(x_recon.view(x_recon.size(0), -1).cpu().numpy())
                
                collected += x.size(0)
                if collected >= num_samples: break
        
        real_data = np.concatenate(real_list)[:num_samples]
        recon_data = np.concatenate(recon_list)[:num_samples]
        combined = np.concatenate([real_data, recon_data])
        
        print(f"[Visualizer] Running t-SNE on {combined.shape[0]} samples...")
        try:
            tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42)
            combined_2d = tsne.fit_transform(combined)
            
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(combined_2d[:num_samples, 0], combined_2d[:num_samples, 1], 
                        c='#1f77b4', label='Real EMG', alpha=0.5, s=15)
            plt.scatter(combined_2d[num_samples:, 0], combined_2d[num_samples:, 1], 
                        c='#ff7f0e', label='Reconstructed', alpha=0.5, s=15)
            
            plt.title("t-SNE: Real vs. Reconstructed Data")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(self.save_dir, "data_distribution_tsne.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            print(f"[Visualizer] Saved {save_path}")
        except Exception as e:
            print(f"[Visualizer] Distribution t-SNE failed: {e}")

    def plot_single_reconstruction(self, dataloader, sample_index=0):
        """
        Saves 'single_recon_sample_{index}.png': 
        8-channel comparison of Input vs Output for one sample.
        """
        self.model.eval()
        try:
            batch = next(iter(dataloader)).to(self.device)
        except StopIteration:
            print("[Visualizer] Dataloader empty.")
            return

        with torch.no_grad():
            recon, _, _ = self.model(batch)
        
        orig = batch[sample_index].cpu().numpy()
        rec = recon[sample_index].cpu().numpy()
        mse = np.mean((orig - rec)**2)
        
        fig, axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
        time_steps = range(orig.shape[1])

        for ch in range(8):
            axes[ch].plot(time_steps, orig[ch], 'k', alpha=0.6, label='Original')
            axes[ch].plot(time_steps, rec[ch], 'r--', label='Recon')
            axes[ch].set_ylabel(f'Ch {ch+1}')
            axes[ch].grid(True, alpha=0.2)
            axes[ch].spines['top'].set_visible(False)
            axes[ch].spines['right'].set_visible(False)
            if ch == 0: axes[ch].legend(loc='upper right')

        plt.suptitle(f"Sample {sample_index} Reconstruction (MSE: {mse:.5f})", y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f"single_recon_sample_{sample_index}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Saved {save_path}")

    def plot_gesture_pipeline(self, df, label_name="Power Grip", label_map=None, duration_sec=2.0):
        """
        Saves 'pipeline_trace_{label}.png': 
        3-Column plot showing Raw -> Preprocessed -> Reconstructed.
        """
        self.model.eval()
        emg_cols = [c for c in df.columns if 'emg' in c.lower()]
        raw_values = df[emg_cols].values
        window_len = int(2000 * duration_sec)
        
        # --- Find Segment ---
        best_start = 0
        found = False
        
        # Strategy A: By Label
        if label_map and label_name in label_map and 'gt' in df.columns:
            target_id = label_map[label_name]
            indices = df.index[df['gt'] == target_id].tolist()
            for idx in indices:
                if idx + window_len < len(df) and all(df['gt'].iloc[idx:idx+100] == target_id): # Check first 100 samples
                    best_start = idx
                    found = True
                    break
        
        # Strategy B: By Energy
        if not found:
            print(f"[Visualizer] '{label_name}' not found by label, scanning for energy...")
            max_var = 0
            for i in range(0, len(df) - window_len, 2000):
                var = np.var(raw_values[i:i+window_len])
                if var > max_var:
                    max_var = var
                    best_start = i

        # --- Process ---
        raw_seg = raw_values[best_start : best_start + window_len]
        
        # Important: Fit scaler on global data for correct relative scaling
        scaler = StandardScaler()
        scaler.fit(raw_values)
        proc_seg = scaler.transform(raw_seg)
        
        inp_tensor = torch.tensor(proc_seg, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            recon_tensor, _, _ = self.model(inp_tensor)
            recon_seg = recon_tensor[0].cpu().numpy()

        # --- Plot ---
        fig, axes = plt.subplots(8, 3, figsize=(18, 12), sharex='col')
        time_axis = np.linspace(0, duration_sec, window_len)
        titles = [f"Raw ({label_name})", "Preprocessed", "Reconstruction"]
        
        for i, title in enumerate(titles):
            axes[0, i].set_title(title, fontweight='bold')

        for ch in range(8):
            # Raw
            axes[ch, 0].plot(time_axis, raw_seg[:, ch], 'k', alpha=0.7, lw=1)
            # Proc
            axes[ch, 1].plot(time_axis, proc_seg[:, ch], '#1f77b4', alpha=0.8, lw=1)
            # Recon
            axes[ch, 2].plot(time_axis, recon_seg[ch], '#d62728', alpha=0.8, lw=1)
            
            for col in range(3):
                axes[ch, col].grid(True, alpha=0.2)
                axes[ch, col].spines['top'].set_visible(False)
                axes[ch, col].spines['right'].set_visible(False)
            
            axes[ch, 0].set_ylabel(f'Ch {ch+1}')

        axes[7, 1].set_xlabel("Time (s)")
        plt.tight_layout()
        
        clean_name = label_name.replace(" ", "_")
        save_path = os.path.join(self.save_dir, f"pipeline_trace_{clean_name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Saved {save_path}")