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
        
        # Constants based on your dataset structure
        self.SAMPLING_RATE = 2000
        self.REP_DURATION_SEC = 2.0
        self.SAMPLES_PER_REP = int(self.SAMPLING_RATE * self.REP_DURATION_SEC) # 4000 samples
        
        # Create directory
        self.save_dir = f"./models/{config['name']}/figs/"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[Visualizer] Saving all figures to: {self.save_dir}")

    def visualize_codebook(self, perplexity=30):
        """Standard Codebook t-SNE + Atomic Patterns"""
        self.model.eval()
        embeddings = self.model.quantizer.embedding.detach().cpu().numpy()
        usage_counts = self.model.quantizer.ema_cluster_size.detach().cpu().numpy()
        top_indices = np.argsort(usage_counts)[::-1]
        
        # t-SNE
        print("[Visualizer] Running t-SNE on codebook...")
        try:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            embeddings_2d = tsne.fit_transform(embeddings)
            
            fig1 = plt.figure(figsize=(10, 8))
            sc = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=np.log1p(usage_counts), cmap='viridis', alpha=0.7, s=30)
            plt.colorbar(sc, label='Log Usage Count')
            plt.title("t-SNE of Codebook Vectors")
            plt.savefig(os.path.join(self.save_dir, "codebook_tsne.png"), bbox_inches='tight')
            plt.close(fig1)
        except Exception as e:
            print(f"[Visualizer] t-SNE failed: {e}")

        # Atomic Patterns
        print("[Visualizer] Decoding top atomic patterns...")
        fig2, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        with torch.no_grad():
            for i, code_idx in enumerate(top_indices[:3]):
                code_vec = self.model.quantizer.embedding[code_idx]
                fake_latent = code_vec.view(1, -1, 1).repeat(1, 1, 20).to(self.device)
                sig = self.model.decoder(fake_latent)[0].cpu().numpy()
                for ch in range(8): axes[i].plot(sig[ch])
                axes[i].set_title(f"Code #{code_idx} (Usage: {int(usage_counts[code_idx])})")
        plt.savefig(os.path.join(self.save_dir, "atomic_patterns.png"), bbox_inches='tight')
        plt.close(fig2)

    def plot_data_distribution(self, dataloader, num_samples=1000):
        """Real vs Recon t-SNE"""
        self.model.eval()
        real_list, recon_list = [], []
        print(f"[Visualizer] Collecting {num_samples} samples for distribution check...")
        
        with torch.no_grad():
            collected = 0
            for x in dataloader:
                x = x.to(self.device)
                x_recon,_, _, _ = self.model(x)
                real_list.append(x.view(x.size(0), -1).cpu().numpy())
                recon_list.append(x_recon.view(x_recon.size(0), -1).cpu().numpy())
                collected += x.size(0)
                if collected >= num_samples: break
        
        combined = np.concatenate([np.concatenate(real_list)[:num_samples], 
                                   np.concatenate(recon_list)[:num_samples]])
        
        try:
            tsne = TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto')
            combined_2d = tsne.fit_transform(combined)
            fig = plt.figure(figsize=(10, 8))
            plt.scatter(combined_2d[:num_samples, 0], combined_2d[:num_samples, 1], c='#1f77b4', label='Real', alpha=0.5)
            plt.scatter(combined_2d[num_samples:, 0], combined_2d[num_samples:, 1], c='#ff7f0e', label='Recon', alpha=0.5)
            plt.legend()
            plt.title("Real vs Reconstructed Distribution")
            plt.savefig(os.path.join(self.save_dir, "data_distribution_tsne.png"), bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"[Visualizer] Distribution t-SNE failed: {e}")

    def plot_single_reconstruction(self, dataloader, sample_index=0):
        """Standard single sample check"""
        self.model.eval()
        try: batch = next(iter(dataloader)).to(self.device)
        except: return
        with torch.no_grad(): recon,_, _, _ = self.model(batch)
        
        orig = batch[sample_index].cpu().numpy()
        rec = recon[sample_index].cpu().numpy()
        
        fig, axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
        for ch in range(8):
            axes[ch].plot(orig[ch], 'k', alpha=0.6, label='Orig')
            axes[ch].plot(rec[ch], 'r--', label='Recon')
            axes[ch].set_ylabel(f'Ch {ch+1}')
        plt.suptitle(f"Sample {sample_index} Recon")
        plt.savefig(os.path.join(self.save_dir, f"single_recon_sample_{sample_index}.png"), bbox_inches='tight')
        plt.close(fig)

    def plot_gesture_pipeline(self, df, label_id, repetition_index=0):
        """
        Saves 'pipeline_trace_label_{id}_rep_{rep}.png'.
        Finds the exact repetition based on known structure (4000 samples/rep).
        
        Args:
            df: The dataframe with 'gt' and 'emg' columns.
            label_id: The integer label to find (e.g., 11).
            repetition_index: Which repetition to plot (0=1st rep of P1, 4=1st rep of P2, etc.)
        """
        self.model.eval()
        
        if 'gt' not in df.columns:
            print("[Visualizer] Error: 'gt' column missing.")
            return

        # 1. Get all indices for this label
        indices = df.index[df['gt'] == label_id].to_numpy()
        
        if len(indices) == 0:
            print(f"[Visualizer] Label {label_id} not found.")
            return

        # 2. Group indices into contiguous blocks
        # A "break" is where index jumps by more than 1
        breaks = np.where(np.diff(indices) > 1)[0]
        # Start points of blocks: [index[0], index[break+1], ...]
        block_starts = [indices[0]] + [indices[i+1] for i in breaks]
        
        # 3. Find valid repetitions within these blocks
        # We know each repetition is 4000 samples.
        valid_starts = []
        
        for start_idx in block_starts:
            # How long is this contiguous block?
            # We find the end of this specific block
            # (In a simpler way: just check forward from start_idx)
            
            # Since we know the structure is rigid (4 reps of 2s), 
            # we can just blindly assume the block contains N valid repetitions 
            # provided the data exists in the dataframe.
            
            # Check 1st Rep in block
            if self._is_valid_rep(df, start_idx, label_id):
                valid_starts.append(start_idx)
            
            # Check 2nd Rep (start + 4000)
            if self._is_valid_rep(df, start_idx + self.SAMPLES_PER_REP, label_id):
                valid_starts.append(start_idx + self.SAMPLES_PER_REP)
                
            # Check 3rd Rep
            if self._is_valid_rep(df, start_idx + 2*self.SAMPLES_PER_REP, label_id):
                valid_starts.append(start_idx + 2*self.SAMPLES_PER_REP)
                
            # Check 4th Rep
            if self._is_valid_rep(df, start_idx + 3*self.SAMPLES_PER_REP, label_id):
                valid_starts.append(start_idx + 3*self.SAMPLES_PER_REP)

        if repetition_index >= len(valid_starts):
            print(f"[Visualizer] Requested Repetition {repetition_index} not found. Max is {len(valid_starts)-1}.")
            return
            
        final_start = valid_starts[repetition_index]
        print(f"[Visualizer] Plotting Label {label_id}, Rep {repetition_index} (Start Index: {final_start})")

        # --- Process & Plot ---
        emg_cols = [c for c in df.columns if 'emg' in c.lower()]
        raw_vals = df[emg_cols].values
        
        raw_seg = raw_vals[final_start : final_start + self.SAMPLES_PER_REP]
        
        scaler = StandardScaler()
        scaler.fit(raw_vals) # Global fit
        proc_seg = scaler.transform(raw_seg)
        
        inp = torch.tensor(proc_seg, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            recon, _, _, _ = self.model(inp)
            recon_seg = recon[0].cpu().numpy()

        fig, axes = plt.subplots(8, 3, figsize=(18, 12), sharex='col')
        t = np.linspace(0, self.REP_DURATION_SEC, self.SAMPLES_PER_REP)
        
        titles = [f"Raw (Label {label_id})", "Preprocessed", "Reconstruction"]
        for i, tit in enumerate(titles): axes[0, i].set_title(tit, fontweight='bold')

        for ch in range(8):
            axes[ch, 0].plot(t, raw_seg[:, ch], 'k', alpha=0.7)
            axes[ch, 1].plot(t, proc_seg[:, ch], '#1f77b4', alpha=0.8)
            axes[ch, 2].plot(t, recon_seg[ch], '#d62728', alpha=0.8)
            axes[ch, 0].set_ylabel(f'Ch {ch+1}')
            for c in range(3): 
                axes[ch, c].grid(True, alpha=0.2)
                axes[ch, c].spines['top'].set_visible(False)
                axes[ch, c].spines['right'].set_visible(False)

        axes[7, 1].set_xlabel("Time (s)")
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"pipeline_trace_label_{label_id}_rep_{repetition_index}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Saved {save_path}")

    def _is_valid_rep(self, df, start_idx, label_id):
        """Helper to check if a 4000-sample window exists and matches the label."""
        if start_idx + self.SAMPLES_PER_REP >= len(df):
            return False
        # Optimization: Check start, middle, and end labels to avoid checking 4000 integers
        # (Assuming contiguous blocks are relatively clean)
        start_ok = (df['gt'].iloc[start_idx] == label_id)
        mid_ok = (df['gt'].iloc[start_idx + self.SAMPLES_PER_REP // 2] == label_id)
        end_ok = (df['gt'].iloc[start_idx + self.SAMPLES_PER_REP - 1] == label_id)
        return start_ok and mid_ok and end_ok