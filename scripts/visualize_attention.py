import os
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from encoded_model import ConditionedGPT, GPTConfig
from encoded_dataset import EncodedEMGDataset
import argparse
import pathlib

def visualize_attention(config_path, vqvae_config_path, model_path, layer_to_viz=0, sample_idx=0, save_dir="./attention_plots"):
    # 1. Load Configs
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(vqvae_config_path, "r") as f:
        vq_conf = yaml.safe_load(f)
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. Initialize Model
    model_args = dict(
        n_layer=config.get('n_layer', 8),
        n_head=config.get('n_head', 8),
        n_embd=config.get('n_embd', 512),
        block_size=config.get('block_size', 75),
        vocab_size=config.get('vocab_size', 512),
        num_classes=config.get('num_classes', 17),
        dropout=0, # No dropout for viz
        bias=config.get('bias', True)
    )
    gptconf = GPTConfig(**model_args)
    model = ConditionedGPT(gptconf)
    
    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model']
    # Fix potential DDP prefixing
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Hook into Attention to capture weights
    # We must force manual attention (disable flash) to get the 'att' matrix
    attention_weights = []
    
    def hook_fn(module, input, output):
        # Re-calculate attention manually to get the weights
        x = input[0]
        B, T, C = x.size()
        q, k, v = module.c_attn(x).split(module.n_embd, dim=2)
        k = k.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        q = q.view(B, T, module.n_head, C // module.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(k.size(-1))))
        # Apply causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = torch.nn.functional.softmax(att, dim=-1)
        attention_weights.append(att.detach().cpu().numpy())

    # Attach hook to the specified layer
    target_layer = model.transformer.h[layer_to_viz].attn
    target_layer.register_forward_hook(hook_fn)

    # 4. Load a sample from Validation Set
    val_data_path = config.get('val_data_path')
    dataset = EncodedEMGDataset(csv_files=[val_data_path])
    X, Y, label = dataset[sample_idx]
    X = X.unsqueeze(0).to(device) # Add batch dim
    label = torch.tensor([label]).to(device)

    # 5. Forward Pass
    with torch.no_grad():
        _ = model(X, labels=label)

    # 5b. Reconstruct the Input Signal for context
    from decoder import VQVAESignalDecoder
    vq_name = vq_conf['name']
    vq_ckpt = f"VQVAE/models/{vq_name}/final_model.pth"
    
    try:
        decoder = VQVAESignalDecoder(vqvae_model_path=vq_ckpt, vqvae_config=vq_conf)
        recon_input = decoder.decode_window(X)[0] # (Time, Channels)
        
        fig_inp, ax_inp = plt.subplots(figsize=(10, 4))
        for c in range(recon_input.shape[1]):
            ax_inp.plot(recon_input[:, c], alpha=0.7)
        ax_inp.set_title(f"Reconstructed Input Signal (Sample {sample_idx} | Label {label.item()})")
        ax_inp.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"input_signal_sample_{sample_idx}.png"))
        plt.close(fig_inp)
        print(f"Saved reconstructed input signal to {save_dir}")
    except Exception as e:
        print(f"Could not reconstruct input signal: {e}")

    # 6. Plotting
    # attention_weights[0] shape: (1, n_head, T, T)
    weights = attention_weights[0][0]
    n_heads = weights.shape[0]
    
    # We will plot a subset of heads if there are too many (e.g., 64)
    heads_to_plot = min(n_heads, 16) 
    cols = 4
    rows = int(np.ceil(heads_to_plot / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f"Layer {layer_to_viz} Attention Heads | Gesture {label.item()} | Sample {sample_idx}", fontsize=20)
    axes = axes.flatten()

    for i in range(heads_to_plot):
        sns.heatmap(weights[i], ax=axes[i], cmap='viridis', cbar=False)
        axes[i].set_title(f"Head {i}")
        axes[i].set_xlabel("Key Position")
        axes[i].set_ylabel("Query Position")

    for j in range(heads_to_plot, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"attention_layer_{layer_to_viz}_sample_{sample_idx}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved attention visualization to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--vqvae_config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--layer', type=int, default=0, help="Which transformer layer to visualize")
    parser.add_argument('--sample', type=int, default=0, help="Which sample index from validation set")
    parser.add_argument('--save_dir', type=str, default=".", help="Where to save the plot")
    args = parser.parse_args()
    
    visualize_attention(args.config, args.vqvae_config, args.ckpt, args.layer, args.sample, args.save_dir)
