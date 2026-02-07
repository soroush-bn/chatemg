import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
def evaluate_model(model, dataloader, device,config ):
    model.eval()
    total_mse = 0
    all_indices = []
    
    with torch.no_grad():
        for i, x in enumerate(dataloader):
            x = x.to(device)
            
            x_recon, _, _, indices = model(x)
            
            mse = F.mse_loss(x_recon, x)
            total_mse += mse.item()
            
            # Collect indices to check codebook usage
            all_indices.append(indices.cpu())
            
            if i == 0:
                orig = x[0].cpu().numpy()       # [Channels, Length]
                recon = x_recon[0].cpu().numpy() # [Channels, Length]
                
                # Plot first 4 channels
                fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                time_steps = range(orig.shape[1])
                
                for ch in range(4):
                    axes[ch].plot(time_steps, orig[ch], label='Original', color='black', alpha=0.7)
                    axes[ch].plot(time_steps, recon[ch], label='Reconstructed', color='#2ca02c', linestyle='--')
                    axes[ch].set_ylabel(f'EMG {ch+1}')
                    axes[ch].legend(loc='upper right')
                    axes[ch].grid(True, alpha=0.3)
                
                plt.xlabel('Time Steps')
                plt.suptitle('Stage 1 Evaluation: Original vs. Reconstructed Signals')
                plt.tight_layout()

                graphs_dir = './graphs'
                os.makedirs(graphs_dir, exist_ok=True)
                plot_path = os.path.join(graphs_dir, f"{config['name']}_stage1_reconstruction.png")
                plt.savefig(plot_path)
                plt.close(fig)

    avg_mse = total_mse / len(dataloader)
    
    all_indices = torch.cat(all_indices).flatten()
    unique_codes = torch.unique(all_indices).numel()
    total_codes = config['codebook_size']
    usage_percent = (unique_codes / total_codes) * 100
    
    print(f"--- Evaluation Results ---")
    print(f"Average Reconstruction MSE: {avg_mse:.5f}")
    print(f"Codebook Usage: {unique_codes}/{total_codes} codes used ({usage_percent:.1f}%)")
    
    if usage_percent < 5.0:
        print("WARNING: Low codebook usage detected. Your model might be suffering from code collapse.")

