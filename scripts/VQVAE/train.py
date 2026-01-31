import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_vqvae(model, dataloader, device, optimizer, config):
    model.train()
    criterion_recon = nn.MSELoss()
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = f"./models/{config['name']}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Only ONE progress bar for the Epochs
    epoch_pbar = tqdm(range(config['number_of_epochs']), desc="Training Progress")

    for epoch in epoch_pbar:
        total_loss = 0
        total_recon = 0
        total_embed = 0

        # Iterate normally without tqdm to prevent log flooding
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            # Forward pass
            x_recon, loss_embed, _ = model(x)

            # Reconstruction Loss
            loss_recon = criterion_recon(x_recon, x)

            # Total Loss (Equation 6)
            loss = loss_recon + config['lambda_loss'] * loss_embed

            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_embed += loss_embed.item()

        # Calculate epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_embed = total_embed / len(dataloader)

        # Print summary ONLY at the end of the epoch
        tqdm.write(f"Epoch [{epoch+1}/{config['number_of_epochs']}] | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Embed: {avg_embed:.4f}")
        
        # --- CHECKPOINTING (Safety Save) ---
        # Save every 5 epochs OR if it's the last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == config['number_of_epochs']:
            save_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            tqdm.write(f"--> Checkpoint saved: {save_path}")

    return model