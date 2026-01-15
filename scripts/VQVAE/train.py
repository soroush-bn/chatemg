from tqdm import tqdm
import torch.nn as nn
import torch.Functional as F

def train_vqvae(model, dataloader,device,optimizer,config):
    model.train()
    criterion_recon = nn.MSELoss()

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
        # tqdm.write ensures it prints nicely above the progress bar
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Embed: {avg_embed:.4f}")
    return model
# Run Training