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

    # determine if this process should perform logging/checkpointing
    master_process = config.get('master_process', True)

    # Optional WandB logging initialization (safe for non-interactive servers)
    if config.get('wandb_log', False) and master_process:
        try:
            import wandb

            # Login for non-interactive environments (servers/LSF)
            wandb_api_key = os.environ.get('WANDB_API_KEY', None)
            if wandb_api_key:
                print("Logging into W&B using WANDB_API_KEY environment variable...")
                wandb.login(key=wandb_api_key)
            else:
                print("Warning: WANDB_API_KEY not found. Attempting to use cached credentials...")
                try:
                    wandb.login()
                except Exception as e:
                    print(f"W&B login failed: {e}")
                    print("Please set WANDB_API_KEY environment variable or run 'wandb login' manually")
                    config['wandb_log'] = False

            if config.get('wandb_log', False):
                wandb.init(project=config.get('wandb_project', 'vqvae'), name=config['name'], config=config)
        except Exception as e:
            print(f"W&B import/init error: {e}")
            config['wandb_log'] = False

    # Only ONE progress bar for the Epochs
    epoch_pbar = tqdm(range(config['number_of_epochs']), desc="Training Progress")

    for epoch in epoch_pbar:
        total_loss = 0
        total_recon = 0
        total_embed = 0
        total_commitment = 0
        total_codebook = 0

        # Iterate normally without tqdm to prevent log flooding
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)

            optimizer.zero_grad()

            # Forward pass
            x_recon, commitment_loss, codebook_loss, _ = model(x)

            # Reconstruction Loss
            loss_recon = criterion_recon(x_recon, x)

            # Embedding Loss (Equation 6) with configurable lambda weighting
            # Scaled by sqrt(code_dim) to normalize for dimensionality
            loss_embed = ((config['lambda_loss'] * commitment_loss) + codebook_loss) / (config['code_dim'] ** 0.5)

            # Total Loss
            loss = loss_recon + loss_embed

            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_embed += loss_embed.item()
            total_commitment += commitment_loss.item()
            total_codebook += codebook_loss.item()

        # Calculate epoch averages
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_embed = total_embed / len(dataloader)
        avg_commitment = total_commitment / len(dataloader)
        avg_codebook = total_codebook / len(dataloader)

        # Print summary ONLY at the end of the epoch
        tqdm.write(f"Epoch [{epoch+1}/{config['number_of_epochs']}] | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Embed: {avg_embed:.4f} | Commitment: {avg_commitment:.4f} | Codebook: {avg_codebook:.4f}")
        # Log to WandB (if enabled)
        if config.get('wandb_log', False) and master_process:
            try:
                import wandb

                # compute MSE explicitly (same as recon here)
                mse_val = avg_recon
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "epoch/total_loss": avg_loss,
                        "epoch/recon_loss": avg_recon,
                        "epoch/embed_loss": avg_embed,
                        "epoch/commitment_loss": avg_commitment,
                        "epoch/codebook_loss": avg_codebook,
                        "epoch/mse": mse_val,
                    }
                )
            except Exception as e:
                print(f"W&B logging error: {e}")
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
            # upload checkpoint to wandb run files (best-effort)
            if config.get('wandb_log', False) and master_process:
                try:
                    import wandb

                    wandb.save(save_path)
                except Exception as e:
                    print(f"W&B save error: {e}")

    # Finish WandB run (if initialized)
    if config.get('wandb_log', False) and master_process:
        try:
            import wandb

            wandb.finish()
        except Exception as e:
            print(f"W&B finish error: {e}")

    return model