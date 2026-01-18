#load data
import os
import pandas as pd
import yaml 
from dataset import EMGDataset
from evaluation import evaluate_model
from torch.utils.data import  DataLoader
from model import SDformerVQVAE
import torch
from train import train_vqvae
with open("vqvae_config.yaml", "r") as file:
    config = yaml.safe_load(file)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Assignment ready on {device}.")
# Initialize Dataset and DataLoader
# Use the dataframe you already loaded in your previous code
dataset = EMGDataset(window_size=config['window_size'],stride=config['stride'])
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

print(f"Dataset created. Shape of one batch: {next(iter(dataloader)).shape}")
#asset that shape of one batch is [batch_size, channels, window_size]
assert next(iter(dataloader)).shape == (config['batch_size'], 8, config['window_size']), f"Unexpected batch shape: {next(iter(dataloader)).shape}"



# create model 

model = SDformerVQVAE(config).to(device)
learning_rate = float(config['learning_rate'])
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(model)
# train model 
train_vqvae(model, dataloader,device,optimizer,config)


os.makedirs("./models/", exist_ok=True)

torch.save(model.state_dict(), f"./models/{config['name']}.pth")
print(f"Model saved to ./models/{config['name']}.pth")

#evaluate model 

evaluate_model(model, dataloader, device,config )

print("Evaluation completed.")
