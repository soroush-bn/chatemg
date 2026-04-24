import torch
import torch.nn as nn

class LatentCNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=17):
        super().__init__()
        # input_size is the code_dim (embedding dimension)
        self.network = nn.Sequential(
            # x shape: [Batch, code_dim, seq_len]
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Global pooling after temporal convolutions
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # x is [Batch, seq_len, code_dim] -> Conv1d expects [Batch, code_dim, seq_len]
        x = x.transpose(1, 2)
        return self.network(x)

# Keeping the name LatentMLP for compatibility or aliasing it
LatentMLP = LatentCNN
