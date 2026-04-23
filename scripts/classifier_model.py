import torch.nn as nn

class LatentMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], num_classes=17):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_classes)
        )

    def forward(self, x):
        # x is [Batch, seq_len, code_dim] -> we pool over time
        x = x.mean(dim=1)
        return self.network(x)
