import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityDrivenVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,lambda_loss = 0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.lambda_loss = lambda_loss
        
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', embedding / embedding.norm(dim=1, keepdim=True))
        
        # store unnormalized embeddings for loss computation
        # This allows MSE loss to have meaningful gradients
        self.register_buffer('embedding_unnormalized', embedding.clone())
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('ema_w_unnormalized', torch.randn(num_embeddings, embedding_dim))
        
        # Flag to track if we have initialized with real data yet
        self.init = False 

    def init_codebook(self, flat_input, flat_input_unnormalized):
        """Initialize codebook using random selection from the first batch of data"""
        print("Initializing codebook with K-means++ strategy (random data selection)...")
        indices = torch.randperm(flat_input.size(0))[:self.num_embeddings]
        
        if len(indices) < self.num_embeddings:
            indices = indices.repeat((self.num_embeddings // len(indices)) + 1)[:self.num_embeddings]
            
        initial_codes = flat_input[indices]
        initial_codes_unnormalized = flat_input_unnormalized[indices]
        self.embedding.data.copy_(F.normalize(initial_codes, p=2, dim=1))
        self.embedding_unnormalized.data.copy_(initial_codes_unnormalized)
        self.ema_w.data.copy_(self.embedding.data * self.decay)  # Sync EMA
        self.ema_w_unnormalized.data.copy_(self.embedding_unnormalized.data * self.decay)
        self.init = True

    def forward(self, inputs):
        # ... [Previous pre-processing code] ...????
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Store unnormalized input for loss computation
        flat_input_unnormalized = flat_input.clone()
        
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        
        if self.training and not self.init:
            self.init_codebook(flat_input_norm, flat_input_unnormalized)
            
        distances = torch.matmul(flat_input_norm, self.embedding.t())  #Z_e(x) * e_j / (||z_e(x)|| * ||e_j||) but we already normalized both, so it's just dot product
        
        # q(z|x)
        encoding_indices = torch.argmax(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        # z_q(x) = e_j where j = argmax sim(z_e(x), e_j)
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view(input_shape)
        
        # EMA Update with Dead Code Revival
        if self.training:
            cluster_size = encodings.sum(0)
            updated_ema_w = torch.matmul(encodings.t(), flat_input_norm)
            updated_ema_w_unnormalized = torch.matmul(encodings.t(), flat_input_unnormalized)
            
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(updated_ema_w, alpha=1 - self.decay)
            self.ema_w_unnormalized.data.mul_(self.decay).add_(updated_ema_w_unnormalized, alpha=1 - self.decay)
            
            # code reset
            # Identify codes that have very low usage (< 1.0 means used less than once on average)
            # We re-initialize them to random encoder outputs from the current batch
            dead_codes = self.ema_cluster_size < 1.0
            if dead_codes.any():
                # Pick random inputs to replace dead codes
                num_dead = dead_codes.sum()
                # Select random inputs
                rand_indices = torch.randint(0, flat_input_norm.size(0), (num_dead,))
                new_codes = flat_input_norm[rand_indices]
                new_codes_unnormalized = flat_input_unnormalized[rand_indices]
                
                # Reset weights and cluster size for these codes
                self.embedding.data[dead_codes] = new_codes
                self.embedding_unnormalized.data[dead_codes] = new_codes_unnormalized
                self.ema_w.data[dead_codes] = new_codes
                self.ema_w_unnormalized.data[dead_codes] = new_codes_unnormalized
                self.ema_cluster_size.data[dead_codes] = 1.0  # Give them a 'fresh start' count
            
            # Normal EMA normalization
            n = self.ema_cluster_size.sum()
            cluster_size_smoothed = (
                (self.ema_cluster_size + self.epsilon) / 
                (n + self.num_embeddings * self.epsilon) * n
            )
            normalised_ema_w = self.ema_w / cluster_size_smoothed.unsqueeze(1)
            self.embedding.data.copy_(F.normalize(normalised_ema_w, p=2, dim=1))
            # Keep unnormalized version in sync for loss computation
            self.embedding_unnormalized.data.copy_(self.ema_w_unnormalized / cluster_size_smoothed.unsqueeze(1))

        # commitment loss on unnormalized vectors for meaningful gradients
        # This loss penalizes the encoder for not committing to the quantized codes
        # Equation: ||z_e(x) - sg[e]||^2 where sg = stop gradient
        commitment_loss = F.mse_loss(flat_input_unnormalized, self.embedding_unnormalized[encoding_indices].detach())
        
        # codebook loss
        # Equation: ||sg[z_e(x)] - e||^2
        codebook_loss = F.mse_loss(self.embedding_unnormalized[encoding_indices], flat_input_unnormalized.detach())
        

        quantized = inputs + (quantized - inputs).detach()
        return quantized.permute(0, 2, 1), commitment_loss, codebook_loss, encoding_indices
    


class ResNetBlock1D(nn.Module):
    """
    ResNet block as described in Table 9:
    Input -> Conv1D -> ReLU -> Conv1D -> Add Input -> ReLU
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        # "input channel = D, output channel = D, kernel size = 3, stride = 1, padding = 1"
        # The paper mentions "dilation growth rate", implying dilation increases.
        # We use 'same' padding logic for dilated convs.
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=1,
                               padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1,
                               padding=padding, dilation=dilation)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + residual)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Architecture as Table 9 of SDformer paper. 

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)

        self.layer3 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3), # Depth=3 dilation growth? interpreted as blocks
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        self.layer4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)

        self.layer5 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3),
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        self.layer6 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Architecture per Table 10

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3),
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        self.layer3_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer3_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.layer4 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3),
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        self.layer5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer5_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        self.layer6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3_conv(self.layer3_up(x))
        x = self.layer4(x)
        x = self.layer5_conv(self.layer5_up(x))
        x = self.layer6(x)
        return x
    


class SDformerVQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            input_dim=config['input_dim'], 
            hidden_dim=config['hidden_dim'],
            output_dim=config['code_dim']
        )
        
        self.decoder = Decoder(
            input_dim=config['code_dim'], 
            hidden_dim=config['hidden_dim'],
            output_dim=config['input_dim']
        )

        self.quantizer = SimilarityDrivenVectorQuantizer(
            num_embeddings=config['codebook_size'],
            embedding_dim=config['code_dim'],
            lambda_loss=config['lambda_loss'],
            decay=config['decay']
        )

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, indices = self.quantizer(z)
        x_recon = self.decoder(z_quantized)
        return x_recon, commitment_loss, codebook_loss, indices