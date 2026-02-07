import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityDrivenVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, lambda_loss=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon
        self.lambda_loss = lambda_loss
        
        # Initialize embeddings and normalize them to unit modulus (Eq 3)
        embedding = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer('embedding', F.normalize(embedding, p=2, dim=1))
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embedding.clone())
        
        self.init = False 

    def init_codebook(self, flat_input_norm):
        """Initialize codebook using random selection from the first batch"""
        indices = torch.randperm(flat_input_norm.size(0))[:self.num_embeddings]
        if len(indices) < self.num_embeddings:
            indices = indices.repeat((self.num_embeddings // len(indices)) + 1)[:self.num_embeddings]
            
        initial_codes = flat_input_norm[indices]
        self.embedding.data.copy_(initial_codes)
        self.ema_w.data.copy_(initial_codes)
        self.init = True

    def forward(self, inputs):
        # inputs shape: [B, D, L] -> [B*L, D]
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        L = input_shape[0] * input_shape[1] # Total latent points across batch
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Normalize encoder output to unit modulus: h_i = z_e(x) / ||z_e(x)||
        h_i = F.normalize(flat_input, p=2, dim=1)
        
        if self.training and not self.init:
            self.init_codebook(h_i)
            
        # Similarity Calculation: h_i · e_j
        # Since both are unit vectors, this is cosine similarity
        distances = torch.matmul(h_i, self.embedding.t())
        
        # Quantization: Select index with maximum similarity
        encoding_indices = torch.argmax(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # h~_i: The selected codebook vectors
        h_tilde_i = torch.matmul(encodings, self.embedding)
        
        # EMA Update Logic
        if self.training:
            cluster_size = encodings.sum(0)
            updated_ema_w = torch.matmul(encodings.t(), h_i)
            
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(updated_ema_w, alpha=1 - self.decay)
            
            # Dead code revival
            dead_codes = self.ema_cluster_size < 1.0
            if dead_codes.any():
                num_dead = dead_codes.sum()
                rand_indices = torch.randint(0, h_i.size(0), (num_dead,))
                self.embedding.data[dead_codes] = h_i[rand_indices]
                self.ema_w.data[dead_codes] = h_i[rand_indices]
                self.ema_cluster_size.data[dead_codes] = 1.0
            
            # Update codebook embeddings (unit modulus)
            n = self.ema_cluster_size.sum()
            cluster_size_smoothed = ((self.ema_cluster_size + self.epsilon) / 
                                     (n + self.num_embeddings * self.epsilon) * n)
            normalized_ema_w = self.ema_w / cluster_size_smoothed.unsqueeze(1)
            self.embedding.data.copy_(F.normalize(normalized_ema_w, p=2, dim=1))

        # --- SD-FORMER SIMILARITY LOSS ---
        # Equation: lambda/L * sum(1 - h_i · sg(h~_i))
        # .detach() acts as the stop-gradient (sg) on the codebook
        dot_product = torch.sum(h_i * h_tilde_i.detach(), dim=1)
        commitment_loss = (self.lambda_loss / L) * torch.sum(1.0 - dot_product)
        
        # In EMA-based VQ, the codebook doesn't use a backprop loss (it uses the EMA update)
        codebook_loss = torch.tensor(0.0).to(inputs.device)

        # Straight-through estimator
        quantized = inputs + (h_tilde_i.view(input_shape) - inputs).detach()
        
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
        # Architecture per Table 9

        # Layer 1: Conv1D (in=d, out=D)
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Layer 2: Downsample 1
        self.layer2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)

        # Layer 3: ResNet
        self.layer3 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3), # Depth=3 dilation growth? interpreted as blocks
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        # Layer 4: Downsample 2
        self.layer4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)

        # Layer 5: ResNet (Same as layer 3)
        self.layer5 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3),
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        # Layer 6: Final projection (in=D, out=output_dim/code_dim)
        # Allows hidden_dim (e.g., 512) to be different from code_dim (e.g., 64)
        self.layer6 = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)

        # Normalization output layer (Eq 3 mentions h_i is unit modulus)
        # We handle this inside the VQ module, but the encoder outputs raw logits/vectors

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

        # Layer 1: Conv1D (in=input_dim/code_dim, out=D)
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Layer 2: ResNet
        self.layer2 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3),
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        # Layer 3: Upsample 1 (Upsample + Conv)
        self.layer3_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer3_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        # Layer 4: ResNet
        self.layer4 = nn.Sequential(
            ResNetBlock1D(hidden_dim, dilation=1),
            ResNetBlock1D(hidden_dim, dilation=3),
            ResNetBlock1D(hidden_dim, dilation=9),
            nn.ReLU()
        )

        # Layer 5: Upsample 2
        self.layer5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer5_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        # Layer 6: Final Refinement
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
        # Pass explicit dimensions to support mismatched hidden/code dims
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
        # 1. Encode
        z = self.encoder(x)

        # 2. Quantize (Similarity Driven)
        z_quantized, commitment_loss, codebook_loss, indices = self.quantizer(z)

        # 3. Decode
        x_recon = self.decoder(z_quantized)

        return x_recon, commitment_loss, codebook_loss, indices