import torch
import torch.nn as nn
import torch.nn.functional as F



class SimilarityDrivenVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        # Initialize embeddings
        embedding = torch.randn(num_embeddings, embedding_dim)
        # We assume unit norm for the codebook as per Eq (3) and (4) implication [cite: 118, 119]
        self.register_buffer('embedding', embedding / embedding.norm(dim=1, keepdim=True))

        # EMA Buffers (for training stability) [cite: 35]
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))

    def forward(self, inputs):
        # inputs shape: [Batch, Channels, Length]
        # Channels corresponds to 'd_c' (embedding_dim) in the paper [cite: 110]
        input_shape = inputs.shape
        # print("input shape is:", inputs.shape)
        # 1. Permute to move embedding dim to the end
        inputs = inputs.permute(0, 2, 1).contiguous()
        assert inputs.shape == (input_shape[0], input_shape[2], input_shape[1]), \
            f"Expected permuted shape {(input_shape[0], input_shape[2], input_shape[1])}, got {inputs.shape}"
        # print("input shape after permutation is:", inputs.shape)

        # 2. Flatten input: [Batch * Length, Embedding_Dim]
        flat_input = inputs.view(-1, self.embedding_dim)
        assert flat_input.shape == (input_shape[0] * input_shape[2], self.embedding_dim), \
            f"Expected flat shape {(input_shape[0] * input_shape[2], self.embedding_dim)}, got {flat_input.shape}"
        # print("input shape after flattening  is:", inputs.shape)

        # --- Normalization (Eq 3) ---
        # "result in a unit modulus length for h_i" [cite: 118]
        flat_input_norm = F.normalize(flat_input, p=2, dim=1)
        assert flat_input_norm.shape == flat_input.shape

        # --- Similarity Calculation (Eq 4) ---
        # "arg max h_i . c_k" (Dot product of normalized vectors) [cite: 120]
        # (B*L, D) @ (K, D).T -> (B*L, K)
        distances = torch.matmul(flat_input_norm, self.embedding.t())
        assert distances.shape == (flat_input.shape[0], self.num_embeddings), \
            f"Expected distances shape {(flat_input.shape[0], self.num_embeddings)}, got {distances.shape}"

        # --- Quantization ---
        # Find index of max similarity [cite: 115]
        encoding_indices = torch.argmax(distances, dim=1)
        assert encoding_indices.shape == (flat_input.shape[0],), \
            f"Expected indices shape {(flat_input.shape[0],)}, got {encoding_indices.shape}"
        # print("encoding indices: ",encoding_indices)
        # Create one-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        assert encodings.shape == (flat_input.shape[0], self.num_embeddings)
        # print("encoding shape:",encodings.shape)
        # Quantize: Select the codebook vectors
        quantized = torch.matmul(encodings, self.embedding)
        assert quantized.shape == flat_input.shape
        # print("quantized shape:",quantized.shape)

        # Reshape back to original dimensions [Batch, Length, Channels]
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[1])

        # --- EMA Update (Training Only) ---
        # Updating codebook without backprop
        if self.training:
            # Usage of each code in this batch
            cluster_size = encodings.sum(0)
            # Sum of input vectors assigned to each code
            updated_ema_w = torch.matmul(encodings.t(), flat_input_norm)

            # Update buffers
            self.ema_cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.ema_w.data.mul_(self.decay).add_(updated_ema_w, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            cluster_size_smoothed = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_embeddings * self.epsilon) * n
            )

            # Normalize updated codebook [cite: 119]
            normalised_ema_w = self.ema_w / cluster_size_smoothed.unsqueeze(1)
            self.embedding.data.copy_(F.normalize(normalised_ema_w, p=2, dim=1))

        # --- Commitment Loss (Eq 6) ---
        # Loss = 1 - similarity [cite: 129]
        # Since vectors are normalized, similarity is just the dot product.
        # We detach quantized to stop gradient flow to the codebook (it's updated via EMA)
        similarity = (flat_input_norm * quantized.view(-1, self.embedding_dim).detach()).sum(dim=1)
        commitment_loss = (1 - similarity).mean()

        # Straight Through Estimator [cite: 131]
        # Pass gradients from 'quantized' directly to 'inputs'
        quantized = inputs + (quantized - inputs).detach()

        # Permute back to [Batch, Channels, Length]
        result = quantized.permute(0, 2, 1)
        assert result.shape == input_shape, f"Final shape mismatch. Expected {input_shape}, got {result.shape}"

        return result, commitment_loss, encoding_indices
    




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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Architecture per Table 9 [cite: 639]

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

        # Layer 6: Final projection (in=D, out=H)
        # In our config D=H=512, so this keeps dimensions same but mixes features
        self.layer6 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

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
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Architecture per Table 10 [cite: 646]

        # Layer 1: Conv1D
        self.layer1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
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
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)
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
        self.encoder = Encoder(config['input_dim'], config['hidden_dim'])
        self.decoder = Decoder(config['input_dim'], config['hidden_dim'])

        self.quantizer = SimilarityDrivenVectorQuantizer(
            num_embeddings=config['codebook_size'],
            embedding_dim=config['code_dim'],
            decay=config['decay']
        )

    def forward(self, x):
        # 1. Encode
        z = self.encoder(x)

        # 2. Quantize (Similarity Driven)
        z_quantized, loss_embed, indices = self.quantizer(z)

        # 3. Decode
        x_recon = self.decoder(z_quantized)

        return x_recon, loss_embed, indices

# Initialize Model
