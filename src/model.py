import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, dict_size: int, k: int, device: str = "cuda"):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k
        self.device = device

        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer('usage_count', torch.zeros(dict_size))
        self.to(device)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.orthogonal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            norm = self.decoder.weight.norm(dim=1, keepdim=True)
            self.decoder.weight.data /= norm

    def encode(self, x):
        x_centered = x - self.decoder_bias
        activations = F.relu(self.encoder(x_centered))
        top_k_values, top_k_indices = activations.topk(self.k, dim=-1, sorted=False)
        sparse_code = torch.zeros_like(activations)
        sparse_code.scatter_(1, top_k_indices, top_k_values)
        return sparse_code, top_k_values, top_k_indices

    def decode(self, sparse_code):
        return self.decoder(sparse_code) + self.decoder_bias

    def forward(self, x):
        sparse_code, _, _ = self.encode(x)
        reconstruction = self.decode(sparse_code)
        return reconstruction, sparse_code