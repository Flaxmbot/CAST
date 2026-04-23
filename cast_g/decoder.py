import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class RVQRefiner(nn.Module):
    """
    Residual Vector Quantization (RVQ).
    Snaps the continuous latent vector to a discrete codebook to prevent drift.
    Includes multiple levels of refinement for high fidelity.
    """
    def __init__(self, d_model: int, codebook_size: int = 1024, num_levels: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_levels = num_levels
        # Multiple codebooks for residual quantization
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(num_levels)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, D]
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices = []
        
        for codebook in self.codebooks:
            # Calculate distances to codebook entries
            dist = torch.cdist(residual, codebook.weight[None, ...]) # [B, S, K]
            indices = dist.argmin(dim=-1) # [B, S]
            all_indices.append(indices)
            
            # Lookup and add to output
            q = codebook(indices)
            quantized_out = quantized_out + q
            residual = residual - q # Update residual for next level
            
        # Straight-through estimator for gradient flow
        quantized_out = x + (quantized_out - x).detach()
        return quantized_out, torch.stack(all_indices, dim=-1)

class ByteDecoder(nn.Module):
    """
    Generates raw bytes from the 'snapped' latent segments.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 256, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, D]
        return self.proj(x) # Logits for 256 bytes
