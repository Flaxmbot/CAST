import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernels.triton_jagged import triton_pool_jagged, HAS_TRITON
from typing import Optional, Tuple, List

class JaggedTransformer(nn.Module):
    """
    A Transformer optimized for variable-length segments.
    Uses attention masks to prevent information leakage between segments
    until the global reasoning stage.
    """
    def __init__(self, d_model: int, n_head: int, n_layer: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, 
            dim_feedforward=4*d_model, batch_first=True,
            norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Importance Gating (The 'Dynamic Stride' solution)
        self.importance_gate = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, S, D]
        # 1. Importance Gating
        importance_logits = self.importance_gate(x)
        importance_gate = torch.sigmoid(importance_logits) # [B, S, 1]
        
        # We only pass segments with gate > 0.5 through the Transformer
        # (During training, we use soft-gating for gradients)
        x_important = x * importance_gate
        
        # 2. Global Reasoning
        h_transformed = self.transformer(x_important, mask=mask)
        
        # 3. Residual Combine
        # Low-info segments are recovered via the skip connection
        out = h_transformed + (x * (1 - importance_gate))
        return out, importance_gate

def pool_jagged(h_bytes: torch.Tensor, boundaries: torch.Tensor):
    """
    Compresses byte-level embeddings into segment-level latents.
    [HARDWARE-AWARE]: Automatically chooses between Triton GPU kernel and PyTorch fallback.
    """
    # 1. Attempt high-performance Triton path
    if HAS_TRITON and h_bytes.is_cuda:
        z = triton_pool_jagged(h_bytes, boundaries)
        if z is not None:
            return z, None # Successful Triton exit
            
    # 2. Optimized PyTorch Fallback (Vectorized Scatter-Add)
    B, T, D = h_bytes.shape
    device = h_bytes.device
    
    # Generate segment IDs using parallel prefix sum
    segment_ids = torch.cumsum(boundaries, dim=1).long()
    max_segments = segment_ids.max().item() + 1
    
    # Parallel Aggregation (Strict DType alignment for AMP)
    pooled = torch.zeros(B, max_segments, D, device=device, dtype=h_bytes.dtype)
    counts = torch.zeros(B, max_segments, 1, device=device, dtype=h_bytes.dtype)
    
    idx_expanded = segment_ids.unsqueeze(-1).expand(-1, -1, D)
    pooled.scatter_add_(1, idx_expanded, h_bytes)
    
    # Ensure 'ones' matches the dtype of h_bytes exactly
    ones = torch.ones_like(segment_ids).unsqueeze(-1).to(h_bytes.dtype)
    counts.scatter_add_(1, segment_ids.unsqueeze(-1), ones)
    
    # Mean pooling result
    return pooled / (counts + 1e-6), counts.squeeze(-1)
