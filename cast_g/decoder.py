"""
CAST-G Decoder — Efficient Segment-Aware Local Decoder.

ARCHITECTURAL FIX (v3.3): Replaces O(T²) full-sequence attention decoder
with O(T) causal convolution decoder. The old decoder used 2 layers of
MultiheadAttention over ALL T bytes, which negated segmentation efficiency.

The new decoder uses:
    1. Unpool: map segment representations [B, S, D] back to [B, T//4, D]
    2. Upsample: expand to [B, T, D] using ConvTranspose1d
    3. Shifted Input: embed original bytes and shift by 1 (autoregressive)
    4. Local Causal Convolution: O(T) processing with limited receptive field
    5. Head: Linear(D, 256) -> byte logits

Key insight: bytes within a segment only need LOCAL context (previous few bytes)
plus the GLOBAL segment context (from the Transformer stack). Full-sequence
attention is wasteful because the global stack already captured long-range deps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalConvBlock(nn.Module):
    """
    Efficient causal convolution block — O(T) instead of O(T²).
    
    Uses depthwise-separable causal convolutions for efficiency:
    - Depthwise conv captures local byte patterns (bigrams, trigrams)
    - Pointwise conv (FFN) mixes features across dimensions
    - Gated activation (SiLU) for better gradient flow
    
    Total receptive field: kernel_size bytes of local context.
    """
    def __init__(self, d_model: int, kernel_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Depthwise causal convolution (groups=d_model for efficiency)
        self.dwconv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            groups=d_model, bias=True
        )
        
        # Pointwise expansion with gated activation (SwiGLU-style)
        self.gate_proj = nn.Linear(d_model, 2 * d_model)
        self.down_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        residual = x
        x = self.norm(x)
        
        # Causal depthwise conv: pad on left only
        # [B, T, D] -> [B, D, T] -> conv -> [B, D, T] -> [B, T, D]
        h = x.transpose(1, 2)
        h = self.dwconv(F.pad(h, (self.kernel_size - 1, 0)))
        h = h.transpose(1, 2)
        
        # Gated pointwise (SwiGLU)
        gate = self.gate_proj(h)
        h = gate[..., :gate.size(-1)//2] * F.silu(gate[..., gate.size(-1)//2:])
        h = self.down_proj(h)
        
        return residual + self.dropout(h)


class CausalLocalDecoder(nn.Module):
    """
    Efficient autoregressive decoder: O(T) instead of O(T²).
    
    Combines global segment context with local byte-level convolutions.
    The global Transformer stack handles long-range dependencies between
    segments; this decoder only needs to model LOCAL byte patterns within
    each segment's prediction window.
    
    Architecture:
        segment_reps[k-1] (shifted) -> unpool -> upsample to byte level
        + shifted byte embeddings (autoregressive input)
        -> 3 layers of CausalConvBlock (kernel=8, O(T))
        -> Linear(D, 256) byte logits
    """
    def __init__(self, d_model: int, n_layers: int = 3, kernel_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Byte embedding for the autoregressive input
        self.byte_embed = nn.Embedding(256, d_model)
        
        # Upsample context from T//4 to T (causal: each output only depends on its input position)
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=4, stride=4
        )
        
        # Projection to combine context + shifted bytes
        self.combine_proj = nn.Linear(2 * d_model, d_model)
        
        # Local causal conv layers — O(T) each
        self.local_layers = nn.ModuleList([
            CausalConvBlock(d_model, kernel_size=kernel_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)
    
    def forward(
        self,
        segment_reps: torch.Tensor,
        segment_ids: torch.Tensor,
        T_encoded: int,
        T_original: int,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            segment_reps: [B, S, D] — shifted global context (segment k-1 for position k)
            segment_ids: [B, T_encoded] — segment ID per encoded position
            T_encoded: T//4
            T_original: T
            idx: [B, T] — original byte indices (for autoregressive input)
        """
        B, S, D = segment_reps.shape
        
        # 1. Unpool segment reps to encoded positions [B, T_encoded, D]
        ids_clamped = segment_ids.clamp(0, S - 1)
        ids_expanded = ids_clamped.unsqueeze(-1).expand(-1, -1, D)
        h_unpooled = torch.gather(segment_reps, 1, ids_expanded)
        
        # 2. Upsample context to byte level [B, T_original, D]
        h_ctx = self.upsample(h_unpooled.transpose(1, 2)).transpose(1, 2)
        h_ctx = h_ctx[:, :T_original, :]
        
        # 3. Embed and Shift Input Bytes (Autoregressive)
        # To predict byte t, we see bytes 0..t-1 only
        h_bytes = self.byte_embed(idx)
        h_bytes_shifted = torch.cat([
            torch.zeros(B, 1, D, device=idx.device, dtype=h_bytes.dtype),
            h_bytes[:, :-1, :]
        ], dim=1)
        
        # 4. Combine context + shifted bytes via learned projection
        # (concatenation + projection preserves more information than addition)
        x = self.combine_proj(torch.cat([h_ctx, h_bytes_shifted], dim=-1))
        
        # 5. Local Causal Refinement — O(T) per layer
        for layer in self.local_layers:
            x = layer(x)
        
        # 6. Predict next bytes
        logits = self.head(self.final_norm(x))
        
        return logits
"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        => Now: __init__(self, d_model, n_layers=3, kernel_size=8, dropout=0.1)
        
    Old CausalLocalDecoder used nn.MultiheadAttention (O(T²)).
    New one uses CausalConvBlock (O(T)).
    Interface is IDENTICAL — drop-in replacement.
"""
