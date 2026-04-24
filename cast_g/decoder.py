"""
CAST-G Decoder — Segment Unpool + Causal Local Decoder.

CRITICAL FIX (v3.2): Replaces the non-autoregressive UpsampleDecoder.
Language models MUST be autoregressive at the prediction level to capture
local dependencies (e.g., 'u' following 'q').

Architecture:
    1. Unpool: map segment representations [B, S, D] back to [B, T//4, D].
    2. Upsample: expand to [B, T, D] using ConvTranspose1d (context).
    3. Shifted Input: embed original bytes [B, T] and shift by 1.
    4. Local Decoder: 2-layer causal Transformer combining context + bytes.
    5. Head: Linear(D, 256) -> byte logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalLocalBlock(nn.Module):
    """Small causal block for the local decoder (SDPA-based)."""
    def __init__(self, d_model: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Causal self-attention
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class CausalLocalDecoder(nn.Module):
    """
    Autoregressive decoder that combines global segment context with
    local byte-level dependencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Byte embedding for the autoregressive input
        self.byte_embed = nn.Embedding(256, d_model)
        
        # Upsample context from T//4 to T
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=4, stride=4
        )
        
        # Local causal layers to fuse context and shifted bytes
        self.local_layers = nn.ModuleList([
            CausalLocalBlock(d_model, n_head=4, dropout=dropout)
            for _ in range(2)
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
            segment_reps: [B, S, D] — global context
            segment_ids: [B, T_encoded] — mapping
            T_encoded: T//4
            T_original: T
            idx: [B, T] — original byte indices (for autoregressive input)
        """
        B, S, D = segment_reps.shape
        
        # 1. Unpool segment reps back to encoded positions [B, T_encoded, D]
        ids_clamped = segment_ids.clamp(0, S - 1)
        ids_expanded = ids_clamped.unsqueeze(-1).expand(-1, -1, D)
        h_unpooled = torch.gather(segment_reps, 1, ids_expanded)
        
        # 2. Upsample context to byte level [B, T_original, D]
        # [B, T_encoded, D] -> [B, D, T_encoded] -> [B, D, T] -> [B, T, D]
        h_ctx = self.upsample(h_unpooled.transpose(1, 2)).transpose(1, 2)
        h_ctx = h_ctx[:, :T_original, :]
        
        # 3. Embed and Shift Input Bytes (Autoregressive)
        # To predict byte t, we must only see bytes 0...t-1.
        # We prepend a zero/start byte and shift.
        # [B, T] -> [B, T, D]
        h_bytes = self.byte_embed(idx)
        h_bytes_shifted = torch.cat([torch.zeros(B, 1, D, device=idx.device), h_bytes[:, :-1, :]], dim=1)
        
        # 4. Combine Context + Bytes
        # The global context 'h_ctx' provides the "theme" for the segment,
        # while 'h_bytes_shifted' provides the local history.
        x = h_ctx + h_bytes_shifted
        
        # 5. Local Causal Refinement
        for layer in self.local_layers:
            x = layer(x)
        
        # 6. Predict next bytes
        logits = self.head(self.final_norm(x))
        
        return logits
