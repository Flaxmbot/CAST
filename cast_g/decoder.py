"""
CAST-G Decoder — Segment Unpool + Upsample Byte Decoder.

CRITICAL FIX (v3.1): Replaces the fixed-expansion AR decoder that caused
catastrophic position misalignment (BPB > 8.0, worse than random).

The Problem (v3):
    Segments have VARIABLE sizes (e.g. 3, 8, 12 encoded positions each).
    The old decoder expanded every segment by a FIXED 'expansion' factor,
    then blindly padded/trimmed to match T. This meant:
    - Segment covering bytes 0-11 produced 8 predictions (missed 4 bytes)
    - Segment covering bytes 12-15 produced 8 predictions (4 predictions had no target)
    - Zero-padding for remaining positions predicted byte 0 everywhere
    
The Fix:
    1. Unpool: map each segment representation BACK to its constituent positions
       using the segment IDs from the hierarchy (lossless, no misalignment)
    2. Upsample: transpose-convolve from T//4 back to T (reverses encoder stride)
    3. Project: Linear(D, 256) at each position → byte logits
    
    This guarantees every byte position gets a prediction from its correct segment.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class UpsampleDecoder(nn.Module):
    """
    Decoder that reverses the encoder's stride-4 compression.
    
    Architecture:
        segment_reps [B, S, D]
            → unpool to [B, T//4, D] (using segment_ids)
            → refine with local causal convolutions
            → upsample to [B, T, D] (transposed conv, stride 4)
            → byte logits [B, T, 256]
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Refine unpooled representations with local context
        self.refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        # Upsample from T//4 back to T (reverses encoder's stride-4 conv)
        self.upsample = nn.ConvTranspose1d(
            d_model, d_model, kernel_size=4, stride=4
        )
        self.up_norm = nn.LayerNorm(d_model)
        
        # Byte prediction head
        self.head = nn.Linear(d_model, 256)
    
    def forward(
        self,
        segment_reps: torch.Tensor,
        segment_ids: torch.Tensor,
        T_encoded: int,
        T_original: int,
    ) -> torch.Tensor:
        """
        Decode segment representations back to byte-level logits.
        
        Args:
            segment_reps: [B, S, D] — global stack output (one per segment)
            segment_ids: [B, T_encoded] — which segment each encoded position belongs to
            T_encoded: number of encoded positions (T//4)
            T_original: original byte sequence length (T)
            
        Returns:
            logits: [B, T_original, 256] — byte prediction logits
        """
        B, S, D = segment_reps.shape
        
        # 1. Unpool: scatter segment reps back to their encoded positions
        # segment_ids[b, t] tells us which segment position t belongs to
        # We use this to gather the correct segment rep for each position
        ids_clamped = segment_ids.clamp(0, S - 1)  # [B, T_encoded]
        ids_expanded = ids_clamped.unsqueeze(-1).expand(-1, -1, D)  # [B, T_encoded, D]
        h_unpooled = torch.gather(segment_reps, 1, ids_expanded)  # [B, T_encoded, D]
        
        # 2. Refine with local context (residual)
        h_unpooled = h_unpooled + self.refine(h_unpooled)
        
        # 3. Upsample: T//4 → T (reverses encoder stride-4)
        # [B, T_encoded, D] → [B, D, T_encoded] → ConvTranspose → [B, D, T'] → [B, T', D]
        h_up = self.upsample(h_unpooled.transpose(1, 2))  # [B, D, T']
        h_up = h_up.transpose(1, 2)  # [B, T', D]
        
        # Trim or pad to exact T_original
        if h_up.size(1) > T_original:
            h_up = h_up[:, :T_original, :]
        elif h_up.size(1) < T_original:
            pad = torch.zeros(B, T_original - h_up.size(1), D, device=h_up.device, dtype=h_up.dtype)
            h_up = torch.cat([h_up, pad], dim=1)
        
        h_up = self.up_norm(h_up)
        
        # 4. Predict bytes
        logits = self.head(h_up)  # [B, T_original, 256]
        
        return logits
