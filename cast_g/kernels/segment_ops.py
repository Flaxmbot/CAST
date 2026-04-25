"""
High-Performance Segment Operations for Jagged Tensors.

Provides vectorized segment pooling and unpooling using PyTorch's
scatter_reduce (2.0+). No Triton required — torch.compile fuses
these into efficient GPU kernels on T4/A100.
"""
import torch
from typing import Tuple, Optional


def segment_pool(
    h_bytes: torch.Tensor,
    segment_ids: torch.Tensor,
    max_segments: Optional[int] = None,
    mode: str = 'mean'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pool byte-level embeddings into segment-level representations.
    
    Args:
        h_bytes: [B, T, D] — byte-level hidden states
        segment_ids: [B, T] — integer segment ID for each byte (from cumsum of boundaries)
        max_segments: Maximum number of segments (for consistent tensor shapes across GPUs).
                      If None, uses the actual max.
        mode: 'mean' or 'sum'
        
    Returns:
        pooled: [B, S, D] — segment-level representations
        counts: [B, S] — number of bytes per segment
        n_segments: [B] — actual number of segments per batch item
    """
    B, T, D = h_bytes.shape
    device = h_bytes.device
    dtype = h_bytes.dtype
    
    # Determine actual segment counts per batch item
    n_segments = segment_ids.max(dim=1).values + 1  # [B]
    
    if max_segments is None:
        # Use a FIXED cap based on sequence length to ensure consistent shapes
        # across GPUs in DataParallel. Each GPU gets same output shape.
        # Cap at T//2 (worst case: boundary every other position)
        max_segments = min(int(n_segments.max().detach().cpu().item()), T // 2)
    max_segments = max(1, int(max_segments))
    
    # Clamp segment_ids to valid range
    segment_ids = segment_ids.clamp(0, max_segments - 1)
    
    # Allocate output tensors with consistent shape
    pooled = torch.zeros(B, max_segments, D, device=device, dtype=dtype)
    counts = torch.zeros(B, max_segments, device=device, dtype=dtype)
    
    # Expand segment_ids for scatter
    idx_expanded = segment_ids.unsqueeze(-1).expand(-1, -1, D).long()
    
    # Scatter-add the byte embeddings into their segments
    pooled.scatter_add_(1, idx_expanded, h_bytes.to(dtype))
    
    # Count bytes per segment
    ones = torch.ones(B, T, device=device, dtype=dtype)
    counts.scatter_add_(1, segment_ids.long(), ones)
    
    if mode == 'mean':
        # Avoid division by zero for empty segments
        safe_counts = counts.unsqueeze(-1).clamp(min=1.0)
        pooled = pooled / safe_counts
    
    return pooled, counts, n_segments


def segment_unpool(
    segment_features: torch.Tensor,
    segment_ids: torch.Tensor,
    T_original: int
) -> torch.Tensor:
    """
    Broadcast segment-level features back to byte-level positions.
    
    Args:
        segment_features: [B, S, D] — segment representations
        segment_ids: [B, T] — segment ID per byte position
        T_original: original sequence length
        
    Returns:
        [B, T_original, D] — byte-level features from their parent segments
    """
    B, S, D = segment_features.shape
    
    # Gather: each byte gets its segment's representation
    idx = segment_ids[:, :T_original].unsqueeze(-1).expand(-1, -1, D).long()
    # Clamp to valid range
    idx = idx.clamp(0, S - 1)
    return torch.gather(segment_features, 1, idx)


def boundaries_to_segment_ids(boundaries: torch.Tensor) -> torch.Tensor:
    """
    Convert a binary boundary mask to segment IDs via cumulative sum.
    
    Args:
        boundaries: [B, T] — binary mask where 1 = start of new segment
        
    Returns:
        segment_ids: [B, T] — integer segment ID per position
    """
    # cumsum gives 1-indexed when boundaries[0]=1, so subtract 1 for 0-indexed
    ids = torch.cumsum(boundaries, dim=1).long() - 1
    return ids.clamp(min=0)
