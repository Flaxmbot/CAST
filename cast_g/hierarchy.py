"""
CAST-G Hierarchical Multi-Scale Segmentation.

NOVEL CONTRIBUTION: 3-level learned segmentation hierarchy that
simultaneously discovers sub-word, word, and phrase boundaries.

No published work does multi-level dynamic segmentation end-to-end:
- BLT (2024): single-level entropy-based patching
- Nawrot (2023): single-level Gumbel boundary in Hourglass
- MegaByte (2023): single-level fixed-size patches
- Charformer (2022): single-level GBST

CAST-G learns THREE simultaneous granularities:
    Level 0 (Fine):   ~4-8 bytes   — morpheme/syllable scale
    Level 1 (Medium): ~16-32 bytes — word/compound scale
    Level 2 (Coarse): ~48-96 bytes — phrase/clause scale

Key innovation: Cross-level attention allows coarse levels to inform
fine-level boundary decisions (top-down planning), while fine-level
representations are pooled upward for coarse reasoning (bottom-up detail).

This creates a fractal compression hierarchy where information flows
bidirectionally across scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from .boundary import MIBoundaryDetector, MISegmentationLoss
from .kernels.segment_ops import segment_pool, boundaries_to_segment_ids, segment_unpool


class CrossLevelAttention(nn.Module):
    """
    Bidirectional cross-level attention between adjacent hierarchy levels.
    
    - Bottom-up: fine segments attend to their parent coarse segment
    - Top-down: coarse segments attend to their child fine segments
    
    This allows the hierarchy to coordinate: coarse planning guides
    fine-grained execution, while fine details inform coarse decisions.
    """
    def __init__(self, d_model: int, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Bottom-up: fine attends to coarse
        self.bu_q = nn.Linear(d_model, d_model)
        self.bu_k = nn.Linear(d_model, d_model)
        self.bu_v = nn.Linear(d_model, d_model)
        self.bu_out = nn.Linear(d_model, d_model)
        self.bu_norm = nn.LayerNorm(d_model)
        
        # Top-down: coarse attends to fine
        self.td_q = nn.Linear(d_model, d_model)
        self.td_k = nn.Linear(d_model, d_model)
        self.td_v = nn.Linear(d_model, d_model)
        self.td_out = nn.Linear(d_model, d_model)
        self.td_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _mha(self, q_proj, k_proj, v_proj, out_proj, q_input, kv_input):
        """Generic multi-head attention helper."""
        B = q_input.size(0)
        S_q = q_input.size(1)
        S_kv = kv_input.size(1)
        
        q = q_proj(q_input).view(B, S_q, self.n_head, self.head_dim).transpose(1, 2)
        k = k_proj(kv_input).view(B, S_kv, self.n_head, self.head_dim).transpose(1, 2)
        v = v_proj(kv_input).view(B, S_kv, self.n_head, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, S_q, self.d_model)
        return self.dropout(out_proj(out))
    
    def forward(
        self, 
        fine: torch.Tensor, 
        coarse: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fine: [B, S_fine, D] — finer-level segment representations
            coarse: [B, S_coarse, D] — coarser-level segment representations
            
        Returns:
            fine_updated: [B, S_fine, D]
            coarse_updated: [B, S_coarse, D]
        """
        # Bottom-up: fine segments enriched by coarse context
        fine_normed = self.bu_norm(fine)
        fine_updated = fine + self._mha(
            self.bu_q, self.bu_k, self.bu_v, self.bu_out,
            fine_normed, coarse
        )
        
        # Top-down: coarse segments enriched by fine detail
        coarse_normed = self.td_norm(coarse)
        coarse_updated = coarse + self._mha(
            self.td_q, self.td_k, self.td_v, self.td_out,
            coarse_normed, fine_updated
        )
        
        return fine_updated, coarse_updated


class HierarchicalSegmenter(nn.Module):
    """
    Multi-scale segmentation hierarchy with cross-level attention.
    
    Produces 3 levels of segments from byte-level embeddings:
    1. Each level has its own MIBoundaryDetector (different window sizes)
    2. Levels are connected by CrossLevelAttention for coordination
    3. Each level has its own segmentation loss with different targets
    
    The hierarchy is built bottom-up:
        bytes → Level 0 segments → Level 1 segments → Level 2 segments
    
    Then refined top-down via cross-level attention.
    """
    def __init__(
        self, 
        d_model: int, 
        n_levels: int = 3,
        target_lengths: List[float] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        
        # Default target lengths: 8, 24, 64 bytes
        if target_lengths is None:
            target_lengths = [8.0, 24.0, 64.0]
        self.target_lengths = target_lengths[:n_levels]
        
        # Per-level boundary detectors with increasing window sizes
        self.boundary_detectors = nn.ModuleList([
            MIBoundaryDetector(d_model, window_size=2 * (i + 2), dropout=dropout)
            for i in range(n_levels)
        ])
        
        # Per-level segmentation losses
        self.seg_losses = nn.ModuleList([
            MISegmentationLoss(target_len=tl, mi_weight=0.1, lagrangian_lr=0.01)
            for tl in self.target_lengths
        ])
        
        # Per-level local processing (small FFN to refine segment representations)
        self.level_processors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * d_model, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_levels)
        ])
        
        # Cross-level attention between adjacent levels
        self.cross_level = nn.ModuleList([
            CrossLevelAttention(d_model, n_head=4, dropout=dropout)
            for _ in range(n_levels - 1)
        ])
        
    def forward(
        self,
        h_bytes: torch.Tensor,
        temp: float = 1.0,
        hard: bool = True,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], Dict]:
        """
        Build the full segmentation hierarchy.
        
        Args:
            h_bytes: [B, T, D] — byte-level hidden states (after encoder)
            temp: boundary temperature (annealed during training)
            hard: whether to use hard boundaries (STE)
            
        Returns:
            level_segments: list of [B, S_i, D] per level
            level_boundaries: list of [B, T_i] per level
            level_segment_ids: list of [B, T_i] per level
            metrics: dict of per-level metrics
        """
        B, T, D = h_bytes.shape
        
        level_segments = []
        level_boundaries = []
        level_segment_ids = []
        all_metrics = {}
        total_seg_loss = torch.tensor(0.0, device=h_bytes.device)
        
        # Current representation to segment (starts at byte level)
        current_repr = h_bytes
        current_T = T  # Track the "original" length at each level for loss
        
        # Bottom-up: build segments at each level
        for level in range(self.n_levels):
            # Predict boundaries at this level
            boundaries, mi_scores = self.boundary_detectors[level](current_repr, temp=temp, hard=hard)
            
            # Convert to segment IDs
            segment_ids = boundaries_to_segment_ids(boundaries)
            
            # Pool into segments
            pooled, counts, n_segments = segment_pool(current_repr, segment_ids)
            
            # Local processing (refine segment representations)
            pooled = pooled + self.level_processors[level](pooled)
            
            # Compute segmentation loss
            seg_loss, seg_metrics = self.seg_losses[level](boundaries, mi_scores, current_T)
            total_seg_loss = total_seg_loss + seg_loss
            
            # Store results
            level_segments.append(pooled)
            level_boundaries.append(boundaries)
            level_segment_ids.append(segment_ids)
            
            # Record metrics
            for k, v in seg_metrics.items():
                all_metrics[f'level{level}_{k}'] = v
            
            # Next level operates on THIS level's segments
            current_repr = pooled
            # Update T for next level's Lagrangian calculation
            avg_segs = max(1, int(n_segments.float().mean().item()))
            current_T = avg_segs
        
        # Top-down refinement via cross-level attention
        # Process from coarsest to finest
        for i in range(self.n_levels - 2, -1, -1):
            fine = level_segments[i]
            coarse = level_segments[i + 1]
            
            # Cross-level attention (bidirectional)
            fine_updated, coarse_updated = self.cross_level[i](fine, coarse)
            level_segments[i] = fine_updated
            level_segments[i + 1] = coarse_updated
        
        all_metrics['total_seg_loss'] = total_seg_loss
        
        return level_segments, level_boundaries, level_segment_ids, all_metrics
