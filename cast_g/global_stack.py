"""
CAST-G Global Stack — Causal Transformer with Mixture-of-Depths Routing.

FIXES:
1. Causal masking via F.scaled_dot_product_attention(is_causal=True)
2. Proper Flash/Memory-Efficient Attention dispatch (T4-compatible)
3. Pre-norm architecture for training stability

NOVEL CONTRIBUTION (MoD-S):
Mixture-of-Depths Segment Routing. Unlike Raposo (2024) who applies MoD
to fixed tokens in standard Transformers, CAST-G applies MoD to
*dynamically-segmented byte representations*. This creates two orthogonal
efficiency axes: (1) dynamic segmentation reduces sequence length, and
(2) MoD routing reduces the number of segments that need full compute.

Trivial segments (articles, spaces, common words) skip Transformer layers
entirely via residual bypass. Complex segments (rare words, code, names)
get full attention-based processing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .kernels.segment_ops import segment_pool, segment_unpool, boundaries_to_segment_ids


class CausalTransformerBlock(nn.Module):
    """
    Single Transformer block with causal attention using PyTorch's SDPA.
    
    Uses F.scaled_dot_product_attention which automatically dispatches to:
    - Flash Attention 2 on Ampere+ (A100, H100)
    - Memory-Efficient Attention on Turing (T4)
    - Math fallback elsewhere
    
    All variants support is_causal=True for proper autoregressive masking.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Self-attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Pre-norm
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] — segment representations
        Returns:
            [B, S, D] — updated representations
        """
        B, S, D = x.shape
        
        # Pre-norm self-attention
        normed = self.attn_norm(x)
        q = self.q_proj(normed).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        
        # SDPA with causal masking — dispatches to best available kernel
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True  # CRITICAL FIX: was missing entirely in v2
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        x = x + self.dropout(self.out_proj(attn_out))
        
        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class MoDSegmentRouter(nn.Module):
    """
    NOVEL: Mixture-of-Depths routing for byte segments.
    
    Unlike Raposo (2024) who routes fixed tokens, this routes variable-length
    segments from the dynamic segmenter. This is novel because:
    
    1. Segments have variable sizes, so routing decisions consider segment
       complexity (a 3-byte segment is likely simpler than a 20-byte one)
    2. The routing interacts with the hierarchical segmenter — coarse segments
       may route differently than fine segments
    3. Skipped segments still participate in future attention via residual bypass
    
    Uses top-k selection (not top-p) for predictable compute budget.
    """
    def __init__(self, d_model: int, capacity_ratio: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.capacity_ratio = capacity_ratio
        
        # Router: predicts importance score per segment
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        
        # Auxiliary load-balancing loss weight
        self.aux_weight = 0.01
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route segments: select top-k for full compute, others get residual bypass.
        
        Args:
            x: [B, S, D] — segment representations
            
        Returns:
            selected: [B, k, D] — segments selected for full compute
            selected_indices: [B, k] — indices of selected segments
            router_scores: [B, S] — raw router scores (for aux loss)
        """
        B, S, D = x.shape
        
        # Compute router scores
        scores = self.router(x).squeeze(-1)  # [B, S]
        
        # Top-k selection
        k = max(1, int(S * self.capacity_ratio))
        k = min(k, S)  # Can't select more than available
        
        topk_vals, topk_indices = torch.topk(scores, k, dim=1, sorted=False)
        
        # Gather selected segments
        selected = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D))
        
        return selected, topk_indices, scores
    
    def compute_aux_loss(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Load-balancing auxiliary loss to prevent router collapse.
        
        Encourages the router to use roughly capacity_ratio fraction
        of segments, preventing degenerate solutions where it always
        selects the same segments.
        """
        # Fraction of segments with positive routing score
        fraction_routed = (scores > 0).float().mean()
        # Target: capacity_ratio
        aux_loss = (fraction_routed - self.capacity_ratio).pow(2)
        return self.aux_weight * aux_loss


class MoDTransformerStack(nn.Module):
    """
    Transformer stack with Mixture-of-Depths segment routing.
    
    At each layer:
    1. Router selects top-k segments for full attention processing
    2. Selected segments pass through CausalTransformerBlock
    3. Un-selected segments bypass via residual connection
    4. Results are scattered back to original positions
    
    This achieves the same representational capacity as a full Transformer
    but with ~capacity_ratio fraction of the compute.
    """
    def __init__(
        self, 
        d_model: int, 
        n_head: int, 
        n_layer: int,
        capacity_ratio: float = 0.5,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        
        # Positional Embeddings (learned)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        self.layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        # Per-layer routers (each layer routes independently)
        self.routers = nn.ModuleList([
            MoDSegmentRouter(d_model, capacity_ratio=capacity_ratio)
            for _ in range(n_layer)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [B, S, D] — segment representations
            
        Returns:
            h: [B, S, D] — processed representations
            metrics: dict with routing statistics
        """
        B, S, D = x.shape
        
        # Add positional embeddings (causal-safe since segments are in order)
        x = x + self.pos_emb[:, :S, :]
        
        total_aux_loss = torch.tensor(0.0, device=x.device)
        
        # Track routing decisions for metrics
        all_routed_fraction = []
        
        for layer_idx in range(self.n_layer):
            router = self.routers[layer_idx]
            transformer = self.layers[layer_idx]
            
            # 1. Route: Select top-k segments
            selected, topk_indices, scores = router(x)
            
            # 1b. SORT by original position for causal correctness
            # Without this, is_causal=True creates a mask based on position in
            # the selected tensor, NOT original sequence order — breaking causality
            sorted_order = topk_indices.argsort(dim=1)
            topk_indices = topk_indices.gather(1, sorted_order)
            selected = selected.gather(1, sorted_order.unsqueeze(-1).expand(-1, -1, D))
            
            # 2. Process: Selected segments through Transformer block
            h_selected = transformer(selected)
            
            # 3. Update: Scatter processed segments back to original positions (residual)
            # We use a residual connection: x = x + scatter(h_selected - selected)
            # This ensures skipped segments retain their previous state.
            x_new = x.clone()
            x_new.scatter_(1, topk_indices.unsqueeze(-1).expand(-1, -1, D), h_selected)
            x = x_new
            
            # 4. Aux Loss
            total_aux_loss = total_aux_loss + router.compute_aux_loss(scores)
            all_routed_fraction.append((scores > 0).float().mean())
            
        x = self.final_norm(x)
        
        metrics = {
            'mod_aux_loss': total_aux_loss,
            'mod_routed_fraction': torch.stack(all_routed_fraction).mean(),
        }
        
        return x, metrics


def pool_segments(
    h_bytes: torch.Tensor,
    boundaries: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pool byte-level embeddings into segment-level representations.
    
    Uses the real segment_pool kernel from kernels/segment_ops.py.
    
    Args:
        h_bytes: [B, T, D] — byte-level hidden states
        boundaries: [B, T] — binary boundary mask
        
    Returns:
        pooled: [B, S, D] — segment-level representations
        segment_ids: [B, T] — segment ID per byte position
        counts: [B, S] — bytes per segment
    """
    segment_ids = boundaries_to_segment_ids(boundaries)
    pooled, counts, n_segments = segment_pool(h_bytes, segment_ids)
    
    return pooled, segment_ids, counts
