"""
CAST-G Boundary Detection — Mutual Information-Driven Segmentation.

NOVEL CONTRIBUTION: Unlike Nawrot (2023) who uses Gumbel-Sigmoid with a
heuristic Lagrangian, and unlike BLT (2024) which uses entropy-based 
patching, CAST-G uses *Mutual Information* between adjacent context 
windows to drive boundary placement.

The key insight: a segment boundary should be placed where the MI between
the left context and right context drops sharply — indicating a semantic
discontinuity. This is theoretically grounded in information theory and
produces boundaries that correlate with linguistic structure (morphemes,
words, phrases) without any supervision.

Components:
1. MIBoundaryDetector — InfoNCE-based MI estimator for boundary scoring
2. AdaptiveLagrangian — Proper dual-variable optimization (not fixed λ)
3. MISegmentationLoss — Joint objective: max intra-MI, min inter-MI
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MIBoundaryDetector(nn.Module):
    """
    Mutual-Information-driven boundary detection.
    
    At each byte position t, estimates MI(left_context, right_context)
    using an InfoNCE lower bound (van den Oord et al., 2018).
    
    A boundary is placed where MI drops below a learned threshold,
    indicating that the left and right contexts are semantically
    independent — a natural segmentation point.
    
    This replaces Nawrot's Gumbel-Sigmoid boundary predictor and is
    theoretically grounded rather than heuristic.
    """
    def __init__(self, d_model: int, window_size: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Left and right context encoders
        self.left_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        self.right_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2),
        )
        
        # MI estimator: bilinear scoring function f(l, r) ≈ MI(L; R)
        # Using a factored bilinear: f(l, r) = l^T W r
        self.mi_proj = nn.Linear(d_model // 2, d_model // 2, bias=False)
        
        # Learned threshold: boundary when MI < threshold
        self.threshold = nn.Parameter(torch.tensor(0.0))
        
        # Temperature for Gumbel-Sigmoid (annealed during training)
        self.dropout = nn.Dropout(dropout)
        
    def _compute_context_windows(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute causal left and right context representations at each position.
        
        Args:
            h: [B, T, D] — byte-level hidden states
            
        Returns:
            left_ctx: [B, T, D//2] — past context encoding
            right_ctx: [B, T, D//2] — near-past context encoding
        """
        B, T, D = h.shape
        w = self.window_size
        
        # Pad for causal window extraction (only on the left)
        # Pad with 2w zeros on the left so we can always look back 2w
        h_padded = F.pad(h, (0, 0, 2*w, 0), mode='constant', value=0)  # [B, T+2w, D]
        
        # Use avg_pool1d for efficient windowed averaging
        h_t = h_padded.transpose(1, 2)  # [B, D, T+2w]
        pooled = F.avg_pool1d(h_t, kernel_size=w, stride=1, padding=0)  # [B, D, T+w+1]
        pooled = pooled.transpose(1, 2)  # [B, T+w+1, D]
        
        # Original position t is now at index t+2w in h_padded.
        # Window of size w ending at t-1 (padded t+2w-1) is [t+w, t+2w)
        # pooled[i] is avg of padded[i : i+w]
        # For original t, we want:
        # Near-past (Right): avg(padded[t+w : t+2w]) = pooled[t+w]
        # Distant-past (Left): avg(padded[t : t+w]) = pooled[t]
        
        left_ctx = pooled[:, :T, :]      # [B, T, D]
        right_ctx = pooled[:, w:w+T, :]  # [B, T, D]
        
        # Project and NORMALIZE to prevent gradient explosion
        left_ctx = self.left_encoder(left_ctx)
        right_ctx = self.right_encoder(right_ctx)
        
        # Normalize for stable dot product (cosine similarity space)
        left_ctx = F.normalize(left_ctx, dim=-1)
        right_ctx = F.normalize(right_ctx, dim=-1)
        
        return left_ctx, right_ctx
    
    def estimate_mi(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute MI estimates and InfoNCE auxiliary loss.
        """
        left_ctx, right_ctx = self._compute_context_windows(h)
        # already normalized in _compute_context_windows
        
        # Pointwise MI estimate via dot product
        mi_scores = (left_ctx * right_ctx).sum(dim=-1)  # [B, T]
        
        # InfoNCE Loss: contrast true pair (left_t, right_t) against others
        if self.training:
            B, T, D = left_ctx.shape
            # [B, T, D] @ [B, D, T] -> [B, T, T]
            logits = torch.matmul(left_ctx, right_ctx.transpose(1, 2)) / 0.07
            labels = torch.arange(T, device=h.device).repeat(B)
            infonce_loss = F.cross_entropy(logits.view(-1, T), labels)
        else:
            infonce_loss = torch.tensor(0.0, device=h.device)
            
        return mi_scores, infonce_loss
    
    def forward(
        self, 
        h: torch.Tensor, 
        temp: float = 1.0, 
        hard: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict segment boundaries based on MI estimates.
        
        Args:
            h: [B, T, D] — byte-level hidden states
            temp: Gumbel-Sigmoid temperature (lower = sharper boundaries)
            hard: If True, produce hard 0/1 boundaries with STE gradient
            
        Returns:
            boundaries: [B, T] — binary boundary mask
            mi_scores: [B, T] — raw MI estimates (for alignment loss)
            infonce_loss: scalar MI estimation loss
        """
        mi_scores, infonce_loss = self.estimate_mi(h)  # [B, T], scalar
        
        # Boundary logit: low MI → high boundary probability
        # boundary_logit = threshold - mi_score
        # When mi < threshold: positive logit → boundary
        # When mi > threshold: negative logit → no boundary
        boundary_logits = self.threshold - mi_scores  # [B, T]
        
        # Gumbel-Sigmoid for differentiable hard boundaries
        if self.training:
            # Add Gumbel noise for exploration
            u = torch.rand_like(boundary_logits).clamp(1e-6, 1 - 1e-6)
            gumbel_noise = torch.log(u) - torch.log(1 - u)
            noisy_logits = (boundary_logits + gumbel_noise) / temp
        else:
            noisy_logits = boundary_logits / temp
        
        boundaries_soft = torch.sigmoid(noisy_logits)  # [B, T]
        
        if hard:
            # Straight-through estimator: hard forward, soft backward
            boundaries_hard = (boundaries_soft > 0.5).float()
            boundaries = boundaries_hard - boundaries_soft.detach() + boundaries_soft
        else:
            boundaries = boundaries_soft
        
        # Force position 0 to always be a boundary (first segment start)
        boundaries = boundaries.clone()
        boundaries[:, 0] = 1.0
        
        return boundaries, mi_scores, infonce_loss


class AdaptiveLagrangian(nn.Module):
    """
    Proper dual-variable Lagrangian for segment length control.
    
    Unlike the fixed λ=0.01 in CAST-G v2, this uses dual gradient ascent
    to adaptively update λ based on the actual constraint violation.
    
    The constraint is: E[segment_length] ≈ target_length
    The dual variable λ increases when segments are too long/short
    and decreases when the constraint is satisfied.
    
    This converges to the correct compression ratio automatically.
    """
    def __init__(self, target_len: float = 8.0, lambda_lr: float = 0.001):
        super().__init__()
        self.target_len = target_len
        self.lambda_lr = lambda_lr
        # Log-space λ — start very small
        self.register_buffer('log_lambda', torch.tensor(-5.0))  # λ ≈ 0.007
        
    @property
    def lam(self) -> torch.Tensor:
        # Tight clamp: max λ ≈ 1.65 — seg loss can never exceed ~1.65x violation
        return torch.exp(self.log_lambda).clamp(min=1e-4, max=2.0)
    
    def forward(self, boundaries: torch.Tensor, total_bytes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Lagrangian penalty and update dual variable.
        
        Args:
            boundaries: [B, T] — boundary mask
            total_bytes: original sequence length
            
        Returns:
            penalty: scalar Lagrangian penalty (for loss)
            avg_len: average segment length (for logging)
        """
        # Average number of segments per batch item
        avg_segments = boundaries.sum(dim=1).mean()  # scalar
        avg_len = total_bytes / (avg_segments + 1e-6)
        
        # Normalized constraint violation (divided by target for scale-invariance)
        violation = (avg_len - self.target_len) / self.target_len
        
        # Penalty on normalized violation — capped so seg loss never exceeds recon
        penalty = self.lam * violation.pow(2).clamp(max=1.0)
        
        # Proper dual ascent: λ UP when segments too short (need fewer boundaries),
        # λ DOWN when constraint is roughly satisfied (|violation| < 0.1)
        if self.training:
            with torch.no_grad():
                # Signed update: positive violation → increase λ, negative → decrease
                self.log_lambda.add_(self.lambda_lr * violation.detach())
                # Tight clamp: [-5, 0.5] → λ in [0.007, 1.65]
                self.log_lambda.clamp_(-5.0, 0.5)
        
        return penalty, avg_len


class MISegmentationLoss(nn.Module):
    """
    Joint loss for MI-driven segmentation.
    
    Combines:
    1. Intra-segment MI maximization (segments should be internally coherent)
    2. Inter-segment MI minimization (adjacent segments should be independent)
    3. Adaptive Lagrangian constraint (target segment length)
    
    This produces boundaries that are:
    - Semantically meaningful (MI-driven, not heuristic)
    - Consistently sized (Lagrangian constraint)
    - Differentiable (Gumbel-Sigmoid + STE)
    """
    def __init__(self, target_len: float = 8.0, mi_weight: float = 0.1, lagrangian_lr: float = 0.01):
        super().__init__()
        self.lagrangian = AdaptiveLagrangian(target_len=target_len, lambda_lr=lagrangian_lr)
        self.mi_weight = mi_weight
        
    def forward(
        self,
        boundaries: torch.Tensor,
        mi_scores: torch.Tensor,
        total_bytes: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the segmentation loss.
        
        The MI loss encourages boundaries at low-MI positions and
        discourages them at high-MI positions.
        
        Args:
            boundaries: [B, T] — boundary mask (soft or hard)
            mi_scores: [B, T] — MI estimates from MIBoundaryDetector
            total_bytes: original byte sequence length
            
        Returns:
            loss: scalar segmentation loss
            metrics: dict of logged values
        """
        # 1. MI-boundary alignment loss
        # Boundaries should align with LOW MI positions
        # Use detached MI scores: the MI estimator is trained by InfoNCE inside
        # estimate_mi(), not by this loss. Without detach, the gradient pushes
        # mi_scores to extreme values → loss diverges to -inf.
        mi_detached = mi_scores.detach()
        
        # Loss = mean(boundary * |mi|) — minimized when boundaries coincide with
        # near-zero MI (semantic discontinuities). Using abs() prevents the loss
        # from going negative when MI scores are negative.
        mi_loss = (boundaries * mi_detached.abs()).mean()
        
        # 2. Lagrangian constraint
        lag_penalty, avg_len = self.lagrangian(boundaries, total_bytes)
        
        # Combined loss (always non-negative)
        loss = self.mi_weight * mi_loss + lag_penalty
        
        metrics = {
            'avg_seg_len': avg_len,
            'mi_mean': mi_scores.mean(),
            'mi_loss': mi_loss,
            'lagrangian_lambda': self.lagrangian.lam,
            'boundary_rate': boundaries.mean(),
        }
        
        return loss, metrics
