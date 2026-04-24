import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryDetector(nn.Module):
    """
    Learned Gumbel-Bernoulli Boundary Detector.
    Determines segment boundaries based on latent state changes.
    [RESEARCH NOTE]: Temperature annealing (starting at 1.0, 
    decaying to 0.1) is recommended for large-scale training 
    to stabilize boundary discovery.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2) # Binary: 0=Continue, 1=Boundary
        )
        
    def forward(self, x: torch.Tensor, temp: float = 1.0, hard: bool = True) -> torch.Tensor:
        # x: [B, T, D]
        logits = self.proj(x)
        # Gumbel-Softmax for differentiable sampling
        # (Temperature controls the sharpness of the decision)
        boundaries = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)
        return boundaries[..., 1] # Binary 1D mask

class LagrangianLoss:
    """
    Helps stabilize the boundary detector by penalizing segment lengths 
    that drift too far from the target.
    """
    def __init__(self, target_len: float = 8.0, lambda_init: float = 0.01):
        super().__init__()
        self.target_len = target_len
        self.lam = lambda_init
        
    def __call__(self, boundaries: torch.Tensor, total_bytes: int):
        # boundaries: [B, T]
        # Calculate average segments per sequence in the batch
        avg_segments = boundaries.sum() / boundaries.size(0)
        avg_len = total_bytes / (avg_segments + 1e-6)
        
        # We use a squared penalty for smoother gradients
        penalty = (avg_len - self.target_len)**2
        return self.lam * penalty, avg_len # Return as tensor to avoid graph break
