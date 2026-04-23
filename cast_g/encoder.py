import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class ConvStem(nn.Module):
    """
    Reduces the sequence length by 4x before SSM processing.
    This solves the high-frequency byte latency.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim, out_dim, 
            kernel_size=4, stride=4, padding=0
        )
        self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x.transpose(1, 2) # [B, D, T]
        x = self.conv(x)
        x = x.transpose(1, 2) # [B, T/4, D]
        return self.ln(x)

class SimpleSSM(nn.Module):
    """
    A Linear Recurrent Unit (LRU) as a parallelizable SSM proxy.
    
    [TECHNICAL NOTE]: For the LinkedIn showcase, this is implemented 
    as a fast sequential loop. In a production B200 deployment, 
    this would be replaced by the fused parallel prefix sum 
    kernel found in kernels/ssm_scan.py.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Lambda (diagonal recurrence)
        self.log_lambda = nn.Parameter(torch.log(torch.exp(torch.randn(d_model) * 0.1) - 1e-6))
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        lam = torch.exp(self.log_lambda).sigmoid() # Stability constraint
        
        # In a real 'fused scan', this would be a parallel prefix sum (O(log T))
        # Here we implement it as a fast sequential loop for the prototype
        # (For production, this would use kernels/ssm_scan.py logic)
        h = torch.zeros(B, D, device=x.device)
        outputs = []
        
        x_b = self.b_proj(x)
        for t in range(T):
            h = lam * h + (1 - lam) * x_b[:, t, :]
            outputs.append(h)
            
        h_all = torch.stack(outputs, dim=1) # [B, T, D]
        return self.c_proj(h_all)

class ByteEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)
        self.stem = ConvStem(d_model, d_model)
        self.ssm = SimpleSSM(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T] (raw bytes)
        x = self.embed(x)      # [B, T, D]
        x = self.stem(x)       # [B, T/4, D]
        x = self.ssm(x)        # [B, T/4, D]
        return x
