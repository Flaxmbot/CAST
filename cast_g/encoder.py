"""
CAST-G Byte Encoder — High-Frequency Local Processing.

Converts raw byte sequences into contextualized embeddings using:
1. Byte embedding (256-entry table)
2. Multi-scale convolutional stem (replaces static stride-4)
3. Parallel Linear Recurrent Unit (replaces sequential loop)

Changes from v2:
- ConvStem now uses multiple kernel sizes for multi-scale local features
- SSM uses parallel associative scan (O(log T) depth) instead of Python for-loop
- Added residual connections and dropout for training stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kernels.fast_scan import parallel_scan


class MultiScaleConvStem(nn.Module):
    """
    Multi-scale convolutional encoder that captures local patterns at
    different granularities before downsampling.
    
    Uses parallel convolutions with kernel sizes [2, 4, 8] to capture
    bigram, quadgram, and octgram patterns simultaneously, then fuses
    and downsamples by stride 4.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Multi-scale parallel convolutions (causal: padding = k-1)
        self.conv_2 = nn.Conv1d(d_model, d_model // 4, kernel_size=2, padding=1)
        self.conv_4 = nn.Conv1d(d_model, d_model // 4, kernel_size=4, padding=3)
        self.conv_8 = nn.Conv1d(d_model, d_model // 4, kernel_size=8, padding=7)
        self.conv_1 = nn.Conv1d(d_model, d_model // 4, kernel_size=1)  # Identity scale
        
        # Fusion and downsampling (causal: padding = k-1)
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=4, stride=4, padding=3)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] — byte embeddings
        Returns:
            [B, T//4, D] — downsampled multi-scale features
        """
        B, T, D = x.shape
        # [B, T, D] -> [B, D, T] for Conv1d
        x_t = x.transpose(1, 2)
        
        # Multi-scale feature extraction (causal: trim future outputs)
        h1 = self.conv_1(x_t)                                 # [B, D//4, T]
        h2 = self.conv_2(x_t)[:, :, :-1]                      # [B, D//4, T]
        h4 = self.conv_4(x_t)[:, :, :-3]                      # [B, D//4, T]
        h8 = self.conv_8(x_t)[:, :, :-7]                      # [B, D//4, T]
        
        # Concatenate multi-scale features
        h_multi = torch.cat([h1, h2, h4, h8], dim=1)  # [B, D, T]
        h_multi = self.act(h_multi)
        
        # Downsample by 4x (causal: trim future outputs)
        h_down = self.downsample(h_multi)  # [B, D, T_new]
        # Transpose convolution with stride 4 and padding 3 produces:
        # output_len = floor((T + 2*3 - 4) / 4) + 1 = floor((T+2)/4) + 1
        # We need exactly T/4. If T=1024, (1024+2)/4 + 1 = 257.
        # Trim to exactly T//4
        target_T = T // 4
        h_down = h_down[:, :, :target_T]
        
        h_down = h_down.transpose(1, 2)    # [B, T//4, D]
        
        return self.dropout(self.norm(h_down))


class ParallelLRU(nn.Module):
    """
    Linear Recurrent Unit with parallel associative scan.
    
    Replaces the sequential for-loop with a log-space prefix sum
    that has O(T) work and O(log T) depth on GPU.
    
    Math: h[t] = λ * h[t-1] + (1-λ) * B(x[t])
          y[t] = C(h[t])
    
    Where λ ∈ (0, 1) is a learned per-dimension decay rate.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Learnable decay rates (initialized for moderate memory)
        # log(sigmoid(x)) parameterization for stability
        self.log_lambda_raw = nn.Parameter(torch.randn(d_model) * 0.5 - 1.0)
        
        self.b_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D] — recurrence output
        """
        B, T, D = x.shape
        
        # Stable decay rate in (0, 1)
        lam = torch.sigmoid(self.log_lambda_raw)  # [D]
        log_lam = torch.log(lam + 1e-8)           # [D]
        
        # Input projection
        x_b = self.b_proj(x)  # [B, T, D]
        values = (1.0 - lam) * x_b  # Scaled input
        
        # Log coefficients: broadcast to [B, T, D]
        log_coeffs = log_lam.unsqueeze(0).unsqueeze(0).expand(B, T, D)
        
        # Parallel scan (O(log T) depth)
        h_all = parallel_scan(log_coeffs, values)
        
        # Output projection with residual
        out = self.c_proj(h_all)
        out = self.dropout(self.norm(out + x))
        
        return out


class ByteEncoder(nn.Module):
    """
    Complete byte-level encoder: Embed → MultiScaleConv → ParallelLRU.
    
    Converts raw byte indices [B, T] into contextualized embeddings [B, T//4, D].
    The 4x downsampling reduces the sequence length before the expensive
    Transformer stack, making attention O((T/4)²) instead of O(T²).
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(256, d_model)
        self.stem = MultiScaleConvStem(d_model, dropout=dropout)
        self.ssm = ParallelLRU(d_model, dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] — raw byte indices (0-255)
        Returns:
            [B, T//4, D] — contextualized byte embeddings
        """
        h = self.embed(x)     # [B, T, D]
        h = self.stem(h)      # [B, T//4, D]
        h = self.ssm(h)       # [B, T//4, D]
        return h
