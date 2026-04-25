"""
Token-Based Baseline Model — Fair Comparison Target.

Standard Transformer with discrete byte vocabulary (256).
Uses the same F.scaled_dot_product_attention as CAST-G for fair
benchmarking (both use Flash/Memory-Efficient Attention when available).

FIX from v2: Proper causal masking via is_causal=True instead of
manual mask construction.

FIX from v3.2: Returns 3-tuple (logits, loss, metrics) for interface
compatibility with benchmarker. Added count_parameters() for consistent
reporting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class BaselineBlock(nn.Module):
    """Pre-norm Transformer block with SDPA causal attention."""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Pre-norm attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Pre-norm FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        # Pre-norm causal self-attention
        normed = self.attn_norm(x)
        q = self.q_proj(normed).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # SDPA with causal masking (same as CAST-G for fair comparison)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.dropout(self.out_proj(attn_out))
        
        # Pre-norm FFN
        x = x + self.ffn(self.ffn_norm(x))
        
        return x


class TokenModel(nn.Module):
    """
    Standard byte-level Transformer baseline.
    
    Uses the same attention backend as CAST-G (SDPA with is_causal=True)
    for a fair throughput comparison. The only architectural difference
    is that this model processes the FULL byte sequence through every
    attention layer, while CAST-G compresses first.
    """
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layer: int = 4,
        n_head: int = 8,
        block_size: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.block_size = block_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            BaselineBlock(d_model, n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs):
        B, T = idx.shape
        
        x = self.token_embedding(idx)
        x = x + self.pos_emb[:, :T, :]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Store for benchmarker compatibility
            self._last_recon_loss = loss.detach()
            
        # Return 3-tuple for interface compatibility with CAST-G
        # metrics_tensor: [1, 4] — [recon_loss, 0, 0, 0]
        if loss is not None:
            metrics = torch.stack([
                loss.detach(),
                torch.tensor(0.0, device=idx.device),
                torch.tensor(0.0, device=idx.device),
                torch.tensor(0.0, device=idx.device),
            ]).unsqueeze(0)
        else:
            metrics = torch.zeros(1, 4, device=idx.device)
        
        return logits, loss, metrics

    def count_parameters(self) -> Dict[str, int]:
        """Report parameter counts by component for consistent benchmarking."""
        embed_params = sum(p.numel() for p in self.token_embedding.parameters()) + self.pos_emb.numel()
        block_params = sum(p.numel() for p in self.blocks.parameters())
        head_params = sum(p.numel() for p in self.head.parameters()) + sum(p.numel() for p in self.ln_f.parameters())
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'embeddings': embed_params,
            'transformer_blocks': block_params,
            'head': head_params,
            'total': total,
        }

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temp: float = 0.7):
        idx = idx.to(next(self.parameters()).device)
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self.forward(idx_cond)
            logits = logits[:, -1, :] / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        self.train()
        return idx
