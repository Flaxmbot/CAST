import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# --- Standard Tokenizer-Based Architecture (The Baseline) ---
class TokenModel(nn.Module):
    """
    Standard Transformer with a Discrete Vocabulary (Character-level).
    This serves as the benchmark baseline.
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 384, n_layer: int = 6, n_head: int = 6, block_size: int = 256):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.head_dim = d_model // n_head
        
        # Traditional Embedding Table
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model)) # Standard learned pos emb
        
        self.blocks = nn.ModuleList([BaselineBlock(d_model, n_head, self.head_dim) for _ in range(n_layer)])
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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        # Standard Lookup
        x = self.token_embedding(idx) 
        x = x + self.pos_emb[:, :T, :]
        
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temp: float = 0.7):
        # FIX: Ensure prompt is on the correct device
        idx = idx.to(next(self.parameters()).device)
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self.forward(idx_cond)
            # Focus on last step and apply temperature
            logits = logits[:, -1, :] / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        self.train()
        return idx

class BaselineBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, head_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        # Use standard mask for consistency
        # MultiheadAttention expects (L, L) mask of boolean or float
        m = None
        if mask is not None:
            m = (mask[0, 0] == 0) # Convert to boolean attention mask
            
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=m, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x
