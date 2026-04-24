"""
CAST-G Decoder — Autoregressive Local Byte Decoder.

CRITICAL FIX: Replaces the single Linear(d_model, 8*256) with a small
causal Transformer that autoregressively models inter-byte dependencies
within each expansion block.

Why this matters: A single linear projection cannot model P(byte_5 | byte_1..4).
Each of the 8 predicted bytes is independent of the others — causing incoherent
generation. MegaByte (Meta, 2023) solves this with a small autoregressive local
decoder, and we do the same here with a more efficient cross-attention design.

Architecture:
    segment_latent → cross-attention → causal self-attention → byte logits
    
    The segment latent provides the "what to say" (content),
    and the autoregressive self-attention provides the "how to spell it" (byte order).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalDecoderBlock(nn.Module):
    """
    Single block of the local autoregressive decoder.
    
    1. Cross-attention from byte positions to the parent segment latent
    2. Causal self-attention between byte positions within the block
    3. Feed-forward network
    
    All using scaled_dot_product_attention for T4-compatible efficiency.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Cross-attention: bytes attend to segment latent
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_k = nn.Linear(d_model, d_model)
        self.cross_v = nn.Linear(d_model, d_model)
        self.cross_out = nn.Linear(d_model, d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        
        # Causal self-attention: bytes attend to previous bytes in block
        self.self_q = nn.Linear(d_model, d_model)
        self.self_k = nn.Linear(d_model, d_model)
        self.self_v = nn.Linear(d_model, d_model)
        self.self_out = nn.Linear(d_model, d_model)
        self.self_norm = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [B*S, L, D] → [B*S, n_head, L, head_dim]"""
        BS, L, D = x.shape
        return x.view(BS, L, self.n_head, self.head_dim).transpose(1, 2)
    
    def forward(self, byte_embeds: torch.Tensor, segment_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_embeds: [B*S, expansion, D] — byte position embeddings
            segment_latent: [B*S, 1, D] — parent segment representation
            
        Returns:
            [B*S, expansion, D] — updated byte representations
        """
        # 1. Cross-attention to segment latent (pre-norm)
        normed = self.cross_norm(byte_embeds)
        q = self._reshape_for_attention(self.cross_q(normed))        # [B*S, H, E, hd]
        k = self._reshape_for_attention(self.cross_k(segment_latent)) # [B*S, H, 1, hd]
        v = self._reshape_for_attention(self.cross_v(segment_latent)) # [B*S, H, 1, hd]
        
        cross_out = F.scaled_dot_product_attention(q, k, v)  # [B*S, H, E, hd]
        cross_out = cross_out.transpose(1, 2).contiguous().view(byte_embeds.shape)
        byte_embeds = byte_embeds + self.dropout(self.cross_out(cross_out))
        
        # 2. Causal self-attention (pre-norm)
        normed = self.self_norm(byte_embeds)
        q = self._reshape_for_attention(self.self_q(normed))
        k = self._reshape_for_attention(self.self_k(normed))
        v = self._reshape_for_attention(self.self_v(normed))
        
        # Causal mask: each byte only attends to previous bytes in the block
        self_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        self_out = self_out.transpose(1, 2).contiguous().view(byte_embeds.shape)
        byte_embeds = byte_embeds + self.dropout(self.self_out(self_out))
        
        # 3. FFN (pre-norm)
        byte_embeds = byte_embeds + self.ffn(self.ffn_norm(byte_embeds))
        
        return byte_embeds


class AutoregressiveLocalDecoder(nn.Module):
    """
    Small causal Transformer that generates 'expansion' bytes per segment.
    
    Unlike the v2 single Linear(d_model, expansion*256), this models
    inter-byte dependencies: P(byte_k | byte_1..k-1, segment_latent).
    
    Architecture:
        1. Each segment latent is "unrolled" into `expansion` byte positions
        2. Byte positions get learned positional embeddings (0..expansion-1)
        3. Cross-attention injects the segment's semantic content
        4. Causal self-attention models byte-to-byte dependencies
        5. Final linear head produces byte logits (256-way)
    """
    def __init__(self, d_model: int, expansion: int = 8, n_layer: int = 2, n_head: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.expansion = expansion
        
        # Positional embeddings for byte positions within a block
        self.pos_embed = nn.Embedding(expansion, d_model)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            LocalDecoderBlock(d_model, n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        # Output head: d_model → 256 byte logits
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256)
        
    def forward(self, segment_latents: torch.Tensor) -> torch.Tensor:
        """
        Expand segment-level latents into byte-level predictions.
        
        Args:
            segment_latents: [B, S, D] — one latent per segment
            
        Returns:
            logits: [B, S * expansion, 256] — byte logits
        """
        B, S, D = segment_latents.shape
        E = self.expansion
        
        # Create byte position embeddings: [E, D]
        positions = torch.arange(E, device=segment_latents.device)
        pos_embeds = self.pos_embed(positions)  # [E, D]
        
        # Broadcast to all segments: [B*S, E, D]
        byte_embeds = pos_embeds.unsqueeze(0).expand(B * S, -1, -1)
        
        # Reshape segment latents for cross-attention: [B*S, 1, D]
        seg_flat = segment_latents.reshape(B * S, 1, D)
        
        # Run through decoder blocks
        for block in self.blocks:
            byte_embeds = block(byte_embeds, seg_flat)
        
        # Project to byte logits
        logits = self.head(self.head_norm(byte_embeds))  # [B*S, E, 256]
        
        # Reshape to [B, S*E, 256]
        logits = logits.view(B, S * E, 256)
        
        return logits
    
    def generate_block(self, segment_latent: torch.Tensor, temp: float = 0.7) -> torch.Tensor:
        """
        Autoregressively generate one block of bytes from a segment latent.
        Used during inference for higher quality generation.
        
        Args:
            segment_latent: [B, 1, D] — single segment latent
            temp: sampling temperature
            
        Returns:
            bytes: [B, expansion] — generated byte indices
        """
        B = segment_latent.size(0)
        D = segment_latent.size(2)
        E = self.expansion
        device = segment_latent.device
        
        # Start with just positional embeddings
        positions = torch.arange(E, device=device)
        byte_embeds = self.pos_embed(positions).unsqueeze(0).expand(B, -1, -1)  # [B, E, D]
        seg_flat = segment_latent  # [B, 1, D]
        
        # Forward through all blocks (causal mask ensures autoregressive)
        for block in self.blocks:
            byte_embeds = block(byte_embeds, seg_flat)
        
        # Get logits and sample
        logits = self.head(self.head_norm(byte_embeds))  # [B, E, 256]
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        
        # Sample each position
        sampled = torch.multinomial(probs.view(B * E, 256), num_samples=1)
        return sampled.view(B, E)
