import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .encoder import ByteEncoder
from .boundary import BoundaryDetector, LagrangianLoss
from .global_stack import JaggedTransformer, pool_jagged
from .decoder import RVQRefiner, ByteDecoder

class CASTGModel(nn.Module):
    """
    The full CAST-G (Generative) Architecture.
    A modular, hardware-aware 'Tokenizer Killer'.
    """
    def __init__(self, d_model: int = 128, n_layer: int = 2, n_head: int = 4):
        super().__init__()
        self.encoder = ByteEncoder(d_model)
        self.boundary_detector = BoundaryDetector(d_model)
        self.global_stack = JaggedTransformer(d_model, n_head, n_layer)
        self.rvq = RVQRefiner(d_model)
        self.decoder = ByteDecoder(d_model)
        
        self.lagrangian = LagrangianLoss(target_len=8.0)
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # 1. Byte Encoding (O(N) recurrence)
        h_bytes = self.encoder(idx)
        
        # 2. Boundary Detection (Learned)
        boundaries = self.boundary_detector(h_bytes, temp=0.5, hard=True)
        
        # 3. Jagged Pooling (Variable-length segments)
        z_continuous, _ = pool_jagged(h_bytes, boundaries)
        
        # 4. Global Transformer Reasoning with Dynamic Stride
        h_global, importance_gate = self.global_stack(z_continuous)
        
        # 5. RVQ Snapping (Error correction)
        z_snapped, _ = self.rvq(h_global)
        
        # 6. Byte Decoding
        logits = self.decoder(z_snapped)
        
        loss = None
        metrics = {}
        if targets is not None:
            # A. Reconstruction Loss (Fidelity)
            # IMPORTANT: We upsample the Logits (S) to match the Full Targets (T).
            # This ensures every byte in the sequence provides a gradient signal.
            B, S, V = logits.shape
            T_original = targets.size(1)
            
            # [B, S, V] -> [B, V, S] -> Interpolate -> [B, V, T] -> [B, T, V]
            logits_upsampled = F.interpolate(
                logits.transpose(1, 2), 
                size=T_original, 
                mode='nearest'
            ).transpose(1, 2)
            
            # Now shapes match: [B*T, 256] and [B*T]
            loss_recon = F.cross_entropy(
                logits_upsampled.reshape(-1, 256), 
                targets.reshape(-1)
            )
            
            # B. Lagrangian Constraint (Segment Length Stability)
            l_penalty, avg_len = self.lagrangian(boundaries, idx.size(1))
            
            # C. Importance Penalty (Encourage Sparsity in Dynamic Stride)
            # Penalize the model for using too many Transformer steps
            loss_sparsity = importance_gate.mean() * 0.05
            
            # Joint Objective: Prioritize reconstruction (spelling) in early training
            loss = (5.0 * loss_recon) + l_penalty + loss_sparsity
            
            metrics['avg_seg_len'] = avg_len
            metrics['importance_sparsity'] = 1.0 - importance_gate.mean().item()
            metrics['loss_recon'] = loss_recon.item()
            
        return logits, loss, metrics

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temp: float = 0.7):
        # Temperature sampling for diverse generation
        self.eval()
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(idx)
            # Take the last segment's last prediction
            last_logits = logits[:, -1, :] / temp # Apply temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_byte], dim=1)
            # Stop if we get a null or repeated sequence (for short benchmark)
            if idx.size(1) > 200: break 
        self.train()
        return idx
