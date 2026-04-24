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
        self.decoder = ByteDecoder(d_model)
        # Weight Tying: Tie decoder projection to input embeddings
        self.decoder.proj.weight = self.encoder.embed.weight
        
        self.lagrangian = LagrangianLoss(target_len=8.0)
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long)) # Persistent step count for annealing
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, step: Optional[int] = None):
        # 1. Byte Encoding (O(N) recurrence)
        h_bytes = self.encoder(idx)
        
        # 2. Boundary Detection (Learned)
        boundaries = self.boundary_detector(h_bytes, temp=0.5, hard=True)
        
        # 3. Jagged Pooling (Variable-length segments)
        z_continuous, _ = pool_jagged(h_bytes, boundaries)
        
        # 4. Global Transformer Reasoning with Dynamic Stride
        h_global, importance_gate = self.global_stack(z_continuous)
        
        # 4. Byte Decoding (Direct from Transformer output)
        logits = self.decoder(h_global)
        
        # [FIX]: Upsample logits back to original byte resolution (T)
        # This ensures generate() can take logits[:, -1, :] as the next byte prediction
        B, S, V = logits.shape
        T_original = idx.size(1)
        
        logits = F.interpolate(
            logits.transpose(1, 2), 
            size=T_original, 
            mode='nearest'
        ).transpose(1, 2)
        
        loss = None
        metrics = {}
        if targets is not None:
            # A. Reconstruction Loss (Fidelity)
            loss_recon = F.cross_entropy(
                logits.reshape(-1, 256), 
                targets.reshape(-1)
            )
            
            # B. Lagrangian Constraint (Segment Length Stability)
            l_penalty, avg_len = self.lagrangian(boundaries, idx.size(1))
            
            # C. Dynamic Entropy Annealing (The 'Showcase' Stability Fix)
            # Use passed step or fallback to buffer
            if step is not None:
                curr_step = torch.tensor(step, device=idx.device)
            else:
                curr_step = self.step_count
            
            anneal_factor = F.relu(1.0 - (curr_step.float() / 5000.0))
            p = torch.sigmoid(self.boundary_detector.proj(h_bytes))
            p = p.clamp(1e-6, 1.0 - 1e-6) # Clamp for numerical stability
            entropy = -p * torch.log(p) - (1-p) * torch.log(1-p)
            loss_entropy = -0.05 * anneal_factor * entropy.mean()
            
            # D. Sparsity Penalty (Encourages Dynamic Stride efficiency)
            loss_sparsity = importance_gate.mean() * 0.05
            
            # Joint Objective
            loss = (5.0 * loss_recon) + l_penalty + loss_sparsity + loss_entropy
            if step is None and self.training: 
                self.step_count.add_(1) # Only auto-increment during training
            
            metrics['avg_seg_len'] = avg_len
            metrics['importance_sparsity'] = 1.0 - importance_gate.mean()
            metrics['loss_recon'] = loss_recon
            
        return logits, loss, metrics

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temp: float = 0.7):
        # FIX: Ensure prompt is on the same device as the model
        idx = idx.to(next(self.parameters()).device)
        
        # Temperature sampling for diverse generation
        self.eval()
        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(idx)
            # Take the last segment's last prediction
            last_logits = logits[:, -1, :] / temp # Apply temperature
            probs = torch.softmax(last_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_byte], dim=1)
        self.train()
        return idx
