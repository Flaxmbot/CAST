"""
CAST-G v3: Hierarchical MI-Segmented Architecture with Mixture-of-Depths.

The full integrated model combining three novel contributions:

1. MI-driven boundary learning (replaces heuristic Lagrangian)
   → boundary.py: MIBoundaryDetector, AdaptiveLagrangian

2. 3-level hierarchical segmentation (sub-word → word → phrase)
   → hierarchy.py: HierarchicalSegmenter with cross-level attention

3. Mixture-of-Depths segment routing (variable compute per segment)
   → global_stack.py: MoDSegmentRouter, MoDTransformerStack

Pipeline:
    Raw Bytes → ByteEncoder → HierarchicalSegmenter → MoD-Transformer → AR-LocalDecoder → Byte Logits

FIXES from v2:
- Causal masking in all attention (was missing — model was "cheating")
- Parallel SSM scan (was sequential Python loop)
- Autoregressive local decoder (was single Linear — no inter-byte deps)
- Removed dead RVQ code
- Removed non-functional Triton stubs
- Real segment pooling kernels
- Proper Flash/Memory-Efficient Attention dispatch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .encoder import ByteEncoder
from .boundary import MIBoundaryDetector, MISegmentationLoss
from .hierarchy import HierarchicalSegmenter
from .global_stack import MoDTransformerStack, pool_segments
from .decoder import CausalLocalDecoder
from .config import get_config

__all__ = [
    'CASTGModel',
    'get_config',
    'CausalLocalDecoder',
]


class CASTGModel(nn.Module):
    """
    CAST-G v3 — Token-Agnostic Neural Architecture.
    
    Novel contributions over prior art (BLT, Nawrot, MegaByte):
    1. MI-driven boundaries: segments placed where mutual information drops
    2. Hierarchical segmentation: 3 simultaneous granularity levels  
    3. MoD routing: trivial segments skip Transformer layers entirely
    
    Config-driven: pass a config dict or a preset name ('small', 'medium').
    """
    def __init__(self, config=None, **kwargs):
        super().__init__()
        
        # Load config
        if config is None:
            config = get_config('small')
        elif isinstance(config, str):
            config = get_config(config)
        
        # Allow kwargs to override config
        config.update(kwargs)
        self.config = config
        
        d_model = config['d_model']
        n_head = config['n_head']
        dropout = config.get('dropout', 0.1)
        
        # 1. Byte Encoder: raw bytes → contextualized embeddings [B, T//4, D]
        self.encoder = ByteEncoder(d_model, dropout=dropout)
        
        # 2. Hierarchical Segmenter: embeddings → multi-level segments
        self.hierarchy = HierarchicalSegmenter(
            d_model=d_model,
            n_levels=config.get('n_hierarchy_levels', 3),
            target_lengths=config.get('hierarchy_targets', [8.0, 24.0, 64.0]),
            dropout=dropout,
        )
        
        # 3. Global Reasoning: MoD-Transformer on Level 0 (finest) segments
        # We reason at the finest level for maximum fidelity, with MoD for efficiency
        self.global_stack = MoDTransformerStack(
            d_model=d_model,
            n_head=n_head,
            n_layer=config.get('global_n_layer', 4),
            capacity_ratio=config.get('mod_capacity', 0.5),
            dropout=dropout,
        )
        
        # 4. Causal Local Decoder (Autoregressive byte prediction)
        self.decoder = CausalLocalDecoder(d_model, dropout=dropout)

        
        # 5. Global Start Token (for causality shift)
        self.start_seg = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Training state
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        """Compute annealed boundary temperature."""
        if step is None:
            step = self.step_count.item()
        
        temp_start = self.config.get('boundary_temp_start', 2.0)
        temp_end = self.config.get('boundary_temp_end', 0.1)
        temp_steps = self.config.get('boundary_temp_steps', 5000)
        
        progress = min(1.0, step / max(1, temp_steps))
        return temp_start + (temp_end - temp_start) * progress
    
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        temp: float = 1.0,
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: bytes -> logits, loss, metrics_tensor.
        """

        metrics = {}

        B, T = idx.shape
        T_encoded = T // 4
        
        # 1. Byte Encoding (O(T) convolution/SSM)
        h_bytes = self.encoder(idx)  # [B, T//4, D]
        
        # 2. Hierarchical Segmentation (MI-driven)
        # Produces segments, boundaries, and auxiliary losses
        level_segments, level_boundaries, level_segment_ids, seg_metrics = \
            self.hierarchy(h_bytes, temp=temp, hard=True)
            
        # We use the finest level (level 0) for the global stack to maintain detail
        fine_segments = level_segments[0]
        fine_segment_ids = level_segment_ids[0]  # [B, T_encoded]
        
        # 3. Global Transformer Stack (O(S²) attention)
        # Processes contextualized segment representations
        h_global, mod_metrics = self.global_stack(fine_segments)
        
        # 4. Strict Causality Shift
        # To predict bytes in segment k, we must only use h_global[k-1].
        B_g, S_g, D_g = h_global.shape
        h_shifted = torch.cat([self.start_seg.expand(B_g, -1, -1), h_global[:, :-1, :]], dim=1)
        
        # 5. Causal Local Decoding (Autoregressive)
        # Combines segment context with shifted input bytes
        logits = self.decoder(h_shifted, fine_segment_ids, T_encoded, T, idx)
        
        # 6. Loss Calculation
        loss = None
        if targets is not None:
            # Reconstruction Loss (Primary)
            loss_recon = F.cross_entropy(logits.view(-1, 256), targets.view(-1))
            
            # Auxiliary Losses (Segmentation & MoD)
            loss_seg = seg_metrics.get('total_seg_loss', torch.tensor(0.0, device=idx.device))
            loss_mod = mod_metrics.get('mod_aux_loss', torch.tensor(0.0, device=idx.device))
            
            # Total Loss
            loss = loss_recon + loss_seg + loss_mod

            
            # Record metrics for gathering
            avg_seg_len = seg_metrics.get('avg_seg_len', torch.tensor(0.0, device=idx.device))
            
            # metrics_tensor: [recon_loss, seg_loss, mod_loss, avg_seg_len]
            metrics_tensor = torch.stack([
                loss_recon.detach(),
                loss_seg.detach(),
                loss_mod.detach(),
                avg_seg_len.detach()
            ])

            
            # Update step counter
            if step is None and self.training:
                self.step_count.add_(1)
        else:
            metrics_tensor = torch.zeros(4, device=idx.device)
        
        return logits, loss, metrics_tensor

    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temp: float = 0.7) -> torch.Tensor:
        """
        Generate bytes autoregressively.
        
        Each step does a full forward pass and samples from the last position.
        
        Args:
            idx: [B, T] — prompt byte indices
            max_new_tokens: maximum number of new bytes to generate
            temp: sampling temperature
            
        Returns:
            [B, T + generated] — full sequence including prompt
        """
        device = next(self.parameters()).device
        idx = idx.to(device)
        self.eval()
        
        for _ in range(max_new_tokens):
            # Use last block_size bytes as context
            block_size = self.config.get('block_size', 1024)
            context = idx[:, -block_size:] if idx.size(1) > block_size else idx
            
            # Full forward pass (no targets → no loss)
            logits, _ = self.forward(context, targets=None)
            
            # Sample from last position
            last_logits = logits[:, -1, :] / temp  # [B, 256]
            probs = torch.softmax(last_logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            idx = torch.cat([idx, next_byte], dim=1)
        
        self.train()
        return idx
    
    def count_parameters(self) -> Dict[str, int]:
        """Report parameter counts by component."""
        components = {
            'encoder': self.encoder,
            'hierarchy': self.hierarchy,
            'global_stack': self.global_stack,
            'decoder': self.decoder,
        }
        
        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n
        
        counts['total'] = total
        return counts
