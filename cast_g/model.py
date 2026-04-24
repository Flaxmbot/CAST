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
from .decoder import AutoregressiveLocalDecoder
from .config import get_config


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
        
        # 4. Autoregressive Local Decoder: segment → expansion bytes
        self.decoder = AutoregressiveLocalDecoder(
            d_model=d_model,
            expansion=config.get('decoder_expansion', 8),
            n_layer=config.get('decoder_n_layer', 2),
            n_head=config.get('decoder_n_head', 4),
            dropout=dropout,
        )
        
        # Training state
        self.register_buffer('step_count', torch.tensor(0, dtype=torch.long))
        
    def get_boundary_temp(self, step: Optional[int] = None) -> float:
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
        step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Full forward pass: bytes → logits.
        
        Args:
            idx: [B, T] — raw byte indices (0-255)
            targets: [B, T] — target byte indices (for loss computation)
            step: training step (for temperature annealing)
            
        Returns:
            logits: [B, T, 256] — byte prediction logits
            loss: scalar total loss (None if no targets)
            metrics: dict of training metrics
        """
        B, T = idx.shape
        
        # 1. Encode bytes
        h_bytes = self.encoder(idx)  # [B, T//4, D]
        T_encoded = h_bytes.size(1)  # T//4 after conv stem
        
        # 2. Hierarchical segmentation
        temp = self.get_boundary_temp(step)
        level_segments, level_boundaries, level_segment_ids, seg_metrics = \
            self.hierarchy(h_bytes, temp=temp, hard=True)
        
        # 3. Global reasoning on Level 0 (finest) segments
        # Level 0 has the most segments and finest granularity
        fine_segments = level_segments[0]
        h_global, mod_metrics = self.global_stack(fine_segments)
        
        # 4. Decode: expand each segment into 'expansion' byte predictions
        logits = self.decoder(h_global)  # [B, S0 * expansion, 256]
        
        # 5. Alignment: match logits to original sequence length T
        if logits.size(1) > T:
            logits = logits[:, :T, :]
        elif logits.size(1) < T:
            padding = torch.zeros(B, T - logits.size(1), 256, device=logits.device, dtype=logits.dtype)
            logits = torch.cat([logits, padding], dim=1)
        
        # 6. Loss computation
        loss = None
        metrics = {**seg_metrics, **mod_metrics}
        
        if targets is not None:
            # A. Reconstruction loss (primary objective)
            loss_recon = F.cross_entropy(
                logits.reshape(-1, 256),
                targets.reshape(-1)
            )
            
            # B. Segmentation loss (MI + Lagrangian, from hierarchy)
            loss_seg = seg_metrics.get('total_seg_loss', torch.tensor(0.0, device=idx.device))
            
            # C. MoD auxiliary loss (load balancing)
            loss_mod = mod_metrics.get('mod_aux_loss', torch.tensor(0.0, device=idx.device))
            
            # Joint objective
            loss = loss_recon + loss_seg + loss_mod
            
            # Compute bits-per-byte for honest benchmarking
            bpb = loss_recon.item() / 0.6931472  # ln(2) ≈ 0.693
            
            metrics['loss_recon'] = loss_recon
            metrics['loss_seg'] = loss_seg
            metrics['loss_mod'] = loss_mod
            metrics['bpb'] = bpb
            
            # Update step counter
            if step is None and self.training:
                self.step_count.add_(1)
        
        return logits, loss, metrics
    
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temp: float = 0.7) -> torch.Tensor:
        """
        Generate bytes autoregressively.
        
        Uses block-autoregressive generation: each forward pass produces
        'expansion' new bytes via the local AR decoder, then those bytes
        are fed back as input for the next block.
        
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
        
        expansion = self.config.get('decoder_expansion', 8)
        steps = (max_new_tokens + expansion - 1) // expansion
        
        for _ in range(steps):
            # Encode current context
            h_bytes = self.encoder(idx)
            
            # Segment (use low temperature for sharp boundaries during generation)
            level_segments, _, _, _ = self.hierarchy(h_bytes, temp=0.1, hard=True)
            
            # Global reasoning on fine segments
            fine_segments = level_segments[0]
            h_global, _ = self.global_stack(fine_segments)
            
            # Generate next block from the LAST segment
            if h_global.size(1) == 0:
                break
                
            last_segment = h_global[:, -1:, :]  # [B, 1, D]
            next_bytes = self.decoder.generate_block(last_segment, temp=temp)  # [B, expansion]
            
            idx = torch.cat([idx, next_bytes], dim=1)
            
            # Safety: limit total sequence length
            max_len = self.config.get('block_size', 1024) * 2
            if idx.size(1) > max_len:
                break
        
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
