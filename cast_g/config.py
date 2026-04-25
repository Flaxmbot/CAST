"""
CAST-G Configuration — Scale Presets.

Two standard configurations for research and development:
- 'small': ~5-8M params — fast iteration, fits single T4 (16GB)
- 'medium': ~50M params — serious experiments, needs 2x T4 via DataParallel
"""


CONFIGS = {
    'small': {
        # Core dimensions
        'd_model': 256,
        'n_head': 8,
        'dropout': 0.1,
        
        # Encoder
        'encoder_type': 'multiscale',  # MultiScaleConvStem + ParallelLRU
        
        # Hierarchy
        'n_hierarchy_levels': 3,
        'hierarchy_targets': [8.0, 24.0, 64.0],  # bytes per segment per level
        
        # Global stack (per hierarchy level)
        'global_n_layer': 4,
        'mod_capacity': 0.5,  # MoD: 50% of segments get full compute
        
        # Decoder (O(T) causal convolutions)
        'decoder_n_layer': 3,
        'decoder_kernel_size': 8,  # 8-byte local context window
        
        # Loss weighting (CRITICAL for training stability)
        'seg_loss_weight': 0.01,    # Segmentation loss weight relative to reconstruction
        'aux_warmup_steps': 200,    # Steps of reconstruction-only training before aux kicks in
        
        # Training
        'block_size': 1024,
        'batch_size': 32,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'max_steps': 10000,
        'warmup_steps': 500,
        'boundary_temp_start': 2.0,
        'boundary_temp_end': 0.1,
        'boundary_temp_steps': 5000,
    },
    
    'medium': {
        'd_model': 512,
        'n_head': 8,
        'dropout': 0.1,
        
        'encoder_type': 'multiscale',
        
        'n_hierarchy_levels': 3,
        'hierarchy_targets': [8.0, 24.0, 64.0],
        
        'global_n_layer': 8,
        'mod_capacity': 0.6,
        
        # Decoder (O(T) causal convolutions)
        'decoder_n_layer': 4,
        'decoder_kernel_size': 12,  # 12-byte local context window
        
        # Loss weighting
        'seg_loss_weight': 0.01,
        'aux_warmup_steps': 500,
        
        'block_size': 1024,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'max_steps': 20000,
        'warmup_steps': 1000,
        'boundary_temp_start': 2.0,
        'boundary_temp_end': 0.1,
        'boundary_temp_steps': 8000,
    },
}


def get_config(name: str = 'small') -> dict:
    """Get a config by name. Returns a copy to prevent mutation."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {list(CONFIGS.keys())}")
    return dict(CONFIGS[name])
