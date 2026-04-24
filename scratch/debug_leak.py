
import torch
from cast_g.model import CASTGModel
from cast_g.config import get_config

def debug():
    device = 'cpu' # Easier to debug on CPU
    config = get_config('small')
    model = CASTGModel(config=config).to(device)
    model.eval()

    B, T = 1, 128
    x = torch.randint(0, 256, (B, T))
    j = 64
    x_mod = x.clone()
    x_mod[0, j] = (x[0, j] + 1) % 256

    print("Checking components...")
    
    # 1. Encoder
    with torch.no_grad():
        h_base = model.encoder(x)
        h_mod = model.encoder(x_mod)
    
    diff_h = (h_base - h_mod).abs().sum(dim=-1)
    if diff_h[0, :j//4].max() > 1e-5:
        print(f"!!! LEAK IN ENCODER at index { (diff_h[0, :j//4] > 1e-5).nonzero()[0].item() }")
    else:
        print("Encoder is causal.")

    # 2. Hierarchy
    with torch.no_grad():
        levels_base, boundaries_base, ids_base, _ = model.hierarchy(h_base)
        levels_mod, boundaries_mod, ids_mod, _ = model.hierarchy(h_mod)
    
    # Check boundaries
    diff_b = (boundaries_base[0] - boundaries_mod[0]).abs()
    if diff_b[0, :j//4].max() > 1e-5:
        print(f"!!! LEAK IN BOUNDARIES at index { (diff_b[0, :j//4] > 1e-5).nonzero()[0].item() }")
    else:
        print("Boundaries are causal.")

    # Check pooled segments
    # This is trickier because segment counts might change.
    # But if boundaries are causal, segment IDs for the past should be identical.
    
    # 3. Global Stack
    # Pick the level 0 segments
    seg_base = levels_base[0]
    seg_mod = levels_mod[0]
    # If boundaries didn't leak, then seg_base and seg_mod should have same prefix?
    # No, because seg_base[k] is pooled from bytes.
    
    with torch.no_grad():
        h_glob_base, _ = model.global_stack(seg_base)
        h_glob_mod, _ = model.global_stack(seg_mod)
    
    # Find the first segment that changed
    min_len = min(seg_base.size(1), seg_mod.size(1))
    diff_seg = (seg_base[:, :min_len] - seg_mod[:, :min_len]).abs().sum(dim=-1)
    if diff_seg[0].max() > 1e-5:
        first_seg_change = (diff_seg[0] > 1e-5).nonzero()[0].item()
        print(f"First segment changed: {first_seg_change}")
    
    diff_glob = (h_glob_base[:, :min_len] - h_glob_mod[:, :min_len]).abs().sum(dim=-1)
    if diff_glob[0].max() > 1e-5:
        first_glob_change = (diff_glob[0] > 1e-5).nonzero()[0].item()
        print(f"First h_glob changed: {first_glob_change}")
        if first_glob_change < first_seg_change:
            print("!!! LEAK IN GLOBAL STACK (Transformer) !!!")
        else:
            print("Global Stack is causal relative to segments.")

if __name__ == "__main__":
    debug()
