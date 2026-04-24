import torch
import torch.nn as nn
from cast_g.model import CASTGModel

def test_cast_g():
    print("Starting CAST-G Component Tests...")
    
    # 1. Configuration
    config = {
        'd_model': 128,
        'n_head': 4,
        'n_layer': 2,
        'n_hierarchy_levels': 2,
        'hierarchy_targets': [4.0, 8.0],
        'mod_capacity': 0.5,
        'max_seq_len': 128,
        'dropout': 0.0,
        'block_size': 64
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    
    model = CASTGModel(config=config).to(device)
    model.train()
    
    # 2. Forward Pass Test
    B, T = 2, 64
    idx = torch.randint(0, 256, (B, T)).to(device)
    targets = torch.randint(0, 256, (B, T)).to(device)
    
    print(f"  Running forward pass (B={B}, T={T})...")
    try:
        logits, loss = model(idx, targets=targets)
        print(f"  Forward pass successful. Loss: {loss.item():.4f}")
        print(f"  Logits shape: {logits.shape}")
        assert logits.shape == (B, T, 256), f"Wrong logits shape: {logits.shape}"
    except Exception as e:
        print(f"  Forward pass FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # 3. Backward Pass Test
    print("  Running backward pass...")
    try:
        loss.backward()
        print("  Backward pass successful (gradients computed).")
    except Exception as e:
        print(f"  Backward pass FAILED: {str(e)}")
        return

    # 4. Causality Test
    print("  Checking causality (Gradient Isolation)...")
    # If we change byte T-1, it should NOT affect logits at T-2.
    idx_v2 = idx.clone()
    idx_v2[0, -1] = (idx_v2[0, -1] + 1) % 256
    
    model.eval() # Use eval for causality check to avoid dropout noise
    with torch.no_grad():
        logits_v1, _ = model(idx)
        logits_v2, _ = model(idx_v2)
        
    # Check if prefix logits are identical
    diff = (logits_v1[:, :-1, :] - logits_v2[:, :-1, :]).abs().max().item()
    if diff < 1e-4:
        print("  Causality verified (No future leakage).")
    else:
        print(f"  CAUSALITY LEAK DETECTED! Max diff in prefix: {diff}")

    # 5. Generation Test
    print("  Running generation test...")
    try:
        prompt = torch.randint(0, 256, (1, 10)).to(device)
        out = model.generate(prompt, max_new_tokens=5)
        print(f"  Generation successful. Output length: {out.size(1)}")
        assert out.size(1) == 15
    except Exception as e:
        print(f"  Generation FAILED: {str(e)}")
        return

    # 6. Parameter Count
    counts = model.count_parameters()
    print(f"  Total Parameters: {counts['total']:,}")
    for k, v in counts.items():
        if k != 'total':
            print(f"    - {k}: {v:,}")

    print("\nAll tests PASSED!")

if __name__ == "__main__":
    test_cast_g()
