
import torch
import torch.nn.functional as F
from cast_g.model import CASTGModel
from cast_g.config import get_config

def test_causality():
    print("--- RUNNING CAUSALITY TEST ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config('small')
    model = CASTGModel(config=config).to(device)
    model.eval()

    B, T = 1, 128
    x = torch.randint(0, 256, (B, T)).to(device)
    
    # 1. Base forward pass
    with torch.no_grad():
        logits_base, _ = model(x)
    
    # 2. Modify input at position j=64
    j = 64
    x_mod = x.clone()
    x_mod[0, j] = (x[0, j] + 1) % 256
    
    # 3. Modified forward pass
    with torch.no_grad():
        logits_mod, _ = model(x_mod)
    
    # 4. Check differences
    diff = (logits_base - logits_mod).abs().sum(dim=-1) # [B, T]
    
    max_past_diff = diff[0, :j].max().item()
    max_future_diff = diff[0, j:].max().item()
    
    print(f"  Max change in past (0 to {j-1}): {max_past_diff:.2e}")
    print(f"  Max change in future ({j} to {T-1}): {max_future_diff:.2e}")
    
    if max_past_diff < 1e-4:
        print("OK: CAUSALITY VERIFIED")
    else:
        print("FAIL: CAUSALITY VIOLATION!")
        # Find where it leaks
        leak_indices = (diff[0, :j] > 1e-4).nonzero()
        if len(leak_indices) > 0:
            print(f"  First leak at index: {leak_indices[0].item()}")

def test_gradient_stability():
    print("\n--- RUNNING GRADIENT STABILITY TEST ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config('small')
    model = CASTGModel(config=config).to(device)
    model.train()
    
    B, T = 4, 128
    x = torch.randint(0, 256, (B, T)).to(device)
    y = torch.randint(0, 256, (B, T)).to(device)
    
    logits, loss = model(x, y)
    loss.backward()
    
    max_grad = 0.0
    max_name = ""
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.abs().max().item()
            if gnorm > max_grad:
                max_grad = gnorm
                max_name = name
    
    print(f"  Max gradient magnitude: {max_grad:.4f} (in {max_name})")
    
    if max_grad < 100.0:
        print("OK: GRADIENTS STABLE.")
    else:
        print("FAIL: GRADIENT EXPLOSION DETECTED!")

if __name__ == "__main__":
    test_causality()
    test_gradient_stability()
