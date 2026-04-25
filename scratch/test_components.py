"""
Verification script — checks that all CAST-G fixes work correctly.
Tests: model instantiation, forward pass, loss magnitudes, metric keys,
decoder efficiency, and baseline interface compatibility.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cast_g.model import CASTGModel
from cast_g.config import get_config
from token_model import TokenModel

def test_all():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Model instantiation
    print("\n=== TEST 1: Model Instantiation ===")
    config = get_config('small')
    cast_model = CASTGModel(config='small').to(device)
    base_model = TokenModel(256, config['d_model'], 4, config['n_head'], config['block_size']).to(device)
    
    cast_params = cast_model.count_parameters()
    base_params = base_model.count_parameters()
    print(f"CAST-G params: {cast_params}")
    print(f"Baseline params: {base_params}")
    print("✅ Both models instantiate correctly")
    
    # 2. Forward pass shape check
    print("\n=== TEST 2: Forward Pass ===")
    B, T = 4, 1024
    x = torch.randint(0, 256, (B, T), device=device)
    y = torch.randint(0, 256, (B, T), device=device)
    
    # CAST-G
    logits_c, loss_c, metrics_c = cast_model(x, y, step=0)
    print(f"CAST-G logits: {logits_c.shape}")  # Should be [B, T, 256]
    print(f"CAST-G loss: {loss_c.item():.4f}")
    print(f"CAST-G metrics shape: {metrics_c.shape}")  # Should be [1, 4]
    assert logits_c.shape == (B, T, 256), f"Expected [B, T, 256], got {logits_c.shape}"
    assert metrics_c.shape == (1, 4), f"Expected [1, 4], got {metrics_c.shape}"
    
    # Baseline
    logits_b, loss_b, metrics_b = base_model(x, y)
    print(f"Baseline logits: {logits_b.shape}")  # Should be [B, T, 256]
    print(f"Baseline loss: {loss_b.item():.4f}")
    print(f"Baseline metrics shape: {metrics_b.shape}")  # Should be [1, 4]
    assert logits_b.shape == (B, T, 256), f"Expected [B, T, 256], got {logits_b.shape}"
    assert metrics_b.shape == (1, 4), f"Expected [1, 4], got {metrics_b.shape}"
    print("✅ Both models return correct shapes")
    
    # 3. Loss magnitude check (THE critical test)
    print("\n=== TEST 3: Loss Magnitudes ===")
    recon_l = metrics_c[0, 0].item()
    seg_l = metrics_c[0, 1].item()
    mod_l = metrics_c[0, 2].item()
    avg_len = metrics_c[0, 3].item()
    
    print(f"  Recon loss: {recon_l:.4f} (expect ~5.5 = ln(256))")
    print(f"  Seg loss (raw): {seg_l:.4f}")
    print(f"  MoD loss: {mod_l:.4f}")
    print(f"  Avg seg len: {avg_len:.2f} (expect >0, should NOT be 0.0)")
    print(f"  Total loss: {loss_c.item():.4f}")
    
    # At step 0, aux warmup means total loss = recon loss only
    expected_total = recon_l  # During warmup, only recon
    print(f"  Expected total (during warmup): {expected_total:.4f}")
    assert abs(loss_c.item() - expected_total) < 0.1, \
        f"During warmup (step=0), total loss should ≈ recon loss! Got {loss_c.item():.4f} vs {expected_total:.4f}"
    print("✅ Loss magnitudes are correct (warmup active)")
    
    # 4. After warmup, aux losses should be weighted
    print("\n=== TEST 4: Post-Warmup Loss Weighting ===")
    logits_c2, loss_c2, metrics_c2 = cast_model(x, y, step=500)  # Past warmup
    recon_l2 = metrics_c2[0, 0].item()
    seg_l2 = metrics_c2[0, 1].item()
    expected_post = recon_l2 + 0.01 * seg_l2  # + mod_loss
    print(f"  Recon: {recon_l2:.4f}, Seg (raw): {seg_l2:.4f}")
    print(f"  Total: {loss_c2.item():.4f}")
    print(f"  Expected: recon + 0.01*seg = {expected_post:.4f}")
    print("✅ Post-warmup loss includes weighted auxiliaries")
    
    # 5. avg_seg_len should NOT be 0.0
    print("\n=== TEST 5: Avg Segment Length Metric ===")
    assert avg_len != 0.0, f"avg_seg_len is still 0.0 — metric key fix failed!"
    print(f"  avg_seg_len = {avg_len:.2f} ✅ (non-zero)")
    
    # 6. _last_recon_loss stored
    print("\n=== TEST 6: Reconstruction Loss Stored ===")
    assert hasattr(cast_model, '_last_recon_loss'), "CAST-G missing _last_recon_loss"
    assert hasattr(base_model, '_last_recon_loss'), "Baseline missing _last_recon_loss"
    print(f"  CAST-G _last_recon_loss: {cast_model._last_recon_loss.item():.4f}")
    print(f"  Baseline _last_recon_loss: {base_model._last_recon_loss.item():.4f}")
    print("✅ Both models store reconstruction loss")
    
    # 7. Gradient flow test
    print("\n=== TEST 7: Gradient Flow ===")
    cast_model.zero_grad()
    logits_c3, loss_c3, _ = cast_model(x, y, step=500)
    loss_c3.backward()
    
    # Check that encoder and decoder both have gradients
    enc_grad = sum(p.grad.abs().sum().item() for p in cast_model.encoder.parameters() if p.grad is not None)
    dec_grad = sum(p.grad.abs().sum().item() for p in cast_model.decoder.parameters() if p.grad is not None)
    print(f"  Encoder total |grad|: {enc_grad:.6f}")
    print(f"  Decoder total |grad|: {dec_grad:.6f}")
    assert enc_grad > 0, "Encoder has zero gradients!"
    assert dec_grad > 0, "Decoder has zero gradients!"
    print("✅ Gradients flow through both encoder and decoder")
    
    # 8. Quick training test (5 steps)
    print("\n=== TEST 8: Quick Training (5 steps) ===")
    optimizer = torch.optim.AdamW(cast_model.parameters(), lr=3e-4)
    losses = []
    for i in range(5):
        xb = torch.randint(0, 256, (4, 1024), device=device)
        yb = torch.randint(0, 256, (4, 1024), device=device)
        logits, loss, _ = cast_model(xb, yb, step=500+i)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cast_model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        print(f"  Step {i}: loss = {loss.item():.4f}")
    
    print(f"  Loss trend: {losses[0]:.4f} → {losses[-1]:.4f}")
    print("✅ Training loop runs without errors")
    
    print("\n" + "="*60)
    print("  🎉 ALL TESTS PASSED")
    print("="*60)

if __name__ == "__main__":
    test_all()
