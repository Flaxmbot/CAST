"""
CAST-G Benchmarker — Honest, Standard Metrics.

Reports:
1. Bits-per-byte (BPB) — THE standard metric for byte-level models
2. Raw throughput (bytes/second) — labeled honestly
3. Segment analysis — what the hierarchy actually learns
4. Standard datasets (enwik8, text8) for comparability

Reference BPB values on enwik8:
- Random (uniform): 8.00 BPB
- Compress (gzip):  ~2.90 BPB
- BLT 1B:          ~1.20 BPB
- BLT 8B:          ~1.00 BPB
- EvaByte 6.5B:    ~1.10 BPB
"""
import torch
import time
import os
import math
import argparse
from cast_g.model import CASTGModel
from cast_g.config import get_config
from token_model import TokenModel
from datasets import load_byte_dataset, get_batch, estimate_bpb

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."


def print_model_specs(name: str, model: torch.nn.Module):
    """Print honest model specifications."""
    if hasattr(model, 'count_parameters'):
        counts = model.count_parameters()
        print(f"\n📊 {name} SPECS:")
        for component, n in counts.items():
            if component != 'total':
                print(f"  • {component}: {n:,} params")
        print(f"  • TOTAL: {counts['total']:,} params")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\n📊 {name} SPECS:")
        print(f"  • Total Parameters: {total:,}")
    print("-" * 40)


def evaluate_bpb(model, data, config, device, n_eval_steps=100, label=""):
    """
    Evaluate bits-per-byte on a dataset.
    
    This is the standard metric — directly comparable to published results.
    """
    model.eval()
    total_loss = 0.0
    n_steps = 0
    block_size = config.get('block_size', 1024)
    batch_size = config.get('batch_size', 16)
    
    with torch.no_grad():
        for i in range(n_eval_steps):
            xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            
            output = model(xb, yb)
            if isinstance(output, tuple) and len(output) == 3:
                _, loss, _ = output
            else:
                _, loss = output
            
            if loss is not None:
                total_loss += loss.item()
                n_steps += 1
            
            if i % 20 == 0:
                current_bpb = estimate_bpb(total_loss / max(1, n_steps))
                print(f"  {label} Eval step {i:3d} | Running BPB: {current_bpb:.4f}")
    
    model.train()
    
    avg_loss = total_loss / max(1, n_steps)
    bpb = estimate_bpb(avg_loss)
    return bpb, avg_loss


def measure_throughput(model, data, config, device, n_steps=50):
    """
    Measure raw throughput in bytes per second.
    
    Labeled honestly: this measures how fast the model processes bytes,
    NOT how well it understands them.
    """
    model.eval()
    block_size = config.get('block_size', 1024)
    batch_size = config.get('batch_size', 16)
    
    # Warmup
    for _ in range(5):
        xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
        with torch.no_grad():
            model(xb, yb)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
            model(xb, yb)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    duration = time.time() - start
    total_bytes = n_steps * batch_size * block_size
    throughput = total_bytes / duration
    
    model.train()
    return throughput, duration


def analyze_segments(model, data, config, device):
    """
    Visualize what the hierarchical segmenter actually learns.
    Shows example segmentations at each level.
    """
    if not hasattr(model, 'hierarchy'):
        print("  (Baseline model — no segmentation)")
        return
    
    model.eval()
    block_size = config.get('block_size', 1024)
    
    # Get a sample
    x = data[:block_size].unsqueeze(0).to(device)
    
    with torch.no_grad():
        h_bytes = model.encoder(x)
        level_segments, level_boundaries, level_segment_ids, _ = \
            model.hierarchy(h_bytes, temp=0.1, hard=True)
    
    # Decode bytes for display
    byte_text = bytes(x[0].cpu().tolist()).decode('utf-8', errors='replace')[:200]
    
    print(f"\n🔬 SEGMENT ANALYSIS (first 200 bytes):")
    print(f"  Text: {byte_text[:80]}...")
    
    for level_idx, boundaries in enumerate(level_boundaries):
        n_boundaries = boundaries[0].sum().item()
        T_level = boundaries.size(1)
        avg_len = T_level / max(1, n_boundaries)
        print(f"  Level {level_idx}: {int(n_boundaries)} segments, avg {avg_len:.1f} units/segment")
    
    model.train()


def run_benchmark(dataset_name: str, config_name: str = 'small'):
    """Run full benchmark on a dataset."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = get_config(config_name)
    
    print(f"\n{'='*60}")
    print(f"  CAST-G v3 BENCHMARK — {dataset_name.upper()} ({config_name})")
    print(f"{'='*60}")
    
    # Load data
    print(f"\n>>> Loading {dataset_name} dataset...")
    if dataset_name in ('enwik8', 'text8'):
        train_data = load_byte_dataset(dataset_name, split='train')
        test_data = load_byte_dataset(dataset_name, split='test')
    else:
        train_data = load_byte_dataset(dataset_name)
        test_data = train_data  # No standard split for multilingual
    
    print(f"  Train: {len(train_data):,} bytes | Test: {len(test_data):,} bytes")
    
    # Initialize models
    cast_model = CASTGModel(config=config_name).to(device)
    base_model = TokenModel(
        vocab_size=256,
        d_model=config['d_model'],
        n_layer=config.get('global_n_layer', 4),
        n_head=config['n_head'],
        block_size=config['block_size'],
    ).to(device)
    
    print_model_specs("CAST-G v3", cast_model)
    print_model_specs("Baseline (Token)", base_model)
    
    # Load weights if available
    for name, model, prefix in [
        ("CAST-G", cast_model, "cast_g"),
        ("Baseline", base_model, "baseline"),
    ]:
        weight_file = os.path.join(OUTPUT_DIR, f"{prefix}_{dataset_name}_production.pt")
        if os.path.exists(weight_file):
            print(f">>> Loading weights: {weight_file}")
            state = torch.load(weight_file, map_location=device)
            model.load_state_dict(state, strict=False)
        else:
            print(f"⚠️ No weights for {name}. Using random init.")
    
    # Compile for speed
    if hasattr(torch, 'compile'):
        print("⚡ Compiling models (mode='reduce-overhead', dynamic=True)...")
        try:
            cast_model = torch.compile(cast_model, mode='reduce-overhead', dynamic=True)
        except Exception as e:
            print(f"  CAST-G compile failed (using eager): {e}")
        try:
            base_model = torch.compile(base_model, mode='reduce-overhead', dynamic=True)
        except Exception as e:
            print(f"  Baseline compile failed (using eager): {e}")
    
    # Evaluate BPB (the honest metric)
    print(f"\n>>> BITS-PER-BYTE EVALUATION ({dataset_name}):")
    cast_bpb, cast_loss = evaluate_bpb(cast_model, test_data, config, device, label="CAST-G")
    base_bpb, base_loss = evaluate_bpb(base_model, test_data, config, device, label="Baseline")
    
    # Throughput
    print(f"\n>>> THROUGHPUT MEASUREMENT:")
    cast_tput, cast_dur = measure_throughput(cast_model, test_data, config, device)
    base_tput, base_dur = measure_throughput(base_model, test_data, config, device)
    
    # Segment analysis
    print(f"\n>>> SEGMENT ANALYSIS:")
    m = cast_model._orig_mod if hasattr(cast_model, '_orig_mod') else cast_model
    analyze_segments(m, test_data, config, device)
    
    # Generation test
    print(f"\n>>> GENERATION TEST:")
    prompt = test_data[:config['block_size']].unsqueeze(0)
    for name, model in [("CAST-G", cast_model), ("Baseline", base_model)]:
        try:
            generated = model.generate(prompt, max_new_tokens=100)
            new_tokens = generated[0, config['block_size']:]
            text = bytes(new_tokens.cpu().tolist()).decode('utf-8', errors='replace')
            print(f"  {name}: {text[:100]}")
        except Exception as e:
            print(f"  {name}: Generation failed — {e}")
    
    # Final honest results table
    print(f"\n{'='*60}")
    print(f"  📊 HONEST RESULTS — {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} | {'Baseline':<15} | {'CAST-G v3':<15}")
    print(f"{'-'*60}")
    print(f"{'Bits-per-Byte (BPB)':<25} | {base_bpb:<15.4f} | {cast_bpb:<15.4f}")
    print(f"{'Raw CE Loss':<25} | {base_loss:<15.4f} | {cast_loss:<15.4f}")
    print(f"{'Throughput (B/s)':<25} | {base_tput:<15,.0f} | {cast_tput:<15,.0f}")
    print(f"{'='*60}")
    
    if cast_bpb < base_bpb:
        print(f"  ✅ CAST-G achieves {base_bpb - cast_bpb:.4f} BPB improvement")
    else:
        print(f"  ⚠️ Baseline wins by {cast_bpb - base_bpb:.4f} BPB (model needs more training)")
    
    print(f"\n  Reference: BLT-8B achieves ~1.0 BPB on enwik8")
    print(f"  Reference: Random baseline is 8.0 BPB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAST-G v3 Benchmarker")
    parser.add_argument("--dataset", type=str, default="enwik8",
                       choices=["enwik8", "text8", "en", "hi", "ja", "zh"],
                       help="Dataset to benchmark on")
    parser.add_argument("--config", type=str, default="small",
                       choices=["small", "medium"],
                       help="Model configuration")
    args = parser.parse_args()
    
    run_benchmark(args.dataset, args.config)
