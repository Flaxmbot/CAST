"""
CAST-G v3 Production Manager.

Interactive CLI for training and benchmarking the CAST-G architecture.
Supports enwik8/text8 standard benchmarks and multilingual datasets.

Usage:
    python manager.py                     # Interactive mode
    python manager.py --mode train --dataset enwik8
    python manager.py --mode bench --dataset enwik8
    python manager.py --mode all          # Full suite
"""
import os
import sys
import torch
import argparse
import math
from cast_g.model import CASTGModel
from cast_g.config import get_config
from token_model import TokenModel
from datasets import load_byte_dataset, get_batch, estimate_bpb

OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."


def setup():
    """Initialize environment and verify GPU availability."""
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("🚀 INITIALIZING CAST-G v3 PRODUCTION SUITE...")
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected. CAST-G requires CUDA.")
        return False
    
    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  ✅ GPU {i}: {name} ({mem:.1f} GB)")
    
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    import warnings
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
    
    sys.path.append(os.path.abspath('.'))
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return True


def print_model_specs(name, model):
    """Print model architecture details."""
    if hasattr(model, 'count_parameters'):
        counts = model.count_parameters()
        print(f"\n📊 {name}:")
        for comp, n in counts.items():
            print(f"  • {comp}: {n:,}")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\n📊 {name}: {total:,} params")
    print(f"  • Device: {next(model.parameters()).device}")
    print("-" * 40)


def unwrap_model(model):
    """Unwrap DataParallel and torch.compile wrappers."""
    m = model
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def train(dataset_name, config_name, steps, batch_size=None):
    """Train both CAST-G and baseline models."""
    device = 'cuda'
    config = get_config(config_name)
    if batch_size is not None:
        config['batch_size'] = batch_size
    
    # Load data
    print(f"\n>>> Loading {dataset_name} dataset...")
    data = load_byte_dataset(dataset_name, split='train' if dataset_name in ('enwik8', 'text8') else 'train')
    print(f"  Dataset size: {len(data):,} bytes")
    
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
    print_model_specs("Token Baseline", base_model)
    
    # Multi-GPU support
    effective_batch = config['batch_size']
    if torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print(f"⚡ {n_gpus} GPUs — wrapping in DataParallel")
        cast_model = torch.nn.DataParallel(cast_model)
        base_model = torch.nn.DataParallel(base_model)
        effective_batch *= n_gpus
        print(f"  Effective batch size: {effective_batch}")
    
    # Compile
    if hasattr(torch, 'compile'):
        print("⚡ Compiling models...")
        try:
            cast_model = torch.compile(cast_model, mode='reduce-overhead', dynamic=True)
        except Exception as e:
            print(f"  CAST-G compile skip: {e}")
        try:
            base_model = torch.compile(base_model, mode='reduce-overhead', dynamic=True)
        except Exception as e:
            print(f"  Baseline compile skip: {e}")
    
    # Train CAST-G
    print(f"\n🔥 [1/2] TRAINING CAST-G ({dataset_name}, {config_name})...")
    cast_save = os.path.join(OUTPUT_DIR, f"cast_g_{dataset_name}_production.pt")
    _train_loop(cast_model, data, steps, device, cast_save, effective_batch, config, is_cast=True)
    
    # Train Baseline
    print(f"\n🔥 [2/2] TRAINING BASELINE ({dataset_name})...")
    base_save = os.path.join(OUTPUT_DIR, f"baseline_{dataset_name}_production.pt")
    _train_loop(base_model, data, steps, device, base_save, effective_batch, config, is_cast=False)


def _train_loop(model, data, steps, device, save_path, batch_size, config, is_cast=True):
    """Core training loop with checkpointing and BPB reporting."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 3e-4),
        weight_decay=config.get('weight_decay', 0.1),
    )
    scaler = torch.amp.GradScaler('cuda')
    block_size = config['block_size']
    
    # Resume logic
    start_step = 0
    ckpt_path = save_path + ".ckpt"
    
    if os.path.exists(save_path):
        print(f"✅ Production weights exist: {save_path}. Skipping.")
        return
    
    if os.path.exists(ckpt_path):
        print(f"📦 Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        start_step = checkpoint['step']
        m = unwrap_model(model)
        m.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if start_step >= steps:
            print(f"✅ Already trained ({start_step}/{steps}). Skipping.")
            if os.path.exists(ckpt_path): os.remove(ckpt_path)
            return
    
    # Warmup scheduler
    warmup_steps = config.get('warmup_steps', 500)
    
    model.train()
    for step in range(start_step, steps):
        # Learning rate warmup
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = config.get('learning_rate', 3e-4) * lr_scale
        
        xb, yb = get_batch(data, batch_size=batch_size, block_size=block_size, device=device)
        
        with torch.amp.autocast('cuda'):
            output = model(xb, yb, step=step)
            if isinstance(output, tuple) and len(output) == 3:
                logits, loss, metrics = output
            else:
                logits, loss = output
                metrics = {}
        
        optimizer.zero_grad()
        actual_loss = loss.mean()
        scaler.scale(actual_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            bpb = estimate_bpb(actual_loss.item())
            extra = ""
            if is_cast and 'level0_avg_seg_len' in metrics:
                seg_len = metrics['level0_avg_seg_len']
                if torch.is_tensor(seg_len):
                    seg_len = seg_len.mean().item()
                extra = f" | Seg: {seg_len:.1f}b"
                mod_frac = metrics.get('mod_routed_fraction', 0)
                extra += f" | MoD: {mod_frac:.0%}"
            print(f"  Step {step:5d} | Loss: {actual_loss.item():.4f} | BPB: {bpb:.4f}{extra}")
        
        # Checkpoint every 500 steps
        if (step + 1) % 500 == 0:
            m = unwrap_model(model)
            ckpt = {
                'step': step + 1,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(ckpt, ckpt_path)
            print(f"💾 Checkpoint at step {step+1}")
    
    # Save final weights
    m = unwrap_model(model)
    torch.save(m.state_dict(), save_path)
    print(f"✅ Saved to {save_path}")
    if os.path.exists(ckpt_path): os.remove(ckpt_path)


def main():
    if not setup():
        return
    
    parser = argparse.ArgumentParser(description="CAST-G v3 Production Manager")
    parser.add_argument("--mode", type=str, default="interactive",
                       choices=["interactive", "train", "bench", "all"])
    parser.add_argument("--dataset", type=str, default="enwik8",
                       choices=["enwik8", "text8", "en", "hi", "ja", "zh"])
    parser.add_argument("--config", type=str, default="small",
                       choices=["small", "medium"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    # Auto non-interactive on Kaggle
    if not sys.stdin.isatty() and args.mode == "interactive":
        print("🤖 Non-interactive. Switching to --mode all")
        args.mode = "all"

    if args.mode == "interactive":
        print("\n--- CAST-G v3 INTERACTIVE MANAGER ---")
        print("1. Train on enwik8 (Standard Benchmark)")
        print("2. Train on text8 (Standard Benchmark)")
        print("3. Train English (TinyStories)")
        print("4. Train Hindi (HindiStories)")
        print("5. Run Benchmarks")
        print("6. Full Suite (enwik8 + text8 + Bench)")
        print("7. Exit")
        
        choice = input("\nSelect (1-7): ")
        if choice == "1": args.mode, args.dataset = "train", "enwik8"
        elif choice == "2": args.mode, args.dataset = "train", "text8"
        elif choice == "3": args.mode, args.dataset = "train", "en"
        elif choice == "4": args.mode, args.dataset = "train", "hi"
        elif choice == "5": args.mode = "bench"
        elif choice == "6": args.mode = "all"
        else: return

    if args.mode == "all":
        for ds in ["enwik8", "text8"]:
            print(f"\n{'='*60}\n🚀 PROCESSING: {ds.upper()}\n{'='*60}")
            train(ds, args.config, args.steps, args.batch_size)
            print(f"\n📈 Benchmarking {ds}...")
            os.system(f"python benchmarker.py --dataset {ds} --config {args.config}")
        print("\n🏆 FULL SUITE COMPLETE.")
        
    elif args.mode == "train":
        train(args.dataset, args.config, args.steps, args.batch_size)
        
    elif args.mode == "bench":
        os.system(f"python benchmarker.py --dataset {args.dataset} --config {args.config}")


if __name__ == "__main__":
    main()
