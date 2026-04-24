import os
import sys
import torch
import argparse
from datasets import load_dataset
from cast_g.model import CASTGModel
from benchmarker import get_batch

# Global Configuration
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

def setup():
    # Fix Windows console encoding issues
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    print("🚀 INITIALIZING CAST-G PRODUCTION SUITE...")
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected. Please go to Runtime -> Change Runtime Type -> T4 GPU.")
        return False
    
    print(f"✅ {torch.cuda.device_count()} GPU(s) detected.")
    
    # [OPTIMIZATION]: Speed up matmuls on T4/A100
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
        
    # Force path injection
    sys.path.append(os.path.abspath('.'))
    os.makedirs("data", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return True

def print_model_specs(name, model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 {name} ARCHITECTURE SPECS:")
    print(f"  • Total Parameters: {total_params:,}")
    print(f"  • Trainable Params: {trainable_params:,}")
    
    # Introspect for specific CAST-G components
    m = model.module if hasattr(model, 'module') else model
    if hasattr(m, 'encoder'):
        print(f"  • Type: Byte-Level Modular (CAST-G)")
        print(f"  • Stride Reduction: 4x (ConvStem)")
        print(f"  • Global Reasoning: {m.global_stack.transformer.layers[0].self_attn.num_heads} Heads, {len(m.global_stack.transformer.layers)} Layers")
        print(f"  • Latent Dim: {m.encoder.embed.embedding_dim}")
    else:
        print(f"  • Type: Discrete Token Baseline")
        print(f"  • Vocabulary: 256 (Raw Bytes)")
        print(f"  • Block Size: {m.block_size}")
        print(f"  • Embedding Dim: {m.token_embedding.embedding_dim}")
    print(f"  • Device Alignment: {next(model.parameters()).device}")
    print("-" * 30)

def download_data(lang_code):
    path = f"data/data_{lang_code}.txt"
    if os.path.exists(path):
        print(f"📦 Found cached dataset: {path}")
        return path

    print(f">>> Downloading {lang_code.upper()} dataset...")
    text = ""
    
    if lang_code == "en":
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        for i, item in enumerate(ds):
            text += item["text"] + "\n\n"
            if i > 5000: break
    elif lang_code == "hi":
        # New high-quality Hindi Stories
        ds = load_dataset("OmAlve/TinyStories-Hindi", split="train", streaming=True)
        for i, item in enumerate(ds):
            text += item["translated"] + "\n\n"
            if i > 5000: break
    elif lang_code == "ja":
        # Japanese Wikipedia Subset
        ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
        for i, item in enumerate(ds):
            text += item["text"] + "\n\n"
            if i > 1000: break # Wikipedia items are longer
    elif lang_code == "zh":
        # Chinese Wikipedia Subset
        ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
        for i, item in enumerate(ds):
            text += item["text"] + "\n\n"
            if i > 1000: break
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"✅ Saved to {path}")
    return path

def train(lang_code, data_path, steps, batch_size=64):
    device = 'cuda'
    from token_model import TokenModel
    
    # Initialize both models
    cast_model = CASTGModel(d_model=256, n_layer=4, n_head=8).to(device)
    base_model = TokenModel(vocab_size=256, d_model=256, n_layer=4, n_head=8, block_size=1024).to(device)
    
    # Report Specs
    print_model_specs("CAST-G (Killer)", cast_model)
    print_model_specs("Token-Baseline", base_model)
    
    # Multi-GPU support (Apply first, then compile)
    if torch.cuda.device_count() > 1:
        print(f"⚡ Multi-GPU detected! Wrapping models in DataParallel.")
        cast_model = torch.nn.DataParallel(cast_model)
        base_model = torch.nn.DataParallel(base_model)
        batch_size = batch_size * torch.cuda.device_count()
        print(f"📈 Scaled effective batch size to {batch_size}")

    # [OPTIMIZATION]: Wrap in torch.compile for massive speedup (PyTorch 2.0+)
    # We compile the wrapped model to ensure all replicas are optimized correctly
    if hasattr(torch, 'compile'):
        print("⚡ Compiling models for maximum performance...")
        cast_model = torch.compile(cast_model)
        base_model = torch.compile(base_model)
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.long)

    # 1. TRAIN CAST-G
    print(f"\n🔥 [1/2] PROCESSING CAST-G ({lang_code})...")
    cast_save = os.path.join(OUTPUT_DIR, f"cast_g_{lang_code}_production.pt")
    run_loop(cast_model, data, steps, device, cast_save, batch_size, show_seg=True)

    # 2. TRAIN BASELINE
    print(f"\n🔥 [2/2] PROCESSING BASELINE ({lang_code})...")
    base_save = os.path.join(OUTPUT_DIR, f"baseline_{lang_code}_production.pt")
    run_loop(base_model, data, steps, device, base_save, batch_size, show_seg=False)

def unwrap_model(model):
    """Deeply unwraps a model from DataParallel and torch.compile wrappers."""
    m = model
    # Unwrap DataParallel
    if hasattr(m, 'module'):
        m = m.module
    # Unwrap torch.compile (OptimizedModule)
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    # Handle the case where they are nested the other way
    if hasattr(m, 'module'):
        m = m.module
    return m

def run_loop(model, data, steps, device, save_path, batch_size, show_seg=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # [RESUME LOGIC]: Load checkpoint if it exists
    start_step = 0
    ckpt_path = save_path + ".ckpt"
    if os.path.exists(ckpt_path):
        print(f"📦 Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        start_step = checkpoint['step']
        m = unwrap_model(model)
        m.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if start_step >= steps:
            print(f"✅ Model already fully trained ({start_step}/{steps} steps). Skipping.")
            return

    model.train()
    for step in range(start_step, steps):
        xb, yb = get_batch(data, batch_size=batch_size, block_size=1024)
        xb, yb = xb.to(device), yb.to(device)
        
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
            avg_seg = metrics.get('avg_seg_len', 0.0)
            if torch.is_tensor(avg_seg): avg_seg = avg_seg.mean().item()
            seg_info = f" | Seg: {avg_seg:.2f} bytes" if show_seg else ""
            print(f"  Step {step:5d} | Loss: {actual_loss.item():.4f}{seg_info}")

        # [PROGRESS SAVING]: Save checkpoint every 500 steps
        if (step + 1) % 500 == 0:
            m = unwrap_model(model)
            ckpt = {
                'step': step + 1,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(ckpt, ckpt_path)
            print(f"💾 Checkpoint saved at step {step+1}")

    # Final Weights (Production Format)
    m = unwrap_model(model)
    torch.save(m.state_dict(), save_path)
    print(f"✅ Production weights saved to {save_path}")
    if os.path.exists(ckpt_path): os.remove(ckpt_path) # Cleanup

def main():
    if not setup(): return
    
    parser = argparse.ArgumentParser(description="CAST-G Production Manager")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "train", "bench", "all"])
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi", "ja", "zh"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Auto-switch to 'all' if not a TTY (Kaggle background/non-interactive)
    if not sys.stdin.isatty() and args.mode == "interactive":
        print("🤖 Non-interactive environment detected. Switching to --mode all")
        args.mode = "all"

    if args.mode == "interactive":
        print("\n--- CAST-G INTERACTIVE MANAGER ---")
        print("1. Train English (TinyStories)")
        print("2. Train Hindi (HindiStories)")
        print("3. Run Official Benchmarks")
        print("4. Train ALL (EN, HI, JA, ZH) + Bench (Full Suite)")
        print("5. Exit")
        
        choice = input("\nSelect an option (1-5): ")
        if choice == "1": args.mode, args.lang = "train", "en"
        elif choice == "2": args.mode, args.lang = "train", "hi"
        elif choice == "3": args.mode = "bench"
        elif choice == "4": args.mode = "all"
        else: return

    if args.mode == "all":
        langs = ["en", "hi", "ja", "zh"]
        for l in langs:
            print(f"\n{'='*50}\n🚀 PROCESSING LANGUAGE: {l.upper()}\n{'='*50}")
            data_path = download_data(l)
            train(l, data_path, args.steps, args.batch_size)
            print(f"📈 Benchmarking {l}...")
            os.system(f"python benchmarker.py --lang {l}")
        print("\n🏆 FULL SUITE COMPLETE.")
        
    elif args.mode == "train":
        data_path = download_data(args.lang)
        train(args.lang, data_path, args.steps, args.batch_size)
        
    elif args.mode == "bench":
        if args.mode == "interactive":
            args.lang = input("Which language to benchmark? (en/hi/ja/zh): ")
        os.system(f"python benchmarker.py --lang {args.lang}")

if __name__ == "__main__":
    main()
