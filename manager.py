import os
import sys
import torch
import argparse
from datasets import load_dataset
from cast_g.model import CASTGModel
from benchmarker import get_batch

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
    # GPU Optimizations
    torch.backends.cudnn.benchmark = True
        
    # Force path injection
    sys.path.append(os.path.abspath('.'))
    os.makedirs("data", exist_ok=True)
    return True

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
            text += item["text"] + "\n\n"
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
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"⚡ Multi-GPU detected! Wrapping models in DataParallel.")
        cast_model = torch.nn.DataParallel(cast_model)
        base_model = torch.nn.DataParallel(base_model)
        # Scale batch size for multiple GPUs
        batch_size = batch_size * torch.cuda.device_count()
        print(f"📈 Scaled effective batch size to {batch_size}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.long)

    # 1. TRAIN CAST-G
    print(f"\n🔥 [1/2] TRAINING CAST-G ({lang_code}) for {steps} steps...")
    run_loop(cast_model, data, steps, device, f"cast_g_{lang_code}_production.pt", batch_size, show_seg=True)

    # 2. TRAIN BASELINE
    print(f"\n🔥 [2/2] TRAINING BASELINE ({lang_code}) for {steps} steps...")
    run_loop(base_model, data, steps, device, f"baseline_{lang_code}_production.pt", batch_size, show_seg=False)

def run_loop(model, data, steps, device, save_path, batch_size, show_seg=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    model.train()
    for step in range(steps):
        xb, yb = get_batch(data, batch_size=batch_size, block_size=1024)
        xb, yb = xb.to(device), yb.to(device)
        
        with torch.amp.autocast('cuda'):
            output = model(xb, yb)
            if isinstance(output, tuple) and len(output) == 3:
                logits, loss, metrics = output
            else:
                logits, loss = output
                metrics = {}
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if step % 100 == 0:
            avg_seg = metrics.get('avg_seg_len', 0.0)
            if torch.is_tensor(avg_seg): avg_seg = avg_seg.item()
            seg_info = f" | Seg: {avg_seg:.2f} bytes" if show_seg else ""
            print(f"  Step {step:5d} | Loss: {loss.item():.4f}{seg_info}")

    # Weights saved as standard state_dict (stripping DataParallel wrapper if present)
    save_obj = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(save_obj, save_path)
    print(f"✅ Weights saved as {save_path}")

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
