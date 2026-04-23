import os
import sys
import torch
from datasets import load_dataset
from cast_g.model import CASTGModel
from benchmarker import get_batch

def setup():
    print("🚀 INITIALIZING CAST-G PRODUCTION SUITE...")
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU detected. Please go to Runtime -> Change Runtime Type -> T4 GPU.")
        return False
    
    # Force path injection
    sys.path.append(os.path.abspath('.'))
    return True

def download_data(choice):
    if choice == "1":
        print(">>> Downloading English TinyStories...")
        # TinyStories is much better for byte-level learning
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        text = ""
        for i, item in enumerate(ds):
            text += item["text"] + "\n\n"
            if i > 5000: break
        path = "data_en.txt"
    else:
        print(">>> Downloading Hindi Stories...")
        # High-quality cleaned stories for better grammar learning
        ds = load_dataset("surajp/hindi_stories", split="train", streaming=True)
        text = ""
        for i, item in enumerate(ds):
            text += item["story"] + "\n\n"
            if i > 2000: break
        path = "data_hi.txt"
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def train(lang_name, data_path, steps):
    device = 'cuda'
    from token_model import TokenModel
    
    # Initialize both models
    cast_model = CASTGModel(d_model=256, n_layer=4, n_head=8).to(device)
    base_model = TokenModel(vocab_size=256, d_model=256, n_layer=4, n_head=8, block_size=1024).to(device)
    
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor([b for b in text.encode('utf-8')], dtype=torch.long)

    # 1. TRAIN CAST-G
    lang_code = "en" if lang_name == "English" else "hi"
    print(f"\n🔥 [1/2] TRAINING CAST-G ({lang_name}) for {steps} steps...")
    run_loop(cast_model, data, steps, device, f"cast_g_{lang_code}_production.pt", show_seg=True)

    # 2. TRAIN BASELINE
    print(f"\n🔥 [2/2] TRAINING BASELINE ({lang_name}) for {steps} steps...")
    run_loop(base_model, data, steps, device, f"baseline_{lang_code}_production.pt", show_seg=False)

def run_loop(model, data, steps, device, save_path, show_seg=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    for step in range(steps):
        xb, yb = get_batch(data, batch_size=32, block_size=1024)
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
            seg_info = f" | Seg: {metrics.get('avg_seg_len', 0.0):.2f} bytes" if show_seg else ""
            print(f"  Step {step:5d} | Loss: {loss.item():.4f}{seg_info}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Weights saved as {save_path}")

def main():
    if not setup(): return
    
    print("\n--- CAST-G INTERACTIVE MANAGER ---")
    print("1. Train English (TinyStories)")
    print("2. Train Hindi (HindiStories)")
    print("3. Run Official Benchmarks")
    print("4. Exit")
    
    choice = input("\nSelect an option (1-4): ")
    
    if choice in ["1", "2"]:
        lang = "English" if choice == "1" else "Hindi"
        steps = int(input(f"How many steps to train {lang}? (Recommended: 10000): "))
        data_path = download_data(choice)
        train(lang, data_path, steps)
    elif choice == "3":
        lang_code = input("Which language to benchmark? (en/hi): ")
        os.system(f"python benchmarker.py --lang {lang_code}")
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
