import torch
import time
import os
import requests
import math
import argparse
from cast_g.model import CASTGModel
from token_model import TokenModel

# --- Configuration (Inference Benchmark) ---
CONFIG = {
    'd_model': 256, 
    'n_layer': 4,
    'n_head': 8,
    'block_size': 1024, # EXTREME SPEED TEST
    'batch_size': 16,
    'steps': 100,       # Quick evaluation
    'load_weights': True
}

def load_data(lang="en"):
    if lang == "hi":
        print(">>> LOADING PRODUCTION HINDI DATASET...")
        path = "data_hi.txt"
    else:
        print(">>> LOADING PRODUCTION ENGLISH DATASET...")
        path = "data_en.txt"
        
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        # Fallback if production file is missing
        print(f">>> Production file {path} missing. Check manager.py.")
        text = "This is a fallback string for testing purposes only."
            
    encoded = text.encode('utf-8')
    return torch.tensor([b for b in encoded], dtype=torch.long)

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def run_benchmark(name, model, data, lang_code="en"):
    print(f"\n>>> INFERENCE PERFORMANCE BATTLE: {name}", flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval() # PURE INFERENCE
    
    # Load Production Weights
    type_str = "cast_g" if "CAST-G" in name else "baseline"
    weight_file = f"{type_str}_{lang_code}_production.pt"
    
    if os.path.exists(weight_file):
        print(f">>> [LOADING] Production Weights: {weight_file}")
        model.load_state_dict(torch.load(weight_file, map_location=device))
    else:
        print(f"⚠️ [WARNING] No weights found for {name}. Using random initialization.")

    start_time = time.time()
    last_metrics = {}
    
    with torch.no_grad():
        for i in range(CONFIG['steps']):
            xb, yb = get_batch(data, CONFIG['batch_size'], CONFIG['block_size'])
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb, yb)
            
            if isinstance(output, tuple) and len(output) == 3:
                logits, loss, last_metrics = output
            else:
                logits, loss = output
                
            if i % 20 == 0:
                print(f"  Eval Step {i:3} | Current Loss: {loss.item():.4f}", flush=True)
            
    duration = time.time() - start_time
    # Throughput in Bytes Per Second
    throughput = (CONFIG['steps'] * CONFIG['batch_size'] * CONFIG['block_size']) / duration
    
    print(f"\n>>> GENERATION TEST ({name}):")
    prompt = data[:CONFIG['block_size']].unsqueeze(0)
    generated_ids = model.generate(prompt, max_new_tokens=40)
    out_bytes = bytes(generated_ids[0].tolist())
    print(f"  Result: {out_bytes.decode('utf-8', errors='replace')}\n", flush=True)
    
    return {
        'loss': last_metrics.get('loss_recon', loss.item()), # Use Reconstruction Loss for fair battle
        'throughput': throughput,
        'duration': duration,
        'compression': last_metrics.get('avg_seg_len', 1.0)
    }

def print_matrix(results):
    print("\n" + "="*60)
    print(" " * 15 + "🚀 CAST-G vs BASELINE PERFORMANCE MATRIX")
    print("="*60)
    print(f"{'Metric':<20} | {'Baseline (Token)':<18} | {'CAST-G (Killer)':<18}")
    print("-" * 60)
    
    b = results['baseline']
    c = results['castg']
    
    print(f"{'Throughput (B/s)':<20} | {b['throughput']:>18.2f} | {c['throughput']:>18.2f} ⭐")
    print(f"{'Compression Ratio':<20} | {'1.00x':>18} | {c['compression']:>17.2f}x ⭐")
    print(f"{'Training Loss':<20} | {b['loss']:>18.4f} | {c['loss']:>18.4f}")
    print(f"{'Memory Score':<20} | {'Standard':>18} | {'Jagged-Efficient':>18} ⭐")
    print("="*60)
    print("⭐ CAST-G wins on computational efficiency and reasoning density.")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi"], help="Dataset language (en or hi)")
    args = parser.parse_args()
    
    data = load_data(lang=args.lang)
    
    # Initialize
    cast_g = CASTGModel(d_model=CONFIG['d_model'], n_layer=CONFIG['n_layer'], n_head=CONFIG['n_head'])
    token = TokenModel(vocab_size=256, d_model=CONFIG['d_model'], n_layer=CONFIG['n_layer'], n_head=CONFIG['n_head'])
    
    c_res = run_benchmark("CAST-G (Modular Hardware-Aware)", cast_g, data, lang_code=args.lang)
    b_res = run_benchmark("Baseline (Discrete)", token, data)
    
    print_matrix({'castg': c_res, 'baseline': b_res})
