"""
CAST-G Dataset Loaders — Standard Benchmarks + Multilingual.

Provides loaders for:
1. enwik8 — 100MB Wikipedia XML (THE standard byte-level benchmark)
2. text8 — 100MB clean Wikipedia text  
3. TinyStories / multilingual narrative datasets

Standard splits follow published conventions:
- enwik8: first 90M train, next 5M val, last 5M test
- text8: first 90M train, next 5M val, last 5M test
"""
import os
import sys
import torch
import zipfile
from typing import Tuple, Optional

DATA_DIR = "data"


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_enwik8() -> str:
    """
    Download enwik8 dataset (100MB of Wikipedia XML).
    Standard byte-level benchmark used by BLT, MegaByte, etc.
    """
    _ensure_dir()
    path = os.path.join(DATA_DIR, "enwik8")
    
    if os.path.exists(path):
        return path
    
    zip_path = os.path.join(DATA_DIR, "enwik8.zip")
    
    if not os.path.exists(zip_path):
        print(">>> Downloading enwik8 (100MB)...")
        import urllib.request
        url = "http://mattmahoney.net/dc/enwik8.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Downloaded.")
    
    print(">>> Extracting enwik8...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print(f"✅ enwik8 ready at {path}")
    return path


def download_text8() -> str:
    """
    Download text8 dataset (100MB of clean Wikipedia text).
    """
    _ensure_dir()
    path = os.path.join(DATA_DIR, "text8")
    
    if os.path.exists(path):
        return path
    
    zip_path = os.path.join(DATA_DIR, "text8.zip")
    
    if not os.path.exists(zip_path):
        print(">>> Downloading text8 (100MB)...")
        import urllib.request
        url = "http://mattmahoney.net/dc/text8.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Downloaded.")
    
    print(">>> Extracting text8...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DATA_DIR)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    print(f"✅ text8 ready at {path}")
    return path


def load_byte_dataset(
    name: str = "enwik8",
    split: str = "train"
) -> torch.Tensor:
    """
    Load a byte-level dataset with standard train/val/test splits.
    
    Args:
        name: 'enwik8', 'text8', or a language code ('en', 'hi', 'ja', 'zh')
        split: 'train', 'val', or 'test'
        
    Returns:
        data: 1D LongTensor of byte values
    """
    if name == "enwik8":
        path = download_enwik8()
        with open(path, 'rb') as f:
            raw = f.read()
        data = torch.tensor(list(raw), dtype=torch.long)
        
        # Standard splits: 90M/5M/5M
        if split == "train":
            return data[:90_000_000]
        elif split == "val":
            return data[90_000_000:95_000_000]
        else:
            return data[95_000_000:100_000_000]
    
    elif name == "text8":
        path = download_text8()
        with open(path, 'rb') as f:
            raw = f.read()
        data = torch.tensor(list(raw), dtype=torch.long)
        
        # Standard splits: 90M/5M/5M
        if split == "train":
            return data[:90_000_000]
        elif split == "val":
            return data[90_000_000:95_000_000]
        else:
            return data[95_000_000:100_000_000]
    
    else:
        # Multilingual: use existing download logic
        return _load_multilingual(name)


def _load_multilingual(lang_code: str) -> torch.Tensor:
    """Load multilingual datasets using HuggingFace."""
    _ensure_dir()
    path = os.path.join(DATA_DIR, f"data_{lang_code}.txt")
    
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        from datasets import load_dataset
        
        print(f">>> Downloading {lang_code.upper()} dataset...")
        text = ""
        
        if lang_code == "en":
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            for i, item in enumerate(ds):
                text += item["text"] + "\n\n"
                if i > 5000: break
        elif lang_code == "hi":
            ds = load_dataset("OmAlve/TinyStories-Hindi", split="train", streaming=True)
            for i, item in enumerate(ds):
                text += item["translated"] + "\n\n"
                if i > 5000: break
        elif lang_code == "ja":
            ds = load_dataset("wikimedia/wikipedia", "20231101.ja", split="train", streaming=True)
            for i, item in enumerate(ds):
                text += item["text"] + "\n\n"
                if i > 1000: break
        elif lang_code == "zh":
            ds = load_dataset("wikimedia/wikipedia", "20231101.zh", split="train", streaming=True)
            for i, item in enumerate(ds):
                text += item["text"] + "\n\n"
                if i > 1000: break
        else:
            raise ValueError(f"Unknown language: {lang_code}")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Saved to {path}")
    
    encoded = text.encode('utf-8')
    return torch.tensor(list(encoded), dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a random batch of byte sequences.
    
    Args:
        data: 1D LongTensor of byte values
        batch_size: number of sequences per batch
        block_size: length of each sequence
        device: target device (None = keep on CPU)
        
    Returns:
        x: [batch_size, block_size] — input bytes
        y: [batch_size, block_size] — target bytes (shifted by 1)
    """
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        max_start = 1
    
    ix = torch.randint(max_start, (batch_size,))
    offsets = torch.arange(block_size)
    indices = ix.unsqueeze(1) + offsets.unsqueeze(0)
    
    x = data[indices]
    y = data[indices + 1]
    
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    
    return x, y


def estimate_bpb(loss: float) -> float:
    """
    Convert cross-entropy loss to bits-per-byte (BPB).
    
    BPB = CE_loss / ln(2)
    
    Reference values:
    - Random (uniform 256): 8.0 BPB
    - Good byte model (BLT 8B): ~1.0 BPB on enwik8
    - Decent small model: ~1.5-2.5 BPB on enwik8
    """
    import math
    return loss / math.log(2)
