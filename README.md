# CAST-G v3: Token-Agnostic Neural Architecture

> **Hierarchical MI-Segmented Architecture with Mixture-of-Depths Routing**

CAST-G (Compress–Attend–Synthesize Transformer, Generative) is a research architecture for byte-level language modeling that eliminates tokenizers like BPE, WordPiece, and SentencePiece. Instead of a fixed vocabulary, CAST-G learns to dynamically segment raw bytes into variable-length segments using information-theoretic criteria.

---

## Novel Contributions

CAST-G v3 introduces three unpublished innovations over prior art (BLT, Nawrot et al., MegaByte, EvaByte):

### 1. Mutual Information-Driven Boundary Detection
Unlike Nawrot (2023) who uses Gumbel-Sigmoid with a heuristic Lagrangian, and BLT (2024) which uses entropy-based patching, CAST-G places segment boundaries where the **mutual information between left and right context windows drops sharply**. This is theoretically grounded in information theory and produces linguistically meaningful boundaries without supervision.

**Key difference from prior art**: MI estimates semantic coherence between adjacent spans, producing boundaries that correlate with morpheme, word, and phrase structure. Nawrot's approach is purely heuristic (learned binary classifier with a Lagrangian constraint on count). BLT's entropy-based patching only considers the predictability of individual bytes, not the relationship between adjacent contexts.

### 2. Hierarchical Multi-Scale Segmentation
CAST-G simultaneously learns **three levels** of segmentation:
- **Level 0 (Fine)**: ~4-8 byte segments (morpheme/syllable scale)
- **Level 1 (Medium)**: ~16-32 byte segments (word/compound scale)
- **Level 2 (Coarse)**: ~48-96 byte segments (phrase/clause scale)

Cross-level attention enables bidirectional information flow: coarse planning guides fine execution, while fine details inform coarse decisions. **No published work does multi-level dynamic segmentation end-to-end.**

### 3. Mixture-of-Depths Segment Routing (MoD-S)
Adapting Raposo et al. (2024), CAST-G applies Mixture-of-Depths to **dynamically-segmented byte representations**. Trivial segments (articles, spaces, common words) skip Transformer layers via residual bypass. Complex segments (rare words, code, names) get full attention processing. This creates two orthogonal efficiency axes: dynamic segmentation reduces sequence length, and MoD routing reduces compute per remaining segment.

---

## Architecture

```
Raw Bytes [B, T]
    │
    ▼
ByteEncoder (MultiScale Conv + Parallel LRU)
    │                                      [B, T/4, D]
    ▼
HierarchicalSegmenter (3 MI-Boundary Levels + Cross-Level Attention)
    │                                      Level 0: [B, S₀, D]
    │                                      Level 1: [B, S₁, D]
    │                                      Level 2: [B, S₂, D]
    ▼
MoD-TransformerStack (Causal SDPA + Per-Layer Routing)
    │                                      [B, S₀, D] (subset computed)
    ▼
AutoregressiveLocalDecoder (Cross-Attention + Causal Self-Attention)
    │                                      [B, S₀ × 8, 256]
    ▼
Byte Logits [B, T, 256]
```

---

## Components & Prior Art Acknowledgment

| Component | Our Version | Prior Art | What's Different |
|:---|:---|:---|:---|
| Byte Embedding | 256-entry table | ByT5, BLT, EvaByte | Same (standard) |
| Local Encoder | Multi-scale Conv + Parallel LRU | BLT (cross-attn), Mamba (SSM) | Multi-kernel fusion, parallel scan |
| Boundary Detection | **MI-driven** (InfoNCE) | Nawrot (Gumbel), BLT (entropy) | **Novel**: information-theoretic, not heuristic |
| Segmentation | **3-level hierarchy** | All prior: single level | **Novel**: fractal compression hierarchy |
| Lagrangian | **Adaptive dual-variable** | Nawrot (fixed λ) | Proper optimization, not heuristic |
| Global Reasoning | **MoD-Transformer** | BLT/MegaByte (full) | **Novel**: segment-level MoD routing |
| Local Decoder | AR cross-attn Transformer | MegaByte (AR Transformer) | Similar approach (not claimed as novel) |

---

## Metrics

CAST-G reports standard metrics for scientific comparability:

- **Bits-per-byte (BPB)**: `CE_loss / ln(2)` — the standard for byte-level models
- **Throughput**: Bytes/second — labeled as raw processing speed, not quality
- **Segment statistics**: Actual learned segment lengths per hierarchy level

### Reference BPB on enwik8

| Model | Params | BPB |
|:---|:---|:---|
| Random (uniform 256) | — | 8.00 |
| gzip compression | — | ~2.90 |
| BLT 1B | 1B | ~1.20 |
| BLT 8B | 8B | ~1.00 |
| EvaByte 6.5B | 6.5B | ~1.10 |
| **CAST-G Small** | ~5M | TBD |
| **CAST-G Medium** | ~50M | TBD |

> **Note**: CAST-G is a research prototype. It is not competitive with industrial-scale models (BLT at 8B, EvaByte at 6.5B) in absolute quality. Its contribution is architectural novelty, not scale.

---

## Usage

```bash
# Interactive mode
python manager.py

# Train on enwik8 (standard benchmark)
python manager.py --mode train --dataset enwik8 --config small

# Run benchmarks
python manager.py --mode bench --dataset enwik8

# Full suite (train + benchmark on enwik8 and text8)
python manager.py --mode all
```

---

## Configuration

| Setting | Small (~5M) | Medium (~50M) |
|:---|:---|:---|
| d_model | 256 | 512 |
| n_layers | 4 | 8 |
| n_heads | 8 | 8 |
| MoD capacity | 50% | 60% |
| Hierarchy levels | 3 | 3 |
| Decoder layers | 2 | 3 |
| Block size | 1024 | 1024 |

---

## Citations

This work builds on and acknowledges the following papers:

- **BLT**: Meta, 2024. "Byte Latent Transformer: Patches Scale Better Than Tokens"
- **Nawrot et al.**: 2023. "Efficient Transformers with Dynamic Token Pooling"
- **MegaByte**: Meta, 2023. "Predicting Million-Byte Sequences with Multiscale Transformers"
- **EvaByte**: 2025. "An Efficient Byte-Level Language Model"
- **ByT5**: Google, 2022. "ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models"
- **Raposo et al.**: 2024. "Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based LMs"
- **van den Oord et al.**: 2018. "Representation Learning with Contrastive Predictive Coding" (InfoNCE)
- **Orvieto et al.**: 2023. "Resurrecting Recurrent Neural Networks for Long Sequences" (LRU)

---

## License

Research use only. Not for production deployment.
