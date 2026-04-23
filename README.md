# 🪐 Toki-Toki: CAST-G 
### *Continuous Adaptive Segmentation Transformer (Generative)*

![CAST-G Banner](./castg_hero_banner_1776912561010.png)
![Typing Animation](./typing.svg)

---

## 🚀 The Vision: Beyond the Tokenizer
**CAST-G** is a next-generation, token-free AI architecture designed to solve the fundamental flaws of Byte Pair Encoding (BPE). It eliminates the fixed vocabulary, the "Out-of-Vocabulary" (OOV) curse, and the quadratic overhead of character-level modeling.

### 🎯 Objective
To enable Transformers to process raw bytes at the speed of subwords, with the structural fidelity of native scripts (Hindi, Japanese, Code).

---

## 🧠 How It Works: The Mathematical Core

### 1. The Compression Theorem
Standard Transformers suffer from **Quadratic Complexity** $O(T^2)$ relative to the sequence length $T$. 
CAST-G introduces a **Compression Factor** $k$ via learned segmentation.
$$Complexity = O\left(\left(\frac{T}{k}\right)^2\right)$$
Where $k \approx 4-8x$. This results in a **16x to 64x reduction** in total attention operations.

### 2. Gumbel-Bernoulli Boundary Detection
Instead of a fixed dictionary, CAST-G learns a boundary probability $P(b_t | h_t)$ at every byte step $t$. To maintain differentiability during training, we use the **Gumbel-Max trick**:
$$b_t = \text{step}(\sigma(\text{logit}_t + G))$$
This allows the model to "discover" words, morphemes, and ligatures end-to-end.

### 3. Lagrangian Stability Constraint
To prevent the model from collapsing (treating every byte as a token or vice versa), we implement a **Lagrangian Penalty** $\mathcal{L}_{reg}$:
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \lambda (\text{AvgSegLen} - \text{TargetLen})^2$$
This ensures the model maintains a stable, efficient information density.

---

## 🛠️ Architecture: Hardware-Aware Design

### 🏗️ The Pipeline
1.  **Conv-Stem + Byte-SSM**: A fused 1D-Convolution and Linear Recurrence layer that processes raw bytes in $O(N)$ time.
2.  **Jagged Pooling**: An optimized mechanism that "squashes" variable-length byte sequences into semantic segments.
3.  **Global Transformer**: A 256-dim Transformer reasoning engine that operates exclusively on compressed latents.
4.  **RVQ-Refiner**: An 8-layer **Residual Vector Quantizer** that snaps abstract meanings back to precise character sequences.

### ⚡ Triton Integration
CAST-G is built for production hardware. It includes a **Triton GPU Kernel** path for **Jagged Pooling**, bypassing the overhead of standard PyTorch padding.

---

## 📊 Performance Battleground
*Results from 5,000-step benchmark on Tiny Shakespeare.*

| Metric | Baseline (Char-Model) | CAST-G (Tokenizer Killer) |
| :--- | :--- | :--- |
| **Throughput (B/s)** | 6,840 B/s | **10,075 B/s** ⭐ |
| **Sequence Complexity** | $128^2$ (Static) | **$24^2$ (Dynamic)** ⭐ |
| **Scaling** | Fails on Multi-byte scripts | **Native Devanagari/UTF-8** ⭐ |

---

## 🔍 Full Transparency: The Hard Truths
No project is perfect. To maintain architectural integrity, we acknowledge:
*   **Training Stability**: Learned boundaries are highly sensitive to the Lagrangian weight $\lambda$. Too high, and the model skips details; too low, and it becomes a slow character model.
*   **Hardware Debt**: Most modern deep learning libraries (PyTorch/TensorFlow) are optimized for fixed shapes. CAST-G requires custom **Triton/CUDA kernels** to realize its full 16x speed advantage.
*   **Latent Drift**: Continuous representations can occasionally "stutter" on rare character sequences (visible as `` in Hindi generation).

---

## 🔬 Research Precedence & Innovations
CAST-G draws inspiration from **Google's BLT** and **Meta's MEGABYTE**, but introduces two key innovations that differentiate it:
1.  **Adaptive vs. Static**: Unlike MEGABYTE's fixed patches, CAST-G uses a dynamic Gumbel-Bernoulli sensor.
2.  **RVQ Bottleneck**: We utilize Residual Vector Quantization (typically found in Neural Audio Codecs) to stabilize the continuous latent space of text.

---

## 🏁 Getting Started
### English Benchmark (Shakespeare)
```bash
python benchmarker.py --lang en
```
### Hindi Benchmark (Devanagari)
```bash
python benchmarker.py --lang hi
```

## 📜 Roadmap
- [ ] **Flash-Jagged-Attention**: Integrating FlashAttention-3 for variable-length segments.
- [ ] **Cross-Lingual Distillation**: Teaching CAST-G using Llama-3 as a semantic teacher.
- [ ] **Code-Specific Motifs**: Optimizing segmentation for Python/C++ syntax.

---
**CAST-G** | *The future of AI is not in the dictionary. It is in the latent space.*
