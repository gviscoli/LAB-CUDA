# LAB 13 — Deep Learning Inference GPU

Accelerazione GPU per i kernel fondamentali dell'inferenza di modelli Transformer/LLM. Questi algoritmi sono eseguiti migliaia di volte per ogni token generato — ottimizzarli è la base di framework come TensorRT, vLLM e FlashAttention.

---

## Algoritmi

### 1. INT8 Quantized MatMul

- **Dimensioni**: M=2048, K=4096, N=2048
- **Quantizzazione**: simmetrica per canale, `scale = max(|W|) / 127`
- **Pipeline**: FP32 → INT8 (quantize) → matmul → FP32 (dequantize)
- **Metrica**: TFLOPS e errore di quantizzazione (MAE)

**Schema quantizzazione simmetrica:**

```
scale_w = max(|W|) / 127
W_int8  = round(clip(W / scale_w, -128, 127))

# Dequantizzazione dopo matmul:
out_fp32 = matmul(X_int8, W_int8) * scale_x * scale_w
```

**Vantaggi INT8**: 4× meno memoria DRAM (32bit → 8bit), 2–4× più throughput su Tensor Core, cache L2 più efficiente (4× più pesi in cache a parità di bytes).

**Tensor Cores RTX 4080**: unità hardware dedicate a matmul 16×16×16 in un singolo clock. Supportano FP16, BF16, INT8, INT4. L'INT8 throughput teorico è 2× l'FP16 (165 vs 82 TOPS).

### 2. Batch Normalization

- **Tensore**: (256, 512, 28, 28) float32 — batch tipico di vision model
- **Normalizzazione**: su (N, H, W) per ciascuno dei C=512 canali
- **Parametri apprendibili**: γ (gamma) e β (beta) per canale

**Formula:**

```
μ_c  = mean(X[:, c, :, :])                ← media per canale
σ²_c = var(X[:, c, :, :])                 ← varianza per canale
Y[:, c, :, :] = γ_c · (X - μ_c) / sqrt(σ²_c + ε) + β_c
```

**Differenza con Layer Norm**: BatchNorm normalizza su (N,H,W) per canale; LayerNorm normalizza su (d_model) per token. BatchNorm richiede batch size > 1; LayerNorm funziona anche con batch=1.

### 3. Softmax (Fused Kernel)

- **Input**: (10,000 × 32,000) float32 — simula output layer LLM con vocab=32K
- **CPU**: NumPy vectorized
- **GPU CuPy**: elementwise ops
- **GPU Numba**: kernel per-riga con shared memory (3 pass: max + exp/sum + normalize)

**Softmax numericamente stabile:**

```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

Senza la sottrazione del `max`, `exp(x_i)` overflowa per logits > ~88. Il kernel Numba fonde le 3 fasi in un unico lancio, evitando 2 round-trip su DRAM rispetto all'implementazione naive a 3 kernel separati.

**Riduzione in shared memory** (pattern butterfly):

```
Thread 0: x[0]+x[256]   Thread 1: x[1]+x[257]  ...  Thread 127: x[127]+x[383]
Thread 0: x[0]+x[128]   Thread 1: x[1]+x[129]  ...  Thread 63:  x[63]+x[127]
...fino a Thread 0: somma totale
```

### 4. Layer Normalization

- **Input**: (512,000 × 1024) float32 — simula 1000 batch × 512 seq_len × d_model=1024
- **Normalizzazione**: su d_model (ultima dimensione) per ogni token
- **Kernel Numba**: fused mean+var+normalize in 3 pass, un blocco per riga

**Formula:**

```
μ    = mean(x)                          ← media sugli elementi del token
σ²   = var(x)                           ← varianza
LN(x)_i = γ_i · (x_i - μ) / sqrt(σ² + ε) + β_i
```

### 5. Scaling Analysis

INT8 vs FP32 matmul per M=N=K ∈ {512, 1024, 2048, 4096, 8192}. Confronta TFLOPS effettivi e overhead di quantizzazione.

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab13-inference/src/run_inference.py
```

Il plot viene salvato in `lab13-inference/outputs/lab13_benchmark.png`.

---

## Risultati attesi

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

| Algoritmo | Dimensione | Speedup atteso | Note |
|-----------|-----------|----------------|------|
| MatMul FP32 | 2048³ | ~30–80x | cuBLAS ottimizzato |
| MatMul INT8 | 2048³ | ~50–120x | 4× meno DRAM reads |
| Batch Norm | (256,512,28,28) | ~20–60x | Reduction + elementwise |
| Softmax | 10K×32K | ~15–40x | Memory-bound, 3 pass |
| Layer Norm | 512K×1024 | ~20–50x | Simile a BatchNorm |

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Quantizzazione simmetrica | `W_int8 = round(W / scale)` con scale=max(|W|)/127; nessun zero-point |
| Tensor Cores | Unità HW per matmul 16×16×16 in 1 clock; attivi con tipi FP16/BF16/INT8 |
| Fused kernel | Combina più operazioni elementari in un unico kernel: riduce round-trip DRAM |
| Reduction in shared memory | Pattern butterfly: O(N) work, O(log N) span — fondamentale per sum/max per riga |
| Arithmetic intensity | FLOP/byte: matmul=O(N), softmax=O(1) → matmul più favorevole per GPU |
| Numerically stable softmax | Sottrae max prima di exp() per prevenire overflow float32 |
| BatchNorm vs LayerNorm | BN: normalizza su batch; LN: normalizza su features — LN non dipende dal batch size |

---

## Tecnologie

- **CuPy** — operazioni vettoirizzate GPU (reduction, elementwise, matmul)
- **Numba CUDA** — kernel fused per softmax e layer norm con shared memory
- **NumPy** — baseline CPU
- **Rich** — tabelle formattate e output colorato

---

## Riferimenti

- [FlashAttention — fused attention kernel](https://arxiv.org/abs/2205.14135)
- [NVIDIA TensorRT INT8 Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
- [Softmax — numerically stable implementation](https://cs231n.github.io/linear-classify/#softmax)
- [Layer Normalization — Ba et al. 2016](https://arxiv.org/abs/1607.06450)
