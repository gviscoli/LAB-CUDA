# LAB 04 — ML Kernels

Kernel fondamentali del deep learning implementati a diversi livelli di astrazione: da kernel CUDA custom con Numba, alle librerie ottimizzate NVIDIA (cuDNN, Flash Attention). Analisi delle prestazioni rispetto al picco teorico dell'hardware.

---

## Algoritmi

### 1. MatMul con Tiling e Shared Memory (Numba CUDA)
- **Dimensione**: matrici 2048 × 2048 (FP32)
- **Tile size**: 16 × 16 elementi per blocco
- **Tecnica**: shared memory tiling — ogni blocco carica un tile dalla DRAM una sola volta, poi riusa i dati dalla SRAM on-chip per tutti i calcoli del tile
- **Picco teorico RTX 4080**: 82.6 TFLOPS (FP32)
- **Metrica**: TFLOPS raggiunti, efficienza rispetto al picco
- **Confronto**: versione naïve (accesso globale) vs tiled (shared memory)
- **Applicazioni**: forward/backward pass di layer densi, weight update

### 2. Convoluzione 2D (PyTorch + cuDNN)
- **Input**: batch=16, canali in/out=64, immagini 224×224, kernel 3×3
- **Backend**: PyTorch chiama cuDNN automaticamente su GPU
- **cuDNN**: libreria NVIDIA ottimizzata per Conv2D (Winograd, FFT, direct)
- **Applicazioni**: CNN per image recognition (ResNet, EfficientNet, YOLO)

### 3. Self-Attention (Transformer)
- **Batch**: 8 sequenze
- **Sequence length**: 1024 token
- **Modello**: d_model = 512, 8 attention heads
- **Backend**: `torch.nn.MultiheadAttention`
- **Flash Attention**: verificata la disponibilità su RTX 4080 Ada (SDPA fused)
  - Flash Attention riduce la complessità da O(n²) a O(n) in memoria
  - Fonde le operazioni QK^T, softmax, ·V in un unico kernel
- **Applicazioni**: LLM (GPT, LLaMA), Vision Transformer, encoder-decoder

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab04-ml-kernels/src/run_kernels.py
```

---

## Output atteso

```
[MatMul naïve]       GPU:  320 ms  |  12.4 TFLOPS  |  15% peak
[MatMul tiled]       GPU:   95 ms  |  41.8 TFLOPS  |  51% peak
[Conv2D cuDNN]       GPU:   28 ms  |  cuDNN: Winograd
[Self-Attention]     GPU:   62 ms  |  Flash Attention: YES
```

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Shared memory tiling | Carica tile in SRAM (48 KB/SM) per riusare dati e ridurre accessi DRAM |
| cuDNN | Libreria NVIDIA con implementazioni hand-tuned di Conv2D (Winograd, FFT) |
| Flash Attention | Kernel fused per self-attention: O(n) memoria invece di O(n²) |
| TFLOPS efficiency | Rapporto tra TFLOPS misurati e picco teorico dell'hardware |
| Tensor Cores | Unità hardware specializzate per GEMM FP16/BF16 (304 su RTX 4080) |

---

## Confronto livelli di astrazione

```
Astrazione alta   PyTorch nn.MultiheadAttention → cuDNN / Flash Attention
                  PyTorch nn.Conv2d              → cuDNN backend
                  cupy.matmul                    → cuBLAS

Astrazione bassa  Numba @cuda.jit tiled matmul   → controllo diretto dei registri
```

---

## Tecnologie

- **Numba CUDA** — kernel MatMul custom con `@cuda.jit` e shared memory
- **PyTorch** — Conv2D e Self-Attention via cuDNN/Flash Attention
- **CuPy** — operazioni dense su GPU come baseline
- **CUDA / cuDNN** — librerie NVIDIA sottostanti
