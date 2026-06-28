# LAB 10 — Image Processing GPU

Accelerazione GPU per algoritmi di image processing. Le operazioni su immagini sono intrinsecamente parallele — ogni pixel può essere processato indipendentemente — e altamente memory-bound, rendendole ideali per la GPU con la sua elevata bandwidth DRAM.

---

## Algoritmi

### 1. Gaussian Blur 2D

- **Immagine**: 4096×4096 float32
- **Sigma**: 3.0 → kernel 7×7
- **CPU**: `scipy.ndimage.gaussian_filter`
- **GPU CuPy**: `cupyx.scipy.ndimage.gaussian_filter`
- **GPU Numba**: kernel separabile 2-pass (row pass + column pass)

**Separabilità del filtro gaussiano:**

Il filtro 2D `G2d = G1d ⊗ G1d` può essere applicato in due pass 1D separati:

```
Pass 1 (righe):  tmp[i,j] = Σ_k G1d[k] * src[i, j+k-3]
Pass 2 (colonne): dst[i,j] = Σ_k G1d[k] * tmp[i+k-3, j]
```

Vantaggio: `O(N × 7)` operazioni invece di `O(N × 49)` → 7× meno operazioni, 7× meno accessi memoria.

**Coefficienti G1d (sigma=3, normalizzati, 7 tap):**
```
[0.1072, 0.1403, 0.1658, 0.1734, 0.1658, 0.1403, 0.1072]
```

### 2. Sobel Edge Detection

- **Immagine**: 4096×4096 float32
- **CPU**: `scipy.ndimage.sobel` su asse x e y, poi `np.hypot`
- **GPU CuPy**: `cupyx.scipy.ndimage.sobel`
- **GPU Numba**: kernel fused che calcola `Gx`, `Gy` e magnitude in un unico pass

**Operatori di Sobel:**

```
Gx = [-1  0 +1]    Gy = [-1 -2 -1]
     [-2  0 +2]         [ 0  0  0]
     [-1  0 +1]         [+1 +2 +1]

Magnitude = sqrt(Gx² + Gy²)
```

Il kernel Numba fused è più efficiente: legge `src` una sola volta per pixel e calcola entrambi i gradienti, eliminando i 2 accessi separati dei pass scipy.

### 3. Bilateral Filter

- **Immagine**: 512×512 float32 (ridotta: complessità O(N²×r²))
- **Raggio**: 5 → finestra 11×11 = 121 operazioni per pixel
- **σ_spaziale**: 3.0 (quanto pesano i pixel vicini)
- **σ_colore**: 0.15 (quanto pesano i pixel con valore simile)
- **CPU**: NumPy patch-based
- **GPU**: CuPy vectorized (stessa logica, tutto su GPU)

**Formula matematica:**

```
BF[p] = (1/Z) Σ_q∈Ω f(||p-q||) · g(|I_p - I_q|) · I_q

f(r) = exp(-r²/2σ_s²)     kernel spaziale gaussiano
g(δ) = exp(-δ²/2σ_c²)     kernel per range (colore)
Z    = Σ_q f(||p-q||) · g(|I_p - I_q|)    normalizzazione
```

Differenza rispetto al Gaussian Blur: il filtro per range `g` preserva i bordi (edge-preserving) perché downweighta i pixel di colore diverso.

### 4. Scaling Analysis

Gaussian blur al variare della risoluzione: 256×256, 512×512, 1024×1024, 2048×2048, 4096×4096.

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab10-imgproc/src/run_imgproc.py
```

Il plot viene salvato in `lab10-imgproc/outputs/lab10_benchmark.png`.

---

## Risultati attesi

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

| Algoritmo | Dimensione | Speedup atteso | Note |
|-----------|-----------|----------------|------|
| Gaussian Blur | 4096×4096 | ~20–60x | Bandwidth-bound, separabile |
| Sobel | 4096×4096 | ~15–40x | Fused kernel elimina un pass |
| Bilateral Filter | 512×512 | ~30–80x | O(N²×r²), molto parallelo |

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Separable filter | Un filtro 2D = due pass 1D: riduce la complessità da O(k²) a O(2k) |
| Halo / ghost cells | Pixel extra caricati in shared memory per evitare accessi non coalescenti al bordo |
| Fused kernel | Combina più operazioni in un unico pass GPU: meno letture di memoria globale |
| Edge-preserving | Proprietà del bilateral filter: non sfoca i bordi, a differenza del Gaussian |
| Bandwidth-bound | Algoritmi limitati dalla velocità della DRAM: beneficiano di GPU per la bandwidth elevata |
| Coalescenza | Accessi consecutivi di thread consecutivi → massima efficienza DRAM |

---

## Tecnologie

- **CuPy / cupyx.scipy.ndimage** — filtri GPU production-quality via cuDNN/CUDA
- **Numba CUDA** — kernel custom separabile (Gaussian) e fused (Sobel)
- **SciPy ndimage** — baseline CPU
- **NumPy** — bilateral filter CPU

---

## Riferimenti

- [CUDA Image Processing](https://docs.nvidia.com/cuda/cuda-samples/index.html#imagedenoising)
- [Bilateral Filter — Tomasi & Manduchi 1998](http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Tomasi98.pdf)
- [GPU Gems 3 — Chapter 40: Incremental Computation of the Gaussian](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-40-incremental-computation-gaussian)
