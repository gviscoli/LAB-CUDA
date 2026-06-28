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

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

### CPU vs GPU — benchmark principale

| Algoritmo | Dimensione | CPU (ms) | GPU (ms) | Speedup |
|-----------|-----------|----------|----------|---------|
| Gaussian Blur (CuPy) | 4096×4096 | 431.96 | 2.23 | **194.0x** |
| Gaussian Blur (Numba) | 4096×4096 | 431.96 | 1.62 | **267.0x** |
| Sobel (CuPy) | 4096×4096 | 583.41 | 60.50 | **9.6x** |
| Sobel (Numba fused) | 4096×4096 | 583.41 | 0.30 | **1924.8x** |
| Bilateral Filter | 512×512 | 273.28 | 11.04 | **24.8x** |

**Nota Sobel**: il kernel CuPy chiama `sobel_x` + `sobel_y` + `hypot` — 3 kernel separati con 3 round-trip DRAM. Il kernel Numba fused legge l'immagine una sola volta e calcola entrambi i gradienti + magnitude in un unico pass → **200× più veloce di CuPy** sulla stessa GPU.

**Nota Gaussian**: il kernel Numba separabile (267x) supera cupyx (194x) perché i due pass 1D (7 operazioni per pixel ciascuno) saturano meglio i warp rispetto al kernel 2D generico di cupyx.

### Scaling Analysis — Gaussian Blur

| Risoluzione | CPU (ms) | GPU (ms) | Speedup | Mem (MB) |
|------------|----------|----------|---------|---------|
| 256×256 | 0.7 | 0.2 | 3.1x | 0.2 |
| 512×512 | 3.1 | 0.5 | 6.3x | 1.0 |
| 1024×1024 | 17.3 | 0.4 | 48.8x | 4.0 |
| 2048×2048 | 105.5 | 1.9 | 56.2x | 16.0 |
| 4096×4096 | 578.2 | 13.8 | 42.0x | 64.0 |

Il break-even CPU/GPU è intorno a **512×512**. Il calo da 56x (2048²) a 42x (4096²) è dovuto alla pressione sulla L2 cache GPU: a 64MB l'immagine non entra in cache e il kernel diventa bandwidth-bound.

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
