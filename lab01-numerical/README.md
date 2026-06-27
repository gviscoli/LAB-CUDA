# LAB 01 — Numerical Computing

Benchmark di algoritmi numerici fondamentali: FFT, algebra lineare densa (DGEMM) e operatori stencil su griglia. Confronto sistematico tra CPU (NumPy/SciPy) e GPU (CuPy + Numba CUDA).

---

## Algoritmi

### 1. FFT 1D
- **Dimensione**: 2²² campioni complessi (~4M)
- **CPU**: `numpy.fft.fft`
- **GPU**: `cupy.fft.fft`
- **Applicazioni**: elaborazione del segnale, solver spettrali per PDE

### 2. FFT 2D
- **Dimensione**: immagine 4096 × 4096
- **CPU**: `numpy.fft.fft2`
- **GPU**: `cupy.fft.fft2`
- **Applicazioni**: image processing, simulazioni in dominio spettrale

### 3. MatMul DGEMM (FP64)
- **Dimensione**: matrici 8192 × 8192
- **Picco teorico RTX 4080**: 82.6 TFLOPS (FP32)
- **Metrica**: TFLOPS raggiunti, efficienza rispetto al picco
- **Applicazioni**: deep learning, simulazioni FEM

### 4. Stencil 2D Laplaciano (CuPy)
- **Dimensione**: griglia 4096 × 4096
- **Operatore**: 5 punti — `u[i±1,j] + u[i,j±1] − 4·u[i,j]`
- **Classificazione**: memory-bound (bassa intensità operazionale)
- **Metrica**: bandwidth effettiva vs 716.8 GB/s teorici
- **Applicazioni**: solver differenze finite per PDE ellittiche

### 5. Stencil 2D con Numba CUDA (shared memory)
- **Kernel custom**: blocchi 16 × 16 con halo
- **Ottimizzazione**: riduzione degli accessi DRAM tramite tiling
- **Confronto**: CuPy naïve vs Numba con shared memory
- **Metrica**: bandwidth effettiva, speedup rispetto alla versione CuPy

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab01-numerical/src/run_numerical.py
```

---

## Output atteso

```
[FFT 1D]      CPU: 450 ms  |  GPU:  12 ms  |  Speedup: 37x
[FFT 2D]      CPU: 890 ms  |  GPU:  35 ms  |  Speedup: 25x
[DGEMM]       CPU: 8200 ms |  GPU:  95 ms  |  Speedup: 86x  |  48.3 TFLOPS
[Stencil CuPy]  CPU: 210ms |  GPU:  18 ms  |  BW: 312 GB/s
[Stencil Numba] CPU: 210ms |  GPU:  11 ms  |  BW: 510 GB/s
```

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Roofline model | Classifica gli algoritmi come memory-bound o compute-bound |
| Ridge point | Intensità operazionale soglia: ~115 FLOP/byte (RTX 4080) |
| Shared memory | SRAM on-chip (~100× più veloce della VRAM) usata dal kernel Numba |
| Halo tiling | Tecnica per riusare dati nelle regioni di confine del blocco |

---

## Tecnologie

- **CuPy** — NumPy drop-in su GPU (FFT, array ops)
- **Numba CUDA** — kernel CUDA scritti in Python con `@cuda.jit`
- **NumPy / SciPy** — baseline CPU
