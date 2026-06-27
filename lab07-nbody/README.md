# LAB 07 — N-Body Gravitational Simulation

Simulazione gravitazionale N-Body su GPU. Il problema N-Body è O(N²) nel numero di interazioni: ogni coppia di corpi si attrae reciprocamente. Con N=4096 si eseguono oltre 16 milioni di calcoli per ogni time step, rendendolo uno dei benchmark più rappresentativi per misurare la potenza di picco (GFLOPS) di una GPU.

---

## Algoritmi

### 1. Brute-Force O(N²) — NumPy vs CuPy

- **N corpi**: 4096 (predefinito)
- **CPU**: NumPy chunked (CHUNK=512) per evitare OOM sulla matrice N×N completa
- **GPU**: CuPy con matrici full N×N — la VRAM RTX 4080 (16 GB) regge agevolmente
- **Softening**: ε² = 1×10⁻⁴ per evitare singolarità a distanza zero
- **Metrica**: GFLOPS effettivi — 26 operazioni floating point per coppia i-j
- **Applicazioni**: astrofisica, simulazioni cosmologiche, dinamica molecolare

### 2. Numba CUDA Tiled — Shared Memory

- **Tile size**: 256 thread (un warp × 8)
- **Pattern**: ogni blocco carica un tile di 256 corpi in shared memory, poi calcola le forze per tutti gli N tile
- **Vantaggio**: riduce gli accessi a global memory di un fattore ~256×
- **Sincronizzazione**: `syncthreads()` dopo il load e dopo il compute di ogni tile
- **JIT**: Numba compila il kernel CUDA in PTX al primo lancio (warm-up necessario)
- **Riferimento**: GPU Gems 3, Chapter 31 — Fast N-Body Simulation with CUDA

### 3. Scaling Analysis — N ∈ {512, 1024, 2048, 4096}

- Misura come lo speedup cresce al crescere di N
- L'N-Body è O(N²): per N grande la GPU scala meglio della CPU perché satura tutti i suoi core
- Output: tabella speedup + grafico PNG

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab07-nbody/src/run_nbody.py
```

Il plot viene salvato automaticamente in `lab07-nbody/outputs/lab07_benchmark.png`.

---

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

### CPU vs GPU — N=4096 corpi

| Algoritmo | CPU (ms) | GPU (ms) | Speedup | GFLOPS |
|-----------|----------|----------|---------|--------|
| N-Body BF (CuPy) | 339.31 | 4.76 | **71.3x** | 91.7 |
| N-Body Numba tiled | 335.92 | 0.42 | **792.9x** | 1029.6 |

Il kernel Numba tiled raggiunge **1 TFLOPS effettivi** grazie al riuso in shared memory — 11× più veloce del kernel CuPy pur operando sulla stessa GPU.

### Scaling Analysis — Speedup vs N

| N | CPU (ms) | GPU (ms) | Speedup | GFLOPS |
|---|----------|----------|---------|--------|
| 512 | 5.4 | 0.33 | 16.1x | 20.4 |
| 1,024 | 21.3 | 0.35 | 61.5x | 78.5 |
| 2,048 | 84.7 | 0.55 | 153.2x | 197.1 |
| 4,096 | 334.6 | 5.00 | 67.0x | 87.3 |

Il picco di speedup è a N=2048 (153×). A N=4096 con CuPy lo speedup scende (67×) perché la matrice N×N completa supera la L2 cache GPU e diventa memory-bound; il kernel Numba tiled compensa proprio questo problema.

### Nota sull'occupancy Numba

Con N=4096 e TILE=256, la grid ha soli 16 blocchi — Numba emette un `NumbaPerformanceWarning` di bassa occupancy. Aumentare N a 8192+ satura completamente tutti gli SM della RTX 4080 e porta lo speedup verso 500–1000×.

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| N-Body O(N²) | Ogni corpo interagisce con tutti gli altri — complessità quadratica |
| Softening ε² | Fattore aggiunto a dist² per evitare singolarità a r→0 |
| GFLOPS | Giga Floating-Point Operations Per Second — misura della potenza di picco |
| Shared memory tiling | Tecnica CUDA per ridurre gli accessi a global memory mediante riuso in shared |
| syncthreads() | Barriera di sincronizzazione tra thread di uno stesso blocco |
| Warp occupancy | Quanti warp attivi per SM — determina l'efficienza del kernel |

---

## Tecnologie

- **CuPy** — simulazione brute-force N×N su GPU con broadcasting vectorizzato
- **Numba CUDA** — kernel custom con `cuda.shared.array` e tiling manuale
- **NumPy** — baseline CPU chunked per evitare OOM
- **matplotlib** — grafico speedup vs N salvato in outputs/

---

## Riferimenti

- [GPU Gems 3 — Fast N-Body Simulation with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)
- [Numba CUDA shared memory](https://numba.readthedocs.io/en/stable/cuda/memory.html)
- [CuPy documentation](https://docs.cupy.dev/)
