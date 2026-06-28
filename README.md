# 🚀 LAB-CUDA — HPC Parallel Computing Lab
### Fase 1: Python + CuPy + Numba CUDA

> Piattaforma di sperimentazione locale per algoritmi ad alte prestazioni,
> progettata per essere portabile su cluster HPC reali (CINECA, AWS HPC, CERN).

**Hardware target**: Intel Core i9 | 96GB RAM | RTX 4080 16GB | Windows 11

---

## Struttura Progetto

```
C:\DATI\Sviluppo\LAB-CUDA\
│
├── README.md                        ← questo file
├── requirements.txt                 ← dipendenze comuni
├── setup.py                         ← installazione pacchetti
├── environment.yml                  ← ambiente Conda (alternativa)
├── .env.example                     ← variabili d'ambiente
│
├── shared/                          ← utility condivise tra tutti i lab
│   ├── utils/
│   │   ├── gpu_info.py              ← info GPU, benchmark baseline
│   │   ├── timer.py                 ← timer CPU vs GPU
│   │   └── plotter.py               ← grafici performance
│   └── datasets/                    ← dataset condivisi
│
├── lab01-numerical/                 ← FFT, Linear Algebra, Stencil
├── lab02-graph/                     ← BFS, PageRank, Shortest Path
├── lab03-montecarlo/                ← Pi, Black-Scholes, Ising Model
├── lab04-ml-kernels/                ← Matmul, Conv2D, Attention custom
├── lab05-fluid-pde/                 ← Navier-Stokes, Heat equation
├── lab06-benchmark/                 ← Confronto CPU vs GPU vs HPC
├── lab07-nbody/                     ← N-Body gravitazionale O(N²) + tiled CUDA
├── lab08-sorting/                   ← Radix sort, Prefix scan, Histogram
├── lab09-finance/                   ← Portfolio VaR, Greeks MC, Volatility surface
├── lab10-imgproc/                   ← Gaussian Blur, Sobel, Bilateral Filter
├── lab11-sparse/                    ← SpMV, Conjugate Gradient, PageRank sparse
├── lab12-bio/                       ← Smith-Waterman, K-mer, Edit Distance
└── lab13-inference/                 ← INT8 MatMul, BatchNorm, Softmax, LayerNorm
```

---

## I 13 Lab

| Lab | Dominio | Algoritmi | Tecnologie |
|-----|---------|-----------|------------|
| [**01** — Numerical](lab01-numerical/README.md) | Numerical | FFT 1D/2D, DGEMM, Stencil 2D | CuPy, Numba CUDA |
| [**02** — Graph](lab02-graph/README.md) | Graph | BFS, PageRank, Betweenness | CuPy sparse, SciPy |
| [**03** — Monte Carlo](lab03-montecarlo/README.md) | Monte Carlo | π, Black-Scholes, Ising 2D | CuPy random, Numba |
| [**04** — ML Kernels](lab04-ml-kernels/README.md) | ML Kernels | MatMul tiled, Conv2D, Attention | Numba CUDA, PyTorch, cuDNN |
| [**05** — Fluid/PDE](lab05-fluid-pde/README.md) | Fluid/PDE | Heat Equation, Navier-Stokes 2D | CuPy, SciPy sparse |
| [**06** — Benchmark](lab06-benchmark/README.md) | Benchmark | Roofline model, scaling analysis | tutti i precedenti |
| [**07** — N-Body](lab07-nbody/README.md) | Fisica | N-Body O(N²), tiled CUDA kernel | CuPy, Numba CUDA |
| [**08** — Sorting](lab08-sorting/README.md) | Primitive | Radix sort, Prefix scan, Histogram | CuPy, Numba |
| [**09** — Finance](lab09-finance/README.md) | Finanza | Portfolio VaR, Greeks MC, Vol surface | CuPy, SciPy |
| [**10** — ImgProc](lab10-imgproc/README.md) | Image | Gaussian Blur, Sobel, Bilateral | CuPy, Numba CUDA, cupyx |
| [**11** — Sparse](lab11-sparse/README.md) | Sparse LA | SpMV, CG solver, PageRank | CuPy sparse, cupyx |
| [**12** — Bio](lab12-bio/README.md) | Genomica | Smith-Waterman, K-mer, Edit dist. | CuPy, Numba CUDA |
| [**13** — Inference](lab13-inference/README.md) | DL Inference | INT8 MatMul, BatchNorm, Softmax | CuPy, Numba CUDA |

---

## Setup Rapido

```powershell
# 1. Crea ambiente virtuale
cd C:\DATI\Sviluppo\LAB-CUDA
python -m venv .venv
.venv\Scripts\activate

# 2. Installa PyTorch con CUDA 12.4
pip install torch==2.6.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Installa dipendenze lab
pip install -r requirements.txt

# 4. Verifica GPU
python shared/utils/gpu_info.py

# 5. Avvia Jupyter
jupyter lab
```

---

## Paradigma di Sviluppo

```
┌─────────────────────────────────────────────────────┐
│  Ogni esperimento segue questo pattern:             │
│                                                     │
│  1. CPU baseline    → NumPy / SciPy puro            │
│  2. GPU CuPy        → drop-in replacement NumPy     │
│  3. GPU Numba       → kernel custom ottimizzato      │
│  4. Benchmark       → speedup, memoria, throughput  │
│  5. HPC-ready       → MPI4Py wrapper per cluster    │
└─────────────────────────────────────────────────────┘
```

---

## Portabilità HPC

Il codice è strutturato per girare su:
- **Locale**: RTX 4080 (questo PC)
- **Cloud HPC**: AWS HPC / Google Cloud HPC
- **Cluster nazionale**: CINECA Leonardo (NVIDIA A100/H100)

Riferimenti ufficiali:
- [CuPy Documentation](https://docs.cupy.dev/)
- [Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CINECA CUDA Guide](https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+GPU+programming)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
