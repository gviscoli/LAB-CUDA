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
└── lab06-benchmark/                 ← Confronto CPU vs GPU vs HPC
```

---

## I 6 Lab

| Lab | Dominio | Algoritmi | Tecnologie |
|-----|---------|-----------|------------|
| **01** | Numerical | FFT, BLAS, Stencil 2D/3D | CuPy, Numba |
| **02** | Graph | BFS, PageRank, SSSP | CuPy, cuGraph |
| **03** | Monte Carlo | π, Black-Scholes, Ising | CuPy random, Numba |
| **04** | ML Kernels | MatMul, Conv2D, Attention | Numba CUDA, Triton |
| **05** | Fluid/PDE | Heat eq., Navier-Stokes 2D | CuPy, Numba stencil |
| **06** | Benchmark | Roofline model, scaling | tutti i precedenti |

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
