# LAB 05 — Fluid Dynamics & PDE Solvers

Solver numerici per equazioni alle derivate parziali (PDE) in fluidodinamica computazionale (CFD). Implementazione con schemi a differenze finite esplicite su GPU tramite CuPy.

---

## Algoritmi

### 1. Equazione del Calore 2D (Heat Equation)
- **Griglia**: 1024 × 1024 punti
- **Time steps**: 500 iterazioni temporali
- **PDE**: `∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)`
- **Schema**: FTCS (Forward-Time Center-Space) — esplicito al primo ordine
- **Stabilità**: criterio CFL 2D → `r = α·dt/dx² ≤ 0.25`
- **Condizione iniziale**: impulso di temperatura al centro del dominio
- **Validazione**: conservazione dell'energia totale nel dominio
- **Applicazioni**: simulazioni termiche, diffusione in materiali, metallurgia

### 2. Navier-Stokes 2D — Lid-Driven Cavity
- **Griglia**: 64 × 64 punti
- **Time steps**: 500 iterazioni
- **Regime**: flusso incomprimibile laminare, Re = 100 (ν = 0.01)
- **Schema**: metodo di proiezione di Chorin (pressure-velocity splitting):
  1. Passo predittore: calcolo velocità intermedia ignorando la pressione
  2. Equazione di Poisson per la pressione — risolta con **solver diretto LU sparse** (scipy)
  3. Correzione della velocità per soddisfare `∇·u = 0`
- **Geometria**: cavità quadrata, lid superiore si muove a u=1, pareti statiche
- **Nota solver**: Jacobi/SOR iterativi NON convergono per questo problema. Le BC miste (Neumann su 3 lati, Dirichlet su 1) producono spettro di Jacobi ρ≈0.9997 → servono ~4000 iter/step. Il solver LU diretto (fattorizzazione una tantum) garantisce pressione esatta ad ogni step.
- **Applicazioni**: CFD, aerodinamica, meteorologia, biomeccanica dei fluidi

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab05-fluid-pde/src/run_fluid-pde.py
```

---

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

| PDE | CPU (ms) | GPU (ms) | Speedup | Note |
|-----|----------|----------|---------|------|
| Heat Equation 2D (1024×1024) | 3536.77 | 40.04 | **88.3x** | Energia conservata ✓ |
| Navier-Stokes 2D (64×64, Re=100) | 180.26 | N/A | — | velocità max = 1.0000 ✓ |

**Nota GPU Navier-Stokes**: lo speedup GPU per NS con loop Python-level è strutturalmente negativo — ogni step lancia decine di kernel CuPy con sincronizzazione implicita. Per ottenere speedup reale serve un loop CUDA monolitico (Numba `@cuda.jit` o kernel custom) che esegua tutti i 500 step senza round-trip CPU↔GPU.

---

## Struttura numerica

### Schema FTCS — Heat Equation
```
u[i,j]^{n+1} = u[i,j]^n + r * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
```

### Passo di proiezione — Navier-Stokes (Chorin)
```
1. b = ρ·(1/dt·∇·uⁿ - (∂u/∂x)² - 2·(∂u/∂y)(∂v/∂x) - (∂v/∂y)²)
2. ∇²p = b   →  LU solve (scipy sparse factorized)
3. u* = uⁿ - uⁿ·∇uⁿ·dt + ν·∇²uⁿ·dt - dt/ρ·∇p
```

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| FTCS scheme | Schema esplicito: semplice ma limitato dal criterio CFL per la stabilità |
| CFL 2D | `ν·dt/dx² ≤ 0.25` per 2D (dimezzato rispetto a 1D per la somma dei termini laplaciani) |
| Chorin projection | Decomposizione pressione-velocità per flussi incomprimibili |
| Sparse direct solver | Fattorizzazione LU una tantum → back-substitution O(N) per step successivi |
| Jacobi spectral radius | Per BCs miste (Neumann+Dirichlet): ρ_J≈0.9997, ~4000 iter necessarie vs 1 per LU |
| Memory-bound | Schemi a differenze finite hanno bassa intensità operazionale (~1-2 FLOP/byte) |

---

## Tecnologie

- **CuPy** — operazioni array su GPU (Heat Equation: 88x speedup)
- **NumPy** — baseline CPU con identica interfaccia
- **scipy.sparse** — costruzione matrice Laplaciana sparsa (4096×4096 per N=64)
- **scipy.sparse.linalg.factorized** — solver LU diretto per pressione
