# LAB 05 — Fluid Dynamics & PDE Solvers

Solver numerici per equazioni alle derivate parziali (PDE) in fluidodinamica computazionale (CFD). Implementazione con schemi a differenze finite esplicite su GPU tramite CuPy.

---

## Algoritmi

### 1. Equazione del Calore 2D (Heat Equation)
- **Griglia**: 1024 × 1024 punti
- **Time steps**: 500 iterazioni temporali
- **PDE**:
  ```
  ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
  ```
- **Schema**: FTCS (Forward-Time Center-Space) — esplicito al primo ordine
- **Stabilità**: criterio CFL → `r = α·dt/dx² ≤ 0.25`
- **Condizioni al contorno**: Dirichlet (temperatura fissa ai bordi)
- **Condizione iniziale**: impulso di temperatura al centro del dominio
- **Validazione**: conservazione dell'energia totale nel dominio
- **Applicazioni**: simulazioni termiche, diffusione in materiali, metallurgia

### 2. Navier-Stokes 2D — Lid-Driven Cavity
- **Griglia**: 256 × 256 punti
- **Time steps**: 200 iterazioni
- **Regime**: flusso incomprimibile laminare, Re = 100
- **Parametri**: ρ = 1, ν = 0.01, dt = 0.001
- **Schema**: metodo di proiezione di Chorin (pressure-velocity splitting)
  1. Passo predittore: calcolo velocità intermedia (`u*`) ignorando la pressione
  2. Equazione di Poisson per la pressione (20 iterazioni Jacobi)
  3. Correzione della velocità per soddisfare la divergenza nulla (`∇·u = 0`)
- **Geometria**: cavità quadrata, lid superiore si muove a u=1, pareti statiche
- **Riferimento**: Barba & Forsyth "CFD Python" (12-step Navier-Stokes course)
- **Applicazioni**: CFD, aerodinamica, meteorologia, biomeccanica dei fluidi

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab05-fluid-pde/src/run_fluid-pde.py
```

---

## Output atteso

```
[Heat Equation]      CPU: 4200 ms  |  GPU: 180 ms  |  Speedup:  23x  |  Energy conserved: YES
[Navier-Stokes]      CPU: 8500 ms  |  GPU: 410 ms  |  Speedup:  21x  |  Re=100 converged
```

---

## Struttura numerica

### Schema FTCS — Heat Equation
```
u[i,j]^{n+1} = u[i,j]^n + r * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j])
```

### Passo di proiezione — Navier-Stokes
```
1. u* = u^n + dt * (-u·∇u + ν·∇²u)        ← advection + diffusion
2. ∇²p = (ρ/dt) · ∇·u*                     ← Poisson per pressione
3. u^{n+1} = u* - (dt/ρ) · ∇p             ← proiezione su spazio incomprimibile
```

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| FTCS scheme | Schema esplicito: semplice ma limitato dal criterio CFL per la stabilità |
| CFL condition | Vincolo `r ≤ 0.25` per la stabilità numerica dell'equazione del calore |
| Chorin projection | Decomposizione pressione-velocità per flussi incomprimibili |
| Stencil pattern | Ogni punto dipende solo dai vicini → alta località, ideale per GPU |
| Memory-bound | Schemi a differenze finite hanno bassa intensità operazionale (~1-2 FLOP/byte) |

---

## Tecnologie

- **CuPy** — operazioni array su GPU (roll, slicing, operazioni element-wise)
- **NumPy** — baseline CPU con identica interfaccia
- **Matplotlib** — visualizzazione dei campi di velocità e temperatura
