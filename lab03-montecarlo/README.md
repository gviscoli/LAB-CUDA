# LAB 03 — Monte Carlo Simulations

Simulazioni stocastiche massicciamente parallele su GPU. I metodi Monte Carlo sono "embarrassingly parallel" — ogni campione è indipendente — il che li rende ideali per ottenere speedup elevati su CUDA.

---

## Algoritmi

### 1. Stima di π con Monte Carlo
- **Campioni**: 100 milioni
- **Metodo**: generazione uniforme di punti in [0,1]², conta quanti cadono nel cerchio unitario
- **Stima**: `π ≈ 4 × (punti nel cerchio) / (punti totali)`
- **GPU**: `cupy.random.uniform` + reduce sum
- **Medie**: 5 run per ridurre la varianza
- **Applicazioni**: validazione di generatori random, introduzione al parallelismo

### 2. Black-Scholes — Pricing di opzioni europee
- **Traiettorie**: 10 milioni
- **Time steps**: 252 (trading days per anno)
- **Parametri**:
  - S₀ = 100 (prezzo iniziale)
  - K = 105 (strike price)
  - r = 0.05 (tasso risk-free)
  - σ = 0.2 (volatilità)
  - T = 1 anno
- **Modello**: Geometric Brownian Motion — `dS = r·S·dt + σ·S·dW`
- **Output**: prezzo atteso dell'opzione call (valore attualizzato del payoff)
- **Applicazioni**: finanza quantitativa, gestione del rischio, pricing derivati

### 3. Modello di Ising 2D
- **Griglia**: 1024 × 1024 spin (±1)
- **Steps**: 1000 passi Metropolis
- **Temperatura**: T ≈ Tc = 2.269 (temperatura critica di fase)
- **Algoritmo**: checkerboard update — aggiornamento alternato di sottoreticoli "neri" e "bianchi" per massimizzare il parallelismo
- **Boundary conditions**: periodiche (toro)
- **Misura**: magnetizzazione media (parametro d'ordine)
- **Applicazioni**: fisica statistica, materiali magnetici, transizioni di fase

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab03-montecarlo/src/run_montecarlo.py
```

---

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

| Algoritmo | CPU (ms) | GPU (ms) | Speedup | Note |
|-----------|----------|----------|---------|------|
| Pi-MC (100M campioni) | 1631.60 | 12.75 | **127.9x** | π ≈ 3.141432 |
| Black-Scholes (10M traj.) | 55821.09 | 679.74 | **82.1x** | Prezzo call: $8.0163 |
| Ising 2D (1024×1024) | 1079.22 | 1247.70 | **0.9x** | GPU più lento — vedi nota |

**Nota Ising 2D**: lo speedup è < 1 perché l'implementazione CuPy usa operazioni array-level (`roll`, `where`, random) con sincronizzazioni implicite a ogni step Metropolis. L'overhead di lancio kernel e sincronizzazione GPU (×1000 step) supera il guadagno computazionale. Per ottenere speedup reale servirebbe un kernel Numba CUDA custom che esegua tutti gli step in un unico lancio, eliminando i round-trip CPU↔GPU.

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Embarrassingly parallel | Campioni indipendenti — zero comunicazione tra thread |
| GBM | Geometric Brownian Motion: modello stocastico per i prezzi degli asset |
| Metropolis algorithm | Accept/reject per campionare dalla distribuzione di Boltzmann |
| Checkerboard update | Tecnica per parallelizzare Metropolis rispettando le dipendenze locali |
| Critical temperature | Tc ≈ 2.269: punto di transizione di fase nel modello di Ising 2D |

---

## Tecnologie

- **CuPy random** — generazione random massiva su GPU (`uniform`, `standard_normal`)
- **Numba CUDA** — kernel Metropolis con aggiornamento in-place
- **NumPy** — baseline CPU
