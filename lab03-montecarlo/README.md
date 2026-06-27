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

## Output atteso

```
[Monte Carlo π]   CPU: 2100 ms  |  GPU:  18 ms  |  Speedup: 116x  |  π ≈ 3.14159
[Black-Scholes]   CPU: 4800 ms  |  GPU:  45 ms  |  Speedup: 107x  |  Price: 8.02
[Ising Model]     CPU: 9500 ms  |  GPU:  85 ms  |  Speedup: 112x  |  |M| ≈ 0.73
```

> Tutti e tre gli algoritmi sono **compute-bound** e embarrassingly parallel: speedup tipicamente > 100×.

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
