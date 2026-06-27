# LAB 09 — Quantitative Finance GPU

Accelerazione GPU per algoritmi di finanza quantitativa. I calcoli di risk management e pricing di derivati sono computazionalmente intensivi e intrinsecamente paralleli — ideali per GPU CUDA.

---

## Algoritmi

### 1. Portfolio VaR con decomposizione di Cholesky

- **Simulazioni**: 5 milioni di scenari
- **Assets**: 100 asset correlati
- **Metodo**: decomposizione di Cholesky per simulare rendimenti correlati

**Matematica:**

La matrice di correlazione `Σ` viene scomposta come:

```
Σ = L · Lᵀ       (decomposizione di Cholesky)
```

Per generare rendimenti correlati da variabili standard normali indipendenti `Z`:

```
correlated = L · Z        → applica la struttura di correlazione
returns    = μ·dt + σ·√dt · correlated   (GBM discretizzato)
```

**Value at Risk (VaR)** al livello di confidenza α:

```
VaR_α = -Q_α(R_portafoglio)
```

dove `Q_α` è il quantile α della distribuzione dei rendimenti simulati.

**Conditional VaR (CVaR / Expected Shortfall)**:

```
CVaR_α = E[R | R ≤ VaR_α]
```

- **GPU**: `cupy.linalg.cholesky` + `cupy.random.standard_normal` + matmul
- **Applicazioni**: stress testing, requisiti patrimoniali Basilea III/IV, FRTB

---

### 2. Options Greeks via Monte Carlo (Finite Differences)

- **Traiettorie**: 10 milioni per ogni MC run
- **Time steps**: 252 (giorni di trading per anno)
- **Parametri**:
  - S₀ = 100 (prezzo corrente)
  - K = 105 (strike price)
  - r = 0.05 (tasso risk-free)
  - σ = 0.2 (volatilità)
  - T = 1 anno
  - dS = 1.0 (step per differenze finite)

**Geometric Brownian Motion (GBM)**:

```
dS = r·S·dt + σ·S·dW
```

Soluzione discreta (schema di Euler-Maruyama):

```
S(t+dt) = S(t) · exp[(r - σ²/2)·dt + σ·√dt·Z]     Z ~ N(0,1)
```

**Greeks via differenze finite centrali**:

```
Delta = ∂V/∂S  ≈  [V(S₀+dS) - V(S₀-dS)] / (2·dS)

Gamma = ∂²V/∂S² ≈  [V(S₀+dS) - 2·V(S₀) + V(S₀-dS)] / dS²
```

- **GPU**: 3 run Black-Scholes MC in parallelo con CuPy
- **Applicazioni**: hedging dinamico (delta-hedging), gestione del rischio su portafogli di derivati

---

### 3. Implied Volatility Surface

- **Simulazioni**: 1 milione per ogni cella della griglia
- **Griglia**: 7 strikes × 4 maturities = 28 combinazioni
- **Strikes**: [85, 90, 95, 100, 105, 110, 115]
- **Maturities**: [0.25, 0.5, 1.0, 2.0] anni
- **Parametri**: S₀ = 100, r = 0.05, σ = 0.20

**Pricing MC per ogni cella (K, T)**:

```
V(K, T) = e^(-rT) · E[max(S_T - K, 0)]
```

La superficie dei prezzi rivela la struttura della volatilità implicita di mercato (volatility smile/skew).

- **GPU**: loop su 28 combinazioni, ogni MC call è accelerata da CuPy
- **Applicazioni**: calibrazione di modelli (Heston, SABR), trading di volatilità, market making

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab09-finance/src/run_finance.py
```

Il plot viene salvato automaticamente in `lab09-finance/outputs/lab09_benchmark.png`.

---

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

### CPU vs GPU

| Algoritmo | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| Portfolio-VaR (5M sim, 100 asset) | 9,839 | 157 | **62.7x** |
| Greeks-MC (10M traiettorie, 252 step) | 57,675 | 853 | **67.6x** |
| ImpVol-Surface (28 celle × 1M sim) | 143,424 | 1,516 | **94.6x** |

La volatility surface da sola richiederebbe **2.4 minuti** su CPU, **1.5 secondi** su GPU.

### Valori finanziari calcolati

**Portfolio VaR** (100 asset correlati, Cholesky):
- VaR 95%: **−0.2522%** (perdita massima giornaliera nel 95% degli scenari)
- CVaR 95%: **−0.3260%** (perdita media nei casi peggiori)

**Options Greeks** (call, S₀=100, K=105, σ=0.20, T=1, r=0.05):
- Prezzo: **$8.02** | Delta: **0.5469** | Gamma: **0.0322**
- Delta ≈ 0.55 atteso per call leggermente OTM — corretto

**Implied Vol Surface** — prezzi call GPU ($):

| T \ K | K=85 | K=90 | K=95 | K=100 | K=105 | K=110 | K=115 |
|-------|------|------|------|-------|-------|-------|-------|
| T=0.25 | 16.22 | 11.67 | 7.71 | 4.60 | 2.48 | 1.19 | 0.51 |
| T=0.50 | 17.64 | 13.51 | 9.88 | 6.88 | 4.58 | 2.92 | 1.76 |
| T=1.00 | 20.49 | 16.69 | 13.35 | 10.43 | 8.02 | 6.04 | 4.46 |
| T=2.00 | 25.43 | 22.04 | 18.94 | 16.14 | 13.66 | 11.44 | 9.59 |

Prezzi decrescenti con lo strike (ITM→OTM), crescenti con la scadenza — comportamento Black-Scholes corretto.

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| GBM | Geometric Brownian Motion: `dS = r·S·dt + σ·S·dW` — modello standard per i prezzi degli asset |
| VaR | Value at Risk: perdita massima attesa al livello di confidenza α (es. 95%) |
| CVaR | Conditional VaR / Expected Shortfall: perdita media nei casi peggiori oltre il VaR |
| Delta | Sensibilità del prezzo dell'opzione rispetto al prezzo del sottostante (`∂V/∂S`) |
| Gamma | Sensibilità del Delta rispetto al prezzo del sottostante (`∂²V/∂S²`) |
| Cholesky | Fattorizzazione `Σ = L·Lᵀ` per simulare variabili casuali correlate |
| Correlated Simulation | Uso della fattorizzazione di Cholesky per modellare dipendenze tra asset |
| Volatility Surface | Struttura tridimensionale dei prezzi (e volatilità implicite) su griglia K×T |
| Risk-Free Rate | Tasso di rendimento privo di rischio `r` usato per attualizzare i payoff |

---

## Tecnologie

- **CuPy** — GPU drop-in replacement per NumPy (random, linalg, elementwise ops)
- **NumPy** — baseline CPU
- **Rich** — output formattato a terminale (tabelle, colori)
- **Matplotlib** — plot del benchmark salvato in `outputs/`
- **CUDA** — architettura sottostante per parallelismo massivo
