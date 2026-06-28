# LAB 11 — Sparse Linear Algebra GPU

Accelerazione GPU per algebra lineare sparsa. I sistemi reali — FEM, CFD, grafi, GNN — generano matrici con milioni di righe ma solo 0.01–0.1% di elementi non-zero: il formato CSR (Compressed Sparse Row) + cuSPARSE rendono questi calcoli praticabili in tempi reali.

---

## Algoritmi

### 1. SpMV — Sparse Matrix-Vector Multiplication

- **Matrice**: 200.000 × 200.000, density 0.0005 (NNZ ≈ 20M)
- **Operazione**: `y = A · x`   dove A e' CSR, x e y sono densi

**Formato CSR (Compressed Sparse Row)**:

```
A_csr = { data[NNZ], indices[NNZ], indptr[N+1] }
Memoria = (NNZ + NNZ + N+1) * sizeof(int/float)
          vs NxN * sizeof(float) per formato denso
```

SpMV e' **bandwidth-bound**: l'intensita' aritmetica e' ~2 FLOP/byte — quindi la velocita' dipende dalla memoria HBM, non dai CUDA core.

- **CPU**: `scipy.sparse.csr_matrix.dot()` — OpenBLAS single-thread
- **GPU**: cuSPARSE `csrmv` — accesso coalescente per righe, warp-level parallelism
- **Applicazioni**: PageRank, FEM assembler, SVM kernel, NLP sparse attention

---

### 2. SpMM — Sparse Matrix × Dense Matrix

- **Matrice sparsa**: 50.000 × 50.000, density 0.001
- **Matrice densa X**: 50.000 × 128   (K=128 simula embedding GNN)
- **Operazione**: `Y = A · X`

**Rilevanza GNN (Graph Neural Networks)**:

Ogni layer di un GNN esegue:
```
H^(l+1) = σ( A_norm · H^(l) · W^(l) )
```
dove `A_norm` e' la matrice di adiacenza normalizzata (sparsa) e `H^(l)` e' la matrice degli embedding (densa). SpMM e' quindi il kernel critico di ogni forward pass GNN.

- **GPU**: cuSPARSE `csrmm2` — tiling ottimizzato per matrice densa K
- **Applicazioni**: GraphSAGE, GCN, GAT, molecular property prediction

---

### 3. PageRank — Graph Laplacian Power Iteration

- **Nodi**: 100.000 (simula web-graph di 100K pagine)
- **Archi**: ~1.000.000 (10 link per pagina in media)
- **Iterazioni**: 50 passi di power iteration

**Algoritmo PageRank (Brin & Page, 1998)**:

```
pr(t+1) = d · A_norm · pr(t) + (1-d)/N
```

dove:
- `d = 0.85` — damping factor (probabilita' di seguire un link)
- `A_norm` — matrice di adiacenza colonna-normalizzata (column-stochastic)
- `(1-d)/N` — teleportation (salto casuale a qualsiasi pagina)

Ogni iterazione e' una SpMV; la convergenza si misura in norma L1 del residuo.

- **Convergenza**: tipicamente < 50 iterazioni per errore < 1e-6
- **GPU**: 50 SpMV consecutive su cuSPARSE — latency hiding con CUDA streams
- **Applicazioni**: search ranking, analisi reti sociali, botnet detection, bioinformatica

---

### 4. Sparse CG Solver — Conjugate Gradient

- **Sistema**: N=50.000, matrice SPD (simmetrica definita positiva)
- **Densita'**: 0.0002 (tipica FEM 2D, ~5 NNZ/riga)
- **Tolleranza**: 1e-5 (residuo relativo)
- **Max iterazioni**: 200

**Algoritmo Conjugate Gradient (Hestenes & Stiefel, 1952)**:

```
Dato Ax = b con A SPD:

r_0 = b - A·x_0,   p_0 = r_0
Per k = 0, 1, 2, ...:
    alpha_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + alpha_k · p_k
    r_{k+1} = r_k - alpha_k · A·p_k
    beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + beta_k · p_k
```

Costo per iterazione: **1 SpMV** + 3 dot products + 2 DAXPY — tutto O(NNZ).

**Convergenza**: `||r_k|| / ||b|| < tol` in O(sqrt(kappa(A))) iterazioni.

- **GPU**: ogni SpMV interno usa cuSPARSE — speedup cumulato su tutte le iterazioni
- **Senza precondizionatore**: mostra le caratteristiche base del metodo
- **Applicazioni**: FEM 2D/3D, equazione di Poisson (CFD), deep learning (K-FAC optimizer)

---

### 5. Scaling Analysis — SpMV al variare di N

Analisi dello speedup GPU/CPU al crescere della dimensione della matrice:

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

### CPU vs GPU — benchmark principale

| Algoritmo | Dettaglio | CPU (ms) | GPU (ms) | Speedup |
|-----------|-----------|----------|----------|---------|
| SpMV | N=50K, NNZ=2.5M | 1.28 | 0.12 | **11.0x** |
| SpMM | N=50K, K=128, NNZ=2.5M | 186.33 | 5.57 | **33.5x** |
| PageRank | N=100K, 1M archi, 50 iter | 33.06 | 4.15 | **8.0x** |
| SparseCG | N=50K, 6 iterazioni, tol=1e-5 | 4.74 | 2.12 | **2.2x** |

**SpMM (33.5x)** batte SpMV (11x) perché K=128 colonne dense aumentano il riuso dei dati in cache — ogni riga sparsa viene caricata una volta e moltiplicata per 128 colonne invece di 1.

**SparseCG (2.2x)** è basso perché converge in sole **6 iterazioni** (matrice ben condizionata): il lancio di 6 kernel GPU ha overhead fisso elevato rispetto al tempo totale.

### Scaling Analysis — SpMV al variare di N

| N | NNZ | CPU (ms) | GPU (ms) | Speedup | BW GPU (GB/s) |
|---|-----|----------|----------|---------|---------------|
| 10,000 | 110K | 0.09 | 0.14 | 0.7x | 3.7 |
| 50,000 | 2.5M | 1.49 | 0.12 | 12.9x | 91.6 |
| 100,000 | 10.1M | 7.11 | 0.23 | **30.8x** | 178.6 |
| 200,000 | 20.2M | 16.31 | 0.76 | 21.4x | 107.9 |
| 500,000 | 25.5M | 26.01 | 48.23 | 0.5x | 2.2 |

- **N=10K (0.7x)**: GPU più lenta — overhead kernel domina su matrice da 110K NNZ
- **N=100K (30.8x)**: picco di speedup — matrice in L2 cache GPU, ottima saturazione
- **N=200K (21.4x)**: calo — matrice 160MB supera la L2, aumentano i cache miss
- **N=500K (0.5x)**: GPU più lenta della CPU — density=0.0001 produce accesso casuale su vettore da 2MB, ogni `x[col_idx]` è un cache miss → irregular random access, il punto debole della GPU

Lo speedup SpMV e' limitato dalla **banda di memoria** (bandwidth-bound), non dai FLOP.
RTX 4080: 716 GB/s vs DDR5: ~70 GB/s → speedup teorico max ~10x per SpMV classico.

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab11-sparse/src/run_sparse.py
```

Il plot viene salvato automaticamente in `lab11-sparse/outputs/lab11_benchmark.png`.

**Prerequisiti**: `scipy` (CPU baseline) e `cupy` (GPU). Per installare:

```powershell
pip install scipy cupy-cuda12x
```

---

## Risultati attesi

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

### CPU vs GPU (valori indicativi)

| Algoritmo | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| SpMV (N=200K, NNZ≈20M) | ~180 | ~15 | **~12x** |
| SpMM (N=50K, K=128) | ~250 | ~20 | **~12x** |
| PageRank (100K nodi, 50 iter) | ~800 | ~60 | **~13x** |
| SparseCG (N=50K, ~100 iter) | ~600 | ~50 | **~12x** |

Lo speedup SpMV e' limitato dalla larghezza di banda della memoria (bandwidth-bound), non dalla potenza di calcolo. La GPU eccelle grazie ai 716 GB/s di HBM vs ~70 GB/s della RAM DDR5.

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| CSR | Compressed Sparse Row: memorizza solo NNZ con `data`, `indices`, `indptr` |
| NNZ | Number of Non-Zeros: misura la "sparsita'" della matrice |
| SpMV | Sparse Matrix-Vector: y = A·x — kernel fondamentale di tutti i solver iterativi |
| SpMM | Sparse Matrix-Matrix: Y = A·X — kernel critico dei Graph Neural Networks |
| Bandwidth-bound | Il collo di bottiglia e' la memoria, non la CPU/GPU FLOP rate |
| CG | Conjugate Gradient: solver iterativo ottimale per sistemi SPD sparsi |
| PageRank | Power iteration su matrice stocastica del grafo — 50 SpMV consecutive |
| cuSPARSE | Libreria NVIDIA per operazioni sparse (SpMV, SpMM, triangular solve) |
| Preconditioner | Trasformazione M^(-1)A per ridurre kappa(A) e accelerare la convergenza CG |
| Dangling nodes | Nodi senza archi uscenti in PageRank — gestiti con normalizzazione |

---

## Tecnologie

- **CuPy sparse** (`cupyx.scipy.sparse`) — wrapper Python per cuSPARSE
- **cuSPARSE** — libreria NVIDIA per algebra lineare sparsa (CSR, COO, BSR, ELL)
- **SciPy sparse** — baseline CPU con OpenBLAS
- **NumPy** — operazioni dense ausiliarie
- **Rich** — output formattato a terminale (tabelle, colori)
- **Matplotlib** — plot del benchmark salvato in `outputs/`
- **CUDA** — architettura sottostante: warp scheduling, HBM bandwidth, atomic ops
