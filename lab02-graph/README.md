# LAB 02 — Graph Algorithms

Algoritmi su grafi accelerati su GPU tramite moltiplicazione matrice-vettore sparsa (SpMV). Confronto CPU (SciPy sparse) vs GPU (CuPy sparse) su grafi sintetici di grandi dimensioni.

---

## Algoritmi

### 1. BFS — Breadth-First Search
- **Grafo**: 50.000 nodi, ~400.000 archi (sparse random)
- **Approccio**: iterative SpMV — la matrice di adiacenza normalizzata viene moltiplicata per il vettore frontiera a ogni livello
- **Formato**: CSR (Compressed Sparse Row)
- **CPU**: `scipy.sparse` matrix-vector multiply
- **GPU**: `cupy.sparse` matrix-vector multiply
- **Applicazioni**: analisi di reti sociali, routing, genomica

### 2. PageRank
- **Grafo**: 50.000 nodi, 50 iterazioni di power method
- **Damping factor**: 0.85 (formula Google originale)
- **Convergenza**: iterazioni di `score = d·A·score + (1−d)/N`
- **CPU**: SciPy sparse matvec
- **GPU**: CuPy sparse matvec
- **Applicazioni**: ranking di pagine web, reti di citazioni, sistemi di raccomandazione

### 3. Betweenness Centrality (Approssimata)
- **Grafo**: 10.000 nodi
- **Metodo**: BFS campionato da 100 sorgenti random
- **Misura**: quante volte un nodo si trova sul percorso più breve tra due nodi
- **Applicazioni**: identificazione di nodi "hub" critici in una rete

---

## Struttura dati

I grafi vengono generati sinteticamente con `scipy.sparse.random` e resi simmetrici (grafi non orientati). Il formato CSR viene usato per:
- accesso efficiente alle righe (iterazione per nodo)
- SpMV ottimizzato (operazione centrale di BFS e PageRank)

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab02-graph/src/run_graph.py
```

---

## Output atteso

```
[BFS]                 CPU: 380 ms  |  GPU:  42 ms  |  Speedup:  9x
[PageRank]            CPU: 620 ms  |  GPU:  55 ms  |  Speedup: 11x
[Betweenness approx]  CPU: 850 ms  |  GPU:  90 ms  |  Speedup:  9x
```

> Gli algoritmi su grafi sono tipicamente **memory-bound**: il vantaggio GPU dipende dalla sparsità e dalla saturazione della banda di memoria.

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| SpMV | Sparse Matrix-Vector multiply: operazione centrale per BFS e PageRank |
| CSR format | Formato compresso per matrici sparse (indici di riga + colonna + valori) |
| Memory-bound | Bassa intensità operazionale → speedup limitato da bandwidth, non da TFLOPS |
| Power iteration | Metodo iterativo per calcolare l'autovettore dominante (PageRank) |

---

## Tecnologie

- **CuPy sparse** — operazioni su matrici sparse su GPU
- **SciPy sparse** — baseline CPU (CSR, COO, SpMV)
- **NumPy** — gestione array e random
