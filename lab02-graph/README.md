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

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11 — grafo 50.000 nodi

| Algoritmo | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| BFS | 13.62 | 1.70 | **8.0x** |
| PageRank (50 iter.) | 30.27 | 3.27 | **9.3x** |
| Betweenness Centrality | 57.77 | N/A | — (solo CPU) |

> Gli algoritmi su grafi sono tipicamente **memory-bound**: il vantaggio GPU è contenuto (4–8×) perché la bassa intensità operazionale non satura il picco computazionale ma la banda di memoria. Per speedup elevati usare cuGraph (RAPIDS).

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
