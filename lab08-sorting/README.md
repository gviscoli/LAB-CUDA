# LAB 08 — Sorting & Parallel Primitives

Primitive fondamentali del GPU computing: algoritmi che costituiscono i building block di quasi ogni kernel CUDA avanzato. Implementazione con NumPy (CPU baseline) e CuPy (GPU via CUB/Thrust), con kernel Numba custom per la riduzione.

---

## Algoritmi

### 1. Radix Sort

- **Input**: array float32 di 50 milioni di elementi
- **CPU**: `numpy.sort` — introsort/timsort ibrido, complessita' O(N log N)
- **GPU**: `cupy.sort` — radix sort a 8 bit per digit, usa CUB/Thrust sotto
- **Paradigma**: sort non-comparativo, O(N·k) con k = numero di digit
- **Applicazioni**: ordinamento particle ID in simulazioni N-body, k-NN search, costruzione BVH per ray tracing, database columnar analytics

**Come funziona il radix sort GPU**: l'array viene processato digit per digit (tipicamente 8 bit alla volta → 4 passate su float32). Per ogni passata si calcola un istogramma, si esegue un prefix scan per calcolare le destinazioni, poi uno scatter. La GPU eccelle perche' tutte e tre le fasi sono massivamente parallele.

### 2. Prefix Scan (cumsum)

- **Input**: array int32 di 100 milioni di elementi
- **CPU**: `numpy.cumsum`
- **GPU**: `cupy.cumsum` — algoritmo work-efficient di Blelloch (1990): up-sweep + down-sweep
- **Complessita'**: O(N) work, O(log N) depth — ottimale per GPU
- **Applicazioni**: BFS (calcolo offset vicini per layer expansion), sparse MatMul (compattazione prodotti parziali), stream compaction (filtraggio parallelo), histogram equalization (CDF cumulativa)

Il prefix scan e' considerato la primitiva piu' importante del GPU computing perche' converte operazioni apparentemente sequenziali in computazione parallela. L'algoritmo di Blelloch sfrutta un albero binario in shared memory per ridurre il numero di sincronizzazioni tra thread.

**Bank conflicts**: una implementazione naive del prefix scan soffre di bank conflicts nella shared memory. La soluzione standard e' usare padding di `CONFLICT_FREE_OFFSET = n >> LOG_NUM_BANKS`.

### 3. Histogram (Parallel Histogram)

- **Input**: array float32 gaussiano di 100 milioni di elementi, 1024 bin
- **CPU**: `numpy.histogram`
- **GPU**: `cupy.histogram` — usa `atomicAdd` su shared memory per bin locali, poi reduce globale
- **Applicazioni**: image processing (histogram equalization), CT scan analysis, feature extraction, radiosity rendering, analisi scientifica di distribuzioni

La sfida principale e' la contention sugli `atomicAdd`: molti thread che scrivono sullo stesso bin serializzano. La soluzione standard usa istogrammi privati in shared memory per ogni blocco, poi una riduzione finale.

**Warp divergence**: se i dati hanno distribuzione non uniforme (es. gaussiana con picchi), alcuni warp impiegano piu' tempo di altri. Il padding degli array all'interno dei blocchi migliora il bilanciamento.

### 4. Parallel Reduction (Sum)

- **Input**: array float32 di 100 milioni di elementi
- **CPU**: `numpy.sum`
- **GPU (CuPy)**: `cupy.sum` — usa CUB device-wide reduction con warp shuffle intrinsics (`__shfl_down_sync`)
- **GPU (Numba)**: kernel `@cuda.reduce` custom — mostra il pattern a due fasi
- **Pattern**: ogni thread somma con il partner a distanza stride decrescente (pattern butterfly), poi `atomicAdd` inter-block

**Warp shuffle**: nelle ultime 5 iterazioni della riduzione (quando stride < 32), si possono usare le istruzioni `__shfl_down_sync` invece di accedere alla shared memory, eliminando le `__syncthreads()` e raddoppiando l'efficienza dell'ultima fase.

---

## Perche' queste primitive sono fondamentali

| Primitiva | Usata in |
|-----------|----------|
| Radix Sort | Costruzione BVH, k-NN, database analytics, particle sorting |
| Prefix Scan | BFS, sparse MatMul, stream compaction, Radix Sort stesso |
| Histogram | Image processing, equalizzazione, analisi dati, rendering |
| Reduction | Loss function, dot product, norme vettoriali, statistiche |

Qualsiasi kernel CUDA complesso (BFS, sparse BLAS, ray tracing, simulazione N-body) usa almeno una di queste primitive. CUB (CUDA Unbound) e Thrust le implementano al massimo delle prestazioni hardware.

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab08-sorting/src/run_sorting.py
```

Il grafico comparativo viene salvato in `lab08-sorting/outputs/lab08_benchmark.png`.

---

## Risultati misurati

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

### CPU vs GPU — Primitives benchmark

| Primitiva | N | CPU (ms) | GPU (ms) | Speedup |
|-----------|---|----------|----------|---------|
| Radix Sort | 50M float32 | 309.72 | 4.11 | **75.4x** |
| Prefix Scan | 100M int32 | 419.44 | 5.31 | **79.0x** |
| Histogram | 100M float32, 1024 bin | 435.38 | 44.46 | **9.8x** |
| Reduction (CuPy) | 100M float32 | 42.78 | 0.65 | **66.3x** |
| Reduction (Numba) | 100M float32 | 42.78 | 1.31 | **32.7x** |

L'Histogram (9.8x) è molto più lento degli altri per via dei conflitti sugli `atomicAdd`: con 1024 bin e 100M elementi gaussiani, i bin centrali ricevono milioni di write concorrenti che si serializzano. Radix Sort e Prefix Scan saturano la bandwidth GPU senza contention.

### Scaling Analysis — Radix Sort GPU vs CPU

| N | CPU (ms) | GPU (ms) | Speedup | Memoria GPU |
|---|----------|----------|---------|-------------|
| 1M | 4.2 | 0.1 | 31.2x | 4 MB |
| 10M | 52.8 | 1.1 | 48.9x | 38 MB |
| 50M | 315.0 | 7.9 | 39.8x | 191 MB |
| 100M | 660.3 | 99.0 | 6.7x | 381 MB |
| 500M | 3999.7 | 42.4 | **94.2x** | 1907 MB |

Il calo anomalo a N=100M (6.7x) è causato dalla frammentazione del memory pool CuPy dopo gli allocation precedenti: Thrust necessita di un buffer di lavoro 2× la dimensione dell'input, che a 381MB può trovarsi su pagine non contigue. A N=500M il pool è già "caldo" e la GPU lavora in piena saturazione.

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Work-efficient parallel scan | Algoritmo di Blelloch: O(N) work, O(log N) span — ideale per GPU |
| Bank conflicts | Accessi multipli thread allo stesso banco di shared memory → serializzazione |
| Warp divergence | Thread dello stesso warp che eseguono branch diversi → sottoutilizzo |
| Warp shuffle (`__shfl_down_sync`) | Scambio dati intra-warp senza shared memory → riduzione piu' veloce |
| atomicAdd | Operazione atomica su memoria globale/shared — necessaria per histogram |
| Stream compaction | Filtraggio parallelo di elementi che soddisfano una condizione (usa prefix scan) |
| CUB / Thrust | Librerie NVIDIA per primitive GPU altamente ottimizzate |
| Radix sort digit | Il sort avviene digit per digit (8 bit): 4 passate su float32 a 32 bit |

---

## Scaling Analysis

Lo script esegue anche un'analisi di scaling del radix sort per N in:
`[1M, 10M, 50M, 100M, 500M]`

Si osserva tipicamente:
- N < 1M: GPU overhead domina, speedup basso o negativo
- N ~ 10M: break-even, GPU inizia a essere competitiva
- N > 50M: GPU in piena saturazione, speedup massimo
- N > VRAM disponibile: `MemoryError` → il loop si ferma automaticamente

---

## Tecnologie

- **CuPy** — sort, cumsum, histogram, sum su GPU (wrappa CUB/Thrust/cuBLAS)
- **Numba** — kernel `@cuda.reduce` custom per la riduzione
- **NumPy** — baseline CPU con interfaccia identica
- **Rich** — output a console formattato con tabelle e colori
- **Matplotlib** — grafici comparativi salvati in `outputs/`
