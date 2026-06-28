# LAB 12 — Bioinformatics GPU

Accelerazione GPU per algoritmi di bioinformatica. Il sequenziamento genomico genera miliardi di basi da allineare, confrontare e analizzare — workload O(N²) che beneficiano enormemente della GPU. I cluster HPC di genomica (CINECA, AWS HealthOmics) sono dominati da questo tipo di carico.

---

## Algoritmi

### 1. Smith-Waterman Local Alignment

- **Sequenze**: L=2000 bp (base pairs) ciascuna
- **Scoring**: match=+2, mismatch=−1, gap=−2
- **CPU**: anti-diagonal wavefront con NumPy vectorized
- **GPU CuPy**: stessa logica su GPU, ogni anti-diagonale in parallelo
- **GPU Numba**: kernel CUDA dove ogni thread processa un elemento dell'anti-diagonale

**Ricorrenza DP:**

```
H[i,j] = max(0,
             H[i-1,j-1] + s(a[i],b[j]),   ← match/mismatch
             H[i-1,j]   + gap,              ← gap in A
             H[i,j-1]   + gap)              ← gap in B

s(a,b) = match se a==b, mismatch altrimenti
```

**Parallelismo anti-diagonale:**

Le celle dell'anti-diagonale `k` (con `i+j=k`) non hanno dipendenze reciproche — ogni thread può calcolarle in parallelo:

```
Antidiagonale k=4:      (1,3), (2,2), (3,1)   → 3 thread paralleli
Antidiagonale k=5:      (1,4), (2,3), (3,2), (4,1)  → 4 thread paralleli
```

**Metrica**: MCUPS (Mega Cell Updates Per Second) = L² / tempo_ms × 10⁻³

### 2. K-mer Counting

- **Sequenza**: 10 milioni di basi
- **k**: 8 → 4⁸ = 65,536 k-mer possibili
- **Encoding**: k-mer → intero in base 4: `kmer = Σ seq[i+j] × 4^(k-1-j)`
- **CPU**: sliding window NumPy + bincount
- **GPU**: CuPy con stessa logica

**Applicazioni**: assemblaggio del genoma (de Bruijn graph), rilevamento di varianti, classificazione metagenomi, ricerca di motivi (motif finding).

**Throughput**: Mbp/s (Mega base pairs per second)

### 3. Edit Distance (Levenshtein) — Batch

- **Batch**: 10,000 coppie di sequenze
- **Lunghezza**: L=64 caratteri
- **CPU**: DP rolling 1D per ogni coppia con NumPy
- **GPU CuPy**: tutte le 10,000 coppie in parallelo (batch DP matriciale)
- **GPU Numba**: ogni thread calcola una coppia con DP in local memory (registri GPU)

**Ricorrenza DP:**

```
D[0,j] = j    (j inserimenti)
D[i,0] = i    (i cancellazioni)
D[i,j] = min(D[i-1,j] + 1,          ← cancellazione
             D[i,j-1] + 1,           ← inserimento
             D[i-1,j-1] + cost(i,j)) ← sostituzione (cost=0 se uguale)
```

Il kernel Numba usa `local.array` per il vettore DP (128 elementi in registri GPU, non shared memory) — ogni thread ha il suo array DP privato senza conflitti.

### 4. Scaling Analysis

Smith-Waterman al variare di L: 500, 1000, 2000, 4000 bp. Mostra come lo speedup scala con L² (complessità quadratica).

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab12-bio/src/run_bio.py
```

Il plot viene salvato in `lab12-bio/outputs/lab12_benchmark.png`.

---

## Risultati attesi

Hardware: Intel Core i9 | RTX 4080 16GB | Windows 11

| Algoritmo | Dimensione | Speedup atteso | Note |
|-----------|-----------|----------------|------|
| Smith-Waterman (CuPy) | L=2000 | ~5–20x | Limitato dalla serialità anti-diagonale |
| Smith-Waterman (Numba) | L=2000 | ~10–30x | Kernel nativo, meno overhead |
| K-mer Counting | 10M bp, k=8 | ~20–60x | Memory-bound, bandwidth GPU |
| Edit Distance batch | 10K coppie | ~30–80x | Completamente parallelo per coppia |

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Wavefront parallelism | Le anti-diagonali di una matrice DP sono indipendenti → parallelizzabili |
| Data dependency | H[i,j] dipende da H[i-1,j-1], H[i-1,j], H[i,j-1] → ordine di calcolo obbligato |
| Local memory | Array nei registri GPU: accesso ~0 latenza, ma dimensione limitata (~16KB per thread) |
| K-mer encoding | Sequenza DNA → intero in base 4: permette di usare bincount invece di dict |
| MCUPS | Mega Cell Updates Per Second: metrica standard per alignment; BLAST raggiunge ~10 GCUPS |
| Memory coalescing | Accessi contigui da thread contigui → massima bandwidth DRAM (critico per k-mer) |

---

## Tecnologie

- **CuPy** — operazioni vectorizzate su array DNA (wavefront, bincount)
- **Numba CUDA** — kernel custom anti-diagonal SW, kernel edit distance per-coppia
- **NumPy** — baseline CPU con broadcasting
- **Rich** — tabelle formattate a terminale

---

## Riferimenti

- [Smith & Waterman 1981 — Local sequence alignment](https://doi.org/10.1016/0022-2836(81)90087-5)
- [NVIDIA CUDA Bioinformatics](https://developer.nvidia.com/nvidia-bio-applications)
- [CUDAlign — GPU Smith-Waterman](https://www.academia.edu/4101469)
- [Jellyfish — GPU k-mer counting](https://github.com/gmarcais/Jellyfish)
