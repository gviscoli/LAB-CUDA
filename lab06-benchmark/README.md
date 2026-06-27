# LAB 06 — Comprehensive Benchmark Suite

Suite di benchmark unificata che aggrega i risultati di tutti i lab precedenti. Fornisce analisi delle prestazioni con il modello Roofline, scaling analysis e tabelle di confronto CPU vs GPU.

---

## Cosa fa questo lab

### 1. Benchmark aggregato (tutti i lab)
- Importa dinamicamente i benchmark ridotti dai lab 01–05
- Esegue ciascuno con dimensioni di problema ridotte per iterazioni rapide
- Se un import fallisce, usa risultati sintetici come fallback
- Produce una tabella unificata con CPU time, GPU time e speedup

### 2. Tabella speedup colorata
| Colore | Soglia | Interpretazione |
|--------|--------|-----------------|
| Rosso | < 5× | GPU non conveniente (overhead domina) |
| Giallo | 5–20× | Vantaggio moderato (spesso memory-bound) |
| Verde | > 20× | Vantaggio significativo (compute-bound) |

### 3. Scaling Analysis
- Esegue MatMul a dimensioni crescenti: 256, 512, 1024, 2048, 4096, 8192
- Identifica il **break-even point**: dimensione minima dove GPU > CPU
- Mostra come lo speedup cresce con la dimensione del problema
- Evidenzia la soglia sotto cui l'overhead CUDA supera il guadagno

### 4. Roofline Model — RTX 4080
Classifica ogni algoritmo nel diagramma Roofline (log-log: TFLOPS vs intensità operazionale):

**Specifiche hardware RTX 4080:**
| Metrica | Valore |
|---------|--------|
| Peak FP32 | 82.58 TFLOPS |
| Peak FP16 | 165.2 TFLOPS |
| Memory bandwidth | 716.8 GB/s |
| VRAM | 16 GB |
| CUDA cores | 9.728 |
| Tensor cores | 304 |
| Ridge point | ~115 FLOP/byte |

**Classificazione algoritmi:**

| Algoritmo | Intensità op. | Classificazione |
|-----------|--------------|-----------------|
| MatMul, Conv2D, Attention | > 115 FLOP/byte | Compute-bound |
| BFS, PageRank, Stencil, Heat Eq. | < 115 FLOP/byte | Memory-bound |

---

## Come eseguire

```powershell
cd C:\DATI\Sviluppo\LAB-CUDA
.venv\Scripts\activate
python lab06-benchmark/src/run_benchmark.py
```

---

## Output generato

I grafici vengono salvati automaticamente in `lab06-benchmark/outputs/` ad ogni esecuzione.

![Benchmark CPU vs GPU](outputs/lab06_complete_benchmark.png)

![Roofline Model RTX 4080](outputs/lab06_roofline.png)

### Console output

```
╔══════════════════════════════════════════════════════════╗
║           CPU vs GPU Benchmark Summary                   ║
╠══════════╦═══════════╦══════════╦═══════════╦═══════════╣
║ Algorithm ║  CPU (ms) ║ GPU (ms) ║ Speedup   ║ Category  ║
╠══════════╬═══════════╬══════════╬═══════════╬═══════════╣
║ FFT 1D   ║      450  ║     12   ║  37.5×    ║ compute   ║
║ DGEMM    ║     8200  ║     95   ║  86.3×  🟢║ compute   ║
║ BFS      ║      380  ║     42   ║   9.0×  🟡║ memory    ║
║ MC-Pi    ║     2100  ║     18   ║ 116.7×  🟢║ compute   ║
║ Heat Eq. ║     4200  ║    180   ║  23.3×  🟢║ memory    ║
╚══════════╩═══════════╩══════════╩═══════════╩═══════════╝
  Average speedup: 54.6×  |  Max speedup: 116.7×
```

---

## Come interpretare il Roofline Model

```
TFLOPS
  │
  │    ██████████████████ Compute Roof (82.6 TFLOPS)
  │   ╱
  │  ╱  Memory Roof (716.8 GB/s × OI)
  │ ╱
  │╱___Ridge point (115 FLOP/byte)
  └────────────────────────────────── OI (FLOP/byte)
       memory-bound │ compute-bound
```

- Algoritmi a **sinistra** del ridge point: limitati dalla bandwidth DRAM
- Algoritmi a **destra** del ridge point: limitati dal throughput computazionale
- Il punto ideale è vicino al tetto (memoria o computazionale)

---

## Concetti chiave

| Concetto | Descrizione |
|----------|-------------|
| Roofline model | Modello grafico che identifica il collo di bottiglia di ogni algoritmo |
| Ridge point | Intensità operazionale dove memoria e compute si equivalgono (115 FLOP/byte per RTX 4080) |
| Operational intensity | FLOP eseguiti per byte letti dalla DRAM |
| Break-even point | Dimensione minima del problema dove il GPU overhead è giustificato |
| Scaling analysis | Studio dello speedup in funzione della dimensione del problema |

---

## Tecnologie

- **tutti i lab** — importazione dinamica dei benchmark
- **Rich** — tabelle colorate in console
- **Matplotlib / Seaborn** — grafici Roofline e bar chart
- **NumPy** — analisi statistica dei risultati
