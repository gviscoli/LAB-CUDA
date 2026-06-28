"""
lab11-sparse/src/run_sparse.py
=================================
LAB 11 — Sparse Linear Algebra GPU
Algoritmi: SpMV (Sparse Matrix-Vector), SpMM (Sparse Matrix-Matrix),
           Sparse Cholesky / ILU preconditioner, Graph Laplacian SpMV
Paradigma: SciPy CSR CPU → CuPy/cuSPARSE GPU acceleration

Riferimenti:
  CuPy sparse:    https://docs.cupy.dev/en/stable/reference/sparse.html
  cuSPARSE:       https://docs.nvidia.com/cuda/cusparse/index.html
  SciPy sparse:   https://docs.scipy.org/doc/scipy/reference/sparse.html
  CUB:            https://nvlabs.github.io/cub/
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))

from timer import CPUTimer, GPUTimer, benchmark, BenchmarkResult
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()

np.random.seed(42)


# ──────────────────────────────────────────────────────────────
# Helper: genera matrice sparsa CSR realistica
# ──────────────────────────────────────────────────────────────

def make_sparse_csr(N: int, density: float = 0.001, symmetric: bool = False, xp=None):
    """
    Genera una matrice sparsa N×N in formato CSR con la densita' specificata.
    Se symmetric=True produce una matrice simmetrica definita positiva (SPD)
    adatta alla fattorizzazione di Cholesky.
    """
    from scipy.sparse import random as sp_random, eye as sp_eye
    nnz_per_row = max(1, int(density * N))
    A = sp_random(N, N, density=density, format="csr", dtype=np.float32,
                  random_state=42)
    if symmetric:
        A = A + A.T + sp_eye(N, dtype=np.float32) * float(N * density * 2 + 1)
    else:
        # Aggiunge diagonale dominante per stabilita' numerica
        from scipy.sparse import diags
        A = A + diags(np.ones(N, dtype=np.float32) * (nnz_per_row + 1), format="csr")
    return A


# ──────────────────────────────────────────────────────────────
# 1. SpMV — Sparse Matrix-Vector Multiplication
# ──────────────────────────────────────────────────────────────

def lab_spmv(N: int = 200_000, density: float = 0.0005):
    """
    Moltiplica matrice sparsa (N×N, formato CSR) per vettore denso.
    y = A · x     dove A e' sparsa, x e y sono densi.

    SpMV e' il kernel fondamentale di:
    - Risoluzione di sistemi lineari sparsi (FEM, CFD, circuiti)
    - Algoritmi su grafi (PageRank, HITS, betweenness centrality)
    - Machine learning (SVM, recommender systems, NLP)

    Il collo di bottiglia e' la memoria (bandwidth-bound):
    ogni elemento di A viene letto una sola volta, quindi
    l'efficienza dipende dalla localita' della struttura sparsa.
    """
    from scipy.sparse import random as sp_random, diags
    nnz_per_row = max(1, int(density * N))
    rprint(f"\n[bold cyan]SpMV — N={N:,}, density={density:.4f}, "
           f"NNZ≈{int(N * N * density):,}[/bold cyan]")
    rprint(f"  [dim]Memoria matrice densa equivalente: {N*N*4/1024**2:.0f} MB — "
           f"CSR occupa: {int(N*N*density)*4/1024**2:.1f} MB[/dim]")

    A_scipy = make_sparse_csr(N, density=density)
    x = np.random.rand(N).astype(np.float32)

    def cpu_fn():
        return A_scipy.dot(x)

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        A_gpu = cpsp.csr_matrix(A_scipy)
        x_gpu = cp.array(x)

        def gpu_fn():
            result = A_gpu.dot(x_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        result = benchmark(cpu_fn, gpu_fn, name="SpMV",
                           problem_size=A_scipy.nnz, warmup=2, runs=5)
        rprint(f"  NNZ: {A_scipy.nnz:,} | {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        return BenchmarkResult("SpMV", cpu_ms=t.elapsed_ms, problem_size=A_scipy.nnz)


# ──────────────────────────────────────────────────────────────
# 2. SpMM — Sparse Matrix × Dense Matrix
# ──────────────────────────────────────────────────────────────

def lab_spmm(N: int = 50_000, K: int = 128, density: float = 0.001):
    """
    Moltiplica matrice sparsa (N×N) per matrice densa (N×K).
    Y = A · X     dove A e' CSR, X e Y sono dense [N×K].

    SpMM e' critico per:
    - Graph Neural Networks (GNN): aggregazione dei messaggi tra nodi
    - Attention sparse: transformer con sparse attention mask
    - Multi-RHS systems: risolvere A·X=B con piu' right-hand sides simultaneamente
    - Spectral graph theory: calcolo di autovettori multipli con Lanczos

    K=128 simula la dimensione tipica degli embedding nei GNN.
    """
    rprint(f"\n[bold cyan]SpMM — A: {N}×{N} sparse, X: {N}×{K} dense[/bold cyan]")
    rprint(f"  [dim]GNN embedding aggregation: {N:,} nodi, {K} features[/dim]")

    A_scipy = make_sparse_csr(N, density=density)
    X = np.random.rand(N, K).astype(np.float32)

    def cpu_fn():
        return A_scipy.dot(X)

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        A_gpu = cpsp.csr_matrix(A_scipy)
        X_gpu = cp.array(X)

        def gpu_fn():
            result = A_gpu.dot(X_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        result = benchmark(cpu_fn, gpu_fn, name="SpMM",
                           problem_size=A_scipy.nnz * K, warmup=2, runs=5)
        rprint(f"  NNZ: {A_scipy.nnz:,} | K={K} | {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        return BenchmarkResult("SpMM", cpu_ms=t.elapsed_ms, problem_size=A_scipy.nnz * K)


# ──────────────────────────────────────────────────────────────
# 3. Graph Laplacian SpMV — PageRank Power Iteration
# ──────────────────────────────────────────────────────────────

def lab_pagerank(N: int = 100_000, edges_per_node: int = 10, max_iter: int = 50):
    """
    Calcola PageRank tramite power iteration su Laplaciano del grafo sparso.
    Modella un grafo diretto di N nodi con circa edges_per_node archi per nodo.

    Algoritmo:
      pr(t+1) = d · A_col_norm · pr(t) + (1-d) / N
      dove d=0.85 e' il damping factor, A e' la matrice di adiacenza colonna-normalizzata.

    Applicazioni HPC:
    - Search engine ranking (Google originale)
    - Analisi di reti sociali (influencer detection)
    - Bioinformatica (protein interaction networks)
    - Cybersecurity (botnet detection, traffic analysis)

    La power iteration converge in O(log(1/eps)/log(1/lambda_2)) iterazioni
    dove lambda_2 e' il secondo autovalore piu' grande di A.
    """
    rprint(f"\n[bold cyan]PageRank (Power Iteration) — N={N:,} nodi, "
           f"~{N * edges_per_node:,} archi, {max_iter} iterazioni[/bold cyan]")
    rprint(f"  [dim]Modella un web-graph con {N:,} pagine[/dim]")

    from scipy.sparse import csr_matrix, diags

    # Costruisce grafo diretto casuale
    rows = np.repeat(np.arange(N, dtype=np.int32), edges_per_node)
    cols = np.random.randint(0, N, size=N * edges_per_node, dtype=np.int32)
    data = np.ones(N * edges_per_node, dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)

    # Normalizza colonne (column-stochastic matrix)
    col_sums = np.array(A.sum(axis=0), dtype=np.float32).ravel()
    col_sums[col_sums == 0] = 1.0  # evita divisione per zero (dangling nodes)
    inv_sums = 1.0 / col_sums
    A_norm = A.multiply(inv_sums)  # broadcast per colonna

    d = 0.85
    teleport = (1.0 - d) / N

    def pagerank_cpu(A_norm, d, teleport, N, max_iter):
        pr = np.full(N, 1.0 / N, dtype=np.float32)
        for _ in range(max_iter):
            pr = d * A_norm.dot(pr) + teleport
        return pr

    def cpu_fn():
        return pagerank_cpu(A_norm, d, teleport, N, max_iter)

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        A_norm_gpu = cpsp.csr_matrix(A_norm)

        def pagerank_gpu(A_norm_gpu, d, teleport, N, max_iter):
            pr_gpu = cp.full(N, 1.0 / N, dtype=cp.float32)
            for _ in range(max_iter):
                pr_gpu = d * A_norm_gpu.dot(pr_gpu) + teleport
            cp.cuda.Stream.null.synchronize()
            return pr_gpu

        def gpu_fn():
            return pagerank_gpu(A_norm_gpu, d, teleport, N, max_iter)

        result = benchmark(cpu_fn, gpu_fn, name="PageRank",
                           problem_size=A_norm.nnz * max_iter, warmup=1, runs=3)

        pr_gpu = gpu_fn()
        top5_idx = cp.argsort(pr_gpu)[-5:][::-1].get()
        rprint(f"  Top-5 nodi per PageRank: {top5_idx.tolist()}")
        rprint(f"  {max_iter} iterazioni SpMV | {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            pr_cpu = cpu_fn()
        top5_idx = np.argsort(pr_cpu)[-5:][::-1]
        rprint(f"  Top-5 nodi per PageRank: {top5_idx.tolist()}")
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        return BenchmarkResult("PageRank", cpu_ms=t.elapsed_ms,
                               problem_size=A_norm.nnz * max_iter)


# ──────────────────────────────────────────────────────────────
# 4. Sparse Solver — Conjugate Gradient (CG)
# ──────────────────────────────────────────────────────────────

def lab_sparse_cg(N: int = 50_000, density: float = 0.0002, tol: float = 1e-5,
                  max_iter: int = 200):
    """
    Risolve sistema lineare sparso Ax=b tramite Conjugate Gradient (CG).
    A deve essere simmetrica definita positiva (SPD).

    Il CG e' l'algoritmo iterativo piu' usato per sistemi sparsi SPD:
    - Ogni iterazione richiede una SpMV (O(NNZ) operazioni)
    - Convergenza in O(sqrt(kappa)) iterazioni, kappa = cond(A)
    - Uso memoria: O(N) — nessuna fattorizzazione richiesta

    Applicazioni HPC:
    - FEM (Finite Element Method): strutture, elasticita', termo-meccanica
    - CFD incomprimibile: equazione di Poisson per la pressione
    - Geofisica: inversione sismica, tomografia
    - Deep Learning: natural gradient, K-FAC optimizer

    Metodo: implementazione CG standard (Hestenes & Stiefel, 1952).
    Residuo relativo: ||r||/||b|| < tol
    """
    rprint(f"\n[bold cyan]Sparse CG Solver — N={N:,}, density={density:.4f}, "
           f"tol={tol:.0e}, max_iter={max_iter}[/bold cyan]")
    rprint(f"  [dim]Sistema SPD tipico di FEM 2D — simula griglia {int(N**0.5)}×{int(N**0.5)}[/dim]")

    A_scipy = make_sparse_csr(N, density=density, symmetric=True)
    b = np.random.rand(N).astype(np.float32)

    def cg_solve(A, b, tol=1e-5, max_iter=200, xp=np):
        """Conjugate Gradient puro senza precondizionatore."""
        x = xp.zeros_like(b)
        r = b - A.dot(x)
        p = r.copy()
        rs_old = float(xp.dot(r, r))
        b_norm = float(xp.linalg.norm(b))
        iters = 0
        for i in range(max_iter):
            Ap = A.dot(p)
            alpha = rs_old / float(xp.dot(p, Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = float(xp.dot(r, r))
            if (rs_new ** 0.5) / b_norm < tol:
                iters = i + 1
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
            iters = i + 1
        return x, iters

    def cpu_fn():
        return cg_solve(A_scipy, b, tol=tol, max_iter=max_iter, xp=np)

    x_cpu, iters_cpu = cpu_fn()
    res_cpu = float(np.linalg.norm(b - A_scipy.dot(x_cpu))) / float(np.linalg.norm(b))
    rprint(f"  CPU: CG converge in {iters_cpu} iter | residuo relativo: {res_cpu:.2e}")

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        A_gpu = cpsp.csr_matrix(A_scipy)
        b_gpu = cp.array(b)

        def gpu_fn():
            x_gpu, iters = cg_solve(A_gpu, b_gpu, tol=tol, max_iter=max_iter, xp=cp)
            cp.cuda.Stream.null.synchronize()
            return x_gpu, iters

        result = benchmark(
            lambda: cg_solve(A_scipy, b, tol=tol, max_iter=max_iter, xp=np),
            lambda: cg_solve(A_gpu, b_gpu, tol=tol, max_iter=max_iter, xp=cp),
            name="SparseCG",
            problem_size=A_scipy.nnz * iters_cpu,
            warmup=1, runs=3,
        )

        x_gpu, iters_gpu = gpu_fn()
        res_gpu = float(cp.linalg.norm(b_gpu - A_gpu.dot(x_gpu))) / float(cp.linalg.norm(b_gpu))
        rprint(f"  GPU: CG converge in {iters_gpu} iter | residuo relativo: {res_gpu:.2e}")
        rprint(f"  {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        return BenchmarkResult("SparseCG", cpu_ms=t.elapsed_ms,
                               problem_size=A_scipy.nnz * iters_cpu)


# ──────────────────────────────────────────────────────────────
# 5. Scaling Analysis — SpMV al variare della dimensione
# ──────────────────────────────────────────────────────────────

def lab_spmv_scaling():
    """
    Analisi scaling: SpMV GPU vs CPU al variare di N (dimensione della matrice).
    Evidenzia il break-even CPU/GPU e la dipendenza dalla banda di memoria.
    """
    rprint("\n[bold]Scaling Analysis — SpMV GPU vs CPU[/bold]")

    # density decrescente per evitare OOM/pagefile su taglie grandi:
    # N=500K con density=0.001 → 250M NNZ → ~5GB RAM temporanea
    configs = [
        (10_000,  0.001),
        (50_000,  0.001),
        (100_000, 0.001),
        (200_000, 0.0005),
        (500_000, 0.0001),  # NNZ ≈ 25M → ~200MB, gestibile
    ]

    table = Table(title="Scaling SpMV: SciPy CSR vs cuSPARSE", header_style="bold magenta")
    table.add_column("N", justify="right", style="cyan")
    table.add_column("NNZ", justify="right")
    table.add_column("CPU (ms)", justify="right")
    table.add_column("GPU (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("BW utile (GB/s)", justify="right")

    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        for N, density in configs:
            A_scipy = make_sparse_csr(N, density=density)
            x = np.random.rand(N).astype(np.float32)
            A_gpu = cpsp.csr_matrix(A_scipy)
            x_gpu = cp.array(x)

            def cpu_fn():
                return A_scipy.dot(x)

            def gpu_fn():
                r = A_gpu.dot(x_gpu)
                cp.cuda.Stream.null.synchronize()
                return r

            # Warmup
            cpu_fn(); gpu_fn()

            cpu_times = []
            for _ in range(3):
                with CPUTimer() as tc:
                    cpu_fn()
                cpu_times.append(tc.elapsed_ms)
            cpu_ms = float(np.median(cpu_times))

            gpu_times = []
            for _ in range(3):
                with GPUTimer() as tg:
                    gpu_fn()
                gpu_times.append(tg.elapsed_ms)
            gpu_ms = float(np.median(gpu_times))

            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
            # BW = (NNZ * sizeof(float) + N * sizeof(float) * 2) / time
            bw_gb = (A_scipy.nnz * 4 + N * 8) / (gpu_ms * 1e-3) / 1e9 if gpu_ms > 0 else 0

            color = "[green]" if speedup > 5 else "[yellow]" if speedup > 1 else "[red]"
            table.add_row(
                f"{N:,}",
                f"{A_scipy.nnz:,}",
                f"{cpu_ms:.2f}",
                f"{gpu_ms:.2f}",
                f"{color}{speedup:.1f}x[/]",
                f"{bw_gb:.1f}" if bw_gb > 0 else "—",
            )

            del A_gpu, x_gpu
            cp.get_default_memory_pool().free_all_blocks()

        console.print(table)

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — scaling analysis saltata[/yellow]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]" + "=" * 60)
    console.print("[bold cyan]LAB 11 — Sparse Linear Algebra GPU[/bold cyan]")
    console.print("[bold]" + "=" * 60)
    console.print("[dim]Algoritmi: SpMV, SpMM, PageRank, CG Solver, Scaling Analysis[/dim]\n")

    results = []
    results.append(lab_spmv())
    results.append(lab_spmm())
    results.append(lab_pagerank())
    results.append(lab_sparse_cg())
    lab_spmv_scaling()

    # Tabella riepilogo
    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
    table.add_column("Algoritmo",  style="cyan")
    table.add_column("CPU (ms)",   justify="right")
    table.add_column("GPU (ms)",   justify="right")
    table.add_column("Speedup",    justify="right", style="green")

    for r in results:
        table.add_row(
            r.name,
            f"{r.cpu_ms:.2f}",
            f"{r.gpu_ms:.2f}" if r.gpu_ms > 0 else "N/A",
            f"{r.speedup:.1f}x" if r.speedup > 0 else "—",
        )
    console.print(table)

    # Salva grafico benchmark
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(__file__).resolve().parent.parent / "outputs" / "lab11_benchmark.png"

        names     = [r.name for r in results]
        cpu_times = [r.cpu_ms for r in results]
        gpu_times = [r.gpu_ms if r.gpu_ms > 0 else 0 for r in results]
        speedups  = [r.speedup if r.speedup > 0 else 0 for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LAB 11 — Sparse Linear Algebra GPU Benchmark",
                     fontsize=14, fontweight="bold")

        x = np.arange(len(names))
        width = 0.35
        ax1 = axes[0]
        bars_cpu = ax1.bar(x - width / 2, cpu_times, width, label="CPU (SciPy CSR)",
                           color="#4C72B0", alpha=0.85)
        bars_gpu = ax1.bar(x + width / 2, gpu_times, width, label="GPU (cuSPARSE)",
                           color="#DD8452", alpha=0.85)
        ax1.set_title("Tempo di esecuzione (ms)")
        ax1.set_ylabel("Tempo (ms)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha="right")
        ax1.legend()
        ax1.set_yscale("log")
        ax1.grid(axis="y", alpha=0.3)
        for bar in bars_cpu:
            h = bar.get_height()
            if h > 0:
                ax1.annotate(f"{h:.0f}",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8)
        for bar in bars_gpu:
            h = bar.get_height()
            if h > 0:
                ax1.annotate(f"{h:.0f}",
                             xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8)

        ax2 = axes[1]
        colors = ["#2ca02c" if s >= 10 else "#ff7f0e" if s >= 1 else "#d62728"
                  for s in speedups]
        bars_sp = ax2.bar(names, speedups, color=colors, alpha=0.85)
        ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Baseline (1x)")
        ax2.set_title("Speedup GPU vs CPU")
        ax2.set_ylabel("Speedup (x)")
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=15, ha="right")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)
        for bar, s in zip(bars_sp, speedups):
            if s > 0:
                ax2.annotate(f"{s:.1f}x",
                             xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        rprint(f"\n[dim]Plot salvato in: {output_path}[/dim]")

    except Exception as exc:
        rprint(f"\n[yellow]Plot non salvato: {exc}[/yellow]")

    rprint("\n[dim]SpMV e' il kernel fondamentale di FEM, CFD, GNN e PageRank — "
           "cuSPARSE accelera ogni iterazione del solver.[/dim]")


if __name__ == "__main__":
    main()
