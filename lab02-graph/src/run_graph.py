"""
lab02-graph/src/run.py
======================
LAB 02 — Graph Algorithms
Algoritmi: BFS, PageRank, Single-Source Shortest Path (SSSP)
Paradigma: NetworkX CPU → CuPy sparse → cuGraph (se disponibile)

Riferimenti:
  cuGraph:   https://docs.rapids.ai/api/cugraph/stable/
  CuPy sparse: https://docs.cupy.dev/en/stable/reference/sparse.html
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))

from timer import CPUTimer, GPUTimer, BenchmarkResult, benchmark
from rich import print as rprint
from rich.table import Table
from rich.console import Console

console = Console()


def generate_graph(n_nodes: int = 100_000, avg_degree: int = 10):
    """
    Genera un grafo random sparso come matrice di adiacenza CSR.
    Usa generazione COO manuale per evitare l'overflow int32 di
    scipy.sparse.random (n*m > 2^31 con n_nodes >= 46341).
    """
    rprint(f"  Generando grafo: {n_nodes:,} nodi, ~{n_nodes*avg_degree:,} archi...")
    from scipy.sparse import csr_matrix

    n_edges = n_nodes * avg_degree
    rows = np.random.randint(0, n_nodes, size=n_edges, dtype=np.int32)
    cols = np.random.randint(0, n_nodes, size=n_edges, dtype=np.int32)
    data = np.ones(n_edges, dtype=np.float32)

    A = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    # Rendi simmetrico (grafo non diretto) e binarizza
    A = A + A.T
    A.data[:] = 1.0
    A.sum_duplicates()

    rprint(f"  Archi effettivi: {A.nnz:,}")
    return A


# ──────────────────────────────────────────────────────────────
# 1. BFS — Breadth First Search
# ──────────────────────────────────────────────────────────────

def lab_bfs(n_nodes: int = 50_000, avg_degree: int = 8):
    """
    BFS dalla sorgente 0 su grafo random.
    GPU: BFS parallelo via CuPy sparse matrix-vector multiplication.
    Applicazione HPC: social network analysis, routing, genomica.
    """
    rprint(f"\n[bold cyan]BFS — {n_nodes:,} nodi[/bold cyan]")
    A = generate_graph(n_nodes, avg_degree)

    def cpu_bfs():
        """BFS CPU via scipy sparse matvec iterato."""
        from scipy.sparse import eye
        visited = np.zeros(n_nodes, dtype=np.float32)
        frontier = np.zeros(n_nodes, dtype=np.float32)
        visited[0] = 1.0
        frontier[0] = 1.0
        level = 0
        while frontier.sum() > 0:
            new_frontier = A.dot(frontier)
            new_frontier = np.where((new_frontier > 0) & (visited == 0), 1.0, 0.0)
            visited = np.where(new_frontier > 0, 1.0, visited)
            frontier = new_frontier
            level += 1
        return visited, level

    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp

        A_gpu = csp.csr_matrix(A)
        visited_gpu = cp.zeros(n_nodes, dtype=cp.float32)
        frontier_gpu = cp.zeros(n_nodes, dtype=cp.float32)

        def gpu_bfs():
            visited = cp.zeros(n_nodes, dtype=cp.float32)
            frontier = cp.zeros(n_nodes, dtype=cp.float32)
            visited[0] = 1.0
            frontier[0] = 1.0
            while frontier.sum() > 0:
                new_frontier = A_gpu.dot(frontier)
                mask = (new_frontier > 0) & (visited == 0)
                new_frontier = cp.where(mask, 1.0, 0.0)
                visited = cp.where(new_frontier > 0, 1.0, visited)
                frontier = new_frontier
            cp.cuda.Stream.null.synchronize()
            return visited

        r = benchmark(cpu_bfs, gpu_bfs, name="BFS", problem_size=n_nodes)
        rprint(f"  {r}")
        return r

    except ImportError:
        with CPUTimer() as t:
            cpu_bfs()
        return BenchmarkResult("BFS", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. PageRank
# ──────────────────────────────────────────────────────────────

def lab_pagerank(n_nodes: int = 50_000, avg_degree: int = 10,
                 iterations: int = 50, damping: float = 0.85):
    """
    PageRank iterativo (power iteration).
    Applicazione HPC: web graph analysis, citation networks, raccomandazione.
    """
    rprint(f"\n[bold cyan]PageRank — {n_nodes:,} nodi, {iterations} iterazioni[/bold cyan]")
    A = generate_graph(n_nodes, avg_degree)

    # Normalizza colonne (colonna stocastica)
    from scipy.sparse import diags
    col_sums = np.array(A.sum(axis=0)).flatten()
    col_sums[col_sums == 0] = 1
    D_inv = diags(1.0 / col_sums)
    M = A.dot(D_inv)  # matrice di transizione

    def cpu_pagerank():
        rank = np.ones(n_nodes, dtype=np.float32) / n_nodes
        for _ in range(iterations):
            rank = damping * M.dot(rank) + (1 - damping) / n_nodes
        return rank

    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp

        M_gpu = csp.csr_matrix(M.astype(np.float32))

        def gpu_pagerank():
            rank = cp.ones(n_nodes, dtype=cp.float32) / n_nodes
            for _ in range(iterations):
                rank = damping * M_gpu.dot(rank) + (1 - damping) / n_nodes
            cp.cuda.Stream.null.synchronize()
            return rank

        r = benchmark(cpu_pagerank, gpu_pagerank, name="PageRank",
                      problem_size=n_nodes)
        rprint(f"  {r}")
        return r

    except ImportError:
        with CPUTimer() as t:
            cpu_pagerank()
        return BenchmarkResult("PageRank", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 3. Betweenness Centrality (approssimata)
# ──────────────────────────────────────────────────────────────

def lab_centrality(n_nodes: int = 10_000, avg_degree: int = 6):
    """
    Centralità approssimata via campionamento BFS.
    Su grafi piccoli per confronto diretto CPU vs GPU.
    """
    rprint(f"\n[bold cyan]Betweenness Centrality (approx) — {n_nodes:,} nodi[/bold cyan]")
    A = generate_graph(n_nodes, avg_degree)
    n_samples = min(100, n_nodes)

    def cpu_centrality():
        from scipy.sparse.csgraph import shortest_path
        sample_idx = np.random.choice(n_nodes, n_samples, replace=False)
        centrality = np.zeros(n_nodes, dtype=np.float32)
        for src in sample_idx[:10]:  # limitato per velocità demo
            dist = shortest_path(A, indices=src, directed=False)
            centrality += (dist < np.inf).astype(np.float32)
        return centrality

    with CPUTimer() as t:
        cpu_centrality()
    rprint(f"  CPU: {t.elapsed_ms:.1f} ms (10 sorgenti campionate)")
    return BenchmarkResult("Centrality", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]=" * 60)
    console.print("[bold cyan]LAB 02 — Graph Algorithms[/bold cyan]")
    console.print("[bold]=" * 60)
    console.print("[dim]Algoritmi: BFS, PageRank, Centrality[/dim]\n")

    results = []
    results.append(lab_bfs())
    results.append(lab_pagerank())
    results.append(lab_centrality())

    table = Table(title="\nRiepilogo", header_style="bold magenta")
    table.add_column("Algoritmo", style="cyan")
    table.add_column("CPU (ms)",  justify="right")
    table.add_column("GPU (ms)",  justify="right")
    table.add_column("Speedup",   justify="right", style="green")

    for r in results:
        table.add_row(
            r.name,
            f"{r.cpu_ms:.2f}",
            f"{r.gpu_ms:.2f}" if r.gpu_ms > 0 else "N/A",
            f"{r.speedup:.1f}x" if r.speedup > 0 else "—",
        )
    console.print(table)

    rprint("\n[dim]💡 Per GPU graph analytics avanzato installa cuGraph (RAPIDS):[/dim]")
    rprint("[dim]   conda install -c rapidsai -c conda-forge cugraph cuda-version=12.4[/dim]")
    rprint("[dim]   Docs: https://docs.rapids.ai/api/cugraph/stable/[/dim]")


if __name__ == "__main__":
    main()
