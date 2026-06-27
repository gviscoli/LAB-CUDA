"""
lab08-sorting/src/run_sorting.py
=================================
LAB 08 — Sorting & Parallel Primitives
Algoritmi: Radix Sort, Prefix Scan (cumsum), Histogram, Reduction
Paradigma: NumPy CPU baseline → CuPy GPU → Numba CUDA kernel custom

Riferimenti:
  CuPy sort:      https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
  CuPy cumsum:    https://docs.cupy.dev/en/stable/reference/generated/cupy.cumsum.html
  Thrust sort:    https://docs.nvidia.com/cuda/thrust/index.html
  Parallel scan:  https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
  Numba reduce:   https://numba.readthedocs.io/en/stable/cuda/reduction.html
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


# ── Numba CUDA reduction kernel (livello modulo) ───────────────────────────────
# Il decorator @cuda.reduce deve stare a livello modulo — non come closure —
# affinché Numba possa compilarlo staticamente (JIT compile-time).
_REDUCE_BLOCK = 256
try:
    from numba import cuda as _numba_cuda, float32 as _nb_f32

    @_numba_cuda.reduce
    def _sum_reduce(a, b):
        return a + b

    _NUMBA_REDUCE_OK = True
except Exception:
    _sum_reduce = None
    _NUMBA_REDUCE_OK = False


# ──────────────────────────────────────────────────────────────
# 1. Radix Sort
# ──────────────────────────────────────────────────────────────

def lab_radix_sort(N: int = 50_000_000):
    """
    Radix Sort su array float32 di N elementi.
    CPU: numpy.sort (introsort/timsort ibrido)
    GPU: cupy.sort — internamente usa CUB/Thrust radix sort.

    Applicazioni HPC: ordinamento di particle ID, k-NN, range query,
    costruzione di BVH per ray tracing, database columnar.
    """
    rprint(f"\n[bold cyan]Radix Sort — N={N:,}[/bold cyan]")

    data = np.random.rand(N).astype(np.float32)

    def cpu_fn():
        return np.sort(data)

    try:
        import cupy as cp
        data_gpu = cp.array(data)

        def gpu_fn():
            result = cp.sort(data_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="RadixSort",
                      problem_size=N, warmup=2, runs=5)
        rprint(f"  {r}")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("RadixSort", cpu_ms=t.elapsed_ms, problem_size=N)


# ──────────────────────────────────────────────────────────────
# 2. Parallel Prefix Sum (Scan)
# ──────────────────────────────────────────────────────────────

def lab_prefix_scan(N: int = 100_000_000):
    """
    Prefix Sum (inclusive scan / cumsum) su array int32 di N elementi.
    CPU: numpy.cumsum
    GPU: cupy.cumsum — usa l'algoritmo work-efficient di Blelloch (1990).

    Il prefix scan e' il building block piu' importante del GPU computing:
    - BFS: calcolo degli offset dei vicini per layer-by-layer expansion
    - Sparse MatMul: compattazione dei prodotti parziali non-zero
    - Stream Compaction: filtraggio parallelo (es. ray-triangle intersection)
    - Histogram Equalization: CDF cumulativa per equalizzazione
    - Radix Sort stesso: usa scan internamente per calcolare le destinazioni

    Complessita': O(N) work, O(log N) depth — ideale per GPU massive parallelism.
    """
    rprint(f"\n[bold cyan]Prefix Scan (cumsum) — N={N:,}[/bold cyan]")
    rprint("  [dim]Primitive fondamentale: BFS layers, sparse MatMul, stream compaction[/dim]")

    data = np.random.randint(0, 100, N, dtype=np.int32)

    def cpu_fn():
        return np.cumsum(data)

    try:
        import cupy as cp
        data_gpu = cp.array(data)

        def gpu_fn():
            result = cp.cumsum(data_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="PrefixScan",
                      problem_size=N, warmup=2, runs=5)
        rprint(f"  {r}")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("PrefixScan", cpu_ms=t.elapsed_ms, problem_size=N)


# ──────────────────────────────────────────────────────────────
# 3. Parallel Histogram
# ──────────────────────────────────────────────────────────────

def lab_histogram(N: int = 100_000_000, bins: int = 1024):
    """
    Istogramma parallelo su array float32 di N elementi con 'bins' bin.
    CPU: numpy.histogram
    GPU: cupy.histogram — usa atomicAdd con shared memory per ridurre contention.

    Applicazioni HPC: image processing, equalizzazione, analisi dati scientifici,
    medical imaging (CT scan), feature extraction, radiosity rendering.

    La versione GPU richiede attenzione a bank conflicts nella shared memory
    e all'uso di atomicAdd che serializza accessi allo stesso bin.
    """
    rprint(f"\n[bold cyan]Histogram — N={N:,}, bins={bins}[/bold cyan]")

    data = np.random.randn(N).astype(np.float32)

    def cpu_fn():
        return np.histogram(data, bins=bins)

    try:
        import cupy as cp
        data_gpu = cp.array(data)

        def gpu_fn():
            result = cp.histogram(data_gpu, bins=bins)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="Histogram",
                      problem_size=N, warmup=2, runs=5)
        rprint(f"  {r}")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("Histogram", cpu_ms=t.elapsed_ms, problem_size=N)


# ──────────────────────────────────────────────────────────────
# 4. Parallel Reduction (Sum)
# ──────────────────────────────────────────────────────────────

def lab_reduction(N: int = 100_000_000):
    """
    Riduzione parallela (somma) su array float32 di N elementi.
    CPU: numpy.sum
    GPU (CuPy): cupy.sum — usa CUB device-wide reduction con warp shuffle.
    GPU (Numba): kernel @cuda.reduce custom — mostra il pattern a due fasi:
      1. Riduzione intra-block (shared memory + syncthreads)
      2. Riduzione inter-block (atomicAdd o secondo kernel)

    La riduzione parallela e' la primitiva piu' studiata in GPU computing.
    Pattern: ogni thread somma con il suo 'partner' a distanza stride decrescente.
    Warp divergence nell'ultima fase puo' essere evitata con __shfl_down_sync.
    """
    rprint(f"\n[bold cyan]Reduction (sum) — N={N:,}[/bold cyan]")

    data = np.random.rand(N).astype(np.float32)

    def cpu_fn():
        return np.sum(data)

    try:
        import cupy as cp
        data_gpu = cp.array(data)

        def gpu_fn():
            result = cp.sum(data_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="Reduction",
                      problem_size=N, warmup=2, runs=5)
        rprint(f"  {r}")

        # ── Numba custom reduction kernel ──────────────────────
        if _NUMBA_REDUCE_OK:
            rprint("  [dim]Numba @cuda.reduce kernel:[/dim]")
            try:
                data_numba = _numba_cuda.to_device(data)

                def numba_gpu_fn():
                    result = _sum_reduce(data_numba)
                    _numba_cuda.synchronize()
                    return result

                # Warmup
                for _ in range(2):
                    numba_gpu_fn()

                numba_times = []
                for _ in range(5):
                    with GPUTimer() as t_nb:
                        numba_gpu_fn()
                    numba_times.append(t_nb.elapsed_ms)

                numba_ms = float(np.median(numba_times))
                numba_speedup = r.cpu_ms / numba_ms if numba_ms > 0 else 0
                rprint(f"  Numba reduce: {numba_ms:.2f} ms  "
                       f"speedup=[green]{numba_speedup:.1f}x[/green] vs CPU")
            except Exception as e:
                rprint(f"  [yellow]Numba reduce error: {e}[/yellow]")
        else:
            rprint("  [dim]Numba non disponibile — solo CuPy[/dim]")

        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()

        # Prova comunque Numba se disponibile
        if _NUMBA_REDUCE_OK:
            try:
                data_numba = _numba_cuda.to_device(data)

                def numba_gpu_fn():
                    result = _sum_reduce(data_numba)
                    _numba_cuda.synchronize()
                    return result

                for _ in range(2):
                    numba_gpu_fn()

                with GPUTimer() as t_nb:
                    numba_gpu_fn()

                rprint(f"  Numba reduce: {t_nb.elapsed_ms:.2f} ms")
            except Exception as e:
                rprint(f"  [yellow]Numba reduce: {e}[/yellow]")

        return BenchmarkResult("Reduction", cpu_ms=t.elapsed_ms, problem_size=N)


# ──────────────────────────────────────────────────────────────
# 5. Scaling Analysis — Sort GPU vs CPU
# ──────────────────────────────────────────────────────────────

def lab_sorting_scaling():
    """
    Analisi scaling: speedup radix sort al variare della dimensione N.
    Mostra il break-even CPU/GPU e la saturazione delle risorse GPU.
    """
    rprint("\n[bold]Scaling Analysis — Radix Sort GPU vs CPU[/bold]")

    sizes = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000]

    table = Table(title="Scaling: numpy.sort vs cupy.sort", header_style="bold magenta")
    table.add_column("N", justify="right", style="cyan")
    table.add_column("CPU (ms)", justify="right")
    table.add_column("GPU (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("Memoria GPU", justify="right")

    try:
        import cupy as cp

        for N in sizes:
            mem_mb = N * 4 / 1024 / 1024  # float32 = 4 bytes
            mem_str = f"{mem_mb:.0f} MB"

            try:
                data = np.random.rand(N).astype(np.float32)

                def cpu_fn():
                    return np.sort(data)

                data_gpu = cp.array(data)

                def gpu_fn():
                    result = cp.sort(data_gpu)
                    cp.cuda.Stream.null.synchronize()
                    return result

                # Warmup
                for _ in range(1):
                    cpu_fn()
                    gpu_fn()

                # Misura CPU
                cpu_times = []
                for _ in range(3):
                    with CPUTimer() as tc:
                        cpu_fn()
                    cpu_times.append(tc.elapsed_ms)
                cpu_ms = float(np.median(cpu_times))

                # Misura GPU
                gpu_times = []
                for _ in range(3):
                    with GPUTimer() as tg:
                        gpu_fn()
                    gpu_times.append(tg.elapsed_ms)
                gpu_ms = float(np.median(gpu_times))

                speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
                color = (
                    "[green]" if speedup > 10
                    else "[yellow]" if speedup > 2
                    else "[red]"
                )
                table.add_row(
                    f"{N:,}",
                    f"{cpu_ms:.1f}",
                    f"{gpu_ms:.1f}",
                    f"{color}{speedup:.1f}x[/]",
                    mem_str,
                )

                # Cleanup GPU memory
                del data_gpu
                cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                table.add_row(
                    f"{N:,}", "—", "—",
                    f"[red]OOM / {e}[/red]",
                    mem_str,
                )
                rprint(f"  [yellow]N={N:,}: {e} — interrotto[/yellow]")
                break

        console.print(table)

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — scaling analysis saltata[/yellow]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]LAB 08 — Sorting & Parallel Primitives[/bold cyan]")
    console.print("=" * 60)
    console.print(
        "[dim]Algoritmi: Radix Sort, Prefix Scan, Histogram, Reduction[/dim]\n"
    )

    results = []
    results.append(lab_radix_sort())
    results.append(lab_prefix_scan())
    results.append(lab_histogram())
    results.append(lab_reduction())
    lab_sorting_scaling()

    # ── Tabella riepilogo ──────────────────────────────────────
    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
    table.add_column("Primitiva",   style="cyan")
    table.add_column("N",           justify="right")
    table.add_column("CPU (ms)",    justify="right")
    table.add_column("GPU (ms)",    justify="right")
    table.add_column("Speedup",     justify="right", style="green")

    for r in results:
        table.add_row(
            r.name,
            f"{r.problem_size:,}" if r.problem_size > 0 else "—",
            f"{r.cpu_ms:.2f}",
            f"{r.gpu_ms:.2f}" if r.gpu_ms > 0 else "N/A",
            f"{r.speedup:.1f}x" if r.speedup > 0 else "—",
        )

    console.print(table)

    rprint(
        "\n[dim]Prefix Scan e' il building block di: "
        "BFS, sparse MatMul, stream compaction, histogram equalization[/dim]"
    )

    # ── Salva grafico speedup ──────────────────────────────────
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))
        from plotter import plot_cpu_vs_gpu
        valid = [r for r in results if r.gpu_ms > 0 and r.speedup > 0]
        if valid:
            _out = Path(__file__).resolve().parent.parent / "outputs" / "lab08_benchmark.png"
            plot_cpu_vs_gpu(
                valid,
                title="LAB 08 — Sorting & Parallel Primitives: CPU vs GPU",
                save_path=str(_out),
            )
    except Exception as e:
        rprint(f"[dim]Grafico non generato: {e}[/dim]")


if __name__ == "__main__":
    main()
