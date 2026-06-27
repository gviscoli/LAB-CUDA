"""
lab01-numerical/src/run.py
==========================
LAB 01 — Algoritmi Numerici
Algoritmi: FFT, BLAS (matmul, dot), Stencil 2D/3D
Paradigma: CPU baseline → GPU CuPy → GPU Numba kernel custom

Riferimenti:
  CuPy FFT:    https://docs.cupy.dev/en/stable/reference/fft.html
  CuPy linalg: https://docs.cupy.dev/en/stable/reference/linalg.html
  Numba CUDA:  https://numba.readthedocs.io/en/stable/cuda/kernels.html
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

# ── Numba CUDA kernel (livello modulo) ──────────────────────────────────────
# cuda.shared.array richiede shape e dtype costanti a compile-time.
# Il kernel deve stare a livello modulo (non come closure) per consentire
# a Numba di risolvere queste costanti durante la JIT compilation.
_STENCIL_BLOCK = 16
try:
    from numba import cuda as _numba_cuda, float32 as _nb_f32

    @_numba_cuda.jit
    def _stencil_kernel(u, out, N):
        tx = _numba_cuda.threadIdx.x
        ty = _numba_cuda.threadIdx.y
        bx = _numba_cuda.blockIdx.x
        by = _numba_cuda.blockIdx.y
        # shape = (BLOCK+2, BLOCK+2) con BLOCK=16 → (18, 18)
        tile = _numba_cuda.shared.array(shape=(18, 18), dtype=_nb_f32)
        i = by * 16 + ty
        j = bx * 16 + tx
        if i < N and j < N:
            tile[ty + 1, tx + 1] = u[i, j]
        if tx == 0 and j > 0:
            tile[ty + 1, 0] = u[i, j - 1]
        if tx == 15 and j < N - 1:        # BLOCK-1 = 15
            tile[ty + 1, 17] = u[i, j + 1]  # BLOCK+1 = 17
        if ty == 0 and i > 0:
            tile[0, tx + 1] = u[i - 1, j]
        if ty == 15 and i < N - 1:        # BLOCK-1 = 15
            tile[17, tx + 1] = u[i + 1, j]  # BLOCK+1 = 17
        _numba_cuda.syncthreads()
        if 1 <= i < N - 1 and 1 <= j < N - 1:
            out[i, j] = (tile[ty,     tx + 1] + tile[ty + 2, tx + 1] +
                         tile[ty + 1, tx]     + tile[ty + 1, tx + 2] -
                         4.0 * tile[ty + 1, tx + 1])

    _NUMBA_CUDA_OK = True
except Exception:
    _stencil_kernel = None
    _NUMBA_CUDA_OK = False


# ──────────────────────────────────────────────────────────────
# 1. FFT
# ──────────────────────────────────────────────────────────────

def lab_fft(N: int = 2**22):
    """
    FFT 1D su segnale di N campioni.
    Applicazione HPC: signal processing, simulazioni fisiche, PDE spettrali.
    """
    rprint(f"\n[bold cyan]FFT 1D — N={N:,}[/bold cyan]")
    signal = np.random.randn(N).astype(np.complex64)

    def cpu_fn():
        return np.fft.fft(signal)

    try:
        import cupy as cp
        signal_gpu = cp.array(signal)
        def gpu_fn():
            result = cp.fft.fft(signal_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="FFT-1D", problem_size=N,
                      bytes_accessed=N * 8 * 2)
        rprint(f"  {r}")
        return r
    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("FFT-1D", cpu_ms=t.elapsed_ms)


def lab_fft2d(N: int = 4096):
    """FFT 2D su immagine NxN — tipico in imaging e PDE spettrali."""
    rprint(f"\n[bold cyan]FFT 2D — {N}x{N}[/bold cyan]")
    img = np.random.randn(N, N).astype(np.complex64)

    def cpu_fn():
        return np.fft.fft2(img)

    try:
        import cupy as cp
        img_gpu = cp.array(img)
        def gpu_fn():
            result = cp.fft.fft2(img_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="FFT-2D", problem_size=N*N)
        rprint(f"  {r}")
        return r
    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("FFT-2D", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. Linear Algebra (BLAS)
# ──────────────────────────────────────────────────────────────

def lab_matmul(N: int = 8192):
    """
    Matrix multiplication NxN (DGEMM).
    Algoritmo fondamentale HPC: deep learning, FEM, simulazioni.
    RTX 4080 Tensor Cores: teorico ~82 TFLOPS FP32.
    """
    rprint(f"\n[bold cyan]MatMul DGEMM — {N}x{N}[/bold cyan]")
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    def cpu_fn():
        return np.dot(A, B)

    try:
        import cupy as cp
        A_gpu = cp.array(A)
        B_gpu = cp.array(B)
        def gpu_fn():
            result = cp.dot(A_gpu, B_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        flops = 2 * N**3
        r = benchmark(cpu_fn, gpu_fn, name="MatMul", problem_size=N,
                      bytes_accessed=3 * N * N * 4)
        achieved_tflops = flops / (r.gpu_ms / 1000) / 1e12
        rprint(f"  {r}")
        rprint(f"  GPU TFLOPS: [green]{achieved_tflops:.1f}[/green] "
               f"/ 82.6 teorici = [cyan]{achieved_tflops/82.6*100:.0f}%[/cyan] efficienza")
        return r
    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("MatMul", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 3. Stencil 2D (Laplaciano)
# ──────────────────────────────────────────────────────────────

def lab_stencil_2d(N: int = 4096):
    """
    Stencil 2D a 5 punti (operatore di Laplace).
    Applicazione HPC: simulazioni fisiche, PDE alle differenze finite.
    Pattern classico memory-bound — interessante per roofline analysis.
    """
    rprint(f"\n[bold cyan]Stencil 2D Laplaciano — {N}x{N}[/bold cyan]")
    u = np.random.randn(N, N).astype(np.float32)

    def cpu_stencil(u):
        return (u[:-2, 1:-1] + u[2:, 1:-1] +
                u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])

    def cpu_fn():
        return cpu_stencil(u)

    try:
        import cupy as cp
        u_gpu = cp.array(u)

        def gpu_fn():
            result = (u_gpu[:-2, 1:-1] + u_gpu[2:, 1:-1] +
                      u_gpu[1:-1, :-2] + u_gpu[1:-1, 2:] - 4 * u_gpu[1:-1, 1:-1])
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="Stencil-2D", problem_size=N*N,
                      bytes_accessed=5 * N * N * 4)
        rprint(f"  {r}")
        return r
    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("Stencil-2D", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# Numba CUDA kernel custom — Stencil ottimizzato
# ──────────────────────────────────────────────────────────────

def lab_stencil_numba(N: int = 4096):
    """
    Stencil 2D implementato con kernel Numba CUDA custom.
    Usa shared memory per ridurre gli accessi DRAM.
    Fonte: https://numba.readthedocs.io/en/stable/cuda/kernels.html
    """
    rprint(f"\n[bold cyan]Stencil 2D Numba CUDA (shared memory) — {N}x{N}[/bold cyan]")

    if not _NUMBA_CUDA_OK:
        rprint("  [yellow]Numba non disponibile[/yellow]")
        return BenchmarkResult("Stencil-Numba", cpu_ms=0)

    import math

    u_cpu = np.random.randn(N, N).astype(np.float32)

    # CPU baseline (stesso operatore di lab_stencil_2d)
    def cpu_fn():
        return (u_cpu[:-2, 1:-1] + u_cpu[2:, 1:-1] +
                u_cpu[1:-1, :-2] + u_cpu[1:-1, 2:] - 4 * u_cpu[1:-1, 1:-1])

    with CPUTimer() as t:
        for _ in range(3):
            cpu_fn()
    cpu_ms = t.elapsed_ms / 3

    # GPU Numba kernel
    u_gpu   = _numba_cuda.to_device(u_cpu)
    out_gpu = _numba_cuda.device_array_like(u_gpu)
    threads = (_STENCIL_BLOCK, _STENCIL_BLOCK)
    blocks  = (math.ceil(N / _STENCIL_BLOCK), math.ceil(N / _STENCIL_BLOCK))

    def numba_fn():
        _stencil_kernel[blocks, threads](u_gpu, out_gpu, N)
        _numba_cuda.synchronize()

    for _ in range(3):
        numba_fn()

    with GPUTimer() as t:
        for _ in range(10):
            numba_fn()
    gpu_ms = t.elapsed_ms / 10

    speedup   = cpu_ms / gpu_ms
    bandwidth = 5 * N * N * 4 / (gpu_ms / 1000) / 1e9
    rprint(f"  Stencil-Numba: CPU={cpu_ms:.2f}ms | GPU={gpu_ms:.2f}ms | "
           f"Speedup=[green]{speedup:.1f}x[/green]")
    rprint(f"  Bandwidth effettiva: [cyan]{bandwidth:.1f} GB/s[/cyan] / 716.8 teorici")

    return BenchmarkResult("Stencil-Numba", cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                           speedup=speedup, problem_size=N*N,
                           throughput_gbs=bandwidth)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]=" * 60)
    console.print("[bold cyan]LAB 01 — Numerical Computing[/bold cyan]")
    console.print("[bold]=" * 60)
    console.print("[dim]Algoritmi: FFT, MatMul BLAS, Stencil 2D[/dim]\n")

    results = []
    results.append(lab_fft())
    results.append(lab_fft2d())
    results.append(lab_matmul())
    results.append(lab_stencil_2d())
    results.append(lab_stencil_numba())

    # Tabella riepilogo
    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
    table.add_column("Algoritmo",  style="cyan")
    table.add_column("CPU (ms)",   justify="right")
    table.add_column("GPU (ms)",   justify="right")
    table.add_column("Speedup",    justify="right", style="green")

    for r in results:
        if r.gpu_ms > 0:
            table.add_row(
                r.name,
                f"{r.cpu_ms:.2f}",
                f"{r.gpu_ms:.2f}",
                f"{r.speedup:.1f}x",
            )

    console.print(table)

    # Salva grafici
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))
        from plotter import plot_cpu_vs_gpu
        valid = [r for r in results if r.gpu_ms > 0]
        if valid:
            plot_cpu_vs_gpu(
                valid,
                title="LAB 01 — Numerical: CPU vs GPU",
                save_path=str(Path(__file__).resolve().parent.parent / "outputs" / "lab01_benchmark.png")
            )
    except Exception as e:
        rprint(f"[dim]Grafico non generato: {e}[/dim]")


if __name__ == "__main__":
    main()
