"""
lab04-ml-kernels/src/run.py
============================
LAB 04 — Machine Learning Kernels
Algoritmi: MatMul ottimizzato, Conv2D, Self-Attention (Transformer)
Paradigma: PyTorch → Numba kernel custom → Triton (se disponibile)

Riferimenti:
  Numba CUDA matmul: https://numba.readthedocs.io/en/stable/cuda/examples.html
  Triton tutorial:   https://triton-lang.org/main/getting-started/tutorials/
  cuBLAS:            https://docs.nvidia.com/cuda/cublas/
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))

from timer import CPUTimer, GPUTimer, benchmark, BenchmarkResult
from rich import print as rprint
from rich.table import Table
from rich.console import Console
import math

console = Console()


# ──────────────────────────────────────────────────────────────
# 1. MatMul con Numba CUDA (tiled shared memory)
# ──────────────────────────────────────────────────────────────

def lab_matmul_numba(N: int = 2048):
    """
    Matrix multiplication con kernel Numba CUDA tiled.
    Usa shared memory per ridurre accessi DRAM.
    Confronta: NumPy CPU vs cuBLAS (CuPy) vs Numba custom kernel.
    """
    rprint(f"\n[bold cyan]MatMul Tiled — {N}x{N}[/bold cyan]")

    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # CPU baseline
    def cpu_fn():
        return np.dot(A, B)

    try:
        from numba import cuda

        TILE = 16

        @cuda.jit
        def matmul_tiled(A, B, C, N):
            """Kernel matmul con shared memory tiling."""
            tx = cuda.threadIdx.x
            ty = cuda.threadIdx.y
            bx = cuda.blockIdx.x
            by = cuda.blockIdx.y

            tile_A = cuda.shared.array(shape=(TILE, TILE), dtype=np.float32)
            tile_B = cuda.shared.array(shape=(TILE, TILE), dtype=np.float32)

            row = by * TILE + ty
            col = bx * TILE + tx

            acc = np.float32(0.0)
            for t in range((N + TILE - 1) // TILE):
                if row < N and t * TILE + tx < N:
                    tile_A[ty, tx] = A[row, t * TILE + tx]
                else:
                    tile_A[ty, tx] = 0.0

                if col < N and t * TILE + ty < N:
                    tile_B[ty, tx] = B[t * TILE + ty, col]
                else:
                    tile_B[ty, tx] = 0.0

                cuda.syncthreads()

                for k in range(TILE):
                    acc += tile_A[ty, k] * tile_B[k, tx]

                cuda.syncthreads()

            if row < N and col < N:
                C[row, col] = acc

        A_gpu = cuda.to_device(A)
        B_gpu = cuda.to_device(B)
        C_gpu = cuda.device_array((N, N), dtype=np.float32)

        threads = (TILE, TILE)
        blocks  = (math.ceil(N / TILE), math.ceil(N / TILE))

        def numba_fn():
            matmul_tiled[blocks, threads](A_gpu, B_gpu, C_gpu, N)
            cuda.synchronize()

        # Warmup
        for _ in range(3):
            numba_fn()

        with GPUTimer() as t:
            for _ in range(5):
                numba_fn()
        gpu_ms = t.elapsed_ms / 5

        with CPUTimer() as tc:
            for _ in range(5):
                cpu_fn()
        cpu_ms = tc.elapsed_ms / 5

        flops = 2 * N**3
        tflops = flops / (gpu_ms / 1000) / 1e12
        speedup = cpu_ms / gpu_ms

        rprint(f"  CPU:          {cpu_ms:.1f} ms")
        rprint(f"  Numba CUDA:   {gpu_ms:.1f} ms")
        rprint(f"  Speedup:      [green]{speedup:.1f}x[/green]")
        rprint(f"  TFLOPS:       [cyan]{tflops:.2f}[/cyan] / 82.6 teorici")

        return BenchmarkResult("MatMul-Numba", cpu_ms=cpu_ms, gpu_ms=gpu_ms,
                               speedup=speedup, problem_size=N)

    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        rprint(f"  CPU: {t.elapsed_ms:.1f} ms (Numba non disponibile)")
        return BenchmarkResult("MatMul-Numba", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. Conv2D — Convoluzione 2D
# ──────────────────────────────────────────────────────────────

def lab_conv2d(batch: int = 16, C: int = 64, H: int = 224,
               W: int = 224, K: int = 64, ksize: int = 3):
    """
    Convoluzione 2D batch via PyTorch (cuDNN sotto).
    Applicazione HPC: deep learning, image processing, segnali.
    """
    rprint(f"\n[bold cyan]Conv2D — batch={batch}, {C}→{K} ch, {H}x{W}, kernel {ksize}x{ksize}[/bold cyan]")

    try:
        import torch

        x_cpu  = torch.randn(batch, C, H, W)
        w_cpu  = torch.randn(K, C, ksize, ksize)
        bias   = torch.randn(K)
        pad    = ksize // 2

        def cpu_fn():
            return torch.nn.functional.conv2d(x_cpu, w_cpu, bias, padding=pad)

        if torch.cuda.is_available():
            x_gpu = x_cpu.cuda()
            w_gpu = w_cpu.cuda()
            b_gpu = bias.cuda()

            def gpu_fn():
                result = torch.nn.functional.conv2d(x_gpu, w_gpu, b_gpu, padding=pad)
                torch.cuda.synchronize()
                return result

            r = benchmark(cpu_fn, gpu_fn, name="Conv2D", problem_size=batch*C*H*W)
            rprint(f"  {r}")
            return r
        else:
            with CPUTimer() as t:
                cpu_fn()
            rprint(f"  CPU: {t.elapsed_ms:.1f} ms (CUDA non disponibile)")
            return BenchmarkResult("Conv2D", cpu_ms=t.elapsed_ms)

    except ImportError:
        rprint("  [yellow]PyTorch non disponibile[/yellow]")
        return BenchmarkResult("Conv2D")


# ──────────────────────────────────────────────────────────────
# 3. Self-Attention (Transformer)
# ──────────────────────────────────────────────────────────────

def lab_attention(batch: int = 8, seq_len: int = 1024,
                  d_model: int = 512, n_heads: int = 8):
    """
    Multi-Head Self-Attention via PyTorch.
    Kernel fondamentale dei Transformer (LLM, Vision Transformer).
    Applicazione HPC: NLP, computer vision, fisica (Equivariant Transformers).
    """
    rprint(f"\n[bold cyan]Self-Attention — batch={batch}, seq={seq_len}, d={d_model}, heads={n_heads}[/bold cyan]")

    try:
        import torch
        import torch.nn as nn

        mha_cpu = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        x_cpu   = torch.randn(batch, seq_len, d_model)

        def cpu_fn():
            with torch.no_grad():
                return mha_cpu(x_cpu, x_cpu, x_cpu)

        if torch.cuda.is_available():
            # Modello GPU separato: .cuda() sposta in-place, quindi mha_cpu
            # diventerebbe GPU. Creiamo un'istanza indipendente.
            mha_gpu = nn.MultiheadAttention(d_model, n_heads, batch_first=True).cuda()
            x_gpu   = x_cpu.cuda()

            def gpu_fn():
                with torch.no_grad():
                    result = mha_gpu(x_gpu, x_gpu, x_gpu)
                torch.cuda.synchronize()
                return result

            r = benchmark(cpu_fn, gpu_fn, name="Attention",
                          problem_size=batch*seq_len*d_model, warmup=2, runs=5)
            rprint(f"  {r}")

            # Flash Attention check
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with torch.amp.autocast('cuda'):
                    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        _ = torch.nn.functional.scaled_dot_product_attention(
                            x_gpu.view(batch, n_heads, seq_len, d_model//n_heads),
                            x_gpu.view(batch, n_heads, seq_len, d_model//n_heads),
                            x_gpu.view(batch, n_heads, seq_len, d_model//n_heads),
                        )
                rprint("  [green]✅ Flash Attention disponibile (RTX 4080 Ada)[/green]")
            except Exception:
                pass

            return r
        else:
            with CPUTimer() as t:
                cpu_fn()
            return BenchmarkResult("Attention", cpu_ms=t.elapsed_ms)

    except ImportError:
        rprint("  [yellow]PyTorch non disponibile[/yellow]")
        return BenchmarkResult("Attention")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]=" * 60)
    console.print("[bold cyan]LAB 04 — ML Kernels[/bold cyan]")
    console.print("[bold]=" * 60)
    console.print("[dim]Algoritmi: MatMul (Numba tiled), Conv2D (cuDNN), Self-Attention[/dim]\n")

    results = []
    results.append(lab_matmul_numba())
    results.append(lab_conv2d())
    results.append(lab_attention())

    table = Table(title="\nRiepilogo", header_style="bold magenta")
    table.add_column("Kernel",    style="cyan")
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

    rprint("\n[dim]💡 Triton (OpenAI) permette kernel GPU custom più veloci di Numba.[/dim]")
    rprint("[dim]   ⚠️  Triton non supporta Windows nativo — richiede Linux o WSL2.[/dim]")


if __name__ == "__main__":
    main()
