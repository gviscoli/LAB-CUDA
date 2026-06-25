"""
shared/utils/gpu_info.py — Diagnostica GPU e baseline hardware
Verifica l'ambiente CUDA e stampa un report completo.
Fonte: https://docs.cupy.dev/ | https://docs.nvidia.com/cuda/
"""
import sys
import os
import platform
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()


def check_environment():
    """Verifica e stampa il report completo dell'ambiente HPC."""

    console.print(Panel.fit(
        "[bold cyan]LAB-CUDA — Environment Check[/bold cyan]\n"
        "[dim]Intel Core i9 | 96GB RAM | RTX 4080 16GB | Windows 11[/dim]",
        border_style="cyan"
    ))

    # ── Sistema ──────────────────────────────────────────────
    rprint("\n[bold]Sistema:[/bold]")
    rprint(f"  OS:      {platform.system()} {platform.release()}")
    rprint(f"  Python:  {sys.version.split()[0]}")
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count(logical=True)
    rprint(f"  CPU:     {cpu_count} thread logici")
    rprint(f"  RAM:     {ram_gb:.1f} GB")

    # ── PyTorch / CUDA ────────────────────────────────────────
    rprint("\n[bold]PyTorch / CUDA:[/bold]")
    try:
        import torch
        rprint(f"  PyTorch: {torch.__version__}")
        rprint(f"  CUDA disponibile: {'[green]✅ Sì[/green]' if torch.cuda.is_available() else '[red]❌ No[/red]'}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                vram = p.total_memory / (1024**3)
                rprint(f"  GPU {i}: [cyan]{p.name}[/cyan] | {vram:.1f}GB VRAM | "
                       f"CUDA {p.major}.{p.minor} | "
                       f"{p.multi_processor_count} SM | "
                       f"{p.max_threads_per_block} thread/block")
    except ImportError:
        rprint("  [red]PyTorch non installato[/red]")

    # ── CuPy ─────────────────────────────────────────────────
    rprint("\n[bold]CuPy:[/bold]")
    try:
        import cupy as cp
        rprint(f"  CuPy:    {cp.__version__}")
        rprint(f"  CUDA:    {cp.cuda.runtime.runtimeGetVersion()}")
        mem_free, mem_total = cp.cuda.runtime.memGetInfo()
        rprint(f"  VRAM:    {mem_free/(1024**3):.1f}GB liberi / "
               f"{mem_total/(1024**3):.1f}GB totali")
        rprint("  Status:  [green]✅ Operativo[/green]")
    except ImportError:
        rprint("  [yellow]CuPy non installato — installa con:[/yellow]")
        rprint("  [dim]pip install cupy-cuda12x[/dim]")
    except Exception as e:
        rprint(f"  [red]Errore CuPy: {e}[/red]")

    # ── Numba ─────────────────────────────────────────────────
    rprint("\n[bold]Numba CUDA:[/bold]")
    try:
        from numba import cuda
        import numba
        rprint(f"  Numba:   {numba.__version__}")
        if cuda.is_available():
            gpu = cuda.get_current_device()
            rprint(f"  Device:  [cyan]{gpu.name.decode()}[/cyan]")
            rprint(f"  Compute: {gpu.compute_capability}")
            rprint("  Status:  [green]✅ Operativo[/green]")
        else:
            rprint("  [red]CUDA non disponibile per Numba[/red]")
    except ImportError:
        rprint("  [yellow]Numba non installato[/yellow]")

    # ── Benchmark rapido ──────────────────────────────────────
    rprint("\n[bold]Benchmark rapido (matmul 4096x4096):[/bold]")
    try:
        import numpy as np
        import time

        n = 4096
        a_cpu = np.random.randn(n, n).astype(np.float32)
        b_cpu = np.random.randn(n, n).astype(np.float32)

        # CPU
        t0 = time.perf_counter()
        _ = np.dot(a_cpu, b_cpu)
        t_cpu = time.perf_counter() - t0
        rprint(f"  CPU (NumPy):  {t_cpu*1000:.1f} ms")

        # GPU CuPy
        try:
            import cupy as cp
            a_gpu = cp.array(a_cpu)
            b_gpu = cp.array(b_cpu)
            cp.cuda.Stream.null.synchronize()

            t0 = time.perf_counter()
            _ = cp.dot(a_gpu, b_gpu)
            cp.cuda.Stream.null.synchronize()
            t_gpu = time.perf_counter() - t0

            speedup = t_cpu / t_gpu
            rprint(f"  GPU (CuPy):   {t_gpu*1000:.1f} ms")
            rprint(f"  [bold green]Speedup: {speedup:.1f}x[/bold green]")
        except Exception:
            rprint("  GPU benchmark non disponibile")

    except Exception as e:
        rprint(f"  [red]Errore benchmark: {e}[/red]")

    # ── Tabella riepilogo lab ─────────────────────────────────
    rprint("")
    table = Table(title="Lab Disponibili", header_style="bold magenta")
    table.add_column("Lab",        style="cyan", width=6)
    table.add_column("Dominio",    style="white")
    table.add_column("Algoritmi",  style="dim")
    table.add_column("Entry point")

    labs = [
        ("01", "Numerical",    "FFT, LinAlg, Stencil",      "lab01-numerical/src/run.py"),
        ("02", "Graph",        "BFS, PageRank, SSSP",        "lab02-graph/src/run.py"),
        ("03", "Monte Carlo",  "π, Black-Scholes, Ising",    "lab03-montecarlo/src/run.py"),
        ("04", "ML Kernels",   "MatMul, Conv2D, Attention",  "lab04-ml-kernels/src/run.py"),
        ("05", "Fluid/PDE",    "Heat eq., Navier-Stokes",    "lab05-fluid-pde/src/run.py"),
        ("06", "Benchmark",    "Roofline, scaling",          "lab06-benchmark/src/run.py"),
    ]
    for lab in labs:
        table.add_row(*lab)
    console.print(table)


if __name__ == "__main__":
    check_environment()
