"""
lab07-nbody/src/run_nbody.py
=============================
LAB 07 — N-Body Gravitational Simulation
Algoritmi: Brute-force O(N²) CuPy, Tiled Numba CUDA kernel, Scaling analysis
Paradigma: NumPy chunked → CuPy full N×N → Numba CUDA shared-memory tiling

Riferimenti:
  N-Body CUDA: https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
  Numba shared memory: https://numba.readthedocs.io/en/stable/cuda/memory.html
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))

from timer import CPUTimer, GPUTimer, benchmark, BenchmarkResult
from rich import print as rprint
from rich.table import Table
from rich.console import Console
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()

# ──────────────────────────────────────────────────────────────
# Numba kernel — definito a scope di modulo (richiesto da cuda.shared.array)
# ──────────────────────────────────────────────────────────────

_NBODY_TILE = 256

try:
    from numba import cuda as _numba_cuda, float32 as _nb_f32
    import math as _math

    @_numba_cuda.jit
    def _nbody_kernel(pos, mass, force, N, eps2):
        tx = _numba_cuda.threadIdx.x
        bx = _numba_cuda.blockIdx.x
        tile_pos  = _numba_cuda.shared.array(shape=(_NBODY_TILE, 3), dtype=_nb_f32)
        tile_mass = _numba_cuda.shared.array(shape=(_NBODY_TILE,),    dtype=_nb_f32)
        i = bx * _NBODY_TILE + tx
        xi = yi = zi = _nb_f32(0.0)
        if i < N:
            xi = pos[i, 0]; yi = pos[i, 1]; zi = pos[i, 2]
        fx = fy = fz = _nb_f32(0.0)
        for t in range((N + _NBODY_TILE - 1) // _NBODY_TILE):
            j = t * _NBODY_TILE + tx
            if j < N:
                tile_pos[tx, 0] = pos[j, 0]
                tile_pos[tx, 1] = pos[j, 1]
                tile_pos[tx, 2] = pos[j, 2]
                tile_mass[tx]   = mass[j]
            else:
                tile_pos[tx, 0] = _nb_f32(0.0)
                tile_pos[tx, 1] = _nb_f32(0.0)
                tile_pos[tx, 2] = _nb_f32(0.0)
                tile_mass[tx]   = _nb_f32(0.0)
            _numba_cuda.syncthreads()
            if i < N:
                for k in range(_NBODY_TILE):
                    dx = tile_pos[k, 0] - xi
                    dy = tile_pos[k, 1] - yi
                    dz = tile_pos[k, 2] - zi
                    dist2 = dx*dx + dy*dy + dz*dz + eps2
                    inv_d3 = _nb_f32(1.0) / (dist2 * _math.sqrt(dist2))
                    m = tile_mass[k]
                    fx += m * dx * inv_d3
                    fy += m * dy * inv_d3
                    fz += m * dz * inv_d3
            _numba_cuda.syncthreads()
        if i < N:
            force[i, 0] = fx; force[i, 1] = fy; force[i, 2] = fz

    _NUMBA_NBODY_OK = True
except Exception:
    _nbody_kernel = None
    _NUMBA_NBODY_OK = False


# ──────────────────────────────────────────────────────────────
# 1. Brute-force N-Body — NumPy vs CuPy
# ──────────────────────────────────────────────────────────────

def lab_nbody_bruteforce(N: int = 4096):
    """
    N-Body gravitazionale brute-force O(N²).
    CPU: NumPy chunked per evitare OOM (matrice N×N troppo grande per RAM).
    GPU: CuPy full N×N — la VRAM RTX 4080 (16GB) regge agevolmente.
    Misura GFLOPS effettivi: 26 operazioni floating point per coppia.
    """
    rprint(f"\n[bold cyan]N-Body Brute-Force — N={N:,} corpi[/bold cyan]")

    np.random.seed(42)
    pos  = np.random.randn(N, 3).astype(np.float32)
    mass = np.random.rand(N).astype(np.float32) + 0.1   # massa > 0
    eps2 = np.float32(1e-4)   # softening al quadrato (evita singolarità)

    # ── CPU: chunked NumPy ──────────────────────────────────────
    CHUNK = 512

    def nbody_cpu(pos, mass, eps2):
        fx = np.zeros(N, np.float32)
        fy = np.zeros(N, np.float32)
        fz = np.zeros(N, np.float32)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            dx = pos[start:end, 0:1] - pos[:, 0]   # (CHUNK, N)
            dy = pos[start:end, 1:2] - pos[:, 1]
            dz = pos[start:end, 2:3] - pos[:, 2]
            dist2 = dx**2 + dy**2 + dz**2 + eps2
            inv_d3 = 1.0 / (dist2 * np.sqrt(dist2))
            fx[start:end] = np.sum(mass * dx * inv_d3, axis=1)
            fy[start:end] = np.sum(mass * dy * inv_d3, axis=1)
            fz[start:end] = np.sum(mass * dz * inv_d3, axis=1)
        return np.stack([fx, fy, fz], axis=1)

    def cpu_fn():
        return nbody_cpu(pos, mass, eps2)

    try:
        import cupy as cp

        pos_gpu  = cp.array(pos)
        mass_gpu = cp.array(mass)

        # ── GPU: CuPy full N×N ─────────────────────────────────
        def nbody_gpu(pos, mass, eps2, cp):
            dx = pos[:, 0:1] - pos[:, 0]
            dy = pos[:, 1:2] - pos[:, 1]
            dz = pos[:, 2:3] - pos[:, 2]
            dist2 = dx**2 + dy**2 + dz**2 + eps2
            inv_d3 = 1.0 / (dist2 * cp.sqrt(dist2))
            return cp.stack(
                [cp.sum(mass * dx * inv_d3, 1),
                 cp.sum(mass * dy * inv_d3, 1),
                 cp.sum(mass * dz * inv_d3, 1)],
                axis=1
            )

        def gpu_fn():
            f = nbody_gpu(pos_gpu, mass_gpu, eps2, cp)
            cp.cuda.Stream.null.synchronize()
            return f

        r = benchmark(cpu_fn, gpu_fn, name="N-Body BF",
                      problem_size=N, warmup=2, runs=5)

        # Calcolo GFLOPS (26 flop per coppia i-j)
        flops  = N * N * 26
        gflops = flops / (r.gpu_ms / 1000) / 1e9
        rprint(f"  Throughput GPU: [green]{gflops:.1f} GFLOPS[/green]")
        rprint(f"  {r}")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("N-Body BF", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. N-Body con kernel Numba CUDA tiled (shared memory)
# ──────────────────────────────────────────────────────────────

def lab_nbody_numba(N: int = 4096):
    """
    N-Body con kernel Numba CUDA che usa shared memory tiling.
    Ogni tile di 256 corpi viene caricato in shared memory per
    ridurre gli accessi a global memory: bandwidth saving ~256×.
    Confronto vs CPU NumPy chunked per misurare lo speedup reale.
    """
    rprint(f"\n[bold cyan]N-Body Numba CUDA Tiled — N={N:,} corpi, tile={_NBODY_TILE}[/bold cyan]")

    if not _NUMBA_NBODY_OK:
        rprint("  [yellow]Numba CUDA non disponibile — skip[/yellow]")
        return BenchmarkResult("N-Body Numba", notes="Numba non disponibile")

    np.random.seed(42)
    pos  = np.random.randn(N, 3).astype(np.float32)
    mass = np.random.rand(N).astype(np.float32) + 0.1
    eps2 = np.float32(1e-4)

    import numba.cuda as nb_cuda

    # Trasferimento dati su device
    d_pos   = nb_cuda.to_device(pos)
    d_mass  = nb_cuda.to_device(mass)
    d_force = nb_cuda.device_array((N, 3), dtype=np.float32)

    blocks  = (N + _NBODY_TILE - 1) // _NBODY_TILE
    threads = _NBODY_TILE

    # ── CPU baseline (chunked NumPy) ──────────────────────────
    CHUNK = 512
    eps2_np = np.float32(1e-4)

    def cpu_fn():
        fx = np.zeros(N, np.float32)
        fy = np.zeros(N, np.float32)
        fz = np.zeros(N, np.float32)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            dx = pos[start:end, 0:1] - pos[:, 0]
            dy = pos[start:end, 1:2] - pos[:, 1]
            dz = pos[start:end, 2:3] - pos[:, 2]
            dist2 = dx**2 + dy**2 + dz**2 + eps2_np
            inv_d3 = 1.0 / (dist2 * np.sqrt(dist2))
            fx[start:end] = np.sum(mass * dx * inv_d3, axis=1)
            fy[start:end] = np.sum(mass * dy * inv_d3, axis=1)
            fz[start:end] = np.sum(mass * dz * inv_d3, axis=1)
        return np.stack([fx, fy, fz], axis=1)

    # ── GPU: kernel Numba CUDA ────────────────────────────────
    def gpu_fn():
        _nbody_kernel[blocks, threads](d_pos, d_mass, d_force, N, eps2)
        nb_cuda.synchronize()

    # Warmup per compilazione JIT
    rprint("  [dim]Compilazione JIT Numba (prima run)...[/dim]")
    gpu_fn()

    r = benchmark(cpu_fn, gpu_fn, name="N-Body Numba",
                  problem_size=N, warmup=2, runs=5)

    flops  = N * N * 26
    gflops = flops / (r.gpu_ms / 1000) / 1e9
    rprint(f"  Throughput GPU (Numba): [green]{gflops:.1f} GFLOPS[/green]")
    rprint(f"  {r}")
    return r


# ──────────────────────────────────────────────────────────────
# 3. Scaling analysis — N = [512, 1024, 2048, 4096]
# ──────────────────────────────────────────────────────────────

def lab_nbody_scaling():
    """
    Analisi di scaling: misura speedup GPU vs CPU al variare di N.
    L'N-Body è O(N²): al crescere di N, la GPU scala molto meglio
    della CPU perché satura i suoi migliaia di core.
    """
    rprint(f"\n[bold cyan]N-Body Scaling Analysis — N = [512, 1024, 2048, 4096][/bold cyan]")

    sizes = [512, 1024, 2048, 4096]
    eps2  = np.float32(1e-4)
    CHUNK = 512

    try:
        import cupy as cp
    except ImportError:
        rprint("  [yellow]CuPy non disponibile — skip scaling[/yellow]")
        return []

    scaling_results = []

    table = Table(title="Scaling N-Body CuPy vs NumPy", header_style="bold magenta")
    table.add_column("N",         style="cyan",  justify="right")
    table.add_column("CPU (ms)",  justify="right")
    table.add_column("GPU (ms)",  justify="right")
    table.add_column("Speedup",   justify="right", style="green")
    table.add_column("GFLOPS",    justify="right", style="yellow")

    for N in sizes:
        np.random.seed(42)
        pos  = np.random.randn(N, 3).astype(np.float32)
        mass = np.random.rand(N).astype(np.float32) + 0.1

        pos_gpu  = cp.array(pos)
        mass_gpu = cp.array(mass)

        def cpu_fn():
            fx = np.zeros(N, np.float32)
            fy = np.zeros(N, np.float32)
            fz = np.zeros(N, np.float32)
            for start in range(0, N, CHUNK):
                end = min(start + CHUNK, N)
                dx = pos[start:end, 0:1] - pos[:, 0]
                dy = pos[start:end, 1:2] - pos[:, 1]
                dz = pos[start:end, 2:3] - pos[:, 2]
                dist2 = dx**2 + dy**2 + dz**2 + eps2
                inv_d3 = 1.0 / (dist2 * np.sqrt(dist2))
                fx[start:end] = np.sum(mass * dx * inv_d3, axis=1)
                fy[start:end] = np.sum(mass * dy * inv_d3, axis=1)
                fz[start:end] = np.sum(mass * dz * inv_d3, axis=1)
            return np.stack([fx, fy, fz], axis=1)

        def gpu_fn():
            dx = pos_gpu[:, 0:1] - pos_gpu[:, 0]
            dy = pos_gpu[:, 1:2] - pos_gpu[:, 1]
            dz = pos_gpu[:, 2:3] - pos_gpu[:, 2]
            dist2 = dx**2 + dy**2 + dz**2 + eps2
            inv_d3 = 1.0 / (dist2 * cp.sqrt(dist2))
            f = cp.stack([cp.sum(mass_gpu*dx*inv_d3, 1),
                          cp.sum(mass_gpu*dy*inv_d3, 1),
                          cp.sum(mass_gpu*dz*inv_d3, 1)], axis=1)
            cp.cuda.Stream.null.synchronize()
            return f

        r = benchmark(cpu_fn, gpu_fn, name=f"N={N}",
                      problem_size=N, warmup=2, runs=5)

        flops  = N * N * 26
        gflops = flops / (r.gpu_ms / 1000) / 1e9

        table.add_row(
            f"{N:,}",
            f"{r.cpu_ms:.1f}",
            f"{r.gpu_ms:.2f}",
            f"{r.speedup:.1f}x",
            f"{gflops:.1f}",
        )

        scaling_results.append((N, r.cpu_ms, r.gpu_ms, r.speedup, gflops))

    console.print(table)
    return scaling_results


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]" + "=" * 60)
    console.print("[bold cyan]LAB 07 — N-Body Gravitational Simulation[/bold cyan]")
    console.print("[bold]" + "=" * 60)
    console.print("[dim]Algoritmi: Brute-Force CuPy, Numba CUDA tiled, Scaling O(N²)[/dim]\n")

    results = []

    r_bf = lab_nbody_bruteforce(N=4096)
    results.append(r_bf)

    r_nb = lab_nbody_numba(N=4096)
    results.append(r_nb)

    scaling = lab_nbody_scaling()

    # ── Riepilogo Rich table ───────────────────────────────────
    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
    table.add_column("Algoritmo",  style="cyan")
    table.add_column("CPU (ms)",   justify="right")
    table.add_column("GPU (ms)",   justify="right")
    table.add_column("Speedup",    justify="right", style="green")
    table.add_column("N corpi",    justify="right")

    for r in results:
        table.add_row(
            r.name,
            f"{r.cpu_ms:.2f}",
            f"{r.gpu_ms:.2f}" if r.gpu_ms > 0 else "N/A",
            f"{r.speedup:.1f}x" if r.speedup > 0 else "—",
            f"{r.problem_size:,}" if r.problem_size else "—",
        )
    console.print(table)

    # ── Bar chart speedup → outputs/ ──────────────────────────
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "lab07_benchmark.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LAB 07 — N-Body GPU vs CPU", fontsize=14, fontweight="bold")

    # Pannello sinistro: speedup BF vs Numba
    valid = [r for r in results if r.speedup > 0]
    if valid:
        names   = [r.name for r in valid]
        speedups = [r.speedup for r in valid]
        bars = axes[0].bar(names, speedups, color=["#2196F3", "#4CAF50"][:len(valid)])
        axes[0].set_title("Speedup GPU vs CPU (N=4096)")
        axes[0].set_ylabel("Speedup (×)")
        axes[0].set_ylim(0, max(speedups) * 1.25)
        for bar, s in zip(bars, speedups):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max(speedups) * 0.02,
                         f"{s:.1f}×", ha="center", va="bottom", fontweight="bold")

    # Pannello destro: scaling speedup vs N
    if scaling:
        ns      = [s[0] for s in scaling]
        sp_vals = [s[3] for s in scaling]
        axes[1].plot(ns, sp_vals, marker="o", color="#E91E63", linewidth=2, markersize=8)
        axes[1].set_title("Scaling Speedup CuPy vs N")
        axes[1].set_xlabel("N (numero di corpi)")
        axes[1].set_ylabel("Speedup (×)")
        axes[1].set_xscale("log", base=2)
        axes[1].grid(True, alpha=0.3)
        for n, s in zip(ns, sp_vals):
            axes[1].annotate(f"{s:.1f}×", (n, s),
                             textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    rprint(f"\n  [dim]Plot salvato in: {out_path}[/dim]")


if __name__ == "__main__":
    main()
