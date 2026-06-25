"""
lab03-montecarlo/src/run.py
============================
LAB 03 — Monte Carlo Simulations
Algoritmi: π estimation, Black-Scholes, Ising Model 2D
Paradigma: NumPy random → CuPy random (GPU) → Numba CUDA kernel

Riferimenti:
  CuPy random: https://docs.cupy.dev/en/stable/reference/random.html
  Numba CUDA random: https://numba.readthedocs.io/en/stable/cuda/random.html
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


# ──────────────────────────────────────────────────────────────
# 1. Stima di π — Monte Carlo classico
# ──────────────────────────────────────────────────────────────

def lab_pi(N: int = 100_000_000):
    """
    Stima π via Monte Carlo: campiona punti random in [0,1]^2
    e conta quelli dentro il cerchio unitario.
    Esempio didattico di parallelismo embarrassingly parallel.
    """
    rprint(f"\n[bold cyan]Stima π Monte Carlo — N={N:,}[/bold cyan]")

    def cpu_fn():
        x = np.random.uniform(0, 1, N).astype(np.float32)
        y = np.random.uniform(0, 1, N).astype(np.float32)
        inside = (x**2 + y**2) <= 1.0
        return 4.0 * inside.sum() / N

    try:
        import cupy as cp

        def gpu_fn():
            x = cp.random.uniform(0, 1, N, dtype=cp.float32)
            y = cp.random.uniform(0, 1, N, dtype=cp.float32)
            inside = (x**2 + y**2) <= 1.0
            pi_est = 4.0 * inside.sum() / N
            cp.cuda.Stream.null.synchronize()
            return float(pi_est)

        r = benchmark(cpu_fn, gpu_fn, name="Pi-MC", problem_size=N,
                      warmup=2, runs=5)

        pi_gpu = gpu_fn()
        rprint(f"  π stimato (GPU): [green]{pi_gpu:.6f}[/green] "
               f"(errore: {abs(pi_gpu - np.pi):.2e})")
        rprint(f"  {r}")
        return r

    except ImportError:
        with CPUTimer() as t:
            pi_cpu = cpu_fn()
        rprint(f"  π stimato (CPU): {pi_cpu:.6f}")
        return BenchmarkResult("Pi-MC", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. Black-Scholes — Option Pricing
# ──────────────────────────────────────────────────────────────

def lab_black_scholes(N: int = 10_000_000):
    """
    Black-Scholes Monte Carlo per pricing opzioni European call.
    Applicazione HPC: finanza quantitativa, risk management.
    N = numero di traiettorie simulate.
    """
    rprint(f"\n[bold cyan]Black-Scholes Monte Carlo — N={N:,} traiettorie[/bold cyan]")

    # Parametri opzione
    S0    = 100.0   # prezzo corrente
    K     = 105.0   # strike price
    r     = 0.05    # tasso risk-free
    sigma = 0.2     # volatilità
    T     = 1.0     # scadenza (anni)
    dt    = T / 252 # step giornaliero
    steps = 252

    def black_scholes_cpu(n):
        S = np.full(n, S0, dtype=np.float32)
        for _ in range(steps):
            Z = np.random.standard_normal(n).astype(np.float32)
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        payoff = np.maximum(S - K, 0)
        return np.exp(-r * T) * payoff.mean()

    def cpu_fn():
        return black_scholes_cpu(N)

    try:
        import cupy as cp

        def gpu_fn():
            S = cp.full(N, S0, dtype=cp.float32)
            for _ in range(steps):
                Z = cp.random.standard_normal(N).astype(cp.float32)
                S *= cp.exp((r - 0.5 * sigma**2) * dt + sigma * cp.sqrt(dt) * Z)
            payoff = cp.maximum(S - K, 0)
            price = float(cp.exp(-r * T) * payoff.mean())
            cp.cuda.Stream.null.synchronize()
            return price

        r = benchmark(cpu_fn, gpu_fn, name="Black-Scholes",
                      problem_size=N, warmup=1, runs=3)
        price = gpu_fn()
        rprint(f"  Prezzo opzione (GPU): [green]${price:.4f}[/green]")
        rprint(f"  {r}")
        return r

    except ImportError:
        with CPUTimer() as t:
            price = cpu_fn()
        rprint(f"  Prezzo opzione (CPU): ${price:.4f}")
        return BenchmarkResult("Black-Scholes", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 3. Ising Model 2D — Fisica Statistica
# ──────────────────────────────────────────────────────────────

def lab_ising(N: int = 1024, steps: int = 1000):
    """
    Modello di Ising 2D con algoritmo di Metropolis.
    Applicazione HPC: fisica della materia condensata, simulazioni statistiche.
    N = dimensione reticolo NxN.
    """
    rprint(f"\n[bold cyan]Ising Model 2D — {N}x{N}, {steps} step Metropolis[/bold cyan]")

    beta = 0.44  # vicino alla temperatura critica Tc = 2/ln(1+sqrt(2)) ≈ 2.269

    def ising_cpu(lattice, beta, steps):
        N = lattice.shape[0]
        for _ in range(steps):
            # Checkerboard update (sublattice A)
            for color in [0, 1]:
                rows, cols = np.mgrid[0:N, 0:N]
                mask = ((rows + cols) % 2 == color)
                i_idx, j_idx = np.where(mask)

                neighbors = (
                    lattice[(i_idx - 1) % N, j_idx] +
                    lattice[(i_idx + 1) % N, j_idx] +
                    lattice[i_idx, (j_idx - 1) % N] +
                    lattice[i_idx, (j_idx + 1) % N]
                )
                dE = 2 * lattice[i_idx, j_idx] * neighbors
                flip = np.random.random(len(i_idx)) < np.exp(-beta * dE)
                lattice[i_idx[flip], j_idx[flip]] *= -1
        return lattice

    lattice_cpu = np.random.choice([-1, 1], size=(N, N)).astype(np.float32)

    def cpu_fn():
        return ising_cpu(lattice_cpu.copy(), beta, min(steps, 10))  # ridotto per demo

    try:
        import cupy as cp

        lattice_gpu = cp.array(lattice_cpu)

        def ising_gpu_step(lattice, beta, color):
            rows = cp.arange(N)
            cols = cp.arange(N)
            I, J = cp.meshgrid(rows, cols, indexing="ij")
            mask = ((I + J) % 2 == color)

            neighbors = (
                cp.roll(lattice, 1, axis=0) +
                cp.roll(lattice, -1, axis=0) +
                cp.roll(lattice, 1, axis=1) +
                cp.roll(lattice, -1, axis=1)
            )
            dE = 2 * lattice * neighbors
            flip = cp.random.random((N, N)) < cp.exp(-beta * dE)
            flip = flip & mask
            lattice[flip] *= -1
            return lattice

        def gpu_fn():
            lat = lattice_gpu.copy()
            for _ in range(steps):
                for color in [0, 1]:
                    lat = ising_gpu_step(lat, beta, color)
            cp.cuda.Stream.null.synchronize()
            magnetization = float(cp.abs(lat.mean()))
            return magnetization

        r = benchmark(cpu_fn, gpu_fn, name="Ising-2D",
                      problem_size=N*N, warmup=1, runs=3)
        mag = gpu_fn()
        rprint(f"  Magnetizzazione media: [green]{mag:.4f}[/green]")
        rprint(f"  {r}")
        return r

    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("Ising-2D", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]=" * 60)
    console.print("[bold cyan]LAB 03 — Monte Carlo Simulations[/bold cyan]")
    console.print("[bold]=" * 60)
    console.print("[dim]Algoritmi: π estimation, Black-Scholes, Ising 2D[/dim]\n")

    results = []
    results.append(lab_pi())
    results.append(lab_black_scholes())
    results.append(lab_ising())

    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
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


if __name__ == "__main__":
    main()
