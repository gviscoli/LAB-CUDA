"""
lab09-finance/src/run_finance.py
=================================
LAB 09 — Quantitative Finance GPU
Algoritmi: Portfolio VaR (Cholesky), Options Greeks (MC), Implied Vol Surface
Paradigma: NumPy → CuPy GPU acceleration

Riferimenti:
  CuPy linalg: https://docs.cupy.dev/en/stable/reference/linalg.html
  CuPy random:  https://docs.cupy.dev/en/stable/reference/random.html
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
# 1. Portfolio VaR — Cholesky Correlated Simulation
# ──────────────────────────────────────────────────────────────

def lab_portfolio_var(N_sims: int = 5_000_000, N_assets: int = 100):
    """
    Simula rendimenti correlati di un portafoglio tramite decomposizione di Cholesky.
    Calcola Value at Risk (VaR) e Conditional VaR (CVaR) al 95%.
    Applicazione HPC: risk management, stress testing, regolamentazione bancaria.
    """
    rprint(f"\n[bold cyan]Portfolio VaR (Cholesky) — N_sims={N_sims:,}, N_assets={N_assets}[/bold cyan]")

    # Setup della matrice di correlazione positiva definita
    A = np.random.randn(N_assets, N_assets).astype(np.float32)
    corr = A @ A.T / N_assets + np.eye(N_assets, dtype=np.float32)
    corr_chol = np.linalg.cholesky(corr).astype(np.float32)
    mu = np.random.uniform(0.05, 0.15, N_assets).astype(np.float32)
    sigma = np.random.uniform(0.1, 0.3, N_assets).astype(np.float32)
    weights = np.ones(N_assets, dtype=np.float32) / N_assets  # equal weight

    def portfolio_var_cpu(N_sims, N_assets, corr_chol, weights, mu, sigma, dt=1/252):
        Z = np.random.standard_normal((N_assets, N_sims)).astype(np.float32)
        correlated = corr_chol @ Z        # (N_assets, N_sims)
        returns = mu[:, None] * dt + sigma[:, None] * np.sqrt(dt) * correlated
        port_returns = weights @ returns   # (N_sims,)
        var_95 = np.percentile(port_returns, 5)
        cvar_95 = port_returns[port_returns <= var_95].mean()
        return var_95, cvar_95

    def cpu_fn():
        return portfolio_var_cpu(N_sims, N_assets, corr_chol, weights, mu, sigma)

    try:
        import cupy as cp

        corr_chol_gpu = cp.array(corr_chol)
        mu_gpu = cp.array(mu)
        sigma_gpu = cp.array(sigma)
        weights_gpu = cp.array(weights)

        def portfolio_var_gpu(N_sims, N_assets, corr_chol, weights, mu, sigma, dt=1/252):
            Z = cp.random.standard_normal((N_assets, N_sims)).astype(cp.float32)
            correlated = corr_chol @ Z
            returns = mu[:, None] * dt + sigma[:, None] * cp.sqrt(dt) * correlated
            port_returns = weights @ returns
            var_95 = float(cp.percentile(port_returns, 5))
            cvar_95 = float(port_returns[port_returns <= var_95].mean())
            cp.cuda.Stream.null.synchronize()
            return var_95, cvar_95

        def gpu_fn():
            return portfolio_var_gpu(N_sims, N_assets, corr_chol_gpu,
                                     weights_gpu, mu_gpu, sigma_gpu)

        result = benchmark(cpu_fn, gpu_fn, name="Portfolio-VaR",
                           problem_size=N_sims * N_assets, warmup=1, runs=3)

        var_95, cvar_95 = gpu_fn()
        rprint(f"  VaR 95% (GPU): [green]{var_95:.4%}[/green] | CVaR 95%: [yellow]{cvar_95:.4%}[/yellow]")
        rprint(f"  {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            var_95, cvar_95 = cpu_fn()
        rprint(f"  VaR 95% (CPU): {var_95:.4%} | CVaR 95%: {cvar_95:.4%}")
        return BenchmarkResult("Portfolio-VaR", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. Options Greeks — Monte Carlo via Finite Differences
# ──────────────────────────────────────────────────────────────

def lab_options_greeks(N: int = 10_000_000):
    """
    Calcola Delta e Gamma di una call europea tramite differenze finite su path MC.
    Delta = dV/dS,  Gamma = d²V/dS²
    Applicazione HPC: hedging dinamico, gestione del rischio su derivati.
    """
    rprint(f"\n[bold cyan]Options Greeks (MC Finite Diff) — N={N:,} traiettorie[/bold cyan]")

    S0 = 100.0
    K = 105.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    dS = 1.0
    steps = 252
    dt = T / steps

    def black_scholes_mc(S0, K, r, sigma, T, N, xp=np):
        S = xp.full(N, S0, dtype=xp.float32)
        dt_local = T / 252
        for _ in range(252):
            Z = xp.random.standard_normal(N).astype(xp.float32)
            S *= xp.exp((r - 0.5 * sigma**2) * dt_local + sigma * xp.sqrt(dt_local) * Z)
        payoff = xp.maximum(S - K, 0)
        return float(xp.exp(-r * T) * payoff.mean())

    def greeks_cpu(S0, K, r, sigma, T, N, dS=1.0):
        price   = black_scholes_mc(S0,      K, r, sigma, T, N, xp=np)
        price_u = black_scholes_mc(S0 + dS, K, r, sigma, T, N, xp=np)
        price_d = black_scholes_mc(S0 - dS, K, r, sigma, T, N, xp=np)
        delta = (price_u - price_d) / (2 * dS)
        gamma = (price_u - 2 * price + price_d) / dS**2
        return price, delta, gamma

    # Benchmark on single BS call for fair timing
    def cpu_fn():
        return black_scholes_mc(S0, K, r, sigma, T, N, xp=np)

    try:
        import cupy as cp

        def gpu_fn():
            result = black_scholes_mc(S0, K, r, sigma, T, N, xp=cp)
            cp.cuda.Stream.null.synchronize()
            return result

        result = benchmark(cpu_fn, gpu_fn, name="Greeks-MC",
                           problem_size=N, warmup=1, runs=3)

        # Compute actual Greeks using GPU
        def black_scholes_mc_gpu(S0_val, K, r, sigma, T, N):
            S = cp.full(N, S0_val, dtype=cp.float32)
            dt_local = T / 252
            for _ in range(252):
                Z = cp.random.standard_normal(N).astype(cp.float32)
                S *= cp.exp((r - 0.5 * sigma**2) * dt_local + sigma * cp.sqrt(dt_local) * Z)
            payoff = cp.maximum(S - K, 0)
            cp.cuda.Stream.null.synchronize()
            return float(cp.exp(-r * T) * payoff.mean())

        price   = black_scholes_mc_gpu(S0,      K, r, sigma, T, N)
        price_u = black_scholes_mc_gpu(S0 + dS, K, r, sigma, T, N)
        price_d = black_scholes_mc_gpu(S0 - dS, K, r, sigma, T, N)
        delta = (price_u - price_d) / (2 * dS)
        gamma = (price_u - 2 * price + price_d) / dS**2

        rprint(f"  Prezzo call (GPU): [green]${price:.4f}[/green]")
        rprint(f"  Delta: [green]{delta:.4f}[/green] | Gamma: [green]{gamma:.6f}[/green]")
        rprint(f"  {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            price = cpu_fn()
        price_u = black_scholes_mc(S0 + dS, K, r, sigma, T, N, xp=np)
        price_d = black_scholes_mc(S0 - dS, K, r, sigma, T, N, xp=np)
        delta = (price_u - price_d) / (2 * dS)
        gamma = (price_u - 2 * price + price_d) / dS**2
        rprint(f"  Prezzo call (CPU): ${price:.4f}")
        rprint(f"  Delta: {delta:.4f} | Gamma: {gamma:.6f}")
        return BenchmarkResult("Greeks-MC", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 3. Implied Volatility Surface
# ──────────────────────────────────────────────────────────────

def lab_implied_vol_surface(N_sims: int = 1_000_000):
    """
    Calcola prezzi di opzioni per una griglia di strikes e maturities via MC.
    Produce la superficie di volatilità implicita (28 combinazioni K×T).
    Applicazione HPC: calibrazione modelli, trading di volatilità, risk management.
    """
    rprint(f"\n[bold cyan]Implied Vol Surface — N_sims={N_sims:,}, griglia 7×4 (28 comb.)[/bold cyan]")

    strikes    = np.array([85, 90, 95, 100, 105, 110, 115], dtype=np.float32)
    maturities = np.array([0.25, 0.5, 1.0, 2.0], dtype=np.float32)
    S0 = 100.0
    r  = 0.05
    sigma_real = 0.2

    def mc_price_cpu(S0, K, r, sigma, T, N):
        steps = max(int(T * 252), 1)
        dt = T / steps
        S = np.full(N, S0, dtype=np.float32)
        for _ in range(steps):
            Z = np.random.standard_normal(N).astype(np.float32)
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        payoff = np.maximum(S - K, 0)
        return float(np.exp(-r * T) * payoff.mean())

    def surface_cpu():
        prices = np.zeros((len(maturities), len(strikes)), dtype=np.float32)
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                prices[i, j] = mc_price_cpu(S0, float(K), r, sigma_real, float(T), N_sims)
        return prices

    def cpu_fn():
        return surface_cpu()

    try:
        import cupy as cp

        def mc_price_gpu(S0, K, r, sigma, T, N):
            steps = max(int(T * 252), 1)
            dt = T / steps
            S = cp.full(N, S0, dtype=cp.float32)
            for _ in range(steps):
                Z = cp.random.standard_normal(N).astype(cp.float32)
                S *= cp.exp((r - 0.5 * sigma**2) * dt + sigma * cp.sqrt(dt) * Z)
            payoff = cp.maximum(S - K, 0)
            cp.cuda.Stream.null.synchronize()
            return float(cp.exp(-r * T) * payoff.mean())

        def surface_gpu():
            prices = np.zeros((len(maturities), len(strikes)), dtype=np.float32)
            for i, T in enumerate(maturities):
                for j, K in enumerate(strikes):
                    prices[i, j] = mc_price_gpu(S0, float(K), r, sigma_real, float(T), N_sims)
            return prices

        def gpu_fn():
            return surface_gpu()

        result = benchmark(cpu_fn, gpu_fn, name="ImpVol-Surface",
                           problem_size=N_sims * len(strikes) * len(maturities),
                           warmup=1, runs=3)

        prices_gpu = gpu_fn()

        # Print compact surface table
        surf_table = Table(title="Option Price Surface (GPU)", header_style="bold blue")
        surf_table.add_column("T \\ K", style="dim")
        for K in strikes:
            surf_table.add_column(f"K={int(K)}", justify="right")
        for i, T in enumerate(maturities):
            row = [f"T={T:.2f}"] + [f"{prices_gpu[i, j]:.2f}" for j in range(len(strikes))]
            surf_table.add_row(*row)
        console.print(surf_table)

        rprint(f"  {result}")
        return result

    except ImportError:
        with CPUTimer() as t:
            prices_cpu = cpu_fn()

        surf_table = Table(title="Option Price Surface (CPU)", header_style="bold blue")
        surf_table.add_column("T \\ K", style="dim")
        for K in strikes:
            surf_table.add_column(f"K={int(K)}", justify="right")
        for i, T in enumerate(maturities):
            row = [f"T={T:.2f}"] + [f"{prices_cpu[i, j]:.2f}" for j in range(len(strikes))]
            surf_table.add_row(*row)
        console.print(surf_table)

        return BenchmarkResult("ImpVol-Surface", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]" + "=" * 60)
    console.print("[bold cyan]LAB 09 — Quantitative Finance GPU[/bold cyan]")
    console.print("[bold]" + "=" * 60)
    console.print("[dim]Algoritmi: Portfolio VaR, Options Greeks, Implied Vol Surface[/dim]\n")

    results = []
    results.append(lab_portfolio_var())
    results.append(lab_options_greeks())
    results.append(lab_implied_vol_surface())

    # Summary table
    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
    table.add_column("Algoritmo",   style="cyan")
    table.add_column("CPU (ms)",    justify="right")
    table.add_column("GPU (ms)",    justify="right")
    table.add_column("Speedup",     justify="right", style="green")

    for r in results:
        table.add_row(
            r.name,
            f"{r.cpu_ms:.2f}",
            f"{r.gpu_ms:.2f}" if r.gpu_ms > 0 else "N/A",
            f"{r.speedup:.1f}x" if r.speedup > 0 else "—",
        )
    console.print(table)

    # Save benchmark plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_path = Path(__file__).resolve().parent.parent / "outputs" / "lab09_benchmark.png"

        names     = [r.name for r in results]
        cpu_times = [r.cpu_ms for r in results]
        gpu_times = [r.gpu_ms if r.gpu_ms > 0 else 0 for r in results]
        speedups  = [r.speedup if r.speedup > 0 else 0 for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LAB 09 — Quantitative Finance GPU Benchmark", fontsize=14, fontweight="bold")

        # Bar chart: CPU vs GPU time
        x = np.arange(len(names))
        width = 0.35
        ax1 = axes[0]
        bars_cpu = ax1.bar(x - width / 2, cpu_times, width, label="CPU (NumPy)", color="#4C72B0", alpha=0.85)
        bars_gpu = ax1.bar(x + width / 2, gpu_times, width, label="GPU (CuPy)",  color="#DD8452", alpha=0.85)
        ax1.set_title("Tempo di esecuzione (ms)")
        ax1.set_ylabel("Tempo (ms)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=15, ha="right")
        ax1.legend()
        ax1.set_yscale("log")
        ax1.grid(axis="y", alpha=0.3)
        for bar in bars_cpu:
            h = bar.get_height()
            ax1.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        for bar in bars_gpu:
            h = bar.get_height()
            if h > 0:
                ax1.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

        # Speedup bar
        ax2 = axes[1]
        colors = ["#2ca02c" if s >= 10 else "#ff7f0e" if s >= 1 else "#d62728" for s in speedups]
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
                ax2.annotate(f"{s:.1f}x", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                             fontsize=9, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        rprint(f"\n[dim]Plot salvato in: {output_path}[/dim]")

    except Exception as exc:
        rprint(f"\n[yellow]Plot non salvato: {exc}[/yellow]")

    rprint("\n[dim]💡 Le banche investment usano GPU per calcolo real-time di Greeks su milioni di contratti.[/dim]")


if __name__ == "__main__":
    main()
