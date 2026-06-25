"""
lab05-fluid-pde/src/run.py
===========================
LAB 05 — Fluid Dynamics & PDE Solvers
Algoritmi: Heat Equation 2D, Navier-Stokes 2D (lid-driven cavity)
Paradigma: NumPy finite differences → CuPy → Numba stencil

Riferimenti:
  Numba stencil: https://numba.readthedocs.io/en/stable/user/stencil.html
  CuPy stencil: https://docs.cupy.dev/en/stable/reference/ndimage.html
  CFD Python:    https://github.com/barbagroup/CFDPython
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
# 1. Heat Equation 2D — Diffusione del calore
# ──────────────────────────────────────────────────────────────

def lab_heat_equation(N: int = 1024, steps: int = 500, alpha: float = 0.01):
    """
    Equazione del calore 2D: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    Schema esplicito alle differenze finite (FTCS).
    Applicazione HPC: simulazioni termiche, diffusione materiali.

    Stabilità: dt ≤ dx²/(4α)
    """
    rprint(f"\n[bold cyan]Heat Equation 2D — {N}x{N}, {steps} passi temporali[/bold cyan]")

    dx = 1.0 / N
    dt = 0.25 * dx**2 / alpha  # stabilità CFL
    r  = alpha * dt / dx**2

    rprint(f"  dt={dt:.2e}, CFL r={r:.3f} (stabile se r≤0.25)")

    # Condizioni iniziali: temperatura 1 al centro, 0 ai bordi
    u = np.zeros((N, N), dtype=np.float32)
    u[N//4:3*N//4, N//4:3*N//4] = 1.0

    def cpu_heat(u, steps, r):
        for _ in range(steps):
            lap = (u[:-2, 1:-1] + u[2:, 1:-1] +
                   u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
            u[1:-1, 1:-1] += r * lap
        return u

    def cpu_fn():
        return cpu_heat(u.copy(), steps, r)

    try:
        import cupy as cp

        u_gpu = cp.array(u)

        def gpu_heat(u, steps, r):
            for _ in range(steps):
                lap = (u[:-2, 1:-1] + u[2:, 1:-1] +
                       u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1])
                u[1:-1, 1:-1] += r * lap
            cp.cuda.Stream.null.synchronize()
            return u

        def gpu_fn():
            return gpu_heat(u_gpu.copy(), steps, r)

        result = benchmark(cpu_fn, gpu_fn, name="Heat-Eq", problem_size=N*N,
                           warmup=1, runs=3)
        rprint(f"  {result}")

        # Energia totale (verifica conservazione)
        u_final = gpu_fn()
        energy = float(u_final.sum() * dx**2)
        rprint(f"  Energia finale: [green]{energy:.4f}[/green] "
               f"(conservata se ≈ {float(u.sum()*dx**2):.4f})")

        return result

    except ImportError:
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("Heat-Eq", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# 2. Navier-Stokes 2D — Lid-Driven Cavity
# ──────────────────────────────────────────────────────────────

def lab_navier_stokes(N: int = 256, steps: int = 200):
    """
    Navier-Stokes 2D incomprimibile — Lid-Driven Cavity.
    Schema: proiezione di Chorin (pressure-velocity splitting).
    Re = 100 (regime laminare stabile).

    Applicazione HPC: CFD, aerodinamica, meteorologia.
    Riferimento: Barba & Forsyth, "CFD Python" (12 passi)
    https://github.com/barbagroup/CFDPython
    """
    rprint(f"\n[bold cyan]Navier-Stokes 2D Lid-Driven Cavity — {N}x{N}, {steps} step[/bold cyan]")

    # Parametri fisici
    rho = 1.0    # densità
    nu  = 0.01   # viscosità cinematica (Re = U*L/nu = 1*1/0.01 = 100)
    dt  = 0.001
    dx = dy = 1.0 / (N - 1)
    nit = 20     # iterazioni Poisson per la pressione

    def build_fields():
        u = np.zeros((N, N), dtype=np.float32)   # velocità x
        v = np.zeros((N, N), dtype=np.float32)   # velocità y
        p = np.zeros((N, N), dtype=np.float32)   # pressione
        return u, v, p

    def pressure_poisson(p, b, dx, dy, nit):
        """Risoluzione equazione di Poisson per la pressione."""
        for _ in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                 (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                (2 * (dx**2 + dy**2)) -
                dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1]
            )
            # Neumann BC per pressione
            p[:, -1] = p[:, -2]
            p[0, :]  = p[1, :]
            p[:, 0]  = p[:, 1]
            p[-1, :] = 0
        return p

    def ns_step_cpu(u, v, p, dx, dy, dt, nu, rho, nit):
        un = u.copy(); vn = v.copy()

        # Source term (divergenza del campo velocità)
        b = np.zeros_like(u)
        b[1:-1, 1:-1] = (
            rho * (1/dt *
                   ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                    (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)) -
                   ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx))**2 -
                   2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2*dy) *
                        (v[1:-1, 2:] - v[1:-1, :-2]) / (2*dx)) -
                   ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy))**2)
        )

        p = pressure_poisson(p, b, dx, dy, nit)

        # Aggiorna velocità
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
            vn[1:-1, 1:-1] * dt/dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
            dt/(2*rho*dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
            nu * (dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, :-2]) +
                  dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[:-2, 1:-1]))
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1] -
            un[1:-1, 1:-1] * dt/dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
            vn[1:-1, 1:-1] * dt/dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
            dt/(2*rho*dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
            nu * (dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                  dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[:-2, 1:-1]))
        )

        # Boundary conditions: lid-driven cavity
        u[-1, :] = 1.0   # lid superiore si muove a velocità 1
        u[0, :]  = 0.0
        u[:, 0]  = 0.0
        u[:, -1] = 0.0
        v[:] = 0.0

        return u, v, p

    def cpu_fn():
        u, v, p = build_fields()
        for _ in range(steps):
            u, v, p = ns_step_cpu(u, v, p, dx, dy, dt, nu, rho, nit)
        return u, v, p

    with CPUTimer() as t:
        u_f, v_f, p_f = cpu_fn()

    rprint(f"  CPU: {t.elapsed_ms:.1f} ms")
    max_vel = np.sqrt(u_f**2 + v_f**2).max()
    rprint(f"  Velocità max: [green]{max_vel:.4f}[/green]")

    # GPU version
    try:
        import cupy as cp

        def gpu_fn():
            u = cp.zeros((N, N), dtype=cp.float32)
            v = cp.zeros((N, N), dtype=cp.float32)
            p = cp.zeros((N, N), dtype=cp.float32)

            for _ in range(steps):
                un = u.copy(); vn = v.copy()

                b = cp.zeros_like(u)
                b[1:-1, 1:-1] = (
                    rho * (1/dt *
                           ((u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)))
                )

                for _ in range(nit):
                    pn = p.copy()
                    p[1:-1, 1:-1] = (
                        ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                         (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                        (2*(dx**2+dy**2)) -
                        dx**2*dy**2/(2*(dx**2+dy**2)) * b[1:-1, 1:-1]
                    )
                    p[:, -1] = p[:, -2]
                    p[0, :]  = p[1, :]
                    p[:, 0]  = p[:, 1]
                    p[-1, :] = 0

                u[1:-1, 1:-1] = (
                    un[1:-1, 1:-1] -
                    un[1:-1, 1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,:-2]) -
                    vn[1:-1, 1:-1]*dt/dy*(un[1:-1,1:-1]-un[:-2,1:-1]) -
                    dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,:-2]) +
                    nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2]) +
                        dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1]))
                )

                u[-1, :] = 1.0
                u[0, :]  = 0.0
                u[:, 0]  = 0.0
                u[:, -1] = 0.0
                v[:] = 0.0

            cp.cuda.Stream.null.synchronize()
            return u, v, p

        with GPUTimer() as tg:
            gpu_fn()

        speedup = t.elapsed_ms / tg.elapsed_ms
        rprint(f"  GPU: {tg.elapsed_ms:.1f} ms | Speedup: [green]{speedup:.1f}x[/green]")

        return BenchmarkResult("Navier-Stokes", cpu_ms=t.elapsed_ms,
                               gpu_ms=tg.elapsed_ms, speedup=speedup,
                               problem_size=N*N)

    except ImportError:
        return BenchmarkResult("Navier-Stokes", cpu_ms=t.elapsed_ms)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n[bold]=" * 60)
    console.print("[bold cyan]LAB 05 — Fluid Dynamics & PDE[/bold cyan]")
    console.print("[bold]=" * 60)
    console.print("[dim]Algoritmi: Heat Equation 2D, Navier-Stokes 2D[/dim]\n")

    results = []
    results.append(lab_heat_equation())
    results.append(lab_navier_stokes())

    table = Table(title="\nRiepilogo", header_style="bold magenta")
    table.add_column("PDE",       style="cyan")
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
