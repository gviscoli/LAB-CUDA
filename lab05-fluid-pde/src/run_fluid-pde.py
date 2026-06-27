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

def lab_navier_stokes(N: int = 64, steps: int = 500):
    """
    Navier-Stokes 2D incomprimibile — Lid-Driven Cavity (Re=100).
    Schema: proiezione di Chorin con solver diretto per la pressione.

    Il Jacobi/SOR iterativo NON funziona per questo problema: le BCs miste
    (Neumann su 3 lati, Dirichlet su 1) danno spettro di Jacobi ρ_J≈0.9997 →
    servono ~4000 iterazioni per convergenza. Con pressione sbagliata il
    divergence di u si accumula step→step fino a overflow.
    Fix: scipy.sparse.linalg.factorized → LU esatto, O(1) per step successivi.
    """
    rprint(f"\n[bold cyan]Navier-Stokes 2D Lid-Driven Cavity — {N}x{N}, {steps} step[/bold cyan]")

    rho = 1.0
    nu  = 0.01   # Re = U*L/nu = 1*1/0.01 = 100
    dx = dy = 1.0 / (N - 1)
    # CFL 2D: ν·dt/(dx²) ≤ 0.25 per diffusione; CFL_adv = u_max·dt/dx ≤ 1
    dt = min(0.20 * dx**2 / nu, 0.5 * dx)
    cfl_diff = nu * dt / dx**2
    rprint(f"  Re={1/nu:.0f}, dt={dt:.2e}, CFL_diff={cfl_diff:.3f}")

    # ── Sparse pressure matrix (built once, factorized once) ──────────────────
    # Laplaciano su punti interni (N-2)×(N-2) con BCs incorporate:
    #   Neumann: bottom (i=0), left (j=0), right (j=N-1)  → ghost cell: p_ghost = p_adj
    #   Dirichlet: top (i=N-1) → p=0, contribuisce 0 al RHS (valore noto)
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import factorized

    M = N - 2
    A = lil_matrix((M * M, M * M), dtype=np.float64)
    for i in range(M):
        for j in range(M):
            k = i * M + j
            d = -2.0 / dx**2 - 2.0 / dy**2
            if i == 0:    d            += 1.0/dy**2        # Neumann bottom
            else:          A[k, (i-1)*M+j] = 1.0/dy**2
            if i < M - 1:  A[k, (i+1)*M+j] = 1.0/dy**2   # top=Dirichlet p=0, nessun contributo
            if j == 0:    d            += 1.0/dx**2        # Neumann left
            else:          A[k, i*M+(j-1)] = 1.0/dx**2
            if j == M - 1: d           += 1.0/dx**2        # Neumann right
            else:          A[k, i*M+(j+1)] = 1.0/dx**2
            A[k, k] = d

    rprint("  Fattorizzazione LU matrice pressione sparse...")
    solve_p = factorized(A.tocsr())

    def compute_b(u, v):
        b = np.zeros_like(u)
        b[1:-1, 1:-1] = rho * (
            (1/dt) * ((u[1:-1,2:]-u[1:-1,:-2])/(2*dx) + (v[2:,1:-1]-v[:-2,1:-1])/(2*dy)) -
            ((u[1:-1,2:]-u[1:-1,:-2])/(2*dx))**2 -
            2*((u[2:,1:-1]-u[:-2,1:-1])/(2*dy) * (v[1:-1,2:]-v[1:-1,:-2])/(2*dx)) -
            ((v[2:,1:-1]-v[:-2,1:-1])/(2*dy))**2
        )
        return b

    def solve_pressure(p, b):
        rhs = b[1:-1, 1:-1].flatten()
        p[1:-1, 1:-1] = solve_p(rhs).reshape((M, M))
        p[:, -1] = p[:, -2]; p[0, :] = p[1, :]
        p[:, 0]  = p[:, 1];  p[-1, :] = 0.0
        return p

    def velocity_update(u, v, un, vn, p):
        u[1:-1,1:-1] = (
            un[1:-1,1:-1] -
            un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,:-2]) -
            vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[:-2,1:-1]) -
            dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,:-2]) +
            nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2]) +
                dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1]))
        )
        v[1:-1,1:-1] = (
            vn[1:-1,1:-1] -
            un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,:-2]) -
            vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[:-2,1:-1]) -
            dt/(2*rho*dy)*(p[2:,1:-1]-p[:-2,1:-1]) +
            nu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2]) +
                dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1]))
        )
        return u, v

    def apply_bc(u, v):
        # lid last so corners inherit wall (u=0), not lid (u=1)
        u[0,:]=0; u[:,0]=0; u[:,-1]=0; u[-1,:]=1.0
        v[0,:]=0; v[-1,:]=0; v[:,0]=0; v[:,-1]=0

    def cpu_ns():
        u = np.zeros((N, N), dtype=np.float64)
        v = np.zeros((N, N), dtype=np.float64)
        p = np.zeros((N, N), dtype=np.float64)
        for _ in range(steps):
            un, vn = u.copy(), v.copy()
            b = compute_b(u, v)
            p = solve_pressure(p, b)
            u, v = velocity_update(u, v, un, vn, p)
            apply_bc(u, v)
        return u, v, p

    with CPUTimer() as t:
        u_f, v_f, p_f = cpu_ns()

    cpu_ms = t.elapsed_ms
    max_vel = float(np.sqrt(u_f**2 + v_f**2).max())
    rprint(f"  CPU: {cpu_ms:.1f} ms  |  velocità max: [green]{max_vel:.4f}[/green]")

    # ── GPU: CuPy sparse CG (cupyx) o fallback a SOR con molte iterazioni ────
    try:
        import cupy as cp
        gpu_ms = None

        try:
            import cupyx.scipy.sparse as cpsp
            import cupyx.scipy.sparse.linalg as cpla
            A_gpu = cpsp.csr_matrix(A.tocsr())

            def gpu_ns():
                u = cp.zeros((N, N), dtype=cp.float64)
                v = cp.zeros((N, N), dtype=cp.float64)
                p = cp.zeros((N, N), dtype=cp.float64)
                for _ in range(steps):
                    un, vn = u.copy(), v.copy()
                    b = cp.zeros_like(u)
                    b[1:-1,1:-1] = rho * (
                        (1/dt)*((u[1:-1,2:]-u[1:-1,:-2])/(2*dx)+(v[2:,1:-1]-v[:-2,1:-1])/(2*dy)) -
                        ((u[1:-1,2:]-u[1:-1,:-2])/(2*dx))**2 -
                        2*((u[2:,1:-1]-u[:-2,1:-1])/(2*dy)*(v[1:-1,2:]-v[1:-1,:-2])/(2*dx)) -
                        ((v[2:,1:-1]-v[:-2,1:-1])/(2*dy))**2
                    )
                    rhs = b[1:-1,1:-1].flatten()
                    p_int, _ = cpla.cg(A_gpu, rhs, tol=1e-8)
                    p[1:-1,1:-1] = p_int.reshape((M, M))
                    p[:,-1]=p[:,-2]; p[0,:]=p[1,:]; p[:,0]=p[:,1]; p[-1,:]=0
                    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                        un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,:-2]) -
                        vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[:-2,1:-1]) -
                        dt/(2*rho*dx)*(p[1:-1,2:]-p[1:-1,:-2]) +
                        nu*(dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2]) +
                            dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1])))
                    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                        un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,:-2]) -
                        vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[:-2,1:-1]) -
                        dt/(2*rho*dy)*(p[2:,1:-1]-p[:-2,1:-1]) +
                        nu*(dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2]) +
                            dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1])))
                    u[0,:]=0; u[:,0]=0; u[:,-1]=0; u[-1,:]=1.0
                    v[0,:]=0; v[-1,:]=0; v[:,0]=0; v[:,-1]=0
                cp.cuda.Stream.null.synchronize()
                return u, v, p

            with GPUTimer() as tg:
                gpu_ns()
            gpu_ms = tg.elapsed_ms
            rprint(f"  GPU (cupyx CG): {gpu_ms:.1f} ms")

        except Exception as e:
            rprint(f"  [yellow]GPU sparse CG non disponibile ({type(e).__name__})[/yellow]")
            rprint("  [dim]NS su GPU richiede loop CUDA custom (Numba) per speedup reale[/dim]")

        if gpu_ms is not None and gpu_ms > 0:
            speedup = cpu_ms / gpu_ms
            rprint(f"  Speedup: [green]{speedup:.1f}x[/green]")
            return BenchmarkResult("Navier-Stokes", cpu_ms=cpu_ms,
                                   gpu_ms=gpu_ms, speedup=speedup, problem_size=N*N)

    except ImportError:
        pass

    return BenchmarkResult("Navier-Stokes", cpu_ms=cpu_ms)


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
