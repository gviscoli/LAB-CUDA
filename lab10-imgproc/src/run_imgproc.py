"""
lab10-imgproc/src/run_imgproc.py
=================================
LAB 10 — Image Processing GPU
Algoritmi: Gaussian Blur, Sobel Edge Detection, Bilateral Filter
Paradigma: SciPy CPU baseline -> CuPy GPU -> Numba CUDA kernel separabile
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

try:
    from scipy import ndimage as _sp_ndimage
    _SCIPY_OK = True
except ImportError:
    _sp_ndimage = None
    _SCIPY_OK = False

# ── Numba CUDA kernels (livello modulo obbligatorio) ───────────────────────────
# Filtro gaussiano separabile 7-tap (sigma=3):
# G1d = exp(-k^2 / (2*sigma^2)) normalizzato, k in [-3..3]
# Valori pre-calcolati: sum=1
_G0, _G1, _G2, _G3 = 0.1072, 0.1403, 0.1658, 0.1734
_G4, _G5, _G6      = 0.1658, 0.1403, 0.1072

try:
    from numba import cuda as _numba_cuda, float32 as _nb_f32
    import math as _math

    @_numba_cuda.jit
    def _gauss_row_kernel(src, tmp, rows, cols,
                          g0, g1, g2, g3, g4, g5, g6):
        """Pass orizzontale: convoluzione 1D per riga, kernel gaussiano 7 tap."""
        row = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        col = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        if row >= rows or col >= cols:
            return
        acc = _nb_f32(0.0)
        for k in range(7):
            c = col + k - 3
            if 0 <= c < cols:
                if   k == 0: w = g0
                elif k == 1: w = g1
                elif k == 2: w = g2
                elif k == 3: w = g3
                elif k == 4: w = g4
                elif k == 5: w = g5
                else:        w = g6
                acc += src[row, c] * w
        tmp[row, col] = acc

    @_numba_cuda.jit
    def _gauss_col_kernel(tmp, dst, rows, cols,
                          g0, g1, g2, g3, g4, g5, g6):
        """Pass verticale: convoluzione 1D per colonna, kernel gaussiano 7 tap."""
        row = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        col = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        if row >= rows or col >= cols:
            return
        acc = _nb_f32(0.0)
        for k in range(7):
            r = row + k - 3
            if 0 <= r < rows:
                if   k == 0: w = g0
                elif k == 1: w = g1
                elif k == 2: w = g2
                elif k == 3: w = g3
                elif k == 4: w = g4
                elif k == 5: w = g5
                else:        w = g6
                acc += tmp[r, col] * w
        dst[row, col] = acc

    @_numba_cuda.jit
    def _sobel_kernel(src, dst, rows, cols):
        """Sobel edge detection: sqrt(Gx^2 + Gy^2)."""
        row = _numba_cuda.blockIdx.y * _numba_cuda.blockDim.y + _numba_cuda.threadIdx.y
        col = _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x + _numba_cuda.threadIdx.x
        if row < rows and col < cols:
            if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
                dst[row, col] = _nb_f32(0.0)
                return
            gx = (-src[row-1, col-1] + src[row-1, col+1]
                  - _nb_f32(2.0) * src[row, col-1] + _nb_f32(2.0) * src[row, col+1]
                  - src[row+1, col-1] + src[row+1, col+1])
            gy = (-src[row-1, col-1] - _nb_f32(2.0) * src[row-1, col] - src[row-1, col+1]
                  + src[row+1, col-1] + _nb_f32(2.0) * src[row+1, col] + src[row+1, col+1])
            dst[row, col] = _math.sqrt(gx * gx + gy * gy)

    _NUMBA_IMG_OK = True
except Exception:
    _gauss_row_kernel = None
    _gauss_col_kernel = None
    _sobel_kernel     = None
    _NUMBA_IMG_OK     = False


# ──────────────────────────────────────────────────────────────
# 1. Gaussian Blur 2D
# ──────────────────────────────────────────────────────────────

def lab_gaussian_blur(H: int = 4096, W: int = 4096):
    """
    Gaussian Blur 2D su immagine H×W float32.
    CPU: scipy.ndimage.gaussian_filter(sigma=3)
    GPU CuPy: cupyx.scipy.ndimage.gaussian_filter(sigma=3)
    GPU Numba: kernel separabile 2-pass (riga + colonna), 7 tap.

    Filtro separabile: G2d = G1d ⊗ G1d → 2 convoluzioni 1D invece di 1 conv 2D.
    Complessità: O(N × 7) invece di O(N × 49), riduzione 7× operazioni.
    """
    rprint(f"\n[bold cyan]Gaussian Blur 2D — {H}×{W} float32[/bold cyan]")

    np.random.seed(42)
    img = np.random.rand(H, W).astype(np.float32)

    def cpu_fn():
        if _SCIPY_OK:
            return _sp_ndimage.gaussian_filter(img, sigma=3.0)
        return np.zeros_like(img)

    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cp_ndimage

        img_gpu = cp.array(img)

        def gpu_fn():
            result = cp_ndimage.gaussian_filter(img_gpu, sigma=3.0)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="GaussianBlur",
                      problem_size=H * W, warmup=2, runs=5)
        throughput = H * W * 4 / r.gpu_ms * 1e-6  # MB/s
        rprint(f"  {r}")
        rprint(f"  Throughput GPU: {throughput:.1f} MB/s")

        if _NUMBA_IMG_OK:
            rprint("  [dim]Numba separable kernel (2-pass):[/dim]")
            try:
                src_d = _numba_cuda.to_device(img)
                tmp_d = _numba_cuda.device_array_like(img)
                dst_d = _numba_cuda.device_array_like(img)

                BLOCK = (16, 16)
                grid = ((W + 15) // 16, (H + 15) // 16)

                def numba_fn():
                    _gauss_row_kernel[grid, BLOCK](
                        src_d, tmp_d, H, W,
                        _G0, _G1, _G2, _G3, _G4, _G5, _G6)
                    _gauss_col_kernel[grid, BLOCK](
                        tmp_d, dst_d, H, W,
                        _G0, _G1, _G2, _G3, _G4, _G5, _G6)
                    _numba_cuda.synchronize()

                for _ in range(2):
                    numba_fn()

                times_nb = []
                for _ in range(5):
                    with GPUTimer() as t:
                        numba_fn()
                    times_nb.append(t.elapsed_ms)
                nb_ms = float(np.median(times_nb))
                rprint(f"  Numba: {nb_ms:.2f} ms  speedup=[green]{r.cpu_ms/nb_ms:.1f}x[/green] vs CPU")
            except Exception as e:
                rprint(f"  [yellow]Numba error: {e}[/yellow]")

        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("GaussianBlur", cpu_ms=t.elapsed_ms, problem_size=H * W)


# ──────────────────────────────────────────────────────────────
# 2. Sobel Edge Detection
# ──────────────────────────────────────────────────────────────

def lab_sobel(H: int = 4096, W: int = 4096):
    """
    Sobel Edge Detection su immagine H×W float32.
    CPU: scipy.ndimage.sobel su entrambi gli assi + magnitude.
    GPU CuPy: cupyx.scipy.ndimage.sobel.
    GPU Numba: kernel custom che calcola Gx, Gy e magnitude in un unico pass.

    Gx = [-1,0,+1; -2,0,+2; -1,0,+1]   Gy = [-1,-2,-1; 0,0,0; +1,+2,+1]
    Magnitude = sqrt(Gx^2 + Gy^2)
    """
    rprint(f"\n[bold cyan]Sobel Edge Detection — {H}×{W} float32[/bold cyan]")

    np.random.seed(42)
    img = np.random.rand(H, W).astype(np.float32)

    def cpu_fn():
        if _SCIPY_OK:
            gx = _sp_ndimage.sobel(img, axis=1)
            gy = _sp_ndimage.sobel(img, axis=0)
            return np.hypot(gx, gy)
        return np.zeros_like(img)

    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cp_ndimage

        img_gpu = cp.array(img)

        def gpu_fn():
            gx = cp_ndimage.sobel(img_gpu, axis=1)
            gy = cp_ndimage.sobel(img_gpu, axis=0)
            result = cp.hypot(gx, gy)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="Sobel",
                      problem_size=H * W, warmup=2, runs=5)
        rprint(f"  {r}")

        if _NUMBA_IMG_OK:
            rprint("  [dim]Numba fused kernel (Gx+Gy+magnitude in 1 pass):[/dim]")
            try:
                src_d = _numba_cuda.to_device(img)
                dst_d = _numba_cuda.device_array_like(img)

                BLOCK = (16, 16)
                grid = ((W + 15) // 16, (H + 15) // 16)

                def numba_fn():
                    _sobel_kernel[grid, BLOCK](src_d, dst_d, H, W)
                    _numba_cuda.synchronize()

                for _ in range(2):
                    numba_fn()

                times_nb = []
                for _ in range(5):
                    with GPUTimer() as t:
                        numba_fn()
                    times_nb.append(t.elapsed_ms)
                nb_ms = float(np.median(times_nb))
                rprint(f"  Numba: {nb_ms:.2f} ms  speedup=[green]{r.cpu_ms/nb_ms:.1f}x[/green] vs CPU")
            except Exception as e:
                rprint(f"  [yellow]Numba error: {e}[/yellow]")

        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("Sobel", cpu_ms=t.elapsed_ms, problem_size=H * W)


# ──────────────────────────────────────────────────────────────
# 3. Bilateral Filter
# ──────────────────────────────────────────────────────────────

def lab_bilateral(H: int = 512, W: int = 512,
                  radius: int = 5, sigma_s: float = 3.0, sigma_c: float = 0.15):
    """
    Bilateral Filter su immagine H×W float32 (immagine più piccola: O(N^2*r^2)).
    CPU: implementazione NumPy patch-based con broadcasting.
    GPU: CuPy vectorized patch-based.

    Formula: BF[p] = Σ_q f(||p-q||) * g(|I_p - I_q|) * I_q / Z
    dove f = kernel spaziale gaussiano, g = kernel per range (colore).

    Applicazioni: denoising con preservazione dei bordi, HDR tone mapping,
    image abstraction, depth super-resolution (joint bilateral).
    """
    rprint(f"\n[bold cyan]Bilateral Filter — {H}×{W}, r={radius}, σ_s={sigma_s}, σ_c={sigma_c}[/bold cyan]")
    rprint("  [dim]Filtro edge-preserving: combina vicinanza spaziale e similarità radiometrica[/dim]")

    np.random.seed(42)
    img = np.random.rand(H, W).astype(np.float32)

    # Pre-calcola kernel spaziale (fisso per ogni pixel)
    ks = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    spatial_kernel = np.exp(-(ax[:, None]**2 + ax[None, :]**2) / (2 * sigma_s**2))

    def cpu_fn():
        img_pad = np.pad(img, radius, mode='reflect')
        out = np.zeros_like(img)
        for dy in range(ks):
            for dx in range(ks):
                shifted = img_pad[dy:dy+H, dx:dx+W]
                color_w = np.exp(-((shifted - img)**2) / (2 * sigma_c**2))
                w = color_w * spatial_kernel[dy, dx]
                out += w * shifted
        # Normalizzazione approssimata (Z calcolata separatamente per semplicità)
        return out

    try:
        import cupy as cp

        img_gpu = cp.array(img)
        spatial_gpu = cp.array(spatial_kernel)

        def gpu_fn():
            img_pad = cp.pad(img_gpu, radius, mode='reflect')
            out = cp.zeros_like(img_gpu)
            for dy in range(ks):
                for dx in range(ks):
                    shifted = img_pad[dy:dy+H, dx:dx+W]
                    color_w = cp.exp(-((shifted - img_gpu)**2) / (2 * sigma_c**2))
                    w = color_w * spatial_gpu[dy, dx]
                    out += w * shifted
            cp.cuda.Stream.null.synchronize()
            return out

        r = benchmark(cpu_fn, gpu_fn, name="BilateralFilter",
                      problem_size=H * W, warmup=1, runs=3)
        rprint(f"  {r}")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("BilateralFilter", cpu_ms=t.elapsed_ms, problem_size=H * W)


# ──────────────────────────────────────────────────────────────
# 4. Scaling Analysis
# ──────────────────────────────────────────────────────────────

def lab_imgproc_scaling():
    """Scaling: Gaussian blur al variare della risoluzione."""
    rprint("\n[bold]Scaling Analysis — Gaussian Blur GPU vs CPU[/bold]")

    resolutions = [(256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

    table = Table(title="Scaling: scipy vs cupyx Gaussian Blur",
                  header_style="bold magenta")
    table.add_column("Risoluzione", style="cyan")
    table.add_column("CPU (ms)", justify="right")
    table.add_column("GPU (ms)", justify="right")
    table.add_column("Speedup", justify="right", style="green")
    table.add_column("Mem (MB)", justify="right")

    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cp_ndimage

        for H, W in resolutions:
            img = np.random.rand(H, W).astype(np.float32)
            img_gpu = cp.array(img)
            mem_mb = H * W * 4 / 1024 / 1024

            cpu_times = []
            for _ in range(3):
                with CPUTimer() as tc:
                    if _SCIPY_OK:
                        _sp_ndimage.gaussian_filter(img, sigma=3.0)
                cpu_times.append(tc.elapsed_ms)

            gpu_times = []
            for _ in range(3):
                with GPUTimer() as tg:
                    cp_ndimage.gaussian_filter(img_gpu, sigma=3.0)
                    cp.cuda.Stream.null.synchronize()
                gpu_times.append(tg.elapsed_ms)

            cpu_ms = float(np.median(cpu_times))
            gpu_ms = float(np.median(gpu_times))
            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
            color = "[green]" if speedup > 10 else "[yellow]" if speedup > 2 else "[red]"
            table.add_row(f"{H}×{W}", f"{cpu_ms:.1f}", f"{gpu_ms:.1f}",
                          f"{color}{speedup:.1f}x[/]", f"{mem_mb:.1f}")

            del img_gpu
            cp.get_default_memory_pool().free_all_blocks()

        console.print(table)

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — scaling saltata[/yellow]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]LAB 10 — Image Processing GPU[/bold cyan]")
    console.print("=" * 60)
    console.print("[dim]Gaussian Blur, Sobel Edge Detection, Bilateral Filter[/dim]\n")

    results = []
    results.append(lab_gaussian_blur())
    results.append(lab_sobel())
    results.append(lab_bilateral())
    lab_imgproc_scaling()

    table = Table(title="\nRiepilogo Speedup GPU vs CPU", header_style="bold magenta")
    table.add_column("Algoritmo",  style="cyan")
    table.add_column("Dimensione", justify="right")
    table.add_column("CPU (ms)",   justify="right")
    table.add_column("GPU (ms)",   justify="right")
    table.add_column("Speedup",    justify="right", style="green")

    for r in results:
        table.add_row(
            r.name,
            f"{r.problem_size:,}" if r.problem_size > 0 else "—",
            f"{r.cpu_ms:.2f}",
            f"{r.gpu_ms:.2f}" if r.gpu_ms > 0 else "N/A",
            f"{r.speedup:.1f}x" if r.speedup > 0 else "—",
        )
    console.print(table)

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared" / "utils"))
        from plotter import plot_cpu_vs_gpu
        valid = [r for r in results if r.gpu_ms > 0 and r.speedup > 0]
        if valid:
            out = Path(__file__).resolve().parent.parent / "outputs" / "lab10_benchmark.png"
            plot_cpu_vs_gpu(valid,
                            title="LAB 10 — Image Processing: CPU vs GPU",
                            save_path=str(out), show=False)
    except Exception as e:
        rprint(f"[dim]Grafico non generato: {e}[/dim]")


if __name__ == "__main__":
    main()
