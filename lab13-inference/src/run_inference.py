"""
lab13-inference/src/run_inference.py
======================================
LAB 13 — Deep Learning Inference GPU
Algoritmi: INT8 Quantized MatMul, Batch Norm, Softmax, Layer Norm
Paradigma: NumPy CPU baseline -> CuPy GPU -> Numba CUDA fused kernel
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

# ── Numba CUDA kernels (livello modulo) ────────────────────────────────────────
_SOFTMAX_BLOCK = 256

try:
    from numba import cuda as _numba_cuda, float32 as _nb_f32
    import math as _math

    @_numba_cuda.jit
    def _softmax_kernel(x, out, N_rows, N_cols):
        """
        Softmax numericamente stabile per riga.
        Un blocco per riga; riduzione in shared memory per max e sum.
        x, out: shape (N_rows, N_cols) float32 in row-major order.
        """
        row = _numba_cuda.blockIdx.x
        tx  = _numba_cuda.threadIdx.x

        sh_max = _numba_cuda.shared.array(shape=_SOFTMAX_BLOCK, dtype=_nb_f32)
        sh_sum = _numba_cuda.shared.array(shape=_SOFTMAX_BLOCK, dtype=_nb_f32)

        if row >= N_rows:
            return

        # Fase 1: trova max parziale per thread
        local_max = _nb_f32(-3.402823466e+38)
        col = tx
        while col < N_cols:
            v = x[row, col]
            if v > local_max:
                local_max = v
            col += _SOFTMAX_BLOCK
        sh_max[tx] = local_max
        _numba_cuda.syncthreads()

        # Riduzione max
        stride = _SOFTMAX_BLOCK // 2
        while stride > 0:
            if tx < stride:
                if sh_max[tx + stride] > sh_max[tx]:
                    sh_max[tx] = sh_max[tx + stride]
            _numba_cuda.syncthreads()
            stride //= 2
        row_max = sh_max[0]
        _numba_cuda.syncthreads()

        # Fase 2: calcola exp e somma parziale
        local_sum = _nb_f32(0.0)
        col = tx
        while col < N_cols:
            ev = _math.exp(x[row, col] - row_max)
            out[row, col] = ev
            local_sum += ev
            col += _SOFTMAX_BLOCK
        sh_sum[tx] = local_sum
        _numba_cuda.syncthreads()

        # Riduzione sum
        stride = _SOFTMAX_BLOCK // 2
        while stride > 0:
            if tx < stride:
                sh_sum[tx] += sh_sum[tx + stride]
            _numba_cuda.syncthreads()
            stride //= 2
        row_sum = sh_sum[0]
        _numba_cuda.syncthreads()

        # Fase 3: normalizza
        col = tx
        while col < N_cols:
            out[row, col] = out[row, col] / row_sum
            col += _SOFTMAX_BLOCK

    @_numba_cuda.jit
    def _layer_norm_kernel(x, out, gamma, beta, N_rows, N_cols, eps):
        """
        Layer Normalization per riga (ultima dimensione).
        Un blocco per riga; mean e var calcolati con riduzione shared memory.
        """
        row = _numba_cuda.blockIdx.x
        tx  = _numba_cuda.threadIdx.x

        sh_val = _numba_cuda.shared.array(shape=_SOFTMAX_BLOCK, dtype=_nb_f32)

        if row >= N_rows:
            return

        # Fase 1: somma parziale per mean
        local_sum = _nb_f32(0.0)
        col = tx
        while col < N_cols:
            local_sum += x[row, col]
            col += _SOFTMAX_BLOCK
        sh_val[tx] = local_sum
        _numba_cuda.syncthreads()

        stride = _SOFTMAX_BLOCK // 2
        while stride > 0:
            if tx < stride:
                sh_val[tx] += sh_val[tx + stride]
            _numba_cuda.syncthreads()
            stride //= 2
        mean = sh_val[0] / _nb_f32(N_cols)
        _numba_cuda.syncthreads()

        # Fase 2: varianza
        local_var = _nb_f32(0.0)
        col = tx
        while col < N_cols:
            d = x[row, col] - mean
            local_var += d * d
            col += _SOFTMAX_BLOCK
        sh_val[tx] = local_var
        _numba_cuda.syncthreads()

        stride = _SOFTMAX_BLOCK // 2
        while stride > 0:
            if tx < stride:
                sh_val[tx] += sh_val[tx + stride]
            _numba_cuda.syncthreads()
            stride //= 2
        inv_std = _nb_f32(1.0) / _math.sqrt(sh_val[0] / _nb_f32(N_cols) + eps)
        _numba_cuda.syncthreads()

        # Fase 3: normalizza + affine transform
        col = tx
        while col < N_cols:
            out[row, col] = gamma[col] * (x[row, col] - mean) * inv_std + beta[col]
            col += _SOFTMAX_BLOCK

    _NUMBA_INF_OK = True
except Exception:
    _softmax_kernel    = None
    _layer_norm_kernel = None
    _NUMBA_INF_OK      = False


# ──────────────────────────────────────────────────────────────
# 1. INT8 Quantized MatMul
# ──────────────────────────────────────────────────────────────

def lab_int8_matmul(M: int = 2048, K: int = 4096, N: int = 2048):
    """
    INT8 Quantized Matrix Multiplication vs FP32 baseline.
    Quantizzazione simmetrica: scale = max(|W|) / 127.
    INT8 occupa 4× meno memoria di FP32 → bandwidth ridotta, Tensor Core ready.

    Pipeline:
      1. Quantizza X e W in INT8 (con scale separati)
      2. MatMul in float32 (CuPy non espone INT8 nativo)
      3. Dequantizza: out_fp32 = out_int8 * scale_x * scale_w
      4. Calcola errore vs FP32 esatto
    """
    rprint(f"\n[bold cyan]INT8 Quantized MatMul — M={M}, K={K}, N={N}[/bold cyan]")
    rprint("  [dim]Quantizzazione simmetrica: scale = max(|W|)/127[/dim]")

    rng = np.random.default_rng(42)
    X_fp32 = rng.standard_normal((M, K)).astype(np.float32)
    W_fp32 = rng.standard_normal((K, N)).astype(np.float32)

    scale_x = float(np.abs(X_fp32).max()) / 127.0
    scale_w = float(np.abs(W_fp32).max()) / 127.0

    X_int8 = np.clip(X_fp32 / scale_x, -128, 127).astype(np.int8)
    W_int8 = np.clip(W_fp32 / scale_w, -128, 127).astype(np.int8)

    def cpu_fp32():
        return np.matmul(X_fp32, W_fp32)

    def cpu_int8():
        out = np.matmul(X_int8.astype(np.float32), W_int8.astype(np.float32))
        return out * scale_x * scale_w

    try:
        import cupy as cp

        X_fp32_gpu = cp.array(X_fp32)
        W_fp32_gpu = cp.array(W_fp32)
        X_int8_gpu = cp.array(X_int8.astype(np.float32))
        W_int8_gpu = cp.array(W_int8.astype(np.float32))

        def gpu_fp32():
            result = cp.matmul(X_fp32_gpu, W_fp32_gpu)
            cp.cuda.Stream.null.synchronize()
            return result

        def gpu_int8():
            out = cp.matmul(X_int8_gpu, W_int8_gpu) * scale_x * scale_w
            cp.cuda.Stream.null.synchronize()
            return out

        r_fp32 = benchmark(cpu_fp32, gpu_fp32, name="MatMul-FP32",
                           problem_size=M * K * N, warmup=2, runs=5)
        r_int8 = benchmark(cpu_int8, gpu_int8, name="MatMul-INT8",
                           problem_size=M * K * N, warmup=2, runs=5)

        tflops_fp32 = 2 * M * K * N / r_fp32.gpu_ms * 1e-12 * 1e3
        tflops_int8 = 2 * M * K * N / r_int8.gpu_ms * 1e-12 * 1e3

        out_exact = cp.matmul(X_fp32_gpu, W_fp32_gpu)
        out_quant = cp.matmul(X_int8_gpu, W_int8_gpu) * scale_x * scale_w
        quant_err = float(cp.mean(cp.abs(out_exact - out_quant)))

        rprint(f"  FP32: {r_fp32}  [{tflops_fp32:.2f} TFLOPS]")
        rprint(f"  INT8: {r_int8}  [{tflops_int8:.2f} TFLOPS]")
        rprint(f"  Errore quantizzazione (MAE): {quant_err:.4f}")
        rprint(f"  Speedup INT8 vs FP32 GPU: [green]{r_fp32.gpu_ms/r_int8.gpu_ms:.2f}x[/green]")

        return r_fp32

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fp32()
        return BenchmarkResult("MatMul-FP32", cpu_ms=t.elapsed_ms, problem_size=M*K*N)


# ──────────────────────────────────────────────────────────────
# 2. Batch Normalization
# ──────────────────────────────────────────────────────────────

def lab_batch_norm(batch: int = 256, C: int = 512, H: int = 28, W: int = 28):
    """
    Batch Normalization su tensore (N, C, H, W) float32.
    Normalizza su (N, H, W) per canale → media e varianza per canale.
    CPU: NumPy. GPU: CuPy.

    Formula:
      mu_c    = mean(X[:, c, :, :])
      sigma_c = std(X[:, c, :, :])
      Y[:, c, :, :] = gamma_c * (X - mu_c) / (sigma_c + eps) + beta_c
    """
    rprint(f"\n[bold cyan]Batch Normalization — ({batch},{C},{H},{W}) float32[/bold cyan]")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((batch, C, H, W)).astype(np.float32)
    gamma = np.ones(C, dtype=np.float32)
    beta  = np.zeros(C, dtype=np.float32)
    eps   = 1e-5

    def cpu_fn():
        mean = X.mean(axis=(0, 2, 3), keepdims=True)
        var  = X.var(axis=(0, 2, 3),  keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + eps)
        return gamma[None, :, None, None] * X_norm + beta[None, :, None, None]

    try:
        import cupy as cp

        X_gpu     = cp.array(X)
        gamma_gpu = cp.array(gamma)
        beta_gpu  = cp.array(beta)

        def gpu_fn():
            mean = X_gpu.mean(axis=(0, 2, 3), keepdims=True)
            var  = X_gpu.var(axis=(0, 2, 3),  keepdims=True)
            X_norm = (X_gpu - mean) / cp.sqrt(var + eps)
            result = gamma_gpu[None, :, None, None] * X_norm + beta_gpu[None, :, None, None]
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="BatchNorm",
                      problem_size=batch * C * H * W, warmup=2, runs=5)
        rprint(f"  {r}")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("BatchNorm", cpu_ms=t.elapsed_ms, problem_size=batch*C*H*W)


# ──────────────────────────────────────────────────────────────
# 3. Softmax (fused Numba kernel)
# ──────────────────────────────────────────────────────────────

def lab_softmax(N_rows: int = 10_000, N_cols: int = 32_000):
    """
    Softmax numericamente stabile su matrice (N_rows × N_cols).
    Simula l'output layer di un LLM con vocabolario di 32K token.

    Formula stabile: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    Sottrae il max per evitare overflow in exp().

    CPU: NumPy. GPU CuPy: vectorized. GPU Numba: kernel per-row con shared memory.
    """
    rprint(f"\n[bold cyan]Softmax — ({N_rows:,} × {N_cols:,}) float32[/bold cyan]")
    rprint(f"  [dim]Simula output layer LLM: {N_rows:,} token × vocab {N_cols:,}[/dim]")

    rng = np.random.default_rng(42)
    X = rng.standard_normal((N_rows, N_cols)).astype(np.float32)

    def cpu_fn():
        x_max = X.max(axis=1, keepdims=True)
        e = np.exp(X - x_max)
        return e / e.sum(axis=1, keepdims=True)

    try:
        import cupy as cp

        X_gpu = cp.array(X)

        def gpu_fn():
            x_max = X_gpu.max(axis=1, keepdims=True)
            e = cp.exp(X_gpu - x_max)
            result = e / e.sum(axis=1, keepdims=True)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="Softmax",
                      problem_size=N_rows * N_cols, warmup=2, runs=5)
        rprint(f"  {r}")

        if _NUMBA_INF_OK:
            rprint("  [dim]Numba fused kernel (max+exp+sum+normalize in 3 pass shared mem):[/dim]")
            try:
                X_d   = _numba_cuda.to_device(X)
                out_d = _numba_cuda.device_array_like(X)

                def numba_fn():
                    _softmax_kernel[N_rows, _SOFTMAX_BLOCK](X_d, out_d, N_rows, N_cols)
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
        return BenchmarkResult("Softmax", cpu_ms=t.elapsed_ms, problem_size=N_rows*N_cols)


# ──────────────────────────────────────────────────────────────
# 4. Layer Normalization
# ──────────────────────────────────────────────────────────────

def lab_layer_norm(N_seq: int = 512_000, d_model: int = 1024):
    """
    Layer Normalization su tensore (N_seq, d_model) float32.
    Normalizza su d_model (ultima dimensione) — diverso da BatchNorm.
    Usata in Transformer: ogni token normalizzato indipendentemente.

    Formula: LN(x)_i = gamma_i * (x_i - mean(x)) / (std(x) + eps) + beta_i
    """
    rprint(f"\n[bold cyan]Layer Normalization — ({N_seq:,}, {d_model}) float32[/bold cyan]")
    rprint(f"  [dim]Simula {N_seq//512:,} batch × 512 seq_len × {d_model} d_model[/dim]")

    rng = np.random.default_rng(42)
    X     = rng.standard_normal((N_seq, d_model)).astype(np.float32)
    gamma = np.ones(d_model,  dtype=np.float32)
    beta  = np.zeros(d_model, dtype=np.float32)
    eps   = 1e-5

    def cpu_fn():
        mean = X.mean(axis=1, keepdims=True)
        var  = X.var(axis=1,  keepdims=True)
        return gamma * (X - mean) / np.sqrt(var + eps) + beta

    try:
        import cupy as cp

        X_gpu     = cp.array(X)
        gamma_gpu = cp.array(gamma)
        beta_gpu  = cp.array(beta)

        def gpu_fn():
            mean = X_gpu.mean(axis=1, keepdims=True)
            var  = X_gpu.var(axis=1,  keepdims=True)
            result = gamma_gpu * (X_gpu - mean) / cp.sqrt(var + eps) + beta_gpu
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="LayerNorm",
                      problem_size=N_seq * d_model, warmup=2, runs=5)
        rprint(f"  {r}")

        if _NUMBA_INF_OK:
            rprint("  [dim]Numba fused kernel (mean+var+normalize in 3 pass shared mem):[/dim]")
            try:
                X_d   = _numba_cuda.to_device(X)
                out_d = _numba_cuda.device_array_like(X)
                g_d   = _numba_cuda.to_device(gamma)
                b_d   = _numba_cuda.to_device(beta)

                def numba_fn():
                    _layer_norm_kernel[N_seq, _SOFTMAX_BLOCK](
                        X_d, out_d, g_d, b_d, N_seq, d_model, eps)
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
        return BenchmarkResult("LayerNorm", cpu_ms=t.elapsed_ms, problem_size=N_seq*d_model)


# ──────────────────────────────────────────────────────────────
# 5. Scaling Analysis — INT8 MatMul
# ──────────────────────────────────────────────────────────────

def lab_inference_scaling():
    """Scaling: FP32 vs INT8 matmul al variare di M=N=K."""
    rprint("\n[bold]Scaling Analysis — FP32 vs INT8 MatMul[/bold]")

    sizes = [512, 1024, 2048, 4096, 8192]

    table = Table(title="Scaling: FP32 vs INT8 MatMul GPU",
                  header_style="bold magenta")
    table.add_column("M=N=K",   style="cyan", justify="right")
    table.add_column("FP32 (ms)", justify="right")
    table.add_column("INT8 (ms)", justify="right")
    table.add_column("INT8/FP32", justify="right", style="green")
    table.add_column("TFLOPS FP32", justify="right")
    table.add_column("TFLOPS INT8", justify="right")

    try:
        import cupy as cp
        rng = np.random.default_rng(42)

        for N in sizes:
            X = rng.standard_normal((N, N)).astype(np.float32)
            W = rng.standard_normal((N, N)).astype(np.float32)
            scale_x = float(np.abs(X).max()) / 127.0
            scale_w = float(np.abs(W).max()) / 127.0
            X8 = np.clip(X / scale_x, -128, 127).astype(np.int8)
            W8 = np.clip(W / scale_w, -128, 127).astype(np.int8)

            X_gpu  = cp.array(X)
            W_gpu  = cp.array(W)
            X8_gpu = cp.array(X8.astype(np.float32))
            W8_gpu = cp.array(W8.astype(np.float32))

            def gpu_fp32():
                cp.matmul(X_gpu, W_gpu)
                cp.cuda.Stream.null.synchronize()

            def gpu_int8():
                cp.matmul(X8_gpu, W8_gpu)
                cp.cuda.Stream.null.synchronize()

            for _ in range(2):
                gpu_fp32(); gpu_int8()

            fp32_times, int8_times = [], []
            for _ in range(5):
                with GPUTimer() as t: gpu_fp32()
                fp32_times.append(t.elapsed_ms)
                with GPUTimer() as t: gpu_int8()
                int8_times.append(t.elapsed_ms)

            fp32_ms = float(np.median(fp32_times))
            int8_ms = float(np.median(int8_times))
            flops   = 2 * N * N * N
            tf_fp32 = flops / fp32_ms * 1e-12 * 1e3
            tf_int8 = flops / int8_ms * 1e-12 * 1e3
            ratio   = fp32_ms / int8_ms if int8_ms > 0 else 0
            color   = "[green]" if ratio > 1.5 else "[yellow]"

            table.add_row(f"{N}", f"{fp32_ms:.2f}", f"{int8_ms:.2f}",
                          f"{color}{ratio:.2f}x[/]",
                          f"{tf_fp32:.2f}", f"{tf_int8:.2f}")

            del X_gpu, W_gpu, X8_gpu, W8_gpu
            cp.get_default_memory_pool().free_all_blocks()

        console.print(table)

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — scaling saltata[/yellow]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]LAB 13 — Deep Learning Inference GPU[/bold cyan]")
    console.print("=" * 60)
    console.print("[dim]INT8 MatMul, Batch Norm, Softmax, Layer Norm[/dim]\n")

    results = []
    results.append(lab_int8_matmul())
    results.append(lab_batch_norm())
    results.append(lab_softmax())
    results.append(lab_layer_norm())
    lab_inference_scaling()

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
            out = Path(__file__).resolve().parent.parent / "outputs" / "lab13_benchmark.png"
            plot_cpu_vs_gpu(valid,
                            title="LAB 13 — DL Inference: CPU vs GPU",
                            save_path=str(out), show=False)
    except Exception as e:
        rprint(f"[dim]Grafico non generato: {e}[/dim]")


if __name__ == "__main__":
    main()
