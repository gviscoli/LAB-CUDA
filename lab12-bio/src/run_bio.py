"""
lab12-bio/src/run_bio.py
=========================
LAB 12 — Bioinformatics GPU
Algoritmi: Smith-Waterman, K-mer Counting, Edit Distance, Scaling
Paradigma: NumPy CPU baseline -> CuPy GPU -> Numba CUDA kernel custom
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

# Alphabet DNA
_ALPHA = np.array([0, 1, 2, 3], dtype=np.int8)  # A=0, C=1, G=2, T=3

# ── Numba CUDA kernels (livello modulo) ────────────────────────────────────────
try:
    from numba import cuda as _numba_cuda, int32 as _nb_i32
    import math as _math

    @_numba_cuda.jit
    def _sw_antidiag_kernel(H, enc_a, enc_b, k,
                            match_score, mismatch_score, gap_score,
                            m, n):
        """
        Smith-Waterman: processa l'anti-diagonale k in parallelo.
        I thread calcolano H[i,j] per tutti (i,j) con i+j == k.
        Ogni thread: indice tid -> i = max(1, k-n+1) + tid, j = k - i.
        """
        i_start = max(1, k - n + 1)
        i_end   = min(k, m)
        tid = _numba_cuda.threadIdx.x + _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x
        i = i_start + tid
        if i > i_end:
            return
        j = k - i
        if j < 1 or j > n:
            return
        # Score match/mismatch
        s = match_score if enc_a[i-1] == enc_b[j-1] else mismatch_score
        diag = H[i-1, j-1] + s
        up   = H[i-1, j]   + gap_score
        left = H[i,   j-1] + gap_score
        val  = diag
        if up   > val: val = up
        if left > val: val = left
        if 0    > val: val = 0
        H[i, j] = val

    @_numba_cuda.jit
    def _edit_dist_kernel(seqs_a_enc, seqs_b_enc, results, L):
        """
        Edit distance (Levenshtein) per un batch di coppie.
        Ogni thread calcola la distanza per una coppia usando DP 1D rolling.
        """
        pair = _numba_cuda.threadIdx.x + _numba_cuda.blockIdx.x * _numba_cuda.blockDim.x
        if pair >= results.shape[0]:
            return
        # DP array in local memory (registri/stack), max L=128
        prev = _numba_cuda.local.array(shape=129, dtype=_nb_i32)
        curr = _numba_cuda.local.array(shape=129, dtype=_nb_i32)
        # Init prima riga
        for j in range(L + 1):
            prev[j] = j
        for i in range(1, L + 1):
            curr[0] = i
            for j in range(1, L + 1):
                if seqs_a_enc[pair, i-1] == seqs_b_enc[pair, j-1]:
                    cost = 0
                else:
                    cost = 1
                del_op  = prev[j]     + 1
                ins_op  = curr[j-1]   + 1
                sub_op  = prev[j-1]   + cost
                val = del_op
                if ins_op < val: val = ins_op
                if sub_op < val: val = sub_op
                curr[j] = val
            # swap
            for j in range(L + 1):
                prev[j] = curr[j]
        results[pair] = prev[L]

    _NUMBA_BIO_OK = True
except Exception:
    _sw_antidiag_kernel = None
    _edit_dist_kernel   = None
    _NUMBA_BIO_OK       = False


def _gen_dna(length: int, rng) -> np.ndarray:
    """Genera sequenza DNA casuale come array int8 (A=0,C=1,G=2,T=3)."""
    return rng.integers(0, 4, size=length, dtype=np.int8)


# ──────────────────────────────────────────────────────────────
# 1. Smith-Waterman Local Alignment
# ──────────────────────────────────────────────────────────────

def lab_smith_waterman(L: int = 2000, match: int = 2,
                       mismatch: int = -1, gap: int = -2):
    """
    Smith-Waterman local sequence alignment su due sequenze di lunghezza L.
    CPU: anti-diagonal wavefront con NumPy (L^2 celle, O(L^2) work).
    GPU Numba: ogni anti-diagonale processata in parallelo su GPU.

    Ricorrenza:
      H[i,j] = max(0,
                   H[i-1,j-1] + s(a[i],b[j]),   # match/mismatch
                   H[i-1,j]   + gap,              # gap in A
                   H[i,j-1]   + gap)              # gap in B
    """
    rprint(f"\n[bold cyan]Smith-Waterman — L={L} bp, match={match}, gap={gap}[/bold cyan]")
    rprint("  [dim]Anti-diagonal wavefront: ogni diagonale è completamente parallela[/dim]")

    rng = np.random.default_rng(42)
    enc_a = _gen_dna(L, rng)
    enc_b = _gen_dna(L, rng)

    def cpu_fn():
        H = np.zeros((L + 1, L + 1), dtype=np.int32)
        for k in range(1, 2 * L + 1):
            i_s = max(1, k - L)
            i_e = min(k, L)
            ia = np.arange(i_s, i_e + 1)
            ja = k - ia
            s = np.where(enc_a[ia - 1] == enc_b[ja - 1], match, mismatch)
            diag = H[ia - 1, ja - 1] + s
            up   = H[ia - 1, ja]     + gap
            left = H[ia,     ja - 1] + gap
            H[ia, ja] = np.maximum(0, np.maximum(diag, np.maximum(up, left)))
        return H.max()

    try:
        import cupy as cp

        enc_a_gpu = cp.array(enc_a)
        enc_b_gpu = cp.array(enc_b)

        def gpu_cupy_fn():
            H = cp.zeros((L + 1, L + 1), dtype=cp.int32)
            for k in range(1, 2 * L + 1):
                i_s = max(1, k - L)
                i_e = min(k, L)
                ia = cp.arange(i_s, i_e + 1)
                ja = k - ia
                s = cp.where(enc_a_gpu[ia - 1] == enc_b_gpu[ja - 1], match, mismatch)
                diag = H[ia - 1, ja - 1] + s
                up   = H[ia - 1, ja]     + gap
                left = H[ia,     ja - 1] + gap
                H[ia, ja] = cp.maximum(0, cp.maximum(diag, cp.maximum(up, left)))
            cp.cuda.Stream.null.synchronize()
            return int(H.max())

        r = benchmark(cpu_fn, gpu_cupy_fn, name="SmithWaterman",
                      problem_size=L * L, warmup=1, runs=3)
        mcups = L * L / r.gpu_ms * 1e-3  # MCUPS = Mega Cell Updates Per Second
        rprint(f"  {r}")
        rprint(f"  Throughput GPU: {mcups:.1f} MCUPS")

        if _NUMBA_BIO_OK:
            rprint("  [dim]Numba anti-diagonal kernel:[/dim]")
            try:
                H_d     = _numba_cuda.to_device(np.zeros((L+1, L+1), dtype=np.int32))
                enc_a_d = _numba_cuda.to_device(enc_a)
                enc_b_d = _numba_cuda.to_device(enc_b)

                THREADS = 256

                def numba_fn():
                    H_d[:] = 0
                    for k in range(1, 2 * L + 1):
                        diag_len = min(k, L) - max(1, k - L) + 1
                        blocks = (diag_len + THREADS - 1) // THREADS
                        _sw_antidiag_kernel[blocks, THREADS](
                            H_d, enc_a_d, enc_b_d, k,
                            match, mismatch, gap, L, L)
                    _numba_cuda.synchronize()

                for _ in range(1):
                    numba_fn()

                times_nb = []
                for _ in range(3):
                    with GPUTimer() as t:
                        numba_fn()
                    times_nb.append(t.elapsed_ms)
                nb_ms = float(np.median(times_nb))
                rprint(f"  Numba: {nb_ms:.1f} ms  speedup=[green]{r.cpu_ms/nb_ms:.1f}x[/green] vs CPU")
            except Exception as e:
                rprint(f"  [yellow]Numba error: {e}[/yellow]")

        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("SmithWaterman", cpu_ms=t.elapsed_ms, problem_size=L * L)


# ──────────────────────────────────────────────────────────────
# 2. K-mer Counting
# ──────────────────────────────────────────────────────────────

def lab_kmer_counting(seq_len: int = 10_000_000, k: int = 8):
    """
    K-mer counting: conta le occorrenze di ogni k-mer in una sequenza DNA.
    k=8 -> 4^8 = 65536 k-mer possibili.
    Encoding: ogni k-mer = intero in base 4 (A=0, C=1, G=2, T=3).

    CPU: sliding window NumPy con np.bincount.
    GPU: sliding window CuPy con cp.bincount.

    Applicazioni: assemblaggio del genoma (de Bruijn graph), rilevamento di
    varianti (variant calling), classificazione metagenomi, ricerca di motivi.
    """
    rprint(f"\n[bold cyan]K-mer Counting — seq={seq_len:,} bp, k={k} (4^k={4**k:,} k-mer)[/bold cyan]")

    rng = np.random.default_rng(42)
    seq = rng.integers(0, 4, size=seq_len, dtype=np.int32)

    # Pre-calcola potenze di 4 per encoding
    powers = np.array([4 ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    n_kmers = seq_len - k + 1

    def cpu_fn():
        # Sliding window: ogni k-mer = dot(seq[i:i+k], powers)
        # Usiamo un approccio rolling: aggiorna incrementalmente
        kmers = np.zeros(n_kmers, dtype=np.int64)
        # Primo k-mer
        kmers[0] = int(np.dot(seq[:k].astype(np.int64), powers))
        base4_k = int(4 ** k)
        for i in range(1, n_kmers):
            kmers[i] = (kmers[i-1] - int(seq[i-1]) * int(powers[0])) * 4 + int(seq[i+k-1])
        return np.bincount(kmers, minlength=4**k)

    try:
        import cupy as cp

        seq_gpu    = cp.array(seq)
        powers_gpu = cp.array(powers)

        def gpu_fn():
            # Approccio vettoriale: conv-like con strides
            idx = cp.arange(n_kmers, dtype=cp.int64)
            # Costruisce matrice n_kmers × k con seq[i:i+k]
            rows = (idx[:, None] + cp.arange(k, dtype=cp.int64)[None, :])
            kmers = cp.sum(seq_gpu[rows] * powers_gpu[None, :], axis=1)
            result = cp.bincount(kmers.astype(cp.int32), minlength=4**k)
            cp.cuda.Stream.null.synchronize()
            return result

        r = benchmark(cpu_fn, gpu_fn, name="KmerCount",
                      problem_size=seq_len, warmup=1, runs=3)
        throughput = seq_len / r.gpu_ms * 1e-3  # Mbp/s
        rprint(f"  {r}")
        rprint(f"  Throughput GPU: {throughput:.1f} Mbp/s")
        return r

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — solo CPU[/yellow]")
        with CPUTimer() as t:
            cpu_fn()
        return BenchmarkResult("KmerCount", cpu_ms=t.elapsed_ms, problem_size=seq_len)


# ──────────────────────────────────────────────────────────────
# 3. Edit Distance — batch
# ──────────────────────────────────────────────────────────────

def lab_edit_distance(n_pairs: int = 10_000, L: int = 64):
    """
    Edit distance (Levenshtein) su un batch di n_pairs coppie, ciascuna di lunghezza L.
    CPU: DP standard NumPy per ogni coppia.
    GPU CuPy: tutte le coppie in parallelo (batch matriciale).
    GPU Numba: ogni thread gestisce una coppia con DP in local memory.

    Operazioni: sostituzione (cost=1), inserimento (cost=1), cancellazione (cost=1).
    Ricorrenza DP:
      D[i,j] = min(D[i-1,j]+1, D[i,j-1]+1, D[i-1,j-1]+cost(i,j))
    """
    rprint(f"\n[bold cyan]Edit Distance batch — {n_pairs:,} coppie × L={L}[/bold cyan]")

    rng = np.random.default_rng(42)
    seqs_a = rng.integers(0, 4, size=(n_pairs, L), dtype=np.int8)
    seqs_b = rng.integers(0, 4, size=(n_pairs, L), dtype=np.int8)

    def cpu_fn():
        results = np.zeros(n_pairs, dtype=np.int32)
        for p in range(n_pairs):
            prev = np.arange(L + 1, dtype=np.int32)
            for i in range(1, L + 1):
                curr = np.empty(L + 1, dtype=np.int32)
                curr[0] = i
                cost = (seqs_a[p, i-1] != seqs_b[p, :]).astype(np.int32)
                curr[1:] = np.minimum(
                    np.minimum(prev[1:] + 1, curr[:L] + 1),
                    prev[:L] + cost
                )
                prev = curr
            results[p] = prev[L]
        return results

    try:
        import cupy as cp

        sa_gpu = cp.array(seqs_a.astype(np.int32))
        sb_gpu = cp.array(seqs_b.astype(np.int32))

        def gpu_fn():
            # Batch DP: prev shape (n_pairs, L+1)
            prev = cp.tile(cp.arange(L + 1, dtype=cp.int32), (n_pairs, 1))
            for i in range(1, L + 1):
                curr = cp.empty((n_pairs, L + 1), dtype=cp.int32)
                curr[:, 0] = i
                cost = (sa_gpu[:, i-1:i] != sb_gpu).astype(cp.int32)
                del_op  = prev[:, 1:] + 1
                ins_op  = curr[:, :L] + 1
                sub_op  = prev[:, :L] + cost
                curr[:, 1:] = cp.minimum(del_op, cp.minimum(ins_op, sub_op))
                prev = curr
            cp.cuda.Stream.null.synchronize()
            return prev[:, L]

        r = benchmark(cpu_fn, gpu_fn, name="EditDistance",
                      problem_size=n_pairs, warmup=1, runs=3)
        rprint(f"  {r}")

        if _NUMBA_BIO_OK and L <= 128:
            rprint("  [dim]Numba kernel (1 thread per coppia):[/dim]")
            try:
                sa_d = _numba_cuda.to_device(seqs_a.astype(np.int32))
                sb_d = _numba_cuda.to_device(seqs_b.astype(np.int32))
                res_d = _numba_cuda.device_array(n_pairs, dtype=np.int32)

                THREADS = 256
                blocks = (n_pairs + THREADS - 1) // THREADS

                def numba_fn():
                    _edit_dist_kernel[blocks, THREADS](sa_d, sb_d, res_d, L)
                    _numba_cuda.synchronize()

                for _ in range(2):
                    numba_fn()

                times_nb = []
                for _ in range(3):
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
        return BenchmarkResult("EditDistance", cpu_ms=t.elapsed_ms, problem_size=n_pairs)


# ──────────────────────────────────────────────────────────────
# 4. Scaling Analysis — Smith-Waterman
# ──────────────────────────────────────────────────────────────

def lab_bio_scaling():
    """Scaling Smith-Waterman CuPy vs NumPy al variare di L."""
    rprint("\n[bold]Scaling Analysis — Smith-Waterman GPU vs CPU[/bold]")

    lengths = [500, 1000, 2000, 4000]

    table = Table(title="Scaling: Smith-Waterman (anti-diagonal wavefront)",
                  header_style="bold magenta")
    table.add_column("L (bp)",   style="cyan", justify="right")
    table.add_column("Celle",    justify="right")
    table.add_column("CPU (ms)", justify="right")
    table.add_column("GPU (ms)", justify="right")
    table.add_column("Speedup",  justify="right", style="green")
    table.add_column("MCUPS GPU", justify="right")

    try:
        import cupy as cp

        rng = np.random.default_rng(42)

        for L in lengths:
            enc_a = rng.integers(0, 4, size=L, dtype=np.int8)
            enc_b = rng.integers(0, 4, size=L, dtype=np.int8)
            enc_a_gpu = cp.array(enc_a.astype(np.int32))
            enc_b_gpu = cp.array(enc_b.astype(np.int32))

            match, mismatch, gap = 2, -1, -2

            def make_cpu(a, b, ll):
                def fn():
                    H = np.zeros((ll+1, ll+1), dtype=np.int32)
                    for k in range(1, 2*ll+1):
                        i_s = max(1, k-ll); i_e = min(k, ll)
                        ia = np.arange(i_s, i_e+1); ja = k - ia
                        s = np.where(a[ia-1] == b[ja-1], match, mismatch)
                        diag = H[ia-1, ja-1]+s; up = H[ia-1, ja]+gap; left = H[ia, ja-1]+gap
                        H[ia, ja] = np.maximum(0, np.maximum(diag, np.maximum(up, left)))
                    return H.max()
                return fn

            def make_gpu(ag, bg, ll):
                def fn():
                    H = cp.zeros((ll+1, ll+1), dtype=cp.int32)
                    for k in range(1, 2*ll+1):
                        i_s = max(1, k-ll); i_e = min(k, ll)
                        ia = cp.arange(i_s, i_e+1); ja = k - ia
                        s = cp.where(ag[ia-1] == bg[ja-1], match, mismatch)
                        diag = H[ia-1, ja-1]+s; up = H[ia-1, ja]+gap; left = H[ia, ja-1]+gap
                        H[ia, ja] = cp.maximum(0, cp.maximum(diag, cp.maximum(up, left)))
                    cp.cuda.Stream.null.synchronize()
                fn = fn
                return fn

            cpu_fn = make_cpu(enc_a, enc_b, L)
            gpu_fn = make_gpu(enc_a_gpu, enc_b_gpu, L)

            cpu_times = []
            for _ in range(2):
                with CPUTimer() as tc:
                    cpu_fn()
                cpu_times.append(tc.elapsed_ms)
            cpu_ms = float(np.median(cpu_times))

            gpu_times = []
            for _ in range(2):
                with GPUTimer() as tg:
                    gpu_fn()
                gpu_times.append(tg.elapsed_ms)
            gpu_ms = float(np.median(gpu_times))

            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
            mcups   = L * L / gpu_ms * 1e-3
            color   = "[green]" if speedup > 5 else "[yellow]" if speedup > 1 else "[red]"
            table.add_row(f"{L:,}", f"{L*L:,}", f"{cpu_ms:.1f}", f"{gpu_ms:.1f}",
                          f"{color}{speedup:.1f}x[/]", f"{mcups:.1f}")

        console.print(table)

    except ImportError:
        rprint("  [yellow]CuPy non disponibile — scaling saltata[/yellow]")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]LAB 12 — Bioinformatics GPU[/bold cyan]")
    console.print("=" * 60)
    console.print("[dim]Smith-Waterman, K-mer Counting, Edit Distance[/dim]\n")

    results = []
    results.append(lab_smith_waterman())
    results.append(lab_kmer_counting())
    results.append(lab_edit_distance())
    lab_bio_scaling()

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
            out = Path(__file__).resolve().parent.parent / "outputs" / "lab12_benchmark.png"
            plot_cpu_vs_gpu(valid,
                            title="LAB 12 — Bioinformatics: CPU vs GPU",
                            save_path=str(out), show=False)
    except Exception as e:
        rprint(f"[dim]Grafico non generato: {e}[/dim]")


if __name__ == "__main__":
    main()
