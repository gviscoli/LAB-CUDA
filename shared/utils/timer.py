"""
shared/utils/timer.py — Timer CPU vs GPU
Utility per misurare e confrontare performance CPU/GPU.
"""
import time
import contextlib
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    cpu_ms: float = 0.0
    gpu_ms: float = 0.0
    speedup: float = 0.0
    problem_size: int = 0
    throughput_gbs: float = 0.0
    notes: str = ""

    def __str__(self):
        return (f"{self.name}: CPU={self.cpu_ms:.2f}ms | "
                f"GPU={self.gpu_ms:.2f}ms | "
                f"Speedup={self.speedup:.1f}x")


class CPUTimer:
    """Context manager per timing CPU."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000


class GPUTimer:
    """Context manager per timing GPU con sincronizzazione CUDA."""
    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        try:
            import cupy as cp
            self._stream = cp.cuda.Stream()
            self._start = cp.cuda.Event()
            self._end = cp.cuda.Event()
            self._start.record(self._stream)
        except ImportError:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        try:
            import cupy as cp
            self._end.record(self._stream)
            self._end.synchronize()
            self.elapsed_ms = cp.cuda.get_elapsed_time(self._start, self._end)
        except ImportError:
            self.elapsed_ms = (time.perf_counter() - self._start_time) * 1000


def benchmark(
    fn_cpu,
    fn_gpu,
    name: str = "benchmark",
    warmup: int = 3,
    runs: int = 10,
    problem_size: int = 0,
    bytes_accessed: int = 0,
) -> BenchmarkResult:
    """
    Esegui benchmark completo CPU vs GPU con warmup e media su N run.

    Args:
        fn_cpu:         funzione CPU da misurare
        fn_gpu:         funzione GPU da misurare
        name:           nome del benchmark
        warmup:         run di riscaldamento (non conteggiati)
        runs:           run da mediare
        problem_size:   dimensione del problema (es. N elementi)
        bytes_accessed: byte letti/scritti (per calcolo bandwidth)
    """
    # Warmup CPU
    for _ in range(warmup):
        fn_cpu()

    # Misura CPU
    cpu_times = []
    for _ in range(runs):
        with CPUTimer() as t:
            fn_cpu()
        cpu_times.append(t.elapsed_ms)
    cpu_ms = np.median(cpu_times)

    # Warmup GPU
    for _ in range(warmup):
        fn_gpu()

    # Misura GPU
    gpu_times = []
    for _ in range(runs):
        with GPUTimer() as t:
            fn_gpu()
        gpu_times.append(t.elapsed_ms)
    gpu_ms = np.median(gpu_times)

    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
    throughput = (bytes_accessed / (gpu_ms / 1000) / 1e9) if bytes_accessed > 0 else 0

    return BenchmarkResult(
        name=name,
        cpu_ms=cpu_ms,
        gpu_ms=gpu_ms,
        speedup=speedup,
        problem_size=problem_size,
        throughput_gbs=throughput,
    )
