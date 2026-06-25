"""
shared/utils/plotter.py — Visualizzazione Performance
Grafici speedup, roofline model, scaling efficiency.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import List
from timer import BenchmarkResult


COLORS = {
    "cpu":    "#4C72B0",
    "gpu":    "#DD8452",
    "speedup":"#55A868",
    "theory": "#C44E52",
}


def plot_cpu_vs_gpu(
    results: List[BenchmarkResult],
    title: str = "CPU vs GPU Performance",
    save_path: str = None,
):
    """Bar chart comparativo CPU vs GPU con speedup annotato."""
    names    = [r.name for r in results]
    cpu_ms   = [r.cpu_ms for r in results]
    gpu_ms   = [r.gpu_ms for r in results]
    speedups = [r.speedup for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Tempi
    bars_cpu = ax1.bar(x - width/2, cpu_ms, width, label="CPU (NumPy)", color=COLORS["cpu"])
    bars_gpu = ax1.bar(x + width/2, gpu_ms, width, label="GPU (CuPy/Numba)", color=COLORS["gpu"])
    ax1.set_xlabel("Algoritmo")
    ax1.set_ylabel("Tempo (ms)")
    ax1.set_title("Tempo di Esecuzione")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.legend()
    ax1.set_yscale("log")

    for bar in bars_cpu:
        ax1.annotate(f"{bar.get_height():.1f}",
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=7)

    # Speedup
    bars_sp = ax2.bar(x, speedups, color=COLORS["speedup"], alpha=0.85)
    ax2.axhline(y=1, color="red", linestyle="--", alpha=0.5, label="Baseline (1x)")
    ax2.set_xlabel("Algoritmo")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("GPU Speedup vs CPU")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.legend()

    for bar, sp in zip(bars_sp, speedups):
        ax2.annotate(f"{sp:.1f}x",
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Salvato: {save_path}")

    plt.show()
    return fig


def plot_roofline(
    peak_flops_tflops: float = 82.6,    # RTX 4080 FP32 peak TFLOPS
    peak_bandwidth_gbs: float = 716.8,  # RTX 4080 memory bandwidth GB/s
    points: List[dict] = None,
    save_path: str = None,
):
    """
    Roofline Model per RTX 4080.
    Mostra dove si posizionano gli algoritmi rispetto ai limiti hardware.
    Fonte: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#roofline-model

    points: [{"name": str, "flops": float, "bytes": float, "achieved_flops": float}]
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Ridge point (operational intensity dove i due tetti si incontrano)
    ridge_oi = peak_flops_tflops * 1e12 / (peak_bandwidth_gbs * 1e9)

    # Asse X: operational intensity (FLOP/byte)
    oi = np.logspace(-2, 4, 1000)

    # Roofline
    memory_roof  = peak_bandwidth_gbs * oi        # GFLOPS
    compute_roof = np.full_like(oi, peak_flops_tflops * 1000)  # GFLOPS
    roof = np.minimum(memory_roof, compute_roof)

    ax.loglog(oi, roof, "k-", linewidth=2.5, label="Roofline RTX 4080")
    ax.axvline(x=ridge_oi, color="gray", linestyle="--", alpha=0.6)
    ax.text(ridge_oi * 1.1, peak_flops_tflops * 800,
            f"Ridge: {ridge_oi:.0f} FLOP/byte", fontsize=9, color="gray")

    # Annotazioni regioni
    ax.text(0.02, peak_bandwidth_gbs * 0.02 * 0.5,
            "Memory\nBound", fontsize=10, color="blue", alpha=0.6)
    ax.text(ridge_oi * 2, peak_flops_tflops * 800,
            "Compute\nBound", fontsize=10, color="red", alpha=0.6)

    # Punti algoritmi
    if points:
        colors_pts = plt.cm.tab10(np.linspace(0, 1, len(points)))
        for pt, color in zip(points, colors_pts):
            oi_pt = pt["flops"] / pt["bytes"]
            ax.scatter([oi_pt], [pt["achieved_flops"] / 1e9],
                       s=120, color=color, zorder=5, label=pt["name"])
            ax.annotate(pt["name"],
                        xy=(oi_pt, pt["achieved_flops"] / 1e9),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Operational Intensity (FLOP/byte)", fontsize=11)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
    ax.set_title("Roofline Model — RTX 4080 Ada Lovelace", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # Hardware specs
    info_text = (f"RTX 4080 Ada Lovelace\n"
                 f"Peak FP32: {peak_flops_tflops} TFLOPS\n"
                 f"Bandwidth: {peak_bandwidth_gbs} GB/s\n"
                 f"VRAM: 16GB GDDR6X")
    ax.text(0.98, 0.05, info_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Roofline salvato: {save_path}")

    plt.show()
    return fig
