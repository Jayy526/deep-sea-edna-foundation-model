"""Phase 9 -- measured computational performance.

Everything here is a measurement taken on this machine. Nothing is estimated
except the explicitly-labelled full-dataset projection, which is derived from
the measured throughput and labelled as a projection.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from models.encoder import SequenceEncoder
from utils.runtime import Timer, environment_report, gpu_memory_mb, process_memory_mb


def batch_size_sweep(
    encoder: SequenceEncoder,
    sequences: Sequence[str],
    batch_sizes: list[int],
    warmup_batches: int = 2,
) -> dict[str, Any]:
    """Measure throughput and peak memory across batch sizes.

    Answers a real engineering question: what batch size should the full run
    use, and where does this 6 GB GPU stop scaling?
    """
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        torch = None
        cuda = False

    rows = []
    for batch_size in batch_sizes:
        if batch_size > len(sequences):
            continue
        # Warm-up (CUDA kernel autotuning, allocator growth) is excluded.
        for i in range(warmup_batches):
            encoder.encode_batch(sequences[i * batch_size : (i + 1) * batch_size])
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        timer = Timer()
        n = 0
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            encoder.encode_batch(chunk)
            n += len(chunk)
        if cuda:
            torch.cuda.synchronize()
        elapsed = timer.stop()

        row = {
            "batch_size": batch_size,
            "sequences": n,
            "elapsed_seconds": round(elapsed, 4),
            "sequences_per_second": round(n / elapsed, 2),
            "milliseconds_per_sequence": round(1000 * elapsed / n, 5),
            "peak_gpu_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if cuda
            else None,
            "peak_gpu_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2)
            if cuda
            else None,
            "process_memory_mb": process_memory_mb(),
        }
        rows.append(row)
        print(
            f"[perf] batch={batch_size:<4} {row['sequences_per_second']:>9,.1f} seq/s  "
            f"gpu_peak={row['peak_gpu_allocated_mb']} MB"
        )

    best = max(rows, key=lambda r: r["sequences_per_second"]) if rows else None
    return {
        "measurement": "batch-size sweep, warm-up excluded, CUDA-synchronised",
        "sequences_per_configuration": len(sequences),
        "rows": rows,
        "best_batch_size": best["batch_size"] if best else None,
        "best_throughput_seq_per_second": best["sequences_per_second"] if best else None,
        "encoder": encoder.describe(),
        "environment": environment_report(),
    }


def full_dataset_projection(
    throughput_seq_per_second: float,
    embedding_dim: int,
    n_reads_full: int,
    label: str,
) -> dict[str, Any]:
    """Project full-dataset cost from MEASURED throughput. Labelled as a projection."""
    seconds = n_reads_full / throughput_seq_per_second
    storage_bytes = n_reads_full * embedding_dim * 4
    return {
        "label": f"PROJECTION for {label} (derived from measured throughput, not executed)",
        "full_dataset_reads": n_reads_full,
        "measured_throughput_seq_per_second": round(throughput_seq_per_second, 2),
        "projected_seconds": round(seconds, 1),
        "projected_minutes": round(seconds / 60, 2),
        "projected_hours": round(seconds / 3600, 3),
        "projected_embedding_storage_bytes": storage_bytes,
        "projected_embedding_storage_gb": round(storage_bytes / 1024**3, 3),
        "basis": "linear extrapolation of measured steady-state throughput at the chosen batch size",
    }
