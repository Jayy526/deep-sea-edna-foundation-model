"""Seeding, timing and resource measurement helpers.

Everything reported here is measured, never estimated. Where a measurement is
unavailable in this environment the value is reported as
``"Not available with the current dataset/experimental setup."`` rather than
being filled in with a guess.
"""

from __future__ import annotations

import os
import platform
import random
import time
from contextlib import contextmanager
from typing import Any

NOT_AVAILABLE = "Not available with the current dataset/experimental setup."


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we can reach, for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def process_memory_mb() -> float | str:
    """Resident set size of this process, in MB."""
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 1024**2, 2)
    except Exception:
        return NOT_AVAILABLE


def gpu_memory_mb() -> dict[str, Any]:
    """Peak CUDA memory for this process, in MB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        return {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
            "total_mb": round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
            ),
        }
    except ImportError:
        return {"available": False}


def environment_report() -> dict[str, Any]:
    """Record the exact environment the experiment ran in."""
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy

        report["numpy"] = numpy.__version__
    except ImportError:
        pass
    for name in ("torch", "transformers", "sklearn", "umap", "pandas", "pyarrow"):
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            report[name] = None
    try:
        import torch

        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["cuda_version"] = torch.version.cuda
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
            )
    except ImportError:
        report["cuda_available"] = False
    return report


class Timer:
    """Wall-clock timer with a throughput helper."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.elapsed = 0.0

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start
        return self.elapsed

    def rate(self, n_items: int) -> dict[str, float]:
        elapsed = self.elapsed or (time.perf_counter() - self.start)
        return {
            "elapsed_seconds": round(elapsed, 4),
            "items": n_items,
            "items_per_second": round(n_items / elapsed, 3) if elapsed > 0 else 0.0,
            "seconds_per_item": round(elapsed / n_items, 8) if n_items else 0.0,
        }


@contextmanager
def timed(label: str, sink: dict | None = None):
    """Context manager that records elapsed seconds into ``sink[label]``."""
    timer = Timer()
    try:
        yield timer
    finally:
        timer.stop()
        if sink is not None:
            sink[label] = round(timer.elapsed, 4)
