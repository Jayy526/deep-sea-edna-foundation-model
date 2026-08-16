"""Embedding-extraction stage driver (works for both encoder families)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from embeddings.store import EmbeddingStore, embedding_health
from models.encoder import SequenceEncoder
from preprocessing.run import load_subset
from utils import config as cfgutil
from utils.runtime import Timer, environment_report, gpu_memory_mb, process_memory_mb


def extract(
    cfg: dict,
    encoder: SequenceEncoder,
    subset_file: str | Path,
    store_name: str,
    batch_size: int,
    shard_size: int,
    run_label: str,
    compatibility: dict | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Encode every read in ``subset_file`` and persist the embeddings.

    ``run_label`` must be either ``"TEST RUN"`` or ``"FULL SUBSET RUN"`` so the
    scope of every metrics file is unambiguous.
    """
    root = cfgutil.output_dir(cfg, "embeddings")
    seq_ids, sequences = load_subset(subset_file)
    total = len(sequences)

    store = EmbeddingStore(root, store_name, encoder.embedding_dim, shard_size)
    if force:
        for path in store.dir.glob("*"):
            path.unlink()
        store = EmbeddingStore(root, store_name, encoder.embedding_dim, shard_size)

    done = store.n_committed
    if done >= total:
        print(f"[encode] '{store_name}' already complete ({done:,} rows). Reusing.")
        import json

        return json.loads(store.manifest_path.read_text(encoding="utf-8")).get(
            "metrics", {"reused": True, "rows": done}
        )
    if done:
        print(f"[encode] Resuming '{store_name}' from row {done:,}/{total:,}")

    seq_ids, sequences = seq_ids[done:], sequences[done:]

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    n_ok = 0
    n_failed = 0
    failures: list[dict] = []
    timer = Timer()
    cursor = 0

    from tqdm import tqdm

    with tqdm(total=total, initial=done, desc=f"encode[{store_name}]", unit="read") as bar:
        for start in range(0, len(sequences), batch_size):
            chunk = sequences[start : start + batch_size]
            chunk_ids = seq_ids[start : start + batch_size]
            try:
                vectors = encoder.encode_batch(chunk)
                if not np.isfinite(vectors).all():
                    raise ValueError("encoder produced non-finite values")
                store.append(list(chunk_ids), vectors)
                n_ok += len(chunk)
            except Exception as exc:  # a failed batch is recorded, never silently dropped
                n_failed += len(chunk)
                failures.append(
                    {"first_id": chunk_ids[0], "n": len(chunk), "error": repr(exc)[:300]}
                )
            cursor += len(chunk)
            bar.update(len(chunk))

    elapsed = timer.stop()
    manifest = store.finalize(
        {
            "encoder": encoder.describe(),
            "input_handling": encoder.input_handling(),
            "source_subset": str(Path(subset_file).name),
            "compatibility": compatibility,
        }
    )

    _, matrix, _ = _load_for_health(root, store_name)
    metrics: dict[str, Any] = {
        "stage": "encode",
        "run_label": run_label,
        "store_name": store_name,
        "encoder": encoder.describe(),
        "input_handling": encoder.input_handling(),
        "source_subset": str(Path(subset_file).name),
        "sequences_requested": total,
        "sequences_processed": done + cursor,
        "sequences_successful": done + n_ok,
        "sequences_failed": n_failed,
        "failures": failures[:20],
        "embedding_dimension": encoder.embedding_dim,
        "batch_size": batch_size,
        "processing_time_seconds": round(elapsed, 3),
        "sequences_per_second": round(n_ok / elapsed, 3) if elapsed else 0.0,
        "seconds_per_sequence": round(elapsed / n_ok, 8) if n_ok else None,
        "milliseconds_per_sequence": round(1000 * elapsed / n_ok, 5) if n_ok else None,
        "process_memory_mb": process_memory_mb(),
        "gpu_memory": gpu_memory_mb(),
        "embedding_storage_bytes": store.storage_bytes(),
        "embedding_storage_mb": round(store.storage_bytes() / 1024**2, 3),
        "bytes_per_embedding": round(store.storage_bytes() / max(manifest["total_rows"], 1), 2),
        "embedding_health": embedding_health(matrix),
        "compatibility_check": compatibility,
        "environment": environment_report(),
        "seed": cfg["seed"],
    }
    cfgutil.save_json(
        metrics, cfgutil.output_dir(cfg, "metrics") / f"encoder_metrics_{store_name}.json"
    )
    print(
        f"[encode] {store_name}: {metrics['sequences_successful']:,} ok, "
        f"{n_failed:,} failed, dim={encoder.embedding_dim}, "
        f"{metrics['sequences_per_second']:,.1f} seq/s"
    )
    return metrics


def _load_for_health(root: Path, name: str):
    from embeddings.store import load_embeddings

    return load_embeddings(root, name)
