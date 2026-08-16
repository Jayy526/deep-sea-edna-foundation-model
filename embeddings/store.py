"""Sharded embedding storage with checkpoint/resume.

Layout::

    outputs/embeddings/<name>/
        manifest.json          -- dim, dtype, shard list, total rows
        shard_00000.npy        -- float32 (rows, dim)
        shard_00000_ids.parquet

Only ``sequence_id`` and ``embedding`` are persisted. No intermediate tensors,
no hidden states, no token IDs are kept.

Resume works at shard granularity: an interrupted run restarts from the first
row not covered by a completed shard, so an expensive encoding job never has to
start over.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class EmbeddingStore:
    """Append-only sharded store for (sequence_id, embedding) pairs."""

    def __init__(self, root: str | Path, name: str, dim: int, shard_size: int = 25_000):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dim = dim
        self.shard_size = shard_size
        self.manifest_path = self.dir / "manifest.json"

        self._shards: list[dict] = []
        self._buffer_vecs: list[np.ndarray] = []
        self._buffer_ids: list[str] = []
        self._buffered = 0

        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("dim") == dim:
                self._shards = [
                    s for s in existing.get("shards", []) if (self.dir / s["file"]).exists()
                ]

    # --- Resume ------------------------------------------------------------

    @property
    def n_committed(self) -> int:
        """Rows already safely written to disk."""
        return sum(shard["rows"] for shard in self._shards)

    def is_complete(self, expected_rows: int) -> bool:
        return self.n_committed >= expected_rows

    # --- Writing -----------------------------------------------------------

    def append(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected embedding dim {self.dim}, got {vectors.shape[1]}"
            )
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        self._buffer_ids.extend(ids)
        self._buffer_vecs.append(np.asarray(vectors, dtype=np.float32))
        self._buffered += len(ids)
        while self._buffered >= self.shard_size:
            self._write_shard(self.shard_size)

    def _write_shard(self, rows: int) -> None:
        block = np.concatenate(self._buffer_vecs, axis=0)
        out, rest = block[:rows], block[rows:]
        ids, id_rest = self._buffer_ids[:rows], self._buffer_ids[rows:]

        index = len(self._shards)
        vec_file = f"shard_{index:05d}.npy"
        id_file = f"shard_{index:05d}_ids.parquet"
        np.save(self.dir / vec_file, out)
        pq.write_table(
            pa.table({"sequence_id": pa.array(ids, pa.string())}),
            self.dir / id_file,
            compression="zstd",
        )
        self._shards.append({"file": vec_file, "ids_file": id_file, "rows": int(out.shape[0])})
        self._write_manifest()

        self._buffer_vecs = [rest] if rest.size else []
        self._buffer_ids = id_rest
        self._buffered = len(id_rest)

    def flush(self) -> None:
        if self._buffered:
            self._write_shard(self._buffered)

    def _write_manifest(self, extra: dict | None = None) -> None:
        manifest: dict[str, Any] = {
            "name": self.name,
            "dim": self.dim,
            "dtype": "float32",
            "shard_size": self.shard_size,
            "shards": self._shards,
            "total_rows": self.n_committed,
        }
        if extra:
            manifest.update(extra)
        elif self.manifest_path.exists():
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for key in ("encoder", "input_handling", "source_subset", "compatibility"):
                if key in previous:
                    manifest[key] = previous[key]
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def finalize(self, metadata: dict | None = None) -> dict:
        self.flush()
        self._write_manifest(metadata)
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    # --- Reading -----------------------------------------------------------

    def storage_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.dir.glob("*") if p.is_file())


def load_embeddings(
    root: str | Path, name: str, limit: int | None = None
) -> tuple[list[str], np.ndarray, dict]:
    """Load a stored embedding matrix and its sequence IDs."""
    directory = Path(root) / name
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    vectors: list[np.ndarray] = []
    ids: list[str] = []
    for shard in manifest["shards"]:
        vectors.append(np.load(directory / shard["file"]))
        ids.extend(
            pq.read_table(directory / shard["ids_file"]).column("sequence_id").to_pylist()
        )
        if limit is not None and sum(v.shape[0] for v in vectors) >= limit:
            break
    matrix = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, manifest["dim"]))
    if limit is not None:
        matrix = matrix[:limit]
        ids = ids[:limit]
    return ids, matrix, manifest


def embedding_health(matrix: np.ndarray) -> dict[str, Any]:
    """Verify embeddings are finite and non-degenerate. Measured, not assumed."""
    finite = np.isfinite(matrix)
    per_dim_std = matrix.std(axis=0) if matrix.size else np.array([])
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "all_finite": bool(finite.all()),
        "n_nan": int(np.isnan(matrix).sum()),
        "n_inf": int(np.isinf(matrix).sum()),
        "value_mean": round(float(matrix.mean()), 6) if matrix.size else None,
        "value_std": round(float(matrix.std()), 6) if matrix.size else None,
        "value_min": round(float(matrix.min()), 6) if matrix.size else None,
        "value_max": round(float(matrix.max()), 6) if matrix.size else None,
        "dead_dimensions": int((per_dim_std == 0).sum()) if per_dim_std.size else 0,
        "l2_norm_mean": round(float(np.linalg.norm(matrix, axis=1).mean()), 6)
        if matrix.size
        else None,
    }
