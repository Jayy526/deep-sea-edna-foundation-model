"""Streaming dataset statistics and per-read metadata storage.

Two things happen in a single pass over the FASTA:

1. Streaming aggregates that would be expensive to recompute (nucleotide
   composition, per-position ambiguity, length histogram, QC reason tally).
2. A per-read metadata table written to Parquet in row groups. Exact
   order statistics (median, percentiles) and read-ID duplication analysis are
   computed afterwards from that table rather than approximated on the fly.

Nothing is sampled here: these are full-dataset numbers.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing.quality_control import REASON_CODES, QCConfig, QCResult

METADATA_SCHEMA = pa.schema(
    [
        ("record_index", pa.int32()),
        ("seq_id", pa.string()),
        ("raw_length", pa.int16()),
        ("effective_length", pa.int16()),
        ("gc_content", pa.float32()),
        ("n_ambiguous", pa.int16()),
        ("n_invalid", pa.int16()),
        ("longest_unambiguous_run", pa.int16()),
        ("max_base_fraction", pa.float32()),
        ("qc_pass", pa.bool_()),
        ("qc_reason", pa.int8()),
        ("lane", pa.int16()),
        ("tile", pa.int32()),
        ("declared_length", pa.int16()),
    ]
)


class MetadataWriter:
    """Buffered Parquet writer for per-read metadata."""

    def __init__(self, path: str | Path, row_group: int = 250_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.row_group = row_group
        self._writer = pq.ParquetWriter(self.path, METADATA_SCHEMA, compression="zstd")
        self._buffer: dict[str, list] = {name: [] for name in METADATA_SCHEMA.names}
        self._n = 0

    def add(self, record, qc: QCResult, header_fields: dict) -> None:
        buf = self._buffer
        buf["record_index"].append(record.index)
        buf["seq_id"].append(record.seq_id)
        buf["raw_length"].append(qc.raw_length)
        buf["effective_length"].append(qc.effective_length)
        buf["gc_content"].append(qc.gc_content)
        buf["n_ambiguous"].append(qc.n_ambiguous)
        buf["n_invalid"].append(qc.n_invalid)
        buf["longest_unambiguous_run"].append(qc.longest_unambiguous_run)
        buf["max_base_fraction"].append(qc.max_base_fraction)
        buf["qc_pass"].append(qc.passed)
        buf["qc_reason"].append(qc.reason)
        buf["lane"].append(header_fields["lane"])
        buf["tile"].append(header_fields["tile"])
        buf["declared_length"].append(header_fields["declared_length"])
        self._n += 1
        if self._n % self.row_group == 0:
            self.flush()

    def flush(self) -> None:
        if not self._buffer["record_index"]:
            return
        table = pa.Table.from_pydict(self._buffer, schema=METADATA_SCHEMA)
        self._writer.write_table(table)
        for values in self._buffer.values():
            values.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


class DatasetAccumulator:
    """Streaming aggregates over the full dataset."""

    def __init__(self, max_position_tracked: int = 512):
        self.n_records = 0
        self.total_bp = 0
        self.total_effective_bp = 0
        self.base_counts: Counter[str] = Counter()
        self.length_hist: Counter[int] = Counter()
        self.effective_length_hist: Counter[int] = Counter()
        self.qc_reason_hist: Counter[int] = Counter()
        self.n_pass = 0
        self.max_position_tracked = max_position_tracked
        self.ambiguous_per_position = np.zeros(max_position_tracked, dtype=np.int64)
        self.coverage_per_position = np.zeros(max_position_tracked, dtype=np.int64)
        self.reads_with_ambiguity = 0
        self.reads_with_invalid = 0
        self.total_ambiguous_bases = 0
        self.total_invalid_bases = 0
        self.header_examples: list[str] = []
        self.declared_length_mismatch = 0

    def update(self, record, qc: QCResult, header_fields: dict, sequence: str) -> None:
        self.n_records += 1
        self.total_bp += qc.raw_length
        self.length_hist[qc.raw_length] += 1
        self.qc_reason_hist[qc.reason] += 1

        if qc.passed:
            self.n_pass += 1
            self.total_effective_bp += qc.effective_length
            self.effective_length_hist[qc.effective_length] += 1

        for base in set(sequence):
            self.base_counts[base] += sequence.count(base)

        if qc.n_ambiguous:
            self.reads_with_ambiguity += 1
            self.total_ambiguous_bases += qc.n_ambiguous
        if qc.n_invalid:
            self.reads_with_invalid += 1
            self.total_invalid_bases += qc.n_invalid

        limit = min(qc.raw_length, self.max_position_tracked)
        self.coverage_per_position[:limit] += 1
        for pos in qc.ambiguous_positions:
            if pos < self.max_position_tracked:
                self.ambiguous_per_position[pos] += 1

        declared = header_fields.get("declared_length")
        if declared is not None and declared != qc.raw_length:
            self.declared_length_mismatch += 1

        if len(self.header_examples) < 10:
            self.header_examples.append(record.header)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_sequence_records": self.n_records,
            "total_base_pairs": self.total_bp,
            "total_base_pairs_after_qc": self.total_effective_bp,
            "reads_passing_qc": self.n_pass,
            "reads_removed_by_qc": self.n_records - self.n_pass,
            "qc_pass_rate": round(self.n_pass / self.n_records, 6)
            if self.n_records
            else 0.0,
            "nucleotide_counts": dict(self.base_counts.most_common()),
            "nucleotide_fractions": {
                base: round(count / self.total_bp, 8)
                for base, count in self.base_counts.most_common()
            }
            if self.total_bp
            else {},
            "reads_with_ambiguous_bases": self.reads_with_ambiguity,
            "total_ambiguous_bases": self.total_ambiguous_bases,
            "reads_with_invalid_characters": self.reads_with_invalid,
            "total_invalid_characters": self.total_invalid_bases,
            "qc_reason_counts": {
                REASON_CODES[code]: count
                for code, count in sorted(self.qc_reason_hist.items())
            },
            "length_histogram": {
                str(k): v for k, v in sorted(self.length_hist.items())
            },
            "effective_length_histogram": {
                str(k): v for k, v in sorted(self.effective_length_hist.items())
            },
            "header_examples": self.header_examples,
            "headers_with_declared_length_mismatch": self.declared_length_mismatch,
        }

    def position_arrays(self) -> dict[str, list[int]]:
        used = int(np.max(np.nonzero(self.coverage_per_position)[0]) + 1) if self.coverage_per_position.any() else 0
        return {
            "ambiguous_per_position": self.ambiguous_per_position[:used].tolist(),
            "coverage_per_position": self.coverage_per_position[:used].tolist(),
        }


def length_stats_from_histogram(hist: dict[int, int]) -> dict[str, float]:
    """Exact min/max/mean/median/percentiles from a length histogram."""
    if not hist:
        return {}
    lengths = np.array(sorted(hist), dtype=np.int64)
    counts = np.array([hist[int(length)] for length in lengths], dtype=np.int64)
    total = int(counts.sum())
    cumulative = np.cumsum(counts)

    def percentile(p: float) -> int:
        target = p * total
        idx = int(np.searchsorted(cumulative, target, side="left"))
        return int(lengths[min(idx, len(lengths) - 1)])

    mean = float((lengths * counts).sum() / total)
    variance = float((counts * (lengths - mean) ** 2).sum() / total)
    return {
        "min": int(lengths[0]),
        "max": int(lengths[-1]),
        "mean": round(mean, 4),
        "median": percentile(0.5),
        "std": round(variance**0.5, 4),
        "p01": percentile(0.01),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p99": percentile(0.99),
        "distinct_values": len(lengths),
        "n": total,
    }


def array_stats(values: np.ndarray) -> dict[str, float]:
    """Descriptive statistics for a numeric column."""
    if values.size == 0:
        return {}
    values = values.astype(np.float64)
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "std": round(float(values.std()), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "p05": round(float(np.percentile(values, 5)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
    }


def read_id_analysis(metadata_path: str | Path) -> dict[str, Any]:
    """Determine empirically WHY sequence IDs occur more than once.

    This does not assume paired-end reads. It measures:
      - the multiplicity distribution of IDs,
      - whether repeated IDs are adjacent in file order (interleaved layout),
      - whether repeated IDs carry identical or different sequences
        (via their independently-measured GC content and length),
      - whether repeated IDs share the same flowcell coordinates.

    Identical flowcell coordinates + adjacent records + differing sequence
    content is the signature of the two mates of one paired-end cluster.
    """
    import pandas as pd

    columns = ["record_index", "seq_id", "gc_content", "raw_length", "lane", "tile"]
    frame = pq.read_table(metadata_path, columns=columns).to_pandas()

    counts = frame["seq_id"].value_counts()
    multiplicity = counts.value_counts().sort_index()

    frame = frame.sort_values("record_index", kind="stable")
    frame["occurrence"] = frame.groupby("seq_id").cumcount()

    duplicated = counts[counts > 1].index
    dup_frame = frame[frame["seq_id"].isin(pd.Index(duplicated))]

    # Are the repeats adjacent in the file?
    adjacency = (
        dup_frame.groupby("seq_id")["record_index"]
        .apply(lambda s: bool(np.all(np.diff(np.sort(s.values)) == 1)))
        .mean()
        if len(dup_frame)
        else float("nan")
    )

    # Do repeated IDs carry the same sequence content?
    pair = dup_frame[dup_frame["occurrence"] < 2].pivot_table(
        index="seq_id", columns="occurrence", values="gc_content"
    )
    identical_gc = float((pair[0] == pair[1]).mean()) if pair.shape[1] == 2 else float("nan")

    coord = dup_frame[dup_frame["occurrence"] < 2].pivot_table(
        index="seq_id", columns="occurrence", values="tile"
    )
    same_tile = float((coord[0] == coord[1]).mean()) if coord.shape[1] == 2 else float("nan")

    return {
        "total_records": int(len(frame)),
        "unique_sequence_ids": int(counts.size),
        "ids_occurring_once": int((counts == 1).sum()),
        "ids_occurring_more_than_once": int((counts > 1).sum()),
        "max_multiplicity": int(counts.max()),
        "multiplicity_distribution": {
            f"{int(k)}_occurrences": int(v) for k, v in multiplicity.items()
        },
        "records_in_duplicated_ids": int(len(dup_frame)),
        "fraction_of_repeats_adjacent_in_file": round(float(adjacency), 6),
        "fraction_of_repeats_with_identical_gc": round(identical_gc, 6),
        "fraction_of_repeats_on_same_tile": round(same_tile, 6),
        "interpretation_basis": (
            "Repeated IDs are classified as paired-end mates only if they are "
            "adjacent in the file, sit on the same flowcell tile, and carry "
            "different sequence content. All three are measured above."
        ),
    }
