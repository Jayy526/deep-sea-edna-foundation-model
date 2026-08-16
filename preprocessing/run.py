"""Stage drivers: full-dataset preprocessing, and subset construction.

``run_preprocess`` makes ONE streaming pass over the whole FASTA. Everything
downstream reads its Parquet output instead of re-parsing 650 MB.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing import fasta_parser, statistics as stats
from preprocessing.quality_control import QCConfig, describe_config, evaluate
from utils import config as cfgutil
from utils.runtime import Timer, environment_report, process_memory_mb

SUBSET_SCHEMA = pa.schema(
    [
        ("record_index", pa.int32()),
        ("seq_id", pa.string()),
        ("sequence", pa.string()),
        ("raw_length", pa.int16()),
        ("effective_length", pa.int16()),
        ("gc_content", pa.float32()),
    ]
)


def metadata_path(cfg: dict) -> Path:
    return cfgutil.output_dir(cfg, "metrics") / "read_metadata.parquet"


def subset_path(cfg: dict, name: str) -> Path:
    return cfgutil.output_dir(cfg, "embeddings") / f"subset_{name}.parquet"


def run_preprocess(cfg: dict, force: bool = False) -> dict[str, Any]:
    """Phase 1 + Phase 2: stream the FASTA, apply QC, emit metadata and metrics."""
    metrics_dir = cfgutil.output_dir(cfg, "metrics")
    meta_path = metadata_path(cfg)
    dataset_metrics_path = metrics_dir / "dataset_metrics.json"

    if meta_path.exists() and dataset_metrics_path.exists() and not force:
        print(f"[preprocess] Reusing existing {meta_path.name} (use --force to redo).")
        import json

        return json.loads(dataset_metrics_path.read_text(encoding="utf-8"))

    fasta = cfgutil.resolve_path(cfg["paths"]["fasta"])
    if not fasta.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta}")

    pre = cfg["preprocessing"]
    qc_cfg = QCConfig.from_dict(cfg["qc"])
    accumulator = stats.DatasetAccumulator(pre["max_position_tracked"])
    writer = stats.MetadataWriter(meta_path, pre["parquet_row_group"])

    file_size = fasta.stat().st_size
    print(f"[preprocess] Streaming {fasta.name} ({file_size / 1024**2:.1f} MB)")
    print(f"[preprocess] QC policy: {qc_cfg.ambiguity_policy}, "
          f"min_length={qc_cfg.min_length}, "
          f"min_effective_length={qc_cfg.min_effective_length}")

    # Baseline RSS before streaming starts. Absolute RSS is dominated by whichever
    # libraries the entry point happened to import (torch alone is several hundred
    # MB), so the meaningful figure for the constant-memory claim is the DELTA
    # attributable to streaming, not the absolute number.
    baseline_memory_mb = process_memory_mb()

    timer = Timer()
    progress_every = pre["progress_every"]
    last_report = time.perf_counter()

    for record in fasta_parser.iter_records(fasta, max_records=pre["max_records"]):
        qc = evaluate(record.sequence, qc_cfg, track_positions=True)
        header_fields = record.header_fields()
        accumulator.update(record, qc, header_fields, record.sequence.upper())
        writer.add(record, qc, header_fields)

        n = accumulator.n_records
        if progress_every and n % progress_every == 0:
            now = time.perf_counter()
            rate = progress_every / (now - last_report)
            last_report = now
            print(
                f"[preprocess] {n:,} records | {rate:,.0f} rec/s | "
                f"pass={accumulator.n_pass:,} | rss={process_memory_mb()} MB",
                flush=True,
            )

    writer.close()
    elapsed = timer.stop()
    # Memory at the END of streaming, before the exact-statistics pass loads
    # columns back in. This is the number that supports the constant-memory
    # claim; the whole-stage peak reported later is necessarily higher.
    streaming_memory_mb = process_memory_mb()
    streaming_delta_mb = (
        round(streaming_memory_mb - baseline_memory_mb, 2)
        if isinstance(streaming_memory_mb, float) and isinstance(baseline_memory_mb, float)
        else None
    )

    print(f"[preprocess] Done: {accumulator.n_records:,} records in {elapsed:,.1f}s "
          f"(streaming added {streaming_delta_mb} MB over a {baseline_memory_mb} MB baseline)")

    # --- Exact statistics from the metadata table --------------------------
    print("[preprocess] Computing exact dataset statistics...")
    table = pq.read_table(
        meta_path, columns=["gc_content", "raw_length", "effective_length", "n_ambiguous", "qc_pass"]
    )
    gc_all = table.column("gc_content").to_numpy(zero_copy_only=False)
    qc_mask = table.column("qc_pass").to_numpy(zero_copy_only=False)
    gc_pass = gc_all[qc_mask]
    n_amb = table.column("n_ambiguous").to_numpy(zero_copy_only=False)

    id_stats = stats.read_id_analysis(meta_path)

    dataset_metrics: dict[str, Any] = {
        "source_file": (
            fasta.relative_to(cfgutil.PROJECT_ROOT).as_posix()
            if fasta.is_relative_to(cfgutil.PROJECT_ROOT)
            else fasta.name
        ),
        "source_file_bytes": file_size,
        "source_file_mb": round(file_size / 1024**2, 2),
        **accumulator.to_dict(),
        "sequence_length": stats.length_stats_from_histogram(
            {int(k): v for k, v in accumulator.length_hist.items()}
        ),
        "effective_length_after_qc": stats.length_stats_from_histogram(
            {int(k): v for k, v in accumulator.effective_length_hist.items()}
        ),
        "gc_content_all_reads": stats.array_stats(gc_all),
        "gc_content_qc_passed_reads": stats.array_stats(gc_pass),
        "ambiguous_bases_per_read": stats.array_stats(n_amb),
        "read_id_analysis": id_stats,
        "position_profile": accumulator.position_arrays(),
    }

    preprocessing_metrics = {
        "stage": "preprocess",
        "scope": "FULL DATASET",
        "records_processed": accumulator.n_records,
        "elapsed_seconds": round(elapsed, 3),
        "records_per_second": round(accumulator.n_records / elapsed, 2) if elapsed else 0,
        "megabytes_per_second": round((file_size / 1024**2) / elapsed, 3) if elapsed else 0,
        "baseline_memory_mb_before_streaming": baseline_memory_mb,
        "streaming_phase_memory_mb": streaming_memory_mb,
        "streaming_phase_memory_delta_mb": streaming_delta_mb,
        "peak_process_memory_mb": process_memory_mb(),
        "memory_note": (
            "Absolute RSS is dominated by imported libraries (the CLI imports torch "
            "and transformers to record the environment), so the figure that supports "
            "the constant-memory claim is streaming_phase_memory_delta_mb: the growth "
            "attributable to streaming the whole FASTA. peak_process_memory_mb "
            "additionally covers the post-hoc exact-statistics pass, which "
            "deliberately reads metadata columns back in to compute exact order "
            "statistics rather than approximating them."
        ),
        "metadata_parquet_bytes": meta_path.stat().st_size,
        "metadata_parquet_mb": round(meta_path.stat().st_size / 1024**2, 2),
        "qc_configuration": describe_config(qc_cfg),
        "environment": environment_report(),
        "seed": cfg["seed"],
        "config_path": cfg.get("_config_path"),
    }

    cfgutil.save_json(dataset_metrics, dataset_metrics_path)
    cfgutil.save_json(preprocessing_metrics, metrics_dir / "preprocessing_metrics.json")
    print(f"[preprocess] Wrote {dataset_metrics_path.name}, preprocessing_metrics.json")
    return dataset_metrics


def build_subset(
    cfg: dict,
    name: str,
    size: int,
    force: bool = False,
) -> Path:
    """Materialise a QC-passed subset of reads for encoding.

    SAMPLING (IMPLEMENTATION DECISION): systematic sampling with a fixed stride
    over QC-passed reads in file order. Reads in this FASTA are ordered by
    flowcell coordinate, not by organism, so a fixed stride gives an unbiased
    spread across the whole run without holding 3M reads in memory. The stride
    is derived from the measured QC-pass count, and the seed is recorded.

    Both the baseline and the foundation-model encoder consume this same file,
    which is what makes the comparison in Phase 7 fair.
    """
    out_path = subset_path(cfg, name)
    if out_path.exists() and not force:
        n_existing = pq.read_metadata(out_path).num_rows
        print(f"[subset] Reusing {out_path.name} ({n_existing:,} reads).")
        return out_path

    import json

    dataset_metrics_path = cfgutil.output_dir(cfg, "metrics") / "dataset_metrics.json"
    if not dataset_metrics_path.exists():
        raise FileNotFoundError("Run the 'preprocess' stage before building a subset.")
    n_pass = json.loads(dataset_metrics_path.read_text(encoding="utf-8"))["reads_passing_qc"]

    stride = max(1, n_pass // size)
    print(f"[subset] Building '{name}': target={size:,} from {n_pass:,} QC-passed reads "
          f"(systematic stride={stride})")

    fasta = cfgutil.resolve_path(cfg["paths"]["fasta"])
    qc_cfg = QCConfig.from_dict(cfg["qc"])

    buffer: dict[str, list] = {n: [] for n in SUBSET_SCHEMA.names}
    kept = 0
    seen_pass = 0
    timer = Timer()

    for record in fasta_parser.iter_records(fasta):
        qc = evaluate(record.sequence, qc_cfg)
        if not qc.passed:
            continue
        if seen_pass % stride == 0 and kept < size:
            buffer["record_index"].append(record.index)
            buffer["seq_id"].append(record.seq_id)
            buffer["sequence"].append(qc.sequence)
            buffer["raw_length"].append(qc.raw_length)
            buffer["effective_length"].append(qc.effective_length)
            buffer["gc_content"].append(qc.gc_content)
            kept += 1
        seen_pass += 1
        if kept >= size:
            break

    pq.write_table(
        pa.Table.from_pydict(buffer, schema=SUBSET_SCHEMA), out_path, compression="zstd"
    )
    print(f"[subset] Wrote {out_path.name}: {kept:,} reads in {timer.stop():.1f}s")

    cfgutil.save_json(
        {
            "subset_name": name,
            "requested_size": size,
            "actual_size": kept,
            "sampling": "systematic (fixed stride over QC-passed reads in file order)",
            "stride": stride,
            "qc_passed_population": n_pass,
            "scope": "SUBSET (not the full dataset)",
            "elapsed_seconds": round(timer.elapsed, 2),
            "seed": cfg["seed"],
        },
        cfgutil.output_dir(cfg, "metrics") / f"subset_{name}_metrics.json",
    )
    return out_path


def build_pair_subset(cfg: dict, name: str, n_pairs: int, force: bool = False) -> Path:
    """Materialise complete paired-end MATE PAIRS (both R1 and R2 of one cluster).

    WHY THIS EXISTS (IMPLEMENTATION DECISION)
    -----------------------------------------
    This dataset has no taxonomic labels, so no supervised metric is available.
    But the paired-end layout supplies one piece of genuine, measured structure
    that was not invented by us: the two mates of a cluster are two
    non-overlapping reads sampled from the SAME source DNA fragment, and
    therefore from the same source organism.

    That gives a label-free retrieval benchmark: given the embedding of R1, can
    the embedding of its true mate R2 be found among distractors? It is a real
    test of whether a representation carries organism-level genomic signal, and
    it applies identically to the baseline and to the foundation model.

    It is NOT a taxonomic label and is never reported as one.

    Emitted rows are strictly ordered R1, R2, R1, R2, ... so pair ``i`` occupies
    rows ``2i`` and ``2i+1``.
    """
    out_path = subset_path(cfg, name)
    if out_path.exists() and not force:
        print(f"[subset] Reusing {out_path.name} ({pq.read_metadata(out_path).num_rows:,} reads).")
        return out_path

    fasta = cfgutil.resolve_path(cfg["paths"]["fasta"])
    qc_cfg = QCConfig.from_dict(cfg["qc"])

    import json

    metrics_file = cfgutil.output_dir(cfg, "metrics") / "dataset_metrics.json"
    n_ids = json.loads(metrics_file.read_text(encoding="utf-8"))["read_id_analysis"][
        "unique_sequence_ids"
    ]
    stride = max(1, n_ids // n_pairs)
    print(f"[subset] Building pair subset '{name}': target={n_pairs:,} pairs "
          f"from {n_ids:,} clusters (systematic stride={stride})")

    buffer: dict[str, list] = {n: [] for n in SUBSET_SCHEMA.names}
    kept = 0
    complete_seen = 0
    previous = None
    timer = Timer()

    for record in fasta_parser.iter_records(fasta):
        qc = evaluate(record.sequence, qc_cfg)
        current = (record, qc) if qc.passed else None
        if (
            previous is not None
            and current is not None
            and previous[0].seq_id == record.seq_id
        ):
            if complete_seen % stride == 0 and kept < n_pairs:
                for rec, res in (previous, current):
                    buffer["record_index"].append(rec.index)
                    buffer["seq_id"].append(rec.seq_id)
                    buffer["sequence"].append(res.sequence)
                    buffer["raw_length"].append(res.raw_length)
                    buffer["effective_length"].append(res.effective_length)
                    buffer["gc_content"].append(res.gc_content)
                kept += 1
            complete_seen += 1
            previous = None  # a mate is consumed exactly once
            if kept >= n_pairs:
                break
        else:
            previous = current

    pq.write_table(
        pa.Table.from_pydict(buffer, schema=SUBSET_SCHEMA), out_path, compression="zstd"
    )
    print(f"[subset] Wrote {out_path.name}: {kept:,} pairs ({kept * 2:,} reads) "
          f"in {timer.stop():.1f}s")

    cfgutil.save_json(
        {
            "subset_name": name,
            "layout": "row 2i and row 2i+1 are the two mates of pair i",
            "requested_pairs": n_pairs,
            "actual_pairs": kept,
            "reads": kept * 2,
            "sampling": "systematic (fixed stride over clusters whose BOTH mates passed QC)",
            "stride": stride,
            "cluster_population": n_ids,
            "scope": "SUBSET (not the full dataset)",
            "elapsed_seconds": round(timer.elapsed, 2),
            "seed": cfg["seed"],
        },
        cfgutil.output_dir(cfg, "metrics") / f"subset_{name}_metrics.json",
    )
    return out_path


def build_rc_pair_subset(cfg: dict, source: str = "pairs", name: str = "pairs_rc") -> Path:
    """Copy a pair subset with the SECOND mate reverse-complemented.

    WHY (IMPLEMENTATION DECISION)
    -----------------------------
    In paired-end sequencing the two mates are read from opposite strands of the
    same fragment. A representation that is reverse-complement invariant (the
    canonical k-mer baseline, by construction) is unaffected by this. A causal
    genomic language model is NOT invariant: GenomeOcean reads sequence in one
    direction and has no built-in notion that a strand and its complement are the
    same molecule.

    Reverse-complementing mate 2 places both mates on the same strand. Comparing
    retrieval with and without this transformation isolates how much of the
    foundation model's difficulty on paired reads is strand-orientation rather
    than a lack of fragment-level signal. Nothing is invented: reverse
    complementation is an exact, information-preserving operation on DNA.
    """
    out_path = subset_path(cfg, name)
    src_path = subset_path(cfg, source)
    table = pq.read_table(src_path)
    data = table.to_pydict()

    complement = str.maketrans("ACGT", "TGCA")
    data["sequence"] = [
        seq.translate(complement)[::-1] if i % 2 == 1 else seq
        for i, seq in enumerate(data["sequence"])
    ]
    pq.write_table(
        pa.Table.from_pydict(data, schema=SUBSET_SCHEMA), out_path, compression="zstd"
    )
    print(f"[subset] Wrote {out_path.name}: {len(data['sequence']):,} reads "
          f"(mate 2 reverse-complemented)")
    return out_path


def load_subset(path: str | Path) -> tuple[list[str], list[str]]:
    """Return (seq_ids, sequences) from a subset Parquet file."""
    table = pq.read_table(path, columns=["seq_id", "sequence"])
    return (
        table.column("seq_id").to_pylist(),
        table.column("sequence").to_pylist(),
    )
