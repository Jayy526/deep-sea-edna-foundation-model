"""Stage 2 -- full-dataset dereplication into unique sequence variants.

WHY THIS IS THE RIGHT FIRST STEP (IMPLEMENTATION DECISION)
----------------------------------------------------------
Stage 1 encoded reads sampled at random. For community structure that is the
wrong unit: 3M reads are not 3M organisms, they are a re-sampling of a much
smaller set of distinct molecules at wildly uneven depth. The natural unit is
the **unique sequence variant** -- an ASV-like entity: one exact sequence,
carrying the number of reads that supported it.

Dereplicating first means:
  * every variant is counted once, so abundance becomes an explicit weight
    rather than an uncontrolled bias in the sample,
  * the foundation model is run over distinct molecules instead of thousands of
    identical copies, which is both cheaper and more informative,
  * diversity statistics become computable at all.

WHAT THIS IS NOT
----------------
These are exact-sequence variants, not denoised ASVs (no DADA2/UNOISE error
model) and not OTUs (no similarity clustering). Sequencing error therefore
inflates the variant count, predominantly as singletons. That is measured and
reported, never corrected for silently. They carry NO taxonomic identity.

MEMORY (IMPLEMENTATION DECISION)
--------------------------------
Holding ~3M sequence strings would cost several hundred MB. Instead pass 1
counts 8-byte BLAKE2b digests in a dict[int, int]; pass 2 recovers the actual
sequence only for variants that will be used downstream. For n distinct
variants the 64-bit collision probability is ~n^2 / 2^65; at n = 3e6 that is
about 2.4e-7, which is recorded in the output rather than assumed away.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing import fasta_parser
from preprocessing.quality_control import QCConfig, evaluate
from utils import config as cfgutil
from utils.runtime import Timer, process_memory_mb

COMPLEMENT = str.maketrans("ACGT", "TGCA")

VARIANT_SCHEMA = pa.schema(
    [
        ("variant_id", pa.int32()),
        ("sequence", pa.string()),
        ("count", pa.int32()),
        ("length", pa.int16()),
        ("gc_content", pa.float32()),
        ("rank", pa.int32()),
    ]
)


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _digest(sequence: str) -> int:
    return int.from_bytes(hashlib.blake2b(sequence.encode(), digest_size=8).digest(), "big")


def variants_path(cfg: dict, canonical: bool | None = None) -> Path:
    """Output path. The canonical (RC-collapsed) run writes to its own files so
    the sensitivity check never overwrites the primary result."""
    if canonical is None:
        canonical = cfg["dereplication"]["canonical"]
    name = "variants_canonical.parquet" if canonical else "variants.parquet"
    return cfgutil.output_dir(cfg, "embeddings") / name


def dereplicate(cfg: dict, force: bool = False, canonical: bool | None = None) -> dict[str, Any]:
    """Two-pass streaming dereplication over the FULL QC-passed dataset."""
    metrics_dir = cfgutil.output_dir(cfg, "metrics")
    if canonical is None:
        canonical = cfg["dereplication"]["canonical"]
    suffix = "_canonical" if canonical else ""
    out_metrics = metrics_dir / f"dereplication_metrics{suffix}.json"
    out_variants = variants_path(cfg, canonical)

    if out_variants.exists() and out_metrics.exists() and not force:
        import json

        print(f"[derep] Reusing {out_variants.name} "
              f"({pq.read_metadata(out_variants).num_rows:,} variants).")
        return json.loads(out_metrics.read_text(encoding="utf-8"))

    fasta = cfgutil.resolve_path(cfg["paths"]["fasta"])
    qc_cfg = QCConfig.from_dict(cfg["qc"])
    derep_cfg = cfg["dereplication"]
    keep_top = derep_cfg["keep_top_variants"]
    min_count = derep_cfg["min_count_for_representative"]

    # ---------------- Pass 1: count digests ------------------------------
    print(f"[derep] Pass 1/2: counting unique variants over the full dataset "
          f"(canonical={canonical})")
    counts: dict[int, int] = {}
    n_reads = 0
    timer = Timer()

    for record in fasta_parser.iter_records(fasta):
        qc = evaluate(record.sequence, qc_cfg)
        if not qc.passed:
            continue
        seq = qc.sequence
        if canonical:
            rc = reverse_complement(seq)
            if rc < seq:
                seq = rc
        key = _digest(seq)
        counts[key] = counts.get(key, 0) + 1
        n_reads += 1
        if n_reads % 500_000 == 0:
            print(f"[derep]   {n_reads:,} reads -> {len(counts):,} variants "
                  f"| rss={process_memory_mb()} MB", flush=True)

    pass1 = timer.stop()
    n_variants = len(counts)
    print(f"[derep] Pass 1 done: {n_reads:,} reads -> {n_variants:,} variants "
          f"in {pass1:,.1f}s")

    # Frequency spectrum: how many variants were seen exactly k times.
    # This is all any diversity index needs, and it is tiny.
    spectrum = Counter(counts.values())

    # Which variants do we need actual sequence for?
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    wanted_pairs = [
        (key, count) for key, count in ranked[:keep_top] if count >= min_count
    ]
    wanted = {key: (rank, count) for rank, (key, count) in enumerate(wanted_pairs)}
    del ranked
    print(f"[derep] Pass 2/2: recovering sequences for {len(wanted):,} variants "
          f"(top {keep_top:,}, min_count {min_count})")

    # ---------------- Pass 2: recover representative sequences ------------
    timer2 = Timer()
    found: dict[int, str] = {}
    for record in fasta_parser.iter_records(fasta):
        if len(found) == len(wanted):
            break
        qc = evaluate(record.sequence, qc_cfg)
        if not qc.passed:
            continue
        seq = qc.sequence
        if canonical:
            rc = reverse_complement(seq)
            if rc < seq:
                seq = rc
        key = _digest(seq)
        if key in wanted and key not in found:
            found[key] = seq
    pass2 = timer2.stop()

    rows = {name: [] for name in VARIANT_SCHEMA.names}
    for key, (rank, count) in sorted(wanted.items(), key=lambda kv: kv[1][0]):
        seq = found.get(key)
        if seq is None:
            continue
        gc = seq.count("G") + seq.count("C")
        at = seq.count("A") + seq.count("T")
        rows["variant_id"].append(rank)
        rows["sequence"].append(seq)
        rows["count"].append(count)
        rows["length"].append(len(seq))
        rows["gc_content"].append(gc / (gc + at) if (gc + at) else 0.0)
        rows["rank"].append(rank + 1)

    pq.write_table(
        pa.Table.from_pydict(rows, schema=VARIANT_SCHEMA), out_variants, compression="zstd"
    )
    print(f"[derep] Wrote {out_variants.name}: {len(rows['variant_id']):,} variants "
          f"({pass2:,.1f}s)")

    covered = sum(rows["count"])
    metrics = {
        "stage": "dereplicate",
        "scope": "FULL DATASET",
        "canonical_reverse_complement_collapse": canonical,
        "reads_dereplicated": n_reads,
        "unique_variants": n_variants,
        "reads_per_variant_mean": round(n_reads / n_variants, 4) if n_variants else 0,
        "variants_written": len(rows["variant_id"]),
        "reads_covered_by_written_variants": covered,
        "fraction_of_reads_covered_by_written_variants": round(covered / n_reads, 6)
        if n_reads
        else 0,
        "singletons": spectrum.get(1, 0),
        "singleton_fraction_of_variants": round(spectrum.get(1, 0) / n_variants, 6)
        if n_variants
        else 0,
        "doubletons": spectrum.get(2, 0),
        "max_variant_count": max(spectrum) if spectrum else 0,
        "frequency_spectrum": {str(k): v for k, v in sorted(spectrum.items())},
        "pass1_seconds": round(pass1, 2),
        "pass2_seconds": round(pass2, 2),
        "peak_process_memory_mb": process_memory_mb(),
        "hash": {
            "algorithm": "BLAKE2b, 8-byte digest",
            "collision_probability_estimate": round(
                (n_variants**2) / 2**65, 12
            ),
            "note": "Counting is on digests, not sequences, to bound memory.",
        },
        "caveats": [
            "Exact-sequence variants: no denoising error model, no similarity clustering.",
            "Sequencing error inflates the variant count, mostly as singletons.",
            "Variants carry no taxonomic identity.",
        ],
        "seed": cfg["seed"],
    }
    cfgutil.save_json(metrics, out_metrics)
    return metrics


def load_variants(cfg: dict, limit: int | None = None):
    """Return (ids, sequences, counts) for the stored variants."""
    table = pq.read_table(
        variants_path(cfg), columns=["variant_id", "sequence", "count"]
    )
    ids = table.column("variant_id").to_pylist()
    seqs = table.column("sequence").to_pylist()
    counts = table.column("count").to_pylist()
    if limit:
        ids, seqs, counts = ids[:limit], seqs[:limit], counts[:limit]
    return ids, seqs, counts
