"""Build a region-matched taxonomic reference by in-silico PCR.

WHY IN-SILICO PCR
-----------------
Our reads cover one short, specific window of the 18S gene: the 37F
hypervariable region, amplified with the Foraminifera-specific primer s14F1
(``AAGGGCACCACAAGAACGC``). Comparing a 151 bp read against full-length 1,800 bp
reference sequences would mostly compare it against sequence it does not
overlap. So the reference is first cut down to the same window: find the primer
site in each reference and keep the read-length stretch that follows it.

That makes reference and query directly comparable, and shrinks the reference
enough that an exact classifier is cheap.

WHY PR2 AND NOT SILVA
---------------------
Measured, not assumed. SILVA 138.2 SSU NR99 (510,495 sequences) contains only
**69 Foraminifera**, of which 67 carry the primer site -- far too few to train
a classifier. PR2 5.0.0 (221,085 sequences) contains **1,547** sequences
carrying the primer site, and every one of them is a foraminiferan, spanning
157 genus-level lineages. That is the well-documented foraminiferal reference
gap in general-purpose databases, and it is why a protist-curated database is
the right choice here.

The 1,547-sequence reference is still small in absolute terms. Labels derived
from it will be incomplete and noisy. That is stated as a limitation and is
precisely the regime TaxDistill was designed for.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from utils import config as cfgutil

# 16-mer core of s14F1. Our reads begin two bases upstream of it, so the
# reference window is cut from (core position - PRIMER_OFFSET).
S14F1_CORE = "GGGCACCACAAGAACG"
PRIMER_OFFSET = 2

RANKS = [
    "domain", "supergroup", "division", "subdivision",
    "class", "order", "family", "genus", "species",
]

REFERENCE_SCHEMA = pa.schema(
    [("accession", pa.string()), ("sequence", pa.string()), ("taxonomy", pa.string())]
    + [(rank, pa.string()) for rank in RANKS]
)


def _iter_fasta_gz(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def load_taxonomy(path: Path) -> dict[str, str]:
    tax: dict[str, str] = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            accession, _, lineage = line.partition("\t")
            tax[accession.strip()] = lineage.strip().rstrip(";")
    return tax


def build_reference(cfg: dict, force: bool = False) -> dict[str, Any]:
    """Extract the amplicon window from every reference carrying the primer site."""
    tcfg = cfg["taxonomy"]
    out_path = cfgutil.resolve_path(tcfg["reference_parquet"])
    fasta = cfgutil.resolve_path(tcfg["pr2_fasta"])
    taxfile = cfgutil.resolve_path(tcfg["pr2_tax"])
    read_length = tcfg["region_length"]

    if out_path.exists() and not force:
        n = pq.read_metadata(out_path).num_rows
        print(f"[reference] Reusing {out_path.name} ({n:,} region-matched references).")
        import json

        meta_path = cfgutil.output_dir(cfg, "metrics") / "reference_metrics.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))

    taxonomy = load_taxonomy(taxfile)
    rows: dict[str, list] = {name: [] for name in REFERENCE_SCHEMA.names}
    scanned = with_primer = kept = 0
    truncated = 0

    for header, sequence in _iter_fasta_gz(fasta):
        scanned += 1
        sequence = sequence.upper().replace("U", "T")
        position = sequence.find(S14F1_CORE)
        if position < 0:
            continue
        with_primer += 1
        start = max(0, position - PRIMER_OFFSET)
        region = sequence[start : start + read_length]
        if len(region) < tcfg["min_region_length"]:
            truncated += 1
            continue
        lineage = taxonomy.get(header, "")
        if not lineage:
            continue
        parts = [p for p in lineage.split(";") if p]
        parts += [""] * (len(RANKS) - len(parts))

        rows["accession"].append(header)
        rows["sequence"].append(region)
        rows["taxonomy"].append(lineage)
        for rank, value in zip(RANKS, parts):
            rows[rank].append(value)
        kept += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(rows, schema=REFERENCE_SCHEMA), out_path, compression="zstd"
    )

    from collections import Counter

    metrics: dict[str, Any] = {
        "stage": "reference",
        "source_database": "PR2 5.0.0 SSU (mothur release)",
        "why_not_silva": (
            "SILVA 138.2 SSU NR99 was downloaded and measured first: of 510,495 "
            "sequences it contains only 69 Foraminifera (67 carrying the primer "
            "site), too few to train a classifier. PR2 carries 1,547."
        ),
        "primer": {
            "name": "s14F1 (Foraminifera-specific, 18S 37F hypervariable region)",
            "core_used": S14F1_CORE,
            "matching": "exact match of the 16-mer core",
        },
        "sequences_scanned": scanned,
        "sequences_with_primer_site": with_primer,
        "references_kept": kept,
        "dropped_region_too_short": truncated,
        "region_length": read_length,
        "lineages": {
            rank: len(set(rows[rank])) for rank in ("class", "order", "family", "genus")
        },
        "class_breakdown": dict(Counter(rows["class"]).most_common()),
        "limitation": (
            "1,547 references is a small database. Coverage of deep-sea benthic "
            "Foraminifera is known to be poor, so a substantial share of query "
            "variants is expected to be unassignable. Labels derived here are "
            "incomplete and noisy by construction."
        ),
    }
    cfgutil.save_json(metrics, cfgutil.output_dir(cfg, "metrics") / "reference_metrics.json")
    print(f"[reference] {kept:,} region-matched references "
          f"({metrics['lineages']['genus']} genera, "
          f"{metrics['lineages']['family']} families) -> {out_path.name}")
    return metrics


def load_reference(cfg: dict, rank: str) -> tuple[list[str], list[str]]:
    """Return (sequences, labels) at the requested taxonomic rank."""
    path = cfgutil.resolve_path(cfg["taxonomy"]["reference_parquet"])
    table = pq.read_table(path, columns=["sequence", rank])
    return table.column("sequence").to_pylist(), table.column(rank).to_pylist()
