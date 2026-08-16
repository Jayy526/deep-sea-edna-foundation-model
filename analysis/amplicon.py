"""Amplicon architecture: how much of each read is actually informative?

MOTIVATION
----------
Stage 1 found that this library is amplicon (marker-gene) data rather than
shotgun: ~90% of reads begin with one of two conserved sequences. That raised a
question neither stage answered: **how much of a 151 bp read is conserved
anchor, and how much is organism-discriminating variable sequence?**

It matters for interpreting every earlier result. If a large fraction of each
read is near-invariant across the whole community, then both the foundation
model and the k-mer baseline are being handed reads whose majority content
carries no discriminating signal — which would bound how well *any*
representation can do at mate retrieval or fine-grained clustering.

METHOD
------
For each amplicon end, per-position base composition is computed across the
abundant variants, **weighted by read abundance** (a variant backed by 400,000
reads should dominate the consensus). Conservation at a position is the
frequency of its most common base: 1.0 means invariant across the community,
0.25 means uniform.

The conserved/variable boundary is found by scanning for the position that
maximises the difference in mean conservation before and after it — a simple
one-dimensional changepoint, reported with the actual values so the reader can
judge whether the split is real.

WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT
-------------------------------------------
Established from the data alone: the conservation architecture, the size of the
conserved block, and the number of genuinely variable positions.

NOT established here: which marker gene this is (16S vs 18S vs organellar SSU),
or any taxonomic identity. Those require a reference database. The conserved
motif's identity is discussed in the report with appropriate hedging; nothing in
this module assigns a gene name.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from analysis.dereplicate import variants_path
from evaluation.structure import amplicon_end_groups
from utils import config as cfgutil

BASES = "ACGT"


def _weighted_profile(
    sequences: list[str], weights: list[int], length: int
) -> tuple[np.ndarray, str, np.ndarray]:
    """Abundance-weighted per-position base frequencies.

    Returns (conservation, consensus, base_frequency_matrix).
    """
    freq = np.zeros((length, len(BASES)), dtype=np.float64)
    for seq, weight in zip(sequences, weights):
        for pos in range(min(length, len(seq))):
            slot = BASES.find(seq[pos])
            if slot >= 0:
                freq[pos, slot] += weight
    totals = freq.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    freq = freq / totals
    conservation = freq.max(axis=1)
    consensus = "".join(BASES[i] for i in freq.argmax(axis=1))
    return conservation, consensus, freq


def _changepoint(conservation: np.ndarray, min_block: int = 15) -> dict[str, Any]:
    """Position maximising the conservation drop between the two halves."""
    best = {"position": None, "before": None, "after": None, "delta": -1.0}
    for cut in range(min_block, len(conservation) - min_block):
        before = float(conservation[:cut].mean())
        after = float(conservation[cut:].mean())
        delta = before - after
        if delta > best["delta"]:
            best = {
                "position": cut,
                "before": round(before, 5),
                "after": round(after, 5),
                "delta": round(delta, 5),
            }
    return best


def characterize(cfg: dict, max_variants: int = 5000) -> dict[str, Any]:
    """Per-amplicon-end conservation architecture over the abundant variants."""
    table = pq.read_table(variants_path(cfg), columns=["sequence", "count"])
    sequences = table.column("sequence").to_pylist()
    counts = table.column("count").to_pylist()

    groups, group_meta = amplicon_end_groups(sequences)
    read_length = Counter(len(s) for s in sequences).most_common(1)[0][0]

    result: dict[str, Any] = {
        "stage": "amplicon_architecture",
        "unit": "unique sequence variant, weighted by read abundance",
        "read_length": read_length,
        "amplicon_end_structure": group_meta,
        "ends": {},
    }

    for end in sorted({int(g) for g in groups if g >= 0}):
        idx = [
            i for i in range(len(sequences))
            if groups[i] == end and len(sequences[i]) == read_length
        ]
        idx.sort(key=lambda i: -counts[i])
        idx = idx[:max_variants]
        if not idx:
            continue

        sub = [sequences[i] for i in idx]
        weights = [counts[i] for i in idx]
        conservation, consensus, _ = _weighted_profile(sub, weights, read_length)
        cut = _changepoint(conservation)

        variable = int((conservation < 0.70).sum())
        invariant = int((conservation > 0.99).sum())
        result["ends"][str(end)] = {
            "variants_used": len(sub),
            "reads_represented": int(sum(weights)),
            "consensus": consensus,
            "mean_conservation": round(float(conservation.mean()), 5),
            "positions_invariant_gt99pct": invariant,
            "positions_variable_lt70pct": variable,
            "informative_fraction_of_read": round(variable / read_length, 5),
            "conserved_block": cut,
            "conserved_block_length": cut["position"],
            "variable_region_length": read_length - cut["position"]
            if cut["position"]
            else None,
            "conservation_profile": [round(float(v), 5) for v in conservation],
            "conserved_block_consensus": consensus[: cut["position"]]
            if cut["position"]
            else None,
        }

    # Community-level summary across both ends, weighted by reads.
    ends = result["ends"].values()
    total_reads = sum(e["reads_represented"] for e in ends)
    if total_reads:
        result["summary"] = {
            "weighted_mean_conservation": round(
                sum(e["mean_conservation"] * e["reads_represented"] for e in ends)
                / total_reads,
                5,
            ),
            "weighted_informative_fraction": round(
                sum(e["informative_fraction_of_read"] * e["reads_represented"] for e in ends)
                / total_reads,
                5,
            ),
            "interpretation": (
                "The informative fraction is the share of positions that actually vary "
                "across the community. Positions outside it are near-identical in every "
                "read and therefore carry no organism-discriminating signal, no matter "
                "which representation encodes them."
            ),
        }

    result["limits"] = {
        "established_from_data": [
            "the conservation architecture of each amplicon end",
            "the size of the conserved block and the number of variable positions",
            "the abundance-weighted consensus sequence",
        ],
        "not_established_here": [
            "which marker gene this is (16S vs 18S vs organellar SSU)",
            "any taxonomic identity",
            "these require a reference database, which is the labelling step",
        ],
    }
    return result


def run(cfg: dict) -> dict[str, Any]:
    result = characterize(cfg)
    cfgutil.save_json(result, cfgutil.output_dir(cfg, "metrics") / "amplicon_metrics.json")
    for end, data in result["ends"].items():
        print(
            f"[amplicon] end {end}: conserved block {data['conserved_block_length']} bp "
            f"(conservation {data['conserved_block']['before']:.3f}) -> variable "
            f"{data['variable_region_length']} bp "
            f"(conservation {data['conserved_block']['after']:.3f}); "
            f"{data['informative_fraction_of_read'] * 100:.1f}% of the read varies"
        )
    if "summary" in result:
        print(f"[amplicon] weighted informative fraction: "
              f"{result['summary']['weighted_informative_fraction'] * 100:.1f}% of each read")
    return result
