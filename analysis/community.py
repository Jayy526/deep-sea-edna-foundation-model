"""Stage 2 -- unsupervised community structure over sequence variants.

The unit of analysis is the unique variant (see analysis/dereplicate.py), and
every summary is **abundance-weighted**: a cluster containing one variant backed
by 400,000 reads is not the same thing as a cluster of 400 singletons, and
reporting only variant counts would hide that.

WHAT CAN AND CANNOT BE CLAIMED
------------------------------
Clusters here are groups of similar sequences. They are NOT taxa, NOT species,
and NOT OTUs in the formal sense (no similarity threshold, no denoising). This
dataset has no taxonomy, so no cluster can be named, and none is. Cluster count
is a property of the embedding and the chosen k, not a species estimate.

What the clustering CAN legitimately support:
  * how concentrated the community is,
  * whether the foundation model and the k-mer baseline organise the same
    variants into the same groups,
  * whether cluster structure tracks measurable sequence properties (GC content,
    amplicon end, abundance) rather than being arbitrary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.dereplicate import variants_path
from preprocessing.run import SUBSET_SCHEMA, subset_path
from utils import config as cfgutil


def build_variant_subset(cfg: dict, name: str = "variants", limit: int | None = None) -> Path:
    """Write the top variants in the standard subset format so the existing
    encoder pipeline can consume them unchanged."""
    out = subset_path(cfg, name)
    table = pq.read_table(variants_path(cfg))
    n = min(limit or table.num_rows, table.num_rows)

    seqs = table.column("sequence").to_pylist()[:n]
    ids = table.column("variant_id").to_pylist()[:n]
    gc = table.column("gc_content").to_pylist()[:n]

    rows = {
        "record_index": list(range(n)),
        "seq_id": [f"variant_{i}" for i in ids],
        "sequence": seqs,
        "raw_length": [len(s) for s in seqs],
        "effective_length": [len(s) for s in seqs],
        "gc_content": gc,
    }
    pq.write_table(
        pa.Table.from_pydict(rows, schema=SUBSET_SCHEMA), out, compression="zstd"
    )
    print(f"[community] Wrote {out.name}: {n:,} variants for encoding")
    return out


def variant_weights(cfg: dict, limit: int) -> tuple[np.ndarray, np.ndarray]:
    """(read counts, gc content) for the encoded variants, aligned by row."""
    table = pq.read_table(variants_path(cfg), columns=["count", "gc_content"])
    counts = np.array(table.column("count").to_pylist()[:limit], dtype=np.float64)
    gc = np.array(table.column("gc_content").to_pylist()[:limit], dtype=np.float64)
    return counts, gc


def _standardize(matrix: np.ndarray) -> np.ndarray:
    data = matrix.astype(np.float64)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def cluster_community(
    matrix: np.ndarray,
    counts: np.ndarray,
    gc: np.ndarray,
    end_groups: np.ndarray,
    k_grid: list[int],
    seed: int = 42,
    silhouette_sample: int = 10000,
) -> dict[str, Any]:
    """Cluster variants and profile the clusters, weighted by read abundance."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    data = _standardize(matrix)
    rng = np.random.default_rng(seed)
    total_reads = float(counts.sum())

    sweep = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_grid:
        if k >= data.shape[0]:
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(data)
        labels_by_k[k] = labels
        sample = min(silhouette_sample, data.shape[0])
        idx = rng.choice(data.shape[0], sample, replace=False)
        sweep.append(
            {
                "k": k,
                "silhouette": round(
                    float(silhouette_score(data[idx], labels[idx], random_state=seed)), 6
                ),
                "inertia": round(float(model.inertia_), 3),
                "largest_cluster_read_share": round(
                    float(
                        max(counts[labels == c].sum() for c in range(k)) / total_reads
                    ),
                    6,
                ),
            }
        )

    best = max(sweep, key=lambda r: r["silhouette"]) if sweep else None
    best_k = best["k"] if best else None
    labels = labels_by_k[best_k] if best_k else np.zeros(data.shape[0], dtype=int)

    # ---- Abundance-weighted cluster profile ------------------------------
    profile = []
    for c in range(best_k or 1):
        mask = labels == c
        if not mask.any():
            continue
        cluster_reads = float(counts[mask].sum())
        ends = end_groups[mask]
        assigned = ends[ends >= 0]
        profile.append(
            {
                "cluster": int(c),
                "n_variants": int(mask.sum()),
                "variant_share": round(float(mask.sum() / len(labels)), 6),
                "n_reads": int(cluster_reads),
                "read_share": round(cluster_reads / total_reads, 6),
                "mean_gc": round(float(gc[mask].mean() * 100), 3),
                "sd_gc": round(float(gc[mask].std() * 100), 3),
                "top_variant_reads": int(counts[mask].max()),
                "dominant_amplicon_end": (
                    int(np.bincount(assigned).argmax()) if assigned.size else None
                ),
                "amplicon_end_purity": (
                    round(float(np.bincount(assigned).max() / assigned.size), 4)
                    if assigned.size
                    else None
                ),
            }
        )
    profile.sort(key=lambda r: -r["read_share"])

    read_shares = np.array([p["read_share"] for p in profile])
    return {
        "n_variants_clustered": int(data.shape[0]),
        "reads_represented": int(total_reads),
        "k_sweep": sweep,
        "selected_k": best_k,
        "selection_criterion": "highest silhouette over the k grid",
        "cluster_profile": profile,
        "clusters_for_50pct_reads": int(np.searchsorted(np.cumsum(read_shares), 0.50) + 1),
        "clusters_for_90pct_reads": int(np.searchsorted(np.cumsum(read_shares), 0.90) + 1),
        "largest_cluster_read_share": round(float(read_shares.max()), 6),
        "mean_amplicon_end_purity": round(
            float(
                np.mean([p["amplicon_end_purity"] for p in profile
                         if p["amplicon_end_purity"] is not None])
            ),
            6,
        ),
        "labels": labels.tolist(),
        "caveat": (
            "Clusters are groups of similar sequences, not taxa. No cluster is "
            "named or assigned a taxonomic identity, because this dataset has no "
            "taxonomy. Cluster count is a property of the embedding and the "
            "selected k, not a species estimate."
        ),
        "seed": seed,
    }


def compare_partitions(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, Any]:
    """Do the two representations group the same variants together?

    Adjusted Rand and adjusted mutual information are chance-corrected, so 0
    means 'no more agreement than random' and 1 means identical partitions.
    """
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        fowlkes_mallows_score,
    )

    n = min(len(labels_a), len(labels_b))
    a, b = np.asarray(labels_a)[:n], np.asarray(labels_b)[:n]
    return {
        "n_variants": int(n),
        "adjusted_rand_index": round(float(adjusted_rand_score(a, b)), 6),
        "adjusted_mutual_information": round(float(adjusted_mutual_info_score(a, b)), 6),
        "fowlkes_mallows": round(float(fowlkes_mallows_score(a, b)), 6),
        "n_clusters_a": int(len(set(a.tolist()))),
        "n_clusters_b": int(len(set(b.tolist()))),
        "interpretation": (
            "Chance-corrected: 0 = agreement no better than random, 1 = identical "
            "partitions. A low value means the two representations disagree about "
            "which variants belong together -- with no labels, neither can be "
            "declared correct."
        ),
    }


def structure_vs_properties(
    labels: np.ndarray, gc: np.ndarray, counts: np.ndarray, end_groups: np.ndarray
) -> dict[str, Any]:
    """Does cluster structure track measurable sequence properties?

    Without taxonomy this is the only way to check that clusters are not
    arbitrary: a partition that explains real variance in GC content and aligns
    with amplicon end is capturing something about the sequences.
    """
    from sklearn.metrics import adjusted_rand_score

    labels = np.asarray(labels)
    valid = end_groups >= 0

    def variance_explained(values: np.ndarray) -> float:
        total = values.var()
        if total == 0:
            return 0.0
        within = sum(
            (labels == c).sum() * values[labels == c].var()
            for c in np.unique(labels)
        ) / len(values)
        return float(1 - within / total)

    return {
        "gc_variance_explained_by_clusters": round(variance_explained(gc), 6),
        "log_abundance_variance_explained_by_clusters": round(
            variance_explained(np.log10(counts + 1)), 6
        ),
        "agreement_with_amplicon_end_ari": round(
            float(adjusted_rand_score(end_groups[valid], labels[valid])), 6
        )
        if valid.any()
        else None,
        "n_variants_with_assigned_end": int(valid.sum()),
        "note": (
            "GC and amplicon end are measured sequence properties, not labels. "
            "High variance explained means the partition is structured rather "
            "than arbitrary; it says nothing about taxonomic correctness."
        ),
    }
