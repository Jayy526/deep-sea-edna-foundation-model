"""Figures 5-8 and 12: embedding analysis, model comparison, performance.

All plots read precomputed coordinates and metrics from disk. The foundation
model is never invoked from this module.

Every scatter states its VISUALIZATION SAMPLE SIZE, which is smaller than the
encoding subset, which is in turn far smaller than the full dataset. The three
numbers are never conflated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt

from utils import config as cfgutil
from visualization import style as S

FOUNDATION_COLOR = S.SERIES[0]
BASELINE_COLOR = S.SERIES[1]


def _figdir(cfg: dict) -> Path:
    return cfgutil.output_dir(cfg, "figures")


def figure_pca_explained_variance(cfg: dict, analyses: dict[str, dict]) -> Path:
    """Q: How many dimensions does each representation actually use?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    colors = {"foundation_model": FOUNDATION_COLOR, "baseline": BASELINE_COLOR}
    names = {
        key: f"{base} ({analyses[key]['embedding_dim']}-d)"
        for key, base in (
            ("foundation_model", "GenomeOcean-500M"),
            ("baseline", "Canonical 4-mer / TNF"),
        )
    }

    ax = axes[0]
    for key, analysis in analyses.items():
        ratio = np.array(analysis["pca"]["explained_variance_ratio"]) * 100
        ax.plot(np.arange(1, len(ratio) + 1), ratio, color=colors[key],
                marker="o", markersize=3.5, label=names[key])
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Variance per component")
    ax.set_yscale("log")
    ax.legend()
    S.clean_axes(ax)

    ax = axes[1]
    # Stagger the 90%-threshold callouts vertically; the two markers sit only a
    # few components apart and their labels would otherwise collide.
    offsets = {"foundation_model": (8, -26), "baseline": (8, -12)}
    for key, analysis in analyses.items():
        cumulative = np.array(analysis["pca"]["cumulative_explained_variance"]) * 100
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=colors[key],
                linewidth=2.2, label=names[key])
        n90 = analysis["pca"]["components_for_90pct"]
        if n90:
            ax.plot([n90], [cumulative[n90 - 1]], marker="o", markersize=8,
                    color=colors[key], markeredgecolor=S.SURFACE, markeredgewidth=2)
            ax.annotate(f"{n90} PCs → 90%", xy=(n90, cumulative[n90 - 1]),
                        xytext=offsets[key], textcoords="offset points",
                        fontsize=9, color=colors[key], fontweight="bold")
    ax.axhline(90, color=S.TEXT_MUTED, linestyle=":", linewidth=1)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("Cumulative variance")
    ax.set_ylim(0, 102)
    ax.legend(loc="lower right")
    S.clean_axes(ax)

    n = analyses["foundation_model"]["n_encoded"]
    S.suptitle(fig, "Figure 5 — PCA explained variance: intrinsic dimensionality of each representation")
    S.caption(
        fig,
        f"Encoding subset: n = {n:,} QC-passed reads, identical for both representations. "
        f"Each matrix is standardized before PCA so the two are on comparable scales. "
        f"A representation whose variance collapses into few components is using less of its nominal width.",
    )
    return S.save(fig, _figdir(cfg) / "pca_explained_variance.png", also_pdf=True)


def _scatter(ax, coords, color_values, title, xlabel, ylabel, cmap="viridis"):
    order = np.argsort(color_values)
    handle = ax.scatter(
        coords[order, 0], coords[order, 1], c=color_values[order],
        s=2.5, alpha=0.55, cmap=cmap, linewidths=0, rasterized=True,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    S.clean_axes(ax, x_grid=True)
    return handle


def _gc_for_index(cfg: dict, store_name: str, subset_name: str) -> np.ndarray | None:
    """GC content of the visualised reads, used as the only honest colour signal.

    There are no taxonomic labels, so points are coloured by a measured
    sequence property rather than by an invented class.
    """
    import pyarrow.parquet as pq

    from analysis.embedding_analysis import load_coords

    idx = load_coords(cfg, store_name, "index")
    if idx is None:
        return None
    path = cfgutil.output_dir(cfg, "embeddings") / f"subset_{subset_name}.parquet"
    gc = pq.read_table(path, columns=["gc_content"]).column("gc_content").to_numpy(
        zero_copy_only=False
    )
    return gc[idx] * 100


def figure_projection(
    cfg: dict, kind: str, analyses: dict[str, dict], subset_name: str, figure_number: str
) -> Path | str:
    """Q: Does either representation organise reads into structure, and does that
    structure track a measured sequence property (GC content)?"""
    from analysis.embedding_analysis import load_coords

    coords = {
        key: load_coords(cfg, analysis["store_name"], kind)
        for key, analysis in analyses.items()
    }
    if any(c is None for c in coords.values()):
        return f"{kind} coordinates unavailable"

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    titles = {
        key: f"{base} ({analyses[key]['embedding_dim']}-d)"
        for key, base in (
            ("foundation_model", "GenomeOcean-500M embedding"),
            ("baseline", "Canonical 4-mer / TNF baseline"),
        )
    }
    axis_label = "PC" if kind == "pca" else "UMAP"
    handle = None
    for ax, (key, analysis) in zip(axes, analyses.items()):
        gc = _gc_for_index(cfg, analysis["store_name"], subset_name)
        handle = _scatter(
            ax, coords[key], gc, titles[key],
            f"{axis_label} 1", f"{axis_label} 2",
        )
    bar = fig.colorbar(handle, ax=axes, fraction=0.03, pad=0.02)
    bar.set_label("GC content (%)", color=S.TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(labelsize=8, color=S.TEXT_SECONDARY)

    n_vis = analyses["foundation_model"]["visualization_sample_size"]
    n_enc = analyses["foundation_model"]["n_encoded"]
    label = "PCA" if kind == "pca" else "UMAP"
    S.suptitle(fig, f"Figure {figure_number} — {label} projection of read embeddings")
    S.caption(
        fig,
        f"VISUALIZATION SAMPLE: n = {n_vis:,} reads, drawn with a fixed seed from the "
        f"{n_enc:,}-read encoding subset (itself drawn from 3,018,522 QC-passed reads). "
        f"Points are coloured by measured GC content, not by taxonomy — this dataset has no "
        f"taxonomic labels, so no class colouring is possible and none is invented.",
    )
    return S.save(fig, _figdir(cfg) / f"{kind}.png", also_pdf=True)


def figure_model_comparison(cfg: dict, comparison: dict) -> Path:
    """Q: Does the foundation model actually beat the k-mer baseline, and at what cost?"""
    ret = comparison["mate_pair_retrieval"]
    rep = comparison["representations"]
    order = ["baseline", "foundation_model"]
    names = ["k-mer / TNF\nbaseline", "GenomeOcean-500M\nfoundation model"]
    colors = [BASELINE_COLOR, FOUNDATION_COLOR]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), layout="constrained")

    ax = axes[0]
    values = [ret[k]["top1_accuracy"] * 100 for k in order]
    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4,
                 fontsize=9.5, color=S.TEXT_SECONDARY)
    chance = ret[order[0]]["chance_top1_accuracy"] * 100
    ax.axhline(chance, color=S.TEXT_MUTED, linestyle=":", linewidth=1.2)
    ax.text(1.45, chance + max(values) * 0.02, f"chance = {chance:.0f}%",
            fontsize=8.5, color=S.TEXT_MUTED, ha="right")
    ax.set_ylabel("Top-1 retrieval accuracy (%)")
    ax.set_title("Mate-pair retrieval")
    ax.set_ylim(0, max(values) * 1.25)
    S.clean_axes(ax)

    ax = axes[1]
    values = [ret[k]["auroc_mate_vs_random"] for k in order]
    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4,
                 fontsize=9.5, color=S.TEXT_SECONDARY)
    ax.axhline(0.5, color=S.TEXT_MUTED, linestyle=":", linewidth=1.2)
    ax.text(1.45, 0.515, "chance = 0.5", fontsize=8.5, color=S.TEXT_MUTED, ha="right")
    ax.set_ylabel("AUROC (mate vs random read)")
    ax.set_title("Mate / non-mate separation")
    ax.set_ylim(0.45, 1.02)
    S.clean_axes(ax)

    ax = axes[2]
    values = [rep[k]["sequences_per_second"] for k in order]
    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values], padding=4,
                 fontsize=9.5, color=S.TEXT_SECONDARY)
    ax.set_yscale("log")
    ax.set_ylabel("Reads encoded per second (log scale)")
    ax.set_title("Inference throughput")
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    S.clean_axes(ax)

    ax = axes[3]
    values = [rep[k]["feature_dimension"] for k in order]
    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=4,
                 fontsize=9.5, color=S.TEXT_SECONDARY)
    ax.set_ylabel("Feature dimension")
    ax.set_title("Representation width")
    ax.set_ylim(0, max(values) * 1.2)
    S.clean_axes(ax)

    n_pairs = ret["foundation_model"]["n_pairs"]
    S.suptitle(fig, "Figure 8 — Baseline vs foundation model on identical reads")
    S.caption(
        fig,
        f"Retrieval measured on {n_pairs:,} paired-end clusters, {ret['foundation_model']['n_distractors_per_query']} "
        f"distractors per query, same reads and same seed for both representations. Throughput is measured on this "
        f"machine (foundation model on GPU, baseline on CPU) — it compares realised cost, not equivalent hardware. "
        f"Taxonomic accuracy and F1 are not computable: this dataset has no ground-truth labels.",
    )
    return S.save(fig, _figdir(cfg) / "model_comparison.png", also_pdf=True)


def figure_computational_performance(cfg: dict, perf: dict, projection: dict) -> Path:
    """Q: Is this computationally feasible, and what limits it?"""
    rows = perf["rows"]
    batch = [r["batch_size"] for r in rows]
    throughput = [r["sequences_per_second"] for r in rows]
    gpu = [r["peak_gpu_allocated_mb"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), layout="constrained")

    ax = axes[0]
    ax.plot(batch, throughput, color=FOUNDATION_COLOR, marker="o", markersize=7)
    for x, y in zip(batch, throughput):
        ax.annotate(f"{y:,.0f}", xy=(x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=S.TEXT_SECONDARY)
    ax.set_xscale("log", base=2)
    ax.set_xticks(batch)
    ax.set_xticklabels([str(b) for b in batch])
    ax.set_xlabel("Batch size (reads)")
    ax.set_ylabel("Reads encoded per second")
    ax.set_title("Throughput vs batch size")
    ax.set_ylim(0, max(throughput) * 1.2)
    S.clean_axes(ax)

    ax = axes[1]
    total = perf["environment"].get("gpu_total_mb")
    bars = ax.bar([str(b) for b in batch], gpu, color=FOUNDATION_COLOR, width=0.6)
    ax.bar_label(bars, labels=[f"{g:,.0f}" for g in gpu], padding=4,
                 fontsize=8.5, color=S.TEXT_SECONDARY)
    if total:
        ax.axhline(total, color=S.STATUS_CRITICAL, linestyle="--", linewidth=1.4)
        ax.text(len(batch) - 0.5, total * 0.955, f"GPU capacity {total:,.0f} MB",
                fontsize=8.5, color=S.STATUS_CRITICAL, ha="right", va="top")
        ax.set_ylim(0, total * 1.1)
    ax.set_xlabel("Batch size (reads)")
    ax.set_ylabel("Peak GPU memory allocated (MB)")
    ax.set_title("Memory headroom")
    S.clean_axes(ax)

    ax = axes[2]
    S.stat_tile(
        ax,
        f"{projection['projected_minutes']:.0f} min",
        "projected to encode all\n3,018,522 QC-passed reads",
        f"at the measured {projection['measured_throughput_seq_per_second']:,.0f} reads/s\n"
        f"producing {projection['projected_embedding_storage_gb']:.1f} GB of embeddings\n"
        f"(projection from measured throughput; not executed)",
        size=24,
    )

    gpu_name = perf["environment"].get("gpu", "CPU")
    S.suptitle(fig, "Figure 12 — Computational performance of the foundation-model encoder")
    S.caption(
        fig,
        f"Measured on {gpu_name} ({perf['environment'].get('gpu_total_mb', 0):,.0f} MB), "
        f"{perf['encoder']['dtype']}, {perf['encoder']['attn_implementation']} attention, "
        f"{perf['sequences_per_configuration']:,} reads per configuration, warm-up excluded and CUDA-synchronised. "
        f"The right panel is a linear projection from measured throughput, clearly labelled as such.",
    )
    return S.save(fig, _figdir(cfg) / "computational_performance.png", also_pdf=True)


def figure_cluster_tendency(cfg: dict, analyses: dict[str, dict]) -> Path:
    """Q: Is there real cluster structure, beyond what a null model produces?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained")
    titles = {
        "foundation_model": "GenomeOcean-500M embedding",
        "baseline": "k-mer / TNF baseline",
    }
    for ax, (key, analysis) in zip(axes, analyses.items()):
        rows = analysis["cluster_tendency"]["per_k"]
        ks = [r["k"] for r in rows]
        ax.plot(ks, [r["real_silhouette"] for r in rows], color=S.SERIES[0],
                marker="o", markersize=6, label="Real embeddings")
        ax.plot(ks, [r["shuffled_null_silhouette"] for r in rows], color=S.SERIES[2],
                marker="s", markersize=5, linestyle="--", label="Dimension-shuffled null")
        ax.set_xlabel("Number of k-means clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title(titles[key])
        ax.legend()
        S.clean_axes(ax)

    n = analyses["foundation_model"]["cluster_tendency"]["analysis_sample_size"]
    S.suptitle(fig, "Figure 6b — Cluster tendency against a null model")
    S.caption(
        fig,
        f"Analysis sample: n = {n:,} reads. The null shuffles each dimension independently, preserving every "
        f"marginal distribution while destroying joint structure; only the gap between the two curves is "
        f"attributable to genuine structure. These clusters have NO verified taxonomic meaning — this is a "
        f"clusterability statistic, not a classification result.",
    )
    return S.save(fig, _figdir(cfg) / "cluster_tendency.png", also_pdf=True)
