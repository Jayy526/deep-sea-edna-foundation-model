"""Diagram of the ACTUALLY IMPLEMENTED architecture.

This module is offline documentation. It is never imported by the inference
path and nothing in it touches a model or an embedding.

The diagram distinguishes, by colour and by an explicit legend:
  * input data
  * preprocessing (full dataset)
  * frozen components (no gradients, no training)
  * hand-crafted baseline
  * offline analysis (reads stored embeddings only)
  * stages that could not run for lack of ground-truth labels
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from utils import config as cfgutil
from visualization import style as S

# Role -> (fill, edge)
PALETTE = {
    "input": ("#dce8f7", "#2a78d6"),
    "preprocess": ("#d9f0e6", "#1baf7a"),
    "frozen": ("#dfe0f2", "#4a3aa7"),
    "baseline": ("#fbe3d6", "#eb6834"),
    "analysis": ("#f9ecc9", "#eda100"),
    "unavailable": ("#efeeea", "#9a9992"),
}

PALETTE["stage2"] = ("#e6e2f5", "#7a5cc4")

LABELS = {
    "input": "Input data",
    "preprocess": "Preprocessing — full dataset",
    "frozen": "Frozen foundation model — no gradients, no training",
    "baseline": "Hand-crafted baseline — 0 parameters",
    "analysis": "Stage 1 — representation analysis on stored embeddings",
    "stage2": "Stage 2 — unsupervised community structure (full dataset)",
    "unavailable": "Not executed — no ground-truth taxonomy in this dataset",
}


def _box(ax, x, y, w, h, text, role, fontsize=9.5, italic=False):
    fill, edge = PALETTE[role]
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=fill, edgecolor=edge, linewidth=1.6,
        linestyle="--" if role == "unavailable" else "-",
    )
    ax.add_patch(box)
    ax.text(
        x, y, text, ha="center", va="center", fontsize=fontsize,
        color=S.TEXT_MUTED if role == "unavailable" else S.TEXT_PRIMARY,
        linespacing=1.35, style="italic" if italic else "normal",
        fontweight="bold" if role == "frozen" else "normal",
    )
    return (x, y, w, h)


def _arrow(ax, start, end, color=None, style="-|>", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=13,
            linewidth=1.5, color=color or "#84837c",
            linestyle="--" if dashed else "-",
            shrinkA=0, shrinkB=0,
        )
    )


def _facts(cfg: dict) -> dict:
    """Pull every number in the diagram from the metrics files.

    The caption claims all figures are measured, so none of them may be
    hard-coded here. Missing files fall back to a dash rather than a guess.
    """
    import json

    metrics = cfgutil.output_dir(cfg, "metrics")

    def load(name: str) -> dict:
        path = metrics / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    dataset = load("dataset_metrics.json")
    pre = load("preprocessing_metrics.json")
    found = load("encoder_metrics_main_genomeocean.json")
    base = load("encoder_metrics_main_kmer4.json")
    derep = load("dereplication_metrics.json")
    community = load("community_metrics.json")
    enc = found.get("encoder", {})
    # Compatibility for the MAIN run, not whichever encode wrote last.
    compat = (
        found.get("compatibility_check")
        or load("model_compatibility_main_genomeocean.json")
        or load("model_compatibility.json")
    )

    def num(value, fmt="{:,.0f}", default="—"):
        return fmt.format(value) if isinstance(value, (int, float)) else default

    return {
        "records": num(dataset.get("total_sequence_records")),
        "mb": num(dataset.get("source_file_mb"), "{:,.1f}"),
        "stream_rate": num(pre.get("records_per_second")),
        "pass": num(dataset.get("reads_passing_qc")),
        "pass_pct": num(
            (dataset.get("qc_pass_rate") or 0) * 100, "{:.2f}"
        ),
        "removed": num(dataset.get("reads_removed_by_qc")),
        "encode_n": num(found.get("sequences_requested")),
        "pairs_n": num(cfg["subset"]["pair_subset_pairs"]),
        "vis_n": num(cfg["subset"]["visualization_size"]),
        "base_rate": num(base.get("sequences_per_second")),
        "found_rate": num(found.get("sequences_per_second")),
        "params": num(enc.get("model_parameters_total")),
        "trainable": num(enc.get("model_parameters_trainable")),
        "found_dim": num(enc.get("embedding_dim")),
        "base_dim": num(base.get("embedding_dimension")),
        "tokens": num(compat.get("tokens_per_read_mean"), "{:.0f}"),
        "token_limit": num(compat.get("model_token_limit")),
        "token_pct": num((compat.get("fraction_of_limit_used") or 0) * 100, "{:.1f}"),
        "dtype": str(enc.get("dtype", "")).replace("torch.", ""),
        "read_bp": num(dataset.get("sequence_length", {}).get("max")),
        # --- stage 2 ---
        "variants": num(derep.get("unique_variants")),
        "variants_encoded": num(community.get("variants_encoded")),
        "variant_read_cover": num(
            (derep.get("fraction_of_reads_covered_by_written_variants") or 0) * 100,
            "{:.1f}%",
        ),
    }


def draw_architecture(cfg: dict) -> Path:
    """Render the implemented pipeline. Returns the PNG path (PDF/SVG alongside)."""
    S.apply_style()
    F = _facts(cfg)
    fig, ax = plt.subplots(figsize=(12.5, 19.5))
    # Left margin extends past 0 to give the stage-2 route its own lane, clear
    # of the stage-1 boxes.
    ax.set_xlim(-0.8, 10)
    # Stage 1 occupies y > 1; stage 2 runs from y ~ 0.3 down to -5; the band
    # below that is reserved for the legend.
    ax.set_ylim(-7.4, 15.25)
    ax.axis("off")
    ax.set_facecolor(S.SURFACE)

    cx = 5.0
    left, right = 2.55, 7.45
    w, h = 4.3, 0.82

    # --- Trunk: input -> preprocessing ------------------------------------
    _box(ax, cx, 14.7, w, h,
         f"RAW DEEP-SEA eDNA FASTA\nSRR26872904 · {F['mb']} MB · {F['records']} records",
         "input")
    _arrow(ax, (cx, 14.29), (cx, 13.93))

    _box(ax, cx, 13.5, w, h,
         f"FASTA STREAMING\nconstant memory · {F['stream_rate']} records/s", "preprocess")
    _arrow(ax, (cx, 13.09), (cx, 12.73))

    _box(ax, cx, 12.3, w, h + 0.18,
         f"QUALITY CONTROL\nvalidate · uppercase · ambiguity policy\n"
         f"{F['pass']} pass ({F['pass_pct']}%) · {F['removed']} removed", "preprocess")
    _arrow(ax, (cx, 11.79), (cx, 11.43))

    _box(ax, cx, 11.0, w, h,
         f"{F['read_bp']} bp QC-PASSED READS\n"
         f"no padding · no concatenation · no assembly", "preprocess")

    # --- Subset selection --------------------------------------------------
    _arrow(ax, (cx, 10.59), (cx, 10.23))
    _box(ax, cx, 9.8, w + 0.5, h + 0.18,
         f"SYSTEMATIC SUBSET SELECTION (fixed stride, seed {cfg['seed']})\n"
         f"{F['encode_n']} reads for representation analysis\n"
         f"{F['pairs_n']} paired-end clusters for retrieval evaluation",
         "preprocess", fontsize=9)

    # --- Fork --------------------------------------------------------------
    ax.plot([cx, cx], [9.29, 8.95], color="#84837c", linewidth=1.5)
    ax.plot([left, right], [8.95, 8.95], color="#84837c", linewidth=1.5)
    _arrow(ax, (left, 8.95), (left, 8.62))
    _arrow(ax, (right, 8.95), (right, 8.62))

    bw = 4.1
    _box(ax, left, 8.05, bw, h + 0.52,
         f"K-MER / TNF BASELINE\ncanonical {cfg['baseline']['k']}-mers, "
         f"reverse-complement collapsed\n"
         f"0 learned parameters · CPU\n{F['base_rate']} reads/s", "baseline", fontsize=8.8)
    _box(ax, right, 8.05, bw, h + 0.52,
         f"GENOMIC FOUNDATION MODEL\nGenomeOcean-500M (frozen backbone)\n"
         f"{F['params']} params · {F['trainable']} trainable\n"
         f"GPU {F['dtype']} · {F['found_rate']} reads/s\n"
         f"{F['read_bp']} bp → {F['tokens']} BPE tokens "
         f"({F['token_pct']}% of {F['token_limit']} limit)", "frozen", fontsize=8.8)

    _arrow(ax, (left, 7.46), (left, 7.12))
    _arrow(ax, (right, 7.46), (right, 7.12))

    _box(ax, left, 6.72, bw, h - 0.1,
         f"BASELINE VECTOR\n{F['base_dim']} dimensions", "baseline")
    _box(ax, right, 6.72, bw, h - 0.1,
         f"DEEP EMBEDDING\n{F['found_dim']} dimensions", "frozen")

    _arrow(ax, (left, 6.31), (left, 5.99))
    _arrow(ax, (right, 6.31), (right, 5.99))
    _box(ax, cx, 5.62, w + 1.4, h - 0.1,
         "SHARDED EMBEDDING STORE  (sequence_id + vector, float32, resumable)",
         "analysis", fontsize=9)

    # --- Merge into analysis ----------------------------------------------
    _arrow(ax, (cx, 5.21), (cx, 4.87))
    _box(ax, cx, 4.5, w + 0.9, h - 0.1,
         "EMBEDDING ANALYSIS  —  offline, model never reloaded", "analysis")

    ax.plot([cx, cx], [4.09, 3.82], color="#84837c", linewidth=1.5)
    ax.plot([2.35, 7.65], [3.82, 3.82], color="#84837c", linewidth=1.5)
    for x in (2.35, 5.0, 7.65):
        _arrow(ax, (x, 3.82), (x, 3.52))

    _box(ax, 2.35, 3.12, 2.75, h - 0.02,
         "PCA\nexplained variance\nintrinsic dimension", "analysis", fontsize=8.8)
    _box(ax, 5.0, 3.12, 2.75, h - 0.02,
         f"UMAP\n{F['vis_n']}-read\nvisualization sample", "analysis", fontsize=8.8)
    _box(ax, 7.65, 3.12, 2.75, h - 0.02,
         "MATE-PAIR RETRIEVAL\nlabel-free, strand- and\nprimer-end-controlled",
         "analysis", fontsize=8.8)

    for x in (2.35, 5.0, 7.65):
        ax.plot([x, x], [2.71, 2.42], color="#84837c", linewidth=1.5)
    ax.plot([2.35, 7.65], [2.42, 2.42], color="#84837c", linewidth=1.5)
    _arrow(ax, (cx, 2.42), (cx, 2.12))

    _box(ax, cx, 1.72, w + 0.9, h - 0.02,
         "BASELINE vs FOUNDATION-MODEL COMPARISON\nsame reads · same seed · same metrics", "analysis")

    # --- STAGE 2: branches off the FULL QC-passed reads, not the subset -----
    # Routed down the left margin so it is visually clear that stage 2 consumes
    # the whole dataset rather than stage 1's sample.
    route_x = -0.45
    qc_left = cx - w / 2
    for seg in (
        ([qc_left, route_x], [11.0, 11.0]),
        ([route_x, route_x], [11.0, 0.30]),
        ([route_x, 2.15], [0.30, 0.30]),
    ):
        ax.plot(seg[0], seg[1], color=PALETTE["stage2"][1], linewidth=1.6,
                linestyle=(0, (6, 3)))
    ax.plot([qc_left], [11.0], marker="o", markersize=5, color=PALETTE["stage2"][1])
    _arrow(ax, (2.15, 0.30), (2.45, 0.30), color=PALETTE["stage2"][1])
    ax.text(route_x + 0.13, 6.0, "STAGE 2 consumes the FULL QC-passed dataset",
            rotation=90, va="center", ha="left", fontsize=8.6,
            color=PALETTE["stage2"][1], fontweight="bold")

    _box(ax, cx, 0.30, w + 1.5, h + 0.18,
         f"FULL-DATASET DEREPLICATION\n"
         f"{F['pass']} reads → {F['variants']} unique sequence variants\n"
         f"read count carried as an explicit abundance weight",
         "stage2", fontsize=8.8)
    _arrow(ax, (cx, -0.21), (cx, -0.55))

    _box(ax, cx, -0.95, w + 1.5, h - 0.08,
         f"{F['variants_encoded']} abundant variants  ({F['variant_read_cover']} of all reads)",
         "stage2", fontsize=9)

    # Fork back into the SAME two encoders -- reused, not reimplemented.
    ax.plot([cx, cx], [-1.36, -1.62], color="#84837c", linewidth=1.5)
    ax.plot([left, right], [-1.62, -1.62], color="#84837c", linewidth=1.5)
    _arrow(ax, (left, -1.62), (left, -1.92))
    _arrow(ax, (right, -1.62), (right, -1.92))
    _box(ax, left, -2.32, bw, h - 0.1,
         "k-mer / TNF\n(same encoder as above)", "baseline", fontsize=8.6)
    _box(ax, right, -2.32, bw, h - 0.1,
         "GenomeOcean-500M\n(same encoder as above)", "frozen", fontsize=8.6)

    ax.plot([left, left], [-2.73, -2.99], color="#84837c", linewidth=1.5)
    ax.plot([right, right], [-2.73, -2.99], color="#84837c", linewidth=1.5)
    ax.plot([left, right], [-2.99, -2.99], color="#84837c", linewidth=1.5)
    _arrow(ax, (cx, -2.99), (cx, -3.29))

    _box(ax, cx, -3.72, w + 1.7, h + 0.26,
         "UNSUPERVISED COMMUNITY STRUCTURE\n"
         "diversity (Hill, Chao1, Good's) · rank abundance · rarefaction\n"
         "abundance-weighted clustering · partition agreement",
         "stage2", fontsize=8.8)

    # --- Blocked stage -----------------------------------------------------
    _arrow(ax, (cx, -4.26), (cx, -4.56), dashed=True)
    _box(ax, cx, -5.0, w + 1.9, h + 0.08,
         "TAXONOMIC CLASSIFIER  →  CONFIDENCE / CALIBRATION ANALYSIS\n"
         "not executed: this dataset contains no ground-truth taxonomy",
         "unavailable", fontsize=9, italic=True)

    # --- Legend ------------------------------------------------------------
    handles = [
        mpatches.Patch(facecolor=PALETTE[k][0], edgecolor=PALETTE[k][1],
                       linewidth=1.5, label=LABELS[k])
        for k in ("input", "preprocess", "frozen", "baseline", "analysis",
                  "stage2", "unavailable")
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.012), ncol=2,
        fontsize=8.8, frameon=True, facecolor=S.SURFACE, edgecolor=S.GRID,
        labelcolor=S.TEXT_SECONDARY, borderpad=0.8, columnspacing=1.6,
        handlelength=1.6, labelspacing=0.7,
    )

    fig.suptitle(
        "Implemented architecture — stages 1 and 2\n"
        "A TaxDistill-inspired adaptation for 151 bp deep-sea eDNA short reads",
        fontsize=13.5, fontweight="bold", color=S.TEXT_PRIMARY, y=0.988,
    )
    fig.text(
        0.5, 0.012,
        "Every number shown is measured, not illustrative. Visualization is strictly offline: it consumes stored "
        "embeddings and is never part of the inference path.\n"
        "The dashed stage is architecturally implemented but was not executed, because populating it would have "
        "required inventing taxonomic labels this dataset does not contain.",
        ha="center", va="bottom", fontsize=8.6, color=S.TEXT_MUTED, linespacing=1.5,
    )

    path = cfgutil.output_dir(cfg, "figures") / "architecture.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=S.SURFACE)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=S.SURFACE)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor=S.SURFACE)
    plt.close(fig)
    print(f"[figure] {path.name} (+ .pdf, .svg)")
    return path
