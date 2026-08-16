"""Figures 1-4 (+ nucleotide composition): dataset characterisation.

Every number plotted here is measured over the FULL dataset (3,026,920 reads).
Nothing is sampled, illustrative, or synthetic. Each figure answers one stated
experimental question, printed in its caption.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt

from utils import config as cfgutil
from visualization import style as S


def _metrics(cfg: dict) -> dict:
    path = cfgutil.output_dir(cfg, "metrics") / "dataset_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _figdir(cfg: dict) -> Path:
    return cfgutil.output_dir(cfg, "figures")


def figure_sequence_length(cfg: dict) -> Path:
    """Q: Are these reads or assembled contigs, and does QC change their length?"""
    m = _metrics(cfg)
    raw = m["sequence_length"]
    eff = m["effective_length_after_qc"]
    eff_hist = {int(k): v for k, v in m["effective_length_histogram"].items()}
    n_total = m["total_sequence_records"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1.35]})

    S.stat_tile(
        axes[0],
        f"{raw['min']} bp",
        "Raw read length",
        f"Identical for all {n_total:,} records\n"
        f"(min = max = {raw['min']}, sd = {raw['std']:g})",
    )

    ax = axes[1]
    lengths = np.array(sorted(eff_hist))
    counts = np.array([eff_hist[int(x)] for x in lengths], dtype=float)
    ax.bar(lengths, counts, width=1.0, color=S.SERIES[0], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Effective read length after quality control (bp)")
    ax.set_ylabel("Reads (log scale)")
    ax.set_title("Length after ambiguity handling")
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.annotate(
        f"{eff_hist.get(151, 0) / eff['n'] * 100:.2f}% retain the full 151 bp",
        xy=(151, eff_hist.get(151, 1)),
        xytext=(-8, -14), textcoords="offset points",
        ha="right", fontsize=9, color=S.TEXT_SECONDARY,
    )
    S.clean_axes(ax)

    S.suptitle(
        fig,
        "Figure 1 — Sequence length: a uniform short-read dataset, not assembled contigs",
    )
    S.caption(
        fig,
        f"Full dataset (n = {n_total:,} records). No read is padded at any stage. "
        f"Shortening occurs only where the longest-unambiguous-run policy trims an "
        f"ambiguous base (mean effective length {eff['mean']:g} bp).",
    )
    return S.save(fig, _figdir(cfg) / "sequence_length.png", also_pdf=True)


def figure_gc_content(cfg: dict) -> Path:
    """Q: What is the GC-content distribution, and is it unimodal?"""
    m = _metrics(cfg)
    table = pq.read_table(
        cfgutil.output_dir(cfg, "metrics") / "read_metadata.parquet",
        columns=["gc_content", "qc_pass"],
    )
    gc = table.column("gc_content").to_numpy(zero_copy_only=False)
    keep = table.column("qc_pass").to_numpy(zero_copy_only=False)
    gc = gc[keep]
    stats = m["gc_content_qc_passed_reads"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0),
                             layout="constrained", gridspec_kw={"width_ratios": [1.6, 1]})

    # GC content of a 151 bp read is discrete: it can only take the values
    # k/151. One bin per attainable value avoids the aliasing comb that an
    # arbitrary bin count produces on discrete data.
    read_bp = int(m["sequence_length"]["max"])
    bins = (np.arange(read_bp + 2) - 0.5) / read_bp * 100

    ax = axes[0]
    ax.hist(gc * 100, bins=bins, color=S.SERIES[0], edgecolor="none")
    ax.axvline(stats["mean"] * 100, color=S.SERIES[1], linewidth=2,
               label=f"mean = {stats['mean'] * 100:.2f}%")
    ax.axvline(stats["median"] * 100, color=S.SERIES[2], linewidth=2, linestyle="--",
               label=f"median = {stats['median'] * 100:.2f}%")
    ax.set_xlim(stats["p05"] * 100 - 12, stats["p95"] * 100 + 12)
    ax.set_xlabel("GC content (%)")
    ax.set_ylabel("Reads")
    ax.set_title("GC content per read (zoomed to the bulk)")
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.legend(loc="upper right")
    S.clean_axes(ax)

    ax = axes[1]
    ax.hist(gc * 100, bins=bins, color=S.SERIES[0], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("GC content (%)")
    ax.set_ylabel("Reads (log scale)")
    ax.set_title("Same data, log scale: the tails")
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    S.clean_axes(ax)

    S.suptitle(
        fig,
        "Figure 2 — GC-content distribution across the deep-sea eDNA read set",
    )
    S.caption(
        fig,
        f"Full dataset, QC-passed reads (n = {stats['n']:,}). "
        f"sd = {stats['std']:.4f}, 5th-95th percentile = "
        f"{stats['p05'] * 100:.1f}%-{stats['p95'] * 100:.1f}%. "
        f"The log panel shows the low- and high-GC tails that the linear panel hides.",
    )
    return S.save(fig, _figdir(cfg) / "gc_content.png", also_pdf=True)


def figure_ambiguity(cfg: dict) -> Path:
    """Q: How much ambiguity is there, and is it positional (a sequencing artefact)?"""
    m = _metrics(cfg)
    profile = m["position_profile"]
    amb = np.array(profile["ambiguous_per_position"], dtype=float)
    cover = np.array(profile["coverage_per_position"], dtype=float)
    rate = np.divide(amb, cover, out=np.zeros_like(amb), where=cover > 0) * 100

    table = pq.read_table(
        cfgutil.output_dir(cfg, "metrics") / "read_metadata.parquet",
        columns=["n_ambiguous"],
    )
    n_amb = table.column("n_ambiguous").to_numpy(zero_copy_only=False)
    n_amb = n_amb[n_amb > 0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9),
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1.7, 1.1]})

    S.stat_tile(
        axes[0],
        f"{m['reads_with_ambiguous_bases'] / m['total_sequence_records'] * 100:.2f}%",
        "of reads contain any ambiguous base",
        f"{m['reads_with_ambiguous_bases']:,} reads, "
        f"{m['total_ambiguous_bases']:,} N bases\n"
        f"{m['reads_with_invalid_characters']:,} reads with invalid characters",
    )

    ax = axes[1]
    ax.bar(np.arange(1, len(rate) + 1), rate, width=1.0, color=S.SERIES[0], edgecolor="none")
    ax.set_xlabel("Position within read (bp, 1-indexed)")
    ax.set_ylabel("Reads with N at this position (%)")
    ax.set_title("Ambiguity is positional, not uniform")
    peak = int(np.argmax(rate)) + 1
    ax.annotate(
        f"cycle {peak}: {rate[peak - 1]:.3f}%",
        xy=(peak, rate[peak - 1]), xytext=(18, -4), textcoords="offset points",
        fontsize=9, color=S.TEXT_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=S.TEXT_MUTED, linewidth=0.8),
    )
    S.clean_axes(ax)

    ax = axes[2]
    top = int(n_amb.max()) if n_amb.size else 1
    ax.hist(n_amb, bins=np.arange(0.5, top + 1.5), color=S.SERIES[0], edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Ambiguous bases per affected read")
    ax.set_ylabel("Reads (log scale)")
    ax.set_title("Ambiguity per affected read")
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    S.clean_axes(ax)

    S.suptitle(
        fig,
        "Figure 3a — Ambiguous-base distribution",
    )
    S.caption(
        fig,
        f"Full dataset (n = {m['total_sequence_records']:,} records). Ambiguity concentrates "
        f"at the first sequencing cycles, the signature of a base-calling artefact rather than "
        f"biological sequence. This matters because the encoder's BPE vocabulary covers ACGT only.",
    )
    return S.save(fig, _figdir(cfg) / "ambiguity.png", also_pdf=True)


def figure_qc_filtering(cfg: dict) -> Path:
    """Q: How much data does QC remove, and for which specific reasons?"""
    m = _metrics(cfg)
    reasons = dict(m["qc_reason_counts"])
    total = m["total_sequence_records"]
    n_pass = reasons.pop("pass", 0)
    removed = {k: v for k, v in sorted(reasons.items(), key=lambda kv: kv[1]) if v > 0}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9),
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1.5]})

    S.stat_tile(
        axes[0],
        f"{n_pass / total * 100:.3f}%",
        "of reads pass quality control",
        f"{n_pass:,} kept / {total - n_pass:,} removed\nout of {total:,} records",
        color=S.STATUS_GOOD,
    )

    ax = axes[1]
    labels = [k.replace("_", " ") for k in removed]
    values = list(removed.values())
    bars = ax.barh(labels, values, color=S.SERIES[1], height=0.55)
    ax.bar_label(bars, labels=[f"{v:,}" for v in values], padding=5,
                 fontsize=9, color=S.TEXT_SECONDARY)
    ax.set_xlabel("Reads removed")
    ax.set_title("Why reads were removed")
    ax.set_xlim(0, max(values) * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(S.thousands))
    S.clean_axes(ax, x_grid=True, y_grid=False)

    S.suptitle(
        fig,
        "Figure 3 — Quality-control filtering outcome",
    )
    S.caption(
        fig,
        "Full dataset. Removal reasons are mutually exclusive (first failing filter wins) and "
        "every threshold is set in configs/default.json. Reasons that removed zero reads are omitted.",
    )
    return S.save(fig, _figdir(cfg) / "qc_filtering.png", also_pdf=True)


def figure_read_id_analysis(cfg: dict) -> Path:
    """Q: Why does every sequence ID occur twice? Evidence, not assumption."""
    m = _metrics(cfg)
    ids = m["read_id_analysis"]

    dist = {int(k.split("_")[0]): v for k, v in ids["multiplicity_distribution"].items()}
    multiplicity_note = ", ".join(
        f"{v:,} IDs appear {k}x" for k, v in sorted(dist.items())
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9),
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1.6]})

    # ID multiplicity is a single value (every ID appears exactly twice), so it is
    # reported as a stat tile rather than a one-category bar chart.
    S.stat_tile(
        axes[0],
        f"{ids['total_records'] / ids['unique_sequence_ids']:.0f}x",
        "records per sequence ID —\nwith no exceptions",
        f"{ids['total_records']:,} records / {ids['unique_sequence_ids']:,} unique IDs\n"
        f"{multiplicity_note}",
    )

    ax = axes[1]
    evidence = [
        ("Repeats adjacent\nin file order", ids["fraction_of_repeats_adjacent_in_file"]),
        ("Repeats on the\nsame flowcell tile", ids["fraction_of_repeats_on_same_tile"]),
        ("Repeats with\nidentical GC content", ids["fraction_of_repeats_with_identical_gc"]),
    ]
    labels = [e[0] for e in evidence]
    values = [e[1] * 100 for e in evidence]
    colors = [S.SERIES[0], S.SERIES[0], S.SERIES[1]]
    bars = ax.barh(labels, values, color=colors, height=0.5)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=5,
                 fontsize=9, color=S.TEXT_SECONDARY)
    ax.set_xlabel("Percentage of repeated-ID groups")
    ax.set_xlim(0, 118)
    ax.set_title("Evidence: what repeated IDs actually are")
    ax.invert_yaxis()
    S.clean_axes(ax, x_grid=True, y_grid=False)

    S.suptitle(
        fig,
        "Figure 4 — Read-ID duplication: measured evidence for a paired-end layout",
    )
    S.caption(
        fig,
        "Full dataset. Every ID occurs exactly twice; both records are always adjacent and always on "
        "the same flowcell tile, yet almost never carry the same sequence content. That combination "
        "identifies them as the two mates (R1/R2) of one paired-end cluster, so the two records are "
        "distinct sequences, not duplicated data.",
    )
    return S.save(fig, _figdir(cfg) / "read_id_analysis.png", also_pdf=True)


def figure_nucleotide_composition(cfg: dict) -> Path:
    """Q: Is base composition balanced, and does it drift along the read?"""
    m = _metrics(cfg)
    fractions = m["nucleotide_fractions"]
    bases = [b for b in ("A", "C", "G", "T") if b in fractions]
    other = {b: v for b, v in fractions.items() if b not in bases}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9),
                             layout="constrained", gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    values = [fractions[b] * 100 for b in bases]
    bars = ax.bar(bases, values, color=S.SERIES[0], width=0.55)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=4,
                 fontsize=9, color=S.TEXT_SECONDARY)
    ax.axhline(25, color=S.TEXT_MUTED, linewidth=1, linestyle=":")
    ax.text(3.45, 25.4, "25% (uniform)", fontsize=8.5, color=S.TEXT_MUTED, ha="right")
    ax.set_ylabel("Share of all sequenced bases (%)")
    ax.set_xlabel("Nucleotide")
    ax.set_title("Genome-wide base composition")
    ax.set_ylim(0, max(values) * 1.25)
    S.clean_axes(ax)

    ax = axes[1]
    labels = list(other)
    vals = [other[b] * 100 for b in labels]
    if labels:
        bars = ax.bar(labels, vals, color=S.SERIES[1], width=0.4)
        ax.bar_label(bars, labels=[f"{v:.5f}%" for v in vals], padding=4,
                     fontsize=9, color=S.TEXT_SECONDARY)
        ax.set_ylim(0, max(vals) * 1.4)
    ax.set_ylabel("Share of all sequenced bases (%)")
    ax.set_xlabel("Non-ACGT character")
    ax.set_title("Non-ACGT characters observed")
    S.clean_axes(ax)

    S.suptitle(
        fig,
        "Figure 3b — Nucleotide composition",
    )
    S.caption(
        fig,
        f"Full dataset: {m['total_base_pairs']:,} sequenced bases across "
        f"{m['total_sequence_records']:,} reads. A+T and G+C are near-balanced, consistent with the "
        f"measured mean GC of {m['gc_content_all_reads']['mean'] * 100:.2f}%.",
    )
    return S.save(fig, _figdir(cfg) / "nucleotide_composition.png", also_pdf=True)


def generate_all(cfg: dict) -> list[Path]:
    S.apply_style()
    return [
        figure_sequence_length(cfg),
        figure_gc_content(cfg),
        figure_ambiguity(cfg),
        figure_qc_filtering(cfg),
        figure_read_id_analysis(cfg),
        figure_nucleotide_composition(cfg),
    ]
