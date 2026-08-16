"""Figures 13-17: Stage 2 unsupervised community structure.

Every number is measured over the FULL dataset (3,018,522 QC-passed reads),
except the clustering, which runs over the encoded variant set and says so.

None of these figures names a taxon, because this dataset has no taxonomy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt

from utils import config as cfgutil
from visualization import style as S

FOUNDATION_COLOR = S.SERIES[0]
BASELINE_COLOR = S.SERIES[1]


def _figdir(cfg: dict) -> Path:
    return cfgutil.output_dir(cfg, "figures")


def figure_dereplication(cfg: dict, derep: dict, diversity: dict) -> Path:
    """Q: How many distinct molecules are actually in this library?"""
    idx = diversity["indices"]
    spectrum = {int(k): v for k, v in derep["frequency_spectrum"].items()}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), layout="constrained",
                             gridspec_kw={"width_ratios": [1, 1.4, 1.3]})

    S.stat_tile(
        axes[0],
        f"{idx['observed_richness_variants']:,}",
        "unique sequence variants",
        f"from {idx['reads']:,} QC-passed reads\n"
        f"({derep['reads_per_variant_mean']:.1f} reads per variant)",
    )

    ax = axes[1]
    counts = np.array(sorted(spectrum))
    freqs = np.array([spectrum[c] for c in counts], dtype=float)
    ax.scatter(counts, freqs, s=12, color=FOUNDATION_COLOR, alpha=0.6, linewidths=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reads supporting a variant (k)")
    ax.set_ylabel("Variants seen exactly k times")
    ax.set_title("Frequency spectrum")
    # Anchored in axes fraction, well clear of the title and the point itself.
    ax.annotate(
        f"{idx['singletons']:,} singletons\n"
        f"{idx['singleton_fraction_of_variants'] * 100:.1f}% of variants,\n"
        f"only {idx['singleton_fraction_of_reads'] * 100:.1f}% of reads",
        xy=(1, spectrum.get(1, 1)), xycoords="data",
        xytext=(0.36, 0.62), textcoords="axes fraction",
        fontsize=8.5, color=S.TEXT_SECONDARY, linespacing=1.4,
        arrowprops=dict(arrowstyle="-", color=S.TEXT_MUTED, linewidth=0.8,
                        connectionstyle="arc3,rad=0.15"),
    )
    S.clean_axes(ax, x_grid=True)

    ax = axes[2]
    names = ["Observed\nrichness", "Chao1\nestimate"]
    values = [idx["observed_richness_variants"], idx["chao1_estimated_richness"]]
    bars = ax.bar(names, values, color=[FOUNDATION_COLOR, S.SERIES[3]], width=0.5)
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in values], padding=4,
                 fontsize=9, color=S.TEXT_SECONDARY)
    ax.set_ylabel("Sequence variants")
    ax.set_title("Richness: observed vs estimated")
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    S.clean_axes(ax)

    S.suptitle(fig, "Figure 13 — Dereplication: 3M reads are not 3M molecules")
    S.caption(
        fig,
        f"Full dataset. Exact-sequence variants: no denoising error model and no similarity "
        f"clustering, so sequencing error inflates richness — visible as the singleton spike. "
        f"The Chao1 estimate ({idx['chao1_over_observed']:.1f}x observed) is driven by those same "
        f"singletons and should be read as an upper bound on molecular diversity, not as a "
        f"species count. Good's coverage = {idx['goods_coverage'] * 100:.2f}%.",
    )
    return S.save(fig, _figdir(cfg) / "dereplication.png", also_pdf=True)


def figure_rank_abundance(cfg: dict, diversity: dict) -> Path:
    """Q: How evenly is the community distributed? (Answer: it is not.)"""
    ra = diversity["rank_abundance"]
    idx = diversity["indices"]
    ranks = np.array(ra["ranks"], dtype=float)
    rel = np.array(ra["relative_abundances"], dtype=float)
    cum = np.array(ra["cumulative_read_fraction"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), layout="constrained",
                             gridspec_kw={"width_ratios": [1.3, 1.3, 1]})

    ax = axes[0]
    ax.plot(ranks, rel * 100, color=FOUNDATION_COLOR, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Variant rank (most to least abundant)")
    ax.set_ylabel("Share of all reads (%)")
    ax.set_title("Rank-abundance (Whittaker) curve")
    S.clean_axes(ax, x_grid=True)

    ax = axes[1]
    ax.plot(ranks, cum * 100, color=FOUNDATION_COLOR, linewidth=2.2)
    ax.set_xscale("log")
    for frac, key, color in (
        (50, "variants_for_50pct_reads", S.SERIES[1]),
        (90, "variants_for_90pct_reads", S.SERIES[2]),
    ):
        n_var = ra[key]
        ax.axhline(frac, color=color, linestyle=":", linewidth=1.2)
        ax.plot([n_var], [frac], marker="o", markersize=8, color=color,
                markeredgecolor=S.SURFACE, markeredgewidth=2)
        ax.annotate(f"{n_var:,} variants → {frac}% of reads",
                    xy=(n_var, frac), xytext=(10, -16), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")
    ax.set_xlabel("Number of variants (ranked by abundance)")
    ax.set_ylabel("Cumulative share of reads (%)")
    ax.set_title("Community concentration")
    ax.set_ylim(0, 104)
    S.clean_axes(ax, x_grid=True)

    S.stat_tile(
        axes[2],
        f"{ra['variants_for_50pct_reads']}",
        "variants make up\nhalf the entire library",
        f"top 10 = {ra['top10_read_share'] * 100:.1f}% of reads\n"
        f"top 100 = {ra['top100_read_share'] * 100:.1f}%\n"
        f"single most abundant = {idx['most_abundant_variant_share'] * 100:.2f}%",
        color=S.SERIES[1],
    )

    S.suptitle(fig, "Figure 14 — Rank abundance: an extremely uneven community")
    S.caption(
        fig,
        f"Full dataset ({idx['reads']:,} reads, {idx['observed_richness_variants']:,} variants). "
        f"Shannon H' = {idx['shannon_H']:.3f}, Pielou evenness J = {idx['pielou_evenness_J']:.3f}, "
        f"inverse Simpson = {idx['inverse_simpson']:.1f}. The Hill numbers make the unevenness "
        f"concrete: {idx['observed_richness_variants']:,} observed variants behave like only "
        f"~{idx['hill_q1_exp_shannon']:.0f} equally-common ones (q=1), or ~{idx['hill_q2_inverse_simpson']:.0f} (q=2).",
    )
    return S.save(fig, _figdir(cfg) / "rank_abundance.png", also_pdf=True)


def figure_rarefaction(cfg: dict, diversity: dict) -> Path:
    """Q: Has this library been sequenced deeply enough?"""
    rf = diversity["rarefaction"]
    idx = diversity["indices"]
    depths = np.array(rf["depths"], dtype=float)
    expected = np.array(rf["expected_richness"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), layout="constrained",
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    ax.plot(depths, expected, color=FOUNDATION_COLOR, linewidth=2.4)
    ax.plot([depths[-1]], [expected[-1]], marker="o", markersize=9,
            color=FOUNDATION_COLOR, markeredgecolor=S.SURFACE, markeredgewidth=2)
    ax.annotate(
        f"full depth:\n{expected[-1]:,.0f} variants",
        xy=(depths[-1], expected[-1]), xytext=(-18, -40), textcoords="offset points",
        ha="right", fontsize=9, color=S.TEXT_SECONDARY,
    )
    ax.set_xlabel("Sequencing depth (reads sampled)")
    ax.set_ylabel("Expected number of variants observed")
    ax.set_title("Rarefaction curve")
    ax.xaxis.set_major_formatter(FuncFormatter(S.thousands))
    ax.yaxis.set_major_formatter(FuncFormatter(S.thousands))
    S.clean_axes(ax)

    S.stat_tile(
        axes[1],
        f"{rf['new_variants_per_additional_read_at_full_depth']:.3f}",
        "new variants per additional read,\nat full sequencing depth",
        f"the curve has not saturated\n"
        f"Good's coverage = {idx['goods_coverage'] * 100:.2f}%\n"
        f"Chao1 suggests ~{idx['chao1_estimated_richness']:,.0f} variants exist",
        color=S.SERIES[3],
    )

    S.suptitle(fig, "Figure 15 — Rarefaction: the library is not sequenced to saturation")
    S.caption(
        fig,
        f"Full dataset, analytic (Hurlbert) rarefaction — deterministic, no random subsampling. "
        f"Still gaining {rf['new_variants_per_additional_read_at_full_depth']:.3f} variants per read at "
        f"{rf['total_reads']:,} reads. Much of that unsaturated tail is sequencing error rather than "
        f"biology, which is exactly why richness here is an upper bound.",
    )
    return S.save(fig, _figdir(cfg) / "rarefaction.png", also_pdf=True)


def figure_community_clusters(cfg: dict, results: dict) -> Path:
    """Q: Do the two representations agree on community structure?"""
    reps = results["representations"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0), layout="constrained")

    ax = axes[0]
    for key, color, name in (
        ("foundation_model", FOUNDATION_COLOR, "GenomeOcean-500M"),
        ("baseline", BASELINE_COLOR, "k-mer / TNF"),
    ):
        sweep = reps[key]["k_sweep"]
        ax.plot([r["k"] for r in sweep], [r["silhouette"] for r in sweep],
                color=color, marker="o", markersize=6, label=name)
        best = reps[key]["selected_k"]
        row = next(r for r in sweep if r["k"] == best)
        ax.plot([best], [row["silhouette"]], marker="o", markersize=11, color=color,
                markeredgecolor=S.SURFACE, markeredgewidth=2.5)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Cluster-count selection")
    ax.legend()
    S.clean_axes(ax)

    ax = axes[1]
    # The two curves are near-identical -- that IS the result -- so the second is
    # drawn dashed and offset in width, otherwise one simply hides the other.
    for key, color, name, style, width in (
        ("foundation_model", FOUNDATION_COLOR, "GenomeOcean-500M", "-", 4.0),
        ("baseline", BASELINE_COLOR, "k-mer / TNF", "--", 2.0),
    ):
        profile = reps[key]["cluster_profile"]
        shares = np.cumsum([p["read_share"] for p in profile]) * 100
        ax.plot(np.arange(1, len(shares) + 1), shares, color=color, linestyle=style,
                linewidth=width, marker="o", markersize=4, label=name, alpha=0.9)
    ax.axhline(90, color=S.TEXT_MUTED, linestyle=":", linewidth=1)
    ax.text(0.04, 0.06, "the two curves coincide", transform=ax.transAxes,
            ha="left", fontsize=8.5, color=S.TEXT_MUTED, style="italic")
    ax.set_xlabel("Clusters, ranked by read abundance")
    ax.set_ylabel("Cumulative share of reads (%)")
    ax.set_title("How concentrated are the clusters?")
    ax.set_ylim(0, 104)
    ax.legend(loc="lower right")
    S.clean_axes(ax)

    ax = axes[2]
    agree = results["partition_agreement"]
    metrics = ["Adjusted\nRand", "Adjusted\nmutual info", "Fowlkes-\nMallows"]
    values = [
        agree["adjusted_rand_index"],
        agree["adjusted_mutual_information"],
        agree["fowlkes_mallows"],
    ]
    bars = ax.bar(metrics, values, color=S.SERIES[2], width=0.5)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4,
                 fontsize=9.5, color=S.TEXT_SECONDARY)
    ax.set_ylabel("Agreement (0 = chance, 1 = identical)")
    ax.set_title("Do the two partitions match?")
    ax.set_ylim(0, max(max(values) * 1.3, 0.25))
    S.clean_axes(ax)

    S.suptitle(fig, "Figure 16 — Community structure: foundation model vs k-mer baseline")
    S.caption(
        fig,
        f"{results['variants_encoded']:,} unique variants representing "
        f"{results['reads_represented']:,} reads ({results['reads_represented'] / 3018522 * 100:.1f}% of the "
        f"QC-passed dataset). Clusters are groups of similar sequences, NOT taxa — this dataset has no "
        f"taxonomy, so no cluster is named and neither partition can be declared correct.",
    )
    return S.save(fig, _figdir(cfg) / "community_clusters.png", also_pdf=True)


def figure_community_map(cfg, results, labels, counts, gc) -> Path:
    """Q: What does the community look like, weighted by actual abundance?"""
    from analysis.embedding_analysis import load_coords
    from analysis.pca import run_pca
    from embeddings.store import load_embeddings

    root = cfgutil.output_dir(cfg, "embeddings")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")

    n = results["variants_encoded"]
    for ax, (key, store, name) in zip(
        axes,
        (
            ("foundation_model", "variants_genomeocean", "GenomeOcean-500M"),
            ("baseline", "variants_kmer4", "k-mer / TNF baseline"),
        ),
    ):
        _, matrix, _ = load_embeddings(root, store)
        _, projected = run_pca(matrix[:n], 2, cfg["seed"])
        # Marker area proportional to read abundance: the visual weight of a
        # variant matches how much of the library it actually represents.
        size = 3 + 260 * (counts / counts.max()) ** 0.5
        order = np.argsort(counts)
        ax.scatter(
            projected[order, 0], projected[order, 1],
            s=size[order], c=labels[key][order],
            cmap="tab20", alpha=0.55, linewidths=0, rasterized=True,
        )
        ax.set_title(f"{name} — {results['representations'][key]['selected_k']} clusters")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        S.clean_axes(ax, x_grid=True)

    S.suptitle(fig, "Figure 17 — Abundance-weighted community map")
    S.caption(
        fig,
        f"{n:,} unique variants, PCA of each representation, coloured by cluster assignment. "
        f"Marker area is proportional to the square root of read abundance, so the few variants "
        f"carrying most of the library are visually dominant — the same weighting used in every "
        f"cluster statistic. Colours are cluster IDs, not taxa.",
    )
    return S.save(fig, _figdir(cfg) / "community_map.png", also_pdf=True)


def figure_amplicon_architecture(cfg: dict, amplicon: dict) -> Path:
    """Q: How much of each 151 bp read actually carries discriminating signal?"""
    ends = amplicon["ends"]
    read_len = amplicon["read_length"]
    fig, axes = plt.subplots(1, len(ends) + 1, figsize=(6.0 * len(ends) + 4.2, 4.2),
                             layout="constrained",
                             gridspec_kw={"width_ratios": [1.5] * len(ends) + [1]})

    for ax, (end, data) in zip(axes, sorted(ends.items())):
        profile = np.array(data["conservation_profile"]) * 100
        cut = data["conserved_block_length"]
        positions = np.arange(1, len(profile) + 1)

        ax.fill_between(positions[:cut], 0, profile[:cut], color=S.SERIES[0],
                        alpha=0.75, linewidth=0, label="conserved anchor")
        ax.fill_between(positions[cut - 1:], 0, profile[cut - 1:], color=S.SERIES[1],
                        alpha=0.75, linewidth=0, label="variable region")
        ax.axhline(25, color=S.TEXT_MUTED, linestyle=":", linewidth=1)
        ax.text(read_len - 2, 26.5, "25% = no conservation", fontsize=8,
                color=S.TEXT_MUTED, ha="right")
        ax.axvline(cut, color=S.TEXT_PRIMARY, linestyle="--", linewidth=1.4)
        # Place each label at the centre of its own block, but clamped inside the
        # axes so a narrow conserved block does not push its label off the edge.
        for text, centre, colour in (
            (f"{cut} bp conserved\n(mean {data['conserved_block']['before'] * 100:.1f}%)",
             cut / 2, S.SERIES[0]),
            (f"{data['variable_region_length']} bp variable\n"
             f"(mean {data['conserved_block']['after'] * 100:.1f}%)",
             cut + (read_len - cut) / 2, S.SERIES[1]),
        ):
            frac = min(max(centre / read_len, 0.13), 0.87)
            ax.text(frac, 0.46, text, transform=ax.transAxes, ha="center",
                    fontsize=8.8, color=colour, fontweight="bold", linespacing=1.4,
                    bbox=dict(facecolor=S.SURFACE, edgecolor="none", alpha=0.82,
                              boxstyle="round,pad=0.3"))
        ax.set_xlabel("Position within read (bp)")
        # Only the leftmost panel carries the (long) y-label.
        if ax is axes[0]:
            ax.set_ylabel("Conservation:\nfrequency of the commonest base (%)")
        ax.set_title(f"Amplicon end {end} — {data['reads_represented']:,} reads")
        ax.set_ylim(0, 105)
        ax.set_xlim(1, read_len)
        ax.legend(loc="lower left", fontsize=8.5)
        S.clean_axes(ax)

    S.stat_tile(
        axes[-1],
        f"{amplicon['summary']['weighted_informative_fraction'] * 100:.0f}%",
        "of each read varies\nacross the community",
        "the rest is near-identical in every read,\n"
        "so it carries no organism-discriminating\n"
        "signal for any representation",
        color=S.SERIES[1],
    )

    S.suptitle(fig, "Figure 18 — Amplicon architecture: how much of a read is actually informative?")
    S.caption(
        fig,
        f"Abundance-weighted over the {amplicon['ends'][sorted(ends)[0]]['variants_used']:,} most abundant "
        f"variants per end, so a variant backed by many reads dominates the consensus. Conservation is the "
        f"frequency of the commonest base at each position; 25% would mean no conservation at all. The dashed "
        f"line is the fitted conserved/variable changepoint. This bounds what ANY representation can achieve: "
        f"positions that do not vary cannot discriminate organisms.",
    )
    return S.save(fig, _figdir(cfg) / "amplicon_architecture.png", also_pdf=True)


def figure_variable_region(cfg: dict, results: dict) -> Path:
    """Q: Does removing the conserved anchor change which representation wins?"""
    v = results["variants"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1]})

    labels = ["Full read\n(151 bp)", "Variable region\nonly"]
    x = np.arange(2)
    width = 0.36

    for ax, (metric, title, ylabel, chance) in zip(
        axes,
        (
            ("top1_accuracy", "Mate retrieval (top-1)", "Top-1 accuracy (%)",
             v["foundation_full_read"]["chance_top1_accuracy"] * 100),
            ("auroc_mate_vs_random", "Mate / non-mate separation", "AUROC", 50.0),
        ),
    ):
        scale = 100
        for i, (name, colour, keys) in enumerate((
            ("GenomeOcean-500M", FOUNDATION_COLOR,
             ("foundation_full_read", "foundation_variable_region")),
            ("k-mer / TNF", BASELINE_COLOR,
             ("baseline_full_read", "baseline_variable_region")),
        )):
            values = [v[k][metric] * scale for k in keys]
            bars = ax.bar(x + i * width, values, width=width, color=colour, label=name)
            fmt = "{:.2f}%" if metric == "top1_accuracy" else "{:.3f}"
            ax.bar_label(bars, labels=[fmt.format(val / (1 if metric == "top1_accuracy" else 100))
                                       for val in values],
                         padding=3, fontsize=8.8, color=S.TEXT_SECONDARY)
        ax.axhline(chance, color=S.TEXT_MUTED, linestyle=":", linewidth=1.2)
        ax.text(-0.42, chance, " chance", fontsize=8.5, color=S.TEXT_MUTED,
                ha="left", va="bottom")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8.5)
        S.clean_axes(ax)
    axes[1].set_ylim(40, 100)

    # Report the PAIRED significance test rather than declaring a winner on a
    # difference that may be noise.
    sig = results.get("significance", {})
    full = sig.get("full_read_foundation_vs_baseline", {})
    var = sig.get("variable_region_foundation_vs_baseline", {})
    base = sig.get("baseline_full_vs_variable", {})
    retained = results["subset"]["fraction_of_sequence_retained"] * 100

    headline = "tied" if not var.get("significant_at_0.05", True) else "differs"
    S.stat_tile(
        axes[2],
        headline,
        "top-1 on the variable region\n(McNemar, paired)",
        f"full read: baseline +{abs(full.get('difference_percentage_points', 0)):.2f} pp "
        f"(p={full.get('p_value', 0):.0e})\n"
        f"variable region: n.s. (p={var.get('p_value', 1):.2f})\n"
        f"trimming costs the baseline "
        f"{abs(base.get('difference_percentage_points', 0)):.2f} pp,\n"
        f"the foundation model nothing",
        color=S.SERIES[2],
        size=24,
    )

    S.suptitle(fig, "Figure 19 — Retrieval on the variable region alone")
    S.caption(
        fig,
        f"Same {v['foundation_variable_region']['n_pairs']:,} paired-end clusters, same seed, same "
        f"amplicon-end control, same strand correction. The only change is that the near-invariant "
        f"conserved anchor has been trimmed from each read, leaving {retained:.0f}% of the sequence. "
        f"This isolates whether the earlier result reflects the encoders or the data they were given. "
        f"Trimming discards real sequence and is a diagnostic, not a proposed default.",
    )
    return S.save(fig, _figdir(cfg) / "variable_region.png", also_pdf=True)
