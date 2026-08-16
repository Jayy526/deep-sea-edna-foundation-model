"""Stage 2 -- diversity statistics from the variant frequency spectrum.

Everything here is computed from the FULL-dataset frequency spectrum (how many
variants were seen exactly k times), so no read is excluded and nothing is
estimated from a subsample except the rarefaction curve, which is explicitly a
subsampling analysis.

WHAT THESE NUMBERS MEAN, AND WHAT THEY DO NOT
---------------------------------------------
These are diversity statistics over *exact sequence variants*, not over taxa.
Sequencing error inflates richness (mostly as singletons), and a single organism
may contribute several variants while two organisms may share one. Richness
here is therefore an upper bound on organismal diversity, not an estimate of it.
Every index below is reported as a property of the variant distribution.

Chao1 and Good's coverage are included precisely because they quantify the
singleton problem rather than hiding it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _spectrum_arrays(spectrum: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """(k, number_of_variants_seen_k_times), sorted by k."""
    items = sorted((int(k), int(v)) for k, v in spectrum.items())
    k = np.array([i[0] for i in items], dtype=np.int64)
    f = np.array([i[1] for i in items], dtype=np.int64)
    return k, f


def diversity_from_spectrum(spectrum: dict[str, int]) -> dict[str, Any]:
    """All indices computable from a frequency spectrum."""
    k, f = _spectrum_arrays(spectrum)
    n_reads = int((k * f).sum())
    richness = int(f.sum())
    if n_reads == 0 or richness == 0:
        return {}

    p = k / n_reads  # relative abundance of a variant seen k times

    shannon = float(-(f * p * np.log(p)).sum())
    simpson = float((f * p**2).sum())  # probability two reads share a variant
    pielou = shannon / math.log(richness) if richness > 1 else 0.0

    singletons = int(f[k == 1].sum()) if (k == 1).any() else 0
    doubletons = int(f[k == 2].sum()) if (k == 2).any() else 0

    # Chao1 (bias-corrected form; valid when doubletons == 0 too)
    chao1 = richness + (singletons * (singletons - 1)) / (2 * (doubletons + 1))
    # Good's coverage: estimated fraction of the community already sampled
    goods_coverage = 1.0 - (singletons / n_reads)

    # Effective numbers (Hill numbers) -- interpretable as "equally-common variants"
    hill0 = float(richness)
    hill1 = float(math.exp(shannon))
    hill2 = float(1.0 / simpson) if simpson > 0 else float("inf")

    # Dominance
    max_k = int(k.max())
    top_share = max_k / n_reads

    return {
        "reads": n_reads,
        "observed_richness_variants": richness,
        "singletons": singletons,
        "doubletons": doubletons,
        "singleton_fraction_of_variants": round(singletons / richness, 6),
        "singleton_fraction_of_reads": round(singletons / n_reads, 8),
        "shannon_H": round(shannon, 6),
        "simpson_D": round(simpson, 8),
        "inverse_simpson": round(hill2, 4),
        "gini_simpson_1_minus_D": round(1 - simpson, 8),
        "pielou_evenness_J": round(pielou, 6),
        "hill_q0_richness": round(hill0, 2),
        "hill_q1_exp_shannon": round(hill1, 4),
        "hill_q2_inverse_simpson": round(hill2, 4),
        "chao1_estimated_richness": round(float(chao1), 2),
        "chao1_over_observed": round(float(chao1) / richness, 4),
        "goods_coverage": round(goods_coverage, 8),
        "most_abundant_variant_reads": max_k,
        "most_abundant_variant_share": round(top_share, 8),
        "interpretation": {
            "hill_q1": "Number of equally-common variants that would give the same Shannon entropy.",
            "chao1": "Lower-bound estimate of true richness given the singleton/doubleton ratio.",
            "goods_coverage": "Estimated fraction of the community represented by the sample.",
            "caveat": (
                "Richness is over exact sequence variants, not taxa. Sequencing error "
                "inflates it; it is an upper bound on organismal diversity."
            ),
        },
    }


def rank_abundance(spectrum: dict[str, int], max_points: int = 5000) -> dict[str, Any]:
    """Rank-abundance (Whittaker) curve reconstructed from the spectrum.

    The spectrum tells us how many variants have each count, so the sorted
    abundance vector is recoverable exactly without ever materialising all
    variants.
    """
    k, f = _spectrum_arrays(spectrum)
    order = np.argsort(-k)
    k, f = k[order], f[order]

    counts = np.repeat(k, f)  # descending abundance
    n_reads = int(counts.sum())
    ranks = np.arange(1, counts.size + 1)

    # Log-spaced sampling so the curve plots cheaply without distortion.
    if counts.size > max_points:
        idx = np.unique(
            np.geomspace(1, counts.size, max_points).astype(np.int64) - 1
        )
    else:
        idx = np.arange(counts.size)

    cumulative = np.cumsum(counts) / n_reads
    return {
        "n_variants": int(counts.size),
        "n_reads": n_reads,
        "ranks": ranks[idx].tolist(),
        "abundances": counts[idx].tolist(),
        "relative_abundances": (counts[idx] / n_reads).tolist(),
        "cumulative_read_fraction": cumulative[idx].tolist(),
        "variants_for_50pct_reads": int(np.searchsorted(cumulative, 0.50) + 1),
        "variants_for_90pct_reads": int(np.searchsorted(cumulative, 0.90) + 1),
        "variants_for_99pct_reads": int(np.searchsorted(cumulative, 0.99) + 1),
        "top10_read_share": round(float(cumulative[min(9, counts.size - 1)]), 6),
        "top100_read_share": round(float(cumulative[min(99, counts.size - 1)]), 6),
        "top1000_read_share": round(float(cumulative[min(999, counts.size - 1)]), 6),
    }


def rarefaction(spectrum: dict[str, int], n_points: int = 40) -> dict[str, Any]:
    """Expected richness vs sequencing depth, by the analytic (Hurlbert) formula.

    Uses the exact expectation of observed richness when subsampling m of n
    reads without replacement, computed in log-space for numerical stability.
    No random subsampling is involved, so the curve is deterministic.
    """
    k, f = _spectrum_arrays(spectrum)
    n = int((k * f).sum())
    richness = int(f.sum())

    depths = np.unique(np.linspace(n / n_points, n, n_points).astype(np.int64))
    from scipy.special import gammaln

    expected = []
    for m in depths:
        # E[S(m)] = S - sum_i C(n-k_i, m) / C(n, m)
        # log C(n-k, m) - log C(n, m)
        with np.errstate(invalid="ignore"):
            log_term = (
                gammaln(n - k + 1) - gammaln(m + 1) - gammaln(n - k - m + 1)
            ) - (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1))
        term = np.exp(log_term)
        term[(n - k - m) < 0] = 0.0
        expected.append(float(richness - (f * term).sum()))

    saturation = expected[-1] / richness if richness else 0.0
    # Slope over the final decade of depth: how fast is richness still growing?
    if len(depths) > 1:
        slope = (expected[-1] - expected[-2]) / (depths[-1] - depths[-2])
    else:
        slope = 0.0

    return {
        "method": "analytic (Hurlbert) rarefaction, deterministic",
        "total_reads": n,
        "observed_richness": richness,
        "depths": depths.tolist(),
        "expected_richness": [round(v, 2) for v in expected],
        "saturation_at_full_depth": round(saturation, 6),
        "new_variants_per_additional_read_at_full_depth": round(float(slope), 8),
        "interpretation": (
            "A curve still climbing steeply at full depth means the library is "
            "under-sampled: more sequencing would keep revealing new variants."
        ),
    }
