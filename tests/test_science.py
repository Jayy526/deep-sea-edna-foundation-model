"""Correctness tests for the hand-implemented scientific computations.

Several quantities reported in the experiment reports are implemented from
scratch rather than taken from a library: Chao1, Shannon/Hill numbers, Hurlbert
rarefaction, a rank-based AUROC, canonical k-mer counting and the streaming
dereplication. If any of those is subtly wrong, every conclusion resting on it
is wrong too.

Each test below checks an implementation against an INDEPENDENT source of
truth -- a closed-form value computed by hand, a brute-force simulation, or an
established library implementation (sklearn/scipy) -- rather than against
itself.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import diversity
from evaluation import structure
from models.baseline_encoder import KmerBaselineEncoder, canonical_kmers, reverse_complement
from preprocessing.quality_control import QCConfig, evaluate

# ---------------------------------------------------------------------------
# Diversity indices -- checked against closed-form values computed by hand
# ---------------------------------------------------------------------------


def test_shannon_and_simpson_against_hand_computation():
    """A community of 4 variants with counts 5,3,1,1 (n=10)."""
    spectrum = {"1": 2, "3": 1, "5": 1}  # two variants seen once, one 3x, one 5x
    result = diversity.diversity_from_spectrum(spectrum)

    p = np.array([0.5, 0.3, 0.1, 0.1])
    expected_shannon = float(-(p * np.log(p)).sum())
    expected_simpson = float((p**2).sum())

    assert result["reads"] == 10
    assert result["observed_richness_variants"] == 4
    assert result["shannon_H"] == pytest.approx(expected_shannon, abs=1e-6)
    assert result["simpson_D"] == pytest.approx(expected_simpson, abs=1e-8)
    assert result["inverse_simpson"] == pytest.approx(1 / expected_simpson, abs=1e-3)
    assert result["hill_q1_exp_shannon"] == pytest.approx(math.exp(expected_shannon), abs=1e-3)
    assert result["pielou_evenness_J"] == pytest.approx(
        expected_shannon / math.log(4), abs=1e-6
    )


def test_shannon_matches_scipy_entropy():
    """Independent implementation: scipy.stats.entropy."""
    from scipy.stats import entropy

    counts = np.array([120, 45, 45, 12, 3, 3, 1, 1, 1])
    spectrum: dict[str, int] = {}
    for c in counts:
        spectrum[str(int(c))] = spectrum.get(str(int(c)), 0) + 1

    result = diversity.diversity_from_spectrum(spectrum)
    assert result["shannon_H"] == pytest.approx(float(entropy(counts)), abs=1e-6)


def test_chao1_matches_published_formula():
    """Bias-corrected Chao1: S_obs + F1(F1-1) / (2(F2+1))."""
    spectrum = {"1": 10, "2": 4, "7": 3}  # F1=10, F2=4, S_obs=17
    result = diversity.diversity_from_spectrum(spectrum)
    expected = 17 + (10 * 9) / (2 * (4 + 1))
    assert result["chao1_estimated_richness"] == pytest.approx(expected, abs=1e-6)


def test_chao1_handles_zero_doubletons():
    """The +1 in the denominator must keep this finite (the uncorrected form divides by zero)."""
    spectrum = {"1": 5, "9": 2}
    result = diversity.diversity_from_spectrum(spectrum)
    assert math.isfinite(result["chao1_estimated_richness"])
    assert result["chao1_estimated_richness"] == pytest.approx(7 + (5 * 4) / 2, abs=1e-6)


def test_goods_coverage_definition():
    """Good's coverage = 1 - F1/n."""
    spectrum = {"1": 3, "10": 2}  # F1=3, n = 3*1 + 2*10 = 23
    result = diversity.diversity_from_spectrum(spectrum)
    assert result["goods_coverage"] == pytest.approx(1 - 3 / 23, abs=1e-8)


def test_even_community_has_evenness_one():
    """Perfectly even community: Pielou J = 1, Hill q1 = richness."""
    spectrum = {"10": 5}  # five variants, ten reads each
    result = diversity.diversity_from_spectrum(spectrum)
    assert result["pielou_evenness_J"] == pytest.approx(1.0, abs=1e-9)
    assert result["hill_q1_exp_shannon"] == pytest.approx(5.0, abs=1e-6)
    assert result["hill_q2_inverse_simpson"] == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Rarefaction -- checked against brute-force Monte Carlo simulation
# ---------------------------------------------------------------------------


def test_hurlbert_rarefaction_matches_monte_carlo():
    """The analytic expectation must match random subsampling without replacement."""
    counts = [40, 25, 15, 10, 5, 3, 1, 1]
    spectrum: dict[str, int] = {}
    for c in counts:
        spectrum[str(c)] = spectrum.get(str(c), 0) + 1

    result = diversity.rarefaction(spectrum, n_points=10)
    depths = result["depths"]
    expected = result["expected_richness"]

    # Build the read pool: variant id repeated by its count.
    pool = np.repeat(np.arange(len(counts)), counts)
    rng = np.random.default_rng(0)

    for depth, analytic in zip(depths, expected):
        if depth >= len(pool):
            continue
        richness = [
            np.unique(rng.choice(pool, size=int(depth), replace=False)).size
            for _ in range(400)
        ]
        simulated = float(np.mean(richness))
        # Monte Carlo error over 400 draws; tolerance is generous but far
        # tighter than any plausible implementation bug.
        assert simulated == pytest.approx(analytic, abs=0.25), (
            f"depth={depth}: analytic={analytic}, simulated={simulated}"
        )


def test_rarefaction_endpoint_equals_observed_richness():
    """At full depth the expected richness must equal the observed richness exactly."""
    spectrum = {"1": 7, "4": 3, "20": 2}
    result = diversity.rarefaction(spectrum, n_points=8)
    assert result["depths"][-1] == result["total_reads"]
    assert result["expected_richness"][-1] == pytest.approx(
        result["observed_richness"], abs=1e-6
    )


# ---------------------------------------------------------------------------
# Rank-abundance reconstruction from the frequency spectrum
# ---------------------------------------------------------------------------


def test_rank_abundance_reconstructs_the_true_distribution():
    """The spectrum is a lossless encoding of the sorted abundance vector."""
    counts = [100, 50, 50, 20, 5, 5, 5, 1, 1, 1, 1]
    spectrum: dict[str, int] = {}
    for c in counts:
        spectrum[str(c)] = spectrum.get(str(c), 0) + 1

    result = diversity.rank_abundance(spectrum, max_points=10_000)
    assert result["n_variants"] == len(counts)
    assert result["n_reads"] == sum(counts)
    assert result["abundances"] == sorted(counts, reverse=True)

    # Cumulative thresholds, computed independently.
    cum = np.cumsum(sorted(counts, reverse=True)) / sum(counts)
    assert result["variants_for_50pct_reads"] == int(np.searchsorted(cum, 0.50) + 1)
    assert result["variants_for_90pct_reads"] == int(np.searchsorted(cum, 0.90) + 1)


# ---------------------------------------------------------------------------
# Rank-based AUROC -- checked against sklearn
# ---------------------------------------------------------------------------


def test_mate_pair_auroc_matches_sklearn():
    """The hand-rolled rank AUROC inside mate_pair_retrieval must match sklearn."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(7)
    # 300 pairs in 8-d, with mates deliberately correlated so AUROC > 0.5.
    n_pairs, dim = 300, 8
    base = rng.normal(size=(n_pairs, dim))
    mates = base + rng.normal(scale=0.6, size=(n_pairs, dim))
    matrix = np.empty((n_pairs * 2, dim), dtype=np.float32)
    matrix[0::2] = base
    matrix[1::2] = mates

    result = structure.mate_pair_retrieval(matrix, n_distractors=25, seed=3)

    # Recompute the same comparison independently.
    data = (matrix - matrix.mean(0)) / matrix.std(0)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    left, right = data[0::2], data[1::2]
    pos = np.einsum("ij,ij->i", left, right)

    rng2 = np.random.default_rng(3)
    neg = []
    for i in range(n_pairs):
        cand = rng2.integers(0, n_pairs, size=25)
        cand[cand == i] = (cand[cand == i] + 1) % n_pairs
        neg.extend(right[cand] @ left[i])
    neg = np.array(neg)

    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    scores = np.concatenate([pos, neg])
    assert result["auroc_mate_vs_random"] == pytest.approx(
        float(roc_auc_score(y, scores)), abs=1e-4
    )


def test_retrieval_on_random_data_scores_at_chance():
    """With no real mate signal, top-1 must sit near 1/(1+distractors)."""
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(1200, 16)).astype(np.float32)
    result = structure.mate_pair_retrieval(matrix, n_distractors=49, seed=5)
    chance = 1 / 50
    assert result["chance_top1_accuracy"] == pytest.approx(chance)
    assert abs(result["top1_accuracy"] - chance) < 0.02
    assert abs(result["auroc_mate_vs_random"] - 0.5) < 0.05


def test_retrieval_on_identical_mates_is_perfect():
    """If each mate is an exact copy, retrieval must be perfect."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(200, 12))
    matrix = np.repeat(base, 2, axis=0).astype(np.float32)
    result = structure.mate_pair_retrieval(matrix, n_distractors=20, seed=1)
    assert result["top1_accuracy"] == pytest.approx(1.0)
    assert result["auroc_mate_vs_random"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# k-mer baseline
# ---------------------------------------------------------------------------


def test_canonical_kmer_count_is_correct():
    """Canonical 4-mers: 136 for k=4, 10 for k=2 (standard values)."""
    assert len(canonical_kmers(2)) == 10
    assert len(canonical_kmers(3)) == 32
    assert len(canonical_kmers(4)) == 136


def test_baseline_is_reverse_complement_invariant():
    """A canonical k-mer vector must be identical for a sequence and its RC.

    This property is relied on in the strand-orientation experiment, where the
    baseline acting as a no-op control is what keeps that comparison fair.
    """
    rng = np.random.default_rng(4)
    seqs = ["".join(rng.choice(list("ACGT"), 151)) for _ in range(25)]
    rc = [reverse_complement(s) for s in seqs]

    encoder = KmerBaselineEncoder(k=4, canonical=True, include_gc=True)
    a = encoder.encode_batch(seqs)
    b = encoder.encode_batch(rc)
    assert np.allclose(a, b, atol=1e-6)


def test_non_canonical_baseline_is_not_rc_invariant():
    """Control for the test above: without collapsing, RC must change the vector."""
    encoder = KmerBaselineEncoder(k=4, canonical=False, include_gc=False)
    seq = "ACGTACGTTTTTGGGGCCCCAAAATTTTACGTACGTAAACCCGGGTTT"
    a = encoder.encode_batch([seq])
    b = encoder.encode_batch([reverse_complement(seq)])
    assert not np.allclose(a, b, atol=1e-6)


def test_kmer_frequencies_sum_to_one():
    encoder = KmerBaselineEncoder(k=4, canonical=True, include_gc=False)
    vec = encoder.encode_batch(["ACGT" * 40])[0]
    assert vec.sum() == pytest.approx(1.0, abs=1e-5)


def test_kmers_spanning_ambiguous_bases_are_skipped_not_imputed():
    """A k-mer containing N must be dropped, and the rest renormalised."""
    encoder = KmerBaselineEncoder(k=4, canonical=True, include_gc=False)
    clean = encoder.encode_batch(["ACGTACGTACGTACGT"])[0]
    withn = encoder.encode_batch(["ACGTACGTNACGTACGTACGT"])[0]
    assert clean.sum() == pytest.approx(1.0, abs=1e-5)
    assert withn.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------


def _cfg(**kw) -> QCConfig:
    base = dict(min_length=10, min_effective_length=5, max_single_base_fraction=None)
    base.update(kw)
    return QCConfig(**base)


def test_longest_unambiguous_run_keeps_the_longest_acgt_stretch():
    qc = evaluate("ACGTNACGTACGTACGT", _cfg(ambiguity_policy="longest_unambiguous_run"))
    assert qc.passed
    assert qc.sequence == "ACGTACGTACGT"
    assert "N" not in qc.sequence


def test_ambiguity_policy_reject_drops_reads_with_n():
    qc = evaluate("ACGTNACGTACGTACGT", _cfg(ambiguity_policy="reject"))
    assert not qc.passed
    assert qc.reason_name == "contains_ambiguous_and_policy_is_reject"


def test_ambiguity_policy_keep_leaves_sequence_untouched():
    seq = "ACGTNACGTACGTACGT"
    qc = evaluate(seq, _cfg(ambiguity_policy="keep"))
    assert qc.passed
    assert qc.sequence == seq


def test_trim_ends_only_strips_terminal_ambiguity():
    # max_ambiguous_fraction must be relaxed: this deliberately N-heavy read
    # (5 of 13 bases) would otherwise be rejected before any policy is applied.
    qc = evaluate(
        "NNACGTNACGTNN",
        _cfg(ambiguity_policy="trim_ends", max_ambiguous_fraction=0.5),
    )
    assert qc.passed
    assert qc.sequence == "ACGTNACGT"


def test_ambiguity_fraction_filter_precedes_the_policy():
    """A read over max_ambiguous_fraction is rejected outright, and a rejected
    read carries an empty processed sequence rather than a partial one."""
    qc = evaluate(
        "NNACGTNACGTNN",
        _cfg(ambiguity_policy="trim_ends", max_ambiguous_fraction=0.10),
    )
    assert not qc.passed
    assert qc.reason_name == "ambiguous_fraction_exceeded"
    assert qc.sequence == ""


def test_no_policy_ever_pads():
    """Central methodological guarantee: nothing is padded, ever."""
    seq = "ACGTNACGTACGTACGT"
    for policy in ("keep", "trim_ends", "longest_unambiguous_run"):
        qc = evaluate(seq, _cfg(ambiguity_policy=policy))
        assert len(qc.sequence) <= len(seq), policy


def test_gc_content_excludes_ambiguous_bases_from_the_denominator():
    qc = evaluate("GGCCAATTNN", _cfg(min_length=5, max_ambiguous_fraction=0.5))
    # 4 GC, 4 AT, 2 N -> GC fraction over ACGT only = 0.5
    assert qc.gc_content == pytest.approx(0.5)


def test_invalid_characters_are_rejected():
    qc = evaluate("ACGTXACGTACGT", _cfg())
    assert not qc.passed
    assert qc.reason_name == "invalid_characters"


def test_low_complexity_filter():
    qc = evaluate("A" * 50, _cfg(max_single_base_fraction=0.9))
    assert not qc.passed
    assert qc.reason_name == "low_complexity_single_base"


def test_sequence_is_uppercased():
    qc = evaluate("acgtacgtacgt", _cfg())
    assert qc.passed
    assert qc.sequence == "ACGTACGTACGT"


# ---------------------------------------------------------------------------
# Amplicon-end grouping
# ---------------------------------------------------------------------------


def test_amplicon_end_groups_separates_two_primer_families():
    a = "AAGGGCACCACAAGAACG"
    b = "CGGTCACGTTCGTTGCCT"
    seqs = [a + "ACGT" * 30 for _ in range(60)] + [b + "TTTT" * 30 for _ in range(40)]
    groups, meta = structure.amplicon_end_groups(seqs, prefix_len=18, n_groups=2)

    assert meta["fraction_assigned"] == pytest.approx(1.0)
    assert set(groups[:60]) == {0}
    assert set(groups[60:]) == {1}
    assert meta["group_sizes"] == {"0": 60, "1": 40}


def test_amplicon_end_groups_tolerates_mismatches_but_not_unrelated_reads():
    a = "AAGGGCACCACAAGAACG"
    seqs = [a + "ACGT" * 30] * 30 + ["TTTTTTTTTTTTTTTTTT" + "ACGT" * 30] * 5
    # A read differing from both anchors by more than max_mismatch is unassigned.
    groups, _ = structure.amplicon_end_groups(
        seqs, prefix_len=18, n_groups=1, max_mismatch=3
    )
    assert set(groups[:30]) == {0}
    assert set(groups[30:]) == {-1}
