"""Label-free evaluation of representation quality.

No ground-truth taxonomy exists for this dataset (see analysis/labels.py), so
supervised metrics -- accuracy, macro F1, confusion matrices, calibration --
are not computable and are reported as such. What IS computable, honestly, is
the following, and it is computed identically for every representation so the
comparison is fair.

1. MATE-PAIR RETRIEVAL (the primary label-free benchmark)
   The two mates of a paired-end cluster are non-overlapping reads from the
   same source DNA fragment, hence the same source organism. Given R1's
   embedding, is R2 ranked above random distractor reads? This is measured
   structure that we did not invent.

   IMPORTANT CONFOUND, stated up front: mates of the same fragment may share
   locus-specific sequence context (in amplicon data, conserved primer-proximal
   regions). A high score therefore demonstrates same-fragment/same-locus
   signal, which is necessary but not sufficient for species-level taxonomy.
   It must not be reported as taxonomic accuracy.

2. CLUSTER TENDENCY vs A NULL MODEL
   Silhouette scores over k-means labels are self-referential on their own, so
   each is paired with the same statistic computed on a dimension-shuffled
   version of the same matrix. The shuffle destroys the correlations between
   dimensions while preserving every marginal distribution, so the gap between
   real and shuffled is the part attributable to genuine structure.

3. INTRINSIC DIMENSIONALITY -- how much of the nominal dimension is used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _standardize(matrix: np.ndarray) -> np.ndarray:
    data = matrix.astype(np.float64)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    return (data - mean) / std


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def amplicon_end_groups(
    sequences: list[str], prefix_len: int = 18, n_groups: int = 2, max_mismatch: int = 4
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign each read to an amplicon-end group from its primer-proximal prefix.

    WHY THIS IS NEEDED (measured, see report §K.2)
    ----------------------------------------------
    This library is amplicon (metabarcoding) data: reads begin with one of a
    small number of conserved primer sequences, and the two mates of a cluster
    are ALWAYS the two opposite ends of the amplicon. A distractor drawn at
    random is therefore ~50% likely to share the QUERY's end rather than the
    mate's, and reads sharing an end share conserved primer sequence. Uncontrolled,
    that confound dominates the retrieval score and inverts it.

    Grouping by prefix lets distractors be matched to the true mate's end, so the
    only thing distinguishing the mate from a distractor is which DNA fragment it
    came from -- which is exactly what the benchmark is supposed to measure.

    This is a technical covariate derived from the sequence data itself, like the
    flowcell tile. It is NOT a taxonomic label and is not used as one.
    """
    from collections import Counter

    counts = Counter(s[:prefix_len] for s in sequences if len(s) >= prefix_len)
    anchors = [prefix for prefix, _ in counts.most_common(n_groups)]

    groups = np.full(len(sequences), -1, dtype=np.int16)
    for i, sequence in enumerate(sequences):
        prefix = sequence[:prefix_len]
        if len(prefix) < prefix_len:
            continue
        best, best_distance = -1, max_mismatch + 1
        for g, anchor in enumerate(anchors):
            distance = sum(a != b for a, b in zip(prefix, anchor))
            if distance < best_distance:
                best, best_distance = g, distance
        groups[i] = best

    assigned = groups >= 0
    return groups, {
        "prefix_len": prefix_len,
        "n_groups": n_groups,
        "max_mismatch": max_mismatch,
        "anchor_prefixes": anchors,
        "anchor_read_counts": [counts[a] for a in anchors],
        "reads_assigned": int(assigned.sum()),
        "reads_unassigned": int((~assigned).sum()),
        "fraction_assigned": round(float(assigned.mean()), 6),
        "group_sizes": {
            str(g): int((groups == g).sum()) for g in range(n_groups)
        },
    }


def mate_pair_retrieval(
    matrix: np.ndarray,
    n_distractors: int = 99,
    seed: int = 42,
    standardize: bool = True,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Can a read's true paired-end mate be retrieved from its embedding?

    ``matrix`` must be laid out so rows ``2i`` and ``2i+1`` are the two mates of
    pair ``i`` (as produced by ``preprocessing.run.build_pair_subset``).

    For each query we rank its true mate against ``n_distractors`` non-mate reads
    by cosine similarity, and report top-1 accuracy, mean reciprocal rank, and
    the separation between mate and non-mate similarity distributions. A
    representation carrying no fragment-level signal scores at chance:
    1/(1 + n_distractors).

    ``groups`` (optional) is a per-ROW amplicon-end assignment from
    ``amplicon_end_groups``. When supplied, distractors are drawn only from reads
    sharing the TRUE MATE's end, which removes the primer-end confound. When
    omitted, distractors are drawn uniformly -- the uncontrolled variant, which
    this dataset shows to be confounded.
    """
    if matrix.shape[0] % 2 != 0:
        raise ValueError("pair matrix must have an even number of rows")
    n_pairs = matrix.shape[0] // 2

    data = _standardize(matrix) if standardize else matrix.astype(np.float64)
    data = _l2_normalize(data)
    left = data[0::2]
    right = data[1::2]

    rng = np.random.default_rng(seed)

    # Candidate pool per query: rows of `right` eligible as distractors.
    controlled = groups is not None
    if controlled:
        right_groups = np.asarray(groups)[1::2]
        pools = {
            g: np.nonzero(right_groups == g)[0]
            for g in np.unique(right_groups)
            if g >= 0
        }
        queries = [i for i in range(n_pairs)
                   if right_groups[i] >= 0 and len(pools[right_groups[i]]) > n_distractors]
    else:
        queries = list(range(n_pairs))

    n_queries = len(queries)
    mate_similarity = np.einsum("ij,ij->i", left[queries], right[queries])

    ranks = np.zeros(n_queries, dtype=np.int64)
    distractor_similarity = np.zeros((n_queries, n_distractors), dtype=np.float64)
    for q, i in enumerate(queries):
        if controlled:
            pool = pools[right_groups[i]]
            candidates = pool[rng.integers(0, len(pool), size=n_distractors)]
        else:
            candidates = rng.integers(0, n_pairs, size=n_distractors)
        candidates[candidates == i] = (candidates[candidates == i] + 1) % n_pairs
        sims = right[candidates] @ left[i]
        distractor_similarity[q] = sims
        ranks[q] = 1 + int((sims > mate_similarity[q]).sum())

    chance = 1.0 / (1 + n_distractors)
    top1 = float((ranks == 1).mean())
    n_pairs = n_queries
    query_ranks = ranks.copy()

    # AUROC of separating true mates from random pairs (rank-based, exact).
    pos = mate_similarity
    neg = distractor_similarity.ravel()
    combined = np.concatenate([pos, neg])
    order = combined.argsort()
    rank_values = np.empty_like(order, dtype=np.float64)
    rank_values[order] = np.arange(1, combined.size + 1)
    # average ties
    auroc = float(
        (rank_values[: pos.size].sum() - pos.size * (pos.size + 1) / 2)
        / (pos.size * neg.size)
    )

    return {
        "benchmark": "paired-end mate retrieval (label-free)",
        "variant": "amplicon-end-controlled" if controlled else "uncontrolled",
        "distractor_sampling": (
            "distractors drawn only from reads sharing the true mate's amplicon end"
            if controlled
            else "distractors drawn uniformly at random (confounded on this dataset)"
        ),
        "n_pairs": n_pairs,
        "n_distractors_per_query": n_distractors,
        "similarity": "cosine on standardized, L2-normalised embeddings",
        "top1_accuracy": round(top1, 6),
        "top5_accuracy": round(float((ranks <= 5).mean()), 6),
        "top10_accuracy": round(float((ranks <= 10).mean()), 6),
        "mean_reciprocal_rank": round(float((1.0 / ranks).mean()), 6),
        "median_rank": float(np.median(ranks)),
        "chance_top1_accuracy": round(chance, 6),
        "lift_over_chance": round(top1 / chance, 3) if chance else None,
        "auroc_mate_vs_random": round(auroc, 6),
        "mean_cosine_true_mates": round(float(pos.mean()), 6),
        "mean_cosine_random_pairs": round(float(neg.mean()), 6),
        "cohens_d": round(
            float(
                (pos.mean() - neg.mean())
                / np.sqrt((pos.var() + neg.var()) / 2)
            ),
            6,
        ),
        "seed": seed,
        "caveat": (
            "Measures same-fragment / same-locus signal, not taxonomic accuracy. "
            "Mates of one fragment may share locus context; this is necessary but "
            "not sufficient evidence of species-level discriminability."
        ),
        # Per-query ranks, so two representations scored on the SAME queries can be
        # compared with a paired test rather than by eyeballing the difference.
        "_query_ranks": query_ranks.tolist(),
        "_query_indices": queries,
    }


def strip_internal(result: dict) -> dict:
    """Drop the per-query arrays before a result is persisted.

    ``mate_pair_retrieval`` carries per-query ranks so paired significance tests
    are possible, but those arrays are tens of thousands of integers and have no
    place in a metrics file. Call this on any result that is about to be saved.
    """
    for key in ("_query_ranks", "_query_indices"):
        result.pop(key, None)
    return result


def compare_retrieval(result_a: dict, result_b: dict, name_a: str, name_b: str) -> dict[str, Any]:
    """Is the difference between two retrieval results real, or noise?

    Both representations are scored on the same queries in the same order, so
    top-1 outcomes are PAIRED. McNemar's exact test on the discordant pairs is
    the correct test; an unpaired comparison of two proportions would overstate
    the uncertainty and ignore that the queries are shared.
    """
    from scipy.stats import binomtest

    ranks_a = np.asarray(result_a["_query_ranks"])
    ranks_b = np.asarray(result_b["_query_ranks"])
    if result_a["_query_indices"] != result_b["_query_indices"]:
        return {"comparable": False,
                "reason": "the two results were scored on different query sets"}

    hit_a = ranks_a == 1
    hit_b = ranks_b == 1
    only_a = int((hit_a & ~hit_b).sum())
    only_b = int((hit_b & ~hit_a).sum())
    discordant = only_a + only_b

    if discordant == 0:
        return {"comparable": True, "identical": True, "n_discordant": 0}

    test = binomtest(only_a, discordant, 0.5)
    difference = float(hit_a.mean() - hit_b.mean())
    # Standard error of the paired difference (McNemar form).
    se = math.sqrt(discordant) / len(ranks_a)

    return {
        "comparable": True,
        "test": "McNemar exact (paired, same queries)",
        "n_queries": int(len(ranks_a)),
        f"top1_{name_a}": round(float(hit_a.mean()), 6),
        f"top1_{name_b}": round(float(hit_b.mean()), 6),
        "difference": round(difference, 6),
        "difference_percentage_points": round(difference * 100, 4),
        "ci95_percentage_points": [
            round((difference - 1.96 * se) * 100, 4),
            round((difference + 1.96 * se) * 100, 4),
        ],
        f"correct_only_by_{name_a}": only_a,
        f"correct_only_by_{name_b}": only_b,
        "p_value": float(test.pvalue),
        "significant_at_0.05": bool(test.pvalue < 0.05),
        "verdict": (
            f"{name_a if difference > 0 else name_b} is better (p={test.pvalue:.2e})"
            if test.pvalue < 0.05
            else f"no significant difference (p={test.pvalue:.3f})"
        ),
    }


def cluster_tendency(
    matrix: np.ndarray,
    k_grid: list[int],
    sample: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """k-means silhouette across k, each against a dimension-shuffled null."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    data = _standardize(matrix)
    if data.shape[0] > sample:
        idx = rng.choice(data.shape[0], sample, replace=False)
        data = data[idx]

    # Null: shuffle each dimension independently. Marginals preserved, joint
    # structure destroyed.
    null = data.copy()
    for j in range(null.shape[1]):
        rng.shuffle(null[:, j])

    results = []
    for k in k_grid:
        if k >= data.shape[0]:
            continue
        row: dict[str, Any] = {"k": k}
        for label, source in (("real", data), ("shuffled_null", null)):
            model = KMeans(n_clusters=k, random_state=seed, n_init=10)
            assignment = model.fit_predict(source)
            row[f"{label}_silhouette"] = round(
                float(silhouette_score(source, assignment, sample_size=min(5000, source.shape[0]), random_state=seed)),
                6,
            )
            row[f"{label}_inertia"] = round(float(model.inertia_), 3)
        row["silhouette_gap_vs_null"] = round(
            row["real_silhouette"] - row["shuffled_null_silhouette"], 6
        )
        results.append(row)

    best = max(results, key=lambda r: r["silhouette_gap_vs_null"]) if results else None
    return {
        "method": "k-means silhouette with a dimension-shuffled null model",
        "analysis_sample_size": int(data.shape[0]),
        "input_dim": int(data.shape[1]),
        "per_k": results,
        "best_k_by_gap": best["k"] if best else None,
        "best_gap": best["silhouette_gap_vs_null"] if best else None,
        "seed": seed,
        "note": (
            "Silhouette on k-means labels is self-referential; only the gap "
            "against the shuffled null is interpretable as genuine structure. "
            "This is a clusterability statistic, NOT classification accuracy, and "
            "the clusters have no verified taxonomic meaning."
        ),
    }


def neighborhood_consistency(
    matrix_a: np.ndarray, matrix_b: np.ndarray, k: int = 20, sample: int = 5000, seed: int = 42
) -> dict[str, Any]:
    """How much do two representations agree about which reads are neighbours?

    Reports mean Jaccard overlap of k-nearest-neighbour sets. A low value means
    the two representations organise the same reads differently -- which is the
    interesting case when neither has labels to be scored against.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n = min(matrix_a.shape[0], matrix_b.shape[0])
    idx = rng.choice(n, min(sample, n), replace=False)
    idx.sort()

    neighbours = []
    for matrix in (matrix_a[:n][idx], matrix_b[:n][idx]):
        model = NearestNeighbors(n_neighbors=k + 1).fit(_standardize(matrix))
        _, indices = model.kneighbors(_standardize(matrix))
        neighbours.append([set(row[1:]) for row in indices])

    overlaps = [
        len(a & b) / len(a | b) for a, b in zip(neighbours[0], neighbours[1])
    ]
    return {
        "metric": "mean Jaccard overlap of k-NN sets between two representations",
        "k": k,
        "sample_size": len(idx),
        "mean_jaccard": round(float(np.mean(overlaps)), 6),
        "median_jaccard": round(float(np.median(overlaps)), 6),
        "interpretation": (
            "1.0 would mean the two representations induce identical local "
            "neighbourhoods; 0.0 means they disagree completely about which "
            "reads are similar."
        ),
        "seed": seed,
    }
