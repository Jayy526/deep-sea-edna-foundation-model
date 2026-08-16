"""Principal component analysis of stored embeddings.

Reads embeddings from disk. The foundation model is never re-run for analysis
or visualisation -- that separation is the whole point of storing embeddings.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA


def run_pca(
    matrix: np.ndarray, n_components: int = 50, seed: int = 42, standardize: bool = True
) -> dict[str, Any]:
    """Fit PCA and report explained variance.

    ``standardize`` centres and scales each dimension first. This matters for a
    fair baseline-vs-foundation comparison: the k-mer vector and the
    foundation embedding live on completely different scales, so PCA on raw
    values would not be comparable between them.
    """
    n_components = int(min(n_components, matrix.shape[0], matrix.shape[1]))
    data = matrix.astype(np.float64)
    if standardize:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1.0
        data = (data - mean) / std

    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(data)

    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)

    def components_for(threshold: float) -> int | None:
        hits = np.nonzero(cumulative >= threshold)[0]
        return int(hits[0]) + 1 if hits.size else None

    return {
        "n_samples": int(matrix.shape[0]),
        "input_dim": int(matrix.shape[1]),
        "n_components": n_components,
        "standardized": standardize,
        "explained_variance_ratio": [round(float(v), 8) for v in ratio],
        "cumulative_explained_variance": [round(float(v), 8) for v in cumulative],
        "variance_pc1": round(float(ratio[0]), 6),
        "variance_pc1_pc2": round(float(cumulative[1]), 6) if n_components > 1 else None,
        "variance_first_10": round(float(cumulative[min(9, n_components - 1)]), 6),
        "variance_captured_by_all_computed": round(float(cumulative[-1]), 6),
        "components_for_50pct": components_for(0.50),
        "components_for_80pct": components_for(0.80),
        "components_for_90pct": components_for(0.90),
        "components_for_95pct": components_for(0.95),
        "effective_dimensionality_participation_ratio": round(
            float((ratio.sum() ** 2) / (ratio**2).sum()), 4
        ),
        "seed": seed,
    }, projected


def intrinsic_dimension_summary(pca_result: dict) -> dict[str, Any]:
    """How much of the nominal dimensionality is actually used?"""
    return {
        "nominal_dimension": pca_result["input_dim"],
        "components_for_90pct_variance": pca_result["components_for_90pct"],
        "participation_ratio": pca_result["effective_dimensionality_participation_ratio"],
        "compression_ratio_at_90pct": (
            round(pca_result["input_dim"] / pca_result["components_for_90pct"], 3)
            if pca_result["components_for_90pct"]
            else None
        ),
    }
