"""UMAP projection of stored embeddings.

COST NOTE: UMAP is O(n log n) with a large constant and is run on a
VISUALISATION SUBSET only. The visualisation sample size is a separate,
explicitly reported number and must never be confused with the dataset size or
the encoding subset size.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def run_umap(
    matrix: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int = 42,
    standardize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project to 2D with UMAP. Returns (coordinates, metadata)."""
    import umap

    data = matrix.astype(np.float32)
    if standardize:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1.0
        data = (data - mean) / std

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=seed,  # fixed seed => reproducible layout (single-threaded)
        verbose=False,
    )
    coords = reducer.fit_transform(data)
    return np.asarray(coords), {
        "method": "UMAP",
        "visualization_sample_size": int(matrix.shape[0]),
        "input_dim": int(matrix.shape[1]),
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "standardized": standardize,
        "seed": seed,
    }


def run_tsne(
    matrix: np.ndarray, perplexity: float = 30.0, seed: int = 42
) -> tuple[np.ndarray, dict[str, Any]]:
    """t-SNE projection. Only run when explicitly justified -- it is the most
    expensive projection here and adds little over UMAP for this dataset."""
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=seed, init="pca"
    )
    coords = tsne.fit_transform(matrix.astype(np.float32))
    return np.asarray(coords), {
        "method": "t-SNE",
        "visualization_sample_size": int(matrix.shape[0]),
        "perplexity": perplexity,
        "seed": seed,
    }
