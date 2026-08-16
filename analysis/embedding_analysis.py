"""Phase 6 -- offline embedding analysis.

This stage reads embeddings from disk. It never loads or runs the foundation
model. That separation is deliberate: analysis and visualisation are cheap and
re-runnable, inference is expensive and run once.

SAMPLE SIZES ARE DISTINCT AND ALL REPORTED:
  * dataset size          -- every read in the FASTA (metrics stage)
  * encoding subset size  -- reads pushed through an encoder
  * visualization sample  -- reads shown in a UMAP/PCA scatter
"""

from __future__ import annotations

from typing import Any

import numpy as np

from analysis.pca import intrinsic_dimension_summary, run_pca
from embeddings.store import embedding_health, load_embeddings
from evaluation import structure
from utils import config as cfgutil


def analyse_representation(
    cfg: dict,
    store_name: str,
    label: str,
    run_umap: bool = True,
) -> dict[str, Any]:
    """PCA + projection + cluster tendency for one stored representation."""
    root = cfgutil.output_dir(cfg, "embeddings")
    acfg = cfg["analysis"]
    seed = cfg["seed"]

    ids, matrix, manifest = load_embeddings(root, store_name)
    print(f"[analyse] {label}: {matrix.shape[0]:,} x {matrix.shape[1]}")

    pca_result, projected = run_pca(matrix, acfg["pca_components"], seed)
    result: dict[str, Any] = {
        "representation": label,
        "store_name": store_name,
        "encoder": manifest.get("encoder", {}),
        "n_encoded": int(matrix.shape[0]),
        "embedding_dim": int(matrix.shape[1]),
        "embedding_health": embedding_health(matrix),
        "pca": pca_result,
        "intrinsic_dimensionality": intrinsic_dimension_summary(pca_result),
        "cluster_tendency": structure.cluster_tendency(
            matrix, acfg["kmeans_k_grid"], acfg["silhouette_sample"], seed
        ),
    }

    # Save PCA coordinates so plotting never recomputes them.
    vis_n = min(acfg.get("visualization_sample_size", 20000), matrix.shape[0])
    rng = np.random.default_rng(seed)
    vis_idx = np.sort(rng.choice(matrix.shape[0], vis_n, replace=False))
    np.save(_coord_path(cfg, store_name, "pca"), projected[vis_idx, :2])
    np.save(_coord_path(cfg, store_name, "index"), vis_idx)
    result["visualization_sample_size"] = int(vis_n)

    if run_umap:
        from analysis.umap_analysis import run_umap as umap_project

        coords, meta = umap_project(
            matrix[vis_idx],
            acfg["umap_neighbors"],
            acfg["umap_min_dist"],
            acfg["umap_metric"],
            seed,
        )
        np.save(_coord_path(cfg, store_name, "umap"), coords)
        result["umap"] = meta
    else:
        result["umap"] = "Not available with the current dataset/experimental setup."

    return result


def _coord_path(cfg: dict, store_name: str, kind: str):
    return cfgutil.output_dir(cfg, "embeddings", "coords") / f"{store_name}_{kind}.npy"


def load_coords(cfg: dict, store_name: str, kind: str) -> np.ndarray | None:
    path = _coord_path(cfg, store_name, kind)
    return np.load(path) if path.exists() else None
