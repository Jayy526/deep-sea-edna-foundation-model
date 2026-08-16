"""Phase 7 -- baseline vs foundation model.

FAIRNESS PROTOCOL
-----------------
Both representations are computed from:
  * the same reads (the same subset Parquet file),
  * after the same quality control,
  * with the same seed,
  * and are evaluated with the same metrics and the same sampling.

No ground-truth taxonomy exists for this dataset, so the accuracy / macro-F1 /
weighted-F1 table specified for the labelled case is not computable and is
reported as unavailable rather than filled in. The comparison below uses only
metrics that are valid without labels.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from embeddings.store import load_embeddings
from evaluation import structure
from utils import config as cfgutil
from utils.runtime import NOT_AVAILABLE


def compare(
    cfg: dict,
    foundation_store: str,
    baseline_store: str,
    foundation_pairs_store: str,
    baseline_pairs_store: str,
) -> dict[str, Any]:
    """Build the full baseline-vs-foundation comparison."""
    root = cfgutil.output_dir(cfg, "embeddings")
    metrics_dir = cfgutil.output_dir(cfg, "metrics")
    acfg = cfg["analysis"]
    seed = cfg["seed"]

    _, foundation, f_manifest = load_embeddings(root, foundation_store)
    _, baseline, b_manifest = load_embeddings(root, baseline_store)
    _, foundation_pairs, _ = load_embeddings(root, foundation_pairs_store)
    _, baseline_pairs, _ = load_embeddings(root, baseline_pairs_store)

    def encoder_metrics(store: str) -> dict:
        path = metrics_dir / f"encoder_metrics_{store}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    f_enc = encoder_metrics(foundation_store)
    b_enc = encoder_metrics(baseline_store)

    # Amplicon-end groups for the paired subset, derived from primer-proximal
    # prefixes. Needed to de-confound the retrieval benchmark -- see
    # evaluation/structure.amplicon_end_groups.
    import pyarrow.parquet as pq

    # Groups always come from the ORIGINAL read orientation, so the control is
    # identical regardless of which pair store is being scored.
    pair_sequences = (
        pq.read_table(
            cfgutil.output_dir(cfg, "embeddings") / "subset_pairs.parquet",
            columns=["sequence"],
        )
        .column("sequence")
        .to_pylist()
    )
    groups, group_meta = structure.amplicon_end_groups(pair_sequences)

    n_distractors = acfg.get("retrieval_distractors", 99)
    result: dict[str, Any] = {
        "protocol": {
            "same_reads": f_manifest.get("source_subset") == b_manifest.get("source_subset"),
            "subset_file": f_manifest.get("source_subset"),
            "reads_compared": int(min(foundation.shape[0], baseline.shape[0])),
            "pair_reads_compared": int(
                min(foundation_pairs.shape[0], baseline_pairs.shape[0])
            ),
            "seed": seed,
            "note": "Identical reads, identical QC, identical seed, identical metrics.",
        },
        "supervised_metrics": {
            "accuracy": NOT_AVAILABLE,
            "macro_precision": NOT_AVAILABLE,
            "macro_recall": NOT_AVAILABLE,
            "macro_f1": NOT_AVAILABLE,
            "weighted_f1": NOT_AVAILABLE,
            "per_class_metrics": NOT_AVAILABLE,
            "reason": (
                "No ground-truth taxonomic labels exist for this dataset "
                "(see outputs/metrics/label_verification.json). Fabricating labels "
                "in order to populate this table would invalidate the experiment."
            ),
        },
        "amplicon_end_structure": group_meta,
        "representations": {},
        "mate_pair_retrieval": {},
        "mate_pair_retrieval_uncontrolled": {},
        "agreement": structure.neighborhood_consistency(
            foundation, baseline, acfg.get("knn_k", 20), 5000, seed
        ),
    }

    for label, matrix, pairs, enc, manifest in (
        ("foundation_model", foundation, foundation_pairs, f_enc, f_manifest),
        ("baseline", baseline, baseline_pairs, b_enc, b_manifest),
    ):
        result["representations"][label] = {
            "name": manifest.get("encoder", {}).get("name"),
            "kind": manifest.get("encoder", {}).get("kind"),
            "feature_dimension": int(matrix.shape[1]),
            "model_parameters_total": manifest.get("encoder", {}).get(
                "model_parameters_total"
            ),
            "model_parameters_frozen": manifest.get("encoder", {}).get(
                "model_parameters_frozen"
            ),
            "model_parameters_trainable": manifest.get("encoder", {}).get(
                "model_parameters_trainable"
            ),
            "device": manifest.get("encoder", {}).get("device"),
            "encoding_time_seconds": enc.get("processing_time_seconds"),
            "sequences_per_second": enc.get("sequences_per_second"),
            "milliseconds_per_sequence": enc.get("milliseconds_per_sequence"),
            "storage_mb": enc.get("embedding_storage_mb"),
            "bytes_per_embedding": enc.get("bytes_per_embedding"),
        }
        # Primary benchmark: distractors matched to the true mate's amplicon end.
        result["mate_pair_retrieval"][label] = structure.strip_internal(
            structure.mate_pair_retrieval(pairs, n_distractors, seed, groups=groups)
        )
        # Kept for transparency: the uncontrolled variant, whose sub-chance score
        # is what exposed the primer-end confound in the first place.
        result["mate_pair_retrieval_uncontrolled"][label] = structure.strip_internal(
            structure.mate_pair_retrieval(pairs, n_distractors, seed)
        )

    # Head-to-head deltas, computed from the measured values above.
    f_ret = result["mate_pair_retrieval"]["foundation_model"]
    b_ret = result["mate_pair_retrieval"]["baseline"]
    f_rep = result["representations"]["foundation_model"]
    b_rep = result["representations"]["baseline"]

    result["head_to_head"] = {
        "mate_retrieval_top1": {
            "foundation_model": f_ret["top1_accuracy"],
            "baseline": b_ret["top1_accuracy"],
            "chance": f_ret["chance_top1_accuracy"],
            "winner": _winner(f_ret["top1_accuracy"], b_ret["top1_accuracy"]),
        },
        "mate_retrieval_auroc": {
            "foundation_model": f_ret["auroc_mate_vs_random"],
            "baseline": b_ret["auroc_mate_vs_random"],
            "chance": 0.5,
            "winner": _winner(
                f_ret["auroc_mate_vs_random"], b_ret["auroc_mate_vs_random"]
            ),
        },
        "mean_reciprocal_rank": {
            "foundation_model": f_ret["mean_reciprocal_rank"],
            "baseline": b_ret["mean_reciprocal_rank"],
            "winner": _winner(
                f_ret["mean_reciprocal_rank"], b_ret["mean_reciprocal_rank"]
            ),
        },
        "feature_dimension": {
            "foundation_model": f_rep["feature_dimension"],
            "baseline": b_rep["feature_dimension"],
            "ratio": round(
                f_rep["feature_dimension"] / b_rep["feature_dimension"], 3
            ),
        },
        "inference_cost": {
            "foundation_model_seq_per_second": f_rep["sequences_per_second"],
            "baseline_seq_per_second": b_rep["sequences_per_second"],
            "baseline_speedup_factor": round(
                b_rep["sequences_per_second"] / f_rep["sequences_per_second"], 2
            )
            if f_rep["sequences_per_second"]
            else None,
        },
        "storage_cost": {
            "foundation_model_bytes_per_embedding": f_rep["bytes_per_embedding"],
            "baseline_bytes_per_embedding": b_rep["bytes_per_embedding"],
        },
    }
    return result


def _winner(foundation_value: float, baseline_value: float, tolerance: float = 1e-6) -> str:
    """Name the better representation, or 'tie'. No claim beyond the measurement."""
    if abs(foundation_value - baseline_value) <= tolerance:
        return "tie"
    return "foundation_model" if foundation_value > baseline_value else "baseline"


def comparison_table(result: dict) -> "list[dict]":
    """Flatten the comparison into rows for outputs/metrics/comparison.csv."""
    f_rep = result["representations"]["foundation_model"]
    b_rep = result["representations"]["baseline"]
    f_ret = result["mate_pair_retrieval"]["foundation_model"]
    b_ret = result["mate_pair_retrieval"]["baseline"]

    rows = [
        ("Accuracy (taxonomic)", NOT_AVAILABLE, NOT_AVAILABLE),
        ("Macro precision", NOT_AVAILABLE, NOT_AVAILABLE),
        ("Macro recall", NOT_AVAILABLE, NOT_AVAILABLE),
        ("Macro F1", NOT_AVAILABLE, NOT_AVAILABLE),
        ("Weighted F1", NOT_AVAILABLE, NOT_AVAILABLE),
        ("Mate retrieval top-1 accuracy", b_ret["top1_accuracy"], f_ret["top1_accuracy"]),
        ("Mate retrieval top-5 accuracy", b_ret["top5_accuracy"], f_ret["top5_accuracy"]),
        ("Mate retrieval MRR", b_ret["mean_reciprocal_rank"], f_ret["mean_reciprocal_rank"]),
        ("Mate vs random AUROC", b_ret["auroc_mate_vs_random"], f_ret["auroc_mate_vs_random"]),
        ("Mate/random effect size (Cohen's d)", b_ret["cohens_d"], f_ret["cohens_d"]),
        ("Feature dimension", b_rep["feature_dimension"], f_rep["feature_dimension"]),
        ("Model parameters", b_rep["model_parameters_total"], f_rep["model_parameters_total"]),
        ("Trainable parameters", b_rep["model_parameters_trainable"], f_rep["model_parameters_trainable"]),
        ("Training time (s) - no training performed; both are fixed extractors", 0, 0),
        ("Inference throughput (seq/s)", b_rep["sequences_per_second"], f_rep["sequences_per_second"]),
        ("Inference time per read (ms)", b_rep["milliseconds_per_sequence"], f_rep["milliseconds_per_sequence"]),
        ("Storage per embedding (bytes)", b_rep["bytes_per_embedding"], f_rep["bytes_per_embedding"]),
        ("Compute device", b_rep["device"], f_rep["device"]),
    ]
    return [
        {"metric": name, "baseline_kmer_tnf": base, "foundation_model_genomeocean": found}
        for name, base, found in rows
    ]
