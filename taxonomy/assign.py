"""Assign reference-derived taxonomy to our sequence variants.

The output is a label table for the supervised branch. Those labels are
**noisy, reference-derived assignments, not ground truth** -- the distinction is
carried through every downstream report.

Two things are measured rather than assumed:

1. **Classifier accuracy**, by k-fold cross-validation on the reference itself.
   This is the honest estimate of how wrong the labels are before they are used
   for anything.
2. **Assignment rate**, i.e. what fraction of our variants can be assigned at
   all above the confidence threshold. Deep-sea benthic Foraminifera are poorly
   represented in reference databases, so an incomplete assignment is the
   expected outcome and is reported, not hidden.

Only variants from amplicon end 0 are classified: those are the ones carrying
the s14F1 primer that the reference window is anchored on. End-1 variants come
from the opposite (s15) end of the amplicon and would need their own
region-matched reference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analysis.dereplicate import variants_path
from evaluation.structure import amplicon_end_groups
from taxonomy.classifier import NaiveBayesClassifier, assignment_summary
from taxonomy.reference import load_reference
from utils import config as cfgutil

LABEL_SCHEMA = pa.schema(
    [
        ("seq_id", pa.string()),
        ("variant_id", pa.int32()),
        ("label", pa.string()),
        ("confidence", pa.float32()),
        ("read_count", pa.int32()),
        ("assigned", pa.bool_()),
    ]
)


def cross_validate(
    sequences: list[str], labels: list[str], k: int, folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Honest accuracy estimate for the labels we are about to generate."""
    rng = np.random.default_rng(seed)
    seqs = np.array(sequences, dtype=object)
    labs = np.array(labels, dtype=object)
    order = rng.permutation(len(seqs))

    correct = total = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    for fold in np.array_split(order, folds):
        mask = np.ones(len(seqs), dtype=bool)
        mask[fold] = False
        model = NaiveBayesClassifier(k=k, seed=seed)
        model.fit(list(seqs[mask]), list(labs[mask]))
        predictions, confidences = model.classify(list(seqs[fold]), n_bootstrap=50)
        for prediction, confidence, truth in zip(predictions, confidences, labs[fold]):
            total += 1
            if prediction == truth:
                correct += 1
                conf_correct.append(float(confidence))
            else:
                conf_wrong.append(float(confidence))

    return {
        "method": f"{folds}-fold cross-validation on the reference database",
        "n": total,
        "accuracy": round(correct / total, 6),
        "error_rate": round(1 - correct / total, 6),
        "mean_confidence_when_correct": round(float(np.mean(conf_correct)), 4),
        "mean_confidence_when_wrong": round(float(np.mean(conf_wrong)), 4)
        if conf_wrong
        else None,
        "note": (
            "Measured on reference sequences, which are cleaner and better "
            "represented than environmental reads. Real label error on our "
            "variants is expected to be HIGHER than this figure."
        ),
    }


def assign(cfg: dict, force: bool = False, rank: str | None = None,
           threshold: float | None = None) -> dict[str, Any]:
    """Classify our variants and write a label table."""
    tcfg = cfg["taxonomy"]
    rank = rank or tcfg["rank"]
    threshold = threshold if threshold is not None else tcfg["confidence_threshold"]
    out_path = cfgutil.resolve_path(f"data/reference/variant_labels_{rank}.parquet")

    ref_seqs, ref_labels = load_reference(cfg, rank)
    keep = [i for i, label in enumerate(ref_labels) if label not in ("", "NA")]
    ref_seqs = [ref_seqs[i] for i in keep]
    ref_labels = [ref_labels[i] for i in keep]

    validation = cross_validate(ref_seqs, ref_labels, tcfg["kmer_size"], seed=cfg["seed"])
    print(f"[assign] classifier CV accuracy at {rank}: {validation['accuracy']:.4f} "
          f"({validation['n']:,} reference sequences, {len(set(ref_labels))} classes)")

    table = pq.read_table(variants_path(cfg), columns=["variant_id", "sequence", "count"])
    variant_ids = table.column("variant_id").to_pylist()
    sequences = table.column("sequence").to_pylist()
    counts = table.column("count").to_pylist()

    groups, _ = amplicon_end_groups(sequences)
    target_end = tcfg["amplicon_end"]
    selected = [i for i in range(len(sequences)) if groups[i] == target_end]
    print(f"[assign] classifying {len(selected):,} variants from amplicon end {target_end} "
          f"(of {len(sequences):,} total)")

    model = NaiveBayesClassifier(k=tcfg["kmer_size"], seed=cfg["seed"])
    training = model.fit(ref_seqs, ref_labels)
    predictions, confidences = model.classify(
        [sequences[i] for i in selected], n_bootstrap=tcfg["bootstrap"]
    )

    assigned = confidences >= threshold
    rows = {
        "seq_id": [f"variant_{variant_ids[i]}" for i in selected],
        "variant_id": [variant_ids[i] for i in selected],
        "label": list(predictions),
        "confidence": confidences.tolist(),
        "read_count": [counts[i] for i in selected],
        "assigned": assigned.tolist(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict(rows, schema=LABEL_SCHEMA), out_path, compression="zstd"
    )

    reads_total = sum(rows["read_count"])
    reads_assigned = sum(c for c, a in zip(rows["read_count"], assigned) if a)
    summary = assignment_summary(predictions, confidences, threshold)

    metrics: dict[str, Any] = {
        "stage": "assign_taxonomy",
        "label_status": (
            "NOISY REFERENCE-DERIVED LABELS -- NOT GROUND TRUTH. Generated by a "
            "naive Bayes classifier against a 1,545-sequence PR2 reference."
        ),
        "rank": rank,
        "reference_training": training,
        "classifier_validation": validation,
        "amplicon_end_classified": target_end,
        "variants_total": len(sequences),
        "variants_considered": len(selected),
        "assignment": summary,
        "reads_represented_by_considered_variants": reads_total,
        "reads_assigned": reads_assigned,
        "read_weighted_assignment_rate": round(reads_assigned / reads_total, 6)
        if reads_total
        else 0,
        "labels_path": str(out_path),
    }
    cfgutil.save_json(
        metrics, cfgutil.output_dir(cfg, "metrics") / f"taxonomy_metrics_{rank}.json"
    )
    print(f"[assign] assigned {summary['n_assigned']:,}/{summary['n_queries']:,} variants "
          f"({summary['assignment_rate'] * 100:.1f}%) at confidence >= {threshold}; "
          f"{summary['n_distinct_classes_assigned']} distinct {rank} labels")
    print(f"[assign] read-weighted assignment rate: "
          f"{metrics['read_weighted_assignment_rate'] * 100:.1f}%")
    return metrics


def load_labels(cfg: dict, rank: str | None = None, assigned_only: bool = True) -> dict[str, str]:
    """seq_id -> label, for the supervised branch."""
    rank = rank or cfg["taxonomy"]["rank"]
    path = cfgutil.resolve_path(f"data/reference/variant_labels_{rank}.parquet")
    table = pq.read_table(path)
    ids = table.column("seq_id").to_pylist()
    labels = table.column("label").to_pylist()
    flags = table.column("assigned").to_pylist()
    return {
        i: l for i, l, a in zip(ids, labels, flags) if (a or not assigned_only)
    }
