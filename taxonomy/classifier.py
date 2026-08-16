"""RDP-style naive Bayes k-mer classifier with bootstrap confidence.

This is the standard algorithm for assigning taxonomy to short rRNA amplicons
(Wang et al. 2007, the RDP Classifier), reimplemented here so the whole pipeline
stays dependency-light and auditable.

METHOD
------
A sequence is represented by the SET of k-mer words it contains (presence, not
count). For taxon ``g`` with ``n_g`` reference sequences, and word ``w`` present
in ``m(w,g)`` of them:

    P(w | g) = (m(w,g) + P_w) / (n_g + 1)          where P_w = (n_w + 0.5) / (N + 1)

``P_w`` is the corpus-wide prior for that word, which is the Bayesian smoothing
that stops an unseen word from zeroing a taxon outright.

Classification takes the argmax over taxa of the summed log-likelihood.

CONFIDENCE
----------
The bootstrap is the whole point of this algorithm and the reason it is used
instead of a plain best-hit search: resample k of the query's words with
replacement, reclassify, repeat. The fraction of resamples that return the
winning taxon is the confidence. Assignments below a threshold are reported as
UNASSIGNED rather than forced -- which matters enormously here, because the
reference database is small and many queries genuinely have no close relative
in it.

WHAT THE OUTPUT IS
------------------
Noisy, reference-derived labels. NOT ground truth. This is exactly the kind of
label TaxDistill takes as input and sets out to correct.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BASES = {"A": 0, "C": 1, "G": 2, "T": 3}


def encode_words(sequence: str, k: int) -> np.ndarray:
    """Unique k-mer word indices present in a sequence.

    k-mers containing a non-ACGT base are skipped rather than imputed.
    """
    n = len(sequence)
    if n < k:
        return np.empty(0, dtype=np.int64)
    values = np.full(n, -1, dtype=np.int64)
    for base, code in BASES.items():
        values[np.frombuffer(sequence.encode(), dtype=np.uint8) == ord(base)] = code

    words = np.zeros(n - k + 1, dtype=np.int64)
    valid = np.ones(n - k + 1, dtype=bool)
    for offset in range(k):
        slice_ = values[offset : offset + n - k + 1]
        valid &= slice_ >= 0
        words = words * 4 + np.maximum(slice_, 0)
    return np.unique(words[valid])


class NaiveBayesClassifier:
    """Word-presence naive Bayes over taxa, with bootstrap confidence."""

    def __init__(self, k: int = 8, seed: int = 42):
        self.k = k
        self.n_words = 4**k
        self.seed = seed
        self.classes: list[str] = []
        self._log_prob: np.ndarray | None = None

    def fit(self, sequences: list[str], labels: list[str]) -> dict[str, Any]:
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        n_classes = len(self.classes)

        presence = np.zeros((n_classes, self.n_words), dtype=np.float32)
        class_counts = np.zeros(n_classes, dtype=np.float64)
        word_totals = np.zeros(self.n_words, dtype=np.float64)

        for sequence, label in zip(sequences, labels):
            words = encode_words(sequence, self.k)
            row = index[label]
            presence[row, words] += 1.0
            class_counts[row] += 1.0
            word_totals[words] += 1.0

        n_total = float(len(sequences))
        word_prior = (word_totals + 0.5) / (n_total + 1.0)
        # P(w|g) = (m(w,g) + P_w) / (n_g + 1)
        probabilities = (presence + word_prior[None, :]) / (class_counts[:, None] + 1.0)
        self._log_prob = np.log(probabilities, dtype=np.float32)

        return {
            "k": self.k,
            "n_words": self.n_words,
            "n_reference_sequences": len(sequences),
            "n_classes": n_classes,
            "smallest_class_size": int(class_counts.min()),
            "largest_class_size": int(class_counts.max()),
            "median_class_size": float(np.median(class_counts)),
            "classes_with_one_reference": int((class_counts == 1).sum()),
            "model_bytes": int(self._log_prob.nbytes),
        }

    def classify(
        self, sequences: list[str], n_bootstrap: int = 100, batch: int = 512
    ) -> tuple[list[str], np.ndarray]:
        """Return (predicted class per query, bootstrap confidence per query)."""
        if self._log_prob is None:
            raise RuntimeError("fit() must be called before classify()")
        rng = np.random.default_rng(self.seed)
        predictions: list[str] = []
        confidences = np.zeros(len(sequences), dtype=np.float32)

        for start in range(0, len(sequences), batch):
            for offset, sequence in enumerate(sequences[start : start + batch]):
                i = start + offset
                words = encode_words(sequence, self.k)
                if words.size == 0:
                    predictions.append("")
                    continue
                scores = self._log_prob[:, words].sum(axis=1)
                best = int(scores.argmax())
                # Bootstrap: 1/8 of the words, as in the RDP Classifier.
                size = max(1, words.size // 8)
                draws = rng.integers(0, words.size, size=(n_bootstrap, size))
                boot = self._log_prob[:, words[draws]].sum(axis=2)  # (classes, boot)
                agree = (boot.argmax(axis=0) == best).mean()
                predictions.append(self.classes[best])
                confidences[i] = agree
        return predictions, confidences


def assignment_summary(
    predictions: list[str], confidences: np.ndarray, threshold: float
) -> dict[str, Any]:
    assigned = confidences >= threshold
    from collections import Counter

    counts = Counter(p for p, a in zip(predictions, assigned) if a)
    return {
        "n_queries": len(predictions),
        "confidence_threshold": threshold,
        "n_assigned": int(assigned.sum()),
        "n_unassigned": int((~assigned).sum()),
        "assignment_rate": round(float(assigned.mean()), 6),
        "n_distinct_classes_assigned": len(counts),
        "mean_confidence": round(float(confidences.mean()), 6),
        "median_confidence": round(float(np.median(confidences)), 6),
        "confidence_deciles": [
            round(float(np.percentile(confidences, p)), 4) for p in range(0, 101, 10)
        ],
        "top_classes": dict(counts.most_common(15)),
    }
