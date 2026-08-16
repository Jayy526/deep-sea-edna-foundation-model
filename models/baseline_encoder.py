"""Conventional k-mer / tetranucleotide-frequency baseline.

PAPER-DERIVED
    TaxDistill's student branch uses hand-crafted tetranucleotide frequencies
    (TNFs) as its sequence representation, following Taxometer/VAMB. The paper
    describes a 103-dimensional TNF vector (the VAMB projection of canonical
    4-mers onto a basis orthogonal to the reverse-complement and
    frequency-sum constraints), concatenated with per-environment abundances.

EXPERIMENTAL ADAPTATION
    We compute canonical k-mer frequencies directly (k=4 -> 136 dimensions)
    and do NOT apply the VAMB 103-d projection. Reason: that projection exists
    to decorrelate features for VAMB's variational autoencoder; here the vector
    is consumed by PCA and a linear probe, for which the raw canonical
    frequencies are the more faithful and more interpretable baseline.

    We omit the abundance features entirely. Abundance requires multiple
    samples mapped against a common assembly; we have a single sample of
    unassembled reads, so those features cannot be computed. This is stated as
    a limitation rather than filled in with a placeholder.

SHORT-READ NOTE
    A 151 bp read contains 148 4-mers. The resulting frequency vector is far
    noisier than the same vector computed over a 2,000 bp contig (~1,997
    k-mers). This sparsity is an intrinsic property of short reads and is the
    central reason the comparison in this project is interesting.
"""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np

from models.encoder import SequenceEncoder

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(kmer: str) -> str:
    return kmer.translate(COMPLEMENT)[::-1]


def canonical_kmers(k: int) -> list[str]:
    """Sorted list of canonical k-mers (a k-mer and its reverse complement collapse)."""
    seen: set[str] = set()
    for tup in itertools.product("ACGT", repeat=k):
        kmer = "".join(tup)
        seen.add(min(kmer, reverse_complement(kmer)))
    return sorted(seen)


class KmerBaselineEncoder(SequenceEncoder):
    """Canonical k-mer frequency vector, optionally with GC content appended."""

    kind = "baseline"

    def __init__(
        self,
        k: int = 4,
        canonical: bool = True,
        include_gc: bool = True,
        include_length: bool = False,
    ):
        if k < 1 or k > 8:
            raise ValueError("k must be between 1 and 8")
        self.k = k
        self.canonical = canonical
        self.include_gc = include_gc
        self.include_length = include_length
        self.name = f"kmer{k}" + ("_canonical" if canonical else "")

        if canonical:
            self.features = canonical_kmers(k)
        else:
            self.features = ["".join(t) for t in itertools.product("ACGT", repeat=k)]
        self._index = {kmer: i for i, kmer in enumerate(self.features)}
        if canonical:
            for kmer in ["".join(t) for t in itertools.product("ACGT", repeat=k)]:
                self._index[kmer] = self._index[min(kmer, reverse_complement(kmer))]

        self._n_kmer_features = len(self.features)
        self._extra = int(include_gc) + int(include_length)

    @property
    def embedding_dim(self) -> int:
        return self._n_kmer_features + self._extra

    def feature_names(self) -> list[str]:
        names = list(self.features)
        if self.include_gc:
            names.append("gc_content")
        if self.include_length:
            names.append("length")
        return names

    def encode_batch(self, sequences: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(sequences), self.embedding_dim), dtype=np.float32)
        k = self.k
        index = self._index
        n_kmer = self._n_kmer_features

        for row, sequence in enumerate(sequences):
            counts = np.zeros(n_kmer, dtype=np.float32)
            total = 0
            for i in range(len(sequence) - k + 1):
                slot = index.get(sequence[i : i + k])
                if slot is not None:  # skips any k-mer containing a non-ACGT base
                    counts[slot] += 1.0
                    total += 1
            if total:
                counts /= total
            out[row, :n_kmer] = counts

            col = n_kmer
            if self.include_gc:
                gc = sequence.count("G") + sequence.count("C")
                at = sequence.count("A") + sequence.count("T")
                out[row, col] = gc / (gc + at) if (gc + at) else 0.0
                col += 1
            if self.include_length:
                out[row, col] = len(sequence)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "label": "EXPERIMENTAL ADAPTATION (TNF baseline, paper-derived concept)",
            "k": self.k,
            "canonical": self.canonical,
            "embedding_dim": self.embedding_dim,
            "n_kmer_features": self._n_kmer_features,
            "includes_gc_content": self.include_gc,
            "includes_length": self.include_length,
            "model_parameters_total": 0,
            "model_parameters_trainable": 0,
            "model_parameters_frozen": 0,
            "device": "cpu",
            "notes": (
                "Deterministic hand-crafted features; no learned parameters. "
                "Abundance features from the paper are not computable from a "
                "single unassembled sample and are omitted."
            ),
        }

    def input_handling(self) -> dict[str, Any]:
        return {
            "transformation": (
                f"Each QC-passed read is scanned with a sliding window of {self.k} bp. "
                "k-mers containing a non-ACGT base are skipped, not imputed."
            ),
            "padding": "none",
            "length_used": "the read's true length after QC",
            "kmers_per_151bp_read": 151 - self.k + 1,
        }
