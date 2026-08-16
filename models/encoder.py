"""The encoder interface.

    SequenceEncoder  (abstract)
        |-- FoundationModelEncoder   -> models/foundation_encoder.py
        |-- BaselineEncoder          -> models/baseline_encoder.py

Any encoder that satisfies this interface can be swapped into the pipeline
without touching preprocessing, embedding storage, or analysis. This is the
seam that lets the paper's exact teacher model be replaced by a different
genomic foundation model later, or vice versa.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


class SequenceEncoder(ABC):
    """Maps DNA strings to fixed-length vectors."""

    #: Short identifier used in filenames and metrics.
    name: str = "encoder"
    #: 'foundation' or 'baseline' -- used to label results.
    kind: str = "unknown"

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of the produced vectors."""

    @abstractmethod
    def encode_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Encode a batch. Returns ``(len(sequences), embedding_dim)`` float32."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Reproducibility metadata: model identity, parameter counts, device, dtype.

        Must report real, measured values -- never placeholders.
        """

    # --- Shared helpers ----------------------------------------------------

    def encode_iter(
        self, sequences: Sequence[str], batch_size: int, progress: bool = True
    ) -> Iterator[np.ndarray]:
        """Yield embedding blocks, one per batch."""
        total = len(sequences)
        iterator = range(0, total, batch_size)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc=f"encode[{self.name}]", unit="batch")
            except ImportError:
                pass
        for start in iterator:
            yield self.encode_batch(sequences[start : start + batch_size])

    def preprocess_sequence(self, sequence: str) -> str:
        """Last transformation applied before encoding. Override to document it."""
        return sequence

    def input_handling(self) -> dict[str, Any]:
        """Exactly what happens to each read before it enters the encoder.

        Required by the short-read handling section of the methodology: no
        encoder in this project may leave this undocumented.
        """
        return {"transformation": "none", "padding": "none"}

    def close(self) -> None:
        """Release resources (GPU memory, file handles)."""
