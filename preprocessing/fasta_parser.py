"""Streaming FASTA parser.

IMPLEMENTATION DECISION
-----------------------
The dataset is ~650 MB / ~3M records. It is never loaded into RAM in full.
Records are produced by a generator that holds at most one record at a time;
callers consume them in configurable batches.

The parser is deliberately tolerant: it does not validate sequence content.
Validation and filtering are the job of ``preprocessing.quality_control`` so
that every filtering decision lives in exactly one, configurable place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

# Illumina read name: <instrument>:<run>:<flowcell>:<lane>:<tile>:<x>:<y>
ILLUMINA_NAME = re.compile(
    r"^(?P<instrument>[^:]+):(?P<run>\d+):(?P<flowcell>[^:]+):"
    r"(?P<lane>\d+):(?P<tile>\d+):(?P<x>\d+):(?P<y>\d+)$"
)
LENGTH_FIELD = re.compile(r"\blength=(\d+)\b")


@dataclass(slots=True)
class FastaRecord:
    """One FASTA record, with the header kept intact.

    ``header`` is preserved verbatim so no information is silently discarded.
    """

    index: int
    seq_id: str
    header: str
    sequence: str

    @property
    def description(self) -> str:
        parts = self.header.split(None, 1)
        return parts[1] if len(parts) > 1 else ""

    def header_fields(self) -> dict:
        """Parse the structured parts of the header we can identify.

        Returns a dict with ``lane``/``tile``/``x``/``y``/``declared_length``
        where parseable, and ``None`` where the header does not follow the
        Illumina convention. Nothing is invented.
        """
        fields: dict = {
            "lane": None,
            "tile": None,
            "x": None,
            "y": None,
            "declared_length": None,
            "instrument": None,
            "flowcell": None,
        }
        tokens = self.header.split()
        for token in tokens[1:]:
            match = ILLUMINA_NAME.match(token)
            if match:
                fields["instrument"] = match.group("instrument")
                fields["flowcell"] = match.group("flowcell")
                fields["lane"] = int(match.group("lane"))
                fields["tile"] = int(match.group("tile"))
                fields["x"] = int(match.group("x"))
                fields["y"] = int(match.group("y"))
                break
        length_match = LENGTH_FIELD.search(self.header)
        if length_match:
            fields["declared_length"] = int(length_match.group(1))
        return fields

    @property
    def accession(self) -> str:
        """The run accession portion of the ID, e.g. ``SRR26872904`` from ``SRR26872904.1``."""
        return self.seq_id.rsplit(".", 1)[0]


def iter_records(
    path: str | Path,
    max_records: int | None = None,
    encoding: str = "utf-8",
) -> Iterator[FastaRecord]:
    """Stream FASTA records one at a time.

    Multi-line sequences are joined. Blank lines are ignored. A record with a
    header but no sequence lines is still yielded (with an empty sequence) so
    that quality control -- not the parser -- decides its fate.
    """
    path = Path(path)
    index = 0
    header: str | None = None
    chunks: list[str] = []

    with open(path, "r", encoding=encoding, newline=None) as handle:
        for line in handle:
            if not line:
                continue
            if line[0] == ">":
                if header is not None:
                    yield _build(index, header, chunks)
                    index += 1
                    if max_records is not None and index >= max_records:
                        return
                header = line[1:].strip()
                chunks = []
            else:
                stripped = line.strip()
                if stripped:
                    chunks.append(stripped)
        if header is not None:
            if max_records is None or index < max_records:
                yield _build(index, header, chunks)


def _build(index: int, header: str, chunks: list[str]) -> FastaRecord:
    seq_id = header.split(None, 1)[0] if header else ""
    return FastaRecord(
        index=index, seq_id=seq_id, header=header, sequence="".join(chunks)
    )


def iter_batches(
    path: str | Path,
    batch_size: int,
    max_records: int | None = None,
) -> Iterator[list[FastaRecord]]:
    """Stream records in fixed-size batches (the last batch may be shorter)."""
    batch: list[FastaRecord] = []
    for record in iter_records(path, max_records=max_records):
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def batched(items: Iterable, batch_size: int) -> Iterator[list]:
    """Generic batching helper for any iterable."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def peek_headers(path: str | Path, n: int = 10) -> list[str]:
    """Return the first ``n`` headers -- used to inspect, not assume, header format."""
    headers = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                headers.append(line[1:].strip())
                if len(headers) >= n:
                    break
    return headers
