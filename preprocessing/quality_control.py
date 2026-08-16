"""Quality control for short reads.

Every filtering decision is configurable and every rejected read is recorded
with an explicit reason code -- nothing is discarded silently.

AMBIGUITY POLICY (IMPLEMENTATION DECISION)
------------------------------------------
GenomeOcean uses a 4096-token BPE vocabulary trained over the ACGT alphabet.
Ambiguous IUPAC codes (N, R, Y, ...) are therefore not representable and would
map to ``[UNK]``. Four policies are provided:

``keep``
    Pass the read through untouched. Ambiguous bases become ``[UNK]`` tokens.
``reject``
    Drop any read containing an ambiguous base.
``trim_ends``
    Strip ambiguous bases from both ends only; interior ambiguity is kept.
``longest_unambiguous_run`` (default)
    Keep the longest contiguous ACGT-only stretch of the read.

The default is ``longest_unambiguous_run`` because it (a) never invents a base,
(b) never pads, and (c) preserves the maximum amount of real, contiguous
genomic signal. It is a lossy operation and the length lost per read is
recorded in the per-read metadata so the cost is auditable.

NOTE ON SHORT READS
-------------------
No policy here pads reads to a target length. The reads are 151 bp and they
enter the encoder as 151 bp (or shorter, after ambiguity handling). See
docs/METHODOLOGY.md for why padding to the paper's 2,000 bp contig threshold
would be scientifically indefensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# Reason codes. 0 == pass. Kept as small ints so 3M rows stay compact.
PASS = 0
REASON_CODES: dict[int, str] = {
    PASS: "pass",
    1: "empty_sequence",
    2: "below_min_length",
    3: "above_max_length",
    4: "invalid_characters",
    5: "ambiguous_fraction_exceeded",
    6: "contains_ambiguous_and_policy_is_reject",
    7: "below_min_effective_length",
    8: "low_complexity_single_base",
    9: "gc_out_of_range",
}

VALID_POLICIES = {"keep", "reject", "trim_ends", "longest_unambiguous_run"}


@dataclass(slots=True)
class QCConfig:
    """Quality-control thresholds. ``None`` disables a filter."""

    min_length: int | None = 100
    max_length: int | None = None
    valid_alphabet: str = "ACGT"
    ambiguity_codes: str = "NRYKMSWBDHV"
    ambiguity_policy: str = "longest_unambiguous_run"
    max_ambiguous_fraction: float | None = 0.10
    min_effective_length: int | None = 100
    max_single_base_fraction: float | None = 0.90
    max_homopolymer_fraction: float | None = 0.90
    min_gc: float | None = None
    max_gc: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "QCConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def __post_init__(self) -> None:
        if self.ambiguity_policy not in VALID_POLICIES:
            raise ValueError(
                f"ambiguity_policy must be one of {sorted(VALID_POLICIES)}, "
                f"got {self.ambiguity_policy!r}"
            )
        self.valid_alphabet = self.valid_alphabet.upper()
        self.ambiguity_codes = self.ambiguity_codes.upper()

    def to_dict(self) -> dict:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass(slots=True)
class QCResult:
    """Outcome of QC for one read."""

    passed: bool
    reason: int
    sequence: str  # the sequence AFTER the ambiguity policy is applied
    raw_length: int
    effective_length: int
    gc_content: float
    n_ambiguous: int
    n_invalid: int
    longest_unambiguous_run: int
    max_base_fraction: float
    ambiguous_positions: tuple[int, ...] = field(default=())

    @property
    def reason_name(self) -> str:
        return REASON_CODES[self.reason]


@lru_cache(maxsize=8)
def _run_regex(alphabet: str) -> re.Pattern[str]:
    return re.compile(f"[{re.escape(alphabet)}]+")


def _longest_run(sequence: str, alphabet: str) -> tuple[int, int, int]:
    """Return (start, end, length) of the longest run of allowed characters.

    Uses a compiled regex rather than a Python-level character loop: this runs
    ~3M times over the full dataset and the difference is minutes.
    """
    best_start = best_end = 0
    best_len = 0
    for match in _run_regex(alphabet).finditer(sequence):
        length = match.end() - match.start()
        if length > best_len:
            best_len = length
            best_start, best_end = match.start(), match.end()
    return best_start, best_end, best_len


def evaluate(sequence: str, cfg: QCConfig, track_positions: bool = False) -> QCResult:
    """Run quality control on a single raw sequence.

    The sequence is uppercased first (case is a formatting artefact, not
    biological information -- lowercase in FASTA conventionally marks
    soft-masked repeats, which we record nowhere because this dataset is
    unmasked raw reads).
    """
    seq = sequence.upper()
    raw_length = len(seq)

    ambiguous = frozenset(cfg.ambiguity_codes)

    # str.count runs in C; a Python character loop here costs minutes over 3M reads.
    counts = {base: seq.count(base) for base in cfg.valid_alphabet}
    n_valid = sum(counts.values())
    n_ambiguous = sum(seq.count(base) for base in cfg.ambiguity_codes)
    n_invalid = raw_length - n_valid - n_ambiguous

    gc = counts.get("G", 0) + counts.get("C", 0)
    at = counts.get("A", 0) + counts.get("T", 0)
    gc_content = gc / (gc + at) if (gc + at) else 0.0

    _, _, longest_run = _longest_run(seq, cfg.valid_alphabet)
    max_base_fraction = (
        max(counts.values(), default=0) / raw_length if raw_length else 0.0
    )

    positions: tuple[int, ...] = ()
    if track_positions and n_ambiguous:
        positions = tuple(i for i, b in enumerate(seq) if b in ambiguous)

    def result(passed: bool, reason: int, out_seq: str) -> QCResult:
        return QCResult(
            passed=passed,
            reason=reason,
            sequence=out_seq,
            raw_length=raw_length,
            effective_length=len(out_seq),
            gc_content=gc_content,
            n_ambiguous=n_ambiguous,
            n_invalid=n_invalid,
            longest_unambiguous_run=longest_run,
            max_base_fraction=max_base_fraction,
            ambiguous_positions=positions,
        )

    # --- Filters applied to the RAW read -----------------------------------
    if raw_length == 0:
        return result(False, 1, "")
    if cfg.min_length is not None and raw_length < cfg.min_length:
        return result(False, 2, "")
    if cfg.max_length is not None and raw_length > cfg.max_length:
        return result(False, 3, "")
    if n_invalid > 0:
        return result(False, 4, "")
    if (
        cfg.max_ambiguous_fraction is not None
        and (n_ambiguous / raw_length) > cfg.max_ambiguous_fraction
    ):
        return result(False, 5, "")
    if cfg.ambiguity_policy == "reject" and n_ambiguous > 0:
        return result(False, 6, "")
    if (
        cfg.max_single_base_fraction is not None
        and max_base_fraction > cfg.max_single_base_fraction
    ):
        return result(False, 8, "")
    if cfg.min_gc is not None and gc_content < cfg.min_gc:
        return result(False, 9, "")
    if cfg.max_gc is not None and gc_content > cfg.max_gc:
        return result(False, 9, "")

    # --- Apply the ambiguity policy ----------------------------------------
    out = _apply_policy(seq, cfg)

    # --- Filters applied to the PROCESSED read ------------------------------
    if cfg.min_effective_length is not None and len(out) < cfg.min_effective_length:
        return result(False, 7, out)

    return result(True, PASS, out)


def _apply_policy(seq: str, cfg: QCConfig) -> str:
    policy = cfg.ambiguity_policy
    if policy in ("keep", "reject"):
        # 'reject' reads that reach here contain no ambiguity, so this is a no-op.
        return seq
    if policy == "trim_ends":
        valid = frozenset(cfg.valid_alphabet)
        start, end = 0, len(seq)
        while start < end and seq[start] not in valid:
            start += 1
        while end > start and seq[end - 1] not in valid:
            end -= 1
        return seq[start:end]
    if policy == "longest_unambiguous_run":
        start, end, _ = _longest_run(seq, cfg.valid_alphabet)
        return seq[start:end]
    raise ValueError(f"Unhandled ambiguity policy: {policy!r}")


def describe_config(cfg: QCConfig) -> dict[str, Any]:
    """A documentation-grade description of the active QC settings."""
    return {
        "thresholds": cfg.to_dict(),
        "reason_codes": REASON_CODES,
        "notes": {
            "padding": "No padding is applied at any point. Reads enter the encoder at their true length.",
            "ambiguity_policy": (
                "longest_unambiguous_run keeps the longest contiguous ACGT stretch; "
                "it never substitutes or invents a base."
            ),
            "case": "Sequences are uppercased; case carries no information in this dataset.",
        },
    }
