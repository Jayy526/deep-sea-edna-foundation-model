"""Does removing the conserved anchor change the representation comparison?

MOTIVATION
----------
``analysis/amplicon.py`` measured that only ~40% of each read varies across the
community; the rest is a near-invariant conserved anchor. That bounds what any
representation can do, and raises a specific question about the earlier results:

    Is the foundation model's lack of advantage a property of the model, or an
    artefact of feeding it reads that are mostly identical to each other?

This module answers it directly. Each read is trimmed to its **variable region
only** — the conserved anchor removed, per amplicon end, using the changepoint
fitted in ``analysis/amplicon.py`` — and the mate-retrieval comparison is re-run
on exactly the same pairs.

ORDER OF OPERATIONS MATTERS
---------------------------
Amplicon end is identified from the primer-proximal prefix, so trimming must
happen in the ORIGINAL read orientation. Mate 2 is reverse-complemented
afterwards (the strand correction established in stage 1), not before —
otherwise its primer sits at the far end and the prefix lookup fails.

WHAT THIS IS NOT
----------------
Trimming discards real sequence. It is a diagnostic to isolate where the
discriminating signal lives, not a proposed preprocessing default. The
full-read result remains the primary one; this is reported alongside it.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from evaluation.structure import amplicon_end_groups
from preprocessing.run import SUBSET_SCHEMA, subset_path
from utils import config as cfgutil

COMPLEMENT = str.maketrans("ACGT", "TGCA")


def build_variable_region_pairs(
    cfg: dict, amplicon: dict, source: str = "pairs", name: str = "pairs_var"
) -> dict[str, Any]:
    """Trim each mate to its variable region, then reverse-complement mate 2."""
    table = pq.read_table(subset_path(cfg, source))
    data = table.to_pydict()
    sequences = data["sequence"]

    groups, group_meta = amplicon_end_groups(sequences)
    cuts = {
        int(end): info["conserved_block_length"]
        for end, info in amplicon["ends"].items()
    }

    trimmed: list[str] = []
    n_trimmed = 0
    kept_bp = 0
    for i, seq in enumerate(sequences):
        end = int(groups[i])
        cut = cuts.get(end)
        if cut is not None and cut < len(seq):
            seq = seq[cut:]
            n_trimmed += 1
        trimmed.append(seq)
        kept_bp += len(seq)

    # Strand correction AFTER trimming, for the reason in the module docstring.
    data["sequence"] = [
        s.translate(COMPLEMENT)[::-1] if i % 2 == 1 else s
        for i, s in enumerate(trimmed)
    ]
    data["effective_length"] = [len(s) for s in data["sequence"]]

    out = subset_path(cfg, name)
    pq.write_table(
        pa.Table.from_pydict(data, schema=SUBSET_SCHEMA), out, compression="zstd"
    )

    original_bp = sum(len(s) for s in sequences)
    meta = {
        "subset_name": name,
        "source": source,
        "reads": len(trimmed),
        "reads_trimmed": n_trimmed,
        "reads_left_whole_unassigned_end": len(trimmed) - n_trimmed,
        "conserved_block_removed_per_end": cuts,
        "total_bp_before": original_bp,
        "total_bp_after": kept_bp,
        "fraction_of_sequence_retained": round(kept_bp / original_bp, 6),
        "mean_length_after": round(kept_bp / len(trimmed), 2),
        "mate2_reverse_complemented": True,
        "amplicon_end_structure": group_meta,
        "note": (
            "Diagnostic subset. Trimming discards real sequence and is not a "
            "proposed preprocessing default."
        ),
    }
    print(f"[varregion] Wrote {out.name}: {len(trimmed):,} reads, "
          f"{meta['fraction_of_sequence_retained'] * 100:.1f}% of sequence retained, "
          f"mean length {meta['mean_length_after']:.0f} bp")
    return meta
