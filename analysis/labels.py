"""Phase 7 -- taxonomic label verification.

Before any classifier is written, this module answers one question with
evidence: does reliable ground-truth taxonomy exist for these reads?

It checks, in order:
  1. an explicitly configured label file (``classification.labels_path``),
  2. any taxonomic content in the FASTA headers themselves,
  3. any sibling annotation file shipped alongside the FASTA.

If none is found, the honest result is "no labels" and the supervised branch of
the pipeline does not run. Labels are NEVER synthesised from clustering,
from similarity heuristics, or from the reads themselves -- doing so would
manufacture the ground truth that the experiment is supposed to test against.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from preprocessing import fasta_parser
from utils import config as cfgutil

# Header tokens that would indicate embedded taxonomy.
TAXONOMY_HINTS = re.compile(
    r"\b(tax(on|onomy)?[_:=]|taxid|kraken|species|genus|family|phylum|"
    r"superkingdom|sk__|k__|p__|c__|o__|f__|g__|s__|OTU|ASV|lineage)\b",
    re.IGNORECASE,
)

CANDIDATE_SUFFIXES = (".tsv", ".csv", ".txt", ".taxonomy", ".tax", ".json")


def verify(cfg: dict, n_headers: int = 20000) -> dict[str, Any]:
    """Determine whether usable ground-truth taxonomic labels exist."""
    fasta = cfgutil.resolve_path(cfg["paths"]["fasta"])
    result: dict[str, Any] = {
        "question": "Does reliable ground-truth taxonomy exist for these reads?",
        "checks": {},
        "labels_available": False,
        "labels_path": None,
        "n_labelled_reads": 0,
        "n_classes": 0,
    }

    # 1. Explicitly configured label file.
    configured = cfg.get("classification", {}).get("labels_path")
    if configured:
        path = cfgutil.resolve_path(configured)
        result["checks"]["configured_label_file"] = {
            "path": cfgutil.portable_path(path),
            "exists": path.exists(),
        }
        if path.exists():
            labels = load_label_file(path)
            result.update(
                labels_available=len(labels) > 0,
                labels_path=cfgutil.portable_path(path),
                n_labelled_reads=len(labels),
                n_classes=len(set(labels.values())),
            )
            return result
    else:
        result["checks"]["configured_label_file"] = {
            "path": None,
            "exists": False,
            "note": "classification.labels_path is not set in the config.",
        }

    # 2. Taxonomy embedded in FASTA headers.
    headers = fasta_parser.peek_headers(fasta, n_headers)
    hits = [h for h in headers if TAXONOMY_HINTS.search(h)]
    token_counts = {len(h.split()) for h in headers}
    result["checks"]["fasta_headers"] = {
        "headers_inspected": len(headers),
        "headers_matching_taxonomy_pattern": len(hits),
        "example_headers": headers[:3],
        "distinct_token_counts": sorted(token_counts),
        "conclusion": (
            "Headers carry only a run accession, an Illumina cluster coordinate "
            "and a declared length. No taxonomic field is present."
            if not hits
            else "Possible taxonomic content detected; inspect manually before use."
        ),
    }

    # 3. Sibling annotation files next to the FASTA.
    siblings = []
    for directory in {fasta.parent, fasta.parent.parent, cfgutil.PROJECT_ROOT}:
        for child in sorted(directory.glob("*")):
            if child.is_file() and child.suffix.lower() in CANDIDATE_SUFFIXES:
                siblings.append(str(child.name))
    result["checks"]["sibling_annotation_files"] = {
        "searched": sorted(
            {cfgutil.portable_path(d) or "." for d in (fasta.parent, cfgutil.PROJECT_ROOT)}
        ),
        "candidates_found": sorted(set(siblings)),
        "conclusion": "No taxonomic annotation file accompanies the FASTA."
        if not siblings
        else "Candidate files found; none are automatically trusted as ground truth.",
    }

    result["labels_available"] = bool(hits)
    result["consequence"] = (
        "Supervised taxonomic classification cannot be scientifically evaluated on "
        "this dataset. The pipeline therefore completes the embedding stage and "
        "reports unsupervised structural analysis only. Classification metrics, "
        "confusion matrices, training curves, and calibration diagrams are "
        "'Not available with the current dataset/experimental setup.'"
        if not result["labels_available"]
        else "Labels detected -- the supervised branch can run."
    )
    result["what_would_be_needed"] = [
        "A per-read taxonomic assignment table (read_id -> taxon) produced by an "
        "independent reference-based classifier such as Kraken2, MMseqs2 or "
        "Metabuli -- the same tools TaxDistill corrects the output of.",
        "OR a mock-community / CAMI2-style dataset where the true source genome of "
        "every read is known by construction.",
        "OR curated reference amplicon sequences (e.g. SILVA/PR2 for 18S) with "
        "assignments at a stated confidence threshold.",
        "In every case the label source, its version, and its own error rate must "
        "be recorded, because TaxDistill's premise is that these labels are noisy.",
    ]
    return result


def load_label_file(path: str | Path) -> dict[str, str]:
    """Load read_id -> taxon from TSV/CSV/JSON. Two columns: id, label."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return {str(k): str(v) for k, v in json.loads(path.read_text()).items()}
    separator = "," if path.suffix.lower() == ".csv" else "\t"
    labels: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(separator)
            if len(parts) < 2:
                continue
            if line_no == 0 and parts[1].strip().lower() in ("label", "taxon", "taxonomy"):
                continue
            labels[parts[0].strip()] = parts[1].strip()
    return labels
