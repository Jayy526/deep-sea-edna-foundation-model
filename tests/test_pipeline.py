"""Correctness tests for the streaming and storage components.

These are the pieces where an optimisation could silently lose or miscount
data: the FASTA parser, the digest-based dereplication (which counts 8-byte
hashes rather than sequences), and the sharded embedding store with its
resume logic. Each is checked against a naive, obviously-correct
implementation over a small synthetic dataset.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.dereplicate import dereplicate, reverse_complement
from embeddings.store import EmbeddingStore, embedding_health, load_embeddings
from preprocessing import fasta_parser
from preprocessing.quality_control import QCConfig, evaluate
from utils import config as cfgutil


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------


def _write_fasta(path: Path, records: list[tuple[str, str]], wrap: int = 0) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for header, seq in records:
            handle.write(f">{header}\n")
            if wrap:
                for i in range(0, len(seq), wrap):
                    handle.write(seq[i : i + wrap] + "\n")
            else:
                handle.write(seq + "\n")


def test_parser_reads_every_record(tmp_path):
    records = [(f"read{i} extra field", "ACGT" * (i + 1)) for i in range(50)]
    path = tmp_path / "t.fasta"
    _write_fasta(path, records)

    parsed = list(fasta_parser.iter_records(path))
    assert len(parsed) == 50
    assert [r.seq_id for r in parsed] == [f"read{i}" for i in range(50)]
    assert [r.sequence for r in parsed] == [s for _, s in records]


def test_parser_joins_multiline_sequences(tmp_path):
    """The real dataset wraps sequences at 70 columns."""
    seq = "ACGT" * 40
    path = tmp_path / "wrapped.fasta"
    _write_fasta(path, [("r1", seq)], wrap=70)
    parsed = list(fasta_parser.iter_records(path))
    assert len(parsed) == 1
    assert parsed[0].sequence == seq


def test_parser_batches_cover_all_records_exactly_once(tmp_path):
    path = tmp_path / "b.fasta"
    _write_fasta(path, [(f"r{i}", "ACGTACGTAC") for i in range(97)])
    batches = list(fasta_parser.iter_batches(path, batch_size=10))
    assert sum(len(b) for b in batches) == 97
    assert len(batches) == 10  # 9 full + 1 partial
    assert [r.seq_id for b in batches for r in b] == [f"r{i}" for i in range(97)]


def test_parser_max_records_is_respected(tmp_path):
    path = tmp_path / "m.fasta"
    _write_fasta(path, [(f"r{i}", "ACGTACGTAC") for i in range(40)])
    assert len(list(fasta_parser.iter_records(path, max_records=7))) == 7


def test_illumina_header_fields_are_parsed(tmp_path):
    path = tmp_path / "h.fasta"
    _write_fasta(
        path, [("SRR26872904.1 LH00271:11:225VV3LT3:5:1101:23320:1032 length=151", "ACGT")]
    )
    record = next(fasta_parser.iter_records(path))
    fields = record.header_fields()
    assert fields["lane"] == 5
    assert fields["tile"] == 1101
    assert fields["x"] == 23320
    assert fields["y"] == 1032
    assert fields["declared_length"] == 151
    assert record.accession == "SRR26872904"


def test_non_illumina_header_yields_nulls_not_guesses(tmp_path):
    path = tmp_path / "p.fasta"
    _write_fasta(path, [("contig_1 len=500", "ACGT")])
    fields = next(fasta_parser.iter_records(path)).header_fields()
    assert fields["lane"] is None
    assert fields["tile"] is None
    assert fields["declared_length"] is None


# ---------------------------------------------------------------------------
# Dereplication -- digest counting vs naive sequence counting
# ---------------------------------------------------------------------------


def _mini_project(tmp_path, records) -> dict:
    """A config pointing at a synthetic FASTA inside tmp_path."""
    fasta = tmp_path / "mini.fasta"
    _write_fasta(fasta, records)
    cfg = cfgutil.load_config()
    cfg["paths"]["fasta"] = str(fasta)
    cfg["paths"]["outputs"] = str(tmp_path / "out")
    cfg["dereplication"] = {
        "canonical": False,
        "keep_top_variants": 10_000,
        "min_count_for_representative": 1,
    }
    return cfg


def test_dereplication_matches_naive_counting(tmp_path):
    """The hash-based streaming count must equal a plain Counter over sequences."""
    rng = np.random.default_rng(0)
    # 30 distinct sequences at wildly different depths, plus unique noise.
    pool = ["".join(rng.choice(list("ACGT"), 120)) for _ in range(30)]
    weights = rng.integers(1, 40, size=30)
    sequences = []
    for seq, w in zip(pool, weights):
        sequences.extend([seq] * int(w))
    sequences.extend("".join(rng.choice(list("ACGT"), 120)) for _ in range(50))
    rng.shuffle(sequences)

    records = [(f"r{i}", s) for i, s in enumerate(sequences)]
    cfg = _mini_project(tmp_path, records)

    metrics = dereplicate(cfg, force=True)

    qc_cfg = QCConfig.from_dict(cfg["qc"])
    naive = Counter(
        evaluate(s, qc_cfg).sequence for s in sequences if evaluate(s, qc_cfg).passed
    )
    assert metrics["unique_variants"] == len(naive)
    assert metrics["reads_dereplicated"] == sum(naive.values())
    assert metrics["max_variant_count"] == max(naive.values())
    assert metrics["singletons"] == sum(1 for v in naive.values() if v == 1)

    # The frequency spectrum must be the exact count-of-counts.
    expected_spectrum = Counter(naive.values())
    assert {int(k): v for k, v in metrics["frequency_spectrum"].items()} == dict(
        expected_spectrum
    )


def test_dereplication_recovers_correct_representative_sequences(tmp_path):
    """Pass 2 must attach the right sequence to the right count."""
    import pyarrow.parquet as pq

    from analysis.dereplicate import variants_path

    sequences = (
        ["ACGT" * 30] * 12
        + ["TTTT" + "ACGT" * 29] * 7
        + ["GGGG" + "ACGT" * 29] * 3
    )
    records = [(f"r{i}", s) for i, s in enumerate(sequences)]
    cfg = _mini_project(tmp_path, records)
    dereplicate(cfg, force=True)

    table = pq.read_table(variants_path(cfg, canonical=False))
    got = dict(
        zip(table.column("sequence").to_pylist(), table.column("count").to_pylist())
    )
    assert got == {"ACGT" * 30: 12, "TTTT" + "ACGT" * 29: 7, "GGGG" + "ACGT" * 29: 3}
    # Ranks must be in descending count order.
    assert table.column("count").to_pylist() == [12, 7, 3]


def test_reverse_complement_palindromes_are_their_own_rc():
    """Guard against a trap: 'ACGT' and its tandem repeats are RC-palindromes,
    so they are useless for testing reverse-complement behaviour."""
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("ACGT" * 30) == "ACGT" * 30
    assert reverse_complement("AACCGGTT") == "AACCGGTT"
    assert reverse_complement("AAAACCCC") == "GGGGTTTT"


def test_canonical_dereplication_merges_reverse_complements(tmp_path):
    """With canonical=True a sequence and its RC must collapse into one variant."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 120))
    rc = reverse_complement(seq)
    assert rc != seq, "test sequence must not be an RC-palindrome"
    records = [(f"r{i}", seq) for i in range(5)] + [
        (f"s{i}", rc) for i in range(3)
    ]
    cfg = _mini_project(tmp_path, records)

    exact = dereplicate(cfg, force=True, canonical=False)
    canonical = dereplicate(cfg, force=True, canonical=True)

    assert exact["unique_variants"] == 2
    assert canonical["unique_variants"] == 1
    assert canonical["max_variant_count"] == 8


def test_dereplication_writes_to_separate_files_per_mode(tmp_path):
    """The sensitivity check must never overwrite the primary result."""
    from analysis.dereplicate import variants_path

    records = [(f"r{i}", "ACGT" * 30) for i in range(4)]
    cfg = _mini_project(tmp_path, records)
    dereplicate(cfg, force=True, canonical=False)
    dereplicate(cfg, force=True, canonical=True)

    assert variants_path(cfg, canonical=False).exists()
    assert variants_path(cfg, canonical=True).exists()
    assert variants_path(cfg, canonical=False) != variants_path(cfg, canonical=True)
    metrics_dir = Path(cfg["paths"]["outputs"]) / "metrics"
    assert (metrics_dir / "dereplication_metrics.json").exists()
    assert (metrics_dir / "dereplication_metrics_canonical.json").exists()


# ---------------------------------------------------------------------------
# Embedding store
# ---------------------------------------------------------------------------


def test_store_round_trip_preserves_order_and_values(tmp_path):
    rng = np.random.default_rng(1)
    ids = [f"seq{i}" for i in range(250)]
    vectors = rng.normal(size=(250, 8)).astype(np.float32)

    store = EmbeddingStore(tmp_path, "s", dim=8, shard_size=60)
    for start in range(0, 250, 37):
        store.append(ids[start : start + 37], vectors[start : start + 37])
    store.finalize({"encoder": {"name": "test"}})

    got_ids, got_vecs, manifest = load_embeddings(tmp_path, "s")
    assert got_ids == ids
    assert np.allclose(got_vecs, vectors)
    assert manifest["total_rows"] == 250
    assert manifest["encoder"]["name"] == "test"


def test_store_resume_reports_committed_rows(tmp_path):
    rng = np.random.default_rng(2)
    ids = [f"s{i}" for i in range(100)]
    vectors = rng.normal(size=(100, 4)).astype(np.float32)

    store = EmbeddingStore(tmp_path, "r", dim=4, shard_size=25)
    store.append(ids[:70], vectors[:70])
    # 70 rows appended -> two complete shards (50 rows) committed, 20 buffered.
    assert store.n_committed == 50

    reopened = EmbeddingStore(tmp_path, "r", dim=4, shard_size=25)
    assert reopened.n_committed == 50
    assert not reopened.is_complete(100)

    reopened.append(ids[50:], vectors[50:])
    reopened.finalize()
    got_ids, got_vecs, _ = load_embeddings(tmp_path, "r")
    assert got_ids == ids
    assert np.allclose(got_vecs, vectors)


def test_store_rejects_wrong_dimension(tmp_path):
    store = EmbeddingStore(tmp_path, "d", dim=8, shard_size=10)
    with pytest.raises(ValueError):
        store.append(["a"], np.zeros((1, 4), dtype=np.float32))


def test_store_rejects_mismatched_id_count(tmp_path):
    store = EmbeddingStore(tmp_path, "m", dim=4, shard_size=10)
    with pytest.raises(ValueError):
        store.append(["a", "b"], np.zeros((3, 4), dtype=np.float32))


def test_embedding_health_detects_non_finite_and_dead_dimensions():
    matrix = np.ones((10, 3), dtype=np.float32)
    matrix[:, 0] = np.linspace(0, 1, 10)  # live
    matrix[0, 1] = np.nan
    health = embedding_health(matrix)
    assert not health["all_finite"]
    assert health["n_nan"] == 1
    # column 2 is constant -> dead
    assert health["dead_dimensions"] >= 1
