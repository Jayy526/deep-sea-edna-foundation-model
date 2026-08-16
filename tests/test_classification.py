"""Tests for the supervised branch, using SYNTHETIC data only.

IMPORTANT
---------
The deep-sea eDNA dataset has no ground-truth taxonomy, so the supervised
branch has never been run on it and no classification result is reported
anywhere in this project. The labels used below are generated inside the test
and are obviously synthetic. Their only purpose is to prove the code path is
correct, so that it is known-good on the day real reference labels exist.

Nothing here writes to the project's outputs directory.

Run with:  python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.classifier import (
    LinearProbe,
    classification_metrics,
    confidence_and_calibration,
    stratified_split,
)

pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_stratified_split_is_disjoint_and_covers_everything():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, size=500)
    train, val, test = stratified_split(y, (0.7, 0.15, 0.15), seed=42)

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(500))


def test_stratified_split_preserves_class_proportions():
    y = np.array([0] * 300 + [1] * 150 + [2] * 50)
    train, val, test = stratified_split(y, (0.7, 0.15, 0.15), seed=42)
    for split in (train, val, test):
        counts = np.bincount(y[split], minlength=3) / len(split)
        assert counts[0] == pytest.approx(0.6, abs=0.06)
        assert counts[1] == pytest.approx(0.3, abs=0.06)
        assert counts[2] == pytest.approx(0.1, abs=0.06)


def test_stratified_split_is_reproducible():
    y = np.random.default_rng(1).integers(0, 4, size=200)
    a = stratified_split(y, (0.7, 0.15, 0.15), seed=7)
    b = stratified_split(y, (0.7, 0.15, 0.15), seed=7)
    for x, z in zip(a, b):
        assert np.array_equal(x, z)


# ---------------------------------------------------------------------------
# Metrics -- checked against sklearn
# ---------------------------------------------------------------------------


def test_classification_metrics_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 4, size=400)
    logits = rng.normal(size=(400, 4))
    # Bias toward the truth so the metrics are not degenerate.
    logits[np.arange(400), y_true] += 1.2
    probabilities = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

    metrics = classification_metrics(y_true, probabilities, ["a", "b", "c", "d"])
    y_pred = probabilities.argmax(1)

    assert metrics["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-6)
    assert metrics["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro"), abs=1e-6
    )
    assert metrics["weighted_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="weighted"), abs=1e-6
    )
    assert np.array(metrics["confusion_matrix"]).sum() == 400
    assert len(metrics["per_class"]) == 4


def test_perfect_predictions_give_perfect_metrics():
    y = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.eye(3)[y]
    metrics = classification_metrics(y, probabilities, ["x", "y", "z"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_brier_score_matches_definition():
    y = np.array([0, 1])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7]])
    cal = confidence_and_calibration(y, probabilities)
    # ((0.8-1)^2 + (0.2-0)^2) + ((0.3-0)^2 + (0.7-1)^2)  / 2
    expected = ((0.04 + 0.04) + (0.09 + 0.09)) / 2
    assert cal["brier_score"] == pytest.approx(expected, abs=1e-9)


def test_perfectly_calibrated_predictions_have_near_zero_ece():
    """Predictions whose confidence equals their true accuracy rate."""
    rng = np.random.default_rng(5)
    n = 20000
    confidence = rng.uniform(0.5, 1.0, size=n)
    correct = rng.random(n) < confidence  # correct exactly `confidence` of the time
    probabilities = np.zeros((n, 2))
    probabilities[:, 0] = confidence
    probabilities[:, 1] = 1 - confidence
    y = np.where(correct, 0, 1)

    cal = confidence_and_calibration(y, probabilities, n_bins=15)
    assert cal["expected_calibration_error"] < 0.02


def test_overconfident_predictions_have_large_ece():
    n = 2000
    probabilities = np.tile([0.99, 0.01], (n, 1))
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1  # only 50% correct despite 99% confidence
    cal = confidence_and_calibration(y, probabilities)
    assert cal["expected_calibration_error"] > 0.4
    assert cal["mean_confidence"] == pytest.approx(0.99, abs=1e-6)


def test_confidence_categories_sum_to_total():
    rng = np.random.default_rng(6)
    probabilities = rng.dirichlet(np.ones(3), size=500)
    y = rng.integers(0, 3, size=500)
    cal = confidence_and_calibration(y, probabilities)
    assert sum(cal["confidence_categories"].values()) == 500


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def test_linear_probe_learns_a_separable_problem():
    """On clearly separable synthetic classes the probe must reach high accuracy."""
    rng = np.random.default_rng(0)
    n_classes, dim, per_class = 4, 32, 150
    centres = rng.normal(scale=3.0, size=(n_classes, dim))
    x = np.concatenate([
        centres[c] + rng.normal(scale=1.0, size=(per_class, dim))
        for c in range(n_classes)
    ]).astype(np.float32)
    y = np.repeat(np.arange(n_classes), per_class)

    train, val, test = stratified_split(y, (0.7, 0.15, 0.15), seed=42)
    probe = LinearProbe(dim, n_classes, "linear", seed=42)
    history = probe.fit(x[train], y[train], x[val], y[val], epochs=40, lr=1e-2)

    metrics = classification_metrics(
        y[test], probe.predict_proba(x[test]), [str(i) for i in range(n_classes)]
    )
    assert metrics["accuracy"] > 0.9
    # Loss must actually decrease.
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert len(history["val_loss"]) == 40


def test_probe_reports_zero_backbone_trainable_parameters():
    """The foundation model stays frozen: only the head is ever trained."""
    probe = LinearProbe(64, 5, "linear", seed=0)
    counts = probe.parameter_counts()
    assert counts["backbone_parameters_trainable"] == 0
    assert counts["head_parameters_total"] == 64 * 5 + 5
    assert counts["head_parameters_trainable"] == counts["head_parameters_total"]


def test_mlp_probe_has_more_parameters_than_linear():
    linear = LinearProbe(64, 5, "linear", seed=0).parameter_counts()
    mlp = LinearProbe(64, 5, "mlp", hidden_dim=128, seed=0).parameter_counts()
    assert mlp["head_parameters_total"] > linear["head_parameters_total"]


def test_probe_is_reproducible_under_a_fixed_seed():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(200, 16)).astype(np.float32)
    y = rng.integers(0, 3, size=200)
    train, val, _ = stratified_split(y, (0.7, 0.15, 0.15), seed=42)

    out = []
    for _ in range(2):
        probe = LinearProbe(16, 3, "linear", seed=123)
        probe.fit(x[train], y[train], x[val], y[val], epochs=5, lr=1e-2)
        out.append(probe.predict_proba(x[:20]))
    assert np.allclose(out[0], out[1], atol=1e-6)


def test_predict_proba_returns_normalised_distributions():
    probe = LinearProbe(8, 4, "linear", seed=0)
    probabilities = probe.predict_proba(np.random.default_rng(0).normal(size=(30, 8)).astype(np.float32))
    assert probabilities.shape == (30, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert (probabilities >= 0).all()


# ---------------------------------------------------------------------------
# Figure rendering (synthetic data, written to a temp dir)
# ---------------------------------------------------------------------------


def test_classification_figures_render(tmp_path):
    """Figures 9-11 have never run on real data (no labels exist), so their
    correctness is only ever exercised here."""
    from utils import config as cfgutil
    from visualization import style as S
    from visualization.classification_figures import generate_all

    rng = np.random.default_rng(0)
    n_classes, dim, per_class = 5, 24, 120
    centres = rng.normal(scale=2.5, size=(n_classes, dim))
    x = np.concatenate([
        centres[c] + rng.normal(size=(per_class, dim)) for c in range(n_classes)
    ]).astype(np.float32)
    y = np.repeat(np.arange(n_classes), per_class)
    names = [f"taxon_{i}" for i in range(n_classes)]

    train, val, test = stratified_split(y, (0.7, 0.15, 0.15), seed=42)
    probe = LinearProbe(dim, n_classes, "linear", seed=42)
    history = probe.fit(x[train], y[train], x[val], y[val], epochs=12, lr=1e-2)
    probabilities = probe.predict_proba(x[test])

    rep = {
        "n_labelled": len(y),
        "n_classes": n_classes,
        "embedding_dim": dim,
        "history": history,
        "metrics": classification_metrics(y[test], probabilities, names),
        "calibration": confidence_and_calibration(y[test], probabilities),
    }
    results = {
        "executed": True,
        "representations": {"foundation_model": rep, "baseline": rep},
    }

    cfg = cfgutil.load_config()
    cfg["paths"]["outputs"] = str(tmp_path)
    S.apply_style()
    paths = generate_all(cfg, results)

    assert len(paths) == 4
    for path in paths:
        assert path.exists() and path.stat().st_size > 5000


def test_no_figures_generated_when_branch_did_not_execute(tmp_path):
    from utils import config as cfgutil
    from visualization.classification_figures import generate_all

    cfg = cfgutil.load_config()
    cfg["paths"]["outputs"] = str(tmp_path)
    assert generate_all(cfg, {"executed": False}) == []
