"""Figures 9-11: supervised classification.

These are generated ONLY when verified ground-truth labels are supplied. On the
deep-sea eDNA dataset used here no such labels exist, so this module has never
been run on real data — its correctness is covered by tests/test_classification.py
using clearly-synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt

from utils import config as cfgutil
from visualization import style as S

FOUNDATION_COLOR = S.SERIES[0]
BASELINE_COLOR = S.SERIES[1]


def _figdir(cfg: dict) -> Path:
    return cfgutil.output_dir(cfg, "figures")


def figure_confusion_matrix(cfg: dict, results: dict, suffix: str = "") -> Path:
    """Figure 9 — where does the classifier confuse which taxa?"""
    reps = results["representations"]
    fig, axes = plt.subplots(1, len(reps), figsize=(6.2 * len(reps), 5.4),
                             layout="constrained", squeeze=False)
    for ax, (label, rep) in zip(axes[0], reps.items()):
        matrix = np.array(rep["metrics"]["confusion_matrix"], dtype=float)
        # Row-normalise: absolute counts are dominated by class imbalance.
        totals = matrix.sum(axis=1, keepdims=True)
        normalised = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
        im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(label.replace("_", " "))
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        names = [c["class"] for c in rep["metrics"]["per_class"]]
        if len(names) <= 25:
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=7)
            ax.set_yticklabels(names, fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Row-normalised rate")

    S.suptitle(fig, "Figure 9 — Confusion matrix (row-normalised)")
    S.caption(
        fig,
        f"Held-out test split only, never seen during training. Rows are normalised "
        f"because absolute counts are dominated by class imbalance.",
    )
    return S.save(fig, _figdir(cfg) / f"confusion_matrix{suffix}.png", also_pdf=True)


def figure_per_class_f1(cfg: dict, results: dict, suffix: str = "") -> Path:
    """Which classes are actually learnable?"""
    reps = results["representations"]
    fig, ax = plt.subplots(figsize=(min(16, 2 + 0.35 * results["representations"][
        next(iter(reps))]["n_classes"]), 4.4), layout="constrained")

    names = [c["class"] for c in reps[next(iter(reps))]["metrics"]["per_class"]]
    width = 0.8 / len(reps)
    for i, (label, rep) in enumerate(reps.items()):
        f1 = [c["f1"] for c in rep["metrics"]["per_class"]]
        colour = FOUNDATION_COLOR if "foundation" in label else BASELINE_COLOR
        ax.bar(np.arange(len(names)) + i * width, f1, width=width,
               color=colour, label=label.replace("_", " "))
    ax.set_xticks(np.arange(len(names)) + width * (len(reps) - 1) / 2)
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("F1 score")
    ax.set_xlabel("Class")
    ax.set_ylim(0, 1.05)
    ax.legend()
    S.clean_axes(ax)

    S.suptitle(fig, "Figure 9b — Per-class F1")
    S.caption(fig, "Held-out test split. Classes with few test examples have high-variance F1.")
    return S.save(fig, _figdir(cfg) / f"per_class_f1{suffix}.png", also_pdf=True)


def figure_training_curves(cfg: dict, results: dict, suffix: str = "") -> Path:
    """Figure 10 — training and validation loss/accuracy per epoch."""
    reps = results["representations"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")

    for label, rep in reps.items():
        colour = FOUNDATION_COLOR if "foundation" in label else BASELINE_COLOR
        history = rep["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], color=colour, linewidth=2,
                     label=f"{label.replace('_', ' ')} — train")
        axes[0].plot(epochs, history["val_loss"], color=colour, linewidth=2,
                     linestyle="--", label=f"{label.replace('_', ' ')} — validation")
        axes[1].plot(epochs, np.array(history["train_accuracy"]) * 100, color=colour,
                     linewidth=2, label=f"{label.replace('_', ' ')} — train")
        axes[1].plot(epochs, np.array(history["val_accuracy"]) * 100, color=colour,
                     linewidth=2, linestyle="--",
                     label=f"{label.replace('_', ' ')} — validation")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.legend(fontsize=8)
        S.clean_axes(ax)

    S.suptitle(fig, "Figure 10 — Training and validation curves")
    S.caption(
        fig,
        "Solid = training split, dashed = validation split. The test split is never "
        "evaluated during training and does not appear here.",
    )
    return S.save(fig, _figdir(cfg) / f"training_curves{suffix}.png", also_pdf=True)


def figure_confidence(cfg: dict, results: dict, suffix: str = "") -> Path:
    """Figure 11 — confidence histogram and reliability diagram."""
    reps = results["representations"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")

    for label, rep in reps.items():
        colour = FOUNDATION_COLOR if "foundation" in label else BASELINE_COLOR
        cal = rep["calibration"]
        bins = [b for b in cal["reliability_bins"] if b["count"] > 0]
        centres = [(b["lower"] + b["upper"]) / 2 for b in bins]
        counts = [b["count"] for b in bins]
        axes[0].plot(centres, counts, color=colour, marker="o", markersize=5,
                     label=label.replace("_", " "))
        axes[1].plot(
            [b["mean_confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            color=colour, marker="o", markersize=6,
            label=f"{label.replace('_', ' ')} (ECE {cal['expected_calibration_error']:.3f})",
        )

    axes[0].set_xlabel("Predicted confidence")
    axes[0].set_ylabel("Test predictions")
    axes[0].set_title("Confidence distribution")
    axes[0].yaxis.set_major_formatter(FuncFormatter(S.thousands))
    axes[0].legend(fontsize=8)
    S.clean_axes(axes[0])

    axes[1].plot([0, 1], [0, 1], color=S.TEXT_MUTED, linestyle=":", linewidth=1.4)
    axes[1].text(0.62, 0.55, "perfect calibration", rotation=34, fontsize=8.5,
                 color=S.TEXT_MUTED)
    axes[1].set_xlabel("Mean predicted confidence")
    axes[1].set_ylabel("Observed accuracy")
    axes[1].set_title("Reliability diagram")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper left")
    S.clean_axes(axes[1])

    S.suptitle(fig, "Figure 11 — Confidence and calibration")
    S.caption(
        fig,
        "Held-out test split. Points below the diagonal indicate overconfidence. "
        "NOTE: low confidence is not evidence of a novel species — novel-taxa "
        "discovery is outside the scope of this implementation.",
    )
    return S.save(fig, _figdir(cfg) / f"confidence{suffix}.png", also_pdf=True)


def generate_all(cfg: dict, results: dict, suffix: str = "") -> list[Path]:
    if not results.get("executed"):
        return []
    return [
        figure_confusion_matrix(cfg, results, suffix),
        figure_per_class_f1(cfg, results, suffix),
        figure_training_curves(cfg, results, suffix),
        figure_confidence(cfg, results, suffix),
    ]
