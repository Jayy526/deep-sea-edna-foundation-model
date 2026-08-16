"""Figures 20-22: TaxDistill knowledge distillation.

The central question these answer is the paper's own claim: does distilling the
foundation model's soft distribution into a lightweight hand-crafted-feature
student actually improve that student?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from utils import config as cfgutil
from visualization import style as S

TEACHER_COLOR = S.SERIES[0]
KD_COLOR = S.SERIES[2]
ALONE_COLOR = S.SERIES[1]

BRANCH_STYLE = {
    "teacher": ("Teacher\n(GenomeOcean + head)", TEACHER_COLOR),
    "student_kd": ("Student + KD\n(TNF + abundance)", KD_COLOR),
    "student_alone": ("Student alone\n(TNF + abundance)", ALONE_COLOR),
}
ORDER = ["teacher", "student_kd", "student_alone"]


def _figdir(cfg: dict) -> Path:
    return cfgutil.output_dir(cfg, "figures")


def figure_distillation(cfg: dict, results: dict) -> list[Path]:
    """Q: Does knowledge distillation improve the lightweight student?"""
    paths = []
    for rank, res in results["ranks"].items():
        branches = res["branches"]
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.2, 1.2, 1]})

        for ax, metric, title in (
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "macro_f1", "Macro F1"),
        ):
            names = [BRANCH_STYLE[b][0] for b in ORDER]
            colours = [BRANCH_STYLE[b][1] for b in ORDER]
            values = [branches[b]["metrics"][metric] for b in ORDER]
            bars = ax.bar(names, values, color=colours, width=0.6)
            ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4,
                         fontsize=9.5, color=S.TEXT_SECONDARY)
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_ylim(0, min(1.05, max(values) * 1.18))
            S.clean_axes(ax)

        effect = res.get("kd_effect", {})
        S.stat_tile(
            axes[2],
            f"{effect.get('accuracy_difference_pp', 0):+.2f} pp",
            "accuracy change from\nadding the KD loss",
            f"{effect.get('verdict', 'not tested')}\n"
            f"macro F1 change {effect.get('macro_f1_difference', 0):+.4f}\n"
            f"{effect.get('corrected_by_kd', 0)} corrected / "
            f"{effect.get('broken_by_kd', 0)} broken\n"
            f"student = {res['student_feature_dim']}-d vs teacher {res['teacher_feature_dim']:,}-d",
            color=KD_COLOR if effect.get("accuracy_difference_pp", 0) > 0 else S.SERIES[1],
            size=24,
        )

        S.suptitle(fig, f"Figure 20 — Knowledge distillation at {rank} level")
        S.caption(
            fig,
            f"{res['n_labelled']:,} labelled variants, {res['n_classes']} classes, "
            f"alpha={res['hyperparameters']['alpha']}, T={res['hyperparameters']['temperature']}. "
            f"Teacher and both students share the split, the seed and the held-out test set; the two "
            f"students share an identical initialisation so the only difference is the KD term. "
            f"Labels are NOISY reference-derived assignments, not ground truth.",
        )
        paths.append(S.save(fig, _figdir(cfg) / f"distillation_{rank}.png", also_pdf=True))
    return paths


def figure_distill_curves(cfg: dict, results: dict) -> list[Path]:
    """Q: How does the KD term behave during training?"""
    paths = []
    for rank, res in results["ranks"].items():
        h = res["history"]
        epochs = np.arange(1, len(h["teacher_train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")

        axes[0].plot(epochs, h["teacher_val_acc"], color=TEACHER_COLOR, linewidth=2,
                     label="teacher")
        axes[0].plot(epochs, h["student_kd_val_acc"], color=KD_COLOR, linewidth=2,
                     label="student + KD")
        axes[0].plot(epochs, h["student_alone_val_acc"], color=ALONE_COLOR, linewidth=2,
                     linestyle="--", label="student alone")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation accuracy")
        axes[0].set_title("Validation accuracy")
        axes[0].legend(fontsize=8.5)
        S.clean_axes(axes[0])

        axes[1].plot(epochs, h["student_kd_kd_loss"], color=KD_COLOR, linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("KD loss (T$^2$ · KL divergence)")
        axes[1].set_title("Teacher-student divergence")
        S.clean_axes(axes[1])

        S.suptitle(fig, f"Figure 21 — Distillation training dynamics ({rank})")
        S.caption(
            fig,
            "Validation split only; the test split is never evaluated during training. "
            "The KD loss measures how far the student's temperature-softened distribution "
            "sits from the teacher's — it falls as the student absorbs the teacher.",
        )
        paths.append(S.save(fig, _figdir(cfg) / f"distillation_curves_{rank}.png", also_pdf=True))
    return paths


def figure_ablation(cfg: dict, ablation: dict) -> Path:
    """Q: How sensitive is distillation to alpha and temperature?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")

    for ax, key, xlabel, title in (
        (axes[0], "alpha_sweep", "KD weight α", "Sensitivity to α"),
        (axes[1], "temperature_sweep", "Temperature T", "Sensitivity to T"),
    ):
        rows = ablation[key]
        if not rows:
            continue
        x = [r["alpha"] if key == "alpha_sweep" else r["temperature"] for r in rows]
        ax.plot(x, [r["student_kd_macro_f1"] for r in rows], color=KD_COLOR,
                marker="o", markersize=6, linewidth=2.2, label="student + KD (macro F1)")
        ax.plot(x, [r["student_kd_accuracy"] for r in rows], color=KD_COLOR,
                marker="s", markersize=5, linewidth=1.6, linestyle=":", label="student + KD (accuracy)")
        alone = rows[0]["student_alone_macro_f1"]
        ax.axhline(alone, color=ALONE_COLOR, linestyle="--", linewidth=1.8,
                   label="student alone (macro F1)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test score")
        ax.set_title(title)
        ax.legend(fontsize=8)
        S.clean_axes(ax)

    S.suptitle(fig, "Figure 22 — Distillation ablations")
    S.caption(
        fig,
        f"Rank: {ablation['rank']}, {ablation['n_labelled']:,} labelled variants, "
        f"{ablation['n_classes']} classes. Each point is a full retrain from an identical "
        f"initialisation with the same split and seed; only α (left) or T (right) changes. "
        f"The dashed line is the same student trained without any KD term.",
    )
    return S.save(fig, _figdir(cfg) / "distillation_ablation.png", also_pdf=True)


def generate_all(cfg: dict, results: dict) -> list[Path]:
    if not results.get("ranks"):
        return []
    return figure_distillation(cfg, results) + figure_distill_curves(cfg, results)
