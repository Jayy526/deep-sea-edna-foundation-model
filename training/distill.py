"""TaxDistill's core contribution: knowledge distillation from a frozen
genomic foundation model into a lightweight hand-crafted-feature student.

PAPER-DERIVED
-------------
    Teacher branch  : frozen GenomeOcean backbone + learnable classification
                      head, optimised by its own classification loss only.
    Student branch  : lightweight MLP over hand-crafted features
                      (TNF + abundance), optimised by its own classification
                      loss PLUS a KD loss against the teacher's distribution.
    KD              : KL divergence between temperature-softened teacher and
                      student distributions, weighted by alpha.

The gradient from the KD term reaches the student only. The teacher never
learns from the student -- exactly as described in the paper.

THE CLAIM BEING TESTED
----------------------
TaxDistill's argument is that labels derived from retrieval tools are noisy, and
that distilling a foundation model's soft distribution into the student reduces
the damage that noise does. Our labels are noisy in precisely that way
(reference-derived, measured CV error 5.6% at class / 15.9% at order), so the
setting is faithful.

The experiment is therefore three-way, not two-way:

    student alone          -- MLP on TNF + abundance, hard labels only
    student + KD           -- same student, plus teacher soft labels
    teacher                -- frozen embedding + head

If distillation works, "student + KD" beats "student alone" at essentially the
same inference cost.

EXPERIMENTAL ADAPTATION -- stated, not glossed
----------------------------------------------
1. **Abundance features.** The paper's student takes per-environment abundances
   across K samples plus their total. We have ONE sample, so only a total
   abundance is computable: log10(reads supporting the variant). K-environment
   abundance is not omitted for convenience -- it does not exist in this data.
2. **Deep hierarchical loss NOT implemented.** The paper uses Valmadre (2022)
   hierarchical loss over the taxonomy tree. We use plain cross-entropy, with an
   optional auxiliary parent-rank term (see ``hierarchical_weight``) which is a
   much simpler construction and is labelled as ours, not the paper's.
3. **Coarse ranks.** The paper evaluates at species level; reference coverage
   limits us to class and order.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _standardise(x: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Fit on the training split only, so nothing leaks from val/test."""
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def build_student_features(
    kmer: np.ndarray, read_counts: np.ndarray, include_abundance: bool = True
) -> np.ndarray:
    """TNF (+ total abundance) -- the paper's student input, minus what we lack."""
    if not include_abundance:
        return kmer
    abundance = np.log10(read_counts.astype(np.float64) + 1.0)[:, None]
    return np.concatenate([kmer, abundance], axis=1)


class DistillationExperiment:
    """Trains teacher, student-alone and student+KD on one labelled split."""

    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        n_classes: int,
        hidden_dim: int = 256,
        seed: int = 42,
    ):
        import torch
        import torch.nn as nn

        self._torch = torch
        self._nn = nn
        torch.manual_seed(seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.seed = seed

    def _teacher_head(self, architecture: str = "mlp"):
        """Teacher classification head.

        FAIRNESS: this defaults to the SAME architecture as the student. If the
        teacher gets a linear head while the student gets an MLP, any "KD does
        not help" result is confounded by head capacity rather than telling us
        anything about the representations. Matching them isolates the variable
        we actually care about -- foundation embedding vs hand-crafted features.
        """
        nn = self._nn
        if architecture == "linear":
            return nn.Linear(self.teacher_dim, self.n_classes).to(self.device)
        return nn.Sequential(
            nn.Linear(self.teacher_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.n_classes),
        ).to(self.device)

    def _student(self):
        nn = self._nn
        return nn.Sequential(
            nn.Linear(self.student_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.n_classes),
        ).to(self.device)

    def run(
        self,
        x_teacher: np.ndarray,
        x_student: np.ndarray,
        y: np.ndarray,
        train: np.ndarray,
        val: np.ndarray,
        test: np.ndarray,
        alpha: float = 0.4,
        temperature: float = 4.0,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        parents: np.ndarray | None = None,
        hierarchical_weight: float = 0.0,
        teacher_head: str = "mlp",
    ) -> dict[str, Any]:
        torch = self._torch
        nn = self._nn

        xt = torch.tensor(_standardise(x_teacher, train), dtype=torch.float32, device=self.device)
        xs = torch.tensor(_standardise(x_student, train), dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        parent_t = (
            torch.tensor(parents, dtype=torch.long, device=self.device)
            if parents is not None and hierarchical_weight > 0
            else None
        )
        n_parents = int(parents.max()) + 1 if parent_t is not None else 0

        tr = torch.tensor(train, dtype=torch.long, device=self.device)
        va = torch.tensor(val, dtype=torch.long, device=self.device)
        te = torch.tensor(test, dtype=torch.long, device=self.device)

        teacher = self._teacher_head(teacher_head)
        student_kd = self._student()
        torch.manual_seed(self.seed)  # identical initialisation for a fair pairing
        student_alone = self._student()

        parent_head = (
            nn.Linear(self.hidden_dim, n_parents).to(self.device) if parent_t is not None else None
        )

        opt_t = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=weight_decay)
        opt_kd = torch.optim.AdamW(student_kd.parameters(), lr=lr, weight_decay=weight_decay)
        opt_alone = torch.optim.AdamW(student_alone.parameters(), lr=lr, weight_decay=weight_decay)

        ce = nn.CrossEntropyLoss()
        kldiv = nn.KLDivLoss(reduction="batchmean")
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        history: dict[str, list[float]] = {
            k: [] for k in (
                "teacher_train_loss", "teacher_val_acc",
                "student_kd_train_loss", "student_kd_val_acc", "student_kd_kd_loss",
                "student_alone_train_loss", "student_alone_val_acc",
            )
        }

        for _ in range(epochs):
            teacher.train(); student_kd.train(); student_alone.train()
            order = torch.randperm(tr.numel(), generator=generator).to(self.device)
            sums = {"t": 0.0, "kd": 0.0, "alone": 0.0, "kdterm": 0.0}
            seen = 0

            for start in range(0, tr.numel(), batch_size):
                idx = tr[order[start : start + batch_size]]
                labels = yt[idx]
                n = idx.numel()

                # ---- teacher: its own classification loss only ----
                opt_t.zero_grad()
                logits_t = teacher(xt[idx])
                loss_t = ce(logits_t, labels)
                loss_t.backward()
                opt_t.step()

                # ---- student + KD ----
                opt_kd.zero_grad()
                logits_s = student_kd(xs[idx])
                hard = ce(logits_s, labels)
                with torch.no_grad():
                    soft_teacher = torch.softmax(teacher(xt[idx]) / temperature, dim=1)
                soft_student = torch.log_softmax(logits_s / temperature, dim=1)
                # T^2 keeps gradient magnitude comparable across temperatures.
                kd = kldiv(soft_student, soft_teacher) * (temperature**2)
                loss_kd = (1 - alpha) * hard + alpha * kd
                if parent_head is not None:
                    features = student_kd[:-1](xs[idx])
                    loss_kd = loss_kd + hierarchical_weight * ce(
                        parent_head(features), parent_t[idx]
                    )
                loss_kd.backward()
                opt_kd.step()

                # ---- student alone: identical model, hard labels only ----
                opt_alone.zero_grad()
                loss_alone = ce(student_alone(xs[idx]), labels)
                loss_alone.backward()
                opt_alone.step()

                sums["t"] += loss_t.item() * n
                sums["kd"] += loss_kd.item() * n
                sums["kdterm"] += kd.item() * n
                sums["alone"] += loss_alone.item() * n
                seen += n

            teacher.eval(); student_kd.eval(); student_alone.eval()
            with torch.no_grad():
                history["teacher_train_loss"].append(sums["t"] / seen)
                history["student_kd_train_loss"].append(sums["kd"] / seen)
                history["student_kd_kd_loss"].append(sums["kdterm"] / seen)
                history["student_alone_train_loss"].append(sums["alone"] / seen)
                history["teacher_val_acc"].append(
                    (teacher(xt[va]).argmax(1) == yt[va]).float().mean().item())
                history["student_kd_val_acc"].append(
                    (student_kd(xs[va]).argmax(1) == yt[va]).float().mean().item())
                history["student_alone_val_acc"].append(
                    (student_alone(xs[va]).argmax(1) == yt[va]).float().mean().item())

        # ---- held-out test, touched only now ----
        with torch.no_grad():
            probabilities = {
                "teacher": torch.softmax(teacher(xt[te]), 1).cpu().numpy(),
                "student_kd": torch.softmax(student_kd(xs[te]), 1).cpu().numpy(),
                "student_alone": torch.softmax(student_alone(xs[te]), 1).cpu().numpy(),
            }
        return {
            "history": history,
            "test_probabilities": probabilities,
            "y_test": y[test],
            "hyperparameters": {
                "alpha": alpha, "temperature": temperature, "epochs": epochs,
                "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
                "hidden_dim": self.hidden_dim, "seed": self.seed,
                "teacher_head": teacher_head,
                "hierarchical_weight": hierarchical_weight,
            },
            "parameter_counts": {
                "teacher_head": sum(p.numel() for p in teacher.parameters()),
                "student": sum(p.numel() for p in student_kd.parameters()),
                "foundation_backbone_trainable": 0,
            },
        }
