"""Lightweight taxonomic classifier over frozen embeddings.

PAPER-DERIVED
    TaxDistill keeps the foundation-model backbone frozen and trains only a
    small head on top of it. We do the same: a linear probe (default) or a
    one-hidden-layer MLP. Nothing larger is warranted -- the point of a probe is
    to measure what the representation already contains, not to add capacity
    that could compensate for a weak representation.

STATUS IN THIS EXPERIMENT: IMPLEMENTED BUT NOT EXECUTED.
    This dataset has no ground-truth taxonomy (see analysis/labels.py), so this
    module has no valid input. It runs unchanged the moment a verified
    read_id -> taxon table is supplied via ``classification.labels_path``.
    It is deliberately not exercised on invented labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class LinearProbe:
    """A linear (or shallow MLP) classifier on frozen embeddings, in PyTorch."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        model: str = "linear",
        hidden_dim: int = 256,
        seed: int = 42,
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self._torch = torch
        self.model_type = model
        if model == "linear":
            self.net = nn.Linear(input_dim, n_classes)
        elif model == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            raise ValueError("model must be 'linear' or 'mlp'")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.seed = seed

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        return {
            "head_parameters_total": total,
            "head_parameters_trainable": trainable,
            "backbone_parameters_trainable": 0,
        }

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
    ) -> dict[str, list[float]]:
        """Train on the training split, monitor on validation. The TEST SPLIT IS
        NEVER SEEN HERE -- it is held out until ``evaluate``."""
        torch = self._torch
        import torch.nn as nn

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        xt = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y_train, dtype=torch.long, device=self.device)
        xv = torch.tensor(x_val, dtype=torch.float32, device=self.device)
        yv = torch.tensor(y_val, dtype=torch.long, device=self.device)

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []
        }
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for _ in range(epochs):
            self.net.train()
            order = torch.randperm(xt.shape[0], generator=generator).to(self.device)
            total_loss = correct = seen = 0
            for start in range(0, xt.shape[0], batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad()
                logits = self.net(xt[idx])
                loss = criterion(logits, yt[idx])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * idx.numel()
                correct += (logits.argmax(1) == yt[idx]).sum().item()
                seen += idx.numel()
            history["train_loss"].append(total_loss / seen)
            history["train_accuracy"].append(correct / seen)

            self.net.eval()
            with torch.no_grad():
                logits = self.net(xv)
                history["val_loss"].append(criterion(logits, yv).item())
                history["val_accuracy"].append(
                    (logits.argmax(1) == yv).float().mean().item()
                )
        return history

    def predict_proba(self, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        torch = self._torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                chunk = torch.tensor(
                    x[start : start + batch_size], dtype=torch.float32, device=self.device
                )
                out.append(torch.softmax(self.net(chunk), dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)


def stratified_split(
    y: np.ndarray, fractions: tuple[float, float, float], seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproducible stratified train/val/test indices."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for label in np.unique(y):
        idx = np.nonzero(y == label)[0]
        rng.shuffle(idx)
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        train.extend(idx[:n_train])
        val.extend(idx[n_train : n_train + n_val])
        test.extend(idx[n_train + n_val :])
    return (
        np.sort(np.array(train)), np.sort(np.array(val)), np.sort(np.array(test))
    )


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]
) -> dict[str, Any]:
    """Accuracy, macro/weighted P-R-F1, per-class metrics, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support, precision_score, recall_score,
    )

    y_pred = probabilities.argmax(axis=1)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "per_class": [
            {
                "class": name,
                "precision": round(float(per_class[0][i]), 6),
                "recall": round(float(per_class[1][i]), 6),
                "f1": round(float(per_class[2][i]), 6),
                "support": int(per_class[3][i]),
            }
            for i, name in enumerate(class_names)
        ],
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(class_names))
        ).tolist(),
        "n_test": int(len(y_true)),
        "n_classes": len(class_names),
    }


def confidence_and_calibration(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 15
) -> dict[str, Any]:
    """Confidence summary, Brier score, ECE, and reliability-diagram bins.

    NOTE: low confidence is NOT evidence of a novel taxon. Novel-taxa discovery
    is outside this implementation and no such inference is drawn anywhere.
    """
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)

    onehot = np.zeros_like(probabilities)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    brier = float(((probabilities - onehot) ** 2).sum(axis=1).mean())

    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            bins.append({"lower": float(lo), "upper": float(hi), "count": 0,
                         "mean_confidence": None, "accuracy": None})
            continue
        mean_conf = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        weight = mask.sum() / len(confidence)
        ece += weight * abs(accuracy - mean_conf)
        bins.append({
            "lower": float(lo), "upper": float(hi), "count": int(mask.sum()),
            "mean_confidence": round(mean_conf, 6), "accuracy": round(accuracy, 6),
        })

    return {
        "mean_confidence": round(float(confidence.mean()), 6),
        "median_confidence": round(float(np.median(confidence)), 6),
        "brier_score": round(brier, 6),
        "expected_calibration_error": round(float(ece), 6),
        "reliability_bins": bins,
        "confidence_categories": {
            "high (>=0.9)": int((confidence >= 0.9).sum()),
            "medium (0.7-0.9)": int(((confidence >= 0.7) & (confidence < 0.9)).sum()),
            "low (<0.7)": int((confidence < 0.7).sum()),
        },
        "caveat": (
            "Low classification confidence is not evidence of a novel species. "
            "Novel-taxa discovery is outside the scope of this implementation."
        ),
    }
