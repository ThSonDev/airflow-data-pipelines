"""Evaluation metrics on class ids (0 NONE, 1 NEG, 2 POS, 3 NEU), arrays shaped (N, 8)."""
import numpy as np
from absa_core import ASPECTS
from sklearn.metrics import balanced_accuracy_score, f1_score

CLASSES = [0, 1, 2, 3]


def compute_metrics(y_true, y_pred):
    """mean_aspect_f1 is the primary metric. macro_f1_flat pools all aspects, NONE included."""
    # No labels arg: classes absent in both truth and prediction are not counted (same as the notebook)
    aspect_f1 = {
        a: float(f1_score(y_true[:, i], y_pred[:, i], average="macro", zero_division=0))
        for i, a in enumerate(ASPECTS)
    }
    flat_true, flat_pred = y_true.ravel(), y_pred.ravel()
    return {
        "mean_aspect_f1": float(np.mean(list(aspect_f1.values()))),
        "aspect_f1": aspect_f1,
        "macro_f1_flat": float(f1_score(flat_true, flat_pred, average="macro", labels=CLASSES, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(flat_true, flat_pred)),
    }
