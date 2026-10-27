"""Regression and classification metrics in original label units."""

from __future__ import annotations

import numpy as np


def _nullable(values: np.ndarray) -> list[float | None]:
    return [float(x) if np.isfinite(x) else None for x in values]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 2 or true.shape != pred.shape or not len(true) or not true.shape[1]:
        raise ValueError("Regression labels/predictions must have matching nonempty [N,D] shape")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite labels and predictions")
    error = pred - true
    squared = np.square(error)
    tss = np.square(true - true.mean(axis=0)).sum(axis=0)
    r2 = np.full(true.shape[1], np.nan)
    np.divide(squared.sum(axis=0), tss, out=r2, where=tss > 0)
    r2 = 1 - r2
    return {
        "rmse": float(np.sqrt(squared.mean())),
        "mae": float(np.abs(error).mean()),
        "r2": float(r2[np.isfinite(r2)].mean()) if np.isfinite(r2).any() else None,
        "rmse_per_dim": np.sqrt(squared.mean(axis=0)).tolist(),
        "mae_per_dim": np.abs(error).mean(axis=0).tolist(),
        "r2_per_dim": _nullable(r2),
    }


def classification_from_confusion(confusion: np.ndarray) -> dict:
    confusion = np.asarray(confusion, dtype=np.float64)
    true_count, pred_count = confusion.sum(axis=1), confusion.sum(axis=0)
    correct = np.diag(confusion)
    recall = np.divide(correct, true_count, out=np.zeros_like(correct), where=true_count > 0)
    f1 = np.divide(2 * correct, true_count + pred_count, out=np.zeros_like(correct), where=true_count + pred_count > 0)
    return {
        "accuracy": float(correct.sum() / confusion.sum()),
        "balanced_accuracy": float(recall[true_count > 0].mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.astype(np.int64).tolist(),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if true.ndim != 1 or true.dtype.kind not in "iu" or not len(true):
        raise ValueError("Classification labels must be a nonempty integer array [N]")
    if pred.ndim == 2:
        if pred.shape != (len(true), num_classes):
            raise ValueError("Classification scores must have shape [N,C]")
        if not np.isfinite(pred).all():
            raise ValueError("Classification predictions must be finite")
        pred = pred.argmax(axis=1)
    if pred.shape != true.shape or pred.dtype.kind not in "iu":
        raise ValueError("Classification predictions must be integer labels or scores")
    if (
        num_classes < 1
        or np.any(true < 0)
        or np.any(true >= num_classes)
        or np.any(pred < 0)
        or np.any(pred >= num_classes)
    ):
        raise ValueError("Class ID outside label_names")
    confusion = np.bincount(true.astype(np.int64) * num_classes + pred, minlength=num_classes**2)
    return classification_from_confusion(confusion.reshape(num_classes, num_classes))


def compute_metrics(
    task_type: str, y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str] | None = None
) -> dict:
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    if task_type == "classification":
        if not label_names:
            raise ValueError("Classification metrics require label_names for the full class vocabulary")
        return classification_metrics(y_true, y_pred, len(label_names))
    raise ValueError(f"Unknown task_type: {task_type}")
