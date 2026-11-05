"""Classification metrics: F1, accuracy, top-k, confusion matrix, calibration."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_classifier(model, texts: list[str], labels: list[str], k: int = 3) -> dict:
    preds = model.predict_batch(texts, k=k)
    y_pred = [p.label for p in preds]
    confidences = [p.confidence for p in preds]
    topk_sets = [[l for l, _ in p.top_k] for p in preds]

    label_names = sorted(set(labels) | set(y_pred))
    cm = confusion_matrix(labels, y_pred, labels=label_names)

    return {
        "n_samples": len(labels),
        "accuracy": round(float(accuracy_score(labels, y_pred)), 4),
        "macro_f1": round(float(f1_score(labels, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(labels, y_pred, average="weighted", zero_division=0)), 4),
        f"top_{k}_accuracy": round(
            float(np.mean([y in s for y, s in zip(labels, topk_sets)])), 4
        ),
        "confusion_matrix": {"labels": label_names, "matrix": cm.tolist()},
        "calibration": calibration_buckets(confidences, [y == p for y, p in zip(labels, y_pred)]),
    }


def calibration_buckets(confidences: list[float], correct: list[bool], n_bins: int = 5) -> list[dict]:
    """Reliability table: does an 0.8-confidence prediction hit 80% accuracy?"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    buckets = []
    conf = np.array(confidences)
    corr = np.array(correct, dtype=float)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        buckets.append(
            {
                "bucket": f"{lo:.1f}–{hi:.1f}",
                "count": int(mask.sum()),
                "avg_confidence": round(float(conf[mask].mean()), 4) if mask.any() else None,
                "accuracy": round(float(corr[mask].mean()), 4) if mask.any() else None,
            }
        )
    return buckets
