from __future__ import annotations

import numpy as np


def bin_index(q: np.ndarray, n_bins: int = 10) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if np.any((q < 0) | (q > 1)) or np.any(np.isnan(q)):
        raise ValueError("Confidence scores must lie in [0, 1]")
    return np.clip(np.floor(q * n_bins).astype(int), 0, n_bins - 1)


def reliability_table(q: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    q = np.asarray(q, dtype=float)
    y = np.asarray(y, dtype=int)
    if q.shape != y.shape:
        raise ValueError("q and y must have the same shape")

    idx = bin_index(q, n_bins)
    total = q.size
    rows = []

    for bin_number in range(n_bins):
        selected = idx == bin_number
        count = int(selected.sum())
        correct = int(y[selected].sum())
        row = {
            "bin": bin_number,
            "lo": bin_number / n_bins,
            "hi": (bin_number + 1) / n_bins,
            "n": count,
            "n_correct": correct,
            "n_incorrect": count - correct,
        }
        if count:
            mean_confidence = float(q[selected].mean())
            accuracy = float(y[selected].mean())
            gap = abs(accuracy - mean_confidence)
            row.update(
                mean_conf=mean_confidence,
                accuracy=accuracy,
                gap=gap,
                ece_contribution=count / total * gap,
                share=count / total,
            )
        else:
            row.update(
                mean_conf=None,
                accuracy=None,
                gap=None,
                ece_contribution=0.0,
                share=0.0,
            )
        rows.append(row)

    return rows


def expected_calibration_error(q: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    return float(sum(row["ece_contribution"] for row in reliability_table(q, y, n_bins)))


def ece_from_bin_summary(
    n: list[int], mean_conf: list[float], accuracy: list[float]
) -> float:
    counts = np.asarray(n, dtype=float)
    total = counts.sum()
    return float(
        np.sum(
            counts
            / total
            * np.abs(np.asarray(accuracy) - np.asarray(mean_conf))
        )
    )
