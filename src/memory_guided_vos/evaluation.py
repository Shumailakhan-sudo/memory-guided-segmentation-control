from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import calibration, confidence_gating, statistics
from .runtime import STAGES, latency_summary, stage_shares


def _f(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def segmentation_summary(rows: list[dict]) -> dict:
    result = {
        "pooled": {
            "n": len(rows),
            "n_sequences": len({row["sequence"] for row in rows}),
            "jaccard": statistics.describe(_f(rows, "jaccard")),
            "boundary_f": statistics.describe(_f(rows, "boundary_f")),
        }
    }

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    result["by_strategy"] = {
        strategy: {
            "n": len(group_rows),
            "jaccard": statistics.describe(_f(group_rows, "jaccard")),
            "boundary_f": statistics.describe(_f(group_rows, "boundary_f")),
        }
        for strategy, group_rows in sorted(grouped.items())
    }
    return result


def calibration_summary(
    rows: list[dict],
    eta: float = 0.50,
    n_bins: int = 10,
    thresholds=confidence_gating.THRESHOLDS,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    confidence = _f(rows, "confidence")
    correct = (_f(rows, "jaccard") >= eta).astype(int)
    nonempty = np.asarray([_truthy(row["nonempty_pred"]) for row in rows])
    sequences = [row["sequence"] for row in rows]

    result = {
        "n": int(confidence.size),
        "n_correct": int(correct.sum()),
        "n_incorrect": int(confidence.size - correct.sum()),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "eta": eta,
        "bins": calibration.reliability_table(confidence, correct, n_bins),
        "ece": calibration.expected_calibration_error(confidence, correct, n_bins),
        "gates": [],
    }

    for threshold in thresholds:
        metrics = confidence_gating.gate_metrics(correct, confidence, nonempty, threshold)
        intervals = {}
        for metric_name in (
            "coverage",
            "error_rejection",
            "false_acceptance",
            "accepted_error",
        ):
            def statistic(indices, metric_name=metric_name, threshold=threshold):
                counts = confidence_gating.gate_counts(
                    correct[indices],
                    confidence_gating.gate(
                        confidence[indices], nonempty[indices], threshold
                    ),
                )
                values = confidence_gating.gate_metrics_from_counts(
                    counts["accepted_correct"],
                    counts["accepted_wrong"],
                    counts["deferred_correct"],
                    counts["deferred_wrong"],
                )
                return values[metric_name]

            intervals[metric_name] = statistics.cluster_bootstrap(
                sequences, statistic, n_resamples=n_boot, seed=seed
            )
