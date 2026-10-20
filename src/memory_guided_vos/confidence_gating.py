from __future__ import annotations

import numpy as np

THRESHOLDS = (0.0, 0.40, 0.55, 0.65)


def gate(q: np.ndarray, nonempty: np.ndarray, q_star: float) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    ne = np.asarray(nonempty, dtype=bool)
    return (q >= q_star) & ne


def gate_counts(y: np.ndarray, g: np.ndarray) -> dict:
    y = np.asarray(y, dtype=bool)
    g = np.asarray(g, dtype=bool)
    return {
        "accepted_correct": int(np.sum(g & y)),
        "accepted_wrong": int(np.sum(g & ~y)),
        "deferred_correct": int(np.sum(~g & y)),
        "deferred_wrong": int(np.sum(~g & ~y)),
    }


def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def gate_metrics_from_counts(ac: int, aw: int, dc: int, dw: int) -> dict:
    n = ac + aw + dc + dw
    wrong = aw + dw
    accepted = ac + aw
    return {
        "n": n,
        "coverage": _ratio(accepted, n),
        "error_rejection": _ratio(dw, wrong),
        "false_acceptance": _ratio(aw, wrong),
        "accepted_error": _ratio(aw, accepted),
    }


def gate_metrics(y: np.ndarray, q: np.ndarray, nonempty: np.ndarray, q_star: float) -> dict:
    counts = gate_counts(y, gate(q, nonempty, q_star))
    return {
        "q_star": q_star,
        **counts,
        **gate_metrics_from_counts(
            counts["accepted_correct"],
            counts["accepted_wrong"],
            counts["deferred_correct"],
            counts["deferred_wrong"],
        ),
    }
