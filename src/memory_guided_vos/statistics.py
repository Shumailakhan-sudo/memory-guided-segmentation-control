from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def describe(x: np.ndarray, ddof: int = 1) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return {"n": 0}

    q1, median, q3 = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(median),
        "sd": float(x.std(ddof=ddof)) if x.size > ddof else None,
        "q1": float(q1),
        "q3": float(q3),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def cluster_bootstrap(
    clusters: Sequence,
    statistic: Callable[[np.ndarray], float | None],
    n_resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    clusters = np.asarray(clusters)
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(ids.size)]
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(clusters.size))
    samples = []
    undefined = 0

    for _ in range(n_resamples):
        selected = rng.integers(0, ids.size, size=ids.size)
        indices = np.concatenate([members[k] for k in selected])
        value = statistic(indices)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            undefined += 1
        else:
            samples.append(value)

    samples = np.asarray(samples, dtype=float)
    alpha = (1 - ci) / 2
    lower, upper = (
        np.percentile(samples, [100 * alpha, 100 * (1 - alpha)])
        if samples.size
        else (np.nan, np.nan)
    )

    return {
        "point": point,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "n_clusters": int(ids.size),
        "n_resamples": n_resamples,
        "n_undefined_resamples": undefined,
        "seed": seed,
        "method": "sequence-cluster percentile bootstrap",
    }


def huber_location(values: np.ndarray, delta: float = 1.0) -> float:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")

    location = float(np.median(values))
    for _ in range(100):
        residuals = values - location
        weights = np.where(
            np.abs(residuals) <= delta,
            1.0,
            delta / np.maximum(np.abs(residuals), 1e-12),
        )
        updated = float(np.sum(weights * values) / np.sum(weights))
        if abs(updated - location) < 1e-10:
            return updated
        location = updated

    return location


def huber_loss(r: np.ndarray, delta: float = 1.0) -> np.ndarray:
    residuals = np.abs(np.asarray(r, dtype=float))
    return np.where(
        residuals <= delta,
        0.5 * residuals * residuals,
        delta * residuals - 0.5 * delta * delta,
    )
