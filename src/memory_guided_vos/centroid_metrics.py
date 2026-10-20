from __future__ import annotations

import warnings

import numpy as np


def trajectory_from_masks(masks: list[np.ndarray]) -> np.ndarray:
    out = np.full((len(masks), 2), np.nan)
    for t, mask in enumerate(masks):
        ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
        if xs.size:
            out[t] = xs.mean(), ys.mean()
    return out


def path_length(c: np.ndarray) -> float:
    c = np.asarray(c, dtype=float)
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return float(np.nansum(steps))


def window_means(c: np.ndarray, tau: int) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    k = c.shape[0] // tau
    if k == 0:
        return np.empty((0, 2))
    blocks = c[: k * tau].reshape(k, tau, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(blocks, axis=1)


def allan_variance_2d(c: np.ndarray, tau: int) -> float:
    if tau < 1:
        raise ValueError("tau must be >= 1")
    means = window_means(c, tau)
    if means.shape[0] < 2:
        return float("nan")
    differences = np.diff(means, axis=0)
    squared = np.sum(differences * differences, axis=1)
    squared = squared[~np.isnan(squared)]
    if squared.size == 0:
        return float("nan")
    return float(0.5 * squared.mean())


def allan_curve(c: np.ndarray, taus: list[int] | range) -> list[dict]:
    rows = []
    n_frames = np.asarray(c).shape[0]
    for tau in taus:
        k = n_frames // tau
        if k < 2:
            continue
        rows.append({
            "tau": int(tau),
            "K": int(k),
            "allan_variance_px2": allan_variance_2d(c, tau),
        })
    return rows
