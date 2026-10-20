from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHA = 0.05
BETA = 3
FPS = 24.0


def area_ratio(areas: np.ndarray) -> tuple[np.ndarray, int]:
    areas = np.asarray(areas, dtype=float)
    positive = np.flatnonzero(areas > 0)
    if positive.size == 0:
        raise ValueError("Object has no valid reference area")
    reference = int(positive[0])
    ratios = np.full(areas.shape, np.nan)
    ratios[reference:] = areas[reference:] / areas[reference]
    return ratios, reference


def low_visibility(ratios: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=float)
    return np.where(np.isnan(ratios), False, ratios < alpha)


def confirmation_indicator(b: np.ndarray, beta: int = BETA) -> np.ndarray:
    b = np.asarray(b, dtype=bool)
    confirmed = np.zeros(b.size, dtype=bool)
    for t in range(beta - 1, b.size):
        window_full = bool(b[t - beta + 1 : t + 1].all())
        previous = b[t - beta] if t - beta >= 0 else False
        confirmed[t] = window_full and not previous
    return confirmed


@dataclass(frozen=True)
class Run:
    start: int
    end: int
    category: str

    @property
    def duration(self) -> int:
        return self.end - self.start + 1

    @property
    def restoration_frame(self) -> int | None:
        return None if self.category in ("censored", "short_censored") else self.end + 1


def extract_runs(b: np.ndarray, beta: int = BETA) -> list[Run]:
    b = np.asarray(b, dtype=bool)
    runs: list[Run] = []
    t, n = 0, b.size

    while t < n:
        if not b[t]:
            t += 1
            continue

        start = t
        while t < n and b[t]:
            t += 1

        end = t - 1
        open_at_end = end == n - 1
        long_enough = end - start + 1 >= beta

        if open_at_end:
            category = "censored" if long_enough else "short_censored"
        else:
            category = "confirmed" if long_enough else "short"

        runs.append(Run(start, end, category))

    return runs


def frames_to_ms(frames: np.ndarray | float, fps: float = FPS) -> np.ndarray | float:
    return np.asarray(frames, dtype=float) * 1000.0 / fps


def reacquisition_delay(j_series: np.ndarray, t_vis: int, eta_rec: float) -> int | None:
    jaccard = np.asarray(j_series, dtype=float)
    for t in range(t_vis, jaccard.size):
        if jaccard[t] >= eta_rec:
            return t - t_vis
    return None


def cantelli_upper_bound(mean: float, sd: float, threshold: float) -> float:
    difference = threshold - mean
    if difference <= 0:
        return 1.0
    return sd * sd / (sd * sd + difference * difference)


def survival_curve(durations: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    durations = np.asarray(durations, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    times = np.unique(durations[observed])
    survival = []
    probability = 1.0

    for t in times:
        at_risk = np.sum(durations >= t)
        events = np.sum((durations == t) & observed)
        if at_risk > 0:
            probability *= 1.0 - events / at_risk
        survival.append(probability)

    return times, np.asarray(survival)
