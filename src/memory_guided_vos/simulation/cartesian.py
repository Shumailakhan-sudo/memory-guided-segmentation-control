from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .target_update import Q_UPDATE, update_goal

DT = 0.05
WORKSPACE = (1.5, 1.5)
TIMEOUT_S = 10.0
SUCCESS_TOL_M = 0.03


@dataclass
class Observation:
    position: np.ndarray | None
    confidence: float | None
    valid: bool


@dataclass
class EpisodeResult:
    success: bool
    closed: bool
    duration_s: float
    path_length_m: float
    final_distance_m: float | None
    n_goal_updates: int
    n_paused_steps: int
    trajectory: np.ndarray = field(repr=False)
    label: str = "reconstruction"


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, [0.0, 0.0], list(WORKSPACE))


def run_episode(
    start: np.ndarray,
    true_target: np.ndarray,
    observations: list[Observation],
    *,
    max_speed_m_s: float,
    closure_radius_m: float,
    q_min: float = Q_UPDATE,
    dt: float = DT,
    timeout_s: float = TIMEOUT_S,
    tol_m: float = SUCCESS_TOL_M,
) -> EpisodeResult:
    if max_speed_m_s <= 0 or closure_radius_m <= 0:
        raise ValueError("max_speed_m_s and closure_radius_m must be positive")
    if not observations:
        raise ValueError("At least one observation is required")

    p = _clip(np.asarray(start, dtype=float))
    target = np.asarray(true_target, dtype=float)
    goal = None
    trajectory = [p.copy()]
    updates = 0
    paused = 0
    n_steps = int(round(timeout_s / dt))
    closed = False
    k = 0

    for k in range(n_steps):
        obs = observations[min(k, len(observations) - 1)]
        decision = update_goal(goal, obs.position, obs.confidence, obs.valid, q_min)
        updates += int(decision.accepted_new)
        goal = decision.goal

        if goal is None:
            paused += 1
            trajectory.append(p.copy())
            continue

        delta = goal - p
        distance = float(np.linalg.norm(delta))
        if distance <= closure_radius_m:
            closed = True
            break

        step = min(distance, max_speed_m_s * dt)
        p = _clip(p + delta / distance * step)
        trajectory.append(p.copy())

    points = np.asarray(trajectory)
    length = (
        float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if len(points) > 1
        else 0.0
    )
    final_distance = float(np.linalg.norm(p - target)) if closed else None

    return EpisodeResult(
        success=bool(closed and final_distance is not None and final_distance <= tol_m),
        closed=closed,
        duration_s=(k + (0 if closed else 1)) * dt,
        path_length_m=length,
        final_distance_m=final_distance,
        n_goal_updates=updates,
        n_paused_steps=paused,
        trajectory=points,
    )
