from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Q_UPDATE = 0.55


@dataclass(frozen=True)
class GoalDecision:
    goal: np.ndarray | None
    accepted_new: bool


def update_goal(
    prev_goal: np.ndarray | None,
    target_est: np.ndarray | None,
    q: float | None,
    valid: bool,
    q_min: float = Q_UPDATE,
) -> GoalDecision:
    if valid and target_est is not None and q is not None and q >= q_min:
        return GoalDecision(np.asarray(target_est, dtype=float), True)
    if prev_goal is not None:
        return GoalDecision(np.asarray(prev_goal, dtype=float), False)
    return GoalDecision(None, False)
