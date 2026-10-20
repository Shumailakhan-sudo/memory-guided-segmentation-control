from __future__ import annotations

import numpy as np

from ..statistics import describe


def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray(
        [
            float(row[key])
            if str(row.get(key, "")).strip() not in ("", "-", "nan")
            else np.nan
            for row in rows
        ],
        dtype=float,
    )


def cartesian_summary(
    rows: list[dict], exclude_grasp_episodes: tuple[str, ...] = ()
) -> dict:
    excluded = set(map(str, exclude_grasp_episodes))
    duration_rows = [row for row in rows if str(row["episode"]) not in excluded]

    result = {
        "n_episodes": len(rows),
        "success_rate": float(np.nanmean(_col(rows, "success"))),
        "confidence": describe(_col(rows, "confidence")),
        "episode_s": describe(_col(rows, "episode_s")),
        "path_m": describe(_col(rows, "path_m")),
    }

    if rows and "grasp_s" in rows[0]:
        result["grasp_s"] = describe(_col(duration_rows, "grasp_s"))
        result["grasp_s_excluded_episodes"] = list(exclude_grasp_episodes)

    if rows and "retries" in rows[0]:
        retries = _col(rows, "retries")
        result["episodes_with_retries"] = int(np.sum(retries > 0))
        result["total_retries"] = int(np.nansum(retries))

    if rows and "occluded" in rows[0]:
        result["occluded_episodes"] = [
            row["episode"] for row in rows if float(row["occluded"]) > 0
        ]

    if rows and "final_action" in rows[0]:
        actions = [row["final_action"] for row in rows]
        result["final_action_counts"] = {
            action: actions.count(action) for action in sorted(set(actions))
        }

    return result


def reward_summary(rows: list[dict]) -> dict:
    result = {"n_episodes": len(rows)}

    for key in (
        "duration_s",
        "cumulative_reward",
        "efficiency",
        "relative_performance",
        "median_reward",
    ):
        if rows and key in rows[0]:
            result[key] = describe(_col(rows, key))

    if rows and "efficiency" in rows[0]:
        ratios = _col(rows, "cumulative_reward") / _col(rows, "duration_s")
        result["max_abs_efficiency_minus_reward_over_duration"] = float(
            np.nanmax(np.abs(ratios - _col(rows, "efficiency")))
        )

    return result


def episode_overlap(cartesian_ids: list[str], reward_ids: list[str]) -> dict:
    cartesian = set(map(str, cartesian_ids))
    reward = set(map(str, reward_ids))

    return {
        "in_both": sorted(cartesian & reward, key=int),
        "reward_only": sorted(reward - cartesian, key=int),
        "cartesian_only": sorted(cartesian - reward, key=int),
    }
