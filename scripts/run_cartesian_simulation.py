from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np

from _common import resolve
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    read_csv_records,
    write_csv_records,
    write_json,
)
from memory_guided_vos.simulation import cartesian as cs
from memory_guided_vos.simulation.reward_analysis import (
    cartesian_summary,
    episode_overlap,
    reward_summary,
)


def summarise(cfg: dict, args) -> int:
    try:
        episode_path = resolve(cfg["episodes_records"])
        reward_path = resolve(cfg["reward_records"])
        episodes = read_csv_records(episode_path)
        rewards = read_csv_records(reward_path)
    except MissingInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    result = {
        "cartesian": cartesian_summary(
            episodes, tuple(cfg["grasp_duration_excluded_episodes"])
        ),
        "reward": reward_summary(rewards),
        "episode_overlap": episode_overlap(
            [row["episode"] for row in episodes],
            [row["episode"] for row in rewards],
        ),
        "scope": "Simulated Cartesian target reaching; not physical grasping.",
        "provenance": provenance(
            [episode_path, reward_path], data_kind=args.data_kind
        ),
    }

    output = (
        resolve(cfg["output"]["dir"])
        / args.data_kind
        / "simulation_summary.json"
    )
    write_json(output, result)
    print(f"{len(episodes)} Cartesian episodes, {len(rewards)} reward episodes -> {output}")
    return 0


def reconstruct(cfg: dict, args) -> int:
    reconstruction_cfg = cfg["reconstruction"]
    try:
        speed = float(reconstruction_cfg["max_speed_m_s"])
        radius = float(reconstruction_cfg["closure_radius_m"])
    except (TypeError, ValueError):
        print(
            "ERROR: Set valid reconstruction.max_speed_m_s and "
            "reconstruction.closure_radius_m values in the configuration.",
            file=sys.stderr,
        )
        return 2

    observation_source = reconstruction_cfg["observations"]
    try:
        observation_path = resolve(observation_source) if observation_source else None
        observations = read_csv_records(observation_path) if observation_path else None
    except MissingInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not observations:
        print("ERROR: Reconstruction requires per-step observation records.", file=sys.stderr)
        return 2

    grouped = defaultdict(list)
    for row in observations:
        grouped[row["episode"]].append(row)

    results = []
    for episode_id, records in grouped.items():
        records.sort(key=lambda row: int(row["step"]))
        inputs = [
            cs.Observation(
                np.array([float(row["x"]), float(row["y"])]),
                float(row["confidence"]),