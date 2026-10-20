from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import resolve
from memory_guided_vos import centroid_metrics as cm
from memory_guided_vos import visibility as vis
from memory_guided_vos.dataset import DavisDataset, load_index_mask, object_masks
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    write_csv_records,
    write_json,
)
from memory_guided_vos.statistics import describe


def analyse(
    ds: DavisDataset,
    names: list[str],
    alpha: float,
    beta: int,
    merge: bool,
    taus,
) -> tuple[list, list, list, list]:
    frame_rows, run_rows, traj_rows, allan_rows = [], [], [], []

    for name in names:
        sequence = ds.sequence(name)
        labels = [load_index_mask(path) for path in sequence.annotations]
        object_ids = sorted({oid for label in labels for oid in object_masks(label, merge)})
        per_object = {oid: [] for oid in object_ids}

        for label in labels:
            masks = object_masks(label, merge)
            for oid in object_ids:
                per_object[oid].append(masks.get(oid, np.zeros(label.shape, dtype=bool)))

        for oid, masks in per_object.items():
            areas = np.array([mask.sum() for mask in masks], dtype=float)
            try:
                ratios, reference_frame = vis.area_ratio(areas)
            except ValueError:
                continue

            low_visibility = vis.low_visibility(ratios, alpha)

            for frame_index in range(len(masks)):
                frame_rows.append({
                    "sequence": name,
                    "object_id": oid,
                    "frame_index": frame_index,
                    "area_px": int(areas[frame_index]),
                    "area_ratio": None if np.isnan(ratios[frame_index]) else float(ratios[frame_index]),
                    "low_visibility": int(low_visibility[frame_index]),
                })

            for run in vis.extract_runs(low_visibility, beta):
                run_rows.append({
                    "sequence": name,
                    "object_id": oid,
                    "t_start": run.start,
                    "t_end": run.end,
                    "gap_frames": run.duration,
                    "gap_ms": float(vis.frames_to_ms(run.duration)),
                    "category": run.category,
                    "t_visible": run.restoration_frame,
                })

            centroids = cm.trajectory_from_masks(masks)
            traj_rows.append({
                "sequence": name,
                "object_id": oid,
                "n_frames": len(masks),
                "path_length_px": cm.path_length(centroids),
                "reference_frame": reference_frame,
            })

            for result in cm.allan_curve(centroids, taus):
                allan_rows.append({"sequence": name, "object_id": oid, **result})

    return frame_rows, run_rows, traj_rows, allan_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze object visibility and centroid motion.")
    parser.add_argument("--config", default="configs/davis.toml")
    parser.add_argument("--root", help="Override the dataset directory")
    parser.add_argument("--out-tag", default="new_run")
    parser.add_argument("--results-root", default="results", help="Output directory")
    parser.add_argument("--data-kind", default="new_experiment", choices=["new_experiment", "synthetic"])
    parser.add_argument("--max-tau", type=int, default=50, help="Maximum Allan variance window")