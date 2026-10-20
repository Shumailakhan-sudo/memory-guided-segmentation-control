from __future__ import annotations

import argparse
import sys

import numpy as np

from _common import resolve

from memory_guided_vos import optical_flow as of
from memory_guided_vos.dataset import DavisDataset, load_index_mask, load_rgb, object_masks
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    require_file,
    write_csv_records,
    write_json,
)
from memory_guided_vos.statistics import describe, huber_location


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute optical flow metrics for video sequences.")
    parser.add_argument("--config", default="configs/raft.toml")
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args(argv)

    cfg = load_config(resolve(args.config))
    data_cfg = load_config(resolve(cfg["data"]["davis_config"]))

    try:
        if cfg["backend"] == "princeton":
            require_file(resolve(cfg["checkpoint"]), "RAFT checkpoint")

        dataset = DavisDataset(
            resolve(data_cfg["root"]),
            data_cfg.get("resolution", "480p"),
        )
        sequences = args.sequences or dataset.sequence_names(data_cfg.get("cohort_year"))
        estimator = of.RaftEstimator(
            cfg["backend"],
            str(resolve(cfg["checkpoint"])),
            cfg["device"],
            cfg["iters"],
        )
    except (MissingInputError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    records = []
    for name in sequences:
        sequence = dataset.sequence(name)
        images = [load_rgb(path) for path in sequence.frames]
        annotations = [
            object_masks(load_index_mask(path), data_cfg.get("merge_objects", True))
            for path in sequence.annotations
        ]

        for frame_index in range(len(images) - 1):
            forward_flow = estimator(images[frame_index], images[frame_index + 1])
            backward_flow = estimator(images[frame_index + 1], images[frame_index])
            residual = of.forward_backward_residual(forward_flow, backward_flow)

            for object_id, mask in annotations[frame_index].items():
                metrics = of.frame_pair_summary(forward_flow, None, mask)
                masked_residual = np.where(mask, residual, np.nan)
                metrics["fb_violation_rate"] = of.consistency_violation_rate(
                    masked_residual,
                    cfg["fb_threshold_px"],
                )
                records.append(
                    {
                        "sequence": name,
                        "object_id": object_id,
                        "frame_index": frame_index,
                        **metrics,
                    }
                )

    output_dir = resolve(cfg["output"]["dir"])
    write_csv_records(output_dir / "flow_pairs.csv", records)

    magnitudes = np.array([row["mean_magnitude_px"] for row in records], dtype=float)
    write_json(
        output_dir / "summary.json",
        {
            "n_rows": len(records),
            "mean_object_flow_px": describe(magnitudes),
            "huber_location_px": huber_location(magnitudes, cfg["huber_delta"]),
            "backend": cfg["backend"],
            "checkpoint": cfg["checkpoint"],
            "provenance": provenance([], data_kind="new_experiment", config=args.config),
        },
    )

    print(f"{len(records)} flow records -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
