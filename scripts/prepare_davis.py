from __future__ import annotations

import argparse
import sys
from collections import Counter

from _common import resolve
from memory_guided_vos.dataset import DavisDataset, build_manifest
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    write_csv_records,
    write_json,
)
from memory_guided_vos.preprocessing import (
    align_to_reference_rate,
    alignment_report,
    short_side_scale,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a dataset manifest.")
    parser.add_argument("--config", default="configs/davis.toml")
    parser.add_argument("--root", help="Dataset directory")
    parser.add_argument("--manifest", help="Manifest output path")
    parser.add_argument(
        "--data-kind",
        default="new_experiment",
        choices=["new_experiment", "synthetic"],
    )
    args = parser.parse_args(argv)

    config = load_config(resolve(args.config))
    root = resolve(args.root or config["root"])

    try:
        dataset = DavisDataset(root, config.get("resolution", "480p"))
        sequences = dataset.sequence_names(config.get("cohort_year"))
    except MissingInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    records = build_manifest(dataset, sequences)
    for record in records:
        sequence = dataset.sequence(record["sequence"])
        aligned = align_to_reference_rate(
            sequence.timestamps(), config["reference_fps"], sequence.n_frames
        )
        report = alignment_report(aligned, sequence.n_frames)
        record.update(
            {f"align_{key}": value for key, value in report.items() if key != "n_source_frames"}
        )
        _, (height, width) = short_side_scale(
            record["height"], record["width"], config["short_side"]
        )
        record["resized_height"] = height
        record["resized_width"] = width

    output = resolve(args.manifest or config["output"]["manifest"])
    write_csv_records(output, records)

    duplicates = [
        digest
        for digest, count in Counter(record["frames_sha256"] for record in records).items()
        if count > 1
    ]
    summary = {
        "n_sequences": len(records),
        "n_frames_total": sum(record["n_frames"] for record in records),
        "n_annotations_total": sum(record["n_annotations"] for record in records),
        "duplicate_frame_hashes": duplicates,
        "all_alignment_identity": all(record["align_is_identity"] for record in records),
        "provenance": provenance(
            [], data_kind=args.data_kind, dataset_root=str(root), config=args.config
        ),
    }
    write_json(output.with_suffix(".json"), summary)

    print(f"{len(records)} sequences, {summary['n_frames_total']} frames -> {output}")
    if duplicates:
        print(f"WARNING: {len(duplicates)} duplicate frame-content hashes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
