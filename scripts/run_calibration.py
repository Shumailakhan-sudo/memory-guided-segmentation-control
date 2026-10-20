from __future__ import annotations

import argparse
import sys

from _common import resolve

from memory_guided_vos.evaluation import calibration_summary
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    read_csv_records,
    write_csv_records,
    write_json,
)
from memory_guided_vos.records import validate_header


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/confidence.toml")
    parser.add_argument("--records", help="Input CSV path")
    parser.add_argument(
        "--data-kind",
        default="original_records",
        choices=["original_records", "new_experiment", "synthetic"],
    )
    parser.add_argument("--out", help="Output directory")
    args = parser.parse_args(argv)

    config = load_config(resolve(args.config))
    records_path = resolve(args.records or config["records"])

    try:
        rows = read_csv_records(records_path)
    except MissingInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    missing_columns = validate_header(
        "calibration_pairs.csv", list(rows[0].keys()) if rows else []
    )
    if missing_columns:
        print(
            f"ERROR: {records_path} lacks required columns {missing_columns}",
            file=sys.stderr,
        )
        return 2

    bootstrap = config["bootstrap"]
    results = calibration_summary(
        rows,
        eta=config["eta"],
        n_bins=config["n_bins"],
        thresholds=config["thresholds"],
        n_boot=bootstrap["n_resamples"],
        seed=bootstrap["seed"],
    )
    results["provenance"] = provenance(
        [records_path], data_kind=args.data_kind, config=args.config
    )

    output_dir = resolve(args.out or config["output"]["dir"]) / args.data_kind
    write_json(output_dir / "calibration_summary.json", results)
    write_csv_records(output_dir / "reliability_bins.csv", results["bins"])
    write_csv_records(
        output_dir / "gate_outcomes.csv",
        [
            {key: value for key, value in gate.items() if key != "bootstrap"}
            for gate in results["gates"]
        ],
    )

    print(
        f"N={results['n']} correct={results['n_correct']} "
        f"ECE={results['ece']:.5f} -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
