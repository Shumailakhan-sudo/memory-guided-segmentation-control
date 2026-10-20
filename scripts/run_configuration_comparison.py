from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict

import numpy as np

from _common import resolve
from memory_guided_vos.evaluation import configuration_summary
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    read_csv_records,
    write_json,
)


def paired_differences(rows: list[dict]) -> list[dict]:
    per = defaultdict(lambda: defaultdict(list))
    for row in rows:
        per[row["configuration"]][row["sequence"]].append(
            (float(row["jaccard"]), float(row["boundary_f"]))
        )

    results = []
    for a, b in itertools.combinations(sorted(per), 2):
        shared = sorted(set(per[a]) & set(per[b]))
        delta_j = [
            np.mean([value[0] for value in per[a][sequence]])
            - np.mean([value[0] for value in per[b][sequence]])
            for sequence in shared
        ]
        delta_f = [
            np.mean([value[1] for value in per[a][sequence]])
            - np.mean([value[1] for value in per[b][sequence]])
            for sequence in shared
        ]
        results.append(
            {
                "a": a,
                "b": b,
                "n_sequences": len(shared),
                "mean_delta_j": float(np.mean(delta_j)) if delta_j else None,
                "mean_delta_f": float(np.mean(delta_f)) if delta_f else None,
            }
        )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics across configurations."
    )
    parser.add_argument("--config", default="configs/configurations.toml")
    parser.add_argument("--records")
    parser.add_argument(
        "--data-kind",
        default="original_records",
        choices=["original_records", "new_experiment", "synthetic"],
    )
    parser.add_argument("--out", help="Output directory")
    args = parser.parse_args(argv)

    config = load_config(resolve(args.config))
    path = resolve(args.records or config["records"])

    try:
        rows = read_csv_records(path)
    except MissingInputError as exc:
        print(f"ERROR: {exc}\nConfiguration records are required.", file=sys.stderr)
        return 2

    results = {
        "configurations": config["configuration"],
        "summary": configuration_summary(rows),
        "paired_differences": paired_differences(rows),
        "provenance": provenance(
            [path], data_kind=args.data_kind, config=args.config
        ),
    }

    output = (
        resolve(args.out or config["output"]["dir"])
        / args.data_kind
        / "configuration_summary.json"
    )
    write_json(output, results)
    print(f"{len(rows)} rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
