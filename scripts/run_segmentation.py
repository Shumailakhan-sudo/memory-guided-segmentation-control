from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from _common import resolve
from memory_guided_vos import metrics
from memory_guided_vos.dataset import DavisDataset, load_index_mask, object_masks
from memory_guided_vos.io_utils import (
    MissingInputError,
    load_config,
    provenance,
    require_file,
    write_csv_records,
    write_json,
)
from memory_guided_vos.prompting import first_valid_reference
from memory_guided_vos.runtime import STAGES, environment_info, latency_summary


STAGE_BOUNDARIES = (
    "preprocess: state initialization and prompt registration, distributed across frames; "
    "inference: retrieval of each propagated prediction; "
    "postprocess: extraction of prediction and reference masks; "
    "publish: metric calculation and output record creation."
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run video object segmentation evaluation.")
    parser.add_argument("--config", default="configs/sam2.toml")
    parser.add_argument("--sequences", nargs="*", help="Sequences to evaluate")
    parser.add_argument("--profile", action="store_true", help="Synchronize CUDA during timing")
    args = parser.parse_args(argv)

    cfg = load_config(resolve(args.config))
    data_cfg = load_config(resolve(cfg["data"]["davis_config"]))

    try:
        for key in ("model_cfg", "checkpoint", "confidence_source"):
            if not cfg.get(key):
                raise MissingInputError(f"Required configuration setting is missing: {key}")

        require_file(resolve(cfg["checkpoint"]), "SAM-2 checkpoint")
        dataset = DavisDataset(
            resolve(data_cfg["root"]), data_cfg.get("resolution", "480p")
        )
        names = args.sequences or dataset.sequence_names(data_cfg.get("cohort_year"))

        import torch

        from memory_guided_vos.sam2_memory import Sam2Config, Sam2VideoSegmenter
    except (MissingInputError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    sync = torch.cuda.synchronize if args.profile and torch.cuda.is_available() else None
    records = []
    timing = []

    for interval in cfg["memory_intervals"]:
        segmenter = Sam2VideoSegmenter(
            Sam2Config(
                cfg["model_cfg"],
                str(resolve(cfg["checkpoint"])),
                cfg["device"],
                cfg["confidence_source"],
                cfg["memory_mechanism"],
                interval,
            )
        )

        for name in names:
            sequence = dataset.sequence(name)
            labels = [load_index_mask(path) for path in sequence.annotations]
            ground_truth = [
                object_masks(label, data_cfg.get("merge_objects", True))
                for label in labels
            ]

            for object_id in sorted({oid for frame in ground_truth for oid in frame}):
                object_masks_by_frame = [
                    frame.get(object_id, np.zeros(label.shape, dtype=bool))
                    for frame, label in zip(ground_truth, labels)
                ]
                reference_frame = first_valid_reference(object_masks_by_frame)

                for prompt in cfg["prompts"]:
                    if sync:
                        sync()
                    start = time.perf_counter()
                    state = segmenter.prepare(
                        str(sequence.frames[0].parent),