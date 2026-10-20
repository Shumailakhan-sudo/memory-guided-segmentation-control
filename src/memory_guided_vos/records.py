from __future__ import annotations

from pathlib import Path

SCHEMAS: dict[str, dict] = {
    "segmentation_frames.csv": {
        "unit": "one frame, object, and prompt strategy",
        "required": ["sequence", "object_id", "frame_index", "strategy", "jaccard", "boundary_f"],
        "optional": [
            "source_frame_index", "nonempty_pred", "confidence", "confidence_source",
            "configuration", "partition",
        ],
    },
    "calibration_pairs.csv": {
        "unit": "one prediction with a valid confidence score",
        "required": ["sequence", "frame_index", "confidence", "jaccard", "nonempty_pred"],
        "optional": ["object_id", "strategy", "configuration", "confidence_source"],
    },
    "timing_frames.csv": {
        "unit": "one processed frame",
        "required": ["frame_index", "preprocess", "inference", "postprocess", "publish"],
        "optional": ["sequence", "strategy", "total", "warmup", "synchronised", "device"],
    },
    "flow_observations.csv": {
        "unit": "one flow observation; motion-group definition is dataset-specific",
        "required": ["motion_group", "sequence", "magnitude_px"],
        "optional": ["frame_index", "fb_residual_px"],
    },
    "centroid_stability.csv": {
        "unit": "one motion group and temporal window scale",
        "required": ["motion_group", "tau", "allan_variance_px2"],
        "optional": ["sequence", "K"],
    },
    "trajectories.csv": {
        "unit": "one motion group",
        "required": ["motion_group", "sequence", "path_length_px"],
        "optional": [],
    },
    "interruption_delays.csv": {
        "unit": "one delay observation; delay definition is dataset-specific",
        "required": ["sequence", "delay_frames"],
        "optional": ["object_id", "strategy", "gap_frames", "t_onset", "t_visible", "definition"],
    },
    "configuration_frames.csv": {
        "unit": "one frame and configuration",
        "required": ["configuration", "sequence", "frame_index", "jaccard", "boundary_f"],
        "optional": ["accepted", "confidence", "centroid_x", "centroid_y"],
    },
    "cartesian_episodes.csv": {
        "unit": "one simulated episode",
        "required": ["episode", "success", "confidence", "episode_s", "path_m"],
        "optional": ["grasp_s", "occluded", "retries", "final_action", "states"],
    },
    "reward_episodes.csv": {
        "unit": "one reward episode",
        "required": ["episode", "duration_s", "cumulative_reward"],
        "optional": [
            "method", "median_reward", "q1", "q3", "success", "efficiency",
            "relative_performance",
        ],
    },
}


def validate_header(file_name: str, header: list[str]) -> list[str]:
    if file_name not in SCHEMAS:
        raise KeyError(f"No schema for {file_name}")
    return [column for column in SCHEMAS[file_name]["required"] if column not in header]


def available_records(original_dir: str | Path) -> dict[str, bool]:
    directory = Path(original_dir)
    return {name: (directory / name).is_file() for name in SCHEMAS}
