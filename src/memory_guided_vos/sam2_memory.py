from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .memory_schedule import hydra_overrides
from .prompting import make_prompt


CONFIDENCE_SOURCES = ("object_score", "mask_prob_mean")


@dataclass
class Sam2Config:
    model_cfg: str
    checkpoint: str
    device: str = "cuda"
    confidence_source: str = ""
    memory_mechanism: str = "memory_temporal_stride"
    memory_interval: int = 1
    offload_video_to_cpu: bool = False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Sam2VideoSegmenter:
    def __init__(self, cfg: Sam2Config):
        if cfg.confidence_source not in CONFIDENCE_SOURCES:
            raise ValueError(f"confidence_source must be one of {CONFIDENCE_SOURCES}")

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as exc:
            raise ImportError(
                "SAM-2 is required: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from exc

        self.cfg = cfg
        overrides = hydra_overrides(cfg.memory_mechanism, cfg.memory_interval)
        self.predictor = build_sam2_video_predictor(
            cfg.model_cfg,
            cfg.checkpoint,
            device=cfg.device,
            hydra_overrides_extra=overrides,
        )

    def prepare(
        self,
        frames_dir: str,
        ref_frame: int,
        ref_mask: np.ndarray,
        prompt_kind: str,
        obj_id: int = 1,
    ):
        import torch

        prompt = make_prompt(prompt_kind, ref_mask)

        with torch.inference_mode():
            state = self.predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=self.cfg.offload_video_to_cpu,
            )

            if prompt["kind"] == "bbox":
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=ref_frame,
                    obj_id=obj_id,
                    box=np.asarray(prompt["box"], dtype=np.float32),
                )
            elif prompt["kind"] == "center_point":
                self.predictor.add_new_points_or_box(
                    state,
                    frame_idx=ref_frame,
                    obj_id=obj_id,
                    points=np.asarray(prompt["points"], dtype=np.float32),
                    labels=np.asarray(prompt["labels"], dtype=np.int32),
                )
            elif prompt["kind"] == "mask_only":
                self.predictor.add_new_mask(
                    state,
                    frame_idx=ref_frame,
                    obj_id=obj_id,
                    mask=prompt["mask"],
                )

        return state

    def propagate(self, state, obj_id: int = 1) -> Iterator[dict]:
        import torch
