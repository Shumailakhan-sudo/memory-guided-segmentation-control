from __future__ import annotations

import numpy as np

INTERVALS = (1, 2, 3, 5)
MECHANISMS = ("memory_temporal_stride", "external_schedule")


def update_frames(n_frames: int, u: int, start: int = 0) -> np.ndarray:
    if u < 1:
        raise ValueError("u must be >= 1")
    return np.arange(start, n_frames, u)


def is_update_frame(t: int, u: int, start: int = 0) -> bool:
    return t >= start and (t - start) % u == 0


def nominal_update_hz(u: int, fps: float = 24.0) -> float:
    return fps / u


def hydra_overrides(mechanism: str, u: int) -> list[str]:
    if mechanism == "memory_temporal_stride":
        return [f"++model.memory_temporal_stride_for_eval={int(u)}"]
    if mechanism == "external_schedule":
        raise NotImplementedError(
            "External memory scheduling requires a custom SAM-2 implementation."
        )
    raise ValueError(f"mechanism must be one of {MECHANISMS}")
