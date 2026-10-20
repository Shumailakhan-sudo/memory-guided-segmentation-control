from __future__ import annotations

import numpy as np
from scipy import ndimage

FB_THRESHOLD_PX = 1.0


def magnitude(flow: np.ndarray) -> np.ndarray:
    flow = np.asarray(flow, dtype=float)
    return np.hypot(flow[..., 0], flow[..., 1])


def warp_sample(
    field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = field.shape[:2]
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    coords = np.stack([y.ravel(), x.ravel()])
    sampled = []
    for channel in range(2):
        values = ndimage.map_coordinates(
            field[..., channel], coords, order=1, mode="nearest"
        ).reshape(x.shape)
        sampled.append(np.where(inside, values, np.nan))
    return sampled[0], sampled[1], inside


def forward_backward_residual(fwd: np.ndarray, bwd: np.ndarray) -> np.ndarray:
    fwd = np.asarray(fwd, dtype=float)
    bwd = np.asarray(bwd, dtype=float)
    h, w = fwd.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    target_x = xx + fwd[..., 0]
    target_y = yy + fwd[..., 1]
    back_x, back_y, _ = warp_sample(bwd, target_x, target_y)
    return np.hypot(fwd[..., 0] + back_x, fwd[..., 1] + back_y)


def consistency_violation_rate(
    residual: np.ndarray, threshold: float = FB_THRESHOLD_PX
) -> float:
    residual = np.asarray(residual, dtype=float)
    valid = ~np.isnan(residual)
    if not valid.any():
        return float("nan")
    return float(np.mean(residual[valid] > threshold))


def frame_pair_summary(
    fwd: np.ndarray,
    bwd: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    mag = magnitude(fwd)
    selected = np.ones(mag.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    result = {
        "n_pixels": int(selected.sum()),
        "mean_magnitude_px": float(mag[selected].mean()) if selected.any() else float("nan"),
        "median_magnitude_px": float(np.median(mag[selected])) if selected.any() else float("nan"),
        "max_magnitude_px": float(mag[selected].max()) if selected.any() else float("nan"),
    }
    if bwd is not None:
        residual = forward_backward_residual(fwd, bwd)
        result["fb_violation_rate"] = consistency_violation_rate(
            np.where(selected, residual, np.nan)
        )
    return result


class RaftEstimator:
    def __init__(
        self,
        backend: str = "princeton",
        checkpoint: str | None = None,
        device: str = "cuda",
        iters: int = 20,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("PyTorch is required for RAFT inference.") from exc
        self.backend = backend
        self.checkpoint = checkpoint
        self.device = device
        self.iters = iters
        self.model = self._load()

    def _load(self):
        import torch

        if self.backend == "torchvision":
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            return raft_large(weights=Raft_Large_Weights.C_T_V2).to(self.device).eval()
        if self.backend == "princeton":