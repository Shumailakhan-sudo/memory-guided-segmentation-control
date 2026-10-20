from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

import numpy as np

STAGES = ("preprocess", "inference", "postprocess", "publish")
DEADLINE_MS = 33.0


class StageTimer:
    def __init__(
        self,
        sync: Callable[[], None] | None = None,
        warmup_frames: int = 0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.sync = sync
        self.warmup_frames = warmup_frames
        self.clock = clock
        self.frames: list[dict[str, float]] = []
        self._current: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        if name not in STAGES:
            raise ValueError(f"Unknown stage {name!r}; expected one of {STAGES}")
        if self.sync:
            self.sync()
        start = self.clock()
        try:
            yield
        finally:
            if self.sync:
                self.sync()
            self._current[name] = (self.clock() - start) * 1000.0

    def end_frame(self) -> None:
        missing = [stage for stage in STAGES if stage not in self._current]
        if missing:
            raise RuntimeError(f"Frame ended without timing stages {missing}")
        record = dict(self._current)
        record["total"] = sum(record[stage] for stage in STAGES)
        self.frames.append(record)
        self._current = {}

    def records(self, include_warmup: bool = False) -> list[dict]:
        start = 0 if include_warmup else self.warmup_frames
        return [
            dict(frame_index=index, warmup=index < self.warmup_frames, **record)
            for index, record in enumerate(self.frames)
            if index >= start
        ]

    def metadata(self) -> dict:
        return {
            "synchronised": self.sync is not None,
            "warmup_frames_excluded": self.warmup_frames,
            "clock": getattr(self.clock, "__name__", str(self.clock)),
        }


def latency_summary(totals_ms: np.ndarray, deadline_ms: float = DEADLINE_MS) -> dict:
    samples = np.asarray(totals_ms, dtype=float)
    if samples.size == 0:
        raise ValueError("No latency samples")

    p50, p90, p95, p99 = np.percentile(samples, [50, 90, 95, 99])
    mean = float(samples.mean())
    return {
        "n_frames": int(samples.size),
        "mean_ms": mean,
        "median_ms": float(p50),
        "sd_ms": float(samples.std(ddof=1)) if samples.size > 1 else None,
        "p90_ms": float(p90),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(samples.max()),
        "nominal_fps": 1000.0 / mean,
        "deadline_ms": deadline_ms,
        "deadline_compliance": float(np.mean(samples <= deadline_ms)),
        "n_deadline_misses": int(np.sum(samples > deadline_ms)),
    }


def stage_shares(stage_means_ms: dict[str, float]) -> dict[str, float]:
    total = sum(stage_means_ms[stage] for stage in STAGES)
    return {stage: stage_means_ms[stage] / total for stage in STAGES}


def environment_info() -> dict:
    import platform

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info
