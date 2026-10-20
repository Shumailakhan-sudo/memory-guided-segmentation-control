from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .io_utils import MissingInputError

DAVIS_FPS = 24.0


@dataclass
class Sequence:
    name: str
    frames: list[Path]
    annotations: list[Path]
    official_splits: dict[str, str] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def timestamps(self) -> np.ndarray:
        return np.arange(self.n_frames) / DAVIS_FPS


class DavisDataset:
    def __init__(self, root: str | Path, resolution: str = "480p"):
        self.root = Path(root)
        self.resolution = resolution
        self.img_dir = self.root / "JPEGImages" / resolution
        self.ann_dir = self.root / "Annotations" / resolution

        for directory in (self.img_dir, self.ann_dir):
            if not directory.is_dir():
                raise MissingInputError(f"Required DAVIS directory not found: {directory}")

    def official_split(self, year: str) -> dict[str, list[str]]:
        splits = {}
        for subset in ("train", "val"):
            path = self.root / "ImageSets" / year / f"{subset}.txt"
            if path.is_file():
                splits[subset] = [
                    name.strip()
                    for name in path.read_text(encoding="utf-8").splitlines()
                    if name.strip()
                ]
        return splits

    def sequence_names(self, year: str | None = None) -> list[str]:
        if year is not None:
            splits = self.official_split(year)
            if not splits:
                raise MissingInputError(
                    f"No train or validation split files found in ImageSets/{year} under {self.root}"
                )
            return sorted(set(splits.get("train", []) + splits.get("val", [])))

        return sorted(path.name for path in self.img_dir.iterdir() if path.is_dir())

    def sequence(self, name: str) -> Sequence:
        frames = sorted((self.img_dir / name).glob("*.jpg"))
        annotations = sorted((self.ann_dir / name).glob("*.png"))

        if not frames:
            raise MissingInputError(f"No frames found for sequence '{name}' in {self.img_dir / name}")

        splits = {}
        for year in ("2016", "2017"):
            for subset, names in self.official_split(year).items():
                if name in names:
                    splits[year] = subset

        return Sequence(
            name=name,
            frames=frames,
            annotations=annotations,
            official_splits=splits,
        )


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_index_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def object_masks(label: np.ndarray, merge: bool = False) -> dict[int, np.ndarray]:
    label = np.asarray(label)
    if merge:
        return {1: (label > 0) & (label != 255)}