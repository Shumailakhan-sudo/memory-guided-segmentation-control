from __future__ import annotations

import numpy as np
from scipy import ndimage


class UndefinedPromptError(NotImplementedError):
    pass


def first_valid_reference(masks: list[np.ndarray]) -> int:
    for index, mask in enumerate(masks):
        if np.any(mask):
            return index
    raise ValueError("No non-empty reference mask in sequence")


def bbox_prompt(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0:
        raise ValueError("Cannot derive a bounding box from an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def center_point_prompt(mask: np.ndarray) -> tuple[int, int]:
    binary_mask = np.asarray(mask, dtype=bool)
    if not binary_mask.any():
        raise ValueError("Cannot derive a centre point from an empty mask")
    distances = ndimage.distance_transform_edt(np.pad(binary_mask, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return int(x), int(y)


def make_prompt(kind: str, mask: np.ndarray) -> dict:
    if kind == "bbox":
        return {"kind": "bbox", "box": bbox_prompt(mask)}
    if kind == "center_point":
        x, y = center_point_prompt(mask)
        return {"kind": "center_point", "points": [(x, y)], "labels": [1]}
    if kind == "none":
        raise UndefinedPromptError("The 'none' initialization mode is not supported")
    if kind == "mask_only":
        return {"kind": "mask_only", "mask": np.asarray(mask, dtype=bool)}
    raise ValueError(f"Unknown prompt kind: {kind!r}")
