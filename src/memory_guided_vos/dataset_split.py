from __future__ import annotations

import numpy as np

WEIGHTS = (0.4, 0.4, 0.2)


def zscore(x: np.ndarray, ref: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    ref = x if ref is None else np.asarray(ref, dtype=float)
    return (x - ref.mean()) / (ref.std(ddof=0) + eps)


def complexity_score(
    mean_flow: np.ndarray,
    low_vis_frac: np.ndarray,
    area_var: np.ndarray,
    weights: tuple[float, float, float] = WEIGHTS,
) -> np.ndarray:
    w1, w2, w3 = weights
    return (
        w1 * zscore(mean_flow)
        + w2 * zscore(low_vis_frac)
        + w3 * zscore(area_var)
    )


def complexity_terciles(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    lo, hi = np.percentile(c, [33.33, 66.67])
    return np.where(c <= lo, 0, np.where(c <= hi, 1, 2))


def stratified_split(
    names: list[str],
    strata: np.ndarray,
    sizes: tuple[int, int, int] = (30, 10, 10),
    seed: int = 42,
) -> dict[str, str]:
    names = list(names)
    strata = np.asarray(strata)

    if len(names) != strata.size or len(set(names)) != len(names):
        raise ValueError("names must be unique and aligned with strata")

    n = len(names)
    if sum(sizes) != n:
        raise ValueError(f"sizes {sizes} do not sum to {n} sequences")

    rng = np.random.default_rng(seed)
    partitions = ("development", "validation", "test")
    random_order = rng.permutation(n)
    order = sorted(range(n), key=lambda i: (strata[i], random_order[i]))
    assigned = np.zeros(3)
    result = {}

    for position, index in enumerate(order):
        deficit = np.asarray(sizes) * (position + 1) / n - assigned
        partition_index = int(np.argmax(deficit))
        assigned[partition_index] += 1
        result[names[index]] = partitions[partition_index]

    return result


def check_partition_integrity(
    assignments: dict[str, str],
    records: list[dict],
    key: str = "sequence",
) -> list[str]:
    seen: dict[str, set] = {}
    for record in records:
        sequence = record[key]
        seen.setdefault(sequence, set()).add(
            record.get("partition", assignments.get(sequence))
        )
    return [
        f"{sequence}: {sorted(partitions)}"
        for sequence, partitions in seen.items()
        if len(partitions) > 1
    ]
