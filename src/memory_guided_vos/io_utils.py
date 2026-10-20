from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import __version__

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_KINDS = {
    "original_records",
    "new_experiment",
    "imported_records",
    "synthetic",
    "unspecified",
}


class MissingInputError(FileNotFoundError):
    pass


def repo_relative(path: str | Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def require_file(path: str | Path, what: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise MissingInputError(f"{what} not found: {path}")
    return path


def load_config(path: str | Path) -> dict:
    with require_file(path, "Configuration file").open("rb") as file:
        return tomllib.load(file)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while block := file.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def provenance(
    inputs: Iterable[str | Path] = (),
    data_kind: str = "unspecified",
    **extra: Any,
) -> dict:
    if data_kind not in DATA_KINDS:
        raise ValueError(f"data_kind must be one of {sorted(DATA_KINDS)}, got {data_kind!r}")

    files = []
    for item in inputs:
        path = Path(item)
        files.append({
            "path": repo_relative(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        })

    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "data_kind": data_kind,
        "inputs": files,
        **extra,
    }


def _json_default(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, Path):
        return value.as_posix()