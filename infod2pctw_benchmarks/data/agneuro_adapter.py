"""Strict AgNeuro ``arrays.npz``/``metadata.json`` loading.

The canonical in-memory layout is always ``[batch, time, feature]``.  Invalid
timestamps are retained in place; adapters may iterate contiguous valid spans
when an author implementation has no mask-aware likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np


@dataclass(frozen=True)
class PreprocessingStats:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    valid_count: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_scale": self.y_scale.tolist(),
            "valid_count": self.valid_count,
            "convention": "per-feature mean/std over the common valid-time mask",
        }


@dataclass(frozen=True)
class BenchmarkDataset:
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    sequence_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    preprocessing_stats: PreprocessingStats | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        mask = np.asarray(self.mask, dtype=bool)
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("x and y must use canonical [B,T,D] layout")
        if x.shape[:2] != y.shape[:2] or mask.shape != x.shape[:2]:
            raise ValueError("x, y and mask must agree in [B,T]")
        if len(self.sequence_ids) != x.shape[0]:
            raise ValueError("sequence_ids must contain exactly one ID per sequence")
        if not np.all(np.isfinite(x[mask])) or not np.all(np.isfinite(y[mask])):
            raise ValueError("valid observations must be finite")
        object.__setattr__(self, "x", x.astype(np.float64, copy=False))
        object.__setattr__(self, "y", y.astype(np.float64, copy=False))
        object.__setattr__(self, "mask", mask)

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape[:2]

    @property
    def name(self) -> str:
        return str(self.metadata.get("dataset_id", self.metadata.get("name", "dataset"))).replace("/", "_")

    def as_layout(self, array: Literal["x", "y"], layout: Literal["BTD", "TBD", "TD"] = "BTD") -> np.ndarray:
        value = self.x if array == "x" else self.y
        if layout == "BTD":
            return value.copy()
        if layout == "TBD":
            return np.transpose(value, (1, 0, 2)).copy()
        if layout == "TD":
            if value.shape[0] != 1:
                raise ValueError("TD conversion is only defined for a single sequence")
            return value[0].copy()
        raise ValueError(f"unknown layout: {layout}")

    def valid_segments(self, minimum_length: int = 2) -> Iterator[tuple[int, slice, np.ndarray, np.ndarray]]:
        """Yield contiguous valid spans without turning timepoints into samples."""
        for batch_index, row in enumerate(self.mask):
            edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
            starts = np.flatnonzero(edges == 1)
            stops = np.flatnonzero(edges == -1)
            for start, stop in zip(starts, stops):
                if stop - start >= minimum_length:
                    span = slice(int(start), int(stop))
                    yield batch_index, span, self.x[batch_index, span], self.y[batch_index, span]

    def standardized(self, standardize_x: bool = True, standardize_y: bool = True) -> "BenchmarkDataset":
        valid = self.mask

        def stats_and_apply(values: np.ndarray, enabled: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not enabled:
                mean = np.zeros(values.shape[-1], dtype=np.float64)
                scale = np.ones(values.shape[-1], dtype=np.float64)
                return values.copy(), mean, scale
            flat = values[valid]
            mean = flat.mean(axis=0)
            scale = flat.std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            transformed = (values - mean) / scale
            transformed = np.where(valid[..., None], transformed, 0.0)
            return transformed, mean, scale

        x, x_mean, x_scale = stats_and_apply(self.x, standardize_x)
        y, y_mean, y_scale = stats_and_apply(self.y, standardize_y)
        stats = PreprocessingStats(x_mean, x_scale, y_mean, y_scale, int(valid.sum()))
        return BenchmarkDataset(x, y, valid.copy(), self.sequence_ids, dict(self.metadata), stats)


def _canonicalize_observations(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 2:
        value = value[None, ...]
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [B,T,D] or [T,D], got {value.shape}")
    return value


def _sequence_ids(metadata: dict[str, Any], batch_size: int) -> tuple[str, ...]:
    for key in ("recording_ids", "trial_ids", "sequence_ids", "session_ids"):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == batch_size:
            return tuple(str(value) for value in values)
    return tuple(f"sequence_{index:04d}" for index in range(batch_size))


def load_agneuro(path: str | Path) -> BenchmarkDataset:
    root = Path(path)
    arrays_path = root / "arrays.npz" if root.is_dir() else root
    metadata_path = arrays_path.with_name("metadata.json")
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        if "x" not in arrays or "y" not in arrays:
            raise KeyError("arrays.npz must contain x and y")
        x = _canonicalize_observations(arrays["x"], "x")
        y = _canonicalize_observations(arrays["y"], "y")
        if "mask" in arrays:
            mask = np.asarray(arrays["mask"], dtype=bool)
        elif "mask_x" in arrays and "mask_y" in arrays:
            mask = np.asarray(arrays["mask_x"], dtype=bool) & np.asarray(arrays["mask_y"], dtype=bool)
        elif "lengths" in arrays:
            lengths = np.asarray(arrays["lengths"]).reshape(-1)
            mask = np.arange(x.shape[1])[None, :] < lengths[:, None]
        else:
            mask = np.ones(x.shape[:2], dtype=bool)
    if mask.ndim == 1:
        mask = mask[None, :]
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata = dict(metadata)
    metadata.setdefault("source_arrays", str(arrays_path.resolve()))
    metadata.setdefault("source_metadata", str(metadata_path.resolve()) if metadata_path.exists() else None)
    return BenchmarkDataset(x, y, mask, _sequence_ids(metadata, x.shape[0]), metadata)
