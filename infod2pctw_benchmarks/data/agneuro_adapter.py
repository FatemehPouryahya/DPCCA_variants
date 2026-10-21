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

    @classmethod
    def from_jsonable(cls, value: dict[str, Any]) -> "PreprocessingStats":
        return cls(
            np.asarray(value["x_mean"], dtype=np.float64),
            np.asarray(value["x_scale"], dtype=np.float64),
            np.asarray(value["y_mean"], dtype=np.float64),
            np.asarray(value["y_scale"], dtype=np.float64),
            int(value["valid_count"]),
        )


@dataclass(frozen=True)
class PredefinedTrialDatasets:
    train: "BenchmarkDataset"
    validation: "BenchmarkDataset"
    test: "BenchmarkDataset"
    source_trial_ids: tuple[str, ...]
    assignments: dict[str, str]

    def to_jsonable(self) -> dict[str, Any]:
        split_ids = {
            "train": list(self.train.sequence_ids),
            "validation": list(self.validation.sequence_ids),
            "test": list(self.test.sequence_ids),
        }
        return {
            "schema_version": "agneuro-predefined-trial-splits-v1",
            "source_trial_ids": list(self.source_trial_ids),
            "assignments": dict(self.assignments),
            "splits": split_ids,
            "counts": {name: len(ids) for name, ids in split_ids.items()},
            "observation_dimensions": {
                "x": int(self.train.x.shape[-1]),
                "y": int(self.train.y.shape[-1]),
            },
            "normalization_fit_split": "train",
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

    def select_sequences(self, indices: list[int], split_name: str | None = None) -> "BenchmarkDataset":
        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or len(selected) == 0:
            raise ValueError("sequence selection must contain at least one index")
        if np.any(selected < 0) or np.any(selected >= self.x.shape[0]):
            raise IndexError("sequence selection is out of range")
        metadata = dict(self.metadata)
        selected_ids = tuple(self.sequence_ids[index] for index in selected)
        for key in ("recording_ids", "trial_ids", "sequence_ids"):
            values = metadata.get(key)
            if isinstance(values, list) and len(values) == self.x.shape[0]:
                metadata[key] = [values[index] for index in selected]
        if split_name is not None:
            metadata["selected_predefined_split"] = split_name
        return BenchmarkDataset(
            self.x[selected].copy(), self.y[selected].copy(), self.mask[selected].copy(),
            selected_ids, metadata,
        )

    def standardized(
        self, standardize_x: bool = True, standardize_y: bool = True,
        statistics: PreprocessingStats | None = None,
    ) -> "BenchmarkDataset":
        valid = self.mask

        if statistics is not None:
            expected_shapes = (
                (statistics.x_mean, self.x.shape[-1], "x_mean"),
                (statistics.x_scale, self.x.shape[-1], "x_scale"),
                (statistics.y_mean, self.y.shape[-1], "y_mean"),
                (statistics.y_scale, self.y.shape[-1], "y_scale"),
            )
            for value, dimension, name in expected_shapes:
                if np.asarray(value).shape != (dimension,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"invalid fitted preprocessing statistic: {name}")
            if np.any(statistics.x_scale <= 0.0) or np.any(statistics.y_scale <= 0.0):
                raise ValueError("fitted preprocessing scales must be positive")

        def stats_and_apply(
            values: np.ndarray, enabled: bool,
            fitted_mean: np.ndarray | None, fitted_scale: np.ndarray | None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not enabled:
                mean = np.zeros(values.shape[-1], dtype=np.float64)
                scale = np.ones(values.shape[-1], dtype=np.float64)
                return values.copy(), mean, scale
            if fitted_mean is None or fitted_scale is None:
                flat = values[valid]
                mean = flat.mean(axis=0)
                scale = flat.std(axis=0)
                scale = np.where(scale > 1e-12, scale, 1.0)
            else:
                mean = np.asarray(fitted_mean, dtype=np.float64)
                scale = np.asarray(fitted_scale, dtype=np.float64)
            transformed = (values - mean) / scale
            transformed = np.where(valid[..., None], transformed, 0.0)
            return transformed, mean, scale

        x, x_mean, x_scale = stats_and_apply(
            self.x, standardize_x,
            statistics.x_mean if statistics is not None else None,
            statistics.x_scale if statistics is not None else None,
        )
        y, y_mean, y_scale = stats_and_apply(
            self.y, standardize_y,
            statistics.y_mean if statistics is not None else None,
            statistics.y_scale if statistics is not None else None,
        )
        stats = statistics or PreprocessingStats(
            x_mean, x_scale, y_mean, y_scale, int(valid.sum())
        )
        return BenchmarkDataset(x, y, valid.copy(), self.sequence_ids, dict(self.metadata), stats)


def predefined_trial_datasets(
    dataset: BenchmarkDataset, standardize_x: bool = True, standardize_y: bool = True,
    statistics: PreprocessingStats | None = None,
    persisted_assignments: dict[str, str] | None = None,
) -> PredefinedTrialDatasets | None:
    """Apply AgNeuro's predefined whole-trial split, if one is declared."""
    joint_metadata = dataset.metadata.get("joint_dataset_metadata")
    if persisted_assignments is None and (
        not isinstance(joint_metadata, dict) or "trial_splits" not in joint_metadata
    ):
        return None
    assignments = (
        persisted_assignments if persisted_assignments is not None
        else joint_metadata["trial_splits"]
    )
    trial_ids = dataset.metadata.get("trial_ids")
    if not isinstance(trial_ids, list) or len(trial_ids) != dataset.x.shape[0]:
        raise ValueError("predefined splits require one metadata trial_id per observation trial")
    trial_ids = tuple(str(value) for value in trial_ids)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("metadata trial_ids must be unique")
    if trial_ids != dataset.sequence_ids:
        raise ValueError("metadata trial_ids do not match observation ordering")
    if not isinstance(assignments, dict) or set(map(str, assignments)) != set(trial_ids):
        raise ValueError("trial_splits must assign every trial_id exactly once")
    assignments = {str(key): str(value) for key, value in assignments.items()}
    allowed = {"train", "validation", "test"}
    if set(assignments.values()) - allowed:
        raise ValueError("trial_splits contains an unknown split label")
    if dataset.x.shape[-1] != 32 or dataset.y.shape[-1] != 16:
        raise ValueError(
            "AgNeuro predefined trials must have observation dimensions x=32 and y=16"
        )
    if not np.all(dataset.mask.any(axis=1)):
        raise ValueError("every predefined trial must contain at least one valid observation")

    indices = {
        name: [index for index, trial_id in enumerate(trial_ids) if assignments[trial_id] == name]
        for name in ("train", "validation", "test")
    }
    counts = {name: len(values) for name, values in indices.items()}
    if counts != {"train": 36, "validation": 12, "test": 12}:
        raise ValueError(f"expected predefined split counts 36/12/12, got {counts}")

    train = dataset.select_sequences(indices["train"], "train").standardized(
        standardize_x, standardize_y, statistics
    )
    fitted_statistics = train.preprocessing_stats
    if fitted_statistics is None:
        raise RuntimeError("training preprocessing statistics were not fitted")
    validation = dataset.select_sequences(indices["validation"], "validation").standardized(
        standardize_x, standardize_y, fitted_statistics
    )
    test = dataset.select_sequences(indices["test"], "test").standardized(
        standardize_x, standardize_y, fitted_statistics
    )
    return PredefinedTrialDatasets(train, validation, test, trial_ids, assignments)


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
