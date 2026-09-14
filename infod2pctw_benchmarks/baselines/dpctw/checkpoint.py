"""Durable, DPCTW-local checkpoint storage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile
from typing import Any

import numpy as np

from data.agneuro_adapter import BenchmarkDataset


SCHEMA_VERSION = 1
PARAMETER_NAMES = (
    "shared_transition",
    "x_private_transition",
    "y_private_transition",
    "shared_process_covariance",
    "x_private_process_covariance",
    "y_private_process_covariance",
    "x_shared_loading",
    "x_private_loading",
    "y_shared_loading",
    "y_private_loading",
    "x_noise",
    "y_noise",
    "initial_mean",
    "initial_covariance",
)


def checkpoint_signature(config: dict[str, Any], dataset: BenchmarkDataset) -> str:
    """Identify checkpoint state that is safe to use for this run."""
    payload = {
        "config": config,
        "x_shape": dataset.x.shape,
        "y_shape": dataset.y.shape,
        "sequence_ids": dataset.sequence_ids,
        "mask_sha256": hashlib.sha256(dataset.mask.tobytes()).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class DPCTWCheckpointStore:
    """Keep two atomic slots so a previous valid checkpoint always remains."""

    def __init__(self, output_dir: str | Path, signature: str):
        self.directory = Path(output_dir) / "checkpoints" / "dpctw"
        self.signature = signature
        self.sequence = -1

    def load(self) -> dict[str, Any] | None:
        candidates = []
        for path in (self.directory / "checkpoint_0.npz", self.directory / "checkpoint_1.npz"):
            loaded = self._load_file(path)
            if loaded is not None:
                candidates.append(loaded)
        if not candidates:
            return None
        newest = max(candidates, key=lambda item: item["sequence"])
        self.sequence = int(newest["sequence"])
        return newest

    def _load_file(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as values:
                metadata = json.loads(str(values["metadata_json"].item()))
                if metadata.get("schema_version") != SCHEMA_VERSION:
                    return None
                if metadata.get("signature") != self.signature:
                    return None
                parameters = {
                    name: np.asarray(values[f"parameter__{name}"]).copy()
                    for name in PARAMETER_NAMES
                }
                pairs = np.asarray(values["path_pairs"], dtype=np.int64)
                offsets = np.asarray(values["path_offsets"], dtype=np.int64)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                return None
            if (
                offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0
                or offsets[-1] != len(pairs) or np.any(np.diff(offsets) < 0)
            ):
                return None
            paths = [pairs[start:stop].copy() for start, stop in zip(offsets[:-1], offsets[1:])]
            return {
                "sequence": int(metadata["sequence"]),
                "state": metadata["state"],
                "parameters": parameters,
                "paths": paths,
            }
        except (OSError, ValueError, KeyError, EOFError, json.JSONDecodeError, zipfile.BadZipFile):
            return None

    def save(
        self, state: dict[str, Any], parameters: dict[str, np.ndarray],
        paths: list[np.ndarray],
    ) -> Path:
        missing = [name for name in PARAMETER_NAMES if name not in parameters]
        if missing:
            raise ValueError(f"DPCTW checkpoint is missing parameters: {missing}")
        self.directory.mkdir(parents=True, exist_ok=True)
        sequence = self.sequence + 1
        slot = sequence % 2
        destination = self.directory / f"checkpoint_{slot}.npz"
        temporary = self.directory / f".{destination.stem}.{uuid.uuid4().hex}.tmp.npz"
        offsets = [0]
        path_values = []
        for path in paths:
            path = np.asarray(path, dtype=np.int64)
            path_values.append(path)
            offsets.append(offsets[-1] + len(path))
        arrays = {
            "metadata_json": np.asarray(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "signature": self.signature,
                "sequence": sequence,
                "state": state,
            }, sort_keys=True)),
            "path_pairs": (
                np.concatenate(path_values, axis=0)
                if path_values else np.empty((0, 2), dtype=np.int64)
            ),
            "path_offsets": np.asarray(offsets, dtype=np.int64),
        }
        arrays.update({
            f"parameter__{name}": np.asarray(parameters[name])
            for name in PARAMETER_NAMES
        })
        try:
            np.savez_compressed(temporary, **arrays)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_descriptor = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        self.sequence = sequence
        return destination
