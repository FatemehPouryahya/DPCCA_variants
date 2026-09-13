"""Linear DPCCA model and common-interface DPCTW baseline."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from benchmark import BenchmarkResult
from data.agneuro_adapter import BenchmarkDataset
from .em import DPCCAParameters, fit_em
from .rts import RTSResult, kalman_filter_smoother


class DPCCAModel:
    def __init__(self, shared_dim: int = 8, x_private_dim: int = 4, y_private_dim: int = 4, seed: int = 0):
        self.shared_dim = int(shared_dim)
        self.x_private_dim = int(x_private_dim)
        self.y_private_dim = int(y_private_dim)
        self.seed = int(seed)
        self.parameters: DPCCAParameters | None = None
        self.history: list[float] = []

    def fit(
        self, pairs: list[tuple[np.ndarray, np.ndarray]], max_iterations: int = 20,
        tolerance: float = 1e-4, warm_start: bool = False,
    ) -> "DPCCAModel":
        self.parameters, self.history = fit_em(
            pairs, self.shared_dim, self.x_private_dim, self.y_private_dim,
            max_iterations=max_iterations, tolerance=tolerance, seed=self.seed,
            initial_parameters=self.parameters if warm_start else None,
        )
        return self

    def smooth_joint(self, x: np.ndarray, y: np.ndarray) -> RTSResult:
        if self.parameters is None:
            raise RuntimeError("model is not fitted")
        from .em import expectation
        return expectation(self.parameters, x, y)

    def _infer_view(self, values: np.ndarray, view: str) -> RTSResult:
        if self.parameters is None:
            raise RuntimeError("model is not fitted")
        p = self.parameters
        ds, du, dv = p.dimensions
        if view == "x":
            indices = np.r_[0:ds, ds:ds + du]
            emission = np.concatenate([p.x_shared_loading, p.x_private_loading], axis=1)
            noise = p.x_noise
        elif view == "y":
            indices = np.r_[0:ds, ds + du:ds + du + dv]
            emission = np.concatenate([p.y_shared_loading, p.y_private_loading], axis=1)
            noise = p.y_noise
        else:
            raise ValueError("view must be x or y")
        transition = p.transition[np.ix_(indices, indices)]
        process = p.process_covariance[np.ix_(indices, indices)]
        initial_covariance = p.initial_covariance[np.ix_(indices, indices)]
        return kalman_filter_smoother(
            values, transition, process, emission, noise,
            p.initial_mean[indices], initial_covariance,
        )

    def infer_x(self, x: np.ndarray) -> RTSResult:
        return self._infer_view(x, "x")

    def infer_y(self, y: np.ndarray) -> RTSResult:
        return self._infer_view(y, "y")

    def parameter_count(self) -> int:
        if self.parameters is None:
            return 0
        p = self.parameters
        fields = (
            p.shared_transition, p.x_private_transition, p.y_private_transition,
            p.shared_process_covariance, p.x_private_process_covariance, p.y_private_process_covariance,
            p.x_shared_loading, p.x_private_loading, p.y_shared_loading, p.y_private_loading,
            p.x_noise, p.y_noise, p.initial_mean, p.initial_covariance,
        )
        return int(sum(value.size for value in fields))

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.parameters is None:
            raise RuntimeError("model is not fitted")
        return {name: np.asarray(value) for name, value in vars(self.parameters).items()}


class DPCTWBaseline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model: DPCCAModel | None = None
        self.alignment = None
        self.training_runtime_seconds = 0.0

    def fit(self, dataset: BenchmarkDataset) -> "DPCTWBaseline":
        from .alignment import alternating_dpctw
        started = time.perf_counter()
        self.model, self.alignment = alternating_dpctw(dataset, self.config)
        self.training_runtime_seconds = time.perf_counter() - started
        return self

    def transform(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        if self.model is None or self.alignment is None:
            raise RuntimeError("fit must be called before transform")
        batch, time_steps = dataset.shape
        ds, du, dv = self.model.shared_dim, self.model.x_private_dim, self.model.y_private_dim
        shared = np.full((batch, time_steps, ds), np.nan)
        x_private = np.full((batch, time_steps, du), np.nan)
        y_private = np.full((batch, time_steps, dv), np.nan)
        x_recon = np.full_like(dataset.x, np.nan)
        y_recon = np.full_like(dataset.y, np.nan)
        self.alignment.x_shared = np.full_like(shared, np.nan)
        self.alignment.y_shared = np.full_like(shared, np.nan)
        started = time.perf_counter()
        for batch_index in range(batch):
            valid_indices = np.flatnonzero(dataset.mask[batch_index])
            if len(valid_indices) < 2:
                continue
            x_result = self.model.infer_x(dataset.x[batch_index, valid_indices])
            y_result = self.model.infer_y(dataset.y[batch_index, valid_indices])
            self.alignment.x_shared[batch_index, valid_indices] = x_result.smoothed_mean[:, :ds]
            self.alignment.y_shared[batch_index, valid_indices] = y_result.smoothed_mean[:, :ds]
            shared[batch_index, valid_indices] = x_result.smoothed_mean[:, :ds]
            x_private[batch_index, valid_indices] = x_result.smoothed_mean[:, ds:]
            y_private[batch_index, valid_indices] = y_result.smoothed_mean[:, ds:]
            p = self.model.parameters
            x_state = x_result.smoothed_mean
            y_state = y_result.smoothed_mean
            x_recon[batch_index, valid_indices] = x_state @ np.concatenate([p.x_shared_loading, p.x_private_loading], axis=1).T
            y_recon[batch_index, valid_indices] = y_state @ np.concatenate([p.y_shared_loading, p.y_private_loading], axis=1).T
        inference_runtime = time.perf_counter() - started
        deviations = []
        lengths = []
        for path in self.alignment.paths:
            lengths.append(int(len(path)))
            if len(path):
                x_index = path[:, 0].astype(float)
                y_index = path[:, 1].astype(float)
                scale = max(1.0, max(x_index.max(), y_index.max()))
                deviations.append(float(np.mean(np.abs(x_index - y_index)) / scale))
            else:
                deviations.append(float("nan"))
        return BenchmarkResult(
            "dpctw", shared, x_private, y_private, x_recon, y_recon, dataset.mask.copy(),
            dataset.sequence_ids,
            {"alignment": self.alignment.history, "final_em_log_likelihood": self.model.history},
            self.training_runtime_seconds, self.model.parameter_count(),
            model_specific={
                "alignment_algorithm": "exact dynamic programming with optional Sakoe-Chiba band",
                "aligned_lengths": lengths,
                "path_deviation_from_diagonal": deviations,
                "shared_latent_axis": "x native time; y-view shared posterior saved in alignment.npz",
                "iteration_zero": self.alignment.initialization,
            }, inference_runtime_seconds=inference_runtime,
        )

    def save(self, output_dir: str | Path) -> None:
        if self.model is None or self.alignment is None:
            raise RuntimeError("no fitted model")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "model.npz", **self.model.state_dict())
        offsets = [0]
        path_values = []
        for path in self.alignment.paths:
            path_values.append(path)
            offsets.append(offsets[-1] + len(path))
        values: dict[str, np.ndarray] = {
            "path_pairs": np.concatenate(path_values, axis=0) if path_values else np.empty((0, 2), dtype=int),
            "path_sequence_offsets": np.asarray(offsets, dtype=int),
        }
        if self.alignment.x_shared is not None:
            values["x_shared_latent"] = self.alignment.x_shared
        if self.alignment.y_shared is not None:
            values["y_shared_latent"] = self.alignment.y_shared
        np.savez_compressed(output / "alignment.npz", **values)
