"""Linear DPCCA model and common-interface DPCTW baseline."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from benchmark import BenchmarkResult
from data.agneuro_adapter import BenchmarkDataset
from .em import DPCCAParameters, fit_em
from .rts import RTSResult, kalman_filter_smoother


class DPCCAModel:
    def __init__(
        self, shared_dim: int = 8, x_private_dim: int = 4,
        y_private_dim: int = 4, seed: int = 0, device: str = "cpu",
    ):
        self.shared_dim = int(shared_dim)
        self.x_private_dim = int(x_private_dim)
        self.y_private_dim = int(y_private_dim)
        self.seed = int(seed)
        self.device = str(device).lower()
        if self.device not in ("cpu", "cuda"):
            raise ValueError("DPCTW device must be 'cpu' or 'cuda'")
        self.backend = None
        if self.device == "cuda":
            from .torch_backend import cuda_backend
            self.backend = cuda_backend()
        self.parameters: DPCCAParameters | None = None
        self.history: list[float] = []

    def fit(
        self, pairs: list[tuple[np.ndarray, np.ndarray]], max_iterations: int = 20,
        tolerance: float = 1e-4, warm_start: bool = False,
        start_iteration: int = 0, initial_history: list[float] | None = None,
        checkpoint_callback: Callable[[int, Any, list[float], bool], None] | None = None,
    ) -> "DPCCAModel":
        if self.device == "cuda":
            from .cuda_em import fit_em as fit_em_cuda
            self.parameters, self.history = fit_em_cuda(
                pairs, self.shared_dim, self.x_private_dim, self.y_private_dim,
                max_iterations=max_iterations, tolerance=tolerance, seed=self.seed,
                initial_parameters=self.parameters if warm_start else None,
                backend=self.backend,
                start_iteration=start_iteration, initial_history=initial_history,
                checkpoint_callback=checkpoint_callback,
            )
        else:
            self.parameters, self.history = fit_em(
                pairs, self.shared_dim, self.x_private_dim, self.y_private_dim,
                max_iterations=max_iterations, tolerance=tolerance, seed=self.seed,
                initial_parameters=self.parameters if warm_start else None,
                start_iteration=start_iteration, initial_history=initial_history,
                checkpoint_callback=checkpoint_callback,
            )
        return self

    def load_state_dict(self, values: dict[str, np.ndarray]) -> None:
        if self.device == "cuda":
            from .cuda_em import DPCCAParameters as CudaDPCCAParameters
            self.parameters = CudaDPCCAParameters(**values).on(self.backend)
        else:
            self.parameters = DPCCAParameters(**{
                name: np.asarray(value) for name, value in values.items()
            })

    def smooth_joint(self, x: np.ndarray, y: np.ndarray) -> RTSResult:
        if self.parameters is None:
            raise RuntimeError("model is not fitted")
        if self.device == "cuda":
            from .cuda_em import expectation
            return expectation(self.parameters, x, y, backend=self.backend)
        from .em import expectation
        return expectation(self.parameters, x, y)

    def _infer_view(self, values: np.ndarray, view: str) -> RTSResult:
        if self.parameters is None:
            raise RuntimeError("model is not fitted")
        p = self.parameters
        if self.device == "cuda":
            from .torch_backend import infer_view_cuda
            return infer_view_cuda(p, values, view, self.backend)
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
        if self.device == "cuda":
            return int(sum(self.backend.size(value) for value in fields))
        return int(sum(value.size for value in fields))

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.parameters is None:
            raise RuntimeError("model is not fitted")
        if self.device == "cuda":
            return {
                name: self.backend.to_numpy(value)
                for name, value in vars(self.parameters).items()
            }
        return {name: np.asarray(value) for name, value in vars(self.parameters).items()}


class DPCTWBaseline:
    def __init__(self, config: dict[str, Any], output_dir: str | Path | None = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.model: DPCCAModel | None = None
        self.alignment = None
        self.training_runtime_seconds = 0.0
        self.validation_dataset: BenchmarkDataset | None = None
        self.validation_history: list[dict[str, Any]] = []
        self.best_validation: dict[str, Any] | None = None
        self.best_model: DPCCAModel | None = None
        self._best_parameter_state: dict[str, np.ndarray] | None = None
        self._training_signature: str | None = None

    def _new_model(self) -> DPCCAModel:
        latent = self.config["latent"]
        return DPCCAModel(
            int(latent["shared_dim"]),
            int(latent["x_private_dim"]),
            int(latent["y_private_dim"]),
            int(self.config.get("seed", 0)),
            device=str(self.config.get("device", "cpu")),
        )

    @staticmethod
    def _parameter_values(parameters: Any) -> dict[str, np.ndarray]:
        values = {}
        for name, value in vars(parameters).items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            values[name] = np.asarray(value).copy()
        return values

    def _model_from_parameters(self, parameters: Any) -> DPCCAModel:
        model = self._new_model()
        values = (
            self._parameter_values(parameters)
            if not isinstance(parameters, dict)
            else {name: np.asarray(value).copy() for name, value in parameters.items()}
        )
        model.load_state_dict(values)
        return model

    @staticmethod
    def _scalar(value: Any) -> float:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
        return float(value)

    def _evaluate_partition(
        self, model: DPCCAModel, dataset: BenchmarkDataset,
    ) -> tuple[dict[str, Any], Any]:
        """Infer per-trial paths/states with fixed global parameters and score them."""
        from .alignment import AlignmentResult, dtw_path, squared_distance_matrix, validate_path

        ds = model.shared_dim
        raw_by_trial: dict[str, float] = {}
        count_by_trial: dict[str, int] = {}
        global_paths: list[np.ndarray] = []
        for batch_index, trial_id in enumerate(dataset.sequence_ids):
            valid_indices = np.flatnonzero(dataset.mask[batch_index])
            if len(valid_indices) < 2:
                raise ValueError(f"sequence {trial_id} has fewer than two valid timestamps")
            x = dataset.x[batch_index, valid_indices]
            y = dataset.y[batch_index, valid_indices]
            x_result = model.infer_x(x)
            y_result = model.infer_y(y)
            if model.device == "cuda":
                x_shared = model.backend.to_numpy(x_result.smoothed_mean)[:, :ds]
                y_shared = model.backend.to_numpy(y_result.smoothed_mean)[:, :ds]
            else:
                x_shared = x_result.smoothed_mean[:, :ds]
                y_shared = y_result.smoothed_mean[:, :ds]
            local_path, _ = dtw_path(
                squared_distance_matrix(x_shared, y_shared, device=model.device),
                self.config.get("alignment", {}).get("band_radius"),
            )
            validate_path(local_path, len(x), len(y))
            aligned_result = model.smooth_joint(
                x[local_path[:, 0]], y[local_path[:, 1]]
            )
            raw_by_trial[trial_id] = -self._scalar(aligned_result.log_likelihood)
            count_by_trial[trial_id] = int(len(local_path))
            global_paths.append(np.column_stack([
                valid_indices[local_path[:, 0]], valid_indices[local_path[:, 1]],
            ]))
        raw_loss = float(sum(raw_by_trial.values()))
        modeled_timestamps = int(sum(count_by_trial.values()))
        normalized_loss = raw_loss / max(1, modeled_timestamps)
        if not np.isfinite(raw_loss) or not np.isfinite(normalized_loss):
            raise FloatingPointError("DPCTW evaluated objective became non-finite")
        objective = {
            "raw_loss": raw_loss,
            "loss_per_valid_modeled_timestamp": normalized_loss,
            "valid_modeled_timestamps": modeled_timestamps,
            "objective_definition": (
                "negative joint linear-Gaussian log likelihood after fixed-parameter "
                "per-trial latent inference and DTW alignment"
            ),
            "per_trial": [
                {
                    "trial_id": trial_id,
                    "raw_loss": float(raw_by_trial[trial_id]),
                    "loss_per_valid_modeled_timestamp": (
                        float(raw_by_trial[trial_id]) / max(1, count_by_trial[trial_id])
                    ),
                    "valid_modeled_timestamps": int(count_by_trial[trial_id]),
                }
                for trial_id in dataset.sequence_ids
            ],
        }
        return objective, AlignmentResult(
            global_paths, [],
            "fixed-global-parameter per-trial latent inference followed by exact DTW",
        )

    def _save_best_checkpoint(
        self, parameters: Any, record: dict[str, Any], signature: str,
    ) -> None:
        values = self._parameter_values(parameters)
        self._best_parameter_state = values
        self.best_model = self._model_from_parameters(values)
        self.best_model.history = list(record.get("em_history", []))
        self.best_validation = dict(record)
        self.best_validation["model_file"] = "in-memory"
        self.best_validation["training_protocol_signature"] = signature
        if self.output_dir is None:
            return
        best_dir = self.output_dir / "checkpoints" / "dpctw" / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{int(record['completed_iteration']):06d}"
        model_path = best_dir / f"model_iteration_{suffix}.npz"
        model_temporary = best_dir / f".model_iteration_{suffix}.tmp.npz"
        np.savez_compressed(model_temporary, **values)
        model_temporary.replace(model_path)
        self.best_validation["model_file"] = str(model_path.relative_to(self.output_dir))
        state = dict(self.best_validation)
        state_path = best_dir / f"state_iteration_{suffix}.json"
        state_temporary = best_dir / f".state_iteration_{suffix}.tmp"
        state_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        state_temporary.replace(state_path)
        selection_temporary = best_dir / ".selection.json.tmp"
        selection_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        selection_temporary.replace(best_dir / "selection.json")

    def _load_best_model(self, signature: str) -> DPCCAModel:
        from .checkpoint import PARAMETER_NAMES

        if self.best_model is not None and self.best_validation is not None:
            if self.best_validation.get("training_protocol_signature") == signature:
                return self.best_model
        if self._best_parameter_state is not None and self.best_validation is not None:
            if self.best_validation.get("training_protocol_signature") != signature:
                raise RuntimeError("selected DPCTW model uses an incompatible protocol")
            self.best_model = self._model_from_parameters(self._best_parameter_state)
            return self.best_model
        if self.output_dir is None:
            raise RuntimeError("no DPCTW best-validation model is available")
        selection_path = self.output_dir / "checkpoints" / "dpctw" / "best" / "selection.json"
        try:
            selection = (
                dict(self.best_validation)
                if self.best_validation is not None
                else json.loads(selection_path.read_text())
            )
            if (
                selection.get("training_protocol_signature") != signature
                or int(selection["completed_iteration"]) < 1
                or not np.isfinite(float(selection["validation_loss_per_valid_modeled_timestamp"]))
            ):
                raise ValueError("invalid selection metadata")
            model_path = self.output_dir / selection["model_file"]
            with np.load(model_path, allow_pickle=False) as values:
                parameters = {name: values[name].copy() for name in values.files}
            if set(parameters) != set(PARAMETER_NAMES):
                raise ValueError("best model parameter set is incomplete")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "DPCTW best-validation checkpoint is missing, incomplete, or incompatible"
            ) from error
        self.best_validation = selection
        self._best_parameter_state = parameters
        self.best_model = self._model_from_parameters(parameters)
        self.best_model.history = list(selection.get("em_history", []))
        return self.best_model

    def fit(
        self, dataset: BenchmarkDataset,
        validation_dataset: BenchmarkDataset | None = None,
    ) -> "DPCTWBaseline":
        from .alignment import alternating_dpctw
        from .checkpoint import DPCTWCheckpointStore, checkpoint_signature

        if validation_dataset is not None and (
            validation_dataset.x.shape[-1] != dataset.x.shape[-1]
            or validation_dataset.y.shape[-1] != dataset.y.shape[-1]
        ):
            raise ValueError("DPCTW validation observation dimensions must match training")
        self.validation_dataset = validation_dataset
        signature = checkpoint_signature(self.config, dataset, validation_dataset)
        self._training_signature = signature

        checkpoint_store = (
            DPCTWCheckpointStore(self.output_dir, signature)
            if self.output_dir is not None else None
        )
        resume = checkpoint_store.load() if checkpoint_store is not None else None
        if resume is not None and validation_dataset is not None:
            state = resume["state"]
            history = state.get("validation_history")
            best = state.get("best_validation")
            if (
                state.get("training_protocol_signature") != signature
                or not isinstance(history, list)
                or (history and not isinstance(best, dict))
            ):
                raise RuntimeError("DPCTW validation checkpoint state is incomplete or incompatible")
            self.validation_history = list(history)
            self.best_validation = dict(best) if isinstance(best, dict) else None
            if self.best_validation is not None:
                self._load_best_model(signature)
        accumulated_runtime = (
            float(resume["state"].get("training_runtime_seconds", 0.0))
            if resume is not None else 0.0
        )
        started = time.perf_counter()

        def save_checkpoint(state, parameters, paths):
            state = dict(state)
            if validation_dataset is not None and state["phase"] in (
                "outer_complete", "final_complete"
            ):
                candidate_model = self._model_from_parameters(parameters)
                validation, _ = self._evaluate_partition(
                    candidate_model, validation_dataset
                )
                completed_iteration = len(self.validation_history) + 1
                row = {
                    "completed_iteration": completed_iteration,
                    "candidate_phase": state["phase"],
                    "completed_outer_iterations": int(
                        state.get("outer_next_iteration", state.get("outer_iteration", 0))
                    ),
                    "raw_validation_loss": float(validation["raw_loss"]),
                    "validation_loss_per_valid_modeled_timestamp": float(
                        validation["loss_per_valid_modeled_timestamp"]
                    ),
                    "valid_modeled_timestamps": int(validation["valid_modeled_timestamps"]),
                    "em_history": list(state.get("em_history", [])),
                }
                self.validation_history.append(row)
                if (
                    self.best_validation is None
                    or row["validation_loss_per_valid_modeled_timestamp"]
                    < float(self.best_validation["validation_loss_per_valid_modeled_timestamp"])
                ):
                    self._save_best_checkpoint(parameters, row, signature)
            if validation_dataset is not None:
                state["validation_history"] = list(self.validation_history)
                state["best_validation"] = self.best_validation
                state["training_protocol_signature"] = signature
            state["training_runtime_seconds"] = (
                accumulated_runtime + time.perf_counter() - started
            )
            if checkpoint_store is not None:
                checkpoint_store.save(
                    state, self._parameter_values(parameters), paths
                )

        self.model, self.alignment = alternating_dpctw(
            dataset, self.config, resume=resume, checkpoint_callback=save_checkpoint,
        )
        self.training_runtime_seconds = (
            accumulated_runtime + time.perf_counter() - started
        )
        if validation_dataset is not None:
            if self.best_validation is None:
                raise RuntimeError("DPCTW produced no valid validation candidate")
            self._load_best_model(signature)
        return self

    def _transform_with_model(
        self, dataset: BenchmarkDataset, model: DPCCAModel, alignment: Any,
    ) -> BenchmarkResult:
        batch, time_steps = dataset.shape
        ds, du, dv = model.shared_dim, model.x_private_dim, model.y_private_dim
        shared = np.full((batch, time_steps, ds), np.nan)
        x_private = np.full((batch, time_steps, du), np.nan)
        y_private = np.full((batch, time_steps, dv), np.nan)
        x_recon = np.full_like(dataset.x, np.nan)
        y_recon = np.full_like(dataset.y, np.nan)
        alignment.x_shared = np.full_like(shared, np.nan)
        alignment.y_shared = np.full_like(shared, np.nan)
        started = time.perf_counter()
        for batch_index in range(batch):
            valid_indices = np.flatnonzero(dataset.mask[batch_index])
            if len(valid_indices) < 2:
                continue
            x_result = model.infer_x(dataset.x[batch_index, valid_indices])
            y_result = model.infer_y(dataset.y[batch_index, valid_indices])
            if model.device == "cuda":
                x_mean = model.backend.to_numpy(x_result.smoothed_mean)
                y_mean = model.backend.to_numpy(y_result.smoothed_mean)
            else:
                x_mean = x_result.smoothed_mean
                y_mean = y_result.smoothed_mean
            alignment.x_shared[batch_index, valid_indices] = x_mean[:, :ds]
            alignment.y_shared[batch_index, valid_indices] = y_mean[:, :ds]
            shared[batch_index, valid_indices] = x_mean[:, :ds]
            x_private[batch_index, valid_indices] = x_mean[:, ds:]
            y_private[batch_index, valid_indices] = y_mean[:, ds:]
            p = model.parameters
            x_state = x_result.smoothed_mean
            y_state = y_result.smoothed_mean
            if model.device == "cuda":
                from .torch_backend import reconstruct_cuda
                x_loading = model.backend.concatenate(
                    [p.x_shared_loading, p.x_private_loading], axis=1
                )
                y_loading = model.backend.concatenate(
                    [p.y_shared_loading, p.y_private_loading], axis=1
                )
                x_recon[batch_index, valid_indices] = reconstruct_cuda(
                    x_state, x_loading, model.backend
                )
                y_recon[batch_index, valid_indices] = reconstruct_cuda(
                    y_state, y_loading, model.backend
                )
            else:
                x_loading = np.concatenate([p.x_shared_loading, p.x_private_loading], axis=1)
                y_loading = np.concatenate([p.y_shared_loading, p.y_private_loading], axis=1)
                x_recon[batch_index, valid_indices] = x_state @ x_loading.T
                y_recon[batch_index, valid_indices] = y_state @ y_loading.T
        inference_runtime = time.perf_counter() - started
        deviations = []
        lengths = []
        for path in alignment.paths:
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
            {
                "alignment": alignment.history,
                "final_em_log_likelihood": model.history,
                "validation": self.validation_history,
            },
            self.training_runtime_seconds, model.parameter_count(),
            model_specific={
                "alignment_algorithm": "exact dynamic programming with optional Sakoe-Chiba band",
                "aligned_lengths": lengths,
                "path_deviation_from_diagonal": deviations,
                "shared_latent_axis": "x native time; y-view shared posterior saved in alignment.npz",
                "iteration_zero": alignment.initialization,
                "device": model.device,
                "cuda_cpu_bound_component": (
                    "exact DTW dynamic-programming recurrence and traceback"
                    if model.device == "cuda" else None
                ),
            }, inference_runtime_seconds=inference_runtime,
        )

    def transform(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        if self.model is None or self.alignment is None:
            raise RuntimeError("fit must be called before transform")
        return self._transform_with_model(dataset, self.model, self.alignment)

    @staticmethod
    def _reconstruction_metrics(
        target: np.ndarray, prediction: np.ndarray, mask: np.ndarray,
        sequence_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        valid = mask[..., None] & np.isfinite(target) & np.isfinite(prediction)
        squared = np.where(valid, (target - prediction) ** 2, np.nan)
        mse = float(np.nanmean(squared))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "evaluated_elements": int(np.isfinite(squared).sum()),
            "space": "training-normalized native-view observations",
            "per_trial": [
                {
                    "trial_id": trial_id,
                    "mse": float(np.nanmean(squared[index])),
                    "rmse": float(np.sqrt(np.nanmean(squared[index]))),
                    "evaluated_elements": int(np.isfinite(squared[index]).sum()),
                }
                for index, trial_id in enumerate(sequence_ids)
            ],
        }

    @staticmethod
    def _path_arrays(paths: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        offsets = [0]
        values = []
        for path in paths:
            values.append(np.asarray(path, dtype=np.int64))
            offsets.append(offsets[-1] + len(path))
        return (
            np.concatenate(values, axis=0) if values else np.empty((0, 2), dtype=np.int64),
            np.asarray(offsets, dtype=np.int64),
        )

    def load_selected_for_evaluation(
        self, dataset: BenchmarkDataset, validation_dataset: BenchmarkDataset,
    ) -> None:
        from .checkpoint import checkpoint_signature

        signature = checkpoint_signature(self.config, dataset, validation_dataset)
        self._training_signature = signature
        self.validation_dataset = validation_dataset
        self._load_best_model(signature)

    def evaluate_test(
        self, dataset: BenchmarkDataset, output_dir: str | Path,
    ) -> tuple[BenchmarkResult, dict[str, Any]]:
        if self._training_signature is None:
            raise RuntimeError("DPCTW test evaluation requires a training protocol")
        model = self._load_best_model(self._training_signature)
        objective, alignment = self._evaluate_partition(model, dataset)
        result = self._transform_with_model(dataset, model, alignment)
        path_pairs, path_offsets = self._path_arrays(alignment.paths)
        alignment_rows = []
        for trial_id, path in zip(dataset.sequence_ids, alignment.paths):
            x_index = path[:, 0].astype(float)
            y_index = path[:, 1].astype(float)
            scale = max(1.0, float(max(x_index.max(), y_index.max())))
            alignment_rows.append({
                "trial_id": trial_id,
                "aligned_length": int(len(path)),
                "path_deviation_from_diagonal": float(
                    np.mean(np.abs(x_index - y_index)) / scale
                ),
            })
        metrics = {
            "model_name": "dpctw",
            "partition": "test",
            "trial_ids": list(dataset.sequence_ids),
            "selected_best_iteration": int(self.best_validation["completed_iteration"]),
            "selected_candidate_phase": self.best_validation["candidate_phase"],
            "selected_checkpoint": self.best_validation["model_file"],
            "selected_validation_loss_per_valid_modeled_timestamp": float(
                self.best_validation["validation_loss_per_valid_modeled_timestamp"]
            ),
            "training_runtime_seconds": float(result.runtime_seconds),
            "inference_runtime_seconds": result.inference_runtime_seconds,
            "parameter_count": int(result.parameter_count),
            "normalization": "statistics fitted on the predefined training trials only",
            "time_alignment": (
                "global model parameters are fixed; each test trial receives independent "
                "view-specific smoothing and exact DTW, and objective timestamps are aligned path pairs"
            ),
            "objective": objective,
            "alignment": alignment_rows,
            "x": self._reconstruction_metrics(
                dataset.x, result.x_reconstruction, dataset.mask, dataset.sequence_ids
            ),
            "y": self._reconstruction_metrics(
                dataset.y, result.y_reconstruction, dataset.mask, dataset.sequence_ids
            ),
        }
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        batch, time_steps = dataset.shape
        time_indices = np.broadcast_to(
            np.arange(time_steps, dtype=np.int64), (batch, time_steps)
        ).copy()
        np.savez_compressed(
            output / "test_latents.npz",
            shared_latent=result.shared_latent,
            x_private_latent=result.x_private_latent,
            y_private_latent=result.y_private_latent,
            x_shared_latent=alignment.x_shared,
            y_shared_latent=alignment.y_shared,
            mask=result.mask,
            trial_ids=np.asarray(dataset.sequence_ids),
            time_indices=time_indices,
            path_pairs=path_pairs,
            path_sequence_offsets=path_offsets,
        )
        np.savez_compressed(
            output / "test_reconstructions.npz",
            x_reconstruction=result.x_reconstruction,
            y_reconstruction=result.y_reconstruction,
            mask=result.mask,
            trial_ids=np.asarray(dataset.sequence_ids),
            time_indices=time_indices,
        )
        return result, metrics

    def save(self, output_dir: str | Path) -> None:
        if self.model is None or self.alignment is None:
            raise RuntimeError("no fitted model")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "model.npz", **self.model.state_dict())
        path_pairs, path_offsets = self._path_arrays(self.alignment.paths)
        values: dict[str, np.ndarray] = {
            "path_pairs": path_pairs,
            "path_sequence_offsets": path_offsets,
        }
        if self.alignment.x_shared is not None:
            values["x_shared_latent"] = self.alignment.x_shared
        if self.alignment.y_shared is not None:
            values["y_shared_latent"] = self.alignment.y_shared
        np.savez_compressed(output / "alignment.npz", **values)
