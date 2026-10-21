"""Thin two-stage adapter around the official InfoDPCCA implementation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from benchmark import BenchmarkResult
from data.agneuro_adapter import BenchmarkDataset


AUTHOR_REPOSITORY = "https://github.com/marcusstang/InfoDPCCA.git"
AUTHOR_COMMIT = "02b2e119c9d7e230b2b2cf62534adacaf219698e"


def _load_author_module():
    source = Path(__file__).resolve().parents[1] / "third_party" / "InfoDPCCA" / "InfoDPCCA.py"
    spec = importlib.util.spec_from_file_location("vendor_infodpcca", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import author implementation at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InfoDPCCABaseline:
    def __init__(self, config: dict[str, Any], output_dir: str | Path | None = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.stage1 = None
        self.stage2 = None
        self.stage1_optimizer = None
        self.stage2_optimizer = None
        self.stage1_svi = None
        self.stage2_svi = None
        self.history: dict[str, list[float]] = {
            "stage_I": [],
            "stage_I_validation_raw": [],
            "stage_I_validation_per_valid_modeled_timestamp": [],
            "stage_II": [],
            "stage_II_validation_raw": [],
            "stage_II_validation_per_valid_modeled_timestamp": [],
        }
        self.training_runtime_seconds = 0.0
        self.validation_dataset: BenchmarkDataset | None = None
        self.best_validation: dict[str, dict[str, Any] | None] = {
            "stage_I": None,
            "stage_II": None,
        }
        self._best_model_state: dict[str, dict[str, Any]] = {}
        self._author = None

    @staticmethod
    def _optimizer(settings: dict[str, Any], clip_default: float):
        from pyro.optim import ClippedAdam
        return ClippedAdam({
            "lr": float(settings.get("lr", 3e-4)),
            "betas": tuple(settings.get("betas", [0.96, 0.999])),
            "clip_norm": float(settings.get("clip_norm", clip_default)),
            "lrd": float(settings.get("lrd", 0.99996)),
            "weight_decay": float(settings.get("weight_decay", 2.0)),
        })

    def _build(self, dataset: BenchmarkDataset) -> tuple[Any, Any]:
        try:
            import pyro
            from pyro.infer import SVI, Trace_ELBO
            import torch
        except ImportError as error:
            raise RuntimeError("InfoDPCCA requires torch, pyro-ppl, matplotlib and scikit-learn") from error
        seed = int(self.config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        self._author = _load_author_module()
        method = self.config.get("infodpcca", {})
        latent = self.config["latent"]
        device = torch.device(method.get("device", "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("InfoDPCCA is configured for CUDA, but no GPU is visible to PyTorch")
        self.stage1 = self._author.InfoDPCCA(
            x_dim=dataset.x.shape[-1], y_dim=dataset.y.shape[-1], z0_dim=int(latent["shared_dim"]),
            emission_x_dim=int(method.get("emission_x_dim", 20)),
            emission_y_dim=int(method.get("emission_y_dim", 20)),
            rnn_x_dim=int(method.get("rnn_x_dim", 30)), rnn_y_dim=int(method.get("rnn_y_dim", 30)),
            num_layers=int(method.get("rnn_layers", 1)),
            rnn_dropout_rate=float(method.get("rnn_dropout_rate", 0.0)),
            beta=float(method.get("beta", 1e-5)), alpha=float(method.get("alpha", 1.0)),
        ).to(device)
        self.stage1.device = device
        self.stage1_optimizer = self._optimizer(method.get("stage_I_optimizer", {}), 10.0)
        self.stage1_svi = SVI(
            self.stage1.model, self.stage1.guide, self.stage1_optimizer, Trace_ELBO()
        )
        return self._author, self.stage1_svi

    def _build_stage2(self, dataset: BenchmarkDataset) -> Any:
        from pyro.infer import SVI, Trace_ELBO

        if self.stage1 is None or self._author is None:
            raise RuntimeError("InfoDPCCA Stage I must be built before Stage II")
        method = self.config.get("infodpcca", {})
        latent = self.config["latent"]
        device = self.stage1.device
        self.stage2 = self._author.InfoDPCCA_2(
            x_dim=dataset.x.shape[-1], y_dim=dataset.y.shape[-1], z0_dim=int(latent["shared_dim"]),
            z1_dim=int(latent["x_private_dim"]), z2_dim=int(latent["y_private_dim"]),
            emission_x_dim=int(method.get("emission_x_dim", 20)),
            emission_y_dim=int(method.get("emission_y_dim", 20)),
            rnn_x_dim=int(method.get("rnn_x_dim", 30)), rnn_y_dim=int(method.get("rnn_y_dim", 30)),
            num_layers=int(method.get("rnn_layers", 1)),
            rnn_dropout_rate=float(method.get("rnn_dropout_rate", 0.0)),
            rnn_x=self.stage1.rnn_x, rnn_y=self.stage1.rnn_y,
            h_x_0=self.stage1.h_x_0, h_y_0=self.stage1.h_y_0,
            combiner_12_0=self.stage1.combiner_12_0,
            res_con=bool(method.get("residual_connection", True)),
            old_emitter_x=self.stage1.emitter_x, old_emitter_y=self.stage1.emitter_y,
            rnn4vi=bool(method.get("rnn4vi", False)),
            rnn_vi_dim=int(method.get("rnn_vi_dim", 60)),
        ).to(device)
        self.stage2.device = device
        self.stage2_optimizer = self._optimizer(method.get("stage_II_optimizer", {}), 5.0)
        self.stage2_svi = SVI(
            self.stage2.model, self.stage2.guide, self.stage2_optimizer, Trace_ELBO()
        )
        return self.stage2_svi

    def _checkpoint_directory(self, stage: str) -> Path | None:
        if self.output_dir is None:
            return None
        return self.output_dir / "checkpoints" / "infodpcca" / stage

    @staticmethod
    def _training_protocol(
        dataset: BenchmarkDataset,
        validation_dataset: BenchmarkDataset | None = None,
    ) -> dict[str, Any]:
        def partition(value: BenchmarkDataset) -> dict[str, Any]:
            return {
                "sequence_ids": list(value.sequence_ids),
                "x_shape": list(value.x.shape),
                "y_shape": list(value.y.shape),
                "mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(value.mask, dtype=np.uint8).tobytes()
                ).hexdigest(),
                "preprocessing": (
                    value.preprocessing_stats.to_jsonable()
                    if value.preprocessing_stats is not None else None
                ),
            }

        protocol = {
            "schema_version": (
                "infodpcca-training-protocol-v2"
                if validation_dataset is not None else "infodpcca-training-protocol-v1"
            ),
            "training": partition(dataset),
        }
        if validation_dataset is not None:
            protocol["validation"] = partition(validation_dataset)
        return protocol

    @staticmethod
    def _protocol_fingerprint(protocol: dict[str, Any]) -> str:
        serialized = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _history_keys(stage: str) -> tuple[str, str, str]:
        return (
            stage,
            f"{stage}_validation_raw",
            f"{stage}_validation_per_valid_modeled_timestamp",
        )

    def _save_checkpoint(
        self, stage: str, model: Any, optimizer: Any, completed_epoch: int,
        stage_completed: bool, runtime_seconds: float,
        training_protocol: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_dir = self._checkpoint_directory(stage)
        if checkpoint_dir is None:
            return
        if training_protocol is None:
            raise RuntimeError("InfoDPCCA checkpoint requires a training-data protocol")
        import torch

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{completed_epoch:06d}"
        model_path = checkpoint_dir / f"model_epoch_{suffix}.pt"
        optimizer_path = checkpoint_dir / f"optimizer_epoch_{suffix}.pt"
        model_temporary = checkpoint_dir / f".model_epoch_{suffix}.tmp"
        optimizer_temporary = checkpoint_dir / f".optimizer_epoch_{suffix}.tmp"
        state_path = checkpoint_dir / "state.json"
        state_temporary = checkpoint_dir / ".state.json.tmp"

        torch.save(model.state_dict(), model_temporary)
        optimizer.save(str(optimizer_temporary))
        model_temporary.replace(model_path)
        optimizer_temporary.replace(optimizer_path)
        state = {
            "stage": stage,
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "stage_completed": bool(stage_completed),
            "model_file": model_path.name,
            "optimizer_file": optimizer_path.name,
            "history": self.history,
            "training_runtime_seconds": float(runtime_seconds),
            "training_protocol": training_protocol,
            "training_protocol_fingerprint": self._protocol_fingerprint(training_protocol),
            "best_validation": self.best_validation[stage],
        }
        state_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        state_temporary.replace(state_path)

    def _load_checkpoint(
        self, stage: str, model: Any, optimizer: Any,
        training_protocol: dict[str, Any],
    ) -> int:
        checkpoint_dir = self._checkpoint_directory(stage)
        if checkpoint_dir is None:
            return 0
        state_path = checkpoint_dir / "state.json"
        if not state_path.exists():
            return 0
        import torch

        try:
            state = json.loads(state_path.read_text())
            completed_epoch = int(state["completed_epoch"])
            saved_protocol = state["training_protocol"]
            saved_fingerprint = state["training_protocol_fingerprint"]
            expected_fingerprint = self._protocol_fingerprint(training_protocol)
            train_key, validation_raw_key, validation_normalized_key = self._history_keys(stage)
            saved_history = state["history"]
            validation_required = training_protocol.get("validation") is not None
            best_validation = state.get("best_validation")
            valid = (
                state.get("stage") == stage
                and completed_epoch >= 1
                and int(state["next_epoch"]) == completed_epoch + 1
                and isinstance(saved_protocol, dict)
                and self._protocol_fingerprint(saved_protocol) == saved_fingerprint
                and saved_fingerprint == expected_fingerprint
                and len(saved_history.get(train_key, [])) == completed_epoch
                and (
                    not validation_required
                    or (
                        len(saved_history.get(validation_raw_key, [])) == completed_epoch
                        and len(saved_history.get(validation_normalized_key, [])) == completed_epoch
                        and isinstance(best_validation, dict)
                        and best_validation.get("training_protocol_fingerprint") == expected_fingerprint
                    )
                )
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            valid = False
        if not valid:
            raise RuntimeError(
                "InfoDPCCA checkpoint uses a different or legacy training-data protocol; "
                "use a new result directory or remove the incompatible checkpoints"
            )
        model_path = checkpoint_dir / state["model_file"]
        optimizer_path = checkpoint_dir / state["optimizer_file"]
        if not model_path.is_file() or not optimizer_path.is_file():
            raise RuntimeError(f"incomplete InfoDPCCA checkpoint recorded by {state_path}")
        model.load_state_dict(torch.load(model_path, map_location=model.device))
        optimizer.load(str(optimizer_path))
        self.history = {
            key: list(state["history"].get(key, [])) for key in self.history
        }
        self.best_validation[stage] = (
            dict(state["best_validation"])
            if isinstance(state.get("best_validation"), dict) else None
        )
        self.training_runtime_seconds = max(
            self.training_runtime_seconds,
            float(state.get("training_runtime_seconds", 0.0)),
        )
        return completed_epoch

    def _save_best_checkpoint(
        self, stage: str, model: Any, completed_epoch: int,
        raw_loss: float, normalized_loss: float,
        training_protocol: dict[str, Any],
    ) -> None:
        import torch

        suffix = f"{completed_epoch:06d}"
        record = {
            "stage": stage,
            "epoch": int(completed_epoch),
            "raw_validation_loss": float(raw_loss),
            "validation_loss_per_valid_modeled_timestamp": float(normalized_loss),
            "model_file": f"best/model_epoch_{suffix}.pt",
            "training_protocol_fingerprint": self._protocol_fingerprint(training_protocol),
        }
        state_copy = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        self._best_model_state[stage] = state_copy
        checkpoint_dir = self._checkpoint_directory(stage)
        if checkpoint_dir is not None:
            best_dir = checkpoint_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model_path = best_dir / f"model_epoch_{suffix}.pt"
            state_path = best_dir / f"state_epoch_{suffix}.json"
            model_temporary = best_dir / f".model_epoch_{suffix}.tmp"
            state_temporary = best_dir / f".state_epoch_{suffix}.tmp"
            torch.save(state_copy, model_temporary)
            model_temporary.replace(model_path)
            state = dict(record)
            state["training_protocol"] = training_protocol
            state_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            state_temporary.replace(state_path)
        self.best_validation[stage] = record

    def load_best_checkpoint(
        self, stage: str, model: Any, training_protocol: dict[str, Any],
    ) -> dict[str, Any]:
        expected_fingerprint = self._protocol_fingerprint(training_protocol)
        checkpoint_dir = self._checkpoint_directory(stage)
        selected = self.best_validation[stage]
        if selected is not None:
            if selected.get("training_protocol_fingerprint") != expected_fingerprint:
                raise RuntimeError(f"selected InfoDPCCA {stage} checkpoint has an incompatible protocol")
            if checkpoint_dir is not None:
                model_path = checkpoint_dir / selected["model_file"]
                if model_path.is_file():
                    import torch
                    model.load_state_dict(torch.load(model_path, map_location=model.device))
                    return dict(selected)
            if stage in self._best_model_state:
                model.load_state_dict(self._best_model_state[stage])
                return dict(selected)
            raise RuntimeError(f"selected InfoDPCCA {stage} best-model file is unavailable")

        records: list[dict[str, Any]] = []
        if checkpoint_dir is not None:
            selection_path = checkpoint_dir / "best" / "selection.json"
            state_paths = (
                [selection_path] if selection_path.is_file()
                else list((checkpoint_dir / "best").glob("state_epoch_*.json"))
            )
            for state_path in state_paths:
                try:
                    state = json.loads(state_path.read_text())
                    loss = float(state["validation_loss_per_valid_modeled_timestamp"])
                    model_path = checkpoint_dir / state["model_file"]
                    if (
                        state.get("stage") != stage
                        or int(state["epoch"]) < 1
                        or not np.isfinite(loss)
                        or state.get("training_protocol_fingerprint") != expected_fingerprint
                        or self._protocol_fingerprint(state["training_protocol"]) != expected_fingerprint
                        or not model_path.is_file()
                    ):
                        continue
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                records.append(state)
        if not records:
            raise RuntimeError(f"no valid InfoDPCCA {stage} best-validation checkpoint is available")
        selected = min(
            records,
            key=lambda value: (
                float(value["validation_loss_per_valid_modeled_timestamp"]),
                int(value["epoch"]),
            ),
        )
        import torch
        model.load_state_dict(torch.load(
            checkpoint_dir / selected["model_file"], map_location=model.device
        ))
        self.best_validation[stage] = {
            key: value for key, value in selected.items() if key != "training_protocol"
        }
        return dict(self.best_validation[stage])

    def _persist_best_selection(self, stage: str, training_protocol: dict[str, Any]) -> None:
        checkpoint_dir = self._checkpoint_directory(stage)
        selected = self.best_validation[stage]
        if checkpoint_dir is None or selected is None:
            return
        best_dir = checkpoint_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        state = dict(selected)
        state["training_protocol"] = training_protocol
        temporary = best_dir / ".selection.json.tmp"
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        temporary.replace(best_dir / "selection.json")

    def _objective_loss(
        self, stage: str, svi: Any, dataset: BenchmarkDataset,
        evaluation_seed: int,
    ) -> dict[str, Any]:
        import torch

        segments = list(dataset.valid_segments(minimum_length=2))
        if not segments:
            raise ValueError(f"InfoDPCCA {stage} objective evaluation requires a valid segment of length >= 2")
        raw_by_trial = {trial_id: 0.0 for trial_id in dataset.sequence_ids}
        count_by_trial = {trial_id: 0 for trial_id in dataset.sequence_ids}
        model = self.stage1 if stage == "stage_I" else self.stage2
        device = model.device
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(evaluation_seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(evaluation_seed))
            for batch_index, _, x, y in segments:
                xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                loss = float(svi.evaluate_loss(xt, yt))
                trial_id = dataset.sequence_ids[batch_index]
                raw_by_trial[trial_id] += loss
                count_by_trial[trial_id] += int(x.shape[0] - 1)
        raw_loss = float(sum(raw_by_trial.values()))
        modeled_timestamps = int(sum(count_by_trial.values()))
        normalized_loss = raw_loss / max(1, modeled_timestamps)
        if not np.isfinite(raw_loss) or not np.isfinite(normalized_loss):
            raise FloatingPointError(f"InfoDPCCA {stage} evaluated objective became non-finite")
        return {
            "raw_loss": raw_loss,
            "loss_per_valid_modeled_timestamp": normalized_loss,
            "valid_modeled_timestamps": modeled_timestamps,
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

    def _train_stage(
        self, stage: str, model: Any, optimizer: Any, svi: Any,
        segments: list[tuple[int, slice, np.ndarray, np.ndarray]],
        epochs: int, start_epoch: int, checkpoint_every: int,
        training_protocol: dict[str, Any],
    ) -> None:
        import torch

        train_key, validation_raw_key, validation_normalized_key = self._history_keys(stage)
        stage_started = time.perf_counter()
        accumulated_runtime = self.training_runtime_seconds
        for epoch in range(start_epoch, epochs):
            loss = 0.0
            count = 0
            for _, _, x, y in segments:
                xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                loss += float(svi.step(xt, yt))
                count += x.shape[0]
            value = loss / max(1, count)
            if not np.isfinite(value):
                raise FloatingPointError(f"InfoDPCCA {stage} objective became non-finite")
            self.history[train_key].append(value)
            completed_epoch = epoch + 1
            if self.validation_dataset is not None:
                validation = self._objective_loss(
                    stage, svi, self.validation_dataset,
                    int(self.config.get("seed", 0))
                    + (100_000 if stage == "stage_I" else 200_000)
                    + completed_epoch,
                )
                raw_loss = float(validation["raw_loss"])
                normalized_loss = float(validation["loss_per_valid_modeled_timestamp"])
                self.history[validation_raw_key].append(raw_loss)
                self.history[validation_normalized_key].append(normalized_loss)
                selected = self.best_validation[stage]
                if (
                    selected is None
                    or normalized_loss
                    < float(selected["validation_loss_per_valid_modeled_timestamp"])
                ):
                    self._save_best_checkpoint(
                        stage, model, completed_epoch, raw_loss, normalized_loss,
                        training_protocol,
                    )
            should_save = (
                self.validation_dataset is not None
                or completed_epoch % checkpoint_every == 0
                or completed_epoch == epochs
            )
            if should_save:
                self._save_checkpoint(
                    stage, model, optimizer, completed_epoch,
                    completed_epoch == epochs,
                    accumulated_runtime + time.perf_counter() - stage_started,
                    training_protocol,
                )
        self.training_runtime_seconds = accumulated_runtime + time.perf_counter() - stage_started

    def fit(
        self, dataset: BenchmarkDataset,
        validation_dataset: BenchmarkDataset | None = None,
    ) -> "InfoDPCCABaseline":
        if validation_dataset is not None and (
            validation_dataset.x.shape[-1] != dataset.x.shape[-1]
            or validation_dataset.y.shape[-1] != dataset.y.shape[-1]
        ):
            raise ValueError("InfoDPCCA validation observation dimensions must match training")
        self.validation_dataset = validation_dataset
        segments = list(dataset.valid_segments(minimum_length=2))
        if not segments:
            raise ValueError("InfoDPCCA requires at least one contiguous valid segment of length >= 2")
        self._build(dataset)
        method = self.config.get("infodpcca", {})
        checkpoint_every = int(method.get("checkpoint_every_epochs", 1))
        if checkpoint_every < 1:
            raise ValueError("InfoDPCCA checkpoint_every_epochs must be positive")
        training_protocol = self._training_protocol(dataset, validation_dataset)

        stage_I_epochs = int(method.get("stage_I_epochs", 1))
        stage_I_start = self._load_checkpoint(
            "stage_I", self.stage1, self.stage1_optimizer, training_protocol
        )
        self._train_stage(
            "stage_I", self.stage1, self.stage1_optimizer, self.stage1_svi,
            segments, stage_I_epochs, stage_I_start, checkpoint_every, training_protocol,
        )
        if validation_dataset is not None:
            self.load_best_checkpoint("stage_I", self.stage1, training_protocol)
            self._persist_best_selection("stage_I", training_protocol)

        self._build_stage2(dataset)
        stage_II_epochs = int(method.get("stage_II_epochs", 1))
        stage_II_start = self._load_checkpoint(
            "stage_II", self.stage2, self.stage2_optimizer, training_protocol
        )
        self._train_stage(
            "stage_II", self.stage2, self.stage2_optimizer, self.stage2_svi,
            segments, stage_II_epochs, stage_II_start, checkpoint_every, training_protocol,
        )
        if validation_dataset is not None:
            self.load_best_checkpoint("stage_II", self.stage2, training_protocol)
            self._persist_best_selection("stage_II", training_protocol)
        return self

    def load_selected_for_evaluation(
        self, dataset: BenchmarkDataset, validation_dataset: BenchmarkDataset,
    ) -> dict[str, Any]:
        """Build both stages and load their selected checkpoints without training."""
        self.validation_dataset = validation_dataset
        self._build(dataset)
        protocol = self._training_protocol(dataset, validation_dataset)
        self.load_best_checkpoint("stage_I", self.stage1, protocol)
        self._build_stage2(dataset)
        self.load_best_checkpoint("stage_II", self.stage2, protocol)
        return protocol

    def validation_loss(
        self, stage: str, dataset: BenchmarkDataset | None = None,
    ) -> float:
        validation = dataset if dataset is not None else self.validation_dataset
        if validation is None:
            raise ValueError("no InfoDPCCA validation dataset is available")
        if stage == "stage_I":
            svi = self.stage1_svi
        elif stage == "stage_II":
            svi = self.stage2_svi
        else:
            raise ValueError(f"unknown InfoDPCCA stage: {stage}")
        if svi is None:
            raise RuntimeError("the requested InfoDPCCA stage has not been built")
        result = self._objective_loss(
            stage, svi, validation, int(self.config.get("seed", 0)) + 250_000
        )
        return float(result["loss_per_valid_modeled_timestamp"])

    def _infer_segment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        import torch
        model = self.stage2
        xt = torch.as_tensor(x, dtype=torch.float32, device=model.device).unsqueeze(0)
        yt = torch.as_tensor(y, dtype=torch.float32, device=model.device).unsqueeze(0)
        length = x.shape[0]
        with torch.no_grad():
            if model.rnn4vi:
                initial = model.h_vi_0.expand(1, 1, model.rnn_vi.hidden_size).contiguous()
                rnn_output, _ = model.rnn_vi(torch.cat([xt, yt], dim=2), initial)
            else:
                hx = model.h_x_0.expand(1, 1, model.rnn_x.hidden_size).contiguous()
                hy = model.h_y_0.expand(1, 1, model.rnn_y.hidden_size).contiguous()
                rx, _ = model.rnn_x(xt, hx)
                ry, _ = model.rnn_y(yt, hy)
                rnn_output = torch.cat([rx, ry], dim=2)
            latent = torch.zeros((1, length, model.z0_dim + model.z1_dim + model.z2_dim), device=model.device)
            xr = torch.zeros_like(xt)
            yr = torch.zeros_like(yt)
            xr[:, 0] = xt[:, 0]
            yr[:, 0] = yt[:, 0]
            for index in range(1, length):
                location, _ = model.combiner_vi(rnn_output[:, index, :])
                latent[:, index] = location
                z0 = location[:, :model.z0_dim]
                z1 = location[:, model.z0_dim:model.z0_dim + model.z1_dim]
                z2 = location[:, model.z0_dim + model.z1_dim:]
                xr[:, index], _ = model.emitter_x(torch.cat([z0, z1], dim=1))
                yr[:, index], _ = model.emitter_y(torch.cat([z0, z2], dim=1))
        values = latent.cpu().numpy()[0]
        return (
            values[:, :model.z0_dim], values[:, model.z0_dim:model.z0_dim + model.z1_dim],
            values[:, model.z0_dim + model.z1_dim:], xr.cpu().numpy()[0], yr.cpu().numpy()[0],
        )

    @staticmethod
    def _modeled_mask(dataset: BenchmarkDataset) -> np.ndarray:
        modeled = np.zeros(dataset.shape, dtype=bool)
        for batch_index, span, _, _ in dataset.valid_segments(minimum_length=2):
            modeled[batch_index, span.start + 1:span.stop] = True
        return modeled

    def transform(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        if self.stage2 is None:
            raise RuntimeError("fit or load_selected_for_evaluation must be called before transform")
        batch, time_steps = dataset.shape
        latent = self.config["latent"]
        arrays = [
            np.full((batch, time_steps, int(latent["shared_dim"])), np.nan),
            np.full((batch, time_steps, int(latent["x_private_dim"])), np.nan),
            np.full((batch, time_steps, int(latent["y_private_dim"])), np.nan),
            np.full_like(dataset.x, np.nan), np.full_like(dataset.y, np.nan),
        ]
        started = time.perf_counter()
        for batch_index, span, x, y in dataset.valid_segments(minimum_length=2):
            inferred = self._infer_segment(x, y)
            for destination, value in zip(arrays, inferred):
                destination[batch_index, span] = value
        inference_runtime = time.perf_counter() - started
        seen: set[int] = set()
        parameter_count = 0
        for parameter in self.stage2.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                parameter_count += parameter.numel()
        return BenchmarkResult(
            "infodpcca", *arrays, dataset.mask.copy(), dataset.sequence_ids, self.history,
            self.training_runtime_seconds, parameter_count,
            model_specific={
                "author_repository": AUTHOR_REPOSITORY, "author_commit": AUTHOR_COMMIT,
                "two_stage_training": True,
                "first_timestamp_behavior": (
                    "the first timestamp of each contiguous valid segment is copied from the "
                    "observation and its latent is zero; it is excluded from prediction metrics"
                ),
                "mask_handling": "independent contiguous valid spans; invalid timestamps retained as NaN outputs",
            }, inference_runtime_seconds=inference_runtime,
        )

    @staticmethod
    def _reconstruction_metrics(
        target: np.ndarray, prediction: np.ndarray, modeled_mask: np.ndarray,
        sequence_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        valid = modeled_mask[..., None] & np.isfinite(target) & np.isfinite(prediction)
        squared = np.where(valid, (target - prediction) ** 2, np.nan)
        aggregate_mse = float(np.nanmean(squared))
        per_trial = []
        for index, trial_id in enumerate(sequence_ids):
            trial_squared = squared[index]
            per_trial.append({
                "trial_id": trial_id,
                "mse": float(np.nanmean(trial_squared)),
                "rmse": float(np.sqrt(np.nanmean(trial_squared))),
                "evaluated_elements": int(np.isfinite(trial_squared).sum()),
            })
        return {
            "mse": aggregate_mse,
            "rmse": float(np.sqrt(aggregate_mse)),
            "evaluated_elements": int(np.isfinite(squared).sum()),
            "per_trial": per_trial,
            "space": "training-normalized observations",
            "evaluated_time_range": (
                "timestamps 2..T within each contiguous valid segment; each segment's first "
                "timestamp is conditioning input copied into reconstruction and is not scored"
            ),
        }

    def evaluate_test(
        self, dataset: BenchmarkDataset, output_dir: str | Path,
    ) -> tuple[BenchmarkResult, dict[str, Any]]:
        """Evaluate and persist held-out Stage-II outputs without parameter updates."""
        selected = self.best_validation["stage_II"]
        if selected is None:
            raise RuntimeError("InfoDPCCA test evaluation requires a selected Stage-II checkpoint")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        best_epoch = int(selected["epoch"])
        objective = self._objective_loss(
            "stage_II", self.stage2_svi, dataset,
            int(self.config.get("seed", 0)) + 300_000 + best_epoch,
        )
        result = self.transform(dataset)
        modeled_mask = self._modeled_mask(dataset)
        metrics = {
            "model_name": "infodpcca",
            "partition": "test",
            "trial_ids": list(dataset.sequence_ids),
            "selected_best_epoch": best_epoch,
            "selected_checkpoint": selected["model_file"],
            "selected_validation_loss_per_valid_modeled_timestamp": float(
                selected["validation_loss_per_valid_modeled_timestamp"]
            ),
            "training_runtime_seconds": float(result.runtime_seconds),
            "inference_runtime_seconds": result.inference_runtime_seconds,
            "parameter_count": int(result.parameter_count),
            "objective": objective,
            "evaluated_time_range": (
                "timestamps 2..T within each contiguous valid segment; each segment's first "
                "timestamp is excluded because InfoDPCCA copies that observation"
            ),
            "x": self._reconstruction_metrics(
                dataset.x, result.x_reconstruction, modeled_mask, dataset.sequence_ids
            ),
            "y": self._reconstruction_metrics(
                dataset.y, result.y_reconstruction, modeled_mask, dataset.sequence_ids
            ),
        }
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
            mask=result.mask,
            modeled_mask=modeled_mask,
            trial_ids=np.asarray(dataset.sequence_ids),
            time_indices=time_indices,
        )
        np.savez_compressed(
            output / "test_reconstructions.npz",
            x_reconstruction=result.x_reconstruction,
            y_reconstruction=result.y_reconstruction,
            mask=result.mask,
            modeled_mask=modeled_mask,
            trial_ids=np.asarray(dataset.sequence_ids),
            time_indices=time_indices,
        )
        return result, metrics

    def save(self, output_dir: str | Path) -> None:
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("no fitted model")
        import torch
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.stage1.state_dict(), output / "stage_I_model.pt")
        torch.save(self.stage2.state_dict(), output / "stage_II_model.pt")
        self.stage1_optimizer.save(str(output / "stage_I_optimizer.pt"))
        self.stage2_optimizer.save(str(output / "stage_II_optimizer.pt"))
