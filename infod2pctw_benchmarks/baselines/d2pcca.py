"""Thin adapter around the canonical author D²PCCA implementation."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from benchmark import BenchmarkResult
from data.agneuro_adapter import BenchmarkDataset


AUTHOR_REPOSITORY = "https://github.com/marcusstang/D2PCCA.git"
AUTHOR_COMMIT = "f88607ac1c3e2905b91298bc359980a9dac0ce93"


def _load_author_module():
    source = Path(__file__).resolve().parents[1] / "third_party" / "D2PCCA" / "D2PCCA.py"
    spec = importlib.util.spec_from_file_location("vendor_d2pcca", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import author implementation at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class D2PCCABaseline:
    """Common benchmark API without changing the vendored model mathematics."""

    def __init__(self, config: dict[str, Any], output_dir: str | Path | None = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.model = None
        self.optimizer = None
        self.svi = None
        self.history: dict[str, list[float]] = {
            "elbo_loss_per_valid_timestamp": [],
            "validation_elbo_loss_raw": [],
            "validation_elbo_loss_per_valid_timestamp": [],
        }
        self.training_runtime_seconds = 0.0
        self.validation_dataset: BenchmarkDataset | None = None
        self.best_validation: dict[str, Any] | None = None
        self._best_model_state: dict[str, Any] | None = None

    def _build(self, dataset: BenchmarkDataset) -> None:
        try:
            import pyro
            from pyro.infer import SVI, Trace_ELBO
            from pyro.optim import ClippedAdam
            import torch
        except ImportError as error:
            raise RuntimeError("D²PCCA requires torch and pyro-ppl; install the benchmark requirements") from error

        seed = int(self.config.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()

        latent = self.config["latent"]
        method = self.config.get("d2pcca", {})
        author = _load_author_module()
        self.model = author.D2PCCA(
            x_dims=[dataset.x.shape[-1], dataset.y.shape[-1]],
            emission_dims=list(method.get("emission_dims", [20, 20])),
            z_dims=[latent["shared_dim"], latent["x_private_dim"], latent["y_private_dim"]],
            transition_dims=list(method.get("transition_dims", [5, 10, 10])),
            rnn_dim=int(method.get("rnn_dim", 50)),
            rnn_layers=int(method.get("rnn_layers", 1)),
            rnn_dropout_rate=float(method.get("rnn_dropout_rate", 0.0)),
            num_iafs=int(method.get("num_iafs", 0)),
            iaf_dim=int(method.get("iaf_dim", 50)),
        )
        requested_device = torch.device(method.get("device", self.model.device))
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "D²PCCA is configured for CUDA, but no GPU is visible to PyTorch"
            )
        self.model.device = requested_device
        self.model.to(requested_device)
        optimizer_config = method.get("optimizer", {})
        self.optimizer = ClippedAdam({
            "lr": float(optimizer_config.get("lr", 3e-4)),
            "betas": tuple(optimizer_config.get("betas", [0.96, 0.999])),
            "clip_norm": float(optimizer_config.get("clip_norm", 10.0)),
            "lrd": float(optimizer_config.get("lrd", 0.99996)),
            "weight_decay": float(optimizer_config.get("weight_decay", 2.0)),
        })
        self.svi = SVI(self.model.model, self.model.guide, self.optimizer, Trace_ELBO())

    def _checkpoint_directory(self) -> Path | None:
        if self.output_dir is None:
            return None
        return self.output_dir / "checkpoints" / "d2pcca"

    @staticmethod
    def _training_protocol(
        dataset: BenchmarkDataset,
        validation_dataset: BenchmarkDataset | None = None,
    ) -> dict[str, Any]:
        preprocessing = (
            dataset.preprocessing_stats.to_jsonable()
            if dataset.preprocessing_stats is not None else None
        )
        protocol = {
            "schema_version": (
                "d2pcca-training-protocol-v2"
                if validation_dataset is not None else "d2pcca-training-protocol-v1"
            ),
            "sequence_ids": list(dataset.sequence_ids),
            "x_shape": list(dataset.x.shape),
            "y_shape": list(dataset.y.shape),
            "mask_sha256": hashlib.sha256(
                np.ascontiguousarray(dataset.mask, dtype=np.uint8).tobytes()
            ).hexdigest(),
            "preprocessing": preprocessing,
        }
        if validation_dataset is not None:
            protocol["validation"] = {
                "sequence_ids": list(validation_dataset.sequence_ids),
                "x_shape": list(validation_dataset.x.shape),
                "y_shape": list(validation_dataset.y.shape),
                "mask_sha256": hashlib.sha256(
                    np.ascontiguousarray(validation_dataset.mask, dtype=np.uint8).tobytes()
                ).hexdigest(),
                "preprocessing": (
                    validation_dataset.preprocessing_stats.to_jsonable()
                    if validation_dataset.preprocessing_stats is not None else None
                ),
            }
        return protocol

    @staticmethod
    def _protocol_fingerprint(protocol: dict[str, Any]) -> str:
        serialized = json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _write_training_checkpoint(
        self, checkpoint_dir: Path,
        completed_epoch: int, runtime_seconds: float,
        training_protocol: dict[str, Any],
    ) -> None:
        import torch

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{completed_epoch:06d}"
        model_path = checkpoint_dir / f"model_epoch_{suffix}.pt"
        optimizer_path = checkpoint_dir / f"optimizer_epoch_{suffix}.pt"
        state_path = checkpoint_dir / f"state_epoch_{suffix}.json"
        model_temporary = checkpoint_dir / f".model_epoch_{suffix}.tmp"
        optimizer_temporary = checkpoint_dir / f".optimizer_epoch_{suffix}.tmp"
        state_temporary = checkpoint_dir / f".state_epoch_{suffix}.tmp"

        torch.save(self.model.state_dict(), model_temporary)
        self.optimizer.save(str(optimizer_temporary))
        model_temporary.replace(model_path)
        optimizer_temporary.replace(optimizer_path)
        state = {
            "completed_epoch": int(completed_epoch),
            "next_epoch": int(completed_epoch) + 1,
            "model_file": model_path.name,
            "optimizer_file": optimizer_path.name,
            "history": self.history,
            "training_runtime_seconds": float(runtime_seconds),
            "training_protocol": training_protocol,
            "training_protocol_fingerprint": self._protocol_fingerprint(training_protocol),
            "best_validation": self.best_validation,
        }
        state_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        state_temporary.replace(state_path)

    def _save_checkpoint(
        self, completed_epoch: int, runtime_seconds: float,
        training_protocol: dict[str, Any],
    ) -> None:
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is not None:
            self._write_training_checkpoint(
                checkpoint_dir, completed_epoch, runtime_seconds, training_protocol
            )

    def _save_resume_checkpoint(
        self, completed_epoch: int, runtime_seconds: float,
        training_protocol: dict[str, Any],
    ) -> None:
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is not None:
            self._write_training_checkpoint(
                checkpoint_dir / "resume", completed_epoch, runtime_seconds, training_protocol
            )

    def _load_latest_checkpoint(self, training_protocol: dict[str, Any]) -> int:
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return 0

        expected_fingerprint = self._protocol_fingerprint(training_protocol)
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        incompatible_checkpoints: list[Path] = []
        state_paths = list(checkpoint_dir.glob("state_epoch_*.json"))
        state_paths.extend((checkpoint_dir / "resume").glob("state_epoch_*.json"))
        validation_required = training_protocol.get("validation") is not None
        for state_path in state_paths:
            try:
                state = json.loads(state_path.read_text())
                completed_epoch = int(state["completed_epoch"])
                suffix = state_path.stem.removeprefix("state_epoch_")
                history = state["history"]
                training_history = history["elbo_loss_per_valid_timestamp"]
                validation_raw = history.get("validation_elbo_loss_raw", [])
                validation_normalized = history.get(
                    "validation_elbo_loss_per_valid_timestamp", []
                )
                runtime_seconds = float(state["training_runtime_seconds"])
                if (
                    completed_epoch < 1
                    or completed_epoch != int(suffix)
                    or int(state["next_epoch"]) != completed_epoch + 1
                    or not isinstance(training_history, list)
                    or len(training_history) != completed_epoch
                    or not isinstance(validation_raw, list)
                    or not isinstance(validation_normalized, list)
                    or (
                        validation_required
                        and (
                            len(validation_raw) != completed_epoch
                            or len(validation_normalized) != completed_epoch
                        )
                    )
                    or (
                        not validation_required
                        and (len(validation_raw) != 0 or len(validation_normalized) != 0)
                    )
                    or runtime_seconds < 0.0
                    or state["model_file"] != f"model_epoch_{suffix}.pt"
                    or state["optimizer_file"] != f"optimizer_epoch_{suffix}.pt"
                ):
                    continue
                model_path = state_path.parent / state["model_file"]
                optimizer_path = state_path.parent / state["optimizer_file"]
                if not model_path.is_file() or not optimizer_path.is_file():
                    continue
                saved_protocol = state.get("training_protocol")
                saved_fingerprint = state.get("training_protocol_fingerprint")
                if (
                    not isinstance(saved_protocol, dict)
                    or not isinstance(saved_fingerprint, str)
                    or self._protocol_fingerprint(saved_protocol) != saved_fingerprint
                    or saved_fingerprint != expected_fingerprint
                ):
                    incompatible_checkpoints.append(state_path)
                    continue
                best_validation = state.get("best_validation")
                if validation_required:
                    if not isinstance(best_validation, dict):
                        continue
                    best_model_path = checkpoint_dir / str(
                        best_validation.get("model_file", "")
                    )
                    if (
                        int(best_validation.get("epoch", 0)) < 1
                        or int(best_validation.get("epoch", 0)) > completed_epoch
                        or not np.isfinite(float(best_validation.get(
                            "validation_loss_per_valid_timestamp", np.nan
                        )))
                        or best_validation.get("training_protocol_fingerprint")
                        != expected_fingerprint
                        or not best_model_path.is_file()
                    ):
                        continue
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            candidates.append((completed_epoch, state_path, state))

        if not candidates:
            if incompatible_checkpoints:
                raise RuntimeError(
                    "D²PCCA checkpoints exist for a different or legacy training-data protocol; "
                    "use a new result directory or remove the incompatible checkpoints"
                )
            return 0

        import torch

        for completed_epoch, state_path, state in sorted(
            candidates, key=lambda candidate: candidate[0], reverse=True
        ):
            model_path = state_path.parent / state["model_file"]
            optimizer_path = state_path.parent / state["optimizer_file"]
            try:
                model_state = torch.load(model_path, map_location=self.model.device)
                self.model.load_state_dict(model_state)
                self.optimizer.load(str(optimizer_path))
            except (OSError, RuntimeError, ValueError, EOFError):
                continue
            history = state.get("history")
            if isinstance(history, dict):
                self.history = {
                    "elbo_loss_per_valid_timestamp": list(
                        history.get("elbo_loss_per_valid_timestamp", [])
                    ),
                    "validation_elbo_loss_raw": list(
                        history.get("validation_elbo_loss_raw", [])
                    ),
                    "validation_elbo_loss_per_valid_timestamp": list(
                        history.get("validation_elbo_loss_per_valid_timestamp", [])
                    ),
                }
            best_validation = state.get("best_validation")
            self.best_validation = (
                dict(best_validation) if isinstance(best_validation, dict) else None
            )
            self.training_runtime_seconds = float(
                state.get("training_runtime_seconds", 0.0)
            )
            return completed_epoch
        return 0

    def _save_best_checkpoint(
        self, completed_epoch: int, raw_loss: float, normalized_loss: float,
        annealing_factor: float, training_protocol: dict[str, Any],
    ) -> None:
        import torch

        suffix = f"{completed_epoch:06d}"
        record = {
            "epoch": int(completed_epoch),
            "raw_validation_loss": float(raw_loss),
            "validation_loss_per_valid_timestamp": float(normalized_loss),
            "annealing_factor": float(annealing_factor),
            "model_file": f"best/model_epoch_{suffix}.pt",
            "training_protocol_fingerprint": self._protocol_fingerprint(training_protocol),
        }
        self._best_model_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is not None:
            best_dir = checkpoint_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model_path = best_dir / f"model_epoch_{suffix}.pt"
            state_path = best_dir / f"state_epoch_{suffix}.json"
            model_temporary = best_dir / f".model_epoch_{suffix}.tmp"
            state_temporary = best_dir / f".state_epoch_{suffix}.tmp"
            torch.save(self._best_model_state, model_temporary)
            model_temporary.replace(model_path)
            state = dict(record)
            state["training_protocol"] = training_protocol
            state_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            state_temporary.replace(state_path)
        self.best_validation = record

    def load_best_checkpoint(self, training_protocol: dict[str, Any]) -> dict[str, Any]:
        """Load the lowest finite validation-loss checkpoint for this protocol."""
        checkpoint_dir = self._checkpoint_directory()
        expected_fingerprint = self._protocol_fingerprint(training_protocol)
        if self.best_validation is not None:
            if (
                self.best_validation.get("training_protocol_fingerprint")
                != expected_fingerprint
            ):
                raise RuntimeError("selected D²PCCA checkpoint has an incompatible protocol")
            if checkpoint_dir is not None:
                model_path = checkpoint_dir / self.best_validation["model_file"]
                if model_path.is_file():
                    import torch
                    self.model.load_state_dict(
                        torch.load(model_path, map_location=self.model.device)
                    )
                    return dict(self.best_validation)
            if self._best_model_state is not None:
                self.model.load_state_dict(self._best_model_state)
                return dict(self.best_validation)
            raise RuntimeError("selected D²PCCA best-model file is unavailable")

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
                    loss = float(state["validation_loss_per_valid_timestamp"])
                    epoch = int(state["epoch"])
                    model_path = checkpoint_dir / state["model_file"]
                    if (
                        not np.isfinite(loss)
                        or epoch < 1
                        or state.get("training_protocol_fingerprint") != expected_fingerprint
                        or self._protocol_fingerprint(state["training_protocol"]) != expected_fingerprint
                        or not model_path.is_file()
                    ):
                        continue
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                records.append(state)
        if records:
            selected = min(
                records,
                key=lambda value: (
                    float(value["validation_loss_per_valid_timestamp"]),
                    int(value["epoch"]),
                ),
            )
            import torch
            model_path = checkpoint_dir / selected["model_file"]
            self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
            self.best_validation = {
                key: value for key, value in selected.items() if key != "training_protocol"
            }
            return dict(self.best_validation)
        if self._best_model_state is not None and self.best_validation is not None:
            self.model.load_state_dict(self._best_model_state)
            return dict(self.best_validation)
        raise RuntimeError("no valid D²PCCA best-validation checkpoint is available")

    def _persist_best_selection(self, training_protocol: dict[str, Any]) -> None:
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is None or self.best_validation is None:
            return
        best_dir = checkpoint_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        selection = dict(self.best_validation)
        selection["training_protocol"] = training_protocol
        temporary = best_dir / ".selection.json.tmp"
        temporary.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        temporary.replace(best_dir / "selection.json")

    def _objective_loss(
        self, dataset: BenchmarkDataset, annealing_factor: float,
        evaluation_seed: int,
    ) -> dict[str, Any]:
        import torch

        segments = list(dataset.valid_segments(minimum_length=2))
        if not segments:
            raise ValueError("D²PCCA objective evaluation requires a valid segment of length >= 2")
        raw_by_trial = {trial_id: 0.0 for trial_id in dataset.sequence_ids}
        count_by_trial = {trial_id: 0 for trial_id in dataset.sequence_ids}
        device = getattr(self.model, "device", torch.device("cpu"))
        fork_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(evaluation_seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(evaluation_seed))
            for batch_index, _, x, y in segments:
                xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                loss = float(self.svi.evaluate_loss([xt, yt], annealing_factor))
                trial_id = dataset.sequence_ids[batch_index]
                raw_by_trial[trial_id] += loss
                count_by_trial[trial_id] += int(x.shape[0])
        raw_loss = float(sum(raw_by_trial.values()))
        valid_timestamps = int(sum(count_by_trial.values()))
        normalized_loss = raw_loss / max(1, valid_timestamps)
        if not np.isfinite(raw_loss) or not np.isfinite(normalized_loss):
            raise FloatingPointError("D²PCCA evaluated objective became non-finite")
        return {
            "raw_loss": raw_loss,
            "loss_per_valid_timestamp": normalized_loss,
            "valid_timestamps": valid_timestamps,
            "per_trial": [
                {
                    "trial_id": trial_id,
                    "raw_loss": float(raw_by_trial[trial_id]),
                    "loss_per_valid_timestamp": (
                        float(raw_by_trial[trial_id]) / max(1, count_by_trial[trial_id])
                    ),
                    "valid_timestamps": int(count_by_trial[trial_id]),
                }
                for trial_id in dataset.sequence_ids
            ],
        }

    def fit(
        self, dataset: BenchmarkDataset,
        validation_dataset: BenchmarkDataset | None = None,
    ) -> "D2PCCABaseline":
        if validation_dataset is not None and (
            validation_dataset.x.shape[-1] != dataset.x.shape[-1]
            or validation_dataset.y.shape[-1] != dataset.y.shape[-1]
        ):
            raise ValueError("D²PCCA validation observation dimensions must match training")
        self.validation_dataset = validation_dataset
        if self.model is None:
            self._build(dataset)
        import torch

        segments = list(dataset.valid_segments(minimum_length=2))
        if not segments:
            raise ValueError("D²PCCA requires at least one contiguous valid segment of length >= 2")
        method = self.config.get("d2pcca", {})
        epochs = int(method.get("epochs", 1))
        annealing_epochs = int(method.get("annealing_epochs", 100))
        minimum = float(method.get("minimum_annealing_factor", 0.01))
        checkpoint_every = int(method.get("checkpoint_every_epochs", 5))
        if checkpoint_every < 1:
            raise ValueError("D²PCCA checkpoint_every_epochs must be positive")
        training_protocol = self._training_protocol(dataset, validation_dataset)
        start_epoch = self._load_latest_checkpoint(training_protocol)
        started = time.perf_counter()
        accumulated_runtime = self.training_runtime_seconds
        for epoch in range(start_epoch, epochs):
            factor = minimum + (1.0 - minimum) * min(1.0, (epoch + 1) / max(1, annealing_epochs))
            total_loss = 0.0
            total_time = 0
            for _, _, x, y in segments:
                xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                total_loss += float(self.svi.step([xt, yt], factor))
                total_time += x.shape[0]
            value = total_loss / max(1, total_time)
            if not np.isfinite(value):
                raise FloatingPointError("D²PCCA author objective became non-finite")
            self.history["elbo_loss_per_valid_timestamp"].append(value)
            completed_epoch = epoch + 1
            if validation_dataset is not None:
                validation_factor = 1.0
                validation = self._objective_loss(
                    validation_dataset, validation_factor,
                    int(self.config.get("seed", 0)) + 100_000 + completed_epoch,
                )
                raw_validation = float(validation["raw_loss"])
                normalized_validation = float(validation["loss_per_valid_timestamp"])
                self.history["validation_elbo_loss_raw"].append(raw_validation)
                self.history["validation_elbo_loss_per_valid_timestamp"].append(
                    normalized_validation
                )
                if (
                    self.best_validation is None
                    or normalized_validation
                    < float(self.best_validation["validation_loss_per_valid_timestamp"])
                ):
                    self._save_best_checkpoint(
                        completed_epoch, raw_validation, normalized_validation,
                        validation_factor, training_protocol,
                    )
                self._save_resume_checkpoint(
                    completed_epoch,
                    accumulated_runtime + time.perf_counter() - started,
                    training_protocol,
                )
            if completed_epoch % checkpoint_every == 0 or completed_epoch == epochs:
                self._save_checkpoint(
                    completed_epoch,
                    accumulated_runtime + time.perf_counter() - started,
                    training_protocol,
                )
        self.training_runtime_seconds = (
            accumulated_runtime + time.perf_counter() - started
        )
        if validation_dataset is not None:
            self.load_best_checkpoint(training_protocol)
            self._persist_best_selection(training_protocol)
        return self

    def validation_loss(
        self, dataset: BenchmarkDataset | None = None,
        annealing_factor: float = 1.0,
    ) -> float:
        """Evaluate the existing SVI loss on validation segments without updates."""
        validation = dataset if dataset is not None else self.validation_dataset
        if validation is None:
            raise ValueError("no D²PCCA validation dataset is available")
        if self.svi is None:
            raise RuntimeError("fit must be called before validation_loss")
        result = self._objective_loss(
            validation, annealing_factor,
            int(self.config.get("seed", 0)) + 200_000,
        )
        return float(result["loss_per_valid_timestamp"])

    def _infer_segment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
        import torch

        if int(self.config.get("d2pcca", {}).get("num_iafs", 0)):
            raise NotImplementedError("analytic latent means are unavailable for the author's IAF posterior")
        model = self.model
        values = [torch.as_tensor(x, dtype=torch.float32, device=model.device).unsqueeze(0),
                  torch.as_tensor(y, dtype=torch.float32, device=model.device).unsqueeze(0)]
        with torch.no_grad():
            combined = torch.cat(values, dim=-1)
            hidden = model.h_0.expand(1, 1, model.rnn.hidden_size).contiguous()
            rnn_output, _ = model.rnn(torch.flip(combined, dims=[1]), hidden)
            rnn_output = torch.flip(rnn_output, dims=[1])
            previous = model.z_q_0.expand(1, model.z_q_0.size(0))
            latent_locations = []
            x_reconstructions = []
            y_reconstructions = []
            for index in range(x.shape[0]):
                location, _ = model.combiner(previous, rnn_output[:, index, :])
                latent_locations.append(location)
                shared = location[:, :model.z_shared_dim]
                x_start = model.z_shared_dim
                x_stop = x_start + model.z_individual_dims[0]
                y_stop = x_stop + model.z_individual_dims[1]
                x_location, _ = model.emitters[0](shared, location[:, x_start:x_stop])
                y_location, _ = model.emitters[1](shared, location[:, x_stop:y_stop])
                x_reconstructions.append(x_location)
                y_reconstructions.append(y_location)
                previous = location
            latent = torch.stack(latent_locations, dim=1).cpu().numpy()[0]
            x_recon = torch.stack(x_reconstructions, dim=1).cpu().numpy()[0]
            y_recon = torch.stack(y_reconstructions, dim=1).cpu().numpy()[0]
        ds, du = model.z_shared_dim, model.z_individual_dims[0]
        return latent[:, :ds], latent[:, ds:ds + du], latent[:, ds + du:], x_recon, y_recon

    def transform(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        if self.model is None:
            raise RuntimeError("fit must be called before transform")
        batch, time_steps = dataset.shape
        ds = int(self.config["latent"]["shared_dim"])
        du = int(self.config["latent"]["x_private_dim"])
        dv = int(self.config["latent"]["y_private_dim"])
        shared = np.full((batch, time_steps, ds), np.nan)
        x_private = np.full((batch, time_steps, du), np.nan)
        y_private = np.full((batch, time_steps, dv), np.nan)
        x_recon = np.full_like(dataset.x, np.nan)
        y_recon = np.full_like(dataset.y, np.nan)
        started = time.perf_counter()
        for batch_index, span, x, y in dataset.valid_segments(minimum_length=2):
            s, u, v, xr, yr = self._infer_segment(x, y)
            shared[batch_index, span] = s
            x_private[batch_index, span] = u
            y_private[batch_index, span] = v
            x_recon[batch_index, span] = xr
            y_recon[batch_index, span] = yr
        inference_runtime = time.perf_counter() - started
        parameter_count = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        return BenchmarkResult(
            "d2pcca", shared, x_private, y_private, x_recon, y_recon, dataset.mask.copy(),
            dataset.sequence_ids, self.history, self.training_runtime_seconds, parameter_count,
            model_specific={
                "author_repository": AUTHOR_REPOSITORY,
                "author_commit": AUTHOR_COMMIT,
                "mask_handling": "independent contiguous valid spans; invalid timestamps retained as NaN outputs",
                "latent_extraction": "deterministic posterior locations from the published reversed-RNN combiner",
            },
            inference_runtime_seconds=inference_runtime,
        )

    @staticmethod
    def _reconstruction_metrics(
        target: np.ndarray, prediction: np.ndarray, mask: np.ndarray,
        sequence_ids: tuple[str, ...],
    ) -> dict[str, Any]:
        valid = mask[..., None] & np.isfinite(target) & np.isfinite(prediction)
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
        }

    def evaluate_test(
        self, dataset: BenchmarkDataset, output_dir: str | Path,
    ) -> tuple[BenchmarkResult, dict[str, Any]]:
        """Evaluate and persist held-out D²PCCA outputs without parameter updates."""
        if self.best_validation is None:
            raise RuntimeError("D²PCCA test evaluation requires a selected validation checkpoint")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        best_epoch = int(self.best_validation["epoch"])
        annealing_factor = float(self.best_validation["annealing_factor"])
        objective = self._objective_loss(
            dataset, annealing_factor,
            int(self.config.get("seed", 0)) + 300_000 + best_epoch,
        )
        result = self.transform(dataset)
        metrics = {
            "model_name": "d2pcca",
            "partition": "test",
            "trial_ids": list(dataset.sequence_ids),
            "selected_best_epoch": best_epoch,
            "selected_checkpoint": self.best_validation["model_file"],
            "selected_validation_loss_per_valid_timestamp": float(
                self.best_validation["validation_loss_per_valid_timestamp"]
            ),
            "annealing_factor": annealing_factor,
            "training_runtime_seconds": float(result.runtime_seconds),
            "inference_runtime_seconds": result.inference_runtime_seconds,
            "parameter_count": int(result.parameter_count),
            "objective": objective,
            "x": self._reconstruction_metrics(
                dataset.x, result.x_reconstruction, dataset.mask, dataset.sequence_ids
            ),
            "y": self._reconstruction_metrics(
                dataset.y, result.y_reconstruction, dataset.mask, dataset.sequence_ids
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
            trial_ids=np.asarray(dataset.sequence_ids),
            time_indices=time_indices,
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
        if self.model is None:
            raise RuntimeError("no fitted model")
        import torch
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output / "model.pt")
        if self.optimizer is not None:
            self.optimizer.save(str(output / "optimizer.pt"))
