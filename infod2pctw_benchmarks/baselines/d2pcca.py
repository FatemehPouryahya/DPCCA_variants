"""Thin adapter around the canonical author D²PCCA implementation."""

from __future__ import annotations

import importlib.util
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
        self.history: dict[str, list[float]] = {"elbo_loss_per_valid_timestamp": []}
        self.training_runtime_seconds = 0.0

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

    def _save_checkpoint(self, completed_epoch: int, runtime_seconds: float) -> None:
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is None:
            return
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
        }
        state_temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        state_temporary.replace(state_path)

    def _load_latest_checkpoint(self) -> int:
        checkpoint_dir = self._checkpoint_directory()
        if checkpoint_dir is None or not checkpoint_dir.exists():
            return 0

        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        for state_path in checkpoint_dir.glob("state_epoch_*.json"):
            try:
                state = json.loads(state_path.read_text())
                completed_epoch = int(state["completed_epoch"])
                suffix = state_path.stem.removeprefix("state_epoch_")
                history = state["history"]["elbo_loss_per_valid_timestamp"]
                runtime_seconds = float(state["training_runtime_seconds"])
                if (
                    completed_epoch < 1
                    or completed_epoch != int(suffix)
                    or int(state["next_epoch"]) != completed_epoch + 1
                    or not isinstance(history, list)
                    or len(history) != completed_epoch
                    or runtime_seconds < 0.0
                    or state["model_file"] != f"model_epoch_{suffix}.pt"
                    or state["optimizer_file"] != f"optimizer_epoch_{suffix}.pt"
                ):
                    continue
                model_path = checkpoint_dir / state["model_file"]
                optimizer_path = checkpoint_dir / state["optimizer_file"]
                if not model_path.is_file() or not optimizer_path.is_file():
                    continue
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            candidates.append((completed_epoch, state_path, state))

        if not candidates:
            return 0

        import torch

        for completed_epoch, _, state in sorted(candidates, reverse=True):
            model_path = checkpoint_dir / state["model_file"]
            optimizer_path = checkpoint_dir / state["optimizer_file"]
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
                    )
                }
            self.training_runtime_seconds = float(
                state.get("training_runtime_seconds", 0.0)
            )
            return completed_epoch
        return 0

    def fit(self, dataset: BenchmarkDataset) -> "D2PCCABaseline":
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
        start_epoch = self._load_latest_checkpoint()
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
            if completed_epoch % checkpoint_every == 0 or completed_epoch == epochs:
                self._save_checkpoint(
                    completed_epoch,
                    accumulated_runtime + time.perf_counter() - started,
                )
        self.training_runtime_seconds = (
            accumulated_runtime + time.perf_counter() - started
        )
        return self

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

    def save(self, output_dir: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("no fitted model")
        import torch
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output / "model.pt")
        if self.optimizer is not None:
            self.optimizer.save(str(output / "optimizer.pt"))
