"""Thin two-stage adapter around the official InfoDPCCA implementation."""

from __future__ import annotations

import importlib.util
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
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage1 = None
        self.stage2 = None
        self.stage1_optimizer = None
        self.stage2_optimizer = None
        self.history: dict[str, list[float]] = {"stage_I": [], "stage_II": []}
        self.training_runtime_seconds = 0.0

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
        author = _load_author_module()
        method = self.config.get("infodpcca", {})
        latent = self.config["latent"]
        device = torch.device(method.get("device", "cpu"))
        self.stage1 = author.InfoDPCCA(
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
        svi1 = SVI(self.stage1.model, self.stage1.guide, self.stage1_optimizer, Trace_ELBO())
        return author, svi1

    def fit(self, dataset: BenchmarkDataset) -> "InfoDPCCABaseline":
        import torch
        import pyro
        from pyro.infer import SVI, Trace_ELBO

        segments = list(dataset.valid_segments(minimum_length=2))
        if not segments:
            raise ValueError("InfoDPCCA requires at least one contiguous valid segment of length >= 2")
        author, svi1 = self._build(dataset)
        method = self.config.get("infodpcca", {})
        started = time.perf_counter()
        for _ in range(int(method.get("stage_I_epochs", 1))):
            loss = 0.0
            count = 0
            for _, _, x, y in segments:
                xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                loss += float(svi1.step(xt, yt))
                count += x.shape[0]
            value = loss / max(1, count)
            if not np.isfinite(value):
                raise FloatingPointError("InfoDPCCA stage-I objective became non-finite")
            self.history["stage_I"].append(value)

        latent = self.config["latent"]
        device = self.stage1.device
        self.stage2 = author.InfoDPCCA_2(
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
        svi2 = SVI(self.stage2.model, self.stage2.guide, self.stage2_optimizer, Trace_ELBO())
        for _ in range(int(method.get("stage_II_epochs", 1))):
            loss = 0.0
            count = 0
            for _, _, x, y in segments:
                xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
                yt = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0)
                loss += float(svi2.step(xt, yt))
                count += x.shape[0]
            value = loss / max(1, count)
            if not np.isfinite(value):
                raise FloatingPointError("InfoDPCCA stage-II objective became non-finite")
            self.history["stage_II"].append(value)
        self.training_runtime_seconds = time.perf_counter() - started
        return self

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

    def transform(self, dataset: BenchmarkDataset) -> BenchmarkResult:
        if self.stage2 is None:
            raise RuntimeError("fit must be called before transform")
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
                "first_timestamp_behavior": "author algorithm copies observations and leaves latent at zero",
                "mask_handling": "independent contiguous valid spans; invalid timestamps retained as NaN outputs",
            }, inference_runtime_seconds=inference_runtime,
        )

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
