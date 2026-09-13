"""Common result, configuration, and provenance contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def git_commit(path: str | Path) -> str | None:
    try:
        requested = Path(path).resolve()
        root = Path(subprocess.check_output(
            ["git", "-C", str(requested), "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()).resolve()
        if root != requested:
            return None
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: tuple[str, ...] = ("numpy", "scipy", "torch", "pyro-ppl", "PyYAML")) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_provenance(
    *, dataset_metadata: dict[str, Any], benchmark_root: str | Path, third_party_root: str | Path | None,
    source_implementation: str, seed: int, runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "dataset_identity": dataset_metadata.get("dataset_id", dataset_metadata.get("source_arrays", "unknown")),
        "benchmark_git_commit": git_commit(benchmark_root),
        "third_party_git_commit": git_commit(third_party_root) if third_party_root else None,
        "source_implementation": source_implementation,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "seed": int(seed),
        "runtime_seconds": float(runtime_seconds),
    }


@dataclass
class BenchmarkResult:
    model_name: str
    shared_latent: np.ndarray | None
    x_private_latent: np.ndarray | None
    y_private_latent: np.ndarray | None
    x_reconstruction: np.ndarray | None
    y_reconstruction: np.ndarray | None
    mask: np.ndarray
    sequence_ids: tuple[str, ...]
    training_history: dict[str, Any]
    runtime_seconds: float
    parameter_count: int
    model_specific: dict[str, Any] = field(default_factory=dict)
    inference_runtime_seconds: float | None = None

    def save(self, output_dir: str | Path, resolved_config: dict[str, Any], provenance: dict[str, Any]) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "resolved_config.yaml").open("w") as stream:
            yaml.safe_dump(resolved_config, stream, sort_keys=False)
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))

        latent_values: dict[str, np.ndarray] = {"mask": np.asarray(self.mask, dtype=bool)}
        for name in ("shared_latent", "x_private_latent", "y_private_latent"):
            value = getattr(self, name)
            if value is not None:
                latent_values[name] = np.asarray(value)
        np.savez_compressed(output / "latents.npz", **latent_values)

        reconstructions: dict[str, np.ndarray] = {}
        if self.x_reconstruction is not None:
            reconstructions["x_reconstruction"] = np.asarray(self.x_reconstruction)
        if self.y_reconstruction is not None:
            reconstructions["y_reconstruction"] = np.asarray(self.y_reconstruction)
        if reconstructions:
            np.savez_compressed(output / "reconstructions.npz", **reconstructions)

        (output / "history.json").write_text(json.dumps(_jsonable(self.training_history), indent=2, sort_keys=True))
        summary = {
            "model_name": self.model_name,
            "sequence_ids": list(self.sequence_ids),
            "training_runtime_seconds": float(self.runtime_seconds),
            "inference_runtime_seconds": self.inference_runtime_seconds,
            "parameter_count": int(self.parameter_count),
            "model_specific": _jsonable(self.model_specific),
        }
        (output / "result.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        metrics_path = output / "metrics.json"
        if not metrics_path.exists():
            metrics_path.write_text("{}\n")
        return output


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_result(output_dir: str | Path) -> BenchmarkResult:
    output = Path(output_dir)
    summary = json.loads((output / "result.json").read_text())
    history = json.loads((output / "history.json").read_text())
    with np.load(output / "latents.npz", allow_pickle=False) as values:
        latents = {key: values[key] for key in values.files}
    recon: dict[str, np.ndarray] = {}
    if (output / "reconstructions.npz").exists():
        with np.load(output / "reconstructions.npz", allow_pickle=False) as values:
            recon = {key: values[key] for key in values.files}
    return BenchmarkResult(
        model_name=summary["model_name"],
        shared_latent=latents.get("shared_latent"),
        x_private_latent=latents.get("x_private_latent"),
        y_private_latent=latents.get("y_private_latent"),
        x_reconstruction=recon.get("x_reconstruction"),
        y_reconstruction=recon.get("y_reconstruction"),
        mask=latents["mask"],
        sequence_ids=tuple(summary["sequence_ids"]),
        training_history=history,
        runtime_seconds=float(summary["training_runtime_seconds"]),
        parameter_count=int(summary["parameter_count"]),
        model_specific=summary.get("model_specific", {}),
        inference_runtime_seconds=summary.get("inference_runtime_seconds"),
    )
