#!/usr/bin/env python3
"""Model-independent evaluation for saved benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmark import load_config
from data.agneuro_adapter import (
    PreprocessingStats,
    load_agneuro,
    predefined_trial_datasets,
)


def _first_array(files: list[Path], keys: tuple[str, ...]) -> np.ndarray | None:
    for file in files:
        if not file.exists():
            continue
        with np.load(file, allow_pickle=False) as values:
            for key in keys:
                if key in values:
                    return np.asarray(values[key])
    return None


def masked_metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool)[..., None] & np.isfinite(target) & np.isfinite(prediction)
    squared = np.where(valid, (target - prediction) ** 2, np.nan)
    feature_mse = np.nanmean(squared, axis=(0, 1))
    return {
        "rmse": float(np.sqrt(np.nanmean(squared))),
        "per_feature_normalized_mse": feature_mse.tolist(),
        "mean_per_feature_normalized_mse": float(np.nanmean(feature_mse)),
        "evaluated_elements": int(valid.sum()),
    }


def evaluate_result(result_dir: str | Path, data_path: str | Path) -> dict[str, Any]:
    result = Path(result_dir)
    config_path = result / "resolved_config.yaml"
    config = load_config(config_path) if config_path.exists() else {"data": {}}
    raw = load_agneuro(data_path)
    summary_path = result / "result.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if summary.get("model_name", result.name) == "d2pcca" and (
        result / "trial_splits.json"
    ).exists():
        from baselines.d2pcca import D2PCCABaseline

        split_manifest = json.loads((result / "trial_splits.json").read_text())
        statistics = PreprocessingStats.from_jsonable(
            json.loads((result / "preprocessing.json").read_text())
        )
        data_config = config.get("data", {})
        splits = predefined_trial_datasets(
            raw,
            bool(data_config.get("standardize_x", True)),
            bool(data_config.get("standardize_y", True)),
            statistics=statistics,
            persisted_assignments=split_manifest["assignments"],
        )
        if splits is None:
            raise RuntimeError("persisted D²PCCA split assignments could not be loaded")
        expected_splits = split_manifest["splits"]
        actual_splits = splits.to_jsonable()["splits"]
        if split_manifest["source_trial_ids"] != list(splits.source_trial_ids):
            raise RuntimeError("persisted D²PCCA trial ordering does not match the dataset")
        if expected_splits != actual_splits:
            raise RuntimeError("persisted D²PCCA split membership does not match the dataset")
        baseline = D2PCCABaseline(config, output_dir=result)
        baseline._build(splits.train)
        protocol = baseline._training_protocol(splits.train, splits.validation)
        baseline.load_best_checkpoint(protocol)
        _, metrics = baseline.evaluate_test(splits.test, result)
        metrics["training_runtime_seconds"] = summary.get("training_runtime_seconds")
        metrics["inference_runtime_seconds"] = summary.get("inference_runtime_seconds")
        metrics["parameter_count"] = summary.get("parameter_count")
        (result / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        return metrics

    if summary.get("model_name", result.name) == "infodpcca" and (
        result / "trial_splits.json"
    ).exists():
        from baselines.infodpcca import InfoDPCCABaseline

        split_manifest = json.loads((result / "trial_splits.json").read_text())
        statistics = PreprocessingStats.from_jsonable(
            json.loads((result / "preprocessing.json").read_text())
        )
        data_config = config.get("data", {})
        splits = predefined_trial_datasets(
            raw,
            bool(data_config.get("standardize_x", True)),
            bool(data_config.get("standardize_y", True)),
            statistics=statistics,
            persisted_assignments=split_manifest["assignments"],
        )
        if splits is None:
            raise RuntimeError("persisted InfoDPCCA split assignments could not be loaded")
        if split_manifest["source_trial_ids"] != list(splits.source_trial_ids):
            raise RuntimeError("persisted InfoDPCCA trial ordering does not match the dataset")
        if split_manifest["splits"] != splits.to_jsonable()["splits"]:
            raise RuntimeError("persisted InfoDPCCA split membership does not match the dataset")
        baseline = InfoDPCCABaseline(config, output_dir=result)
        baseline.load_selected_for_evaluation(splits.train, splits.validation)
        _, metrics = baseline.evaluate_test(splits.test, result)
        metrics["training_runtime_seconds"] = summary.get("training_runtime_seconds")
        metrics["inference_runtime_seconds"] = summary.get("inference_runtime_seconds")
        metrics["parameter_count"] = summary.get("parameter_count")
        (result / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        return metrics

    if summary.get("model_name", result.name) == "dpctw" and (
        result / "trial_splits.json"
    ).exists():
        from baselines.dpctw.model import DPCTWBaseline

        split_manifest = json.loads((result / "trial_splits.json").read_text())
        statistics = PreprocessingStats.from_jsonable(
            json.loads((result / "preprocessing.json").read_text())
        )
        data_config = config.get("data", {})
        splits = predefined_trial_datasets(
            raw,
            bool(data_config.get("standardize_x", True)),
            bool(data_config.get("standardize_y", True)),
            statistics=statistics,
            persisted_assignments=split_manifest["assignments"],
        )
        if splits is None:
            raise RuntimeError("persisted DPCTW split assignments could not be loaded")
        if split_manifest["source_trial_ids"] != list(splits.source_trial_ids):
            raise RuntimeError("persisted DPCTW trial ordering does not match the dataset")
        if split_manifest["splits"] != splits.to_jsonable()["splits"]:
            raise RuntimeError("persisted DPCTW split membership does not match the dataset")
        baseline = DPCTWBaseline(config, output_dir=result)
        baseline.load_selected_for_evaluation(splits.train, splits.validation)
        _, metrics = baseline.evaluate_test(splits.test, result)
        metrics["training_runtime_seconds"] = summary.get("training_runtime_seconds")
        metrics["inference_runtime_seconds"] = summary.get("inference_runtime_seconds")
        metrics["parameter_count"] = summary.get("parameter_count")
        (result / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        return metrics

    data_config = config.get("data", {})
    dataset = raw.standardized(
        bool(data_config.get("standardize_x", True)), bool(data_config.get("standardize_y", True))
    )
    files = [result / "reconstructions.npz", result / "results.npz", result / "latents.npz"]
    x_reconstruction = _first_array(files, ("x_reconstruction", "recon_x", "x_hat"))
    y_reconstruction = _first_array(files, ("y_reconstruction", "recon_y", "y_hat"))
    metrics: dict[str, Any] = {
        "model_name": summary.get("model_name", result.name),
        "training_runtime_seconds": summary.get("training_runtime_seconds"),
        "inference_runtime_seconds": summary.get("inference_runtime_seconds"),
        "parameter_count": summary.get("parameter_count"),
    }
    if x_reconstruction is not None:
        metrics["x"] = masked_metrics(dataset.x, x_reconstruction, dataset.mask)
    if y_reconstruction is not None:
        metrics["y"] = masked_metrics(dataset.y, y_reconstruction, dataset.mask)
    if not any(key in metrics for key in ("x", "y")):
        metrics["reconstruction_note"] = "No recognized reconstruction arrays were exported by this method."
    if (result / "alignment.npz").exists():
        with np.load(result / "alignment.npz", allow_pickle=False) as alignment:
            offsets = alignment["path_sequence_offsets"]
            paths = alignment["path_pairs"]
            metrics["alignment"] = {
                "aligned_lengths": np.diff(offsets).astype(int).tolist(),
                "path_deviation_from_diagonal": [
                    float(np.mean(np.abs(path[:, 0] - path[:, 1])) / max(1, np.max(path)))
                    for path in (paths[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1))
                ],
            }
    (result / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--summary", default="comparison.csv")
    arguments = parser.parse_args()
    rows = [evaluate_result(path, arguments.data) for path in arguments.results]
    flat_rows = []
    for row in rows:
        flat_rows.append({
            "model_name": row["model_name"],
            "rmse_x": row.get("x", {}).get("rmse"), "rmse_y": row.get("y", {}).get("rmse"),
            "training_runtime_seconds": row.get("training_runtime_seconds"),
            "inference_runtime_seconds": row.get("inference_runtime_seconds"),
            "parameter_count": row.get("parameter_count"),
        })
    with Path(arguments.summary).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
