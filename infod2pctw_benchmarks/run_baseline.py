#!/usr/bin/env python3
"""Train one baseline and save the common artifact contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark import build_provenance, load_config
from data.agneuro_adapter import load_agneuro, predefined_trial_datasets


ROOT = Path(__file__).resolve().parent


def run(model_name: str, data_path: str | Path, config_path: str | Path, output_dir: str | Path | None = None) -> Path:
    config = load_config(config_path)
    raw = load_agneuro(data_path)
    data_config = config.get("data", {})
    standardize_x = bool(data_config.get("standardize_x", True))
    standardize_y = bool(data_config.get("standardize_y", True))
    predefined = (
        predefined_trial_datasets(raw, standardize_x, standardize_y)
        if model_name in ("d2pcca", "infodpcca", "dpctw") else None
    )
    dataset = (
        predefined.train if predefined is not None
        else raw.standardized(standardize_x, standardize_y)
    )
    output = Path(output_dir) if output_dir else ROOT / "results" / dataset.name / model_name
    if model_name == "d2pcca":
        from baselines.d2pcca import D2PCCABaseline, AUTHOR_REPOSITORY
        baseline = D2PCCABaseline(config, output_dir=output)
        third_party = ROOT / "third_party" / "D2PCCA"
        source = f"{AUTHOR_REPOSITORY}:D2PCCA.py"
    elif model_name == "infodpcca":
        from baselines.infodpcca import InfoDPCCABaseline, AUTHOR_REPOSITORY
        baseline = InfoDPCCABaseline(config, output_dir=output)
        third_party = ROOT / "third_party" / "InfoDPCCA"
        source = f"{AUTHOR_REPOSITORY}:InfoDPCCA.py"
    elif model_name == "dpctw":
        from baselines.dpctw.model import DPCTWBaseline
        baseline = DPCTWBaseline(config, output_dir=output)
        third_party = None
        source = "local faithful linear-Gaussian DPCCA/DPCTW implementation"
    else:
        raise ValueError(f"unknown model: {model_name}")

    if predefined is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "trial_splits.json").write_text(
            json.dumps(predefined.to_jsonable(), indent=2, sort_keys=True) + "\n"
        )
        (output / "preprocessing.json").write_text(
            json.dumps(dataset.preprocessing_stats.to_jsonable(), indent=2, sort_keys=True) + "\n"
        )
        baseline.fit(dataset, validation_dataset=predefined.validation)
        result, _ = baseline.evaluate_test(predefined.test, output)
    else:
        baseline.fit(dataset)
        result = baseline.transform(dataset)
    provenance = build_provenance(
        dataset_metadata=dataset.metadata, benchmark_root=ROOT, third_party_root=third_party,
        source_implementation=source, seed=int(config.get("seed", 0)),
        runtime_seconds=result.runtime_seconds + (result.inference_runtime_seconds or 0.0),
    )
    result.save(output, config, provenance)
    baseline.save(output)
    if dataset.preprocessing_stats is not None:
        (output / "preprocessing.json").write_text(
            json.dumps(dataset.preprocessing_stats.to_jsonable(), indent=2, sort_keys=True)
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("d2pcca", "infodpcca", "dpctw"))
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    print(run(arguments.model, arguments.data, arguments.config, arguments.output))


if __name__ == "__main__":
    main()
