import json

import numpy as np
import torch
import yaml

from baselines.d2pcca import D2PCCABaseline
from benchmark import BenchmarkResult
from data.agneuro_adapter import BenchmarkDataset, predefined_trial_datasets
from evaluate import evaluate_result


def test_d2pcca_test_evaluation_reuses_persisted_protocol(monkeypatch, tmp_path):
    trial_ids = tuple(f"synthetic-trial-{index:04d}" for index in range(60))
    assignments = {
        trial_id: "train" if index < 36 else "validation" if index < 48 else "test"
        for index, trial_id in enumerate(trial_ids)
    }
    x = np.repeat(np.arange(60, dtype=float)[:, None, None], 4, axis=1)
    x = np.repeat(x, 32, axis=2)
    y = np.repeat((2.0 * np.arange(60, dtype=float))[:, None, None], 4, axis=1)
    y = np.repeat(y, 16, axis=2)
    mask = np.ones((60, 4), dtype=bool)
    metadata = {
        "dataset_id": "synthetic",
        "trial_ids": list(trial_ids),
        "joint_dataset_metadata": {"trial_splits": assignments},
    }
    raw = BenchmarkDataset(x, y, mask, trial_ids, metadata)
    splits = predefined_trial_datasets(raw)
    assert splits is not None

    data_dir = tmp_path / "data"
    result_dir = tmp_path / "result"
    data_dir.mkdir()
    result_dir.mkdir()
    np.savez(data_dir / "arrays.npz", x=x, y=y, mask=mask)
    (data_dir / "metadata.json").write_text(json.dumps(metadata))
    (result_dir / "trial_splits.json").write_text(
        json.dumps(splits.to_jsonable(), indent=2, sort_keys=True)
    )
    preprocessing_text = json.dumps(
        splits.train.preprocessing_stats.to_jsonable(), indent=2, sort_keys=True
    )
    (result_dir / "preprocessing.json").write_text(preprocessing_text)
    config = {
        "seed": 0,
        "latent": {"shared_dim": 3, "x_private_dim": 3, "y_private_dim": 3},
        "data": {"standardize_x": True, "standardize_y": True},
        "d2pcca": {"epochs": 1},
    }
    (result_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (result_dir / "result.json").write_text(json.dumps({
        "model_name": "d2pcca",
        "training_runtime_seconds": 1.5,
        "inference_runtime_seconds": 0.0,
        "parameter_count": 1,
    }))

    evaluated_means = []

    class FakeSVI:
        def step(self, values, factor):
            raise AssertionError("test evaluation must not perform optimizer updates")

        def evaluate_loss(self, values, factor):
            evaluated_means.append(float(values[0].mean()))
            return float(values[0].square().sum() + values[1].square().sum())

    def fake_build(self, dataset):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.device = torch.device("cpu")
        self.svi = FakeSVI()

    def fake_transform(self, dataset):
        batch, time_steps = dataset.shape
        zeros = np.zeros((batch, time_steps, 3), dtype=float)
        return BenchmarkResult(
            "d2pcca", zeros, zeros.copy(), zeros.copy(), dataset.x.copy(), dataset.y.copy(),
            dataset.mask.copy(), dataset.sequence_ids, self.history, 1.5, 1,
            inference_runtime_seconds=0.0,
        )

    monkeypatch.setattr(D2PCCABaseline, "_build", fake_build)
    monkeypatch.setattr(D2PCCABaseline, "transform", fake_transform)

    creator = D2PCCABaseline(config, output_dir=result_dir)
    creator._build(splits.train)
    protocol = creator._training_protocol(splits.train, splits.validation)
    creator._save_best_checkpoint(1, 12.0, 1.0, 0.01, protocol)
    creator._persist_best_selection(protocol)

    first = evaluate_result(result_dir, data_dir)
    first_latents = {
        key: value.copy()
        for key, value in np.load(result_dir / "test_latents.npz", allow_pickle=False).items()
    }
    second = evaluate_result(result_dir, data_dir)
    second_latents = {
        key: value.copy()
        for key, value in np.load(result_dir / "test_latents.npz", allow_pickle=False).items()
    }

    assert first == second
    assert first["trial_ids"] == list(trial_ids[48:])
    assert [row["trial_id"] for row in first["objective"]["per_trial"]] == list(trial_ids[48:])
    assert len(evaluated_means) == 24
    assert min(evaluated_means) > 1.0
    assert (result_dir / "preprocessing.json").read_text() == preprocessing_text
    assert first_latents["shared_latent"].shape == (12, 4, 3)
    assert first_latents["x_private_latent"].shape == (12, 4, 3)
    assert first_latents["y_private_latent"].shape == (12, 4, 3)
    assert first_latents["trial_ids"].tolist() == list(trial_ids[48:])
    assert np.array_equal(first_latents["mask"], mask[48:])
    assert np.array_equal(first_latents["time_indices"], np.tile(np.arange(4), (12, 1)))
    for key in first_latents:
        assert np.array_equal(first_latents[key], second_latents[key])
    assert (result_dir / "test_metrics.json").is_file()
    assert (result_dir / "test_reconstructions.npz").is_file()
