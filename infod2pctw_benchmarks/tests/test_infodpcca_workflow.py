import json

import numpy as np
import pytest
import torch
import yaml

from baselines.infodpcca import InfoDPCCABaseline
from benchmark import BenchmarkResult
from data.agneuro_adapter import BenchmarkDataset, predefined_trial_datasets
from evaluate import evaluate_result


def info_config(stage_I_epochs=1, stage_II_epochs=1):
    return {
        "seed": 0,
        "latent": {"shared_dim": 2, "x_private_dim": 1, "y_private_dim": 1},
        "data": {"standardize_x": True, "standardize_y": True},
        "infodpcca": {
            "stage_I_epochs": stage_I_epochs,
            "stage_II_epochs": stage_II_epochs,
            "checkpoint_every_epochs": 1,
            "emission_x_dim": 4,
            "emission_y_dim": 4,
            "rnn_x_dim": 5,
            "rnn_y_dim": 5,
            "rnn4vi": False,
            "residual_connection": True,
            "stage_I_optimizer": {"weight_decay": 0.0},
            "stage_II_optimizer": {"weight_decay": 0.0},
        },
    }


def tiny_partitions():
    training = BenchmarkDataset(
        np.zeros((1, 4, 3)), np.zeros((1, 4, 2)), np.ones((1, 4), bool),
        ("train",),
    )
    validation = BenchmarkDataset(
        np.full((1, 4, 3), 10.0), np.full((1, 4, 2), 20.0),
        np.ones((1, 4), bool), ("validation",),
    )
    return training, validation


class FakeOptimizer:
    def __init__(self):
        self.steps = 0

    def save(self, path):
        torch.save({"steps": self.steps}, path)

    def load(self, path):
        self.steps = int(torch.load(path)["steps"])


class FakeSVI:
    def __init__(self, model, optimizer, validation_losses, base):
        self.model = model
        self.optimizer = optimizer
        self.validation_losses = iter(validation_losses)
        self.base = base
        self.step_means = []
        self.validation_means = []

    def step(self, x, y):
        self.step_means.append((float(x.mean()), float(y.mean())))
        self.optimizer.steps += 1
        with torch.no_grad():
            self.model.weight.fill_(self.base + self.optimizer.steps)
        return 1.0

    def evaluate_loss(self, x, y):
        self.validation_means.append((float(x.mean()), float(y.mean())))
        return float(next(self.validation_losses))


def test_two_stage_validation_best_selection_and_initialization(tmp_path):
    training, validation = tiny_partitions()
    baseline = InfoDPCCABaseline(info_config(3, 3), output_dir=tmp_path)
    stage2_initializations = []
    svis = {}

    def fake_build(dataset):
        baseline.stage1 = torch.nn.Linear(1, 1, bias=False)
        baseline.stage1.device = torch.device("cpu")
        baseline.stage1_optimizer = FakeOptimizer()
        baseline.stage1_svi = FakeSVI(
            baseline.stage1, baseline.stage1_optimizer, (30.0, 6.0, 12.0), 0.0
        )
        svis["stage_I"] = baseline.stage1_svi
        baseline._author = object()
        return baseline._author, baseline.stage1_svi

    def fake_build_stage2(dataset):
        stage2_initializations.append(float(baseline.stage1.weight.item()))
        baseline.stage2 = torch.nn.Linear(1, 1, bias=False)
        baseline.stage2.device = torch.device("cpu")
        baseline.stage2_optimizer = FakeOptimizer()
        baseline.stage2_svi = FakeSVI(
            baseline.stage2, baseline.stage2_optimizer, (3.0, 15.0, 9.0), 100.0
        )
        svis["stage_II"] = baseline.stage2_svi
        return baseline.stage2_svi

    baseline._build = fake_build
    baseline._build_stage2 = fake_build_stage2
    baseline.fit(training, validation_dataset=validation)

    assert svis["stage_I"].step_means == [(0.0, 0.0)] * 3
    assert svis["stage_II"].step_means == [(0.0, 0.0)] * 3
    assert svis["stage_I"].validation_means == [(10.0, 20.0)] * 3
    assert svis["stage_II"].validation_means == [(10.0, 20.0)] * 3
    assert baseline.history["stage_I_validation_per_valid_modeled_timestamp"] == [10.0, 2.0, 4.0]
    assert baseline.history["stage_II_validation_per_valid_modeled_timestamp"] == [1.0, 5.0, 3.0]
    assert baseline.best_validation["stage_I"]["epoch"] == 2
    assert baseline.best_validation["stage_II"]["epoch"] == 1
    assert stage2_initializations == [2.0]
    assert float(baseline.stage2.weight.item()) == 101.0
    for stage, epoch in (("stage_I", 2), ("stage_II", 1)):
        selection = json.loads(
            (tmp_path / "checkpoints" / "infodpcca" / stage / "best" / "selection.json").read_text()
        )
        assert selection["epoch"] == epoch
        state = json.loads(
            (tmp_path / "checkpoints" / "infodpcca" / stage / "state.json").read_text()
        )
        assert state["best_validation"]["epoch"] == epoch


def test_validation_resume_preserves_stage_optimizer_history_best_and_protocol(tmp_path):
    rng = np.random.default_rng(17)
    raw = BenchmarkDataset(
        rng.normal(size=(1, 5, 3)), rng.normal(size=(1, 5, 2)),
        np.ones((1, 5), bool), ("train",),
    )
    training = raw.standardized()
    validation = BenchmarkDataset(
        raw.x + 0.25, raw.y - 0.25, raw.mask, ("validation",),
    ).standardized(statistics=training.preprocessing_stats)
    config = info_config(1, 2)
    interrupted = InfoDPCCABaseline(config, output_dir=tmp_path)
    save_checkpoint = interrupted._save_checkpoint

    def interrupt_after_stage_II_epoch_one(
        stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds,
        training_protocol,
    ):
        save_checkpoint(
            stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds,
            training_protocol,
        )
        if stage == "stage_II" and completed_epoch == 1:
            raise RuntimeError("simulated Stage-II interruption")

    interrupted._save_checkpoint = interrupt_after_stage_II_epoch_one
    with pytest.raises(RuntimeError, match="simulated Stage-II interruption"):
        interrupted.fit(training, validation_dataset=validation)

    stage_I_state_path = tmp_path / "checkpoints" / "infodpcca" / "stage_I" / "state.json"
    stage_I_state_before = stage_I_state_path.read_text()
    stage_II_state = json.loads(
        (tmp_path / "checkpoints" / "infodpcca" / "stage_II" / "state.json").read_text()
    )
    assert stage_II_state["completed_epoch"] == 1
    assert stage_II_state["best_validation"]["epoch"] == 1
    assert stage_II_state["training_protocol"]["training"]["sequence_ids"] == ["train"]
    assert stage_II_state["training_protocol"]["validation"]["sequence_ids"] == ["validation"]

    probe = InfoDPCCABaseline(config, output_dir=tmp_path)
    probe._build(training)
    protocol = probe._training_protocol(training, validation)
    probe._load_checkpoint("stage_I", probe.stage1, probe.stage1_optimizer, protocol)
    probe.load_best_checkpoint("stage_I", probe.stage1, protocol)
    probe._build_stage2(training)
    assert probe._load_checkpoint("stage_II", probe.stage2, probe.stage2_optimizer, protocol) == 1
    saved_optimizer = torch.load(
        tmp_path / "checkpoints" / "infodpcca" / "stage_II" / stage_II_state["optimizer_file"]
    )
    assert set(probe.stage2_optimizer._state_waiting_to_be_consumed) == set(saved_optimizer)

    resumed = InfoDPCCABaseline(config, output_dir=tmp_path).fit(
        training, validation_dataset=validation
    )
    assert stage_I_state_path.read_text() == stage_I_state_before
    assert len(resumed.history["stage_I"]) == 1
    assert len(resumed.history["stage_I_validation_raw"]) == 1
    assert len(resumed.history["stage_II"]) == 2
    assert len(resumed.history["stage_II_validation_raw"]) == 2
    assert resumed.best_validation["stage_I"] is not None
    assert resumed.best_validation["stage_II"] is not None

    changed_validation = BenchmarkDataset(
        validation.x, validation.y, validation.mask, ("different-validation",),
        validation.metadata, validation.preprocessing_stats,
    )
    with pytest.raises(RuntimeError, match="different or legacy training-data protocol"):
        InfoDPCCABaseline(config, output_dir=tmp_path).fit(
            training, validation_dataset=changed_validation
        )


def _agneuro_fixture():
    trial_ids = tuple(f"synthetic-trial-{index:04d}" for index in range(60))
    assignments = {
        trial_id: "train" if index < 36 else "validation" if index < 48 else "test"
        for index, trial_id in enumerate(trial_ids)
    }
    x = np.repeat(np.arange(60, dtype=float)[:, None, None], 5, axis=1)
    x = np.repeat(x, 32, axis=2)
    y = np.repeat((2.0 * np.arange(60, dtype=float))[:, None, None], 5, axis=1)
    y = np.repeat(y, 16, axis=2)
    mask = np.ones((60, 5), dtype=bool)
    mask[:, 2] = False
    metadata = {
        "dataset_id": "synthetic",
        "trial_ids": list(trial_ids),
        "joint_dataset_metadata": {"trial_splits": assignments},
    }
    return trial_ids, x, y, mask, metadata


def test_standalone_test_evaluation_reuses_split_preprocessing_and_selected_stage_II(
    monkeypatch, tmp_path,
):
    trial_ids, x, y, mask, metadata = _agneuro_fixture()
    raw = BenchmarkDataset(x, y, mask, trial_ids, metadata)
    splits = predefined_trial_datasets(raw)
    assert splits is not None
    assert (len(splits.train.sequence_ids), len(splits.validation.sequence_ids), len(splits.test.sequence_ids)) == (36, 12, 12)
    np.testing.assert_allclose(splits.train.x[splits.train.mask].mean(axis=0), 0.0, atol=1e-12)
    assert float(splits.validation.x[splits.validation.mask].mean()) > 1.0
    assert float(splits.test.x[splits.test.mask].mean()) > float(
        splits.validation.x[splits.validation.mask].mean()
    )

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
    config = info_config()
    (result_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (result_dir / "result.json").write_text(json.dumps({
        "model_name": "infodpcca",
        "training_runtime_seconds": 1.5,
        "inference_runtime_seconds": 0.25,
        "parameter_count": 7,
    }))
    evaluated_means = []

    class EvaluationSVI:
        def evaluate_loss(self, x_value, y_value):
            evaluated_means.append(float(x_value.mean()))
            return float(x_value.square().sum() + y_value.square().sum())

    def fake_build(self, dataset):
        self.stage1 = torch.nn.Linear(1, 1, bias=False)
        self.stage1.device = torch.device("cpu")
        self.stage1_optimizer = FakeOptimizer()
        self.stage1_svi = EvaluationSVI()
        self._author = object()
        return self._author, self.stage1_svi

    def fake_build_stage2(self, dataset):
        self.stage2 = torch.nn.Linear(1, 1, bias=False)
        self.stage2.device = torch.device("cpu")
        self.stage2_optimizer = FakeOptimizer()
        self.stage2_svi = EvaluationSVI()
        return self.stage2_svi

    def fake_transform(self, dataset):
        batch, time_steps = dataset.shape
        shared = np.zeros((batch, time_steps, 2), dtype=float)
        private = np.zeros((batch, time_steps, 1), dtype=float)
        return BenchmarkResult(
            "infodpcca", shared, private, private.copy(), dataset.x.copy(), dataset.y.copy(),
            dataset.mask.copy(), dataset.sequence_ids, self.history, 1.5, 7,
            inference_runtime_seconds=0.25,
        )

    monkeypatch.setattr(InfoDPCCABaseline, "_build", fake_build)
    monkeypatch.setattr(InfoDPCCABaseline, "_build_stage2", fake_build_stage2)
    monkeypatch.setattr(InfoDPCCABaseline, "transform", fake_transform)

    creator = InfoDPCCABaseline(config, output_dir=result_dir)
    creator._build(splits.train)
    protocol = creator._training_protocol(splits.train, splits.validation)
    creator._save_best_checkpoint("stage_I", creator.stage1, 1, 12.0, 1.0, protocol)
    creator._persist_best_selection("stage_I", protocol)
    creator._build_stage2(splits.train)
    creator._save_best_checkpoint("stage_II", creator.stage2, 1, 9.0, 0.75, protocol)
    creator._persist_best_selection("stage_II", protocol)

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
    assert len(evaluated_means) == 48
    assert min(evaluated_means) > 1.0
    assert first["objective"]["valid_modeled_timestamps"] == 24
    assert first["x"]["evaluated_elements"] == 24 * 32
    assert first["y"]["evaluated_elements"] == 24 * 16
    assert "timestamps 2..T" in first["evaluated_time_range"]
    assert (result_dir / "preprocessing.json").read_text() == preprocessing_text
    assert first_latents["shared_latent"].shape == (12, 5, 2)
    assert first_latents["x_private_latent"].shape == (12, 5, 1)
    assert first_latents["y_private_latent"].shape == (12, 5, 1)
    assert first_latents["trial_ids"].tolist() == list(trial_ids[48:])
    assert np.array_equal(first_latents["mask"], mask[48:])
    expected_modeled = np.tile(np.array([False, True, False, False, True]), (12, 1))
    assert np.array_equal(first_latents["modeled_mask"], expected_modeled)
    assert np.array_equal(first_latents["time_indices"], np.tile(np.arange(5), (12, 1)))
    for key in first_latents:
        assert np.array_equal(first_latents[key], second_latents[key])
    assert (result_dir / "test_metrics.json").is_file()
    assert (result_dir / "test_reconstructions.npz").is_file()
