import json

import numpy as np
import pytest

from baselines.d2pcca import D2PCCABaseline
from baselines.infodpcca import InfoDPCCABaseline
from data.agneuro_adapter import BenchmarkDataset, PreprocessingStats


def dataset():
    rng = np.random.default_rng(3)
    return BenchmarkDataset(
        rng.normal(size=(1, 6, 3)), rng.normal(size=(1, 6, 2)),
        np.ones((1, 6), dtype=bool), ("tiny",),
    )


def d2_config():
    return {
        "seed": 0, "latent": {"shared_dim": 2, "x_private_dim": 1, "y_private_dim": 1},
        "d2pcca": {"epochs": 1, "emission_dims": [4, 4], "transition_dims": [3, 3, 3],
                     "rnn_dim": 5, "annealing_epochs": 1, "optimizer": {"weight_decay": 0.0}},
    }


def info_config():
    return {
        "seed": 0, "latent": {"shared_dim": 2, "x_private_dim": 1, "y_private_dim": 1},
        "infodpcca": {"stage_I_epochs": 1, "stage_II_epochs": 1, "emission_x_dim": 4,
                       "emission_y_dim": 4, "rnn_x_dim": 5, "rnn_y_dim": 5,
                       "rnn4vi": False, "residual_connection": True,
                       "stage_I_optimizer": {"weight_decay": 0.0},
                       "stage_II_optimizer": {"weight_decay": 0.0}},
    }


def test_d2pcca_adapter_construction_and_latent_extraction():
    baseline = D2PCCABaseline(d2_config()).fit(dataset())
    result = baseline.transform(dataset())
    assert result.shared_latent.shape == (1, 6, 2)
    assert np.all(np.isfinite(result.shared_latent))


def test_d2pcca_resumes_latest_complete_checkpoint(tmp_path):
    config = d2_config()
    config["d2pcca"].update({"epochs": 3, "checkpoint_every_epochs": 1})
    training = dataset().standardized()
    baseline = D2PCCABaseline(config, output_dir=tmp_path)
    save_checkpoint = baseline._save_checkpoint

    def interrupt_after_first_epoch(completed_epoch, runtime_seconds, training_protocol):
        save_checkpoint(completed_epoch, runtime_seconds, training_protocol)
        if completed_epoch == 1:
            raise RuntimeError("simulated interruption")

    baseline._save_checkpoint = interrupt_after_first_epoch
    with pytest.raises(RuntimeError, match="simulated interruption"):
        baseline.fit(training)

    checkpoint_dir = tmp_path / "checkpoints" / "d2pcca"
    first_state = json.loads((checkpoint_dir / "state_epoch_000001.json").read_text())
    (checkpoint_dir / "state_epoch_999999.json").write_text(json.dumps({
        "completed_epoch": 999999,
        "model_file": "missing_model.pt",
        "optimizer_file": "missing_optimizer.pt",
    }))

    assert first_state["training_protocol"]["sequence_ids"] == ["tiny"]
    assert first_state["training_protocol"]["preprocessing"] == (
        training.preprocessing_stats.to_jsonable()
    )
    assert first_state["training_protocol_fingerprint"]
    probe = D2PCCABaseline(config, output_dir=tmp_path)
    probe._build(training)
    assert probe._load_latest_checkpoint(probe._training_protocol(training)) == 1
    import torch
    saved_optimizer_state = torch.load(checkpoint_dir / "optimizer_epoch_000001.pt")
    assert saved_optimizer_state
    assert set(probe.optimizer._state_waiting_to_be_consumed) == set(saved_optimizer_state)

    changed_membership = BenchmarkDataset(
        training.x, training.y, training.mask, ("different-trial",),
        training.metadata, training.preprocessing_stats,
    )
    with pytest.raises(RuntimeError, match="different or legacy training-data protocol"):
        D2PCCABaseline(config, output_dir=tmp_path).fit(changed_membership)
    stats = training.preprocessing_stats
    changed_normalization = BenchmarkDataset(
        training.x, training.y, training.mask, training.sequence_ids, training.metadata,
        PreprocessingStats(
            stats.x_mean + 1.0, stats.x_scale, stats.y_mean, stats.y_scale, stats.valid_count,
        ),
    )
    with pytest.raises(RuntimeError, match="different or legacy training-data protocol"):
        D2PCCABaseline(config, output_dir=tmp_path).fit(changed_normalization)

    legacy_state = dict(first_state)
    legacy_state.pop("training_protocol")
    legacy_state.pop("training_protocol_fingerprint")
    state_path = checkpoint_dir / "state_epoch_000001.json"
    state_path.write_text(json.dumps(legacy_state))
    with pytest.raises(RuntimeError, match="different or legacy training-data protocol"):
        D2PCCABaseline(config, output_dir=tmp_path).fit(training)
    state_path.write_text(json.dumps(first_state))

    resumed = D2PCCABaseline(config, output_dir=tmp_path).fit(training)
    assert len(resumed.history["elbo_loss_per_valid_timestamp"]) == 3
    final_state = json.loads((checkpoint_dir / "state_epoch_000003.json").read_text())
    assert final_state["completed_epoch"] == 3
    assert final_state["next_epoch"] == 4
    assert final_state["training_runtime_seconds"] >= first_state["training_runtime_seconds"]
    assert (checkpoint_dir / "model_epoch_000003.pt").is_file()
    assert (checkpoint_dir / "optimizer_epoch_000003.pt").is_file()


def test_d2pcca_fit_updates_only_training_segments_and_validation_is_read_only():
    import torch

    train = BenchmarkDataset(
        np.zeros((2, 4, 3)), np.zeros((2, 4, 2)), np.ones((2, 4), bool),
        ("train-a", "train-b"),
    )
    validation = BenchmarkDataset(
        np.ones((1, 4, 3)), np.ones((1, 4, 2)), np.ones((1, 4), bool),
        ("validation-a",),
    )
    calls = {"step": 0, "evaluate": 0}

    class FakeSVI:
        def step(self, values, factor):
            calls["step"] += 1
            return float(values[0].shape[1])

        def evaluate_loss(self, values, factor):
            calls["evaluate"] += 1
            return float(values[0].shape[1])

    baseline = D2PCCABaseline(d2_config())
    baseline.model = torch.nn.Linear(1, 1)
    baseline.model.device = torch.device("cpu")
    baseline.svi = FakeSVI()
    baseline.fit(train, validation_dataset=validation)
    assert calls == {"step": 2, "evaluate": 1}
    assert baseline.history["validation_elbo_loss_raw"] == [4.0]
    assert baseline.history["validation_elbo_loss_per_valid_timestamp"] == [1.0]
    assert baseline.validation_dataset.sequence_ids == ("validation-a",)
    parameters_before = [value.detach().clone() for value in baseline.model.parameters()]
    assert baseline.validation_loss() == 1.0
    for before, after in zip(parameters_before, baseline.model.parameters()):
        assert torch.equal(before, after)
    assert calls == {"step": 2, "evaluate": 2}


def test_d2pcca_best_checkpoint_survives_worse_epochs(tmp_path):
    import torch

    config = d2_config()
    config["d2pcca"].update({"epochs": 3, "checkpoint_every_epochs": 3})
    train = dataset()
    validation = dataset()

    class FakeOptimizer:
        def save(self, path):
            torch.save({"optimizer": "unchanged"}, path)

    class FakeSVI:
        def __init__(self, model):
            self.model = model
            self.epoch = 0
            self.validation_losses = iter((4.0, 8.0, 6.0))

        def step(self, values, factor):
            self.epoch += 1
            with torch.no_grad():
                self.model.weight.fill_(self.epoch)
            return 1.0

        def evaluate_loss(self, values, factor):
            return next(self.validation_losses)

    baseline = D2PCCABaseline(config, output_dir=tmp_path)
    baseline.model = torch.nn.Linear(1, 1, bias=False)
    baseline.model.device = torch.device("cpu")
    baseline.optimizer = FakeOptimizer()
    baseline.svi = FakeSVI(baseline.model)
    baseline.fit(train, validation_dataset=validation)

    assert baseline.history["validation_elbo_loss_per_valid_timestamp"] == [
        4.0 / 6.0, 8.0 / 6.0, 1.0,
    ]
    assert baseline.best_validation["epoch"] == 1
    assert baseline.best_validation["validation_loss_per_valid_timestamp"] == 4.0 / 6.0
    assert float(baseline.model.weight.item()) == 1.0
    best_states = sorted((tmp_path / "checkpoints" / "d2pcca" / "best").glob("state_*.json"))
    assert [path.name for path in best_states] == ["state_epoch_000001.json"]
    selection = json.loads(
        (tmp_path / "checkpoints" / "d2pcca" / "best" / "selection.json").read_text()
    )
    assert selection["epoch"] == 1
    final_state = json.loads(
        (tmp_path / "checkpoints" / "d2pcca" / "state_epoch_000003.json").read_text()
    )
    assert final_state["best_validation"]["epoch"] == 1


def test_d2pcca_validation_history_and_best_selection_resume(tmp_path):
    config = d2_config()
    config["d2pcca"].update({"epochs": 3, "checkpoint_every_epochs": 3})
    training = dataset().standardized()
    validation = dataset().standardized(statistics=training.preprocessing_stats)
    baseline = D2PCCABaseline(config, output_dir=tmp_path)
    save_resume = baseline._save_resume_checkpoint

    def interrupt_after_first_epoch(completed_epoch, runtime_seconds, training_protocol):
        save_resume(completed_epoch, runtime_seconds, training_protocol)
        if completed_epoch == 1:
            raise RuntimeError("simulated validation interruption")

    baseline._save_resume_checkpoint = interrupt_after_first_epoch
    with pytest.raises(RuntimeError, match="simulated validation interruption"):
        baseline.fit(training, validation_dataset=validation)

    resumed = D2PCCABaseline(config, output_dir=tmp_path).fit(
        training, validation_dataset=validation
    )
    validation_history = resumed.history["validation_elbo_loss_per_valid_timestamp"]
    assert len(validation_history) == 3
    assert resumed.best_validation["validation_loss_per_valid_timestamp"] == min(
        validation_history
    )
    assert resumed.best_validation["epoch"] == validation_history.index(min(validation_history)) + 1
    resume_state = json.loads(
        (tmp_path / "checkpoints" / "d2pcca" / "resume" / "state_epoch_000003.json").read_text()
    )
    assert len(resume_state["history"]["validation_elbo_loss_raw"]) == 3
    assert resume_state["best_validation"]["epoch"] == resumed.best_validation["epoch"]


def test_infodpcca_adapter_construction_and_latent_extraction():
    baseline = InfoDPCCABaseline(info_config()).fit(dataset())
    result = baseline.transform(dataset())
    assert result.shared_latent.shape == (1, 6, 2)
    assert result.x_private_latent.shape[-1] == 1
    assert result.y_private_latent.shape[-1] == 1
    assert np.all(np.isfinite(result.shared_latent))


def test_infodpcca_resumes_interrupted_stage_I(tmp_path):
    config = info_config()
    config["infodpcca"].update({
        "stage_I_epochs": 2,
        "stage_II_epochs": 1,
        "checkpoint_every_epochs": 1,
    })
    baseline = InfoDPCCABaseline(config, output_dir=tmp_path)
    save_checkpoint = baseline._save_checkpoint

    def interrupt_after_first_epoch(
        stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds,
        training_protocol,
    ):
        save_checkpoint(
            stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds,
            training_protocol,
        )
        if stage == "stage_I" and completed_epoch == 1:
            raise RuntimeError("simulated interruption")

    baseline._save_checkpoint = interrupt_after_first_epoch
    with pytest.raises(RuntimeError, match="simulated interruption"):
        baseline.fit(dataset())

    resumed = InfoDPCCABaseline(config, output_dir=tmp_path).fit(dataset())
    assert len(resumed.history["stage_I"]) == 2
    assert len(resumed.history["stage_II"]) == 1
    state = json.loads(
        (tmp_path / "checkpoints" / "infodpcca" / "stage_I" / "state.json").read_text()
    )
    assert state["completed_epoch"] == 2
    assert state["stage_completed"] is True


def test_infodpcca_resumes_stage_II_without_rerunning_stage_I(tmp_path):
    config = info_config()
    config["infodpcca"].update({
        "stage_I_epochs": 1,
        "stage_II_epochs": 2,
        "checkpoint_every_epochs": 1,
    })
    baseline = InfoDPCCABaseline(config, output_dir=tmp_path)
    save_checkpoint = baseline._save_checkpoint

    def interrupt_after_first_epoch(
        stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds,
        training_protocol,
    ):
        save_checkpoint(
            stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds,
            training_protocol,
        )
        if stage == "stage_II" and completed_epoch == 1:
            raise RuntimeError("simulated interruption")

    baseline._save_checkpoint = interrupt_after_first_epoch
    with pytest.raises(RuntimeError, match="simulated interruption"):
        baseline.fit(dataset())

    stage_I_state_path = (
        tmp_path / "checkpoints" / "infodpcca" / "stage_I" / "state.json"
    )
    stage_I_state_before = stage_I_state_path.read_text()
    resumed = InfoDPCCABaseline(config, output_dir=tmp_path).fit(dataset())
    assert stage_I_state_path.read_text() == stage_I_state_before
    assert len(resumed.history["stage_I"]) == 1
    assert len(resumed.history["stage_II"]) == 2
    state = json.loads(
        (tmp_path / "checkpoints" / "infodpcca" / "stage_II" / "state.json").read_text()
    )
    assert state["completed_epoch"] == 2
    assert state["stage_completed"] is True
