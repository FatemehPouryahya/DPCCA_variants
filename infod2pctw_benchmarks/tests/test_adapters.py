import json

import numpy as np
import pytest

from baselines.d2pcca import D2PCCABaseline
from baselines.infodpcca import InfoDPCCABaseline
from data.agneuro_adapter import BenchmarkDataset


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

    def interrupt_after_first_epoch(stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds):
        save_checkpoint(stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds)
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

    def interrupt_after_first_epoch(stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds):
        save_checkpoint(stage, model, optimizer, completed_epoch, stage_completed, runtime_seconds)
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
