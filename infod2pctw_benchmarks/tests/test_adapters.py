import numpy as np

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
