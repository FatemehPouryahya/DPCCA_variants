import numpy as np

from baselines.dpctw.alignment import dtw_path, squared_distance_matrix, validate_path
from baselines.dpctw.model import DPCCAModel
from baselines.dpctw.rts import kalman_filter_smoother


def synthetic(seed=0, length=40):
    rng = np.random.default_rng(seed)
    transition = np.diag([0.85, 0.7, 0.6])
    process = 0.08 * np.eye(3)
    emission = rng.normal(size=(5, 3))
    noise = np.full(5, 0.05)
    latent = np.zeros((length, 3))
    for index in range(1, length):
        latent[index] = transition @ latent[index - 1] + rng.multivariate_normal(np.zeros(3), process)
    observed = latent @ emission.T + rng.normal(scale=np.sqrt(noise), size=(length, 5))
    return transition, process, emission, noise, latent, observed


def test_rts_shapes_and_finite_covariances():
    transition, process, emission, noise, _, observed = synthetic()
    result = kalman_filter_smoother(observed, transition, process, emission, noise, np.zeros(3), np.eye(3))
    assert result.smoothed_mean.shape == (40, 3)
    assert result.smoothed_covariance.shape == (40, 3, 3)
    assert result.lag_covariance.shape == (40, 3, 3)
    assert np.isfinite(result.log_likelihood)
    assert np.linalg.eigvalsh(result.smoothed_covariance).min() > 0


def test_em_is_finite_and_deterministic():
    *_, observed = synthetic()
    pairs = [(observed[:, :3], observed[:, 3:])]
    first = DPCCAModel(1, 1, 1, seed=7).fit(pairs, max_iterations=3)
    second = DPCCAModel(1, 1, 1, seed=7).fit(pairs, max_iterations=3)
    assert np.all(np.isfinite(first.history))
    np.testing.assert_allclose(first.parameters.emission, second.parameters.emission)


def test_dtw_constraints_and_shift_recovery():
    time = np.linspace(0, 4 * np.pi, 60)
    x = np.c_[np.sin(time), np.cos(time)]
    shift = 5
    y = np.vstack([np.repeat(x[[0]], shift, axis=0), x[:-shift]])
    path, objective = dtw_path(squared_distance_matrix(x, y), band_radius=10)
    validate_path(path, len(x), len(y))
    interior = path[(path[:, 0] > 8) & (path[:, 0] < 50)]
    assert abs(np.median(interior[:, 1] - interior[:, 0]) - shift) <= 1
    assert np.isfinite(objective)
