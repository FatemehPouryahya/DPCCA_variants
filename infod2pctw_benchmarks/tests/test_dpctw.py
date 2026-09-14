import numpy as np
import pytest
import torch

from baselines.dpctw.alignment import dtw_path, squared_distance_matrix, validate_path
from baselines.dpctw.checkpoint import DPCTWCheckpointStore, checkpoint_signature
from baselines.dpctw.model import DPCCAModel, DPCTWBaseline
from baselines.dpctw.rts import kalman_filter_smoother
from data.agneuro_adapter import BenchmarkDataset


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


def dpctw_dataset(seed=11, length=18):
    rng = np.random.default_rng(seed)
    shared = np.sin(np.linspace(0, 2.0 * np.pi, length))[:, None]
    x = np.concatenate([shared, np.cos(shared)], axis=1)
    y = np.concatenate([shared, shared * shared], axis=1)
    x += rng.normal(scale=0.015, size=x.shape)
    y += rng.normal(scale=0.015, size=y.shape)
    return BenchmarkDataset(
        x[None, ...], y[None, ...], np.ones((1, length), dtype=bool), ("synthetic",)
    )


def dpctw_config(device):
    return {
        "seed": 3,
        "device": device,
        "latent": {"shared_dim": 1, "x_private_dim": 1, "y_private_dim": 1},
        "dpcca": {"max_em_iterations": 2, "tolerance": 1e-8},
        "alignment": {"max_iterations": 2, "tolerance": 1e-8, "band_radius": 4},
    }


def test_dpctw_cpu_smoke():
    dataset = dpctw_dataset()
    baseline = DPCTWBaseline(dpctw_config("cpu")).fit(dataset)
    result = baseline.transform(dataset)
    assert result.shared_latent.shape == (1, 18, 1)
    assert result.x_reconstruction.shape == dataset.x.shape
    assert result.y_reconstruction.shape == dataset.y.shape
    assert np.all(np.isfinite(result.shared_latent))
    assert result.model_specific["device"] == "cpu"
    validate_path(baseline.alignment.paths[0], 18, 18)


def assert_same_dpctw_fit(actual, expected):
    assert actual.alignment.history == expected.alignment.history
    assert len(actual.alignment.paths) == len(expected.alignment.paths)
    for actual_path, expected_path in zip(actual.alignment.paths, expected.alignment.paths):
        assert np.array_equal(actual_path, expected_path)
    assert actual.model.history == expected.model.history
    for name, expected_value in expected.model.state_dict().items():
        np.testing.assert_allclose(actual.model.state_dict()[name], expected_value)


def test_dpctw_resumes_after_completed_em_iteration(tmp_path, monkeypatch):
    dataset = dpctw_dataset()
    config = dpctw_config("cpu")
    expected = DPCTWBaseline(config).fit(dataset)
    output = tmp_path / "em_interruption"
    original_save = DPCTWCheckpointStore.save
    interrupted = False
    resumed_states = []

    def interrupt_after_first_em(self, state, parameters, paths):
        nonlocal interrupted
        result = original_save(self, state, parameters, paths)
        if not interrupted and state["phase"] == "outer_em" and state["em_next_iteration"] == 1:
            interrupted = True
            raise RuntimeError("simulated Slurm termination")
        if interrupted:
            resumed_states.append(dict(state))
        return result

    monkeypatch.setattr(DPCTWCheckpointStore, "save", interrupt_after_first_em)
    with pytest.raises(RuntimeError, match="simulated Slurm termination"):
        DPCTWBaseline(config, output_dir=output).fit(dataset)

    store = DPCTWCheckpointStore(output, checkpoint_signature(config, dataset))
    saved = store.load()
    assert saved["state"]["phase"] == "outer_em"
    assert saved["state"]["em_next_iteration"] == 1
    saved_runtime = saved["state"]["training_runtime_seconds"]

    resumed = DPCTWBaseline(config, output_dir=output).fit(dataset)
    assert resumed_states[0]["phase"] == "outer_em"
    assert resumed_states[0]["em_next_iteration"] == 2
    assert resumed.training_runtime_seconds >= saved_runtime
    assert_same_dpctw_fit(resumed, expected)


def test_dpctw_resumes_after_completed_outer_iteration(tmp_path, monkeypatch):
    dataset = dpctw_dataset()
    config = dpctw_config("cpu")
    expected = DPCTWBaseline(config).fit(dataset)
    output = tmp_path / "outer_interruption"
    original_save = DPCTWCheckpointStore.save
    interrupted = False
    resumed_states = []

    def interrupt_after_first_outer(self, state, parameters, paths):
        nonlocal interrupted
        result = original_save(self, state, parameters, paths)
        if not interrupted and state["phase"] == "outer_complete" and state["outer_next_iteration"] == 1:
            interrupted = True
            raise RuntimeError("simulated Slurm termination")
        if interrupted:
            resumed_states.append(dict(state))
        return result

    monkeypatch.setattr(DPCTWCheckpointStore, "save", interrupt_after_first_outer)
    with pytest.raises(RuntimeError, match="simulated Slurm termination"):
        DPCTWBaseline(config, output_dir=output).fit(dataset)

    store = DPCTWCheckpointStore(output, checkpoint_signature(config, dataset))
    saved = store.load()
    assert saved["state"]["phase"] == "outer_complete"
    assert saved["state"]["outer_next_iteration"] == 1

    resumed = DPCTWBaseline(config, output_dir=output).fit(dataset)
    assert resumed_states[0]["phase"] == "outer_em"
    assert resumed_states[0]["outer_iteration"] == 1
    assert_same_dpctw_fit(resumed, expected)


def test_dpctw_resumes_final_em_without_repeating_iteration(tmp_path, monkeypatch):
    dataset = dpctw_dataset()
    config = dpctw_config("cpu")
    expected = DPCTWBaseline(config).fit(dataset)
    output = tmp_path / "final_em_interruption"
    original_save = DPCTWCheckpointStore.save
    interrupted = False
    resumed_states = []

    def interrupt_during_final_em(self, state, parameters, paths):
        nonlocal interrupted
        result = original_save(self, state, parameters, paths)
        if not interrupted and state["phase"] == "final_em" and state["em_next_iteration"] == 1:
            interrupted = True
            raise RuntimeError("simulated Slurm termination")
        if interrupted:
            resumed_states.append(dict(state))
        return result

    monkeypatch.setattr(DPCTWCheckpointStore, "save", interrupt_during_final_em)
    with pytest.raises(RuntimeError, match="simulated Slurm termination"):
        DPCTWBaseline(config, output_dir=output).fit(dataset)

    resumed = DPCTWBaseline(config, output_dir=output).fit(dataset)
    assert resumed_states[0]["phase"] == "final_em"
    assert resumed_states[0]["em_next_iteration"] == 2
    assert_same_dpctw_fit(resumed, expected)

    completed = DPCTWCheckpointStore(output, checkpoint_signature(config, dataset)).load()
    assert completed["state"]["phase"] == "final_complete"

    import baselines.dpctw.em as cpu_em

    def reject_repeated_em(*args, **kwargs):
        raise AssertionError("completed final EM was repeated")

    monkeypatch.setattr(cpu_em, "expectation", reject_repeated_em)
    restarted_completed = DPCTWBaseline(config, output_dir=output).fit(dataset)
    assert_same_dpctw_fit(restarted_completed, expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is not available")
def test_dpctw_cpu_cuda_regression():
    dataset = dpctw_dataset()
    cpu = DPCTWBaseline(dpctw_config("cpu")).fit(dataset)
    cuda = DPCTWBaseline(dpctw_config("cuda")).fit(dataset)
    cpu_result = cpu.transform(dataset)
    cuda_result = cuda.transform(dataset)

    np.testing.assert_allclose(
        cuda_result.shared_latent, cpu_result.shared_latent, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda_result.x_private_latent, cpu_result.x_private_latent, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda_result.y_private_latent, cpu_result.y_private_latent, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda.alignment.x_shared, cpu.alignment.x_shared, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda.alignment.y_shared, cpu.alignment.y_shared, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda_result.x_reconstruction, cpu_result.x_reconstruction, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda_result.y_reconstruction, cpu_result.y_reconstruction, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        cuda.model.history, cpu.model.history, rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        [item["alignment_objective"] for item in cuda.alignment.history],
        [item["alignment_objective"] for item in cpu.alignment.history],
        rtol=2e-4,
        atol=2e-5,
    )
    assert len(cuda.alignment.history) == len(cpu.alignment.history)
    assert [item["path_changed"] for item in cuda.alignment.history] == [
        item["path_changed"] for item in cpu.alignment.history
    ]
    assert np.array_equal(cuda.alignment.paths[0], cpu.alignment.paths[0])
    assert cuda_result.model_specific["device"] == "cuda"
    assert cuda_result.model_specific["cuda_cpu_bound_component"] == (
        "exact DTW dynamic-programming recurrence and traceback"
    )
