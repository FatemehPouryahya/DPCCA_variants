import json

import numpy as np
import pytest
import yaml

from baselines.dpctw.checkpoint import DPCTWCheckpointStore, checkpoint_signature
from baselines.dpctw.model import DPCTWBaseline
from data.agneuro_adapter import BenchmarkDataset, predefined_trial_datasets
from evaluate import evaluate_result
from run_baseline import run


def config(outer_iterations=1):
    return {
        "seed": 4,
        "device": "cpu",
        "latent": {"shared_dim": 1, "x_private_dim": 1, "y_private_dim": 1},
        "data": {"standardize_x": True, "standardize_y": True},
        "dpcca": {"max_em_iterations": 1, "tolerance": 1e-12},
        "alignment": {
            "max_iterations": outer_iterations,
            "tolerance": 1e-12,
            "band_radius": 4,
        },
    }


def small_dataset(identifier, offset=0.0, trials=2, length=6, x_dim=3, y_dim=2):
    time = np.linspace(-1.0, 1.0, length)
    x = np.stack([
        np.stack([
            time + offset + trial * 0.1,
            time ** 2 + 0.2 * trial,
            np.sin(time + trial),
        ], axis=1)[:, :x_dim]
        for trial in range(trials)
    ])
    y = np.stack([
        np.stack([
            np.cos(time + trial) + offset,
            time * (trial + 1),
        ], axis=1)[:, :y_dim]
        for trial in range(trials)
    ])
    return BenchmarkDataset(
        x, y, np.ones((trials, length), bool),
        tuple(f"{identifier}-{index}" for index in range(trials)),
    )


def test_validation_uses_fixed_model_and_training_updates_only_training(monkeypatch, tmp_path):
    training = small_dataset("train", offset=0.0)
    validation = small_dataset("validation", offset=100.0)
    import baselines.dpctw.em as em

    original_maximization = em.maximization
    update_means = []

    def recording_maximization(parameters, pairs, results):
        update_means.extend(float(x.mean()) for x, _ in pairs)
        return original_maximization(parameters, pairs, results)

    monkeypatch.setattr(em, "maximization", recording_maximization)
    baseline = DPCTWBaseline(config(), output_dir=tmp_path).fit(
        training, validation_dataset=validation
    )
    assert update_means
    assert max(abs(value) for value in update_means) < 10.0
    assert len(baseline.validation_history) == 2
    assert [row["candidate_phase"] for row in baseline.validation_history] == [
        "outer_complete", "final_complete",
    ]
    before = {
        name: value.copy() for name, value in baseline.best_model.state_dict().items()
    }
    objective, alignment = baseline._evaluate_partition(
        baseline.best_model, validation
    )
    assert np.isfinite(objective["raw_loss"])
    assert len(alignment.paths) == len(validation.sequence_ids)
    for name, value in before.items():
        np.testing.assert_array_equal(baseline.best_model.state_dict()[name], value)


def test_strict_best_selection_is_independent_of_final_model(monkeypatch, tmp_path):
    training = small_dataset("train")
    validation = small_dataset("validation", offset=0.3)
    losses = iter((30.0, 10.0, 20.0))
    original = DPCTWBaseline._evaluate_partition

    def scheduled_validation(self, model, dataset):
        objective, alignment = original(self, model, dataset)
        raw = next(losses)
        count = objective["valid_modeled_timestamps"]
        objective["raw_loss"] = raw
        objective["loss_per_valid_modeled_timestamp"] = raw / count
        return objective, alignment

    monkeypatch.setattr(DPCTWBaseline, "_evaluate_partition", scheduled_validation)
    baseline = DPCTWBaseline(config(outer_iterations=2), output_dir=tmp_path).fit(
        training, validation_dataset=validation
    )
    assert len(baseline.validation_history) == 3
    assert baseline.best_validation["completed_iteration"] == 2
    assert baseline.best_validation["candidate_phase"] == "outer_complete"
    assert baseline.best_validation["validation_loss_per_valid_modeled_timestamp"] == (
        baseline.validation_history[1]["validation_loss_per_valid_modeled_timestamp"]
    )
    selection = json.loads(
        (tmp_path / "checkpoints" / "dpctw" / "best" / "selection.json").read_text()
    )
    assert selection["completed_iteration"] == 2
    assert (tmp_path / selection["model_file"]).is_file()
    assert baseline.model is not baseline.best_model
    assert (tmp_path / "checkpoints" / "dpctw" / "checkpoint_0.npz").is_file()
    assert (tmp_path / "checkpoints" / "dpctw" / "checkpoint_1.npz").is_file()


def test_validation_history_and_best_model_resume_at_next_outer_iteration(monkeypatch, tmp_path):
    training = small_dataset("train")
    validation = small_dataset("validation", offset=0.2)
    cfg = config(outer_iterations=2)
    original_save = DPCTWCheckpointStore.save
    interrupted = False

    def interrupt_after_first_outer(self, state, parameters, paths):
        nonlocal interrupted
        result = original_save(self, state, parameters, paths)
        if (
            not interrupted
            and state["phase"] == "outer_complete"
            and state["outer_next_iteration"] == 1
        ):
            interrupted = True
            raise RuntimeError("simulated interruption after validated outer iteration")
        return result

    monkeypatch.setattr(DPCTWCheckpointStore, "save", interrupt_after_first_outer)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        DPCTWBaseline(cfg, output_dir=tmp_path).fit(
            training, validation_dataset=validation
        )
    signature = checkpoint_signature(cfg, training, validation)
    state = DPCTWCheckpointStore(tmp_path, signature).load()["state"]
    assert state["outer_next_iteration"] == 1
    assert len(state["validation_history"]) == 1
    assert state["best_validation"]["completed_iteration"] == 1
    saved_runtime = state["training_runtime_seconds"]

    resumed = DPCTWBaseline(cfg, output_dir=tmp_path).fit(
        training, validation_dataset=validation
    )
    assert [row["completed_iteration"] for row in resumed.validation_history] == [1, 2, 3]
    assert resumed.training_runtime_seconds >= saved_runtime
    assert resumed.best_validation["completed_iteration"] in (1, 2, 3)

    changed = BenchmarkDataset(
        validation.x, validation.y, validation.mask,
        tuple(f"changed-{index}" for index in range(len(validation.sequence_ids))),
        validation.metadata, validation.preprocessing_stats,
    )
    with pytest.raises(RuntimeError, match="different/legacy"):
        DPCTWBaseline(cfg, output_dir=tmp_path).fit(
            training, validation_dataset=changed
        )


def agneuro_arrays():
    trial_ids = tuple(f"synthetic-trial-{index:04d}" for index in range(60))
    assignments = {
        trial_id: "train" if index < 36 else "validation" if index < 48 else "test"
        for index, trial_id in enumerate(trial_ids)
    }
    time = np.linspace(-1.0, 1.0, 5)
    x = np.empty((60, 5, 32), dtype=float)
    y = np.empty((60, 5, 16), dtype=float)
    for trial in range(60):
        x[trial] = (
            trial + time[:, None] * np.linspace(0.2, 1.0, 32)[None, :]
            + np.linspace(-0.5, 0.5, 32)[None, :]
        )
        y[trial] = (
            2.0 * trial + time[:, None] * np.linspace(1.0, 0.2, 16)[None, :]
            + np.linspace(0.4, -0.4, 16)[None, :]
        )
    mask = np.ones((60, 5), dtype=bool)
    mask[:, 2] = False
    metadata = {
        "dataset_id": "synthetic",
        "trial_ids": list(trial_ids),
        "joint_dataset_metadata": {"trial_splits": assignments},
    }
    return trial_ids, x, y, mask, metadata


def test_predefined_dpctw_end_to_end_and_standalone_reproducibility(tmp_path):
    trial_ids, x, y, mask, metadata = agneuro_arrays()
    raw = BenchmarkDataset(x, y, mask, trial_ids, metadata)
    splits = predefined_trial_datasets(raw)
    assert splits is not None
    assert (
        len(splits.train.sequence_ids),
        len(splits.validation.sequence_ids),
        len(splits.test.sequence_ids),
    ) == (36, 12, 12)
    np.testing.assert_allclose(
        splits.train.x[splits.train.mask].mean(axis=0), 0.0, atol=1e-12
    )
    assert float(splits.validation.x[splits.validation.mask].mean()) > 1.0
    assert float(splits.test.x[splits.test.mask].mean()) > float(
        splits.validation.x[splits.validation.mask].mean()
    )

    data_dir = tmp_path / "data"
    result_dir = tmp_path / "result"
    config_path = tmp_path / "config.yaml"
    data_dir.mkdir()
    np.savez(data_dir / "arrays.npz", x=x, y=y, mask=mask)
    (data_dir / "metadata.json").write_text(json.dumps(metadata))
    config_path.write_text(yaml.safe_dump(config()))
    run("dpctw", data_dir, config_path, result_dir)

    split_manifest = json.loads((result_dir / "trial_splits.json").read_text())
    assert split_manifest["counts"] == {"test": 12, "train": 36, "validation": 12}
    assert split_manifest["splits"]["test"] == list(trial_ids[48:])
    result_summary = json.loads((result_dir / "result.json").read_text())
    assert result_summary["sequence_ids"] == list(trial_ids[48:])
    selection = json.loads(
        (result_dir / "checkpoints" / "dpctw" / "best" / "selection.json").read_text()
    )
    best_path = result_dir / selection["model_file"]
    before_best = best_path.read_bytes()
    before_final = (result_dir / "model.npz").read_bytes()

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
    assert best_path.read_bytes() == before_best
    assert (result_dir / "model.npz").read_bytes() == before_final
    assert first["trial_ids"] == list(trial_ids[48:])
    assert first_latents["shared_latent"].shape == (12, 5, 1)
    assert first_latents["x_private_latent"].shape == (12, 5, 1)
    assert first_latents["y_private_latent"].shape == (12, 5, 1)
    assert first_latents["trial_ids"].tolist() == list(trial_ids[48:])
    assert np.array_equal(first_latents["mask"], mask[48:])
    assert np.array_equal(first_latents["time_indices"], np.tile(np.arange(5), (12, 1)))
    assert first_latents["path_sequence_offsets"].shape == (13,)
    assert first_latents["path_pairs"].shape[1] == 2
    for key in first_latents:
        if np.issubdtype(first_latents[key].dtype, np.number):
            assert np.array_equal(first_latents[key], second_latents[key], equal_nan=True)
        else:
            assert np.array_equal(first_latents[key], second_latents[key])
    assert (result_dir / "test_metrics.json").is_file()
    assert (result_dir / "test_reconstructions.npz").is_file()
