import json

import numpy as np
import pytest

from data.agneuro_adapter import BenchmarkDataset, load_agneuro, predefined_trial_datasets


def test_agneuro_loading_ids_shapes_and_standardization(tmp_path):
    x = np.arange(30, dtype=float).reshape(1, 5, 6)
    y = np.arange(10, dtype=float).reshape(1, 5, 2)
    mask = np.array([[True, True, False, True, True]])
    np.savez(tmp_path / "arrays.npz", x=x, y=y, mask=mask)
    (tmp_path / "metadata.json").write_text(json.dumps({"recording_ids": ["stable-id"], "dataset_id": "test/a"}))
    dataset = load_agneuro(tmp_path)
    assert dataset.x.shape == (1, 5, 6)
    assert dataset.sequence_ids == ("stable-id",)
    assert dataset.as_layout("x", "TBD").shape == (5, 1, 6)
    assert dataset.as_layout("x", "TD").shape == (5, 6)
    standardized = dataset.standardized()
    np.testing.assert_allclose(standardized.x[mask].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(standardized.x[mask].std(axis=0), 1.0, atol=1e-12)
    assert len(list(standardized.valid_segments())) == 2


def test_td_rejects_multiple_sequences():
    dataset = BenchmarkDataset(np.zeros((2, 3, 1)), np.zeros((2, 3, 1)), np.ones((2, 3), bool), ("a", "b"))
    with pytest.raises(ValueError):
        dataset.as_layout("x", "TD")


def test_predefined_trials_split_before_training_only_standardization():
    trial_ids = tuple(f"synthetic-trial-{index:04d}" for index in range(60))
    assignments = {
        trial_id: "train" if index < 36 else "validation" if index < 48 else "test"
        for index, trial_id in enumerate(trial_ids)
    }
    x = np.zeros((60, 4, 32), dtype=float)
    y = np.zeros((60, 4, 16), dtype=float)
    x[:36] = np.arange(36)[:, None, None]
    y[:36] = 2.0 * np.arange(36)[:, None, None]
    x[36:48] = 10_000.0
    y[36:48] = 20_000.0
    x[48:] = 30_000.0
    y[48:] = 40_000.0
    mask = np.tile(np.array([True, True, False, True]), (60, 1))
    raw = BenchmarkDataset(
        x, y, mask, trial_ids,
        {"trial_ids": list(trial_ids), "joint_dataset_metadata": {"trial_splits": assignments}},
    )

    splits = predefined_trial_datasets(raw)
    assert splits is not None
    assert (len(splits.train.sequence_ids), len(splits.validation.sequence_ids),
            len(splits.test.sequence_ids)) == (36, 12, 12)
    assert splits.train.sequence_ids == trial_ids[:36]
    assert splits.validation.sequence_ids == trial_ids[36:48]
    assert splits.test.sequence_ids == trial_ids[48:]
    assert set(splits.train.sequence_ids).isdisjoint(splits.validation.sequence_ids)
    assert set(splits.train.sequence_ids).isdisjoint(splits.test.sequence_ids)
    np.testing.assert_allclose(splits.train.preprocessing_stats.x_mean, 17.5)
    np.testing.assert_allclose(splits.train.preprocessing_stats.y_mean, 35.0)
    assert np.all(splits.validation.x[splits.validation.mask] > 900.0)
    assert np.all(splits.test.x[splits.test.mask] > 2_000.0)
    assert np.array_equal(splits.validation.mask, mask[36:48])
    assert len(list(splits.train.valid_segments())) == 36


def test_predefined_split_is_optional_for_backward_compatibility():
    raw = BenchmarkDataset(
        np.zeros((1, 3, 2)), np.zeros((1, 3, 1)), np.ones((1, 3), bool), ("one",),
    )
    assert predefined_trial_datasets(raw) is None
    assert raw.standardized().shape == raw.shape
