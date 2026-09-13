import json

import numpy as np
import pytest

from data.agneuro_adapter import BenchmarkDataset, load_agneuro


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
