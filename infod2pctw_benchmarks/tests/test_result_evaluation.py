import json

import numpy as np

from benchmark import BenchmarkResult, load_result
from evaluate import masked_metrics


def test_result_serialization(tmp_path):
    values = np.ones((1, 4, 2))
    result = BenchmarkResult(
        "test", values, values[..., :1], values[..., :1], values, values,
        np.ones((1, 4), bool), ("id",), {"loss": [1.0]}, 0.1, 12,
    )
    result.save(tmp_path, {"seed": 0}, {"seed": 0})
    loaded = load_result(tmp_path)
    assert loaded.model_name == "test"
    np.testing.assert_array_equal(loaded.shared_latent, values)
    assert json.loads((tmp_path / "metrics.json").read_text()) == {}


def test_evaluation_metrics_respect_mask():
    target = np.array([[[0.0], [2.0], [100.0]]])
    prediction = np.array([[[0.0], [0.0], [-100.0]]])
    metrics = masked_metrics(target, prediction, np.array([[True, True, False]]))
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(2.0))
    assert metrics["evaluated_elements"] == 2
