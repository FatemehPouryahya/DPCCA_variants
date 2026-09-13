"""Small PyTorch/CUDA helpers used by the CUDA DPCTW path."""

from __future__ import annotations

import numpy as np

from .backend import NumericalBackend, backend_for
from .cuda_rts import kalman_filter_smoother


def cuda_backend() -> NumericalBackend:
    return backend_for("cuda")


def squared_distance_matrix_cuda(x, y):
    backend = cuda_backend()
    x = backend.asarray(x)
    y = backend.asarray(y)
    x_norm = backend.sum(x * x, axis=1)[:, None]
    y_norm = backend.sum(y * y, axis=1)[None, :]
    return backend.to_numpy(backend.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0))


def _inverse_sqrt_cuda(covariance, backend: NumericalBackend, ridge: float = 1e-6):
    eigenvalues, eigenvectors = backend.eigh((covariance + covariance.T) * 0.5)
    return (
        eigenvectors * (1.0 / backend.sqrt(backend.maximum(eigenvalues, ridge)))
    ) @ eigenvectors.T


def observation_initialization_cuda(
    x: np.ndarray, y: np.ndarray, dimensions: int,
) -> tuple[np.ndarray, np.ndarray]:
    backend = cuda_backend()
    x = backend.asarray(x)
    y = backend.asarray(y)
    paired = min(len(x), len(y))
    x_centered = x - backend.mean(x[:paired], axis=0, keepdims=True)
    y_centered = y - backend.mean(y[:paired], axis=0, keepdims=True)
    normalizer = max(1, paired - 1)
    x_whitener = _inverse_sqrt_cuda(x_centered[:paired].T @ x_centered[:paired] / normalizer, backend)
    y_whitener = _inverse_sqrt_cuda(y_centered[:paired].T @ y_centered[:paired] / normalizer, backend)
    cross = x_centered[:paired].T @ y_centered[:paired] / normalizer
    left, _, right_t = backend.svd(x_whitener @ cross @ y_whitener)
    available = min(dimensions, left.shape[1], right_t.shape[0])
    x_scores = backend.zeros((len(x), dimensions))
    y_scores = backend.zeros((len(y), dimensions))
    x_scores[:, :available] = x_centered @ x_whitener @ left[:, :available]
    y_scores[:, :available] = y_centered @ y_whitener @ right_t.T[:, :available]
    return backend.to_numpy(x_scores), backend.to_numpy(y_scores)


def infer_view_cuda(parameters, values: np.ndarray, view: str, backend: NumericalBackend):
    ds, du, dv = parameters.dimensions
    if view == "x":
        indices = backend.concatenate([backend.arange(ds), backend.arange(ds, ds + du)])
        emission = backend.concatenate(
            [parameters.x_shared_loading, parameters.x_private_loading], axis=1
        )
        noise = parameters.x_noise
    elif view == "y":
        indices = backend.concatenate([
            backend.arange(ds), backend.arange(ds + du, ds + du + dv)
        ])
        emission = backend.concatenate(
            [parameters.y_shared_loading, parameters.y_private_loading], axis=1
        )
        noise = parameters.y_noise
    else:
        raise ValueError("view must be x or y")
    transition = parameters.transition[indices][:, indices]
    process = parameters.process_covariance[indices][:, indices]
    initial_covariance = parameters.initial_covariance[indices][:, indices]
    return kalman_filter_smoother(
        values, transition, process, emission, noise,
        parameters.initial_mean[indices], initial_covariance, backend=backend,
    )


def reconstruct_cuda(state, loading, backend: NumericalBackend) -> np.ndarray:
    return backend.to_numpy(state @ loading.T)
