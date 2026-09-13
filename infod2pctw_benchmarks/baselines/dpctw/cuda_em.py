"""Constrained EM updates for the linear DPCCA state-space model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .backend import NumericalBackend, backend_for, backend_from_array
from .cuda_rts import RTSResult, kalman_filter_smoother


def _psd(value, backend: NumericalBackend, floor: float = 1e-6):
    value = (value + value.T) * 0.5
    eigenvalues, eigenvectors = backend.eigh(value)
    return (eigenvectors * backend.maximum(eigenvalues, floor)) @ eigenvectors.T


@dataclass
class DPCCAParameters:
    shared_transition: Any
    x_private_transition: Any
    y_private_transition: Any
    shared_process_covariance: Any
    x_private_process_covariance: Any
    y_private_process_covariance: Any
    x_shared_loading: Any
    x_private_loading: Any
    y_shared_loading: Any
    y_private_loading: Any
    x_noise: Any
    y_noise: Any
    initial_mean: Any
    initial_covariance: Any

    @property
    def backend(self) -> NumericalBackend:
        return backend_from_array(self.shared_transition)

    def on(self, backend: NumericalBackend) -> "DPCCAParameters":
        return DPCCAParameters(*(backend.asarray(value) for value in vars(self).values()))

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return (
            self.shared_transition.shape[0],
            self.x_private_transition.shape[0],
            self.y_private_transition.shape[0],
        )

    @property
    def transition(self):
        backend = self.backend
        ds, du, dv = self.dimensions
        value = backend.zeros((ds + du + dv, ds + du + dv))
        value[:ds, :ds] = self.shared_transition
        value[ds:ds + du, ds:ds + du] = self.x_private_transition
        value[ds + du:, ds + du:] = self.y_private_transition
        return value

    @property
    def process_covariance(self):
        backend = self.backend
        ds, du, dv = self.dimensions
        value = backend.zeros((ds + du + dv, ds + du + dv))
        value[:ds, :ds] = self.shared_process_covariance
        value[ds:ds + du, ds:ds + du] = self.x_private_process_covariance
        value[ds + du:, ds + du:] = self.y_private_process_covariance
        return value

    @property
    def emission(self):
        backend = self.backend
        ds, du, dv = self.dimensions
        x_dim = self.x_shared_loading.shape[0]
        y_dim = self.y_shared_loading.shape[0]
        value = backend.zeros((x_dim + y_dim, ds + du + dv))
        value[:x_dim, :ds] = self.x_shared_loading
        value[:x_dim, ds:ds + du] = self.x_private_loading
        value[x_dim:, :ds] = self.y_shared_loading
        value[x_dim:, ds + du:] = self.y_private_loading
        return value

    @property
    def observation_covariance(self):
        return self.backend.concatenate([self.x_noise, self.y_noise])


def _pca_loadings(data, dimensions: int, rng: np.random.Generator, backend: NumericalBackend):
    _, _, right = backend.svd(data - backend.mean(data, axis=0))
    available = min(dimensions, right.shape[0])
    loading = backend.zeros((data.shape[1], dimensions))
    loading[:, :available] = right[:available].T
    if available < dimensions:
        random_values = rng.normal(
            scale=0.05, size=(data.shape[1], dimensions - available)
        )
        loading[:, available:] = backend.asarray(random_values)
    return loading


def initialize_parameters(
    sequences: Iterable[tuple[Any, Any]], shared_dim: int, x_private_dim: int,
    y_private_dim: int, seed: int = 0, backend: NumericalBackend | None = None,
) -> DPCCAParameters:
    backend = backend or backend_for("cpu")
    pairs = [(backend.asarray(x), backend.asarray(y)) for x, y in sequences]
    if not pairs:
        raise ValueError("at least one aligned sequence is required")
    x = backend.concatenate([pair[0] for pair in pairs])
    y = backend.concatenate([pair[1] for pair in pairs])
    rng = np.random.default_rng(seed)
    x_loading = _pca_loadings(x, shared_dim + x_private_dim, rng, backend)
    y_loading = _pca_loadings(y, shared_dim + y_private_dim, rng, backend)
    latent_dim = shared_dim + x_private_dim + y_private_dim
    return DPCCAParameters(
        0.9 * backend.eye(shared_dim),
        0.9 * backend.eye(x_private_dim),
        0.9 * backend.eye(y_private_dim),
        0.1 * backend.eye(shared_dim),
        0.1 * backend.eye(x_private_dim),
        0.1 * backend.eye(y_private_dim),
        x_loading[:, :shared_dim],
        x_loading[:, shared_dim:],
        y_loading[:, :shared_dim],
        y_loading[:, shared_dim:],
        backend.full((x.shape[1],), 0.5),
        backend.full((y.shape[1],), 0.5),
        backend.zeros((latent_dim,)),
        backend.eye(latent_dim),
    )


def expectation(
    parameters: DPCCAParameters, x, y, backend: NumericalBackend | None = None,
) -> RTSResult:
    backend = backend or parameters.backend
    observations = backend.concatenate([backend.asarray(x), backend.asarray(y)], axis=1)
    return kalman_filter_smoother(
        observations, parameters.transition, parameters.process_covariance,
        parameters.emission, parameters.observation_covariance,
        parameters.initial_mean, parameters.initial_covariance, backend=backend,
    )


def _expected_second(result: RTSResult, backend: NumericalBackend):
    return result.smoothed_covariance + backend.einsum(
        "ti,tj->tij", result.smoothed_mean, result.smoothed_mean
    )


def _update_chain(
    results: list[RTSResult], indices, backend: NumericalBackend,
):
    dimension = len(indices)
    denominator = backend.zeros((dimension, dimension))
    numerator = backend.zeros((dimension, dimension))
    transition_count = 0
    for result in results:
        second = _expected_second(result, backend)[:, indices][:, :, indices]
        means = result.smoothed_mean[:, indices]
        cross = result.lag_covariance[:, indices][:, :, indices] + backend.einsum(
            "ti,tj->tij", means, means
        )
        cross[1:] = (
            result.lag_covariance[1:][:, indices][:, :, indices]
            + backend.einsum("ti,tj->tij", means[1:], means[:-1])
        )
        denominator += backend.sum(second[:-1], axis=0)
        numerator += backend.sum(cross[1:], axis=0)
        transition_count += max(0, len(means) - 1)
    transition = numerator @ backend.pinv(denominator)
    covariance_sum = backend.zeros_like(denominator)
    for result in results:
        second = _expected_second(result, backend)[:, indices][:, :, indices]
        means = result.smoothed_mean[:, indices]
        cross = backend.copy(result.lag_covariance[:, indices][:, :, indices])
        cross[1:] += backend.einsum("ti,tj->tij", means[1:], means[:-1])
        for index in range(1, len(means)):
            covariance_sum += (
                second[index] - transition @ cross[index].T - cross[index] @ transition.T
                + transition @ second[index - 1] @ transition.T
            )
    return transition, _psd(covariance_sum / max(1, transition_count), backend)


def _update_emission(
    observations: list[Any], results: list[RTSResult], shared_indices,
    private_indices, backend: NumericalBackend, loading_iterations: int = 50,
    tolerance: float = 1e-8,
):
    """Alternating Eq. 14/15 updates for ``W_i`` and ``B_i``."""
    dimension = observations[0].shape[1]
    shared_second = backend.zeros((len(shared_indices), len(shared_indices)))
    private_second = backend.zeros((len(private_indices), len(private_indices)))
    private_shared = backend.zeros((len(private_indices), len(shared_indices)))
    x_shared = backend.zeros((dimension, len(shared_indices)))
    x_private = backend.zeros((dimension, len(private_indices)))
    count = 0
    for values, result in zip(observations, results):
        second = _expected_second(result, backend)
        shared_mean = result.smoothed_mean[:, shared_indices]
        private_mean = result.smoothed_mean[:, private_indices]
        shared_second += backend.sum(
            second[:, shared_indices][:, :, shared_indices], axis=0
        )
        private_second += backend.sum(
            second[:, private_indices][:, :, private_indices], axis=0
        )
        private_shared += backend.sum(
            second[:, private_indices][:, :, shared_indices], axis=0
        )
        x_shared += values.T @ shared_mean
        x_private += values.T @ private_mean
        count += len(values)
    shared_loading = x_shared @ backend.pinv(shared_second)
    private_loading = x_private @ backend.pinv(private_second)
    for _ in range(loading_iterations):
        old_shared = backend.copy(shared_loading)
        old_private = backend.copy(private_loading)
        shared_loading = (
            x_shared - private_loading @ private_shared
        ) @ backend.pinv(shared_second)
        private_loading = (
            x_private - shared_loading @ private_shared.T
        ) @ backend.pinv(private_second)
        change = max(
            backend.norm(shared_loading - old_shared),
            backend.norm(private_loading - old_private),
        )
        if change < tolerance:
            break
    indices = backend.concatenate([shared_indices, private_indices])
    loading = backend.concatenate([shared_loading, private_loading], axis=1)
    residual_diagonal = backend.zeros((dimension,))
    for values, result in zip(observations, results):
        means = result.smoothed_mean[:, indices]
        second = _expected_second(result, backend)[:, indices][:, :, indices]
        predicted = means @ loading.T
        expected_prediction_square = backend.einsum(
            "di,tij,dj->td", loading, second, loading
        )
        residual_diagonal += backend.sum(
            values * values - 2.0 * values * predicted + expected_prediction_square,
            axis=0,
        )
    isotropic_variance = max(
        backend.scalar(backend.sum(residual_diagonal)) / max(1, count * dimension),
        1e-6,
    )
    return shared_loading, private_loading, backend.full((dimension,), isotropic_variance)


def maximization(
    parameters: DPCCAParameters, pairs: list[tuple[Any, Any]],
    results: list[RTSResult], backend: NumericalBackend | None = None,
) -> DPCCAParameters:
    backend = backend or parameters.backend
    ds, du, dv = parameters.dimensions
    shared = backend.arange(ds)
    x_private = backend.arange(ds, ds + du)
    y_private = backend.arange(ds + du, ds + du + dv)
    shared_transition, shared_q = _update_chain(results, shared, backend)
    x_transition, x_q = _update_chain(results, x_private, backend)
    y_transition, y_q = _update_chain(results, y_private, backend)
    x_shared_loading, x_private_loading, x_noise = _update_emission(
        [pair[0] for pair in pairs], results, shared, x_private, backend
    )
    y_shared_loading, y_private_loading, y_noise = _update_emission(
        [pair[1] for pair in pairs], results, shared, y_private, backend
    )
    initial_mean = backend.mean(
        backend.stack([result.smoothed_mean[0] for result in results]), axis=0
    )
    initial_covariance = backend.mean(backend.stack([
        result.smoothed_covariance[0]
        + backend.outer(
            result.smoothed_mean[0] - initial_mean,
            result.smoothed_mean[0] - initial_mean,
        )
        for result in results
    ]), axis=0)
    return DPCCAParameters(
        shared_transition, x_transition, y_transition, shared_q, x_q, y_q,
        x_shared_loading, x_private_loading, y_shared_loading, y_private_loading,
        x_noise, y_noise, initial_mean, _psd(initial_covariance, backend),
    )


def fit_em(
    pairs: list[tuple[Any, Any]], shared_dim: int, x_private_dim: int,
    y_private_dim: int, max_iterations: int = 20, tolerance: float = 1e-4,
    seed: int = 0, initial_parameters: DPCCAParameters | None = None,
    backend: NumericalBackend | None = None,
) -> tuple[DPCCAParameters, list[float]]:
    backend = backend or backend_for("cpu")
    pairs = [(backend.asarray(x), backend.asarray(y)) for x, y in pairs]
    parameters = (
        initial_parameters.on(backend) if initial_parameters is not None
        else initialize_parameters(
            pairs, shared_dim, x_private_dim, y_private_dim, seed, backend=backend
        )
    )
    history: list[float] = []
    for _ in range(max_iterations):
        results = [expectation(parameters, x, y, backend) for x, y in pairs]
        objective = float(sum(result.log_likelihood for result in results))
        if not np.isfinite(objective):
            raise FloatingPointError("DPCCA EM objective became non-finite")
        history.append(objective)
        if len(history) > 1:
            relative = abs(history[-1] - history[-2]) / max(1.0, abs(history[-2]))
            if relative < tolerance:
                break
        parameters = maximization(parameters, pairs, results, backend)
    return parameters, history

