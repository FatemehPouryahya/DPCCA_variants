"""Constrained EM updates for the linear DPCCA state-space model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .rts import RTSResult, kalman_filter_smoother


def _psd(value: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    value = (value + value.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


@dataclass
class DPCCAParameters:
    shared_transition: np.ndarray
    x_private_transition: np.ndarray
    y_private_transition: np.ndarray
    shared_process_covariance: np.ndarray
    x_private_process_covariance: np.ndarray
    y_private_process_covariance: np.ndarray
    x_shared_loading: np.ndarray
    x_private_loading: np.ndarray
    y_shared_loading: np.ndarray
    y_private_loading: np.ndarray
    x_noise: np.ndarray
    y_noise: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return (self.shared_transition.shape[0], self.x_private_transition.shape[0], self.y_private_transition.shape[0])

    @property
    def transition(self) -> np.ndarray:
        ds, du, dv = self.dimensions
        value = np.zeros((ds + du + dv, ds + du + dv))
        value[:ds, :ds] = self.shared_transition
        value[ds:ds + du, ds:ds + du] = self.x_private_transition
        value[ds + du:, ds + du:] = self.y_private_transition
        return value

    @property
    def process_covariance(self) -> np.ndarray:
        ds, du, dv = self.dimensions
        value = np.zeros((ds + du + dv, ds + du + dv))
        value[:ds, :ds] = self.shared_process_covariance
        value[ds:ds + du, ds:ds + du] = self.x_private_process_covariance
        value[ds + du:, ds + du:] = self.y_private_process_covariance
        return value

    @property
    def emission(self) -> np.ndarray:
        ds, du, dv = self.dimensions
        x_dim = self.x_shared_loading.shape[0]
        y_dim = self.y_shared_loading.shape[0]
        value = np.zeros((x_dim + y_dim, ds + du + dv))
        value[:x_dim, :ds] = self.x_shared_loading
        value[:x_dim, ds:ds + du] = self.x_private_loading
        value[x_dim:, :ds] = self.y_shared_loading
        value[x_dim:, ds + du:] = self.y_private_loading
        return value

    @property
    def observation_covariance(self) -> np.ndarray:
        return np.concatenate([self.x_noise, self.y_noise])


def _pca_loadings(data: np.ndarray, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    _, _, right = np.linalg.svd(data - data.mean(axis=0), full_matrices=False)
    available = min(dimensions, right.shape[0])
    loading = np.zeros((data.shape[1], dimensions))
    loading[:, :available] = right[:available].T
    if available < dimensions:
        loading[:, available:] = rng.normal(scale=0.05, size=(data.shape[1], dimensions - available))
    return loading


def initialize_parameters(
    sequences: Iterable[tuple[np.ndarray, np.ndarray]], shared_dim: int, x_private_dim: int,
    y_private_dim: int, seed: int = 0,
) -> DPCCAParameters:
    pairs = list(sequences)
    if not pairs:
        raise ValueError("at least one aligned sequence is required")
    x = np.concatenate([pair[0] for pair in pairs], axis=0)
    y = np.concatenate([pair[1] for pair in pairs], axis=0)
    rng = np.random.default_rng(seed)
    x_loading = _pca_loadings(x, shared_dim + x_private_dim, rng)
    y_loading = _pca_loadings(y, shared_dim + y_private_dim, rng)
    latent_dim = shared_dim + x_private_dim + y_private_dim
    return DPCCAParameters(
        0.9 * np.eye(shared_dim), 0.9 * np.eye(x_private_dim), 0.9 * np.eye(y_private_dim),
        0.1 * np.eye(shared_dim), 0.1 * np.eye(x_private_dim), 0.1 * np.eye(y_private_dim),
        x_loading[:, :shared_dim], x_loading[:, shared_dim:],
        y_loading[:, :shared_dim], y_loading[:, shared_dim:],
        np.full(x.shape[1], 0.5), np.full(y.shape[1], 0.5),
        np.zeros(latent_dim), np.eye(latent_dim),
    )


def expectation(parameters: DPCCAParameters, x: np.ndarray, y: np.ndarray) -> RTSResult:
    observations = np.concatenate([x, y], axis=1)
    return kalman_filter_smoother(
        observations, parameters.transition, parameters.process_covariance,
        parameters.emission, parameters.observation_covariance,
        parameters.initial_mean, parameters.initial_covariance,
    )


def _expected_second(result: RTSResult) -> np.ndarray:
    return result.smoothed_covariance + np.einsum("ti,tj->tij", result.smoothed_mean, result.smoothed_mean)


def _update_chain(results: list[RTSResult], indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dimension = len(indices)
    denominator = np.zeros((dimension, dimension))
    numerator = np.zeros_like(denominator)
    transition_count = 0
    for result in results:
        second = _expected_second(result)[:, indices][:, :, indices]
        means = result.smoothed_mean[:, indices]
        cross = result.lag_covariance[:, indices][:, :, indices] + np.einsum("ti,tj->tij", means, means)
        # Correct the second vector in E[z_t z_{t-1}^T].
        cross[1:] = result.lag_covariance[1:][:, indices][:, :, indices] + np.einsum("ti,tj->tij", means[1:], means[:-1])
        denominator += second[:-1].sum(axis=0)
        numerator += cross[1:].sum(axis=0)
        transition_count += max(0, len(means) - 1)
    transition = numerator @ np.linalg.pinv(denominator)
    covariance_sum = np.zeros_like(denominator)
    for result in results:
        second = _expected_second(result)[:, indices][:, :, indices]
        means = result.smoothed_mean[:, indices]
        cross = result.lag_covariance[:, indices][:, :, indices]
        cross[1:] += np.einsum("ti,tj->tij", means[1:], means[:-1])
        for index in range(1, len(means)):
            covariance_sum += (
                second[index] - transition @ cross[index].T - cross[index] @ transition.T
                + transition @ second[index - 1] @ transition.T
            )
    return transition, _psd(covariance_sum / max(1, transition_count))


def _update_emission(
    observations: list[np.ndarray], results: list[RTSResult], shared_indices: np.ndarray,
    private_indices: np.ndarray, loading_iterations: int = 50, tolerance: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating Eq. 14/15 updates for ``W_i`` and ``B_i``."""
    dimension = observations[0].shape[1]
    shared_second = np.zeros((len(shared_indices), len(shared_indices)))
    private_second = np.zeros((len(private_indices), len(private_indices)))
    private_shared = np.zeros((len(private_indices), len(shared_indices)))
    x_shared = np.zeros((dimension, len(shared_indices)))
    x_private = np.zeros((dimension, len(private_indices)))
    count = 0
    for values, result in zip(observations, results):
        second = _expected_second(result)
        shared_mean = result.smoothed_mean[:, shared_indices]
        private_mean = result.smoothed_mean[:, private_indices]
        shared_second += second[:, shared_indices][:, :, shared_indices].sum(axis=0)
        private_second += second[:, private_indices][:, :, private_indices].sum(axis=0)
        private_shared += second[:, private_indices][:, :, shared_indices].sum(axis=0)
        x_shared += values.T @ shared_mean
        x_private += values.T @ private_mean
        count += len(values)
    shared_loading = x_shared @ np.linalg.pinv(shared_second)
    private_loading = x_private @ np.linalg.pinv(private_second)
    for _ in range(loading_iterations):
        old_shared = shared_loading.copy()
        old_private = private_loading.copy()
        shared_loading = (x_shared - private_loading @ private_shared) @ np.linalg.pinv(shared_second)
        private_loading = (x_private - shared_loading @ private_shared.T) @ np.linalg.pinv(private_second)
        change = max(np.linalg.norm(shared_loading - old_shared), np.linalg.norm(private_loading - old_private))
        if change < tolerance:
            break
    indices = np.concatenate([shared_indices, private_indices])
    loading = np.concatenate([shared_loading, private_loading], axis=1)
    residual_diagonal = np.zeros(dimension)
    for values, result in zip(observations, results):
        means = result.smoothed_mean[:, indices]
        second = _expected_second(result)[:, indices][:, :, indices]
        predicted = means @ loading.T
        expected_prediction_square = np.einsum("di,tij,dj->td", loading, second, loading)
        residual_diagonal += (values * values - 2.0 * values * predicted + expected_prediction_square).sum(axis=0)
    isotropic_variance = max(float(residual_diagonal.sum() / max(1, count * dimension)), 1e-6)
    return shared_loading, private_loading, np.full(dimension, isotropic_variance)


def maximization(parameters: DPCCAParameters, pairs: list[tuple[np.ndarray, np.ndarray]], results: list[RTSResult]) -> DPCCAParameters:
    ds, du, dv = parameters.dimensions
    shared = np.arange(ds)
    x_private = np.arange(ds, ds + du)
    y_private = np.arange(ds + du, ds + du + dv)
    shared_transition, shared_q = _update_chain(results, shared)
    x_transition, x_q = _update_chain(results, x_private)
    y_transition, y_q = _update_chain(results, y_private)
    x_shared_loading, x_private_loading, x_noise = _update_emission(
        [pair[0] for pair in pairs], results, shared, x_private
    )
    y_shared_loading, y_private_loading, y_noise = _update_emission(
        [pair[1] for pair in pairs], results, shared, y_private
    )
    initial_mean = np.mean([result.smoothed_mean[0] for result in results], axis=0)
    initial_covariance = np.mean([
        result.smoothed_covariance[0] + np.outer(result.smoothed_mean[0] - initial_mean, result.smoothed_mean[0] - initial_mean)
        for result in results
    ], axis=0)
    return DPCCAParameters(
        shared_transition, x_transition, y_transition, shared_q, x_q, y_q,
        x_shared_loading, x_private_loading, y_shared_loading, y_private_loading,
        x_noise, y_noise, initial_mean, _psd(initial_covariance),
    )


def fit_em(
    pairs: list[tuple[np.ndarray, np.ndarray]], shared_dim: int, x_private_dim: int,
    y_private_dim: int, max_iterations: int = 20, tolerance: float = 1e-4,
    seed: int = 0, initial_parameters: DPCCAParameters | None = None,
) -> tuple[DPCCAParameters, list[float]]:
    parameters = initial_parameters or initialize_parameters(pairs, shared_dim, x_private_dim, y_private_dim, seed)
    history: list[float] = []
    for _ in range(max_iterations):
        results = [expectation(parameters, x, y) for x, y in pairs]
        objective = float(sum(result.log_likelihood for result in results))
        if not np.isfinite(objective):
            raise FloatingPointError("DPCCA EM objective became non-finite")
        history.append(objective)
        if len(history) > 1:
            relative = abs(history[-1] - history[-2]) / max(1.0, abs(history[-2]))
            if relative < tolerance:
                break
        parameters = maximization(parameters, pairs, results)
    return parameters, history
