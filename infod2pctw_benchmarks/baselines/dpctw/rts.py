"""Kalman filtering and Rauch--Tung--Striebel smoothing."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RTSResult:
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    smoothed_mean: np.ndarray
    smoothed_covariance: np.ndarray
    lag_covariance: np.ndarray
    log_likelihood: float


def _symmetrize(value: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    value = (value + value.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _update_diagonal_noise(
    mean: np.ndarray, covariance: np.ndarray, observation: np.ndarray,
    emission: np.ndarray, noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Information-form update, avoiding a large observation-space inverse."""
    inverse_noise = 1.0 / np.maximum(noise, 1e-10)
    inverse_covariance = np.linalg.inv(covariance)
    information = inverse_covariance + emission.T @ (inverse_noise[:, None] * emission)
    posterior_covariance = _symmetrize(np.linalg.inv(information))
    residual = observation - emission @ mean
    rhs = emission.T @ (inverse_noise * residual)
    posterior_mean = mean + posterior_covariance @ rhs

    sign_p, logdet_p = np.linalg.slogdet(covariance)
    sign_i, logdet_i = np.linalg.slogdet(information)
    if sign_p <= 0 or sign_i <= 0:
        raise np.linalg.LinAlgError("non-positive covariance in Kalman update")
    logdet = np.log(noise).sum() + logdet_p + logdet_i
    quadratic = residual @ (inverse_noise * residual) - rhs @ posterior_covariance @ rhs
    log_likelihood = -0.5 * (observation.size * np.log(2.0 * np.pi) + logdet + quadratic)
    return posterior_mean, posterior_covariance, float(log_likelihood)


def kalman_filter_smoother(
    observations: np.ndarray, transition: np.ndarray, process_covariance: np.ndarray,
    emission: np.ndarray, observation_covariance: np.ndarray,
    initial_mean: np.ndarray, initial_covariance: np.ndarray,
    observation_mask: np.ndarray | None = None,
) -> RTSResult:
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError("observations must have shape [T,D]")
    length, observation_dim = observations.shape
    latent_dim = initial_mean.size
    if observation_mask is None:
        observation_mask = np.ones_like(observations, dtype=bool)
    else:
        observation_mask = np.asarray(observation_mask, dtype=bool)
        if observation_mask.ndim == 1:
            observation_mask = np.broadcast_to(observation_mask[:, None], observations.shape)
    filtered_mean = np.empty((length, latent_dim))
    filtered_covariance = np.empty((length, latent_dim, latent_dim))
    predicted_mean = np.empty_like(filtered_mean)
    predicted_covariance = np.empty_like(filtered_covariance)
    log_likelihood = 0.0

    diagonal_noise = observation_covariance.ndim == 1
    for index in range(length):
        if index == 0:
            mean = np.asarray(initial_mean, dtype=np.float64)
            covariance = _symmetrize(np.asarray(initial_covariance, dtype=np.float64))
        else:
            mean = transition @ filtered_mean[index - 1]
            covariance = _symmetrize(transition @ filtered_covariance[index - 1] @ transition.T + process_covariance)
        predicted_mean[index] = mean
        predicted_covariance[index] = covariance
        valid = observation_mask[index] & np.isfinite(observations[index])
        if valid.any():
            selected_emission = emission[valid]
            selected_observation = observations[index, valid]
            if diagonal_noise:
                mean, covariance, increment = _update_diagonal_noise(
                    mean, covariance, selected_observation, selected_emission, observation_covariance[valid]
                )
            else:
                selected_noise = observation_covariance[np.ix_(valid, valid)]
                innovation = selected_observation - selected_emission @ mean
                innovation_covariance = _symmetrize(selected_emission @ covariance @ selected_emission.T + selected_noise)
                gain = np.linalg.solve(innovation_covariance, selected_emission @ covariance).T
                mean = mean + gain @ innovation
                covariance = _symmetrize(covariance - gain @ selected_emission @ covariance)
                sign, logdet = np.linalg.slogdet(innovation_covariance)
                increment = -0.5 * (valid.sum() * np.log(2 * np.pi) + logdet + innovation @ np.linalg.solve(innovation_covariance, innovation))
            log_likelihood += float(increment)
        filtered_mean[index] = mean
        filtered_covariance[index] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    smoother_gain = np.zeros((max(0, length - 1), latent_dim, latent_dim))
    for index in range(length - 2, -1, -1):
        gain = np.linalg.solve(predicted_covariance[index + 1], transition @ filtered_covariance[index]).T
        smoother_gain[index] = gain
        smoothed_mean[index] = filtered_mean[index] + gain @ (smoothed_mean[index + 1] - predicted_mean[index + 1])
        smoothed_covariance[index] = _symmetrize(
            filtered_covariance[index] + gain @ (smoothed_covariance[index + 1] - predicted_covariance[index + 1]) @ gain.T
        )
    lag_covariance = np.zeros((length, latent_dim, latent_dim))
    for index in range(1, length):
        lag_covariance[index] = smoothed_covariance[index] @ smoother_gain[index - 1].T
    return RTSResult(
        filtered_mean, filtered_covariance, predicted_mean, predicted_covariance,
        smoothed_mean, smoothed_covariance, lag_covariance, log_likelihood,
    )
