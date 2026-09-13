"""CUDA Kalman filtering and Rauch--Tung--Striebel smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backend import NumericalBackend, backend_for


@dataclass
class RTSResult:
    filtered_mean: Any
    filtered_covariance: Any
    predicted_mean: Any
    predicted_covariance: Any
    smoothed_mean: Any
    smoothed_covariance: Any
    lag_covariance: Any
    log_likelihood: float


def _symmetrize(value, backend: NumericalBackend, floor: float = 1e-8):
    value = (value + value.T) * 0.5
    eigenvalues, eigenvectors = backend.eigh(value)
    return (eigenvectors * backend.maximum(eigenvalues, floor)) @ eigenvectors.T


def _update_diagonal_noise(
    mean, covariance, observation, emission, noise, backend: NumericalBackend,
):
    """Information-form update, avoiding a large observation-space inverse."""
    inverse_noise = 1.0 / backend.maximum(noise, 1e-10)
    inverse_covariance = backend.inv(covariance)
    information = inverse_covariance + emission.T @ (inverse_noise[:, None] * emission)
    posterior_covariance = _symmetrize(backend.inv(information), backend)
    residual = observation - emission @ mean
    rhs = emission.T @ (inverse_noise * residual)
    posterior_mean = mean + posterior_covariance @ rhs

    sign_p, logdet_p = backend.slogdet(covariance)
    sign_i, logdet_i = backend.slogdet(information)
    if backend.scalar(sign_p) <= 0 or backend.scalar(sign_i) <= 0:
        raise np.linalg.LinAlgError("non-positive covariance in Kalman update")
    logdet = backend.sum(backend.log(noise)) + logdet_p + logdet_i
    quadratic = residual @ (inverse_noise * residual) - rhs @ posterior_covariance @ rhs
    log_likelihood = -0.5 * (
        backend.size(observation) * np.log(2.0 * np.pi) + logdet + quadratic
    )
    return posterior_mean, posterior_covariance, backend.scalar(log_likelihood)


def kalman_filter_smoother(
    observations, transition, process_covariance, emission, observation_covariance,
    initial_mean, initial_covariance, observation_mask=None,
    backend: NumericalBackend | None = None,
) -> RTSResult:
    backend = backend or backend_for("cpu")
    observations = backend.asarray(observations)
    transition = backend.asarray(transition)
    process_covariance = backend.asarray(process_covariance)
    emission = backend.asarray(emission)
    observation_covariance = backend.asarray(observation_covariance)
    initial_mean = backend.asarray(initial_mean)
    initial_covariance = backend.asarray(initial_covariance)
    if observations.ndim != 2:
        raise ValueError("observations must have shape [T,D]")
    length, _ = observations.shape
    latent_dim = backend.size(initial_mean)
    if observation_mask is None:
        observation_mask = backend.ones_like_bool(observations)
    else:
        observation_mask = backend.as_bool(observation_mask)
        if observation_mask.ndim == 1:
            observation_mask = backend.broadcast_to(observation_mask[:, None], observations.shape)
    filtered_mean = backend.empty((length, latent_dim))
    filtered_covariance = backend.empty((length, latent_dim, latent_dim))
    predicted_mean = backend.empty_like(filtered_mean)
    predicted_covariance = backend.empty_like(filtered_covariance)
    log_likelihood = 0.0

    diagonal_noise = observation_covariance.ndim == 1
    for index in range(length):
        if index == 0:
            mean = initial_mean
            covariance = _symmetrize(initial_covariance, backend)
        else:
            mean = transition @ filtered_mean[index - 1]
            covariance = _symmetrize(
                transition @ filtered_covariance[index - 1] @ transition.T + process_covariance,
                backend,
            )
        predicted_mean[index] = mean
        predicted_covariance[index] = covariance
        valid = observation_mask[index] & backend.finite(observations[index])
        if backend.any(valid):
            selected_emission = emission[valid]
            selected_observation = observations[index, valid]
            if diagonal_noise:
                mean, covariance, increment = _update_diagonal_noise(
                    mean, covariance, selected_observation, selected_emission,
                    observation_covariance[valid], backend,
                )
            else:
                selected_noise = observation_covariance[valid][:, valid]
                innovation = selected_observation - selected_emission @ mean
                innovation_covariance = _symmetrize(
                    selected_emission @ covariance @ selected_emission.T + selected_noise,
                    backend,
                )
                gain = backend.solve(
                    innovation_covariance, selected_emission @ covariance
                ).T
                mean = mean + gain @ innovation
                covariance = _symmetrize(
                    covariance - gain @ selected_emission @ covariance, backend
                )
                _, logdet = backend.slogdet(innovation_covariance)
                increment = -0.5 * (
                    int(backend.scalar(backend.sum(valid))) * np.log(2 * np.pi)
                    + logdet
                    + innovation @ backend.solve(innovation_covariance, innovation)
                )
                increment = backend.scalar(increment)
            log_likelihood += increment
        filtered_mean[index] = mean
        filtered_covariance[index] = covariance

    smoothed_mean = backend.copy(filtered_mean)
    smoothed_covariance = backend.copy(filtered_covariance)
    smoother_gain = backend.zeros((max(0, length - 1), latent_dim, latent_dim))
    for index in range(length - 2, -1, -1):
        gain = backend.solve(
            predicted_covariance[index + 1], transition @ filtered_covariance[index]
        ).T
        smoother_gain[index] = gain
        smoothed_mean[index] = filtered_mean[index] + gain @ (
            smoothed_mean[index + 1] - predicted_mean[index + 1]
        )
        smoothed_covariance[index] = _symmetrize(
            filtered_covariance[index]
            + gain @ (smoothed_covariance[index + 1] - predicted_covariance[index + 1]) @ gain.T,
            backend,
        )
    lag_covariance = backend.zeros((length, latent_dim, latent_dim))
    for index in range(1, length):
        lag_covariance[index] = smoothed_covariance[index] @ smoother_gain[index - 1].T
    return RTSResult(
        filtered_mean, filtered_covariance, predicted_mean, predicted_covariance,
        smoothed_mean, smoothed_covariance, lag_covariance, float(log_likelihood),
    )

