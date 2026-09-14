"""DPCTW alternating latent-space temporal alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from data.agneuro_adapter import BenchmarkDataset


def squared_distance_matrix(x: np.ndarray, y: np.ndarray, device: str = "cpu") -> np.ndarray:
    if device == "cuda":
        from .torch_backend import squared_distance_matrix_cuda
        return squared_distance_matrix_cuda(x, y)
    x_norm = np.sum(x * x, axis=1)[:, None]
    y_norm = np.sum(y * y, axis=1)[None, :]
    return np.maximum(x_norm + y_norm - 2.0 * x @ y.T, 0.0)


def dtw_path(distance: np.ndarray, band_radius: int | None = None) -> tuple[np.ndarray, float]:
    """Exact DTW with fixed endpoints and unit monotone/continuous steps."""
    distance = np.asarray(distance, dtype=np.float64)
    if distance.ndim != 2 or min(distance.shape) == 0:
        raise ValueError("distance must be a non-empty [Tx,Ty] matrix")
    n, m = distance.shape
    if band_radius is not None:
        band_radius = max(int(band_radius), abs(n - m))
    cumulative = np.full((n, m), np.inf)
    predecessor = np.full((n, m), -1, dtype=np.int8)
    cumulative[0, 0] = distance[0, 0]
    for i in range(n):
        lower = 0 if band_radius is None else max(0, i - band_radius)
        upper = m if band_radius is None else min(m, i + band_radius + 1)
        for j in range(lower, upper):
            if i == 0 and j == 0:
                continue
            candidates = (
                cumulative[i - 1, j - 1] if i and j else np.inf,
                cumulative[i - 1, j] if i else np.inf,
                cumulative[i, j - 1] if j else np.inf,
            )
            step = int(np.argmin(candidates))
            best = candidates[step]
            if np.isfinite(best):
                cumulative[i, j] = distance[i, j] + best
                predecessor[i, j] = step
    if not np.isfinite(cumulative[-1, -1]):
        raise ValueError("no feasible DTW path under the requested band")
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i or j:
        step = predecessor[i, j]
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            raise RuntimeError("broken DTW predecessor chain")
        path.append((i, j))
    path.reverse()
    return np.asarray(path, dtype=int), float(cumulative[-1, -1])


def validate_path(path: np.ndarray, x_length: int, y_length: int) -> None:
    path = np.asarray(path)
    if path.shape[0] == 0 or tuple(path[0]) != (0, 0) or tuple(path[-1]) != (x_length - 1, y_length - 1):
        raise ValueError("DTW endpoint constraint violated")
    steps = np.diff(path, axis=0)
    allowed = ((steps[:, 0] >= 0) & (steps[:, 1] >= 0) & (steps.sum(axis=1) >= 1)
               & (steps[:, 0] <= 1) & (steps[:, 1] <= 1))
    if not np.all(allowed):
        raise ValueError("DTW monotonicity/continuity constraint violated")


def _inverse_sqrt(covariance: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh((covariance + covariance.T) * 0.5)
    return (eigenvectors * (1.0 / np.sqrt(np.maximum(eigenvalues, ridge)))) @ eigenvectors.T


def observation_initialization(
    x: np.ndarray, y: np.ndarray, dimensions: int, device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Published CTW-style CCA projections ``V_i^T X_i`` for iteration zero."""
    if device == "cuda":
        from .torch_backend import observation_initialization_cuda
        return observation_initialization_cuda(x, y, dimensions)
    paired = min(len(x), len(y))
    x_centered = x - x[:paired].mean(axis=0, keepdims=True)
    y_centered = y - y[:paired].mean(axis=0, keepdims=True)
    normalizer = max(1, paired - 1)
    x_whitener = _inverse_sqrt(x_centered[:paired].T @ x_centered[:paired] / normalizer)
    y_whitener = _inverse_sqrt(y_centered[:paired].T @ y_centered[:paired] / normalizer)
    cross = x_centered[:paired].T @ y_centered[:paired] / normalizer
    left, _, right_t = np.linalg.svd(x_whitener @ cross @ y_whitener, full_matrices=False)
    available = min(dimensions, left.shape[1], right_t.shape[0])
    x_scores = np.zeros((len(x), dimensions))
    y_scores = np.zeros((len(y), dimensions))
    x_scores[:, :available] = x_centered @ x_whitener @ left[:, :available]
    y_scores[:, :available] = y_centered @ y_whitener @ right_t.T[:, :available]
    return x_scores, y_scores


@dataclass
class AlignmentResult:
    paths: list[np.ndarray]
    history: list[dict[str, Any]]
    initialization: str
    x_shared: np.ndarray | None = None
    y_shared: np.ndarray | None = None


def alternating_dpctw(
    dataset: BenchmarkDataset, config: dict[str, Any], resume: dict[str, Any] | None = None,
    checkpoint_callback: Callable[[dict[str, Any], Any, list[np.ndarray]], None] | None = None,
):
    from .model import DPCCAModel

    latent = config["latent"]
    dpcca_config = config.get("dpcca", {})
    alignment_config = config.get("alignment", {})
    device = str(config.get("device", "cpu")).lower()
    if device not in ("cpu", "cuda"):
        raise ValueError("DPCTW device must be 'cpu' or 'cuda'")
    ds = int(latent["shared_dim"])
    compact: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for batch_index in range(dataset.shape[0]):
        indices = np.flatnonzero(dataset.mask[batch_index])
        if len(indices) < 2:
            raise ValueError(f"sequence {dataset.sequence_ids[batch_index]} has fewer than two valid timestamps")
        x = dataset.x[batch_index, indices]
        y = dataset.y[batch_index, indices]
        compact.append((x, y, indices))

    model = DPCCAModel(
        ds, int(latent["x_private_dim"]), int(latent["y_private_dim"]),
        int(config.get("seed", 0)), device=device,
    )
    if resume is not None:
        local_paths = [np.asarray(path, dtype=int) for path in resume["paths"]]
        if len(local_paths) != len(compact):
            raise ValueError("DPCTW checkpoint path count does not match the dataset")
        for path, (x, y, _) in zip(local_paths, compact):
            validate_path(path, len(x), len(y))
        model.load_state_dict(resume["parameters"])
        state = resume["state"]
        history = list(state.get("alignment_history", []))
        previous_objective = state.get("previous_objective")
        last_outer_converged = bool(state.get("outer_converged", False))
        phase = str(state["phase"])
    else:
        local_paths = []
        for x, y, _ in compact:
            x_initial, y_initial = observation_initialization(x, y, ds, device=device)
            path, _ = dtw_path(
                squared_distance_matrix(x_initial, y_initial, device=device),
                alignment_config.get("band_radius"),
            )
            local_paths.append(path)
        state = {}
        history = []
        previous_objective = None
        last_outer_converged = False
        phase = "outer_start"

    max_alignment_iterations = int(alignment_config.get("max_iterations", 5))
    if phase == "outer_em":
        outer_start = int(state["outer_iteration"])
        proceed_to_final = False
    elif phase == "outer_complete":
        outer_start = int(state["outer_next_iteration"])
        proceed_to_final = bool(state.get("outer_converged", False)) or outer_start >= max_alignment_iterations
    elif phase in ("final_em", "final_complete"):
        outer_start = max_alignment_iterations
        proceed_to_final = True
    elif phase == "outer_start":
        outer_start = 0
        proceed_to_final = False
    else:
        raise ValueError(f"unknown DPCTW checkpoint phase: {phase}")

    for iteration in range(outer_start, max_alignment_iterations) if not proceed_to_final else ():
        aligned = [(x[path[:, 0]], y[path[:, 1]]) for (x, y, _), path in zip(compact, local_paths)]
        resume_em = phase == "outer_em" and iteration == outer_start
        em_start = int(state.get("em_next_iteration", 0)) if resume_em else 0
        em_history = list(state.get("em_history", [])) if resume_em else []
        em_complete = bool(state.get("em_complete", False)) if resume_em else False

        def save_outer_em(next_iteration, parameters, current_history, complete):
            if checkpoint_callback is not None:
                checkpoint_callback({
                    "phase": "outer_em",
                    "outer_iteration": iteration,
                    "em_next_iteration": next_iteration,
                    "em_complete": bool(complete),
                    "em_history": list(current_history),
                    "alignment_history": list(history),
                    "previous_objective": previous_objective,
                    "outer_converged": False,
                }, parameters, local_paths)

        if em_complete:
            model.history = em_history
        else:
            model.fit(
                aligned, max_iterations=int(dpcca_config.get("max_em_iterations", 20)),
                tolerance=float(dpcca_config.get("tolerance", 1e-4)),
                warm_start=resume_em or iteration > 0,
                start_iteration=em_start, initial_history=em_history,
                checkpoint_callback=save_outer_em,
            )
        new_paths: list[np.ndarray] = []
        objective = 0.0
        for x, y, _ in compact:
            x_shared = model.infer_x(x).smoothed_mean[:, :ds]
            y_shared = model.infer_y(y).smoothed_mean[:, :ds]
            distance = squared_distance_matrix(x_shared, y_shared, device=device)
            path, cost = dtw_path(distance, alignment_config.get("band_radius"))
            validate_path(path, len(x), len(y))
            new_paths.append(path)
            objective += cost
        changed = any(not np.array_equal(old, new) for old, new in zip(local_paths, new_paths))
        relative = None if previous_objective is None else abs(previous_objective - objective) / max(1.0, abs(previous_objective))
        history.append({
            "iteration": iteration, "alignment_objective": float(objective),
            "relative_objective_change": relative, "path_changed": bool(changed),
            "em_iterations": len(model.history),
            "em_final_log_likelihood": float(model.history[-1]) if model.history else None,
        })
        local_paths = new_paths
        converged = (
            previous_objective is not None and relative is not None
            and relative < float(alignment_config.get("tolerance", 1e-4)) and not changed
        )
        last_outer_converged = bool(converged)
        if checkpoint_callback is not None:
            checkpoint_callback({
                "phase": "outer_complete",
                "outer_iteration": iteration,
                "outer_next_iteration": iteration + 1,
                "outer_converged": bool(converged),
                "em_next_iteration": len(model.history),
                "em_complete": True,
                "em_history": list(model.history),
                "alignment_history": list(history),
                "previous_objective": float(objective),
            }, model.parameters, local_paths)
        phase = "outer_start"
        state = {}
        previous_objective = objective
        if converged:
            break

    final_aligned = [(x[path[:, 0]], y[path[:, 1]]) for (x, y, _), path in zip(compact, local_paths)]
    if phase == "final_complete":
        model.history = list(state.get("em_history", []))
    else:
        resume_final_em = phase == "final_em"
        em_start = int(state.get("em_next_iteration", 0)) if resume_final_em else 0
        em_history = list(state.get("em_history", [])) if resume_final_em else []
        em_complete = bool(state.get("em_complete", False)) if resume_final_em else False

        def save_final_em(next_iteration, parameters, current_history, complete):
            if checkpoint_callback is not None:
                checkpoint_callback({
                    "phase": "final_em",
                    "outer_iteration": len(history),
                    "em_next_iteration": next_iteration,
                    "em_complete": bool(complete),
                    "em_history": list(current_history),
                    "alignment_history": list(history),
                    "previous_objective": previous_objective,
                    "outer_converged": last_outer_converged,
                }, parameters, local_paths)

        if em_complete:
            model.history = em_history
        else:
            model.fit(
                final_aligned, max_iterations=int(dpcca_config.get("max_em_iterations", 20)),
                tolerance=float(dpcca_config.get("tolerance", 1e-4)), warm_start=True,
                start_iteration=em_start, initial_history=em_history,
                checkpoint_callback=save_final_em,
            )
        if checkpoint_callback is not None:
            checkpoint_callback({
                "phase": "final_complete",
                "outer_iteration": len(history),
                "em_next_iteration": len(model.history),
                "em_complete": True,
                "em_history": list(model.history),
                "alignment_history": list(history),
                "previous_objective": previous_objective,
                "outer_converged": last_outer_converged,
            }, model.parameters, local_paths)
    global_paths = [np.column_stack([indices[path[:, 0]], indices[path[:, 1]]])
                    for (_, _, indices), path in zip(compact, local_paths)]
    return model, AlignmentResult(
        global_paths, history,
        "CTW-style CCA observation projections V_i^T X_i on synchronized native observations",
    )
