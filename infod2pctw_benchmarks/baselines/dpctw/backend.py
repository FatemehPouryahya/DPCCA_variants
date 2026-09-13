"""Low-level NumPy or PyTorch/CUDA numerical operations for DPCTW."""

from __future__ import annotations

from typing import Any

import numpy as np


class NumericalBackend:
    """Small array/linalg adapter; DPCTW algorithmic control flow lives elsewhere."""

    def __init__(self, device: str = "cpu", _torch_device: str | None = None):
        self.device = str(device).lower()
        if self.device not in ("cpu", "cuda"):
            raise ValueError("DPCTW device must be 'cpu' or 'cuda'")
        self.torch = None
        self.torch_device = None
        if self.device == "cuda":
            import torch
            self.torch_device = _torch_device or "cuda"
            if self.torch_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("DPCTW device is 'cuda', but PyTorch cannot access a CUDA GPU")
            self.torch = torch

    def asarray(self, value: Any):
        if self.device == "cpu":
            return np.asarray(value, dtype=np.float64)
        return self.torch.as_tensor(value, dtype=self.torch.float64, device=self.torch_device)

    def as_bool(self, value: Any):
        if self.device == "cpu":
            return np.asarray(value, dtype=bool)
        return self.torch.as_tensor(value, dtype=self.torch.bool, device=self.torch_device)

    def to_numpy(self, value: Any) -> np.ndarray:
        if self.device == "cpu":
            return np.asarray(value)
        return value.detach().cpu().numpy()

    def scalar(self, value: Any) -> float:
        return float(value if self.device == "cpu" else value.item())

    def size(self, value: Any) -> int:
        return int(value.size if self.device == "cpu" else value.numel())

    def zeros(self, shape: tuple[int, ...]):
        if self.device == "cpu":
            return np.zeros(shape)
        return self.torch.zeros(shape, dtype=self.torch.float64, device=self.torch_device)

    def empty(self, shape: tuple[int, ...]):
        if self.device == "cpu":
            return np.empty(shape)
        return self.torch.empty(shape, dtype=self.torch.float64, device=self.torch_device)

    def eye(self, dimension: int):
        if self.device == "cpu":
            return np.eye(dimension)
        return self.torch.eye(dimension, dtype=self.torch.float64, device=self.torch_device)

    def full(self, shape: tuple[int, ...], fill_value: float):
        if self.device == "cpu":
            return np.full(shape, fill_value)
        return self.torch.full(shape, fill_value, dtype=self.torch.float64, device=self.torch_device)

    def empty_like(self, value: Any):
        return np.empty_like(value) if self.device == "cpu" else self.torch.empty_like(value)

    def zeros_like(self, value: Any):
        return np.zeros_like(value) if self.device == "cpu" else self.torch.zeros_like(value)

    def ones_like_bool(self, value: Any):
        if self.device == "cpu":
            return np.ones_like(value, dtype=bool)
        return self.torch.ones_like(value, dtype=self.torch.bool)

    def copy(self, value: Any):
        return value.copy() if self.device == "cpu" else value.clone()

    def concatenate(self, values: list[Any], axis: int = 0):
        if self.device == "cpu":
            return np.concatenate(values, axis=axis)
        return self.torch.cat(values, dim=axis)

    def stack(self, values: list[Any], axis: int = 0):
        if self.device == "cpu":
            return np.stack(values, axis=axis)
        return self.torch.stack(values, dim=axis)

    def broadcast_to(self, value: Any, shape: tuple[int, ...]):
        if self.device == "cpu":
            return np.broadcast_to(value, shape)
        return self.torch.broadcast_to(value, shape)

    def arange(self, start: int, stop: int | None = None):
        if stop is None:
            start, stop = 0, start
        if self.device == "cpu":
            return np.arange(start, stop)
        return self.torch.arange(start, stop, device=self.torch_device)

    def finite(self, value: Any):
        return np.isfinite(value) if self.device == "cpu" else self.torch.isfinite(value)

    def any(self, value: Any) -> bool:
        return bool(np.any(value)) if self.device == "cpu" else bool(self.torch.any(value).item())

    def maximum(self, value: Any, floor: float):
        if self.device == "cpu":
            return np.maximum(value, floor)
        return self.torch.clamp(value, min=floor)

    def sum(self, value: Any, axis: int | None = None):
        if self.device == "cpu":
            return np.sum(value, axis=axis)
        return self.torch.sum(value, dim=axis)

    def mean(self, value: Any, axis: int | None = None, keepdims: bool = False):
        if self.device == "cpu":
            return np.mean(value, axis=axis, keepdims=keepdims)
        return self.torch.mean(value, dim=axis, keepdim=keepdims)

    def log(self, value: Any):
        return np.log(value) if self.device == "cpu" else self.torch.log(value)

    def sqrt(self, value: Any):
        return np.sqrt(value) if self.device == "cpu" else self.torch.sqrt(value)

    def einsum(self, equation: str, *values: Any):
        if self.device == "cpu":
            return np.einsum(equation, *values)
        return self.torch.einsum(equation, *values)

    def outer(self, left: Any, right: Any):
        return np.outer(left, right) if self.device == "cpu" else self.torch.outer(left, right)

    def norm(self, value: Any) -> float:
        if self.device == "cpu":
            return float(np.linalg.norm(value))
        return float(self.torch.linalg.vector_norm(value).item())

    def eigh(self, value: Any):
        return np.linalg.eigh(value) if self.device == "cpu" else self.torch.linalg.eigh(value)

    def svd(self, value: Any):
        if self.device == "cpu":
            return np.linalg.svd(value, full_matrices=False)
        return self.torch.linalg.svd(value, full_matrices=False)

    def inv(self, value: Any):
        return np.linalg.inv(value) if self.device == "cpu" else self.torch.linalg.inv(value)

    def pinv(self, value: Any):
        return np.linalg.pinv(value) if self.device == "cpu" else self.torch.linalg.pinv(value)

    def solve(self, left: Any, right: Any):
        return np.linalg.solve(left, right) if self.device == "cpu" else self.torch.linalg.solve(left, right)

    def slogdet(self, value: Any):
        return np.linalg.slogdet(value) if self.device == "cpu" else self.torch.linalg.slogdet(value)


def backend_for(device: str = "cpu") -> NumericalBackend:
    return NumericalBackend(device)


def backend_from_array(value: Any) -> NumericalBackend:
    module = type(value).__module__.split(".", 1)[0]
    if module == "torch":
        return NumericalBackend("cuda", _torch_device=str(value.device))
    return NumericalBackend("cpu")
