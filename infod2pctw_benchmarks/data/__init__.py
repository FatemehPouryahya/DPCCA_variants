"""Data adapters for the benchmark suite."""

from .agneuro_adapter import BenchmarkDataset, PreprocessingStats, load_agneuro

__all__ = ["BenchmarkDataset", "PreprocessingStats", "load_agneuro"]
