"""Deterministic bootstrap summaries used by causal-intervention plots."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sample mean and percentile bootstrap interval along axis 0."""
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("values must contain at least one sample")
    rng = np.random.default_rng(seed)
    n_samples = values.shape[0]
    boot = np.empty((n_resamples, *values.shape[1:]), dtype=np.float32)
    chunk_size = 100
    for start in range(0, n_resamples, chunk_size):
        end = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, n_samples, size=(end - start, n_samples))
        boot[start:end] = values[indices].mean(axis=1, dtype=np.float32)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [tail, 1.0 - tail], axis=0)
    return values.mean(axis=0), low, high
