"""Deterministic coefficient constructions for dimension-extension experiments."""

from __future__ import annotations
import numpy as np


BASE_SPARSE_BETA = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=float)
TARGET_SQUARED_NORM = 2.2


def make_sparse_beta(d: int) -> np.ndarray:
    """Return (1, .8, .6, .4, .2, 0, ..., 0) with exactly ``d`` entries."""
    if d < BASE_SPARSE_BETA.size:
        raise ValueError(f"Sparse construction requires d >= {BASE_SPARSE_BETA.size}.")

    beta = np.zeros(int(d), dtype=float)
    beta[: BASE_SPARSE_BETA.size] = BASE_SPARSE_BETA
    return beta


def make_dense_beta(d: int, squared_norm: float = TARGET_SQUARED_NORM) -> np.ndarray:
    """Return the norm-matched dense vector sqrt(squared_norm / d) * 1_d."""
    if d <= 0:
        raise ValueError("d must be positive.")
    if squared_norm <= 0.0:
        raise ValueError("squared_norm must be positive.")

    coefficient = np.sqrt(float(squared_norm) / float(d))
    return np.full(int(d), coefficient, dtype=float)


def make_proposal_beta(target_beta: np.ndarray) -> np.ndarray:
    """Return the paper's proposal coefficient vector, 0.8 * target beta."""
    target_beta = np.asarray(target_beta, dtype=float)
    if target_beta.ndim != 1:
        raise ValueError("target_beta must be one-dimensional.")
    return 0.8 * target_beta
