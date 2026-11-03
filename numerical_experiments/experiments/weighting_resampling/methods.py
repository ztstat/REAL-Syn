"""Equal-budget resampling and subset-SNIS methods."""

from __future__ import annotations
import hashlib
from typing import Any
import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("ESS requires finite nonnegative one-dimensional weights.")
    total = float(np.sum(values))
    square_sum = float(np.sum(values**2))
    if total <= 0.0 or square_sum <= 0.0:
        raise ValueError("ESS requires positive total mass.")
    return total**2 / square_sum


def normalize_probabilities(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("Probabilities require finite nonnegative weights.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("Probability mass must be positive.")
    probabilities = values / total
    probabilities = probabilities / float(np.sum(probabilities))
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise RuntimeError("Probability normalization failed.")
    return probabilities


def _uniforms(rng: Any, size: int) -> np.ndarray:
    if hasattr(rng, "random_sample"):
        return np.asarray(rng.random_sample(size), dtype=float)
    return np.asarray(rng.random(size), dtype=float)


def indices_fingerprint(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def multiplicity_summary(multiplicities: np.ndarray, budget: int) -> dict[str, Any]:
    counts = np.asarray(multiplicities)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("Multiplicities must be a one-dimensional integer array.")
    if np.any(counts < 0) or int(np.sum(counts)) != int(budget):
        raise ValueError("Multiplicities must be nonnegative and sum to B.")
    indices = np.flatnonzero(counts)
    if indices.size == 0:
        raise RuntimeError("Resampling produced no selected candidate.")
    nonzero = counts[indices].astype(float)
    return {
        "multiplicities": counts.astype(np.int64, copy=False),
        "indices": indices,
        "weights": nonzero,
        "draws": np.repeat(indices, counts[indices]),
        "draw_count": int(budget),
        "unique_count": int(indices.size),
        "maximum_multiplicity": int(np.max(counts)),
        "multiplicity_ess": effective_sample_size(nonzero),
    }


def residual_resample(
    normalized_weights: np.ndarray,
    budget: int,
    rng: Any,
) -> dict[str, Any]:
    probabilities = normalize_probabilities(normalized_weights)
    if budget <= 0:
        raise ValueError("B must be positive.")
    expected = budget * probabilities
    deterministic = np.floor(expected).astype(np.int64)
    residual_draws = int(budget - np.sum(deterministic))
    residual_mass = expected - deterministic
    counts = deterministic.copy()
    if residual_draws > 0:
        total_residual = float(np.sum(residual_mass))
        if total_residual <= 0.0:
            raise RuntimeError("Positive residual draw count has no residual mass.")
        residual_probabilities = residual_mass / total_residual
        draws = rng.choice(
            probabilities.size,
            size=residual_draws,
            replace=True,
            p=residual_probabilities,
        )
        counts += np.bincount(draws, minlength=probabilities.size)
    result = multiplicity_summary(counts, budget)
    result.update(
        {
            "deterministic_draw_count": int(np.sum(deterministic)),
            "residual_draw_count": residual_draws,
        }
    )
    return result


def stratified_resample(
    normalized_weights: np.ndarray,
    budget: int,
    rng: Any,
) -> dict[str, Any]:
    probabilities = normalize_probabilities(normalized_weights)
    if budget <= 0:
        raise ValueError("B must be positive.")
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    positions = (np.arange(budget, dtype=float) + _uniforms(rng, budget)) / budget
    draws = np.searchsorted(cdf, positions, side="right")
    draws = np.minimum(draws, probabilities.size - 1)
    if np.any(draws < 0) or np.any(draws >= probabilities.size):
        raise RuntimeError("Stratified resampling generated an invalid index.")
    counts = np.bincount(draws, minlength=probabilities.size)
    result = multiplicity_summary(counts, budget)
    result["draws"] = draws.astype(np.int64, copy=False)
    return result


def uniform_subset(
    candidate_count: int,
    budget: int,
    rng: Any,
) -> np.ndarray:
    if candidate_count <= 0 or budget <= 0 or budget > candidate_count:
        raise ValueError("Subset sizes must satisfy 0 < B <= M.")
    indices = np.asarray(
        rng.choice(candidate_count, size=budget, replace=False), dtype=np.int64
    )
    indices.sort()
    if np.unique(indices).size != budget:
        raise RuntimeError("Uniform subset does not contain B distinct indices.")
    return indices


def subset_ratio_diagnostics(ratios: np.ndarray) -> dict[str, float]:
    values = np.asarray(ratios, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("Subset ratios must be finite and strictly positive.")
    return {
        "subset_ratio_mean": float(np.mean(values)),
        "subset_ratio_maximum": float(np.max(values)),
        "subset_ratio_p99": float(np.quantile(values, 0.99)),
        "subset_ratio_p99p9": float(np.quantile(values, 0.999)),
        "subset_ratio_ess": effective_sample_size(values),
    }


def subsampled_snis(
    full_ratios: np.ndarray,
    subset_indices: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    ratios = np.asarray(full_ratios, dtype=float)
    indices = np.asarray(subset_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size != budget:
        raise ValueError("Subsampled-SNIS requires exactly B indices.")
    if np.unique(indices).size != budget:
        raise ValueError("Subsampled-SNIS indices must be distinct.")
    subset = ratios[indices]
    diagnostics = subset_ratio_diagnostics(subset)
    weights = budget * subset / float(np.sum(subset))
    if not np.isclose(float(np.sum(weights)), budget, rtol=0.0, atol=1e-10):
        raise RuntimeError("Subsampled-SNIS total weight is not B.")
    return {
        "indices": indices.copy(),
        "weights": weights,
        "raw_ratios": subset,
        "clipped_ratios": subset.copy(),
        "clipping_threshold": None,
        "clipped_count": 0,
        "clipped_fraction": 0.0,
        "raw_ess": diagnostics["subset_ratio_ess"],
        "clipped_ess": diagnostics["subset_ratio_ess"],
        "final_weight_ess": effective_sample_size(weights),
        "subset_fingerprint": indices_fingerprint(indices),
        **diagnostics,
        "maximum_clipped_ratio": float(np.max(subset)),
    }


def clipped_snis(
    full_ratios: np.ndarray,
    subset_indices: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    ratios = np.asarray(full_ratios, dtype=float)
    indices = np.asarray(subset_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size != budget:
        raise ValueError("Clipped-SNIS requires exactly B indices.")
    if np.unique(indices).size != budget:
        raise ValueError("Clipped-SNIS indices must be distinct.")
    subset = ratios[indices]
    diagnostics = subset_ratio_diagnostics(subset)
    threshold = float(np.mean(subset) * np.sqrt(budget))
    clipped = np.minimum(subset, threshold)
    weights = budget * clipped / float(np.sum(clipped))
    clipped_count = int(np.count_nonzero(subset > threshold))
    raw_ess = diagnostics["subset_ratio_ess"]
    clipped_ess = effective_sample_size(clipped)
    if clipped_ess + 1e-10 < raw_ess:
        raise RuntimeError("Clipping unexpectedly reduced ESS.")
    if not np.isclose(float(np.sum(weights)), budget, rtol=0.0, atol=1e-10):
        raise RuntimeError("Clipped-SNIS total weight is not B.")
    return {
        "indices": indices.copy(),
        "weights": weights,
        "raw_ratios": subset,
        "clipped_ratios": clipped,
        "clipping_threshold": threshold,
        "clipped_count": clipped_count,
        "clipped_fraction": clipped_count / budget,
        "raw_ess": raw_ess,
        "clipped_ess": clipped_ess,
        "final_weight_ess": effective_sample_size(weights),
        "subset_fingerprint": indices_fingerprint(indices),
        **diagnostics,
        "maximum_clipped_ratio": float(np.max(clipped)),
    }
