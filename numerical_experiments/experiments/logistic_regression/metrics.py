"""Metrics and finite-sample diagnostics for the classification extension."""


from __future__ import annotations
import math
from typing import Any
import numpy as np


METHOD_ORDER = [
    "Real-only",
    "Uniform acceptance",
    "Density-ratio-only acceptance",
    "Influence-only acceptance",
    "REAL-Syn",
]


METHOD_KEYS = {
    "Real-only": "real_only",
    "Uniform acceptance": "uniform",
    "Density-ratio-only acceptance": "density_ratio_only",
    "Influence-only acceptance": "influence_only",
    "REAL-Syn": "real_syn",
}


def effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("ESS requires a nonempty one-dimensional vector.")
    denominator = float(np.sum(weights**2))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("ESS denominator must be finite and positive.")
    ess = float(np.sum(weights) ** 2 / denominator)
    return ess, float(ess / weights.size)


def ratio_summary(ratio: np.ndarray) -> dict[str, float]:
    ratio = np.asarray(ratio, dtype=float)
    ess, relative = effective_sample_size(ratio)
    return {
        "ratio_minimum": float(np.min(ratio)),
        "ratio_median": float(np.median(ratio)),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_p90": float(np.quantile(ratio, 0.90)),
        "ratio_p95": float(np.quantile(ratio, 0.95)),
        "ratio_p99": float(np.quantile(ratio, 0.99)),
        "ratio_maximum": float(np.max(ratio)),
        "candidate_ess": ess,
        "candidate_relative_ess": relative,
    }


def final_weight_summary(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    ess, relative = effective_sample_size(weights)
    return {
        "final_weight_ess": ess,
        "final_weight_relative_ess": relative,
        "final_weight_p95": float(np.quantile(weights, 0.95)),
        "final_weight_p99": float(np.quantile(weights, 0.99)),
        "final_weight_maximum": float(np.max(weights)),
    }


def binary_auroc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes.")
    order = np.argsort(probabilities, kind="mergesort")
    sorted_scores = probabilities[order]
    ranks = np.empty(probabilities.size, dtype=float)
    start = 0
    while start < probabilities.size:
        end = start + 1
        while end < probabilities.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum_positive = float(np.sum(ranks[y_true == 1]))
    return (
        rank_sum_positive - positives * (positives + 1) / 2.0
    ) / float(positives * negatives)


def binary_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Average precision, grouping equal-score thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        raise ValueError("AUPRC requires at least one positive.")
    order = np.argsort(-probabilities, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = probabilities[order]
    cumulative_tp = 0
    cumulative_count = 0
    previous_recall = 0.0
    average_precision = 0.0
    start = 0
    while start < sorted_y.size:
        end = start + 1
        while end < sorted_y.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        cumulative_tp += int(np.sum(sorted_y[start:end] == 1))
        cumulative_count += end - start
        recall = cumulative_tp / float(positives)
        precision = cumulative_tp / float(cumulative_count)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    return float(average_precision)


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    true_probabilities: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    true_probabilities = np.asarray(true_probabilities, dtype=float)
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    log_loss = float(
        np.mean(-(y_true * np.log(clipped) + (1.0 - y_true) * np.log1p(-clipped)))
    )
    return {
        "test_log_loss": log_loss,
        "brier_score": float(np.mean((probabilities - y_true) ** 2)),
        "probability_mse": float(np.mean((probabilities - true_probabilities) ** 2)),
        "auroc": binary_auroc(y_true.astype(int), probabilities),
        "auprc": binary_auprc(y_true.astype(int), probabilities),
        "classification_error": float(
            np.mean((probabilities >= 0.5).astype(int) != y_true.astype(int))
        ),
        "fitted_probability_below_1e-6_fraction": float(
            np.mean(probabilities < 1e-6)
        ),
        "fitted_probability_above_1_minus_1e-6_fraction": float(
            np.mean(probabilities > 1.0 - 1e-6)
        ),
    }


def parameter_squared_error(theta: np.ndarray, beta_true: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)
    return float(np.sum((theta[1:] - beta_true) ** 2))


def mean_sd_se_ci(values: list[float]) -> dict[str, float | int | None]:
    values_array = np.asarray(values, dtype=float)
    mean = float(np.mean(values_array))
    if values_array.size < 2:
        return {
            "n": int(values_array.size),
            "mean": mean,
            "sd": None,
            "se": None,
            "ci_low": None,
            "ci_high": None,
        }
    sd = float(np.std(values_array, ddof=1))
    se = sd / math.sqrt(values_array.size)
    return {
        "n": int(values_array.size),
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - 1.959963984540054 * se,
        "ci_high": mean + 1.959963984540054 * se,
    }


def coefficient_summary(
    coefficient_records: list[dict[str, Any]],
    beta_true: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Return complete-parameter covariance and slope recovery summaries."""
    output: dict[str, dict[str, Any]] = {}
    beta_true = np.asarray(beta_true, dtype=float)
    for method in METHOD_ORDER:
        matrix = np.asarray(
            [
                row["theta"]
                for row in coefficient_records
                if row["method"] == method
            ],
            dtype=float,
        )
        if matrix.shape[0] < 2:
            covariance_trace = None
        else:
            covariance_trace = float(np.trace(np.cov(matrix, rowvar=False, ddof=1)))
        slope_matrix = matrix[:, 1:]
        output[method] = {
            "coefficient_covariance_trace": covariance_trace,
            "coefficient_covariance_includes_intercept": True,
            "parameter_mse": float(
                np.mean(np.sum((slope_matrix - beta_true[None, :]) ** 2, axis=1))
            ),
            "squared_bias": float(
                np.sum((np.mean(slope_matrix, axis=0) - beta_true) ** 2)
            ),
            "mean_complete_parameter": np.mean(matrix, axis=0).tolist(),
        }
    return output
