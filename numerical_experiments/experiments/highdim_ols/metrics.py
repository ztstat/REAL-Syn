"""Small reporting helpers for the higher-dimensional simulation extension."""


from __future__ import annotations
import csv
import json
import math
from typing import Iterable
import numpy as np


METHOD_ORDER = [
    "real only",
    "uniform",
    "density ratio only",
    "influence only",
    "Real-Syn",
]


def effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    """Return ESS and relative ESS for a finite one-dimensional weight vector."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("ESS weights must be one-dimensional.")
    if weights.size == 0:
        return float("nan"), float("nan")

    denominator = float(np.sum(weights ** 2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan"), float("nan")

    ess = float(np.sum(weights) ** 2 / denominator)
    return ess, float(ess / weights.size)


def saturation_rate(pi: np.ndarray, tolerance: float = 1e-12) -> float:
    """Fraction of candidate acceptance probabilities clipped at one."""
    pi = np.asarray(pi, dtype=float)
    if pi.size == 0:
        return float("nan")
    return float(np.mean(pi >= 1.0 - tolerance))


def ratio_summary(ratio: np.ndarray) -> dict:
    """Return requested finite-candidate density-ratio diagnostics."""
    ratio = np.asarray(ratio, dtype=float)
    ess, relative_ess = effective_sample_size(ratio)
    return {
        "ratio_minimum": float(np.min(ratio)),
        "ratio_median": float(np.median(ratio)),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_p90": float(np.quantile(ratio, 0.90)),
        "ratio_p95": float(np.quantile(ratio, 0.95)),
        "ratio_p99": float(np.quantile(ratio, 0.99)),
        "ratio_maximum": float(np.max(ratio)),
        "candidate_ess": ess,
        "candidate_relative_ess": relative_ess,
    }


def final_weight_summary(weights: np.ndarray) -> dict:
    """Return upper-tail diagnostics for actually accepted final weights."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("Final weights must be one-dimensional.")
    if weights.size == 0:
        return {
            "final_weight_p95": None,
            "final_weight_p99": None,
            "final_weight_maximum": None,
        }
    return {
        "final_weight_p95": float(np.quantile(weights, 0.95)),
        "final_weight_p99": float(np.quantile(weights, 0.99)),
        "final_weight_maximum": float(np.max(weights)),
    }


def vp_summary(vp: np.ndarray, ridge: float) -> dict:
    """Return raw and ridge-stabilized Vp spectral diagnostics."""
    vp = np.asarray(vp, dtype=float)
    symmetric_vp = 0.5 * (vp + vp.T)
    raw_eigenvalues = np.linalg.eigvalsh(symmetric_vp)
    stabilized = symmetric_vp + float(ridge) * np.eye(vp.shape[0])
    stabilized_eigenvalues = np.linalg.eigvalsh(stabilized)
    return {
        "vp_condition_number": float(np.linalg.cond(symmetric_vp)),
        "vp_minimum_eigenvalue": float(np.min(raw_eigenvalues)),
        "vp_maximum_eigenvalue": float(np.max(raw_eigenvalues)),
        "vp_stabilized_condition_number": float(np.linalg.cond(stabilized)),
        "vp_stabilized_minimum_eigenvalue": float(np.min(stabilized_eigenvalues)),
        "vp_stabilized_maximum_eigenvalue": float(np.max(stabilized_eigenvalues)),
    }


def score_diagnostics(influence: np.ndarray, acceptance_score: np.ndarray) -> dict:
    """Summarize standardized-score norms h and REAL-Syn scores r*h."""
    influence = np.asarray(influence, dtype=float)
    acceptance_score = np.asarray(acceptance_score, dtype=float)
    if influence.ndim != 1 or acceptance_score.ndim != 1:
        raise ValueError("Score diagnostics require one-dimensional arrays.")
    if influence.shape != acceptance_score.shape or influence.size == 0:
        raise ValueError("Influence and acceptance scores must have equal nonzero size.")

    h_finite_mask = np.isfinite(influence)
    score_finite_mask = np.isfinite(acceptance_score)
    h_finite = influence[h_finite_mask]
    score_finite = acceptance_score[score_finite_mask]
    if h_finite.size == 0 or score_finite.size == 0:
        raise ValueError("Score diagnostics contain no finite values.")

    h_mean = float(np.mean(h_finite))
    h_cv = (
        float(np.std(h_finite, ddof=0) / h_mean)
        if h_mean != 0.0
        else None
    )
    return {
        "h_median": float(np.median(h_finite)),
        "h_p95": float(np.quantile(h_finite, 0.95)),
        "h_p99": float(np.quantile(h_finite, 0.99)),
        "h_maximum": float(np.max(h_finite)),
        "h_coefficient_of_variation": h_cv,
        "h_nonfinite_fraction": float(1.0 - np.mean(h_finite_mask)),
        "real_syn_score_median": float(np.median(score_finite)),
        "real_syn_score_p95": float(np.quantile(score_finite, 0.95)),
        "real_syn_score_p99": float(np.quantile(score_finite, 0.99)),
        "real_syn_score_maximum": float(np.max(score_finite)),
    }


def count_nonfinite(arrays: Iterable[np.ndarray]) -> tuple[int, int]:
    """Count NaN and Inf entries across a collection of arrays."""
    nan_count = 0
    inf_count = 0
    for array in arrays:
        values = np.asarray(array, dtype=float)
        nan_count += int(np.count_nonzero(np.isnan(values)))
        inf_count += int(np.count_nonzero(np.isinf(values)))
    return nan_count, inf_count


def parameter_errors(
    beta_full: np.ndarray,
    target_beta: np.ndarray,
    experiment_arm: str,
) -> dict:
    """Return per-repetition slope-recovery errors."""
    beta_full = np.asarray(beta_full, dtype=float)
    target_beta = np.asarray(target_beta, dtype=float)
    slopes = beta_full[1:]
    if slopes.shape != target_beta.shape:
        raise ValueError(
            f"Slope vector shape {slopes.shape} does not match target {target_beta.shape}."
        )

    squared_errors = (slopes - target_beta) ** 2
    output = {
        "parameter_squared_error": float(np.sum(squared_errors)),
        "active_error": None,
        "nuisance_error": None,
    }
    if experiment_arm == "primary_sparse":
        output["active_error"] = float(np.sum(squared_errors[:5]))
        if target_beta.size > 5:
            output["nuisance_error"] = float(np.sum(slopes[5:] ** 2))
    return output


def aggregate_parameter_summary(
    coefficient_records: list[dict],
    target_beta: np.ndarray,
    experiment_arm: str,
) -> dict:
    """Aggregate SquaredBias and ParameterMSE for every method."""
    target_beta = np.asarray(target_beta, dtype=float)
    output = {}
    for method in METHOD_ORDER:
        method_records = [
            record for record in coefficient_records if record["method"] == method
        ]
        slope_matrix = np.asarray(
            [record["beta_full"][1:] for record in method_records],
            dtype=float,
        )
        mean_slopes = np.mean(slope_matrix, axis=0)
        squared_bias = float(np.sum((mean_slopes - target_beta) ** 2))
        parameter_mse = float(
            np.mean(np.sum((slope_matrix - target_beta.reshape(1, -1)) ** 2, axis=1))
        )
        method_output = {
            "squared_bias": squared_bias,
            "parameter_mse": parameter_mse,
            "mean_slope_estimate": mean_slopes.tolist(),
            "active_error_mean": None,
            "nuisance_error_mean": None,
        }
        if experiment_arm == "primary_sparse":
            method_output["active_error_mean"] = float(
                np.mean(
                    np.sum(
                        (slope_matrix[:, :5] - target_beta[:5].reshape(1, -1)) ** 2,
                        axis=1,
                    )
                )
            )
            if target_beta.size > 5:
                method_output["nuisance_error_mean"] = float(
                    np.mean(np.sum(slope_matrix[:, 5:] ** 2, axis=1))
                )
        output[method] = method_output
    return output


def mean_se(values: list[float]) -> tuple[float, float | None]:
    """Return mean and the ordinary ddof=1 Monte Carlo standard error."""
    values_array = np.asarray(values, dtype=float)
    mean_value = float(np.mean(values_array))
    if values_array.size < 2:
        return mean_value, None
    se_value = float(np.std(values_array, ddof=1) / math.sqrt(values_array.size))
    return mean_value, se_value


def write_coefficient_estimates(path: str, records: list[dict], dimension: int) -> None:
    """Write intercept and every slope for every repetition and method."""
    fieldnames = ["rep", "method", "intercept"] + [
        f"beta_{index + 1}" for index in range(dimension)
    ]
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            beta_full = np.asarray(record["beta_full"], dtype=float)
            row = {
                "rep": int(record["rep"]),
                "method": record["method"],
                "intercept": float(beta_full[0]),
            }
            for index, coefficient in enumerate(beta_full[1:]):
                row[f"beta_{index + 1}"] = float(coefficient)
            writer.writerow(row)


def write_dict_rows(path: str, rows: list[dict]) -> None:
    """Write a homogeneous list of dictionaries as CSV."""
    if not rows:
        raise ValueError("Cannot write an empty row collection.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, obj: dict) -> None:
    """Save JSON with strict rejection of NaN/Inf."""
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(obj, file_obj, indent=2, allow_nan=False)
