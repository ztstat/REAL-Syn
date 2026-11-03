"""Aggregation for weighting and resampling comparisons."""


from __future__ import annotations
from typing import Any
import numpy as np
from scipy.stats import t as student_t
from . import core


CONTRASTS = (
    ("REAL-Syn", "Uniform"),
    ("REAL-Syn", "DRE-only"),
    ("REAL-Syn", "Influence-only"),
    ("REAL-Syn", "Multinomial-SIR-600"),
    ("REAL-Syn", "Residual-SIR-600"),
    ("REAL-Syn", "Stratified-SIR-600"),
    ("REAL-Syn", "Subsampled-SNIS-600"),
    ("REAL-Syn", "Clipped-SNIS-600"),
    ("REAL-Syn", "Real-only"),
    ("Multinomial-SIR-600", "Residual-SIR-600"),
    ("Multinomial-SIR-600", "Stratified-SIR-600"),
    ("Subsampled-SNIS-600", "Clipped-SNIS-600"),
)


def _mean_optional(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    return None if not values else float(np.mean(values))


def _se(values: np.ndarray) -> float:
    if values.size < 2:
        raise ValueError("Monte Carlo SE requires at least two repetitions.")
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def aggregate(
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    run_label: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    truth = core.importance.true_coefficients(source, "gaussian_linear")
    summaries: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    for method in core.METHODS:
        group = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["rep"]),
        )
        if len(group) < 2:
            raise ValueError(f"{method} aggregation needs at least two rows.")
        vectors = np.asarray(
            [
                [row[f"theta_{index}"] for index in range(6)]
                for row in group
            ],
            dtype=float,
        )
        covariance = np.cov(vectors, rowvar=False, ddof=1)
        mean_vector = np.mean(vectors, axis=0)
        squared_bias = float(
            (mean_vector - truth) @ (mean_vector - truth)
        )
        trace = float(np.trace(covariance))
        finite_variance = (len(group) - 1.0) / len(group) * trace
        test_mse = np.asarray(
            [row["test_mse"] for row in group], dtype=float
        )
        noiseless = np.asarray(
            [row["noiseless_prediction_mse"] for row in group], dtype=float
        )
        parameter_mse = np.asarray(
            [row["parameter_squared_error"] for row in group], dtype=float
        )
        summaries.append(
            {
                "run_label": run_label,
                "experiment_id": "weighting_resampling",
                "method": method,
                "repetitions": len(group),
                "mean_test_mse": float(np.mean(test_mse)),
                "mc_se_test_mse": _se(test_mse),
                "mean_noiseless_test_mse": float(np.mean(noiseless)),
                "mc_se_noiseless_test_mse": _se(noiseless),
                "coefficient_covariance_trace": trace,
                "slope_covariance_trace": float(
                    np.trace(covariance[1:, 1:])
                ),
                "intercept_variance": float(covariance[0, 0]),
                "mean_parameter_mse": float(np.mean(parameter_mse)),
                "parameter_squared_bias": squared_bias,
                "finite_sample_variance_component": finite_variance,
                "variance_plus_squared_bias": finite_variance + squared_bias,
                "decomposition_absolute_error": abs(
                    float(np.mean(parameter_mse))
                    - (finite_variance + squared_bias)
                ),
                "mean_downstream_draws": _mean_optional(
                    group, "downstream_draws"
                ),
                "mean_unique_rows": _mean_optional(
                    group, "downstream_unique_synthetic_rows"
                ),
                "mean_synthetic_total_weight": _mean_optional(
                    group, "synthetic_total_weight"
                ),
                "mean_final_weight_or_multiplicity_ess": _mean_optional(
                    group, "final_weight_or_multiplicity_ess"
                ),
                "mean_selection_or_resampling_seconds": _mean_optional(
                    group, "selection_or_resampling_seconds"
                ),
                "mean_downstream_fitting_seconds": _mean_optional(
                    group, "downstream_fitting_seconds"
                ),
                "mean_total_runtime_seconds": _mean_optional(
                    group, "total_runtime_seconds"
                ),
                "clipping_activation_rate": _mean_optional(
                    [
                        {
                            "clipping_activated": int(
                                bool(row["clipping_activated"])
                            )
                        }
                        for row in group
                        if row["clipping_activated"] is not None
                    ],
                    "clipping_activated",
                ),
                "finite_value_failure_count": int(
                    sum(int(row["finite_value_failure"]) for row in group)
                ),
                "solver_failure_count": int(
                    sum(int(row["solver_failure"]) for row in group)
                ),
                "warning_count": int(
                    sum(int(row["warning_flag"]) for row in group)
                ),
            }
        )
        for row_index in range(6):
            for column_index in range(6):
                covariance_rows.append(
                    {
                        "run_label": run_label,
                        "experiment_id": "weighting_resampling",
                        "method": method,
                        "row": row_index,
                        "column": column_index,
                        "covariance": float(
                            covariance[row_index, column_index]
                        ),
                    }
                )

    lookup = {
        (int(row["rep"]), str(row["method"])): row for row in rows
    }
    repetitions = sorted({int(row["rep"]) for row in rows})
    contrasts: list[dict[str, Any]] = []
    for first, second in CONTRASTS:
        differences = np.asarray(
            [
                float(lookup[(rep, first)]["test_mse"])
                - float(lookup[(rep, second)]["test_mse"])
                for rep in repetitions
            ],
            dtype=float,
        )
        mean = float(np.mean(differences))
        se = _se(differences)
        critical = float(student_t.ppf(0.975, differences.size - 1))
        contrasts.append(
            {
                "run_label": run_label,
                "experiment_id": "weighting_resampling",
                "first_method": first,
                "second_method": second,
                "contrast": f"{first} minus {second}",
                "metric": "test_mse",
                "paired_repetitions": differences.size,
                "mean_difference": mean,
                "mc_se_difference": se,
                "t_critical_975": critical,
                "ci95_lower": mean - critical * se,
                "ci95_upper": mean + critical * se,
                "median_difference": float(np.median(differences)),
                "fraction_first_lower": float(
                    np.mean(differences < 0.0)
                ),
            }
        )
    return summaries, covariance_rows, contrasts
