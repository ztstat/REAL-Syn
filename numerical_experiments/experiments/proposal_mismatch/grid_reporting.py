"""Aggregation for the local proposal grid."""


from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import t
from . import grid_core as core


TRUE_PARAMETERS = np.asarray([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])


WITHIN_CELL_COMPARISONS = (
    ("REAL-Syn", "DRE-only", "REAL-Syn_minus_DRE-only"),
    ("REAL-Syn", "SIR", "REAL-Syn_minus_SIR"),
    ("REAL-Syn", "Uniform", "REAL-Syn_minus_Uniform"),
    (
        "REAL-Syn",
        "Influence-only",
        "REAL-Syn_minus_Influence-only",
    ),
    ("REAL-Syn", "Real-only", "REAL-Syn_minus_Real-only"),
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "N/A" if row.get(key) is None else row.get(key)
                    for key in fieldnames
                }
            )


def mean_se(values: list[float] | np.ndarray) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    return (
        float(np.mean(vector)),
        (
            float(np.std(vector, ddof=1) / np.sqrt(vector.size))
            if vector.size > 1
            else 0.0
        ),
    )


def paired_summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=float)
    mean, se = mean_se(vector)
    critical = (
        float(t.ppf(0.975, df=vector.size - 1))
        if vector.size > 1
        else 0.0
    )
    return {
        "n_pairs": int(vector.size),
        "mean_difference": mean,
        "standard_error": se,
        "t95_lower": mean - critical * se,
        "t95_upper": mean + critical * se,
        "median_difference": float(np.median(vector)),
        "fraction_first_lower": float(np.mean(vector < 0.0)),
    }


def coefficient_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[f"theta_{index}"]) for index in range(6)])


def covariance(values: np.ndarray) -> np.ndarray:
    vectors = np.asarray(values, dtype=float)
    if vectors.shape[0] <= 1:
        return np.zeros((6, 6))
    return np.cov(vectors, rowvar=False, ddof=1)


def method_summary_rows_full(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    for grid_cell, mu_x, sigma_x in core.GRID:
        for method in core.METHODS:
            group = sorted(
                [
                    row
                    for row in rows
                    if row["grid_cell"] == grid_cell
                    and row["method"] == method
                ],
                key=lambda row: int(row["rep"]),
            )
            vectors = np.vstack([coefficient_vector(row) for row in group])
            mean_vector = np.mean(vectors, axis=0)
            covariance_matrix = covariance(vectors)
            covariance_trace = float(np.trace(covariance_matrix))
            variance = (
                (len(group) - 1.0) / len(group) * covariance_trace
            )
            squared_bias = float(
                (mean_vector - TRUE_PARAMETERS)
                @ (mean_vector - TRUE_PARAMETERS)
            )
            summary: dict[str, Any] = {
                "grid_cell": grid_cell,
                "mu_x": mu_x,
                "sigma_x": sigma_x,
                "c_prop": sigma_x**2,
                "method": method,
                "repetitions": len(group),
                "coefficient_covariance_trace": covariance_trace,
                "finite_sample_parameter_variance_component": variance,
                "parameter_squared_bias": squared_bias,
                "variance_plus_squared_bias": variance + squared_bias,
            }
            for metric in core.METRICS:
                summary[f"mean_{metric}"], summary[f"se_{metric}"] = (
                    mean_se([float(row[metric]) for row in group])
                )
            for field in (
                "accepted_count",
                "candidate_ratio_ess",
                "candidate_ratio_relative_ess",
                "final_weight_ess",
                "synthetic_total_weight",
                "covariance_diagnostic",
                "saturation_fraction",
                "pilot_parameter_squared_error",
                "vp_condition_number",
                "sir_unique_candidate_count",
                "sir_multiplicity_ess",
            ):
                values = [
                    float(row[field])
                    for row in group
                    if row.get(field) is not None
                ]
                summary[f"mean_{field}"] = (
                    float(np.mean(values)) if values else None
                )
            for field in (
                "lower_log_ratio_clipping_count",
                "upper_log_ratio_clipping_count",
                "candidate_ratio_nonfinite_count",
                "coefficient_nonfinite_count",
                "metric_nonfinite_count",
                "covariance_diagnostic_nonfinite_count",
                "solver_failure_count",
                "execution_failure_count",
            ):
                summary[f"total_{field}"] = int(
                    sum(int(row[field]) for row in group)
                )
            for index, value in enumerate(mean_vector):
                summary[f"mean_theta_{index}"] = float(value)
            summaries.append(summary)
            for row_index in range(6):
                covariance_rows.append(
                    {
                        "grid_cell": grid_cell,
                        "mu_x": mu_x,
                        "sigma_x": sigma_x,
                        "c_prop": sigma_x**2,
                        "method": method,
                        "row_parameter": f"theta_{row_index}",
                        **{
                            f"theta_{column_index}": float(
                                covariance_matrix[
                                    row_index, column_index
                                ]
                            )
                            for column_index in range(6)
                        },
                    }
                )
    return summaries, covariance_rows


def paired_contrast_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["rep"]), row["grid_cell"], row["method"]): row
        for row in rows
    }
    repetitions = sorted({int(row["rep"]) for row in rows})
    output: list[dict[str, Any]] = []
    for grid_cell, mu_x, sigma_x in core.GRID:
        for first, second, label in WITHIN_CELL_COMPARISONS:
            differences = np.asarray(
                [
                    float(
                        lookup[(rep, grid_cell, first)]["test_mse"]
                    )
                    - float(
                        lookup[(rep, grid_cell, second)]["test_mse"]
                    )
                    for rep in repetitions
                ]
            )
            output.append(
                {
                    "grid_cell": grid_cell,
                    "mu_x": mu_x,
                    "sigma_x": sigma_x,
                    "c_prop": sigma_x**2,
                    "first_method": first,
                    "second_method": second,
                    "comparison": label,
                    "metric": "test_mse",
                    **paired_summary(differences),
                }
            )
    return output


def ratio_summary_rows(
    analytic: list[dict[str, Any]],
    ratio_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analytic_lookup = {row["grid_cell"]: row for row in analytic}
    output: list[dict[str, Any]] = []
    for grid_cell, mu_x, sigma_x in core.GRID:
        group = [
            row for row in ratio_rows if row["grid_cell"] == grid_cell
        ]
        analytic_row = analytic_lookup[grid_cell]
        summary: dict[str, Any] = {
            "grid_cell": grid_cell,
            "mu_x": mu_x,
            "sigma_x": sigma_x,
            "c_prop": sigma_x**2,
            "density_ratio_bounded": analytic_row["bounded"],
            "second_moment_finite": analytic_row["second_moment_finite"],
            "population_relative_ess": analytic_row[
                "population_relative_ess"
            ],
            "total_lower_log_ratio_clipping_count": int(
                sum(
                    int(row["lower_log_ratio_clipping_count"])
                    for row in group
                )
            ),
            "total_upper_log_ratio_clipping_count": int(
                sum(
                    int(row["upper_log_ratio_clipping_count"])
                    for row in group
                )
            ),
            "total_nonfinite_ratio_count": int(
                sum(int(row["nonfinite_ratio_count"]) for row in group)
            ),
            "maximum_generic_log_ratio_difference": float(
                max(
                    float(row["generic_log_ratio_max_abs_difference"])
                    for row in group
                )
            ),
        }
        for field in (
            "mean_candidate_ratio",
            "ratio_ess",
            "ratio_relative_ess",
            "ratio_p99",
            "maximum_ratio",
        ):
            summary[f"mean_{field}"], summary[f"se_{field}"] = mean_se(
                [float(row[field]) for row in group]
            )
        output.append(summary)
    return output


def diagnostics_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for grid_cell, mu_x, sigma_x in core.GRID:
        for method in core.METHODS:
            group = [
                row
                for row in rows
                if row["grid_cell"] == grid_cell
                and row["method"] == method
            ]
            probability_minima = [
                float(row["minimum_inclusion_probability"])
                for row in group
                if row.get("minimum_inclusion_probability") is not None
            ]
            probability_maxima = [
                float(row["maximum_inclusion_probability"])
                for row in group
                if row.get("maximum_inclusion_probability") is not None
            ]
            output.append(
                {
                    "grid_cell": grid_cell,
                    "mu_x": mu_x,
                    "sigma_x": sigma_x,
                    "c_prop": sigma_x**2,
                    "method": method,
                    "repetitions": len(group),
                    "minimum_inclusion_probability": (
                        min(probability_minima) if probability_minima else None
                    ),
                    "maximum_inclusion_probability": (
                        max(probability_maxima) if probability_maxima else None
                    ),
                    "minimum_accepted_count": int(
                        min(int(row["accepted_count"]) for row in group)
                    ),
                    "maximum_vp_condition_number": float(
                        max(float(row["vp_condition_number"]) for row in group)
                    ),
                    "empty_selection_count": int(
                        sum(
                            int(row["accepted_count"]) <= 0
                            for row in group
                            if method not in ("Real-only",)
                        )
                    ),
                    **{
                        field: int(sum(int(row[field]) for row in group))
                        for field in (
                            "candidate_ratio_nonfinite_count",
                            "coefficient_nonfinite_count",
                            "metric_nonfinite_count",
                            "covariance_diagnostic_nonfinite_count",
                            "solver_failure_count",
                            "execution_failure_count",
                        )
                    },
                }
            )
    return output
