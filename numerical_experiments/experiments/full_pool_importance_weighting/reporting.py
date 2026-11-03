"""Aggregation, covariance calculations, paired contrasts, and figures."""


from __future__ import annotations
import csv
import math
from pathlib import Path
from typing import Any
import matplotlib


matplotlib.use("Agg")


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t
from . import core


METHOD_STYLES = {
    "Real-only": ("#111827", "o", "--"),
    "Uniform-IW": ("#6B7280", "s", "-"),
    "DRE-only": ("#1F77B4", "^", "-"),
    "REAL-Syn": ("#D62728", "o", "-"),
    "All-IW-matched": ("#9467BD", "D", "-"),
    "All-IW-full": ("#2CA02C", "v", "--"),
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with Path(path).open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("Monte Carlo SE requires at least two repetitions.")
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1) / math.sqrt(values.size)),
    )


def build_summaries(
    metric_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    source_config: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    truth = core.true_coefficients(source_config)
    gp = core.target_design_second_moment(source_config)
    summary_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for alpha in core.ALPHAS:
        for method in core.METHODS:
            selected_metrics = sorted(
                (
                    row
                    for row in metric_rows
                    if row["method"] == method
                    and np.isclose(float(row["alpha"]), alpha)
                ),
                key=lambda row: int(row["rep"]),
            )
            selected_parameters = sorted(
                (
                    row
                    for row in parameter_rows
                    if row["method"] == method
                    and np.isclose(float(row["alpha"]), alpha)
                ),
                key=lambda row: int(row["rep"]),
            )
            selected_diagnostics = sorted(
                (
                    row
                    for row in diagnostic_rows
                    if row["method"] == method
                    and np.isclose(float(row["alpha"]), alpha)
                ),
                key=lambda row: int(row["rep"]),
            )
            repetitions = len(selected_metrics)
            vectors = np.asarray(
                [
                    [
                        float(row[f"theta_{index}"])
                        for index in range(truth.size)
                    ]
                    for row in selected_parameters
                ]
            )
            covariance = np.atleast_2d(np.cov(vectors.T, ddof=1))
            trace_covariance = float(np.trace(covariance))
            slope_trace = float(np.trace(covariance[1:, 1:]))
            intercept_variance = float(covariance[0, 0])
            mean_vector = np.mean(vectors, axis=0)
            squared_bias = float(np.sum((mean_vector - truth) ** 2))
            variance_component = (
                (repetitions - 1) / repetitions * trace_covariance
            )
            parameter_errors = np.asarray(
                [
                    float(row["parameter_squared_error"])
                    for row in selected_metrics
                ]
            )
            mean_parameter_error, se_parameter_error = mean_se(
                parameter_errors
            )
            test_values = np.asarray(
                [float(row["test_mse"]) for row in selected_metrics]
            )
            noiseless_values = np.asarray(
                [
                    float(row["noiseless_prediction_mse"])
                    for row in selected_metrics
                ]
            )
            mean_test, se_test = mean_se(test_values)
            mean_noiseless, se_noiseless = mean_se(noiseless_values)
            accepted = np.asarray(
                [
                    float(row["actual_accepted_count"])
                    for row in selected_diagnostics
                ]
            )
            summary_rows.append(
                {
                    "alpha": alpha,
                    "method": method,
                    "repetitions": repetitions,
                    "candidate_pool_size": 6000,
                    "mean_accepted_synthetic_rows": float(
                        np.mean(accepted)
                    ),
                    "se_accepted_synthetic_rows": float(
                        np.std(accepted, ddof=1) / math.sqrt(repetitions)
                    ),
                    "mean_test_mse": mean_test,
                    "se_test_mse": se_test,
                    "mean_noiseless_prediction_mse": mean_noiseless,
                    "se_noiseless_prediction_mse": se_noiseless,
                    "mean_parameter_squared_error": mean_parameter_error,
                    "se_parameter_squared_error": se_parameter_error,
                    "squared_bias": squared_bias,
                    "trace_coefficient_covariance": trace_covariance,
                    "empirical_parameter_mse_decomposition": (
                        variance_component + squared_bias
                    ),
                    "decomposition_absolute_error": abs(
                        mean_parameter_error
                        - variance_component
                        - squared_bias
                    ),
                }
            )
            covariance_rows.append(
                {
                    "alpha": alpha,
                    "method": method,
                    "repetitions": repetitions,
                    "covariance_denominator": repetitions - 1,
                    "trace_coefficient_covariance": trace_covariance,
                    "slope_only_covariance_trace": slope_trace,
                    "intercept_variance": intercept_variance,
                    "squared_bias": squared_bias,
                    "variance_component_finite_sample": variance_component,
                    "mean_parameter_squared_error": mean_parameter_error,
                    "variance_plus_squared_bias": (
                        variance_component + squared_bias
                    ),
                    "decomposition_absolute_error": abs(
                        mean_parameter_error
                        - variance_component
                        - squared_bias
                    ),
                    "target_weighted_covariance_trace": float(
                        np.trace(gp @ covariance)
                    ),
                }
            )
            for row_index in range(truth.size):
                for column_index in range(truth.size):
                    matrix_rows.append(
                        {
                            "alpha": alpha,
                            "method": method,
                            "row_index": row_index,
                            "column_index": column_index,
                            "covariance": float(
                                covariance[row_index, column_index]
                            ),
                        }
                    )
    return summary_rows, covariance_rows, matrix_rows


def build_paired_contrasts(
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["rep"]), row["method"], float(row["alpha"])): row
        for row in metric_rows
    }
    repetitions = len({int(row["rep"]) for row in metric_rows})
    critical = float(student_t.ppf(0.975, df=repetitions - 1))
    output: list[dict[str, Any]] = []
    for alpha in core.ALPHAS:
        for baseline in core.PAIRED_BASELINES:
            for metric in core.PARAMETER_METRICS:
                differences = np.asarray(
                    [
                        float(lookup[(rep, "REAL-Syn", alpha)][metric])
                        - float(lookup[(rep, baseline, alpha)][metric])
                        for rep in range(repetitions)
                    ]
                )
                mean_value, se_value = mean_se(differences)
                output.append(
                    {
                        "alpha": alpha,
                        "contrast": f"REAL-Syn minus {baseline}",
                        "baseline": baseline,
                        "metric": metric,
                        "paired_repetitions": repetitions,
                        "mean_paired_difference": mean_value,
                        "se_paired_difference": se_value,
                        "t_critical_975": critical,
                        "ci95_low": mean_value - critical * se_value,
                        "ci95_high": mean_value + critical * se_value,
                        "median_paired_difference": float(
                            np.median(differences)
                        ),
                        "fraction_real_syn_lower": float(
                            np.mean(differences < 0.0)
                        ),
                    }
                )
    return output


def _plot_series(
    summary_rows: list[dict[str, Any]],
    *,
    y_field: str,
    y_label: str,
    output: Path,
    error_field: str | None = None,
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    for method in core.METHODS:
        rows = sorted(
            (row for row in summary_rows if row["method"] == method),
            key=lambda row: float(row["alpha"]),
        )
        x_values = np.asarray([float(row["alpha"]) for row in rows])
        y_values = np.asarray([float(row[y_field]) for row in rows])
        color, marker, linestyle = METHOD_STYLES[method]
        if error_field is None:
            axis.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.7,
                markersize=5,
                label=method,
            )
        else:
            errors = np.asarray([float(row[error_field]) for row in rows])
            axis.errorbar(
                x_values,
                y_values,
                yerr=errors,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4.5,
                capsize=2.5,
                label=method,
            )
    axis.set_xlabel(r"Acceptance fraction $\alpha$")
    axis.set_ylabel(y_label)
    axis.set_xticks(core.ALPHAS)
    axis.grid(True, alpha=0.22)
    axis.legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    summary_rows: list[dict[str, Any]], output: Path
) -> None:
    _plot_series(
        summary_rows,
        y_field="mean_test_mse",
        error_field="se_test_mse",
        y_label="Mean test MSE",
        output=output / "figure_1_mean_test_mse_vs_alpha.png",
    )
    _plot_series(
        summary_rows,
        y_field="mean_noiseless_prediction_mse",
        error_field="se_noiseless_prediction_mse",
        y_label="Mean noiseless prediction MSE",
        output=output
        / "figure_2_mean_noiseless_prediction_mse_vs_alpha.png",
    )
    _plot_series(
        summary_rows,
        y_field="trace_coefficient_covariance",
        y_label="Trace of coefficient covariance",
        output=output / "figure_3_covariance_trace_vs_alpha.png",
    )
    _plot_series(
        summary_rows,
        y_field="mean_parameter_squared_error",
        error_field="se_parameter_squared_error",
        y_label="Mean full parameter squared error",
        output=output / "figure_4_parameter_mse_vs_alpha.png",
    )
    _plot_series(
        summary_rows,
        y_field="mean_accepted_synthetic_rows",
        y_label="Mean accepted synthetic rows",
        output=output / "figure_5_mean_accepted_rows_vs_alpha.png",
    )
