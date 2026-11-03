"""Aggregation for the proposal stress grid."""


from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import t
from .stress_core import CLIPPING_METHODS, GRID, METHODS, METRICS


TRUE_PARAMETERS = np.asarray([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])


WITHIN_CELL_COMPARISONS = (
    ("REAL-Syn", "DRE-only", "REAL-Syn_minus_DRE-only"),
    ("REAL-Syn", "Uniform-IW", "REAL-Syn_minus_Uniform-IW"),
    ("REAL-Syn", "SIR-600", "REAL-Syn_minus_SIR-600"),
    (
        "Sqrt-clipped-SNIS",
        "SNIS-matched",
        "Sqrt-clipped-SNIS_minus_SNIS-matched",
    ),
    (
        "ESS-clipped-SNIS",
        "SNIS-matched",
        "ESS-clipped-SNIS_minus_SNIS-matched",
    ),
    ("REAL-Syn", "All-IW-full", "REAL-Syn_minus_All-IW-full"),
    (
        "REAL-Syn",
        "All-IW-matched",
        "REAL-Syn_minus_All-IW-matched",
    ),
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
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "N/A"
                        if row.get(key) is None
                        else row.get(key)
                    )
                    for key in fieldnames
                }
            )


def mean_se(values: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    mean = float(np.mean(vector))
    se = (
        float(np.std(vector, ddof=1) / np.sqrt(vector.size))
        if vector.size > 1
        else 0.0
    )
    return mean, se


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


def covariance_trace(vectors: np.ndarray) -> float:
    array = np.asarray(vectors, dtype=float)
    if array.shape[0] <= 1:
        return 0.0
    return float(np.trace(np.cov(array, rowvar=False, ddof=1)))


def aggregate_methods(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["grid_cell"], row["method"])].append(row)
    summary: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    for grid_cell, mu_x, sigma_x in GRID:
        for method in METHODS:
            group = sorted(
                groups[(grid_cell, method)], key=lambda item: int(item["rep"])
            )
            vectors = np.vstack([coefficient_vector(row) for row in group])
            mean_vector = np.mean(vectors, axis=0)
            covariance = (
                np.cov(vectors, rowvar=False, ddof=1)
                if vectors.shape[0] > 1
                else np.zeros((6, 6))
            )
            trace = float(np.trace(covariance))
            variance_component = (
                (vectors.shape[0] - 1.0) / vectors.shape[0] * trace
            )
            squared_bias = float(
                (mean_vector - TRUE_PARAMETERS)
                @ (mean_vector - TRUE_PARAMETERS)
            )
            item: dict[str, Any] = {
                "grid_cell": grid_cell,
                "mu_x": mu_x,
                "sigma_x": sigma_x,
                "c": sigma_x**2,
                "method": method,
                "repetitions": len(group),
                "parameter_covariance_trace": trace,
                "finite_sample_parameter_variance_component": variance_component,
                "parameter_squared_bias": squared_bias,
                "variance_plus_squared_bias": variance_component + squared_bias,
            }
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in group])
                item[f"mean_{metric}"], item[f"se_{metric}"] = mean_se(values)
            for key in (
                "accepted_count",
                "synthetic_total_weight",
                "final_weight_ess",
                "relative_final_weight_ess",
                "clipped_fraction",
                "sir_unique_candidate_count",
                "sir_multiplicity_ess",
            ):
                values = [
                    float(row[key])
                    for row in group
                    if row.get(key) not in (None, "N/A")
                ]
                item[f"mean_{key}"] = (
                    float(np.mean(values)) if values else None
                )
            clipping_values = [
                float(bool(row["clipping_activated"]))
                for row in group
                if row.get("clipping_activated") not in (None, "N/A")
            ]
            item["clipping_activation_frequency"] = (
                float(np.mean(clipping_values)) if clipping_values else None
            )
            for index, value in enumerate(mean_vector):
                item[f"mean_theta_{index}"] = float(value)
            summary.append(item)
            for row_index in range(6):
                covariance_rows.append(
                    {
                        "grid_cell": grid_cell,
                        "method": method,
                        "row_parameter": f"theta_{row_index}",
                        **{
                            f"theta_{column_index}": float(
                                covariance[row_index, column_index]
                            )
                            for column_index in range(6)
                        },
                    }
                )
    return summary, covariance_rows


def build_pairwise_contrasts(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["rep"]), row["grid_cell"], row["method"]): row
        for row in rows
    }
    repetitions = sorted({int(row["rep"]) for row in rows})
    output: list[dict[str, Any]] = []
    for grid_cell, _, _ in GRID:
        for first, second, label in WITHIN_CELL_COMPARISONS:
            for metric in METRICS:
                differences = np.asarray(
                    [
                        float(lookup[(rep, grid_cell, first)][metric])
                        - float(lookup[(rep, grid_cell, second)][metric])
                        for rep in repetitions
                    ]
                )
                output.append(
                    {
                        "comparison_scope": "within_cell",
                        "grid_cell": grid_cell,
                        "reference_grid_cell": grid_cell,
                        "first_method": first,
                        "second_method": second,
                        "comparison": label,
                        "metric": metric,
                        **paired_summary(differences),
                    }
                )
    for grid_cell, _, _ in GRID[1:]:
        for method in METHODS:
            for metric in METRICS:
                differences = np.asarray(
                    [
                        float(lookup[(rep, grid_cell, method)][metric])
                        - float(lookup[(rep, "G1", method)][metric])
                        for rep in repetitions
                    ]
                )
                output.append(
                    {
                        "comparison_scope": "grid_cell_minus_G1",
                        "grid_cell": grid_cell,
                        "reference_grid_cell": "G1",
                        "first_method": method,
                        "second_method": method,
                        "comparison": f"{grid_cell}_minus_G1",
                        "metric": metric,
                        **paired_summary(differences),
                    }
                )
    return output


def build_covariance_bootstrap(
    rows: list[dict[str, Any]],
    seed: int,
    resamples: int,
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["rep"]), row["grid_cell"], row["method"]): coefficient_vector(
            row
        )
        for row in rows
    }
    repetitions = sorted({int(row["rep"]) for row in rows})
    n = len(repetitions)
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(0, n, size=(int(resamples), n))
    output: list[dict[str, Any]] = []

    def add(
        scope: str,
        grid_cell: str,
        reference_cell: str,
        first: str,
        second: str,
        label: str,
        first_vectors: np.ndarray,
        second_vectors: np.ndarray,
    ) -> None:
        observed = covariance_trace(first_vectors) - covariance_trace(
            second_vectors
        )
        bootstrap = np.asarray(
            [
                covariance_trace(first_vectors[index])
                - covariance_trace(second_vectors[index])
                for index in indices
            ]
        )
        output.append(
            {
                "comparison_scope": scope,
                "grid_cell": grid_cell,
                "reference_grid_cell": reference_cell,
                "first_method": first,
                "second_method": second,
                "comparison": label,
                "repetitions": n,
                "bootstrap_resamples": int(resamples),
                "observed_covariance_trace_difference": observed,
                "bootstrap_mean_difference": float(np.mean(bootstrap)),
                "bootstrap_se": float(np.std(bootstrap, ddof=1)),
                "bootstrap_95_lower": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_95_upper": float(np.quantile(bootstrap, 0.975)),
                "fraction_bootstrap_first_lower": float(
                    np.mean(bootstrap < 0.0)
                ),
            }
        )

    for grid_cell, _, _ in GRID:
        for first, second, label in WITHIN_CELL_COMPARISONS:
            first_vectors = np.vstack(
                [lookup[(rep, grid_cell, first)] for rep in repetitions]
            )
            second_vectors = np.vstack(
                [lookup[(rep, grid_cell, second)] for rep in repetitions]
            )
            add(
                "within_cell",
                grid_cell,
                grid_cell,
                first,
                second,
                label,
                first_vectors,
                second_vectors,
            )
    for grid_cell, _, _ in GRID[1:]:
        for method in METHODS:
            first_vectors = np.vstack(
                [lookup[(rep, grid_cell, method)] for rep in repetitions]
            )
            second_vectors = np.vstack(
                [lookup[(rep, "G1", method)] for rep in repetitions]
            )
            add(
                "grid_cell_minus_G1",
                grid_cell,
                "G1",
                method,
                method,
                f"{grid_cell}_minus_G1",
                first_vectors,
                second_vectors,
            )
    return output


def severity_outputs(
    analytic: list[dict[str, Any]],
    ratio_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ratio_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    method_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in ratio_rows:
        ratio_groups[row["grid_cell"]].append(row)
    for row in method_rows:
        method_groups[(row["grid_cell"], row["method"])].append(row)
    summary: list[dict[str, Any]] = []
    detail: list[dict[str, Any]] = []
    analytic_lookup = {row["grid_cell"]: row for row in analytic}
    for grid_cell, mu_x, sigma_x in GRID:
        severity = analytic_lookup[grid_cell]
        ratios = ratio_groups[grid_cell]
        empirical_ress = np.asarray(
            [float(row["ratio_relative_ess"]) for row in ratios]
        )
        item = {
            "grid_cell": grid_cell,
            "mu_x": mu_x,
            "sigma_x": sigma_x,
            "c": sigma_x**2,
            "density_ratio_bounded": severity["bounded"],
            "boundedness_reason": severity["boundedness_reason"],
            "minimum_boundedness_matrix_eigenvalue": float(
                np.min(severity["boundedness_matrix_eigenvalues"])
            ),
            "maximum_boundedness_matrix_eigenvalue": float(
                np.max(severity["boundedness_matrix_eigenvalues"])
            ),
            "second_moment_finite": severity["second_moment_finite"],
            "minimum_second_moment_matrix_eigenvalue": float(
                np.min(severity["second_moment_matrix_eigenvalues"])
            ),
            "log_second_moment": severity["log_second_moment"],
            "population_relative_ess": severity["population_relative_ess"],
            "mean_empirical_relative_ess": float(np.mean(empirical_ress)),
            "se_empirical_relative_ess": mean_se(empirical_ress)[1],
            "mean_ratio_p99": float(
                np.mean([float(row["ratio_p99"]) for row in ratios])
            ),
            "mean_ratio_p99p9": float(
                np.mean([float(row["ratio_p99p9"]) for row in ratios])
            ),
            "mean_maximum_ratio": float(
                np.mean([float(row["maximum_ratio"]) for row in ratios])
            ),
            "total_lower_log_ratio_clipping_count": int(
                sum(
                    int(row["lower_log_ratio_clipping_count"])
                    for row in ratios
                )
            ),
            "total_upper_log_ratio_clipping_count": int(
                sum(
                    int(row["upper_log_ratio_clipping_count"])
                    for row in ratios
                )
            ),
        }
        for method in CLIPPING_METHODS:
            group = method_groups[(grid_cell, method)]
            item[f"{method}_clipping_activation_frequency"] = float(
                np.mean([bool(row["clipping_activated"]) for row in group])
            )
            item[f"{method}_mean_clipped_fraction"] = float(
                np.mean([float(row["clipped_fraction"]) for row in group])
            )
        summary.append(item)
        detail.append(
            {
                "grid_cell": grid_cell,
                "target_mean": severity["target_mean"].tolist(),
                "target_covariance": severity["target_covariance"].tolist(),
                "proposal_mean": severity["proposal_mean"].tolist(),
                "proposal_covariance": severity[
                    "proposal_covariance"
                ].tolist(),
                "boundedness_matrix_eigenvalues": severity[
                    "boundedness_matrix_eigenvalues"
                ].tolist(),
                "second_moment_matrix_eigenvalues": severity[
                    "second_moment_matrix_eigenvalues"
                ].tolist(),
                "boundedness_tolerance": severity[
                    "boundedness_tolerance"
                ],
            }
        )
    return summary, detail
