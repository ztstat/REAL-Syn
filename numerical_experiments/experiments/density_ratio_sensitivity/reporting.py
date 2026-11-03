"""Aggregation for density-ratio sensitivity."""


from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import t
from . import core


TRUE_PARAMETERS = np.asarray([0.0, 1.0, 0.8, 0.6, 0.4, 0.2])


WITHIN_REGIME_COMPARISONS = (
    ("REAL-Syn", "DRE-only", "REAL-Syn_minus_DRE-only"),
    ("REAL-Syn", "Uniform-IW", "REAL-Syn_minus_Uniform-IW"),
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
        raise ValueError(f"Cannot write empty CSV: {path.name}")
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


def mean_se(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    return (
        float(np.mean(data)),
        float(np.std(data, ddof=1) / np.sqrt(data.size))
        if data.size > 1
        else 0.0,
    )


def coefficient_vector(row: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(row[f"theta_{index}"]) for index in range(6)])


def covariance(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.shape[0] <= 1:
        return np.zeros((6, 6))
    return np.cov(values, rowvar=False, ddof=1)


def paired_summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    mean, se = mean_se(data)
    critical = (
        float(t.ppf(0.975, df=data.size - 1)) if data.size > 1 else 0.0
    )
    return {
        "n_pairs": int(data.size),
        "mean_difference": mean,
        "standard_error": se,
        "t95_lower": mean - critical * se,
        "t95_upper": mean + critical * se,
        "median_difference": float(np.median(data)),
        "fraction_first_lower": float(np.mean(data < 0.0)),
    }


def ratio_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    numeric_keys = (
        "mean_oracle_ratio",
        "mean_transformed_ratio",
        "ratio_ess",
        "ratio_relative_ess",
        "log_ratio_rmse_vs_oracle",
        "log_ratio_mae_vs_oracle",
        "pearson_log_ratio_correlation_vs_oracle",
        "spearman_ratio_correlation_vs_oracle",
        "ratio_p50",
        "ratio_p90",
        "ratio_p95",
        "ratio_p99",
        "ratio_p99p9",
        "maximum_ratio",
        "p99_tail_compression_ratio",
        "maximum_tail_compression_ratio",
        "mean_absolute_ratio_difference_vs_oracle",
        "mean_absolute_log_ratio_difference_vs_oracle",
    )
    for regime, shrinkage in core.RATIO_REGIMES:
        group = [row for row in rows if row["ratio_regime"] == regime]
        item: dict[str, Any] = {
            "ratio_regime": regime,
            "lambda": shrinkage,
            "repetitions": len(group),
            "ideal_unclipped_population_mean_transformed_ratio": 1.0,
            "total_inherited_oracle_lower_clipping_count": int(
                sum(
                    int(row["inherited_oracle_lower_clipping_count"])
                    for row in group
                )
            ),
            "total_inherited_oracle_upper_clipping_count": int(
                sum(
                    int(row["inherited_oracle_upper_clipping_count"])
                    for row in group
                )
            ),
            "total_transformed_ratio_additional_clipping_count": int(
                sum(
                    int(row["transformed_ratio_additional_clipping_count"])
                    for row in group
                )
            ),
            "total_nonfinite_count": int(
                sum(int(row["nonfinite_count"]) for row in group)
            ),
        }
        for key in numeric_keys:
            values = np.asarray([float(row[key]) for row in group])
            item[f"mean_{key}"], item[f"se_{key}"] = mean_se(values)
        output.append(item)
    return output


def method_summary_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    for regime, shrinkage in core.RATIO_REGIMES:
        for method in core.METHODS:
            group = sorted(
                [
                    row
                    for row in rows
                    if row["ratio_regime"] == regime
                    and row["method"] == method
                ],
                key=lambda row: int(row["rep"]),
            )
            vectors = np.vstack([coefficient_vector(row) for row in group])
            mean_vector = np.mean(vectors, axis=0)
            cov = covariance(vectors)
            trace = float(np.trace(cov))
            variance = (len(group) - 1.0) / len(group) * trace
            squared_bias = float(
                (mean_vector - TRUE_PARAMETERS)
                @ (mean_vector - TRUE_PARAMETERS)
            )
            item: dict[str, Any] = {
                "ratio_regime": regime,
                "lambda": shrinkage,
                "method": method,
                "repetitions": len(group),
                "coefficient_covariance_trace": trace,
                "finite_sample_parameter_variance_component": variance,
                "parameter_squared_bias": squared_bias,
                "variance_plus_squared_bias": variance + squared_bias,
            }
            for metric in core.METRICS:
                values = np.asarray([float(row[metric]) for row in group])
                item[f"mean_{metric}"], item[f"se_{metric}"] = mean_se(values)
            for key in (
                "accepted_count",
                "synthetic_rows_used",
                "synthetic_total_weight",
                "final_weight_ess",
                "relative_final_weight_ess",
                "p95_final_weight",
                "p99_final_weight",
                "maximum_final_weight",
                "saturation_fraction",
            ):
                values = [
                    float(row[key])
                    for row in group
                    if row.get(key) is not None
                ]
                item[f"mean_{key}"] = (
                    float(np.mean(values)) if values else None
                )
            for index, value in enumerate(mean_vector):
                item[f"mean_theta_{index}"] = float(value)
            output.append(item)
            for row_index in range(6):
                covariance_rows.append(
                    {
                        "ratio_regime": regime,
                        "lambda": shrinkage,
                        "method": method,
                        "row_parameter": f"theta_{row_index}",
                        **{
                            f"theta_{column_index}": float(
                                cov[row_index, column_index]
                            )
                            for column_index in range(6)
                        },
                    }
                )
    return output, covariance_rows


def acceptance_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = (
        "pearson_pi_correlation_vs_oracle",
        "spearman_pi_correlation_vs_oracle",
        "pearson_acceptance_score_correlation_vs_oracle",
        "spearman_acceptance_score_correlation_vs_oracle",
        "mean_absolute_pi_difference_vs_oracle",
        "maximum_absolute_pi_difference_vs_oracle",
        "saturation_fraction",
        "saturation_fraction_difference_vs_oracle",
        "accepted_mask_overlap_count",
        "accepted_mask_overlap_fraction_of_oracle",
        "accepted_mask_jaccard_similarity",
    )
    for regime, shrinkage in core.RATIO_REGIMES:
        for method in core.ACCEPTANCE_DIAGNOSTIC_METHODS:
            group = [
                row
                for row in rows
                if row["ratio_regime"] == regime
                and row["method"] == method
            ]
            item: dict[str, Any] = {
                "ratio_regime": regime,
                "lambda": shrinkage,
                "method": method,
                "repetitions": len(group),
            }
            for key in keys:
                values = np.asarray([float(row[key]) for row in group])
                item[f"mean_{key}"], item[f"se_{key}"] = mean_se(values)
            output.append(item)
    return output


def paired_contrast_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["rep"]), row["ratio_regime"], row["method"]): row
        for row in rows
    }
    repetitions = sorted({int(row["rep"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method in core.METHODS:
        for regime, shrinkage in core.RATIO_REGIMES[1:]:
            for metric in core.METRICS:
                differences = np.asarray(
                    [
                        float(lookup[(rep, regime, method)][metric])
                        - float(
                            lookup[(rep, "shrinkage_000", method)][metric]
                        )
                        for rep in repetitions
                    ]
                )
                output.append(
                    {
                        "comparison_scope": "regime_minus_oracle",
                        "ratio_regime": regime,
                        "reference_regime": "shrinkage_000",
                        "lambda": shrinkage,
                        "first_method": method,
                        "second_method": method,
                        "comparison": f"{regime}_minus_shrinkage_000",
                        "metric": metric,
                        **paired_summary(differences),
                    }
                )
    for regime, shrinkage in core.RATIO_REGIMES:
        for first, second, label in WITHIN_REGIME_COMPARISONS:
            for metric in core.METRICS:
                differences = np.asarray(
                    [
                        float(lookup[(rep, regime, first)][metric])
                        - float(lookup[(rep, regime, second)][metric])
                        for rep in repetitions
                    ]
                )
                output.append(
                    {
                        "comparison_scope": "within_regime",
                        "ratio_regime": regime,
                        "reference_regime": regime,
                        "lambda": shrinkage,
                        "first_method": first,
                        "second_method": second,
                        "comparison": label,
                        "metric": metric,
                        **paired_summary(differences),
                    }
                )
    return output


def covariance_bootstrap_rows(
    rows: list[dict[str, Any]], seed: int, resamples: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], np.ndarray] = {}
    repetitions = sorted({int(row["rep"]) for row in rows})
    for regime, _ in core.RATIO_REGIMES:
        for method in core.METHODS:
            group = sorted(
                [
                    row
                    for row in rows
                    if row["ratio_regime"] == regime
                    and row["method"] == method
                ],
                key=lambda row: int(row["rep"]),
            )
            groups[(regime, method)] = np.vstack(
                [coefficient_vector(row) for row in group]
            )
    rng = np.random.RandomState(int(seed))
    indices = rng.randint(
        0, len(repetitions), size=(int(resamples), len(repetitions))
    )

    def trace(values: np.ndarray) -> float:
        return float(np.trace(covariance(values)))

    def add(
        scope: str,
        regime: str,
        reference: str,
        first: str,
        second: str,
        label: str,
    ) -> dict[str, Any]:
        first_values = groups[(regime, first)]
        second_values = groups[(reference, second)]
        observed = trace(first_values) - trace(second_values)
        bootstrap = np.asarray(
            [
                trace(first_values[index]) - trace(second_values[index])
                for index in indices
            ]
        )
        return {
            "comparison_scope": scope,
            "ratio_regime": regime,
            "reference_regime": reference,
            "lambda": core.LAMBDA_BY_REGIME[regime],
            "first_method": first,
            "second_method": second,
            "comparison": label,
            "repetitions": len(repetitions),
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

    output: list[dict[str, Any]] = []
    for method in core.METHODS:
        for regime, _ in core.RATIO_REGIMES[1:]:
            output.append(
                add(
                    "regime_minus_oracle",
                    regime,
                    "shrinkage_000",
                    method,
                    method,
                    f"{regime}_minus_shrinkage_000",
                )
            )
    for regime, _ in core.RATIO_REGIMES:
        for first, second, label in WITHIN_REGIME_COMPARISONS:
            output.append(
                add(
                    "within_regime",
                    regime,
                    regime,
                    first,
                    second,
                    label,
                )
            )
    return output


def degradation_rows(
    rows: list[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    lookup = {
        (int(row["rep"]), row["ratio_regime"], row["method"]): row
        for row in rows
    }
    repetitions = sorted({int(row["rep"]) for row in rows})
    output: list[dict[str, Any]] = []
    for method in core.METHODS:
        for regime, shrinkage in core.RATIO_REGIMES[1:]:
            for metric in core.METRICS:
                oracle_method = np.asarray(
                    [
                        float(
                            lookup[(rep, "shrinkage_000", method)][metric]
                        )
                        for rep in repetitions
                    ]
                )
                real_only = np.asarray(
                    [
                        float(
                            lookup[
                                (rep, "shrinkage_000", "Real-only")
                            ][metric]
                        )
                        for rep in repetitions
                    ]
                )
                misspecified = np.asarray(
                    [
                        float(lookup[(rep, regime, method)][metric])
                        for rep in repetitions
                    ]
                )
                numerator = float(np.mean(misspecified - oracle_method))
                denominator = float(np.mean(real_only - oracle_method))
                valid = denominator > float(tolerance)
                output.append(
                    {
                        "ratio_regime": regime,
                        "lambda": shrinkage,
                        "method": method,
                        "metric": metric,
                        "raw_degradation_numerator": numerator,
                        "raw_oracle_gain_denominator": denominator,
                        "denominator_tolerance": float(tolerance),
                        "denominator_valid": bool(valid),
                        "degradation_fraction": (
                            numerator / denominator if valid else None
                        ),
                    }
                )
    return output


def method_summary_rows_full(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct method summaries from repetition-level rows."""
    summaries, covariance_rows = method_summary_rows(rows)
    lookup = {
        (row["ratio_regime"], row["method"]): [
            item
            for item in rows
            if item["ratio_regime"] == row["ratio_regime"]
            and item["method"] == row["method"]
        ]
        for row in summaries
    }
    for summary in summaries:
        group = lookup[(summary["ratio_regime"], summary["method"])]
        summary.update(
            {
                "mean_candidate_ratio_ess": float(
                    np.mean(
                        [float(item["candidate_ratio_ess"]) for item in group]
                    )
                ),
                "mean_candidate_ratio_relative_ess": float(
                    np.mean(
                        [
                            float(item["candidate_ratio_relative_ess"])
                            for item in group
                        ]
                    )
                ),
                "total_ratio_nonfinite_count": int(
                    sum(
                        int(item["candidate_ratio_nonfinite_count"])
                        for item in group
                    )
                ),
                "total_coefficient_nonfinite_count": int(
                    sum(
                        int(item["coefficient_nonfinite_count"])
                        for item in group
                    )
                ),
                "total_metric_nonfinite_count": int(
                    sum(
                        int(item["metric_nonfinite_count"])
                        for item in group
                    )
                ),
                "total_solver_failure_count": int(
                    sum(int(item["solver_failure_count"]) for item in group)
                ),
                "total_execution_failure_count": int(
                    sum(
                        int(item["execution_failure_count"]) for item in group
                    )
                ),
                "total_inherited_oracle_lower_clipping_count": int(
                    sum(
                        int(item["inherited_oracle_lower_clipping_count"])
                        for item in group
                    )
                ),
                "total_inherited_oracle_upper_clipping_count": int(
                    sum(
                        int(item["inherited_oracle_upper_clipping_count"])
                        for item in group
                    )
                ),
                "total_transformed_ratio_additional_clipping_count": int(
                    sum(
                        int(
                            item[
                                "transformed_ratio_additional_clipping_count"
                            ]
                        )
                        for item in group
                    )
                ),
            }
        )
    return summaries, covariance_rows


def gain_retained_rows(
    rows: list[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    """Compute REAL-Syn augmentation gains retained."""
    grouped = {
        (regime, method): [
            row
            for row in rows
            if row["ratio_regime"] == regime and row["method"] == method
        ]
        for regime, _ in core.RATIO_REGIMES
        for method in core.METHODS
    }

    def mean_metric(regime: str, method: str, metric: str) -> float:
        return float(
            np.mean(
                [
                    float(row[metric])
                    for row in grouped[(regime, method)]
                ]
            )
        )

    def covariance_trace(regime: str, method: str) -> float:
        vectors = np.vstack(
            [coefficient_vector(row) for row in grouped[(regime, method)]]
        )
        return float(np.trace(covariance(vectors)))

    output: list[dict[str, Any]] = []
    real_mse = mean_metric("shrinkage_000", "Real-only", "test_mse")
    oracle_real_syn_mse = mean_metric(
        "shrinkage_000", "REAL-Syn", "test_mse"
    )
    mse_denominator = real_mse - oracle_real_syn_mse
    real_covariance = covariance_trace("shrinkage_000", "Real-only")
    oracle_real_syn_covariance = covariance_trace(
        "shrinkage_000", "REAL-Syn"
    )
    covariance_denominator = (
        real_covariance - oracle_real_syn_covariance
    )
    for regime, shrinkage in core.RATIO_REGIMES:
        current_mse = mean_metric(regime, "REAL-Syn", "test_mse")
        current_covariance = covariance_trace(regime, "REAL-Syn")
        for metric, real_value, current_value, oracle_value, denominator in (
            (
                "test_mse",
                real_mse,
                current_mse,
                oracle_real_syn_mse,
                mse_denominator,
            ),
            (
                "coefficient_covariance_trace",
                real_covariance,
                current_covariance,
                oracle_real_syn_covariance,
                covariance_denominator,
            ),
        ):
            valid = abs(denominator) > float(tolerance)
            output.append(
                {
                    "ratio_regime": regime,
                    "lambda": shrinkage,
                    "method": "REAL-Syn",
                    "metric": metric,
                    "real_only_value": real_value,
                    "real_syn_lambda_value": current_value,
                    "real_syn_lambda0_value": oracle_value,
                    "gain_numerator": real_value - current_value,
                    "lambda0_gain_denominator": denominator,
                    "denominator_tolerance": float(tolerance),
                    "denominator_valid": bool(valid),
                    "gain_retained": (
                        (real_value - current_value) / denominator
                        if valid
                        else None
                    ),
                }
            )
    return output


def diagnostics_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for regime, shrinkage in core.RATIO_REGIMES:
        for method in core.METHODS:
            group = [
                row
                for row in rows
                if row["ratio_regime"] == regime
                and row["method"] == method
            ]
            probability_values = [
                float(row["minimum_inclusion_probability"])
                for row in group
                if row.get("minimum_inclusion_probability") is not None
            ] + [
                float(row["maximum_inclusion_probability"])
                for row in group
                if row.get("maximum_inclusion_probability") is not None
            ]
            output.append(
                {
                    "ratio_regime": regime,
                    "lambda": shrinkage,
                    "method": method,
                    "repetitions": len(group),
                    "minimum_inclusion_probability": (
                        float(np.min(probability_values))
                        if probability_values
                        else None
                    ),
                    "maximum_inclusion_probability": (
                        float(np.max(probability_values))
                        if probability_values
                        else None
                    ),
                    "mean_candidate_ratio_relative_ess": float(
                        np.mean(
                            [
                                float(
                                    row["candidate_ratio_relative_ess"]
                                )
                                for row in group
                            ]
                        )
                    ),
                    "mean_final_weight_ess": (
                        float(
                            np.mean(
                                [
                                    float(row["final_weight_ess"])
                                    for row in group
                                    if row.get("final_weight_ess") is not None
                                ]
                            )
                        )
                        if any(
                            row.get("final_weight_ess") is not None
                            for row in group
                        )
                        else None
                    ),
                    "empty_selection_count": int(
                        sum(
                            int(row["accepted_count"]) <= 0
                            for row in group
                            if row.get("accepted_count") is not None
                        )
                    ),
                    "ratio_nonfinite_count": int(
                        sum(
                            int(row["candidate_ratio_nonfinite_count"])
                            for row in group
                        )
                    ),
                    "coefficient_nonfinite_count": int(
                        sum(
                            int(row["coefficient_nonfinite_count"])
                            for row in group
                        )
                    ),
                    "metric_nonfinite_count": int(
                        sum(
                            int(row["metric_nonfinite_count"])
                            for row in group
                        )
                    ),
                    "solver_failure_count": int(
                        sum(
                            int(row["solver_failure_count"]) for row in group
                        )
                    ),
                    "execution_failure_count": int(
                        sum(
                            int(row["execution_failure_count"])
                            for row in group
                        )
                    ),
                }
            )
    return output
