"""End-to-end logistic REAL-Syn classification simulation."""


from __future__ import annotations
import csv
import hashlib
import time
from pathlib import Path
from typing import Any
import numpy as np
from . import metrics as logistic_metrics
from . import parameters as logistic_parameters
from .model import (
    compute_score_psi,
    fit_weighted_logistic,
    predict_probability,
)


METHOD_ORDER = logistic_metrics.METHOD_ORDER


METHOD_KEYS = logistic_metrics.METHOD_KEYS


def compute_calibrated_pi(
    score: np.ndarray,
    alpha: float,
    n_iterations: int = 60,
) -> tuple[np.ndarray, float]:
    """Build pi=min(1,c*score) with empirical mean pi equal to alpha."""
    score = np.maximum(np.asarray(score, dtype=float), 0.0)
    if score.ndim != 1 or score.size == 0:
        raise ValueError("Acceptance scores must be a nonempty vector.")
    if not np.all(np.isfinite(score)):
        raise ValueError("Acceptance scores must be finite.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one.")
    if np.all(score == 0.0):
        raise ValueError("Cannot calibrate an all-zero score.")
    low = 0.0
    high = 1.0
    while float(np.mean(np.minimum(1.0, high * score))) < float(alpha):
        high *= 2.0
    for _ in range(int(n_iterations)):
        midpoint = 0.5 * (low + high)
        if float(np.mean(np.minimum(1.0, midpoint * score))) < float(alpha):
            low = midpoint
        else:
            high = midpoint
    pi = np.minimum(1.0, high * score)
    return np.clip(pi, 0.0, 1.0), float(high)


def sample_with_pi(
    x: np.ndarray,
    y: np.ndarray,
    pi: np.ndarray,
    ratio: np.ndarray,
    alpha: float,
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Bernoulli acceptance followed by W=alpha*r/pi."""
    pi = np.clip(np.asarray(pi, dtype=float), 0.0, 1.0)
    ratio = np.asarray(ratio, dtype=float)
    accepted_mask = rng.random_sample(pi.size) < pi
    accepted_pi = np.clip(pi[accepted_mask], 1e-12, 1.0)
    return {
        "x": np.asarray(x, dtype=float)[accepted_mask],
        "y": np.asarray(y, dtype=int)[accepted_mask],
        "weight": float(alpha) * ratio[accepted_mask] / accepted_pi,
        "accepted_mask": accepted_mask,
        "accepted_pi": accepted_pi,
        "accepted_ratio": ratio[accepted_mask],
        "accept_rate": float(np.mean(accepted_mask)),
    }


def estimate_vp(psi: np.ndarray) -> np.ndarray:
    """Uncentered score second moment."""
    psi = np.asarray(psi, dtype=float)
    return psi.T @ psi / float(psi.shape[0])


def standardized_score_norm(
    psi: np.ndarray,
    vp_inverse: np.ndarray,
) -> np.ndarray:
    quadratic = np.sum((psi @ vp_inverse) * psi, axis=1)
    return np.sqrt(np.maximum(quadratic, 0.0))


def estimate_vw(
    psi: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_psi = np.asarray(weights, dtype=float)[:, None] * np.asarray(
        psi, dtype=float
    )
    return weighted_psi.T @ weighted_psi / float(weighted_psi.shape[0])


def covariance_diagnostic(
    vp_inverse: np.ndarray,
    vw: np.ndarray,
) -> float:
    return float(np.trace(vp_inverse @ vw) / vp_inverse.shape[0])


def _fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _stream_rng(rep_seed: int, offset: int) -> np.random.RandomState:
    return np.random.RandomState((int(rep_seed) + int(offset)) % (2**32 - 1))


def _method_sample_bundle(
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    ratio: np.ndarray,
    influence: np.ndarray,
    alpha: float,
    rep_seed: int,
    offsets: dict[str, int],
) -> dict[str, dict[str, Any] | None]:
    score_by_method = {
        "Uniform acceptance": np.ones(candidate_y.size, dtype=float),
        "Density-ratio-only acceptance": ratio,
        "Influence-only acceptance": influence,
        "REAL-Syn": ratio * influence,
    }
    samples: dict[str, dict[str, Any] | None] = {"Real-only": None}
    for method, score in score_by_method.items():
        if method == "Uniform acceptance":
            pi = np.full(candidate_y.size, float(alpha), dtype=float)
            calibration_constant = float(alpha)
        else:
            pi, calibration_constant = compute_calibrated_pi(score, alpha)
        sample = sample_with_pi(
            candidate_x,
            candidate_y,
            pi,
            ratio,
            alpha,
            _stream_rng(rep_seed, offsets[METHOD_KEYS[method]]),
        )
        sample["pi"] = pi
        sample["calibration_constant"] = calibration_constant
        samples[method] = sample
    return samples


def run_single_repetition(
    config: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
    true_test_probability: np.ndarray,
    size_override: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Run one paired repetition and return only in-memory records."""
    start_time = time.perf_counter()
    sizes = {
        "n_real": int(config["n_real"]),
        "n_candidates": int(config["n_candidates"]),
    }
    if size_override is not None:
        sizes.update({key: int(value) for key, value in size_override.items()})
    seed_config = config["seeds"]
    rep_seed = int(seed_config["base_seed"]) + int(seed_config["rep_seed_stride"]) * int(rep)
    offsets = {key: int(value) for key, value in seed_config["stream_offsets"].items()}

    target_distribution = config["target"]
    proposal_distribution = config["proposal"]
    real_x, real_y = logistic_parameters.sample_joint(
        sizes["n_real"],
        float(config["pi_p"]),
        target_distribution,
        _stream_rng(rep_seed, offsets["real_data"]),
    )
    candidate_x, candidate_y = logistic_parameters.sample_joint(
        sizes["n_candidates"],
        float(config["pi_q"]),
        proposal_distribution,
        _stream_rng(rep_seed, offsets["candidate_data"]),
    )
    paired_data_fingerprint = _fingerprint(
        real_x, real_y, candidate_x, candidate_y, test_x, test_y
    )

    pilot = fit_weighted_logistic(real_x, real_y, None, config["solver"])
    real_psi = compute_score_psi(real_x, real_y, pilot)
    vp = estimate_vp(real_psi)
    vp_ridge = float(config["vp_ridge"])
    stabilized_vp = vp + vp_ridge * np.eye(vp.shape[0])
    vp_inverse = np.linalg.inv(stabilized_vp)

    clipped_log_ratio, raw_log_ratio = logistic_parameters.oracle_log_density_ratio(
        candidate_x,
        candidate_y,
        config,
        apply_clipping=True,
    )
    ratio = np.exp(clipped_log_ratio)
    candidate_psi = compute_score_psi(candidate_x, candidate_y, pilot)
    influence = standardized_score_norm(candidate_psi, vp_inverse)
    samples = _method_sample_bundle(
        candidate_x,
        candidate_y,
        ratio,
        influence,
        float(config["alpha"]),
        rep_seed,
        offsets,
    )

    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    model_bundle: dict[str, dict[str, Any]] = {}
    beta_true = np.asarray(config["true_logistic"]["beta"], dtype=float)
    solver_failure_count = int(not pilot["converged"])
    empty_selection_count = 0
    nonfinite_arrays: list[np.ndarray] = [
        real_x,
        real_y,
        candidate_x,
        candidate_y,
        vp,
        vp_inverse,
        raw_log_ratio,
        clipped_log_ratio,
        ratio,
        influence,
    ]

    for method in METHOD_ORDER:
        sample = samples[method]
        if sample is None:
            fit_x = real_x
            fit_y = real_y
            fit_weight = None
            accepted_prevalence = None
            accepted_count = None
            accept_rate = None
            saturation_rate = None
            weight_diagnostics = {
                "final_weight_ess": None,
                "final_weight_relative_ess": None,
                "final_weight_p95": None,
                "final_weight_p99": None,
                "final_weight_maximum": None,
            }
            # Covariance diagnostic describes the accepted, weighted synthetic
            # contribution. Real-only has no synthetic block and therefore no Vw.
            method_covariance_diagnostic = None
        else:
            accepted_count = int(sample["y"].size)
            if accepted_count == 0:
                empty_selection_count += 1
                raise RuntimeError(f"{method} accepted no candidates.")
            fit_x = np.vstack([real_x, sample["x"]])
            fit_y = np.concatenate([real_y, sample["y"]])
            fit_weight = np.concatenate(
                [np.ones(real_y.size, dtype=float), sample["weight"]]
            )
            accepted_prevalence = float(np.mean(sample["y"]))
            accept_rate = float(sample["accept_rate"])
            saturation_rate = float(np.mean(sample["pi"] >= 1.0 - 1e-12))
            weight_diagnostics = logistic_metrics.final_weight_summary(
                sample["weight"]
            )
            accepted_psi = compute_score_psi(sample["x"], sample["y"], pilot)
            vw = estimate_vw(accepted_psi, sample["weight"])
            method_covariance_diagnostic = covariance_diagnostic(vp_inverse, vw)
            nonfinite_arrays.extend(
                [sample["pi"], sample["weight"], accepted_psi, vw]
            )

        fitted = fit_weighted_logistic(
            fit_x,
            fit_y,
            fit_weight,
            config["solver"],
        )
        model_bundle[method] = fitted
        solver_failure_count += int(not fitted["converged"])
        probabilities = predict_probability(fitted, test_x)
        evaluated = logistic_metrics.evaluate_probabilities(
            test_y,
            probabilities,
            true_test_probability,
        )
        parameter_error = logistic_metrics.parameter_squared_error(
            fitted["theta"], beta_true
        )
        metric_rows.append(
            {
                "rep": int(rep),
                "method": method,
                "paired_data_fingerprint": paired_data_fingerprint,
                **evaluated,
                "parameter_squared_error": parameter_error,
                "covariance_diagnostic": method_covariance_diagnostic,
                "accepted_positive_prevalence": accepted_prevalence,
                "accepted_count": accepted_count,
                "accept_rate": accept_rate,
                "acceptance_saturation_rate": saturation_rate,
                **weight_diagnostics,
                "model_converged": bool(fitted["converged"]),
                "model_status": fitted["status"],
                "model_iterations": int(fitted["iterations"]),
                "model_gradient_norm": float(fitted["gradient_norm"]),
                "model_hessian_condition_number": float(
                    fitted["hessian_condition_number"]
                ),
            }
        )
        coefficient_rows.append(
            {
                "rep": int(rep),
                "method": method,
                "theta": np.asarray(fitted["theta"], dtype=float).copy(),
            }
        )
        nonfinite_arrays.extend([fitted["theta"], probabilities])

    ratio_diagnostics = logistic_metrics.ratio_summary(ratio)
    raw_eigenvalues = np.linalg.eigvalsh(0.5 * (vp + vp.T))
    diagnostic_row: dict[str, Any] = {
        "rep": int(rep),
        "paired_data_fingerprint": paired_data_fingerprint,
        "real_positive_prevalence": float(np.mean(real_y)),
        "proposal_positive_prevalence": float(np.mean(candidate_y)),
        **ratio_diagnostics,
        "ratio_lower_clipping_count": int(
            np.count_nonzero(raw_log_ratio < -float(config["ratio"]["clip_log_ratio"]))
        ),
        "ratio_upper_clipping_count": int(
            np.count_nonzero(raw_log_ratio > float(config["ratio"]["clip_log_ratio"]))
        ),
        "vp_condition_number": float(np.linalg.cond(vp)),
        "vp_minimum_eigenvalue": float(np.min(raw_eigenvalues)),
        "vp_ridge": vp_ridge,
        "pilot_converged": bool(pilot["converged"]),
        "pilot_status": pilot["status"],
        "pilot_iterations": int(pilot["iterations"]),
        "pilot_final_gradient_norm": float(pilot["gradient_norm"]),
        "pilot_hessian_condition_number": float(pilot["hessian_condition_number"]),
    }
    for row in metric_rows:
        key = METHOD_KEYS[row["method"]]
        for field in [
            "accepted_positive_prevalence",
            "accepted_count",
            "accept_rate",
            "acceptance_saturation_rate",
            "final_weight_ess",
            "final_weight_relative_ess",
            "final_weight_p95",
            "final_weight_p99",
            "final_weight_maximum",
            "model_converged",
            "model_status",
            "model_iterations",
            "model_gradient_norm",
            "model_hessian_condition_number",
            "fitted_probability_below_1e-6_fraction",
            "fitted_probability_above_1_minus_1e-6_fraction",
        ]:
            diagnostic_row[f"{key}_{field}"] = row[field]

    nan_count = 0
    inf_count = 0
    for values in nonfinite_arrays:
        numeric = np.asarray(values, dtype=float)
        nan_count += int(np.count_nonzero(np.isnan(numeric)))
        inf_count += int(np.count_nonzero(np.isinf(numeric)))
    diagnostic_row.update(
        {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "solver_failure_count": solver_failure_count,
            "empty_selection_count": empty_selection_count,
            "runtime_seconds": float(time.perf_counter() - start_time),
        }
    )
    return {
        "metric_rows": metric_rows,
        "coefficient_rows": coefficient_rows,
        "diagnostic_row": diagnostic_row,
    }


def _paired_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordinary_comparisons = [
        "Density-ratio-only acceptance",
        "Uniform acceptance",
        "Real-only",
    ]
    ordinary_metrics = [
        "test_log_loss",
        "brier_score",
        "probability_mse",
        "parameter_squared_error",
        "auroc",
        "auprc",
        "classification_error",
    ]
    covariance_comparisons = [
        "Density-ratio-only acceptance",
        "Uniform acceptance",
        "Influence-only acceptance",
    ]
    output: list[dict[str, Any]] = []
    repetitions = sorted({int(row["rep"]) for row in metric_rows})
    for rep in repetitions:
        rep_rows = {
            row["method"]: row for row in metric_rows if int(row["rep"]) == rep
        }
        real_syn = rep_rows["REAL-Syn"]
        for baseline in ordinary_comparisons:
            for metric in ordinary_metrics:
                output.append(
                    {
                        "rep": rep,
                        "contrast": f"REAL-Syn minus {baseline}",
                        "baseline": baseline,
                        "metric": metric,
                        "difference": float(real_syn[metric] - rep_rows[baseline][metric]),
                        "favorable_direction": (
                            "negative_is_better"
                            if metric
                            in {
                                "test_log_loss",
                                "brier_score",
                                "probability_mse",
                                "parameter_squared_error",
                                "covariance_diagnostic",
                            }
                            else "positive_is_better"
                        ),
                    }
                )
        for baseline in covariance_comparisons:
            output.append(
                {
                    "rep": rep,
                    "contrast": f"REAL-Syn minus {baseline}",
                    "baseline": baseline,
                    "metric": "covariance_diagnostic",
                    "difference": float(
                        real_syn["covariance_diagnostic"]
                        - rep_rows[baseline]["covariance_diagnostic"]
                    ),
                    "favorable_direction": "negative_is_better",
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_coefficients(
    path: Path,
    coefficient_rows: list[dict[str, Any]],
    dimension: int,
) -> None:
    rows: list[dict[str, Any]] = []
    for record in coefficient_rows:
        theta = np.asarray(record["theta"], dtype=float)
        row: dict[str, Any] = {
            "rep": int(record["rep"]),
            "method": record["method"],
            "intercept": float(theta[0]),
        }
        for index in range(dimension):
            row[f"beta_{index + 1}"] = float(theta[index + 1])
        rows.append(row)
    _write_csv(path, rows)


def _summary(
    metric_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    metrics = [
        "test_log_loss",
        "brier_score",
        "probability_mse",
        "auroc",
        "auprc",
        "classification_error",
        "parameter_squared_error",
        "covariance_diagnostic",
    ]
    coefficients = logistic_metrics.coefficient_summary(
        coefficient_rows,
        np.asarray(config["true_logistic"]["beta"], dtype=float),
    )
    methods: dict[str, Any] = {}
    for method in METHOD_ORDER:
        selected = [row for row in metric_rows if row["method"] == method]
        method_summary = {}
        for metric in metrics:
            values = [
                float(row[metric])
                for row in selected
                if row[metric] is not None
            ]
            method_summary[metric] = (
                logistic_metrics.mean_sd_se_ci(values)
                if values
                else {
                    "n": 0,
                    "mean": None,
                    "sd": None,
                    "se": None,
                    "ci_low": None,
                    "ci_high": None,
                }
            )
        method_summary.update(coefficients[method])
        methods[method] = method_summary
    return {
        "arm": config["arm"],
        "planned_repetitions": int(config["repetitions"]),
        "executed_repetitions": len({int(row["rep"]) for row in metric_rows}),
        "methods": methods,
    }
