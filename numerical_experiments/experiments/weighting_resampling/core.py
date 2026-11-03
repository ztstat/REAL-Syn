from __future__ import annotations
import time
from typing import Any
import numpy as np
from common import importance as importance
from .methods import (clipped_snis, effective_sample_size, indices_fingerprint, normalize_probabilities, residual_resample, stratified_resample, subsampled_snis, uniform_subset)

METHODS = (
    "Real-only",
    "Uniform",
    "DRE-only",
    "Influence-only",
    "REAL-Syn",
    "Multinomial-SIR-600",
    "Residual-SIR-600",
    "Stratified-SIR-600",
    "Subsampled-SNIS-600",
    "Clipped-SNIS-600",
)

SELECTION_METHODS = (
    "Uniform",
    "DRE-only",
    "Influence-only",
    "REAL-Syn",
)

SIR_METHODS = (
    "Multinomial-SIR-600",
    "Residual-SIR-600",
    "Stratified-SIR-600",
)

SIR_COVARIANCE_DEFINITION = (
    "draw_level_score_second_moment:"
    "(1/B)*sum_i multiplicity_i*psi_i*psi_i^T"
)

SNIS_METHODS = ("Subsampled-SNIS-600", "Clipped-SNIS-600")

METRICS = (
    "test_mse",
    "noiseless_prediction_mse",
    "parameter_squared_error",
)

METHOD_ALIASES = {
    "Real-only": "Real-only",
    "Uniform": "Uniform-IW",
    "DRE-only": "DRE-only",
    "REAL-Syn": "REAL-Syn",
    "Multinomial-SIR-600": "SIR-600",
}

N_REAL = 40

N_CANDIDATES = 6000

N_TEST = 8000

DIMENSION = 5

PARAMETER_DIMENSION = 6

ALPHA = 0.1

BUDGET = 600

SEED_MODULUS = 2**32 - 1

def stream_seed(rep_seed: int, offset: int) -> int:
    return int((int(rep_seed) + int(offset)) % SEED_MODULUS)


def importance_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "distribution": "gaussian_linear",
        "alpha": ALPHA,
        "n_candidates": N_CANDIDATES,
        "synthetic_budget": BUDGET,
        "seeds": {
            "base_seed": 0,
            "rep_seed_stride": 1337,
            "sir_resampling_offset": int(
                config["seeds"]["multinomial_sir_offset"]
            ),
        },
    }


def fixed_test_data(
    config: dict[str, Any], source: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    return importance.fixed_test_data(importance_config(config), source)


def prepare_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    prepared = importance.prepare_repetition(
        importance_config(config), source, rep, test_x, test_y
    )
    rep_seed = int(source["seed"]) + int(
        source["pipeline"]["rep_seed_stride"]
    ) * rep
    seeds = config["seeds"]
    method_seeds = {
        "Real-only": rep_seed,
        "Uniform": importance.original_pipeline.stream_seed(
            rep_seed, "sample_uniform"
        ),
        "DRE-only": importance.original_pipeline.stream_seed(
            rep_seed, "sample_ratio_only"
        ),
        "Influence-only": importance.original_pipeline.stream_seed(
            rep_seed, "sample_influence_only"
        ),
        "REAL-Syn": importance.original_pipeline.stream_seed(
            rep_seed, "sample_theory"
        ),
        "Multinomial-SIR-600": stream_seed(
            rep_seed, int(seeds["multinomial_sir_offset"])
        ),
        "Residual-SIR-600": stream_seed(
            rep_seed, int(seeds["residual_sir_offset"])
        ),
        "Stratified-SIR-600": stream_seed(
            rep_seed, int(seeds["stratified_sir_offset"])
        ),
        "Subsampled-SNIS-600": stream_seed(
            rep_seed, int(seeds["snis_subset_offset"])
        ),
        "Clipped-SNIS-600": stream_seed(
            rep_seed, int(seeds["snis_subset_offset"])
        ),
    }
    subset_indices = uniform_subset(
        N_CANDIDATES,
        BUDGET,
        np.random.RandomState(method_seeds["Subsampled-SNIS-600"]),
    )
    prepared["method_seeds"] = method_seeds
    prepared["snis_subset_indices"] = subset_indices
    prepared["snis_subset_fingerprint"] = indices_fingerprint(subset_indices)
    prepared["pilot_state"] = importance.original_model.fit_reference_model(
        prepared["real_x"], prepared["real_y"], prepared["fit_config"]
    )
    prepared["vp_inverse"] = importance.original_math.invert_positive_matrix(
        prepared["vp"], float(source["pipeline"]["vp_ridge"])
    )
    return prepared


def _base_sample() -> dict[str, Any]:
    return {
        "candidate_indices": None,
        "synthetic_x": None,
        "synthetic_y": None,
        "weights": None,
        "pi": None,
        "downstream_draws": 0,
        "unique_rows": 0,
        "accepted_count": None,
        "maximum_multiplicity": None,
        "multiplicity_ess": None,
        "subset_ratio_ess": None,
        "raw_subset_ess": None,
        "clipped_subset_ess": None,
        "subset_ratio_mean": None,
        "subset_ratio_maximum": None,
        "subset_ratio_p99": None,
        "subset_ratio_p99p9": None,
        "maximum_clipped_ratio": None,
        "clipping_threshold": None,
        "clipped_count": None,
        "clipped_fraction": None,
        "subset_fingerprint": None,
        "multiplicities": None,
        "raw_subset_ratios": None,
        "clipped_subset_ratios": None,
    }


def _adapt_sample(method: str, sample: dict[str, Any]) -> dict[str, Any]:
    result = _base_sample()
    result.update(
        {
            "candidate_indices": sample["candidate_indices"],
            "synthetic_x": sample["synthetic_x"],
            "synthetic_y": sample["synthetic_y"],
            "weights": sample["weights"],
            "pi": sample["pi"],
        }
    )
    if method == "Real-only":
        return result
    indices = np.asarray(sample["candidate_indices"], dtype=np.int64)
    if method in SELECTION_METHODS:
        result.update(
            {
                "downstream_draws": int(indices.size),
                "unique_rows": int(indices.size),
                "accepted_count": int(indices.size),
            }
        )
    elif method == "Multinomial-SIR-600":
        multiplicities = np.zeros(N_CANDIDATES, dtype=np.int64)
        multiplicities[indices] = np.asarray(sample["weights"], dtype=np.int64)
        result.update(
            {
                "downstream_draws": BUDGET,
                "unique_rows": int(indices.size),
                "maximum_multiplicity": int(np.max(multiplicities)),
                "multiplicity_ess": effective_sample_size(
                    multiplicities[indices].astype(float)
                ),
                "multiplicities": multiplicities,
            }
        )
    return result


def build_method_sample(
    method: str,
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if method in METHOD_ALIASES:
        base_sample = importance.build_method_sample(
            METHOD_ALIASES[method], prepared, importance_config(config)
        )
        return _adapt_sample(method, base_sample)

    ratio = np.asarray(prepared["ratio"], dtype=float)
    candidate_x = prepared["candidate_x"]
    candidate_y = prepared["candidate_y"]
    result = _base_sample()
    if method == "Influence-only":
        pi, _ = importance.calibrate_pi(
            prepared["influence"], ALPHA, "gaussian_linear"
        )
        rng = np.random.RandomState(prepared["method_seeds"][method])
        indices = np.flatnonzero(rng.random_sample(N_CANDIDATES) < pi)
        if indices.size == 0:
            raise RuntimeError("Influence-only produced an empty selection.")
        weights = ALPHA * ratio[indices] / np.clip(
            pi[indices], 1e-12, 1.0
        )
        result.update(
            {
                "candidate_indices": indices,
                "synthetic_x": candidate_x[indices],
                "synthetic_y": candidate_y[indices],
                "weights": weights,
                "pi": pi,
                "downstream_draws": int(indices.size),
                "unique_rows": int(indices.size),
                "accepted_count": int(indices.size),
            }
        )
        return result

    probabilities = normalize_probabilities(ratio)
    if method == "Residual-SIR-600":
        resampled = residual_resample(
            probabilities,
            BUDGET,
            np.random.RandomState(prepared["method_seeds"][method]),
        )
    elif method == "Stratified-SIR-600":
        resampled = stratified_resample(
            probabilities,
            BUDGET,
            np.random.RandomState(prepared["method_seeds"][method]),
        )
    else:
        resampled = None
    if resampled is not None:
        indices = resampled["indices"]
        result.update(
            {
                "candidate_indices": indices,
                "synthetic_x": candidate_x[indices],
                "synthetic_y": candidate_y[indices],
                "weights": resampled["weights"],
                "downstream_draws": BUDGET,
                "unique_rows": int(indices.size),
                "maximum_multiplicity": resampled[
                    "maximum_multiplicity"
                ],
                "multiplicity_ess": resampled["multiplicity_ess"],
                "multiplicities": resampled["multiplicities"],
            }
        )
        return result

    subset = prepared["snis_subset_indices"]
    if method == "Subsampled-SNIS-600":
        snis = subsampled_snis(ratio, subset, BUDGET)
    elif method == "Clipped-SNIS-600":
        snis = clipped_snis(ratio, subset, BUDGET)
    else:
        raise ValueError(f"Unknown method: {method}")
    indices = snis["indices"]
    result.update(
        {
            "candidate_indices": indices,
            "synthetic_x": candidate_x[indices],
            "synthetic_y": candidate_y[indices],
            "weights": snis["weights"],
            "downstream_draws": BUDGET,
            "unique_rows": BUDGET,
            "subset_ratio_ess": snis["subset_ratio_ess"],
            "raw_subset_ess": snis["raw_ess"],
            "clipped_subset_ess": snis["clipped_ess"],
            "subset_ratio_mean": snis["subset_ratio_mean"],
            "subset_ratio_maximum": snis["subset_ratio_maximum"],
            "subset_ratio_p99": snis["subset_ratio_p99"],
            "subset_ratio_p99p9": snis["subset_ratio_p99p9"],
            "maximum_clipped_ratio": snis["maximum_clipped_ratio"],
            "clipping_threshold": snis["clipping_threshold"],
            "clipped_count": snis["clipped_count"],
            "clipped_fraction": snis["clipped_fraction"],
            "subset_fingerprint": snis["subset_fingerprint"],
            "raw_subset_ratios": snis["raw_ratios"],
            "clipped_subset_ratios": snis["clipped_ratios"],
        }
    )
    return result


def fit_coefficients(
    prepared: dict[str, Any], sample: dict[str, Any]
) -> np.ndarray:
    return importance.fit_coefficients(
        prepared,
        sample["synthetic_x"],
        sample["synthetic_y"],
        sample["weights"],
    )


def evaluate_coefficients(
    coefficients: np.ndarray, prepared: dict[str, Any]
) -> dict[str, float]:
    return importance.evaluate_coefficients(coefficients, prepared)


def covariance_diagnostic(
    prepared: dict[str, Any], sample: dict[str, Any], method: str
) -> float | None:
    if method == "Real-only":
        return None
    psi_dimension = int(prepared["vp"].shape[0])
    if method in SIR_METHODS:
        psi = importance.original_model.compute_psi(
            sample["synthetic_x"],
            sample["synthetic_y"],
            prepared["pilot_state"],
            prepared["fit_config"],
        )
        vw = sir_draw_level_vw_from_psi(
            psi,
            np.asarray(sample["weights"], dtype=float),
            BUDGET,
        )
    else:
        vw = importance.original_math.estimate_vw_from_accepted(
            accepted_x=sample["synthetic_x"],
            accepted_y=sample["synthetic_y"],
            accepted_weight=sample["weights"],
            reference_model=prepared["pilot_state"],
            config=prepared["fit_config"],
            psi_dim=psi_dimension,
            compute_psi_fn=importance.original_model.compute_psi,
        )
    value = float(
        np.trace(prepared["vp_inverse"] @ vw) / psi_dimension
    )
    if not np.isfinite(value):
        raise RuntimeError(f"{method} covariance diagnostic is nonfinite.")
    return value


def sir_draw_level_vw_from_psi(
    psi: np.ndarray,
    multiplicities: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Compute the draw-level score second moment for compressed SIR samples.

    This is equivalent to averaging score outer products over the
    uncompressed draws. For residual and stratified resampling,
    it is a descriptive diagnostic rather than an independent-draw
    variance estimate.
    """
    values = np.asarray(psi, dtype=float)
    counts = np.asarray(multiplicities, dtype=float)
    if values.ndim != 2 or counts.shape != (values.shape[0],):
        raise ValueError("SIR psi rows and multiplicities are misaligned.")
    if (
        int(budget) <= 0
        or np.any(~np.isfinite(values))
        or np.any(~np.isfinite(counts))
        or np.any(counts <= 0.0)
        or np.any(counts != np.floor(counts))
        or float(np.sum(counts)) != float(budget)
    ):
        raise ValueError("SIR multiplicities must be positive integers summing to B.")
    return (values.T @ (counts[:, None] * values)) / float(budget)


def run_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
    method_order: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    prepared = prepare_repetition(config, source, rep, test_x, test_y)
    run_label = str(config["run_label"])
    order = METHODS if method_order is None else method_order
    if set(order) != set(METHODS) or len(order) != len(METHODS):
        raise ValueError("Method order must contain each method once.")
    ratio_ess = effective_sample_size(prepared["ratio"])
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    samples: dict[str, dict[str, Any]] = {}
    truth = importance.true_coefficients(source, "gaussian_linear")

    for method in order:
        total_start = time.perf_counter()
        selection_start = time.perf_counter()
        sample = build_method_sample(method, prepared, config)
        selection_seconds = time.perf_counter() - selection_start
        fitting_start = time.perf_counter()
        coefficients = fit_coefficients(prepared, sample)
        fitting_seconds = time.perf_counter() - fitting_start
        total_seconds = time.perf_counter() - total_start
        if coefficients.shape != (PARAMETER_DIMENSION,) or np.any(
            ~np.isfinite(coefficients)
        ):
            raise RuntimeError(f"{method} produced invalid coefficients.")
        metrics = evaluate_coefficients(coefficients, prepared)
        if any(not np.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"{method} produced a nonfinite metric.")
        covariance_value = covariance_diagnostic(prepared, sample, method)
        weights = sample["weights"]
        if weights is None:
            total_weight = final_ess = None
        else:
            weights = np.asarray(weights, dtype=float)
            if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
                raise RuntimeError(f"{method} produced invalid weights.")
            total_weight = float(np.sum(weights))
            final_ess = effective_sample_size(weights)
        error = coefficients - truth
        row = {
            "run_label": run_label,
            "experiment_id": "weighting_resampling",
            "rep": rep,
            "rep_seed": prepared["rep_seed"],
            "master_seed": prepared["rep_seed"],
            "method_substream_seed": prepared["method_seeds"][method],
            "method": method,
            "n_real": N_REAL,
            "M_cand": N_CANDIDATES,
            "B": BUDGET,
            "alpha": ALPHA,
            "candidate_generation_count": (
                0 if method == "Real-only" else N_CANDIDATES
            ),
            "downstream_draws": sample["downstream_draws"],
            "downstream_unique_synthetic_rows": sample["unique_rows"],
            "realized_accepted_count": sample["accepted_count"],
            "synthetic_total_weight": total_weight,
            "candidate_ratio_ess": ratio_ess,
            "candidate_ratio_relative_ess": ratio_ess / N_CANDIDATES,
            "subset_ratio_ess": sample["subset_ratio_ess"],
            "raw_subset_ratio_ess": sample["raw_subset_ess"],
            "clipped_subset_ratio_ess": sample["clipped_subset_ess"],
            "final_weight_or_multiplicity_ess": final_ess,
            "subset_ratio_mean": sample["subset_ratio_mean"],
            "subset_ratio_maximum": sample["subset_ratio_maximum"],
            "subset_ratio_p99": sample["subset_ratio_p99"],
            "subset_ratio_p99p9": sample["subset_ratio_p99p9"],
            "maximum_clipped_ratio": sample["maximum_clipped_ratio"],
            "clipping_threshold": sample["clipping_threshold"],
            "clipping_activated": (
                None
                if sample["clipped_count"] is None
                else bool(sample["clipped_count"] > 0)
            ),
            "clipped_count": sample["clipped_count"],
            "clipped_fraction": sample["clipped_fraction"],
            "maximum_multiplicity": sample["maximum_multiplicity"],
            "test_mse": metrics["test_mse"],
            "noiseless_prediction_mse": metrics[
                "noiseless_prediction_mse"
            ],
            "parameter_squared_error": metrics["parameter_squared_error"],
            "slope_parameter_squared_error": metrics[
                "slope_parameter_squared_error"
            ],
            "intercept_squared_error": metrics["intercept_squared_error"],
            **{
                f"theta_{index}": float(coefficients[index])
                for index in range(PARAMETER_DIMENSION)
            },
            **{
                f"squared_error_theta_{index}": float(error[index] ** 2)
                for index in range(PARAMETER_DIMENSION)
            },
            "covariance_diagnostic": covariance_value,
            "selection_or_resampling_seconds": selection_seconds,
            "selection_method_seconds": selection_seconds,
            "downstream_fitting_seconds": fitting_seconds,
            "total_runtime_seconds": total_seconds,
            "subset_fingerprint": sample["subset_fingerprint"],
            "selected_candidate_indices_fingerprint": (
                None
                if sample["candidate_indices"] is None
                else indices_fingerprint(
                    np.asarray(sample["candidate_indices"], dtype=np.int64)
                )
            ),
            **prepared["fingerprints"],
            "finite_value_failure": 0,
            "solver_failure": 0,
            "warning_flag": 0,
        }
        rows.append(row)
        parameter_rows.append(
            {
                "run_label": run_label,
                "experiment_id": "weighting_resampling",
                "rep": rep,
                "rep_seed": prepared["rep_seed"],
                "method": method,
                **{
                    f"theta_{index}": float(coefficients[index])
                    for index in range(PARAMETER_DIMENSION)
                },
            }
        )
        samples[method] = sample

    rows.sort(key=lambda row: METHODS.index(str(row["method"])))
    parameter_rows.sort(
        key=lambda row: METHODS.index(str(row["method"]))
    )
    snis_subset_error = int(
        not np.array_equal(
            samples["Subsampled-SNIS-600"]["candidate_indices"],
            samples["Clipped-SNIS-600"]["candidate_indices"],
        )
    )
    sanity_row = {
        "run_label": run_label,
        "experiment_id": "weighting_resampling",
        "rep": rep,
        "rep_seed": prepared["rep_seed"],
        "shared_snis_subset_error": snis_subset_error,
        "snis_subset_unique_count_error": abs(
            int(
                np.unique(
                    samples["Subsampled-SNIS-600"]["candidate_indices"]
                ).size
            )
            - BUDGET
        ),
        "maximum_sir_draw_count_error": max(
            abs(float(np.sum(samples[method]["weights"])) - float(BUDGET))
            for method in SIR_METHODS
        ),
        "maximum_snis_total_weight_error": max(
            abs(float(np.sum(samples[method]["weights"])) - BUDGET)
            for method in SNIS_METHODS
        ),
        "minimum_selection_count": min(
            int(samples[method]["accepted_count"])
            for method in SELECTION_METHODS
        ),
        "empty_selection_count": sum(
            int(samples[method]["accepted_count"] == 0)
            for method in SELECTION_METHODS
        ),
        "candidate_data_fingerprint": prepared["fingerprints"][
            "candidate_data_fingerprint"
        ],
        "test_data_fingerprint": prepared["fingerprints"][
            "test_data_fingerprint"
        ],
    }
    ratio_row = {
        "run_label": run_label,
        "experiment_id": "weighting_resampling",
        "rep": rep,
        "rep_seed": prepared["rep_seed"],
        "candidate_count": N_CANDIDATES,
        "mean_ratio": float(np.mean(prepared["ratio"])),
        "ratio_ess": ratio_ess,
        "ratio_relative_ess": ratio_ess / N_CANDIDATES,
        "ratio_p99": float(np.quantile(prepared["ratio"], 0.99)),
        "ratio_p99p9": float(np.quantile(prepared["ratio"], 0.999)),
        "maximum_ratio": float(np.max(prepared["ratio"])),
        "lower_log_ratio_clipping_count": int(
            np.count_nonzero(
                prepared["raw_log_ratio"]
                < -float(source["ratio"]["clip_log_ratio"])
            )
        ),
        "upper_log_ratio_clipping_count": int(
            np.count_nonzero(
                prepared["raw_log_ratio"]
                > float(source["ratio"]["clip_log_ratio"])
            )
        ),
        **prepared["fingerprints"],
    }
    return {
        "per_rep_rows": rows,
        "parameter_rows": parameter_rows,
        "ratio_row": ratio_row,
        "sanity_row": sanity_row,
    }


def seed_manifest(
    config: dict[str, Any], source: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "run_label": config["run_label"],
        "experiment_id": "weighting_resampling",
        "base_seed": source["seed"],
        "rep_seed_stride": source["pipeline"]["rep_seed_stride"],
        "fixed_test_seed": importance.original_pipeline.stream_seed(
            int(source["seed"]), "fixed_data"
        ),
        "repetitions": [],
    }
    for rep in range(repetitions):
        rep_seed = int(source["seed"]) + int(
            source["pipeline"]["rep_seed_stride"]
        ) * rep
        prepared_seeds = {
            "real_data": importance.original_pipeline.stream_seed(
                rep_seed, "real_data"
            ),
            "candidate_data": importance.original_pipeline.stream_seed(
                rep_seed, "candidate"
            ),
            "uniform_acceptance": importance.original_pipeline.stream_seed(
                rep_seed, "sample_uniform"
            ),
            "dre_acceptance": importance.original_pipeline.stream_seed(
                rep_seed, "sample_ratio_only"
            ),
            "influence_acceptance": importance.original_pipeline.stream_seed(
                rep_seed, "sample_influence_only"
            ),
            "real_syn_acceptance": importance.original_pipeline.stream_seed(
                rep_seed, "sample_theory"
            ),
            "multinomial_sir": stream_seed(
                rep_seed, int(config["seeds"]["multinomial_sir_offset"])
            ),
            "residual_sir": stream_seed(
                rep_seed, int(config["seeds"]["residual_sir_offset"])
            ),
            "stratified_sir": stream_seed(
                rep_seed, int(config["seeds"]["stratified_sir_offset"])
            ),
            "snis_subset": stream_seed(
                rep_seed, int(config["seeds"]["snis_subset_offset"])
            ),
        }
        output["repetitions"].append(
            {"rep": rep, "master_seed": rep_seed, "streams": prepared_seeds}
        )
    return output
