from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from scipy.stats import spearmanr
from common import pipeline_math as original_math
from common import ols_pipeline as original_pipeline
from common import ols_model as original_model
from common import ratio_methods as ratio_methods

RATIO_REGIMES = (
    ("shrinkage_000", 0.00),
    ("shrinkage_005", 0.05),
    ("shrinkage_010", 0.10),
    ("shrinkage_015", 0.15),
    ("shrinkage_020", 0.20),
)

REGIME_NAMES = tuple(name for name, _ in RATIO_REGIMES)

LAMBDA_BY_REGIME = dict(RATIO_REGIMES)

METHODS = (
    "Real-only",
    "Uniform-IW",
    "DRE-only",
    "REAL-Syn",
    "All-IW-matched",
)

SELECTION_METHODS = ("Uniform-IW", "DRE-only", "REAL-Syn")

ACCEPTANCE_DIAGNOSTIC_METHODS = ("DRE-only", "REAL-Syn")

METRICS = (
    "test_mse",
    "noiseless_prediction_mse",
    "parameter_squared_error",
)

N_REAL = 40

N_CANDIDATES = 6000

N_TEST = 8000

ALPHA = 0.1

RANDOM_SEED_MODULUS = 2**32 - 1

def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def stream_seed(rep_seed: int, offset: int) -> int:
    return int((int(rep_seed) + int(offset)) % RANDOM_SEED_MODULUS)


def effective_sample_size(weights: np.ndarray) -> float:
    return ratio_methods.effective_sample_size(np.asarray(weights, dtype=float))


def correlation(first: np.ndarray, second: np.ndarray, kind: str) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if kind == "pearson":
        if np.array_equal(first, second):
            return 1.0
        return float(np.corrcoef(first, second)[0, 1])
    statistic = float(spearmanr(first, second).statistic)
    return 1.0 if abs(statistic - 1.0) <= 1e-14 else statistic


def fixed_test_data(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return ratio_methods.fixed_test_data(source)


def prepare_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    rep_seed = int(config["seeds"]["base_seed"]) + int(
        config["seeds"]["rep_seed_stride"]
    ) * int(rep)
    rep_config = original_pipeline.copy_config_with_seed(
        source, original_pipeline.stream_seed(rep_seed, "real_data")
    )
    data = original_pipeline.load_misspecified_real_data(rep_config)
    real_x, real_y = data["train_x"], data["train_y"]
    generator = original_pipeline.fit_misspecified_generator(
        real_x, real_y, rep_config
    )
    candidate_seed = original_pipeline.stream_seed(rep_seed, "candidate")
    candidate_x, candidate_y = original_pipeline.generate_misspecified_candidates(
        generator,
        N_CANDIDATES,
        np.random.RandomState(candidate_seed),
        rep_config,
    )
    pilot_state = original_model.fit_reference_model(real_x, real_y, rep_config)
    pilot = np.asarray(pilot_state["beta_full"], dtype=float)
    vp = original_math.estimate_vp(
        real_x,
        real_y,
        pilot_state,
        rep_config,
        compute_psi_fn=original_model.compute_psi,
    )
    vp_inverse = original_math.invert_positive_matrix(
        vp, float(source["pipeline"]["vp_ridge"])
    )
    influence = original_math.compute_influence_norm(
        candidate_x,
        candidate_y,
        pilot_state,
        vp_inverse,
        rep_config,
        original_model.compute_psi,
    )
    acceptance_seeds = {
        "Uniform-IW": original_pipeline.stream_seed(rep_seed, "sample_uniform"),
        "DRE-only": original_pipeline.stream_seed(rep_seed, "sample_ratio_only"),
        "REAL-Syn": original_pipeline.stream_seed(rep_seed, "sample_theory"),
    }
    base_uniforms = {
        method: np.random.RandomState(seed).random_sample(N_CANDIDATES)
        for method, seed in acceptance_seeds.items()
    }
    prepared = {
        "rep": int(rep),
        "rep_seed": rep_seed,
        "source": source,
        "fit_config": rep_config,
        "real_x": real_x,
        "real_y": real_y,
        "candidate_x": candidate_x,
        "candidate_y": candidate_y,
        "test_x": test_x,
        "test_y": test_y,
        "pilot": pilot,
        "pilot_state": pilot_state,
        "vp": vp,
        "vp_inverse": vp_inverse,
        "influence": influence,
        "base_uniforms": base_uniforms,
        "seeds": {
            "rep_seed": rep_seed,
            "real_data": original_pipeline.stream_seed(rep_seed, "real_data"),
            "candidate_generation": candidate_seed,
            "fixed_test": original_pipeline.stream_seed(
                int(source["seed"]), "fixed_data"
            ),
            **{
                f"{method}_acceptance": seed
                for method, seed in acceptance_seeds.items()
            },
        },
    }
    prepared["fingerprints"] = {
        "real_data_fingerprint": array_fingerprint(real_x, real_y),
        "candidate_data_fingerprint": array_fingerprint(
            candidate_x, candidate_y
        ),
        "test_data_fingerprint": array_fingerprint(test_x, test_y),
        "pilot_fingerprint": array_fingerprint(pilot),
        "vp_fingerprint": array_fingerprint(vp),
        "influence_fingerprint": array_fingerprint(influence),
        **{
            f"{method}_uniform_fingerprint": array_fingerprint(values)
            for method, values in base_uniforms.items()
        },
    }
    for array in (
        real_x,
        real_y,
        candidate_x,
        candidate_y,
        test_x,
        test_y,
        pilot,
        vp,
        influence,
    ):
        if np.any(~np.isfinite(array)):
            raise RuntimeError("Prepared repetition contains nonfinite data.")
    return prepared


def build_ratio_states(prepared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    oracle = ratio_methods.oracle_ratio_state(
        prepared["candidate_x"],
        prepared["candidate_y"],
        prepared["source"],
    )
    oracle_ratio = np.asarray(oracle["ratio"], dtype=float)
    states: dict[str, dict[str, Any]] = {}
    for regime, shrinkage in RATIO_REGIMES:
        if shrinkage == 0.0:
            ratio = oracle_ratio.copy()
            log_ratio = np.asarray(oracle["log_ratio"], dtype=float).copy()
        else:
            ratio = (1.0 - shrinkage) * oracle_ratio + shrinkage
            log_ratio = np.log(ratio)
        nonfinite = int(
            np.count_nonzero(~np.isfinite(ratio))
            + np.count_nonzero(~np.isfinite(log_ratio))
        )
        if nonfinite or np.any(ratio <= 0.0):
            raise RuntimeError(f"{regime} has an invalid transformed ratio.")
        states[regime] = {
            "regime": regime,
            "lambda": shrinkage,
            "oracle_ratio": oracle_ratio,
            "oracle_log_ratio": np.asarray(oracle["log_ratio"], dtype=float),
            "oracle_raw_log_ratio": np.asarray(
                oracle["raw_log_ratio"], dtype=float
            ),
            "ratio": ratio,
            "log_ratio": log_ratio,
            "inherited_oracle_lower_clipping_count": (
                oracle["lower_clipping_count"]
            ),
            "inherited_oracle_upper_clipping_count": (
                oracle["upper_clipping_count"]
            ),
            "transformed_ratio_additional_clipping_count": 0,
            "nonfinite_count": nonfinite,
        }
    if tuple(states) != REGIME_NAMES:
        raise RuntimeError("Ratio-regime registry mismatch.")
    return states


def build_method_sample(
    method: str,
    ratio: np.ndarray,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    return ratio_methods.build_method_sample(
        method, ratio, prepared["influence"], prepared
    )


def fit_coefficients(
    prepared: dict[str, Any], sample: dict[str, Any]
) -> np.ndarray:
    return ratio_methods.fit_coefficients(prepared, sample)


def evaluate_coefficients(
    coefficients: np.ndarray, prepared: dict[str, Any]
) -> dict[str, float]:
    metrics = ratio_methods.evaluate_coefficients(coefficients, prepared)
    truth = ratio_methods.true_coefficients(prepared["source"])
    error = np.asarray(coefficients, dtype=float) - truth
    metrics.update(
        {
            "slope_parameter_squared_error": float(error[1:] @ error[1:]),
            "intercept_squared_error": float(error[0] ** 2),
        }
    )
    return metrics


def ratio_diagnostic_row(
    prepared: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    ratio = state["ratio"]
    oracle_ratio = state["oracle_ratio"]
    log_error = state["log_ratio"] - state["oracle_log_ratio"]
    quantiles = np.quantile(ratio, [0.5, 0.9, 0.95, 0.99, 0.999])
    oracle_p99 = float(np.quantile(oracle_ratio, 0.99))
    oracle_maximum = float(np.max(oracle_ratio))
    return {
        "rep": prepared["rep"],
        "ratio_regime": state["regime"],
        "lambda": state["lambda"],
        "mean_oracle_ratio": float(np.mean(oracle_ratio)),
        "mean_transformed_ratio": float(np.mean(ratio)),
        "ideal_unclipped_population_mean_transformed_ratio": 1.0,
        "ratio_ess": effective_sample_size(ratio),
        "ratio_relative_ess": effective_sample_size(ratio) / N_CANDIDATES,
        "log_ratio_rmse_vs_oracle": float(
            np.sqrt(np.mean(log_error**2))
        ),
        "log_ratio_mae_vs_oracle": float(np.mean(np.abs(log_error))),
        "pearson_log_ratio_correlation_vs_oracle": correlation(
            state["log_ratio"], state["oracle_log_ratio"], "pearson"
        ),
        "spearman_ratio_correlation_vs_oracle": correlation(
            ratio, oracle_ratio, "spearman"
        ),
        "ratio_p50": float(quantiles[0]),
        "ratio_p90": float(quantiles[1]),
        "ratio_p95": float(quantiles[2]),
        "ratio_p99": float(quantiles[3]),
        "ratio_p99p9": float(quantiles[4]),
        "maximum_ratio": float(np.max(ratio)),
        "inherited_oracle_lower_clipping_count": (
            state["inherited_oracle_lower_clipping_count"]
        ),
        "inherited_oracle_upper_clipping_count": (
            state["inherited_oracle_upper_clipping_count"]
        ),
        "transformed_ratio_additional_clipping_count": (
            state["transformed_ratio_additional_clipping_count"]
        ),
        "nonfinite_count": state["nonfinite_count"],
        "p99_tail_compression_ratio": float(quantiles[3] / oracle_p99),
        "maximum_tail_compression_ratio": float(
            np.max(ratio) / oracle_maximum
        ),
        "mean_absolute_ratio_difference_vs_oracle": float(
            np.mean(np.abs(ratio - oracle_ratio))
        ),
        "mean_absolute_log_ratio_difference_vs_oracle": float(
            np.mean(np.abs(log_error))
        ),
        **prepared["fingerprints"],
    }


def _mask(indices: np.ndarray) -> np.ndarray:
    result = np.zeros(N_CANDIDATES, dtype=bool)
    result[np.asarray(indices, dtype=int)] = True
    return result


def run_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    prepared = prepare_repetition(config, source, rep, test_x, test_y)
    states = build_ratio_states(prepared)
    prepared["ratio_states"] = states
    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    samples: dict[tuple[str, str], dict[str, Any]] = {}

    real_coefficients = fit_coefficients(
        prepared,
        {
            "x": None,
            "y": None,
            "weights": None,
            "indices": None,
            "pi": None,
            "calibration_constant": None,
        },
    )
    for regime, shrinkage in RATIO_REGIMES:
        state = states[regime]
        ratio = state["ratio"]
        for method in METHODS:
            if method == "Real-only":
                sample = {
                    "x": None,
                    "y": None,
                    "weights": None,
                    "indices": None,
                    "pi": None,
                    "calibration_constant": None,
                }
                coefficients = real_coefficients.copy()
            else:
                sample = build_method_sample(method, ratio, prepared)
                coefficients = fit_coefficients(prepared, sample)
            metrics = evaluate_coefficients(coefficients, prepared)
            if np.any(~np.isfinite(coefficients)) or any(
                not np.isfinite(value) for value in metrics.values()
            ):
                raise RuntimeError(f"{regime}/{method} is nonfinite.")
            weights = sample["weights"]
            pi = sample["pi"]
            if weights is None:
                accepted_count = synthetic_rows = total_weight = None
                weight_ess = relative_weight_ess = p95 = p99 = maximum = None
            else:
                synthetic_rows = int(weights.size)
                accepted_count = (
                    synthetic_rows if method in SELECTION_METHODS else None
                )
                total_weight = float(np.sum(weights))
                weight_ess = effective_sample_size(weights)
                relative_weight_ess = weight_ess / synthetic_rows
                p95 = float(np.quantile(weights, 0.95))
                p99 = float(np.quantile(weights, 0.99))
                maximum = float(np.max(weights))
            row = {
                "rep": rep,
                "rep_seed": prepared["rep_seed"],
                "ratio_regime": regime,
                "lambda": shrinkage,
                "method": method,
                "n_real": N_REAL,
                "n_candidates": N_CANDIDATES,
                "n_test": N_TEST,
                "alpha": ALPHA,
                **metrics,
                "per_rep_squared_parameter_error": metrics[
                    "parameter_squared_error"
                ],
                **{
                    f"theta_{index}": float(value)
                    for index, value in enumerate(coefficients)
                },
                "accepted_count": accepted_count,
                "synthetic_rows_used": synthetic_rows,
                "synthetic_total_weight": total_weight,
                "final_weight_ess": weight_ess,
                "relative_final_weight_ess": relative_weight_ess,
                "p95_final_weight": p95,
                "p99_final_weight": p99,
                "maximum_final_weight": maximum,
                "mean_inclusion_probability": (
                    None if pi is None else float(np.mean(pi))
                ),
                "acceptance_calibration_constant": sample[
                    "calibration_constant"
                ],
                "saturation_fraction": (
                    None
                    if pi is None
                    else float(np.mean(pi >= 1.0 - 1e-12))
                ),
                "minimum_inclusion_probability": (
                    None if pi is None else float(np.min(pi))
                ),
                "maximum_inclusion_probability": (
                    None if pi is None else float(np.max(pi))
                ),
                "candidate_ratio_ess": effective_sample_size(ratio),
                "candidate_ratio_relative_ess": (
                    effective_sample_size(ratio) / N_CANDIDATES
                ),
                "candidate_ratio_nonfinite_count": state[
                    "nonfinite_count"
                ],
                "inherited_oracle_lower_clipping_count": state[
                    "inherited_oracle_lower_clipping_count"
                ],
                "inherited_oracle_upper_clipping_count": state[
                    "inherited_oracle_upper_clipping_count"
                ],
                "transformed_ratio_additional_clipping_count": state[
                    "transformed_ratio_additional_clipping_count"
                ],
                "coefficient_nonfinite_count": int(
                    np.count_nonzero(~np.isfinite(coefficients))
                ),
                "metric_nonfinite_count": int(
                    sum(not np.isfinite(value) for value in metrics.values())
                ),
                "solver_failure_count": 0,
                "execution_failure_count": 0,
                **prepared["fingerprints"],
            }
            rows.append(row)
            parameter_rows.append(
                {
                    "rep": rep,
                    "ratio_regime": regime,
                    "lambda": shrinkage,
                    "method": method,
                    **{
                        f"theta_{index}": float(value)
                        for index, value in enumerate(coefficients)
                    },
                }
            )
            samples[(regime, method)] = sample

    acceptance_rows: list[dict[str, Any]] = []
    for regime, shrinkage in RATIO_REGIMES:
        for method in ACCEPTANCE_DIAGNOSTIC_METHODS:
            current = samples[(regime, method)]
            oracle = samples[("shrinkage_000", method)]
            current_pi = current["pi"]
            oracle_pi = oracle["pi"]
            ratio = states[regime]["ratio"]
            oracle_ratio = states["shrinkage_000"]["ratio"]
            if method == "DRE-only":
                current_score = ratio
                oracle_score = oracle_ratio
            else:
                current_score = ratio * prepared["influence"]
                oracle_score = oracle_ratio * prepared["influence"]
            current_mask = _mask(current["indices"])
            oracle_mask = _mask(oracle["indices"])
            intersection = int(np.count_nonzero(current_mask & oracle_mask))
            union = int(np.count_nonzero(current_mask | oracle_mask))
            current_saturation = float(
                np.mean(current_pi >= 1.0 - 1e-12)
            )
            oracle_saturation = float(
                np.mean(oracle_pi >= 1.0 - 1e-12)
            )
            acceptance_rows.append(
                {
                    "rep": rep,
                    "ratio_regime": regime,
                    "lambda": shrinkage,
                    "method": method,
                    "pearson_pi_correlation_vs_oracle": correlation(
                        current_pi, oracle_pi, "pearson"
                    ),
                    "spearman_pi_correlation_vs_oracle": correlation(
                        current_pi, oracle_pi, "spearman"
                    ),
                    "pearson_acceptance_score_correlation_vs_oracle": (
                        correlation(current_score, oracle_score, "pearson")
                    ),
                    "spearman_acceptance_score_correlation_vs_oracle": (
                        correlation(current_score, oracle_score, "spearman")
                    ),
                    "mean_absolute_pi_difference_vs_oracle": float(
                        np.mean(np.abs(current_pi - oracle_pi))
                    ),
                    "maximum_absolute_pi_difference_vs_oracle": float(
                        np.max(np.abs(current_pi - oracle_pi))
                    ),
                    "saturation_fraction": current_saturation,
                    "saturation_fraction_difference_vs_oracle": (
                        current_saturation - oracle_saturation
                    ),
                    "accepted_mask_overlap_count": intersection,
                    "accepted_mask_overlap_fraction_of_oracle": (
                        intersection / max(1, int(np.count_nonzero(oracle_mask)))
                    ),
                    "accepted_mask_jaccard_similarity": intersection
                    / max(1, union),
                }
            )

    maximum_mean_pi_error = 0.0
    maximum_selected_weight_error = 0.0
    maximum_all_iw_weight_error = 0.0
    maximum_transform_identity_error = 0.0
    maximum_normalization_identity_error = 0.0
    maximum_spearman_error = 0.0
    ordering_mismatch_count = 0
    monotonicity_violation_count = 0
    empty_selection_count = 0
    oracle_ratio = states["shrinkage_000"]["ratio"]
    previous_deviation = np.abs(oracle_ratio - 1.0)
    for regime, shrinkage in RATIO_REGIMES:
        ratio = states[regime]["ratio"]
        expected_ratio = (1.0 - shrinkage) * oracle_ratio + shrinkage
        maximum_transform_identity_error = max(
            maximum_transform_identity_error,
            float(np.max(np.abs(ratio - expected_ratio))),
        )
        maximum_normalization_identity_error = max(
            maximum_normalization_identity_error,
            abs(
                float(np.mean(ratio))
                - ((1.0 - shrinkage) * float(np.mean(oracle_ratio)) + shrinkage)
            ),
        )
        deviation = np.abs(ratio - 1.0)
        monotonicity_violation_count += int(
            np.count_nonzero(deviation > previous_deviation + 1e-14)
        )
        previous_deviation = deviation
        ordering_mismatch_count += int(
            np.count_nonzero(
                np.argsort(ratio, kind="stable")
                != np.argsort(oracle_ratio, kind="stable")
            )
        )
        maximum_spearman_error = max(
            maximum_spearman_error,
            abs(correlation(ratio, oracle_ratio, "spearman") - 1.0),
        )
        for method in SELECTION_METHODS:
            sample = samples[(regime, method)]
            indices = sample["indices"]
            empty_selection_count += int(indices.size == 0)
            expected_weight = (
                ALPHA * ratio[indices] / sample["pi"][indices]
            )
            maximum_selected_weight_error = max(
                maximum_selected_weight_error,
                float(np.max(np.abs(sample["weights"] - expected_weight))),
            )
            maximum_mean_pi_error = max(
                maximum_mean_pi_error,
                abs(float(np.mean(sample["pi"])) - ALPHA),
            )
        matched = samples[(regime, "All-IW-matched")]
        maximum_all_iw_weight_error = max(
            maximum_all_iw_weight_error,
            float(np.max(np.abs(matched["weights"] - ALPHA * ratio))),
        )

    sanity = {
        "rep": rep,
        "maximum_mean_pi_error": maximum_mean_pi_error,
        "maximum_selected_weight_error": maximum_selected_weight_error,
        "maximum_all_iw_matched_weight_error": maximum_all_iw_weight_error,
        "maximum_transform_identity_error": maximum_transform_identity_error,
        "maximum_no_normalization_identity_error": (
            maximum_normalization_identity_error
        ),
        "maximum_spearman_ordering_error": maximum_spearman_error,
        "ordering_mismatch_count": ordering_mismatch_count,
        "monotonicity_violation_count": monotonicity_violation_count,
        "analytic_population_mean_one_maximum_error": 0.0,
        "transformed_ratio_additional_clipping_count": int(
            sum(
                state["transformed_ratio_additional_clipping_count"]
                for state in states.values()
            )
        ),
        "ratio_nonfinite_count": int(
            sum(state["nonfinite_count"] for state in states.values())
        ),
        "empty_selection_count": empty_selection_count,
        "regime_count": len(states),
        "method_count": len(METHODS),
        **prepared["fingerprints"],
    }
    return {
        "rows": rows,
        "parameter_rows": parameter_rows,
        "ratio_rows": [
            ratio_diagnostic_row(prepared, states[regime])
            for regime in REGIME_NAMES
        ],
        "acceptance_rows": acceptance_rows,
        "sanity_row": sanity,
        "prepared": prepared,
        "ratio_states": states,
        "samples": samples,
    }


def seed_manifest(
    config: dict[str, Any], source: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    base = int(config["seeds"]["base_seed"])
    stride = int(config["seeds"]["rep_seed_stride"])
    stream_names = (
        "real_data",
        "candidate",
        "sample_uniform",
        "sample_ratio_only",
        "sample_theory",
    )
    return {
        "base_seed": base,
        "rep_seed_stride": stride,
        "fixed_test_seed": original_pipeline.stream_seed(
            int(source["seed"]), "fixed_data"
        ),
        "repetitions": [
            {
                "rep": rep,
                "rep_seed": base + stride * rep,
                "streams": {
                    name: original_pipeline.stream_seed(
                        base + stride * rep, name
                    )
                    for name in stream_names
                },
            }
            for rep in range(repetitions)
        ],
    }
