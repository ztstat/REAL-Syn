from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from common import pipeline_math as original_math
from common import ols_pipeline as original_pipeline
from common import ols_model as original_model

METHODS = (
    "Real-only",
    "Uniform-IW",
    "DRE-only",
    "REAL-Syn",
    "All-IW-full",
    "All-IW-matched",
    "SNIS-matched",
    "Sqrt-clipped-SNIS",
    "ESS-clipped-SNIS",
    "SIR-600",
)

SELECTION_METHODS = ("Uniform-IW", "DRE-only", "REAL-Syn")

FULL_POOL_METHODS = (
    "All-IW-full",
    "All-IW-matched",
    "SNIS-matched",
    "Sqrt-clipped-SNIS",
    "ESS-clipped-SNIS",
)

EQUAL_ROW_METHODS = (*SELECTION_METHODS, "SIR-600")

METRICS = (
    "test_mse",
    "noiseless_prediction_mse",
    "parameter_squared_error",
)

ALPHA = 0.1

M = 6000

BUDGET = 600

ESS_TARGET = 600

def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def stream_rng(rep_seed: int, offset: int) -> np.random.RandomState:
    return np.random.RandomState((int(rep_seed) + int(offset)) % (2**32 - 1))


def true_coefficients(source: dict[str, Any], distribution: str) -> np.ndarray:
    section = source["data"] if distribution == "gaussian_linear" else source["target"]
    return np.asarray(
        [float(section["intercept"]), *np.asarray(section["beta"], dtype=float)]
    )


def target_section(source: dict[str, Any], distribution: str) -> dict[str, Any]:
    return source["data"] if distribution == "gaussian_linear" else source["target"]


def fixed_test_data(
    config: dict[str, Any], source: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    if config["distribution"] == "gaussian_linear":
        fixed_config = original_pipeline.copy_config_with_seed(
            source,
            original_pipeline.stream_seed(int(source["seed"]), "fixed_data"),
        )
        fixed_data = original_pipeline.load_misspecified_real_data(fixed_config)
        return fixed_data["test_x"], fixed_data["test_y"]
    raise ValueError("Only the Gaussian linear distribution is supported.")


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or weights.size == 0
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("ESS weights must be finite, nonnegative, and nonempty.")
    total = float(np.sum(weights))
    sum_squares = float(np.sum(weights**2))
    if total <= 0.0 or sum_squares <= 0.0:
        raise ValueError("ESS weights must have positive mass.")
    return total**2 / sum_squares


def calibrate_pi(
    score: np.ndarray, alpha: float, distribution: str
) -> tuple[np.ndarray, float]:
    if distribution == "gaussian_linear":
        pi, constant = original_pipeline.compute_calibrated_pi(
            score=np.asarray(score, dtype=float), alpha=float(alpha)
        )
    else:
        raise ValueError("Unsupported distribution.")
    pi = np.asarray(pi, dtype=float)
    if (
        np.any(~np.isfinite(pi))
        or np.any(pi < 0.0)
        or np.any(pi > 1.0)
    ):
        raise RuntimeError("Invalid calibrated inclusion probabilities.")
    if not np.isclose(float(np.mean(pi)), alpha, rtol=0.0, atol=2e-12):
        raise RuntimeError("Mean inclusion probability missed alpha.")
    return pi, float(constant)


def ess_clipping(
    ratio: np.ndarray, target: float
) -> dict[str, Any]:
    ratio = np.asarray(ratio, dtype=float)
    raw_ess = effective_sample_size(ratio)
    if raw_ess >= float(target):
        return {
            "weights": ratio.copy(),
            "threshold": None,
            "threshold_state": "infinity_no_clipping",
            "activated": False,
            "clipped_count": 0,
            "ess_before": raw_ess,
            "ess_after": raw_ess,
            "largest_feasible_check_ess": raw_ess,
        }
    low = float(np.min(ratio))
    high = float(np.max(ratio))
    for _ in range(100):
        middle = 0.5 * (low + high)
        middle_ess = effective_sample_size(np.minimum(ratio, middle))
        if middle_ess >= float(target):
            low = middle
        else:
            high = middle
    threshold = low
    clipped = np.minimum(ratio, threshold)
    after = effective_sample_size(clipped)
    next_threshold = min(
        float(np.max(ratio)),
        np.nextafter(threshold, float("inf")) * (1.0 + 1e-10),
    )
    next_ess = effective_sample_size(np.minimum(ratio, next_threshold))
    if after < float(target) - 1e-8:
        raise RuntimeError("ESS clipping failed to reach the target.")
    return {
        "weights": clipped,
        "threshold": threshold,
        "threshold_state": "finite_clipping_threshold",
        "activated": True,
        "clipped_count": int(np.count_nonzero(ratio > threshold)),
        "ess_before": raw_ess,
        "ess_after": after,
        "largest_feasible_check_ess": next_ess,
    }


def _prepare_gaussian(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    rep_seed = int(source["seed"]) + int(
        source["pipeline"]["rep_seed_stride"]
    ) * rep
    rep_config = original_pipeline.copy_config_with_seed(
        source, original_pipeline.stream_seed(rep_seed, "real_data")
    )
    rep_data = original_pipeline.load_misspecified_real_data(rep_config)
    real_x, real_y = rep_data["train_x"], rep_data["train_y"]
    pilot_state = original_model.fit_reference_model(real_x, real_y, rep_config)
    pilot = np.asarray(pilot_state["beta_full"], dtype=float)
    vp = original_math.estimate_vp(
        real_x,
        real_y,
        pilot_state,
        rep_config,
        compute_psi_fn=original_model.compute_psi,
    )
    vp_inv = original_math.invert_positive_matrix(
        vp, float(source["pipeline"]["vp_ridge"])
    )
    generator = original_pipeline.fit_misspecified_generator(
        real_x, real_y, rep_config
    )
    ratio_state = original_pipeline.fit_misspecified_ratio_model(
        real_x,
        real_y,
        generator,
        np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "ratio_model")
        ),
        rep_config,
    )
    candidate_x, candidate_y = (
        original_pipeline.generate_misspecified_candidates(
            generator,
            int(source["pipeline"]["n_candidates"]),
            np.random.RandomState(
                original_pipeline.stream_seed(rep_seed, "candidate")
            ),
            rep_config,
        )
    )
    theory = original_math.compute_theory_statistics(
        x=candidate_x,
        y=candidate_y,
        reference_model=pilot_state,
        ratio_state=ratio_state,
        vp_inv=vp_inv,
        config=rep_config,
        compute_log_ratio_fn=original_pipeline.compute_misspecified_log_ratio,
        compute_psi_fn=original_model.compute_psi,
    )
    ratio = np.asarray(theory["ratio"], dtype=float)
    score = np.asarray(theory["score"], dtype=float)
    influence = score / ratio
    raw_log_ratio = (
        original_pipeline.linear_gaussian_joint_logpdf(
            candidate_x,
            candidate_y,
            rep_config["data"],
            int(rep_config["data"]["x_dim"]),
        )
        - original_pipeline.linear_gaussian_joint_logpdf(
            candidate_x,
            candidate_y,
            rep_config["synthetic_candidate"],
            int(rep_config["data"]["x_dim"]),
        )
    )
    clip = float(rep_config["ratio"]["clip_log_ratio"])
    method_rngs = {
        "Uniform-IW": np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "sample_uniform")
        ),
        "DRE-only": np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "sample_ratio_only")
        ),
        "REAL-Syn": np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "sample_theory")
        ),
        "SIR-600": stream_rng(
            rep_seed, int(config["seeds"]["sir_resampling_offset"])
        ),
    }
    return {
        "distribution": "gaussian_linear",
        "rep": rep,
        "rep_seed": rep_seed,
        "source": source,
        "fit_config": rep_config,
        "real_x": real_x,
        "real_y": real_y,
        "candidate_x": candidate_x,
        "candidate_y": candidate_y,
        "ratio": ratio,
        "score": score,
        "influence": influence,
        "pilot": pilot,
        "vp": vp,
        "test_x": test_x,
        "test_y": test_y,
        "raw_log_ratio": raw_log_ratio,
        "clipped_log_ratio": np.clip(raw_log_ratio, -clip, clip),
        "clip_lower": -clip,
        "clip_upper": clip,
        "method_rngs": method_rngs,
    }


def prepare_repetition(
    config: dict[str, Any],
    source: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    if config["distribution"] == "gaussian_linear":
        prepared = _prepare_gaussian(config, source, rep, test_x, test_y)
    else:
        raise ValueError("Unsupported distribution.")
    if prepared["ratio"].size != M:
        raise RuntimeError("Candidate pool is not exactly 6000.")
    arrays = (
        prepared["ratio"],
        prepared["score"],
        prepared["influence"],
        prepared["pilot"],
    )
    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise RuntimeError("Prepared repetition contains a nonfinite value.")
    prepared["fingerprints"] = {
        "real_data_fingerprint": array_fingerprint(
            prepared["real_x"], prepared["real_y"]
        ),
        "candidate_data_fingerprint": array_fingerprint(
            prepared["candidate_x"], prepared["candidate_y"]
        ),
        "ratio_fingerprint": array_fingerprint(prepared["ratio"]),
        "pilot_fingerprint": array_fingerprint(prepared["pilot"]),
        "influence_fingerprint": array_fingerprint(prepared["influence"]),
        "test_data_fingerprint": array_fingerprint(test_x, test_y),
    }
    return prepared


def fit_coefficients(
    prepared: dict[str, Any],
    synthetic_x: np.ndarray | None,
    synthetic_y: np.ndarray | None,
    synthetic_weights: np.ndarray | None,
) -> np.ndarray:
    if synthetic_x is None:
        return np.asarray(prepared["pilot"], dtype=float).copy()
    synthetic_weights = np.asarray(synthetic_weights, dtype=float)
    if (
        np.any(~np.isfinite(synthetic_weights))
        or np.any(synthetic_weights < 0.0)
        or float(np.sum(synthetic_weights)) <= 0.0
    ):
        raise ValueError("Invalid synthetic training weights.")
    if prepared["distribution"] == "gaussian_linear":
        model = original_math.fit_augmented_model(
            real_x=prepared["real_x"],
            real_y=prepared["real_y"],
            synth_x=np.asarray(synthetic_x),
            synth_y=np.asarray(synthetic_y),
            synth_weight=synthetic_weights,
            config=prepared["fit_config"],
            fit_final_model_fn=original_model.fit_final_model,
        )
        return np.asarray(model["beta_full"], dtype=float)
    raise ValueError("Unsupported distribution.")


def evaluate_coefficients(
    coefficients: np.ndarray, prepared: dict[str, Any]
) -> dict[str, float]:
    coefficients = np.asarray(coefficients, dtype=float)
    design = np.column_stack(
        [np.ones(prepared["test_x"].shape[0]), prepared["test_x"]]
    )
    prediction = design @ coefficients
    target = target_section(prepared["source"], prepared["distribution"])
    true_mean = float(target["intercept"]) + prepared["test_x"] @ np.asarray(
        target["beta"], dtype=float
    )
    truth = true_coefficients(prepared["source"], prepared["distribution"])
    error = coefficients - truth
    return {
        "test_mse": float(np.mean((prediction - prepared["test_y"]) ** 2)),
        "noiseless_prediction_mse": float(
            np.mean((prediction - true_mean) ** 2)
        ),
        "parameter_squared_error": float(error @ error),
        "slope_parameter_squared_error": float(error[1:] @ error[1:]),
        "intercept_squared_error": float(error[0] ** 2),
    }


def _empty_method_fields() -> dict[str, Any]:
    return {
        "candidate_indices": None,
        "synthetic_x": None,
        "synthetic_y": None,
        "weights": None,
        "synthetic_rows_used": None,
        "unique_synthetic_rows_used": None,
        "pi": None,
        "calibration_constant": None,
        "clipping_threshold": None,
        "clipping_threshold_state": None,
        "clipping_activated": None,
        "clipped_count": None,
        "ess_before_clipping": None,
        "ess_after_clipping": None,
        "largest_feasible_check_ess": None,
        "sir_total_draws": None,
        "sir_unique_candidates": None,
        "sir_maximum_multiplicity": None,
        "sir_multiplicity_ess": None,
        "sir_expanded_aggregate_coefficient_difference": None,
    }


def build_method_sample(
    method: str,
    prepared: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    result = _empty_method_fields()
    ratio = prepared["ratio"]
    candidate_x = prepared["candidate_x"]
    candidate_y = prepared["candidate_y"]
    alpha = float(config["alpha"])
    budget = int(config["synthetic_budget"])
    if method == "Real-only":
        return result
    if method in SELECTION_METHODS:
        if method == "Uniform-IW":
            pi = np.full(M, alpha)
            constant = alpha
        elif method == "DRE-only":
            pi, constant = calibrate_pi(ratio, alpha, prepared["distribution"])
        else:
            pi, constant = calibrate_pi(
                prepared["score"], alpha, prepared["distribution"]
            )
        uniforms = prepared["method_rngs"][method].random_sample(M)
        mask = uniforms < pi
        if not np.any(mask):
            raise RuntimeError(f"{method} produced an empty selection.")
        indices = np.flatnonzero(mask)
        weights = alpha * ratio[indices] / np.clip(pi[indices], 1e-12, 1.0)
        result.update(
            {
                "candidate_indices": indices,
                "synthetic_x": candidate_x[indices],
                "synthetic_y": candidate_y[indices],
                "weights": weights,
                "synthetic_rows_used": int(indices.size),
                "unique_synthetic_rows_used": int(indices.size),
                "pi": pi,
                "calibration_constant": float(constant),
            }
        )
        return result
    if method == "All-IW-full":
        weights = ratio.copy()
    elif method == "All-IW-matched":
        weights = alpha * ratio
    elif method == "SNIS-matched":
        weights = budget * ratio / float(np.sum(ratio))
    elif method == "Sqrt-clipped-SNIS":
        threshold = float(np.mean(ratio) * np.sqrt(M))
        clipped = np.minimum(ratio, threshold)
        weights = budget * clipped / float(np.sum(clipped))
        result.update(
            {
                "clipping_threshold": threshold,
                "clipping_threshold_state": "finite_prespecified_sqrt_rule",
                "clipping_activated": bool(np.any(ratio > threshold)),
                "clipped_count": int(np.count_nonzero(ratio > threshold)),
                "ess_before_clipping": effective_sample_size(ratio),
                "ess_after_clipping": effective_sample_size(clipped),
            }
        )
    elif method == "ESS-clipped-SNIS":
        clipped_result = ess_clipping(ratio, float(config["ess_target"]))
        clipped = clipped_result["weights"]
        weights = budget * clipped / float(np.sum(clipped))
        result.update(
            {
                "clipping_threshold": clipped_result["threshold"],
                "clipping_threshold_state": clipped_result["threshold_state"],
                "clipping_activated": clipped_result["activated"],
                "clipped_count": clipped_result["clipped_count"],
                "ess_before_clipping": clipped_result["ess_before"],
                "ess_after_clipping": clipped_result["ess_after"],
                "largest_feasible_check_ess": clipped_result[
                    "largest_feasible_check_ess"
                ],
            }
        )
    elif method == "SIR-600":
        probabilities = ratio / float(np.sum(ratio))
        if (
            np.any(~np.isfinite(probabilities))
            or np.any(probabilities < 0.0)
            or not np.isclose(
                float(np.sum(probabilities)), 1.0, rtol=0.0, atol=2e-14
            )
        ):
            raise RuntimeError("Invalid SIR probabilities.")
        draws = prepared["method_rngs"][method].choice(
            M, size=budget, replace=True, p=probabilities
        )
        multiplicities = np.bincount(draws, minlength=M)
        indices = np.flatnonzero(multiplicities)
        weights = multiplicities[indices].astype(float)
        aggregate_coefficients = fit_coefficients(
            prepared, candidate_x[indices], candidate_y[indices], weights
        )
        expanded_coefficients = fit_coefficients(
            prepared,
            candidate_x[draws],
            candidate_y[draws],
            np.ones(budget),
        )
        multiplicity_ess = budget**2 / float(np.sum(multiplicities**2))
        result.update(
            {
                "candidate_indices": indices,
                "synthetic_x": candidate_x[indices],
                "synthetic_y": candidate_y[indices],
                "weights": weights,
                "synthetic_rows_used": budget,
                "unique_synthetic_rows_used": int(indices.size),
                "sir_total_draws": budget,
                "sir_unique_candidates": int(indices.size),
                "sir_maximum_multiplicity": int(np.max(multiplicities)),
                "sir_multiplicity_ess": multiplicity_ess,
                "sir_expanded_aggregate_coefficient_difference": float(
                    np.max(
                        np.abs(
                            aggregate_coefficients - expanded_coefficients
                        )
                    )
                ),
            }
        )
        return result
    else:
        raise ValueError(f"Unknown method: {method}")
    indices = np.arange(M)
    result.update(
        {
            "candidate_indices": indices,
            "synthetic_x": candidate_x,
            "synthetic_y": candidate_y,
            "weights": np.asarray(weights, dtype=float),
            "synthetic_rows_used": M,
            "unique_synthetic_rows_used": M,
        }
    )
    return result
