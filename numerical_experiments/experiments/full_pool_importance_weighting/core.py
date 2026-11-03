from __future__ import annotations
import hashlib
from typing import Any
import numpy as np
from common import pipeline_math as original_math
from common import ols_pipeline as original_pipeline
from common import ols_model as original_model

ALPHAS = (0.02, 0.05, 0.10, 0.20, 0.50, 1.00)

METHODS = (
    "Real-only",
    "Uniform-IW",
    "DRE-only",
    "REAL-Syn",
    "All-IW-matched",
    "All-IW-full",
)

SELECTED_METHODS = ("Uniform-IW", "DRE-only", "REAL-Syn")

PAIRED_BASELINES = ("Uniform-IW", "DRE-only", "All-IW-matched")

PARAMETER_METRICS = (
    "test_mse",
    "noiseless_prediction_mse",
    "parameter_squared_error",
)

def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def true_coefficients(source_config: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            float(source_config["data"]["intercept"]),
            *np.asarray(source_config["data"]["beta"], dtype=float),
        ]
    )


def fixed_test_data(
    source_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    base_seed = int(source_config["seed"])
    fixed_config = original_pipeline.copy_config_with_seed(
        source_config,
        original_pipeline.stream_seed(base_seed, "fixed_data"),
    )
    fixed_data = original_pipeline.load_misspecified_real_data(fixed_config)
    return fixed_data["test_x"], fixed_data["test_y"]


def seed_manifest(
    source_config: dict[str, Any], repetitions: int
) -> dict[str, Any]:
    base_seed = int(source_config["seed"])
    stride = int(source_config["pipeline"]["rep_seed_stride"])
    streams = original_pipeline.SEED_STREAM_OFFSETS
    manifest: dict[str, Any] = {
        "base_seed": base_seed,
        "rep_seed_stride": stride,
        "fixed_test_stream_seed": original_pipeline.stream_seed(
            base_seed, "fixed_data"
        ),
        "stream_offsets": streams,
        "repetitions": [],
    }
    for rep in range(repetitions):
        rep_seed = base_seed + stride * rep
        manifest["repetitions"].append(
            {
                "rep": rep,
                "rep_seed": rep_seed,
                "streams": {
                    name: original_pipeline.stream_seed(rep_seed, name)
                    for name in streams
                },
            }
        )
    return manifest


def effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=float)
    if weights.size == 0:
        raise ValueError("Weight ESS requires a nonempty vector.")
    total = float(np.sum(weights))
    sum_squares = float(np.sum(weights**2))
    if not np.isfinite(total) or sum_squares <= 0.0:
        raise ValueError("Weight ESS inputs are invalid.")
    ess = total**2 / sum_squares
    return ess, ess / weights.size


def calibrated_pi(score: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    pi, constant = original_pipeline.compute_calibrated_pi(
        score=np.asarray(score, dtype=float),
        alpha=float(alpha),
    )
    pi = np.asarray(pi, dtype=float)
    if np.any(~np.isfinite(pi)) or np.any(pi < 0.0) or np.any(pi > 1.0):
        raise RuntimeError("Calibrated inclusion probabilities are invalid.")
    if not np.isclose(np.mean(pi), alpha, rtol=0.0, atol=2e-12):
        raise RuntimeError("Calibrated mean inclusion probability missed alpha.")
    return pi, float(constant)


def sample_from_uniforms(
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    ratio: np.ndarray,
    pi: np.ndarray,
    alpha: float,
    uniforms: np.ndarray,
) -> dict[str, np.ndarray]:
    mask = np.asarray(uniforms) < np.asarray(pi)
    if not np.any(mask):
        raise RuntimeError("A selected method produced an empty sample.")
    accepted_pi = np.clip(np.asarray(pi)[mask], 1e-12, 1.0)
    weights = float(alpha) * np.asarray(ratio)[mask] / accepted_pi
    return {
        "x": np.asarray(candidate_x)[mask],
        "y": np.asarray(candidate_y)[mask],
        "weights": weights,
        "mask": mask,
    }


def fit_augmented(
    real_x: np.ndarray,
    real_y: np.ndarray,
    synthetic_x: np.ndarray,
    synthetic_y: np.ndarray,
    synthetic_weights: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    model = original_math.fit_augmented_model(
        real_x=real_x,
        real_y=real_y,
        synth_x=synthetic_x,
        synth_y=synthetic_y,
        synth_weight=synthetic_weights,
        config=config,
        fit_final_model_fn=original_model.fit_final_model,
    )
    return np.asarray(model["beta_full"], dtype=float)


def evaluate_coefficients(
    coefficients: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    source_config: dict[str, Any],
) -> dict[str, float]:
    coefficients = np.asarray(coefficients, dtype=float)
    design = np.column_stack([np.ones(test_x.shape[0]), test_x])
    predictions = design @ coefficients
    target = source_config["data"]
    true_mean = float(target["intercept"]) + test_x @ np.asarray(
        target["beta"], dtype=float
    )
    truth = true_coefficients(source_config)
    error = coefficients - truth
    return {
        "test_mse": float(np.mean((predictions - test_y) ** 2)),
        "noiseless_prediction_mse": float(
            np.mean((predictions - true_mean) ** 2)
        ),
        "parameter_squared_error": float(error @ error),
        "slope_parameter_squared_error": float(error[1:] @ error[1:]),
        "intercept_squared_error": float(error[0] ** 2),
    }


def _method_rows(
    *,
    rep: int,
    alpha: float,
    method: str,
    coefficients: np.ndarray,
    metrics: dict[str, float],
    ratio_diagnostics: dict[str, float | int | str],
    candidate_count: int,
    accepted_count: int,
    pi: np.ndarray | None,
    weights: np.ndarray | None,
    calibration_constant: float | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if weights is None:
        total_weight = None
        weight_ess = None
        relative_weight_ess = None
        maximum_weight = None
    else:
        weight_ess, relative_weight_ess = effective_sample_size(weights)
        total_weight = float(np.sum(weights))
        maximum_weight = float(np.max(weights))
    if pi is None:
        mean_pi = None
        saturation_fraction = None
    else:
        mean_pi = float(np.mean(pi))
        saturation_fraction = float(np.mean(pi >= 1.0 - 1e-12))
    metric_row = {
        "rep": rep,
        "method": method,
        "alpha": alpha,
        "candidate_pool_size": candidate_count,
        **metrics,
        "candidate_ratio_ess": ratio_diagnostics["ratio_ess"],
        "candidate_ratio_relative_ess": ratio_diagnostics[
            "ratio_relative_ess"
        ],
    }
    parameter_row = {
        "rep": rep,
        "method": method,
        "alpha": alpha,
        **{
            f"theta_{index}": float(value)
            for index, value in enumerate(coefficients)
        },
    }
    diagnostic_row = {
        "rep": rep,
        "method": method,
        "alpha": alpha,
        "candidate_pool_size": candidate_count,
        "actual_accepted_count": accepted_count,
        "acceptance_rate": accepted_count / candidate_count,
        "mean_inclusion_probability": mean_pi,
        "saturation_fraction": saturation_fraction,
        "calibration_constant": calibration_constant,
        "synthetic_total_weight": total_weight,
        "synthetic_weight_ess": weight_ess,
        "relative_weight_ess": relative_weight_ess,
        "maximum_synthetic_weight": maximum_weight,
        "candidate_ratio_ess": ratio_diagnostics["ratio_ess"],
        "candidate_ratio_relative_ess": ratio_diagnostics[
            "ratio_relative_ess"
        ],
        "candidate_maximum_ratio": ratio_diagnostics["maximum_ratio"],
        "lower_clipping_count": ratio_diagnostics[
            "lower_clipping_count"
        ],
        "upper_clipping_count": ratio_diagnostics[
            "upper_clipping_count"
        ],
    }
    return metric_row, parameter_row, diagnostic_row


def run_repetition(
    source_config: dict[str, Any],
    rep: int,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, Any]:
    base_seed = int(source_config["seed"])
    stride = int(source_config["pipeline"]["rep_seed_stride"])
    rep_seed = base_seed + stride * rep
    rep_config = original_pipeline.copy_config_with_seed(
        source_config,
        original_pipeline.stream_seed(rep_seed, "real_data"),
    )
    rep_data = original_pipeline.load_misspecified_real_data(rep_config)
    real_x = rep_data["train_x"]
    real_y = rep_data["train_y"]
    reference_model = original_model.fit_reference_model(
        real_x, real_y, rep_config
    )
    vp = original_math.estimate_vp(
        real_x,
        real_y,
        reference_model,
        rep_config,
        compute_psi_fn=original_model.compute_psi,
    )
    vp_inv = original_math.invert_positive_matrix(
        vp, float(source_config["pipeline"]["vp_ridge"])
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
    candidate_x, candidate_y = original_pipeline.generate_misspecified_candidates(
        generator,
        int(source_config["pipeline"]["n_candidates"]),
        np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "candidate")
        ),
        rep_config,
    )
    theory = original_math.compute_theory_statistics(
        x=candidate_x,
        y=candidate_y,
        reference_model=reference_model,
        ratio_state=ratio_state,
        vp_inv=vp_inv,
        config=rep_config,
        compute_log_ratio_fn=original_pipeline.compute_misspecified_log_ratio,
        compute_psi_fn=original_model.compute_psi,
    )
    ratio = np.asarray(theory["ratio"], dtype=float)
    score = np.asarray(theory["score"], dtype=float)
    candidate_count = ratio.size
    if candidate_count != 6000:
        raise RuntimeError("Candidate pool is not fixed at 6000.")
    ratio_ess, ratio_relative_ess = effective_sample_size(ratio)
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
    clip_value = float(rep_config["ratio"]["clip_log_ratio"])
    ratio_diagnostics: dict[str, float | int | str] = {
        "rep": rep,
        "ratio_ess": ratio_ess,
        "ratio_relative_ess": ratio_relative_ess,
        "mean_ratio": float(np.mean(ratio)),
        "maximum_ratio": float(np.max(ratio)),
        "lower_clipping_count": int(
            np.count_nonzero(raw_log_ratio < -clip_value)
        ),
        "upper_clipping_count": int(
            np.count_nonzero(raw_log_ratio > clip_value)
        ),
        "real_data_fingerprint": array_fingerprint(real_x, real_y),
        "candidate_data_fingerprint": array_fingerprint(
            candidate_x, candidate_y
        ),
        "ratio_fingerprint": array_fingerprint(ratio),
        "test_data_fingerprint": array_fingerprint(test_x, test_y),
    }
    uniforms = {
        "Uniform-IW": np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "sample_uniform")
        ).rand(candidate_count),
        "DRE-only": np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "sample_ratio_only")
        ).rand(candidate_count),
        "REAL-Syn": np.random.RandomState(
            original_pipeline.stream_seed(rep_seed, "sample_theory")
        ).rand(candidate_count),
    }

    real_coefficients = np.asarray(
        reference_model["beta_full"], dtype=float
    )
    all_full_coefficients = fit_augmented(
        real_x,
        real_y,
        candidate_x,
        candidate_y,
        ratio,
        rep_config,
    )
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    outputs: dict[tuple[float, str], dict[str, Any]] = {}
    previous_pi: dict[str, np.ndarray] = {}
    previous_count: dict[str, int] = {}
    minimum_pi_increment = float("inf")
    minimum_count_increment = float("inf")
    maximum_mean_pi_error = 0.0

    for alpha in ALPHAS:
        dre_pi, dre_constant = calibrated_pi(ratio, alpha)
        real_syn_pi, real_syn_constant = calibrated_pi(score, alpha)
        pi_values = {
            "Uniform-IW": np.full(candidate_count, alpha, dtype=float),
            "DRE-only": dre_pi,
            "REAL-Syn": real_syn_pi,
        }
        constants: dict[str, float | None] = {
            "Uniform-IW": alpha,
            "DRE-only": (
                dre_constant if np.isfinite(dre_constant) else None
            ),
            "REAL-Syn": (
                real_syn_constant
                if np.isfinite(real_syn_constant)
                else None
            ),
        }
        for method in SELECTED_METHODS:
            pi = pi_values[method]
            maximum_mean_pi_error = max(
                maximum_mean_pi_error, abs(float(np.mean(pi)) - alpha)
            )
            if method in previous_pi:
                minimum_pi_increment = min(
                    minimum_pi_increment,
                    float(np.min(pi - previous_pi[method])),
                )
                if np.any(pi < previous_pi[method] - 2e-12):
                    raise RuntimeError(
                        f"{method} pi is not nondecreasing in alpha."
                    )
            previous_pi[method] = pi
            sample = sample_from_uniforms(
                candidate_x,
                candidate_y,
                ratio,
                pi,
                alpha,
                uniforms[method],
            )
            count = int(np.count_nonzero(sample["mask"]))
            if method in previous_count:
                increment = count - previous_count[method]
                minimum_count_increment = min(
                    minimum_count_increment, float(increment)
                )
                if increment < 0:
                    raise RuntimeError(
                        f"{method} accepted count decreased in alpha."
                    )
            previous_count[method] = count
            coefficients = fit_augmented(
                real_x,
                real_y,
                sample["x"],
                sample["y"],
                sample["weights"],
                rep_config,
            )
            outputs[(alpha, method)] = {
                "coefficients": coefficients,
                "weights": sample["weights"],
                "mask": sample["mask"],
                "pi": pi,
            }

        all_matched_weights = alpha * ratio
        all_matched_coefficients = fit_augmented(
            real_x,
            real_y,
            candidate_x,
            candidate_y,
            all_matched_weights,
            rep_config,
        )
        outputs[(alpha, "Real-only")] = {
            "coefficients": real_coefficients,
            "weights": None,
            "mask": None,
            "pi": None,
        }
        outputs[(alpha, "All-IW-matched")] = {
            "coefficients": all_matched_coefficients,
            "weights": all_matched_weights,
            "mask": np.ones(candidate_count, dtype=bool),
            "pi": None,
        }
        outputs[(alpha, "All-IW-full")] = {
            "coefficients": all_full_coefficients,
            "weights": ratio,
            "mask": np.ones(candidate_count, dtype=bool),
            "pi": None,
        }

        for method in METHODS:
            result = outputs[(alpha, method)]
            coefficients = result["coefficients"]
            metrics = evaluate_coefficients(
                coefficients, test_x, test_y, rep_config
            )
            if method == "Real-only":
                accepted_count = 0
                pi = None
                weights = None
                constant = None
            elif method in SELECTED_METHODS:
                accepted_count = int(np.count_nonzero(result["mask"]))
                pi = result["pi"]
                weights = result["weights"]
                constant = constants[method]
            else:
                accepted_count = candidate_count
                pi = None
                weights = result["weights"]
                constant = None
            rows = _method_rows(
                rep=rep,
                alpha=alpha,
                method=method,
                coefficients=coefficients,
                metrics=metrics,
                ratio_diagnostics=ratio_diagnostics,
                candidate_count=candidate_count,
                accepted_count=accepted_count,
                pi=pi,
                weights=weights,
                calibration_constant=constant,
            )
            metric_rows.append(rows[0])
            parameter_rows.append(rows[1])
            diagnostic_rows.append(rows[2])

    endpoint_methods = (
        "Uniform-IW",
        "DRE-only",
        "REAL-Syn",
        "All-IW-matched",
        "All-IW-full",
    )
    endpoint_reference = outputs[(1.0, "All-IW-full")]
    alpha_one_coefficient_difference = 0.0
    alpha_one_weight_difference = 0.0
    for method in endpoint_methods:
        current = outputs[(1.0, method)]
        alpha_one_coefficient_difference = max(
            alpha_one_coefficient_difference,
            float(
                np.max(
                    np.abs(
                        current["coefficients"]
                        - endpoint_reference["coefficients"]
                    )
                )
            ),
        )
        alpha_one_weight_difference = max(
            alpha_one_weight_difference,
            float(
                np.max(
                    np.abs(
                        current["weights"] - endpoint_reference["weights"]
                    )
                )
            ),
        )
        if int(np.count_nonzero(current["mask"])) != candidate_count:
            raise RuntimeError(f"{method} did not accept all rows at alpha=1.")

    alpha_point = 0.10
    reference_specs = {
        "Uniform-IW": (
            np.full(candidate_count, alpha_point),
            "sample_uniform",
        ),
        "DRE-only": (
            calibrated_pi(ratio, alpha_point)[0],
            "sample_ratio_only",
        ),
        "REAL-Syn": (
            calibrated_pi(score, alpha_point)[0],
            "sample_theory",
        ),
    }
    alpha_point_coefficient_difference = 0.0
    alpha_point_mask_mismatch_count = 0
    for method, (pi, stream) in reference_specs.items():
        reference_sample = original_math.sample_with_pi(
            x=candidate_x,
            y=candidate_y,
            pi=pi,
            ratio=ratio,
            alpha=alpha_point,
            rng=np.random.RandomState(
                original_pipeline.stream_seed(rep_seed, stream)
            ),
        )
        current = outputs[(alpha_point, method)]
        alpha_point_mask_mismatch_count += int(
            np.count_nonzero(
                reference_sample["accepted_mask"] != current["mask"]
            )
        )
        reference_coefficients = fit_augmented(
            real_x,
            real_y,
            reference_sample["x"],
            reference_sample["y"],
            reference_sample["weight"],
            rep_config,
        )
        alpha_point_coefficient_difference = max(
            alpha_point_coefficient_difference,
            float(
                np.max(
                    np.abs(
                        reference_coefficients - current["coefficients"]
                    )
                )
            ),
        )

    sanity = {
        "rep": rep,
        "candidate_pool_size": candidate_count,
        "maximum_mean_pi_error": maximum_mean_pi_error,
        "minimum_pi_increment": minimum_pi_increment,
        "minimum_accepted_count_increment": minimum_count_increment,
        "alpha_one_maximum_weight_difference": (
            alpha_one_weight_difference
        ),
        "alpha_one_maximum_coefficient_difference": (
            alpha_one_coefficient_difference
        ),
        "alpha_0p1_mask_mismatch_count": (
            alpha_point_mask_mismatch_count
        ),
        "alpha_0p1_maximum_coefficient_difference": (
            alpha_point_coefficient_difference
        ),
    }
    return {
        "metric_rows": metric_rows,
        "parameter_rows": parameter_rows,
        "diagnostic_rows": diagnostic_rows,
        "ratio_row": ratio_diagnostics,
        "sanity": sanity,
    }


def target_design_second_moment(
    source_config: dict[str, Any],
) -> np.ndarray:
    target = source_config["data"]
    mean = np.asarray(target["x_mean"], dtype=float)
    std = np.asarray(target["x_std"], dtype=float)
    second = np.diag(std**2) + np.outer(mean, mean)
    return np.block(
        [
            [np.ones((1, 1)), mean.reshape(1, -1)],
            [mean.reshape(-1, 1), second],
        ]
    )
