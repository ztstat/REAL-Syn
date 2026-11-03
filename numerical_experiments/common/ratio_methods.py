from __future__ import annotations
from typing import Any
import numpy as np
from common import pipeline_math as original_math
from common import ols_pipeline as original_pipeline
from common import ols_model as original_model

METHODS = ("Real-only", "Uniform-IW", "DRE-only", "REAL-Syn", "All-IW-matched")
SELECTION_METHODS = ("Uniform-IW", "DRE-only", "REAL-Syn")
N_CANDIDATES = 6000
ALPHA = 0.1

def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or weights.size == 0
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError("ESS input must be finite, nonnegative, and nonempty.")
    total = float(np.sum(weights))
    sum_squares = float(np.sum(weights**2))
    if total <= 0.0 or sum_squares <= 0.0:
        raise ValueError("ESS input must have positive mass.")
    return total**2 / sum_squares


def oracle_ratio_state(
    candidate_x: np.ndarray,
    candidate_y: np.ndarray,
    source: dict[str, Any],
) -> dict[str, Any]:
    dimension = int(source["data"]["x_dim"])
    raw_log_ratio = original_pipeline.linear_gaussian_joint_logpdf(
        candidate_x, candidate_y, source["data"], dimension
    ) - original_pipeline.linear_gaussian_joint_logpdf(
        candidate_x,
        candidate_y,
        source["synthetic_candidate"],
        dimension,
    )
    clip = float(source["ratio"]["clip_log_ratio"])
    log_ratio = np.clip(raw_log_ratio, -clip, clip)
    return {
        "raw_log_ratio": raw_log_ratio,
        "log_ratio": log_ratio,
        "ratio": np.exp(log_ratio),
        "target_shrinkage": None,
        "proposal_shrinkage": None,
        "target_covariance_minimum_eigenvalue": None,
        "proposal_covariance_minimum_eigenvalue": None,
        "lower_clipping_count": int(np.count_nonzero(raw_log_ratio < -clip)),
        "upper_clipping_count": int(np.count_nonzero(raw_log_ratio > clip)),
        "nonfinite_count": 0,
        "feature_space": "joint_XY_known",
        "uses_y": True,
        "standardization_training_source": None,
        "standardization_mean": None,
        "standardization_standard_deviation": None,
    }


def fixed_test_data(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    fixed_config = original_pipeline.copy_config_with_seed(
        source,
        original_pipeline.stream_seed(int(source["seed"]), "fixed_data"),
    )
    data = original_pipeline.load_misspecified_real_data(fixed_config)
    return data["test_x"], data["test_y"]


def calibrate_pi(score: np.ndarray) -> tuple[np.ndarray, float]:
    pi, constant = original_pipeline.compute_calibrated_pi(score, ALPHA)
    pi = np.asarray(pi, dtype=float)
    if np.any(~np.isfinite(pi)) or np.any((pi < 0.0) | (pi > 1.0)):
        raise RuntimeError("Invalid inclusion probabilities.")
    if not np.isclose(float(np.mean(pi)), ALPHA, rtol=0.0, atol=2e-12):
        raise RuntimeError("Mean inclusion probability missed alpha.")
    return pi, float(constant)


def build_method_sample(
    method: str,
    ratio: np.ndarray,
    influence: np.ndarray,
    prepared: dict[str, Any],
) -> dict[str, Any]:
    if method == "Real-only":
        return {
            "x": None,
            "y": None,
            "weights": None,
            "indices": None,
            "pi": None,
            "calibration_constant": None,
        }
    if method == "Uniform-IW":
        pi = np.full(N_CANDIDATES, ALPHA)
        constant = ALPHA
    elif method == "DRE-only":
        pi, constant = calibrate_pi(ratio)
    elif method == "REAL-Syn":
        pi, constant = calibrate_pi(ratio * influence)
    elif method == "All-IW-matched":
        return {
            "x": prepared["candidate_x"],
            "y": prepared["candidate_y"],
            "weights": ALPHA * ratio,
            "indices": np.arange(N_CANDIDATES),
            "pi": None,
            "calibration_constant": None,
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    mask = prepared["base_uniforms"][method] < pi
    if not np.any(mask):
        raise RuntimeError(f"{method} produced an empty selection.")
    indices = np.flatnonzero(mask)
    weights = ALPHA * ratio[indices] / np.clip(pi[indices], 1e-12, 1.0)
    return {
        "x": prepared["candidate_x"][indices],
        "y": prepared["candidate_y"][indices],
        "weights": weights,
        "indices": indices,
        "pi": pi,
        "calibration_constant": constant,
    }


def fit_coefficients(
    prepared: dict[str, Any], sample: dict[str, Any]
) -> np.ndarray:
    if sample["x"] is None:
        return prepared["pilot"].copy()
    model = original_math.fit_augmented_model(
        real_x=prepared["real_x"],
        real_y=prepared["real_y"],
        synth_x=sample["x"],
        synth_y=sample["y"],
        synth_weight=sample["weights"],
        config=prepared["fit_config"],
        fit_final_model_fn=original_model.fit_final_model,
    )
    return np.asarray(model["beta_full"], dtype=float)


def true_coefficients(source: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [source["data"]["intercept"], *source["data"]["beta"]], dtype=float
    )


def evaluate_coefficients(
    coefficients: np.ndarray, prepared: dict[str, Any]
) -> dict[str, float]:
    test_design = np.column_stack(
        [np.ones(prepared["test_x"].shape[0]), prepared["test_x"]]
    )
    prediction = test_design @ coefficients
    target = prepared["source"]["data"]
    noiseless_target = float(target["intercept"]) + prepared["test_x"] @ np.asarray(
        target["beta"], dtype=float
    )
    error = coefficients - true_coefficients(prepared["source"])
    return {
        "test_mse": float(np.mean((prediction - prepared["test_y"]) ** 2)),
        "noiseless_prediction_mse": float(
            np.mean((prediction - noiseless_target) ** 2)
        ),
        "parameter_squared_error": float(error @ error),
    }
