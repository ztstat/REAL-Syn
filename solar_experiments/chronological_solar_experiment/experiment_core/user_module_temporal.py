"""
Temporal SHARPs regression module aligned with `Practical_Implement.tex`.

This module keeps the same linear-regression, DRE, influence, weighting, and
evaluation callbacks as the current SHARPs experiment. The selected artifact
cache supplies either the final frame's 20 covariates or all 1,200 coordinates
from the 60-frame history. Both views use standardized base-10 endpoint flux
as the target.

The temporal CSVs are normalized once into a compact memory-mapped cache:

- `q_prop(x, y) = tilde_q(y) q_phi(x | y)`
- `r_hat(x, y) = p(x, y) / q_prop(x, y)` via LightGBM classifier DRE
- `theta0_hat` and `psi_hat(x, y)` for the plug-in path
"""

import numpy as np

from .temporal_artifact_data import (
    inverse_condition_values,
    load_temporal_artifact_bundle,
    strip_log_prefix,
)
from .sharps_lightgbm_dre import (
    DEFAULT_LGBM_PARAMS,
    SPLIT_NAMES,
    SourceSplit,
    evaluate_dre_diagnostics,
    fit_platt_calibrator,
    predict_log_ratio,
    predict_ratio,
    prepare_split,
    split_source_arrays,
    train_lgbm_dre,
)

_ARTIFACT_BUNDLE_CACHE = {}
_RATIO_STATE_CACHE = {}
M_CLASS_TAIL_MSE_METRIC_NAME = "m_class_tail_mse"
M_CLASS_TAIL_THRESHOLD_CONFIG_KEY = "m_class_tail_flux_threshold_raw"


def clear_runtime_caches():
    """Release repetition-specific arrays and fitted models in a local run."""
    _ARTIFACT_BUNDLE_CACHE.clear()
    _RATIO_STATE_CACHE.clear()


def _freeze_cache_value(value):
    """
    Convert nested config values into hashable cache-key components.
    """
    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze_cache_value(nested_value))
            for key, nested_value in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    return value


def get_default_model_config():
    """
    Return defaults for the temporal linear-regression experiments.
    """
    lgbm_params = dict(DEFAULT_LGBM_PARAMS)
    lgbm_params["n_jobs"] = 4
    lgbm_params["min_child_samples"] = 500
    return {
        "data": {},
        "artifacts": {
            "split_label": "chronological_split",
            "temporal_cache_manifest_path": None,
            "pipeline_test_fraction": 0.20,
            "condition_normalization_mode": "log10",
        },
        "ratio": {
            "split_fractions": {
                "train": 0.72,
                "valid": 0.08,
                "calibration": 0.10,
                "test": 0.10,
            },
            "proposal_sample_ratio": 1.0,
            "add_log_y": False,
            "calibration": "platt",
            "clip_log_ratio": 20.0,
            "clip_prob": 1e-12,
            "lgbm_params": lgbm_params,
        },
        "model": {},
        "pipeline": {
            "make_xy_band_plots": False,
            "q_prop_subsample_fraction": 1.0,
        },
    }


def _validate_feature_target_shapes(x, y, x_name, y_name):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 2:
        raise ValueError(f"{x_name} must be a 2D array.")
    if y.ndim != 1:
        raise ValueError(f"{y_name} must be a 1D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x_name} and {y_name} must have the same number of rows.")

    return x, y


def _make_internal_rng(config, offset=0):
    """
    Build a deterministic RNG from the current repetition seed.

    Citation: `Practical_Implement.tex`, practical Monte Carlo steps that
    approximate expectations under `q_prop(x, y)` using a candidate pool.
    """
    return np.random.RandomState(int(config["seed"]) + int(offset))


def _resolve_pipeline_split_seed(config):
    """
    Resolve the seed used for the real-data train/test split.
    """
    if "seed" not in config:
        raise KeyError("config must define top-level 'seed' for SHARPs splits.")
    return int(config["seed"])


def _resolve_ratio_fit_seed(config):
    """
    Resolve the seed used for DRE fitting.
    """
    if "dre_seed" not in config:
        raise KeyError("config must define top-level 'dre_seed' for SHARPs DRE fitting.")
    return int(config["dre_seed"])


def _make_ratio_fit_rng(config, offset=0):
    """
    Build a deterministic RNG for DRE fitting from the current repetition seed.
    """
    return np.random.RandomState(_resolve_ratio_fit_seed(config) + int(offset))


def _resolve_q_prop_subsample_fraction(config):
    """
    Resolve the active fraction of the full synthetic proposal pool.
    """
    q_prop_subsample_fraction = float(config["pipeline"]["q_prop_subsample_fraction"])
    if not (0.0 < q_prop_subsample_fraction <= 1.0):
        raise ValueError(
            "config['pipeline']['q_prop_subsample_fraction'] must lie in (0, 1]."
        )
    return q_prop_subsample_fraction


def _deterministic_q_prop_subsample_indices(pool_size, fraction, rng):
    """
    Draw one deterministic without-replacement subset of the full q_prop pool.
    """
    pool_size = int(pool_size)
    if pool_size <= 0:
        raise ValueError("pool_size must be positive.")

    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must lie in (0, 1].")

    if fraction >= 1.0:
        return np.arange(pool_size, dtype=int)

    active_size = max(1, int(round(fraction * pool_size)))
    active_size = min(active_size, pool_size)
    return np.asarray(rng.permutation(pool_size)[:active_size], dtype=int)


def _resolve_artifact_config(config):
    """
    Resolve the artifact config for the current repetition seed.
    """
    resolved_artifact_config = dict(config["artifacts"])
    resolved_artifact_config["pipeline_split_seed"] = _resolve_pipeline_split_seed(config)
    return resolved_artifact_config


def _resolve_ratio_config(config):
    """
    Resolve the DRE config for the current repetition seed.
    """
    resolved_ratio_config = dict(config["ratio"])
    resolved_ratio_config["fit_seed"] = _resolve_ratio_fit_seed(config)
    return resolved_ratio_config


def _load_artifact_bundle(config):
    """
    Load `D_real` and the empirical artifact-backed `q_prop(x, y)` pool.

    Citation: `Practical_Implement.tex`, Algorithm input
    `D_real = {(X_i, Y_i)}_{i=1}^n` and the proposal law
    `q_prop(x, y) = tilde_q(y) q_phi(x | y)`.
    """
    artifact_config = _resolve_artifact_config(config)
    cache_key = _freeze_cache_value(artifact_config)

    if cache_key not in _ARTIFACT_BUNDLE_CACHE:
        _ARTIFACT_BUNDLE_CACHE[cache_key] = load_temporal_artifact_bundle(
            artifact_config
        )

    return _ARTIFACT_BUNDLE_CACHE[cache_key]


def _build_ratio_cache_key(config):
    """
    Build a cache key for one repetition-specific ratio model within one process.
    """
    return (
        _freeze_cache_value(_resolve_artifact_config(config)),
        _freeze_cache_value(_resolve_ratio_config(config)),
    )


def _compute_n_q_prop_samples(n_real, config):
    """
    Convert a real-split size into a Monte Carlo draw count from `q_prop`.

    Citation: `Practical_Implement.tex`, "What Needs to Be Estimated and How",
    items on calibrating `kappa` and estimating the rejection rule with a
    candidate synthetic pool `{(X_j, Y_j)}_{j=1}^M ~ q_prop(x, y)`.
    """
    n_real = int(n_real)
    if n_real <= 0:
        raise ValueError("n_real must be positive.")

    proposal_sample_ratio = float(config["ratio"]["proposal_sample_ratio"])
    if proposal_sample_ratio <= 0.0:
        raise ValueError("config['ratio']['proposal_sample_ratio'] must be positive.")

    return max(1, int(round(proposal_sample_ratio * n_real)))


def _sample_from_q_prop_pool(generator_state, n_samples, rng, replace=True, return_indices=False):
    """
    Sample empirical proposal draws from the artifact-backed `q_prop` pool.

    Citation: `Practical_Implement.tex`, Algorithm section and the Monte
    Carlo candidate pool `{(X_j, Y_j)}_{j=1}^M ~ q_prop(x, y)`.
    """
    n_samples = int(n_samples)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    q_prop_x = np.asarray(generator_state["q_prop_x"], dtype=float)
    q_prop_y = np.asarray(generator_state["q_prop_y"], dtype=float)
    q_prop_source_index = generator_state.get("q_prop_source_index")
    if q_prop_x.ndim != 2 or q_prop_y.ndim != 1:
        raise ValueError("generator_state q_prop pool has invalid shapes.")
    if q_prop_x.shape[0] != q_prop_y.shape[0]:
        raise ValueError("generator_state q_prop pool row counts must match.")
    if q_prop_x.shape[0] == 0:
        raise ValueError("generator_state q_prop pool must be non-empty.")
    if q_prop_source_index is not None:
        q_prop_source_index = np.asarray(q_prop_source_index, dtype=int)
        if q_prop_source_index.ndim != 1:
            raise ValueError("generator_state q_prop_source_index must be a 1D array.")
        if q_prop_source_index.shape[0] != q_prop_x.shape[0]:
            raise ValueError(
                "generator_state q_prop_source_index must match the q_prop pool row count."
            )

    if replace:
        indices = rng.randint(0, q_prop_x.shape[0], size=n_samples)
    else:
        if n_samples > q_prop_x.shape[0]:
            raise ValueError(
                "n_samples cannot exceed q_prop pool size when sampling without replacement."
            )
        indices = rng.permutation(q_prop_x.shape[0])[:n_samples]

    sampled_x = q_prop_x[indices].copy()
    sampled_y = q_prop_y[indices].copy()

    if return_indices:
        if q_prop_source_index is None:
            source_index = np.asarray(indices, dtype=int).copy()
        else:
            source_index = q_prop_source_index[indices].copy()
        return sampled_x, sampled_y, source_index

    return sampled_x, sampled_y


def _add_intercept(x):
    """
    Add an intercept column of ones to the feature matrix.
    """
    x = np.asarray(x, dtype=float)
    ones = np.ones((x.shape[0], 1), dtype=float)
    return np.concatenate([ones, x], axis=1)


def _weighted_linear_regression_fit(x, y, sample_weight):
    """
    Fit exact weighted least squares with an intercept term.
    """
    x, y = _validate_feature_target_shapes(x, y, "x", "y")
    x_with_intercept = _add_intercept(x)

    sample_weight = np.asarray(sample_weight, dtype=float)

    if sample_weight.ndim != 1:
        raise ValueError("sample_weight must be a 1D array.")
    if sample_weight.shape[0] != x.shape[0]:
        raise ValueError("sample_weight length must match x rows.")
    if np.any(~np.isfinite(sample_weight)):
        raise ValueError("sample_weight contains non-finite values.")
    if np.any(sample_weight < 0.0):
        raise ValueError("sample_weight must be non-negative.")

    sqrt_weight = np.sqrt(sample_weight)
    weighted_x = x_with_intercept * sqrt_weight.reshape(-1, 1)
    weighted_y = y * sqrt_weight
    beta_full, _, _, _ = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)
    return {
        "beta_full": beta_full,
    }


def _ordinary_least_squares_fit(x, y):
    """
    Fit standard OLS with an intercept term.

    Citation: `Practical_Implement.tex`, "What Needs to Be Estimated and How",
    item "Estimate the target parameter theta^*" with the plug-in estimator
    `theta0_hat`.
    """
    x, y = _validate_feature_target_shapes(x, y, "x", "y")
    x_with_intercept = _add_intercept(x)
    beta_full, _, _, _ = np.linalg.lstsq(x_with_intercept, y, rcond=None)
    return {
        "beta_full": beta_full,
    }


def _predict_linear_regression(model_state, x):
    """
    Predict y values using the fitted linear regression model.
    """
    x = np.asarray(x, dtype=float)
    return _add_intercept(x) @ model_state["beta_full"]


def estimate_parameter_covariance_trace(model_state, x, y, sample_weight, config):
    """
    Estimate trace(Cov_hat(beta_hat)) for one OLS/WLS fit.
    """
    x, y = _validate_feature_target_shapes(x, y, "x", "y")
    x_with_intercept = _add_intercept(x)
    beta_full = np.asarray(model_state["beta_full"], dtype=float)

    residual = y - x_with_intercept @ beta_full
    n_samples = x_with_intercept.shape[0]
    param_dim = x_with_intercept.shape[1]
    if n_samples <= param_dim:
        return float("nan")

    if sample_weight is None:
        xtwx = x_with_intercept.T @ x_with_intercept
        sigma_sq_hat = float(np.sum(residual ** 2) / (n_samples - param_dim))
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must be a 1D array.")
        if sample_weight.shape[0] != x.shape[0]:
            raise ValueError("sample_weight length must match x rows.")
        if np.any(~np.isfinite(sample_weight)):
            raise ValueError("sample_weight contains non-finite values.")
        if np.any(sample_weight < 0.0):
            raise ValueError("sample_weight must be non-negative.")

        xtwx = x_with_intercept.T @ (sample_weight.reshape(-1, 1) * x_with_intercept)
        sigma_sq_hat = float(
            np.sum(sample_weight * (residual ** 2)) / (n_samples - param_dim)
        )

    covariance = sigma_sq_hat * np.linalg.pinv(xtwx)
    return float(np.trace(covariance))


def load_real_data(config):
    """
    Load normalized real train/test arrays from the SHARPs downstream task.

    Citation: `Practical_Implement.tex`, Algorithm input
    `D_real = {(X_i, Y_i)}_{i=1}^n`.
    """
    artifact_bundle = _load_artifact_bundle(config)
    data = {
        "train_x": artifact_bundle["train_x"],
        "train_y": artifact_bundle["train_y"],
        "test_x": artifact_bundle["test_x"],
        "test_y": artifact_bundle["test_y"],
    }
    if "ratio_train_x" in artifact_bundle:
        data["ratio_train_x"] = artifact_bundle["ratio_train_x"]
        data["ratio_train_y"] = artifact_bundle["ratio_train_y"]
    return data


def fit_reference_model(train_x, train_y, config):
    """
    Fit the plug-in reference model `theta0_hat`.

    Citation: `Practical_Implement.tex`, "What Needs to Be Estimated and How",
    item "Estimate the target parameter theta^*" with the plug-in estimator
    `theta0_hat`.
    """
    return _ordinary_least_squares_fit(train_x, train_y)


def compute_psi(x, y, reference_model, config):
    """
    Compute the plug-in score `psi_hat(x, y)`.

    Citation: `Practical_Implement.tex`, "What Needs to Be Estimated and How",
    item defining `psi_hat(x, y) := nabla_theta L(x, y, theta0_hat)`.
    """
    x, y = _validate_feature_target_shapes(x, y, "x", "y")
    x_with_intercept = _add_intercept(x)
    residual = x_with_intercept @ reference_model["beta_full"] - y
    return 2.0 * residual.reshape(-1, 1) * x_with_intercept


def fit_generator(train_x, train_y, config):
    """
    Prepare an empirical proposal sampler for `q_prop(x, y)`.

    Citation: `Practical_Implement.tex`, Algorithm section and
    "Key Theoretical Quantities" for `q_prop(x, y) = tilde_q(y) q_phi(x | y)`.
    The external SHARPs workflow already fit `q_phi` and emitted Monte Carlo
    draws from `q_prop`, so this function loads that joint artifact pool.
    """
    train_x, train_y = _validate_feature_target_shapes(
        train_x,
        train_y,
        "train_x",
        "train_y",
    )

    artifact_bundle = _load_artifact_bundle(config)
    full_q_prop_x = artifact_bundle["q_prop_x"]
    full_q_prop_y = artifact_bundle["q_prop_y"]
    normalized_condition_name = artifact_bundle["normalized_condition_name"]
    condition_stats = artifact_bundle["metadata"]["normalization_stats"][
        normalized_condition_name
    ]

    if full_q_prop_x.shape[1] != train_x.shape[1]:
        raise ValueError("Real-data and q_prop feature dimensions must match.")

    q_prop_subsample_fraction = _resolve_q_prop_subsample_fraction(config)
    subsample_rng = _make_ratio_fit_rng(config, offset=3)
    q_prop_source_index = _deterministic_q_prop_subsample_indices(
        pool_size=full_q_prop_x.shape[0],
        fraction=q_prop_subsample_fraction,
        rng=subsample_rng,
    )
    if q_prop_source_index.shape[0] == full_q_prop_x.shape[0]:
        q_prop_x = full_q_prop_x
        q_prop_y = full_q_prop_y
    else:
        q_prop_x = full_q_prop_x[q_prop_source_index].copy()
        q_prop_y = full_q_prop_y[q_prop_source_index].copy()

    return {
        "q_prop_x": q_prop_x,
        "q_prop_y": q_prop_y,
        "q_prop_source_index": q_prop_source_index,
        "q_prop_pool_size": int(full_q_prop_x.shape[0]),
        "q_prop_active_pool_size": int(q_prop_x.shape[0]),
        "q_prop_subsample_fraction": q_prop_subsample_fraction,
        "x_dim": int(q_prop_x.shape[1]),
        "q_prop_source_path": artifact_bundle["q_prop_joint_csv_path"],
        "q_prop_feature_names": artifact_bundle["normalized_feature_names"],
        "q_prop_condition_name": normalized_condition_name,
        "condition_normalization_mode": artifact_bundle.get(
            "condition_normalization_mode",
            config.get("artifacts", {}).get("condition_normalization_mode", "unknown"),
        ),
        "condition_log_mean": float(condition_stats["mean"]),
        "condition_log_std": float(condition_stats["std"]),
        "condition_inverse_transform": artifact_bundle["condition_inverse_transform"],
        "condition_plot_label": artifact_bundle["condition_plot_label"],
    }


def generate_candidates(generator_state, n_candidates, rng, config):
    """
    Draw candidate proposal samples from the empirical `q_prop(x, y)` pool.

    Citation: `Practical_Implement.tex`, Algorithm section and
    "Key Theoretical Quantities" for the proposal law `q_prop(x, y)`.
    """
    return _sample_from_q_prop_pool(
        generator_state,
        n_candidates,
        rng,
        replace=False,
        return_indices=True,
    )


def fit_ratio_model(train_x, train_y, generator_state, rng, config):
    """
    Estimate `r_hat(x, y) = p(x, y) / q_prop(x, y)` by source discrimination.

    Citation: `Practical_Implement.tex`, "What Needs to Be Estimated and How",
    item "Estimate the density-ratio term" and the definition
    `r(x, y) := p(x, y) / q_prop(x, y)`.
    """
    train_x, train_y = _validate_feature_target_shapes(
        train_x,
        train_y,
        "train_x",
        "train_y",
    )

    ratio_config = _resolve_ratio_config(config)
    add_log_y = bool(ratio_config["add_log_y"])
    split_fractions = ratio_config["split_fractions"]
    ratio_cache_key = _build_ratio_cache_key(config)
    if ratio_cache_key in _RATIO_STATE_CACHE:
        return _RATIO_STATE_CACHE[ratio_cache_key]

    real_split_rng = _make_ratio_fit_rng(config, offset=1)
    q_prop_rng = _make_ratio_fit_rng(config, offset=2)

    real_splits = split_source_arrays(
        train_x,
        train_y,
        split_fractions=split_fractions,
        rng=real_split_rng,
    )

    prepared_splits = {}
    for split_name in SPLIT_NAMES:
        n_real_split = real_splits[split_name]["X"].shape[0]
        n_q_prop_split = _compute_n_q_prop_samples(n_real_split, config)
        q_prop_x, q_prop_y = _sample_from_q_prop_pool(
            generator_state,
            n_q_prop_split,
            q_prop_rng,
        )

        prepared_splits[split_name] = prepare_split(
            SourceSplit(
                X_real=real_splits[split_name]["X"],
                Y_real=real_splits[split_name]["Y"],
                X_syn=q_prop_x,
                Y_syn=q_prop_y,
                omega_syn=np.ones(n_q_prop_split, dtype=float),
            ),
            add_log_y=add_log_y,
        )

    model = train_lgbm_dre(
        train_split=prepared_splits["train"],
        valid_split=prepared_splits["valid"],
        params=ratio_config["lgbm_params"],
    )

    calibration_mode = ratio_config["calibration"]
    if calibration_mode == "platt":
        calibrator = fit_platt_calibrator(
            model=model,
            cal_split=prepared_splits["calibration"],
        )
    elif calibration_mode == "none":
        calibrator = None
    else:
        raise ValueError(
            "config['ratio']['calibration'] must be 'platt' or 'none'."
        )

    ratio_state = {
        "model": model,
        "calibrator": calibrator,
        "add_log_y": add_log_y,
        "clip_log_ratio": float(ratio_config["clip_log_ratio"]),
        "clip_prob": float(ratio_config["clip_prob"]),
        "q_prop_description": "empirical q_prop(x, y) pool from artifact synthetic CSV",
        "q_prop_source_path": generator_state["q_prop_source_path"],
        "diagnostics": evaluate_dre_diagnostics(
            model=model,
            test_split=prepared_splits["test"],
            calibrator=calibrator,
        ),
        "split_sizes": {
            split_name: {
                "n_real": int(prepared_splits[split_name].Z_real.shape[0]),
                "n_q_prop": int(prepared_splits[split_name].Z_syn.shape[0]),
            }
            for split_name in SPLIT_NAMES
        },
    }
    _RATIO_STATE_CACHE[ratio_cache_key] = ratio_state
    return ratio_state


def compute_log_ratio(x, y, ratio_state, config):
    """
    Compute `log r_hat(x, y)` for `r_hat(x, y) = p(x, y) / q_prop(x, y)`.

    Citation: `Practical_Implement.tex`, "What Needs to Be Estimated and How",
    item "Estimate the density-ratio term".
    """
    x, y = _validate_feature_target_shapes(x, y, "x", "y")
    add_log_y = bool(ratio_state["add_log_y"])
    calibrator = ratio_state["calibrator"]

    if calibrator is None:
        log_ratio = predict_log_ratio(
            model=ratio_state["model"],
            X=x,
            Y=y,
            add_log_y=add_log_y,
        )
    else:
        ratio_hat = predict_ratio(
            model=ratio_state["model"],
            calibrator=calibrator,
            X=x,
            Y=y,
            add_log_y=add_log_y,
            clip_prob=ratio_state["clip_prob"],
        )
        log_ratio = np.log(ratio_hat)

    clip_log_ratio = float(ratio_state["clip_log_ratio"])
    return np.clip(np.asarray(log_ratio, dtype=float), -clip_log_ratio, clip_log_ratio)


def fit_final_model(x, y, sample_weight, config):
    """
    Fit the final weighted regression model for `theta_hat`.
    """
    if sample_weight is None:
        return _ordinary_least_squares_fit(x, y)
    return _weighted_linear_regression_fit(
        x=x,
        y=y,
        sample_weight=sample_weight,
    )


def _resolve_m_class_tail_threshold(config):
    """
    Convert the physical M1.0 flux boundary to normalized target space.

    Temporal regression uses ``(log10(flux) - mean) / scale`` as its target,
    so the tail mask must apply the identical transform to ``1e-5 W/m^2``.
    A missing threshold disables this chronological-only diagnostic.
    """
    artifact_config = config.get("artifacts", {})
    raw_threshold = artifact_config.get(
        M_CLASS_TAIL_THRESHOLD_CONFIG_KEY
    )
    if raw_threshold is None:
        return None

    raw_threshold = float(raw_threshold)
    if not np.isfinite(raw_threshold) or raw_threshold <= 0.0:
        raise ValueError(
            "M-class tail flux threshold must be finite and positive."
        )

    artifact_bundle = _load_artifact_bundle(config)
    if artifact_bundle["condition_normalization_mode"] != "log10":
        raise ValueError(
            "M-class tail MSE requires standardized log10(flux) targets."
        )
    condition_name = str(artifact_bundle["normalized_condition_name"])
    condition_stats = artifact_bundle["metadata"][
        "normalization_stats"
    ][condition_name]
    condition_mean = float(condition_stats["mean"])
    condition_scale = float(condition_stats["std"])
    if not np.isfinite(condition_scale) or condition_scale <= 0.0:
        raise ValueError(
            "Temporal condition standardization scale must be positive."
        )

    log10_threshold = float(np.log10(raw_threshold))
    standardized_threshold = (
        log10_threshold - condition_mean
    ) / condition_scale
    return {
        "raw": raw_threshold,
        "log10": log10_threshold,
        "standardized": float(standardized_threshold),
    }


def evaluate_model(model_state, test_x, test_y, config):
    """
    Evaluate overall MSE and, for chronological grids, M-class tail MSE.
    """
    test_y = np.asarray(test_y, dtype=float)
    prediction = _predict_linear_regression(model_state, test_x)
    mse = float(np.mean((prediction - test_y) ** 2))
    metrics = {
        "primary_metric_name": "mse",
        "primary_metric_value": mse,
        "mse": mse,
    }
    tail_threshold = _resolve_m_class_tail_threshold(config)
    if tail_threshold is None:
        return metrics

    tail_mask = test_y >= tail_threshold["standardized"]
    tail_test_count = int(np.count_nonzero(tail_mask))
    if tail_test_count == 0:
        tail_mse = float("nan")
    else:
        tail_mse = float(
            np.mean((prediction[tail_mask] - test_y[tail_mask]) ** 2)
        )
    metrics.update(
        {
            M_CLASS_TAIL_MSE_METRIC_NAME: tail_mse,
            "m_class_tail_test_count": tail_test_count,
            "m_class_tail_test_fraction": float(
                tail_test_count / test_y.shape[0]
            ),
            "m_class_tail_flux_threshold_raw": tail_threshold["raw"],
            "m_class_tail_flux_threshold_log10": tail_threshold["log10"],
            "m_class_tail_flux_threshold_standardized": (
                tail_threshold["standardized"]
            ),
            "m_class_tail_threshold_inclusive": True,
        }
    )
    return metrics
