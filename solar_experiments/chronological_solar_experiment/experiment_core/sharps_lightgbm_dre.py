from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


SPLIT_NAMES = ("train", "valid", "calibration", "test")


DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 5000,
    "min_child_weight": 1e-2,
    "min_split_gain": 0.05,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 20.0,
    "max_bin": 511,
    "min_data_in_bin": 5,
    "subsample_for_bin": 500_000,
    "use_missing": True,
    "zero_as_missing": False,
    "random_state": 123,
    "n_jobs": 16,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


@dataclass(frozen=True)
class SourceSplit:
    X_real: np.ndarray
    Y_real: np.ndarray
    X_syn: np.ndarray
    Y_syn: np.ndarray
    omega_syn: np.ndarray


@dataclass(frozen=True)
class PreparedSplit:
    Z: np.ndarray
    source_label: np.ndarray
    sample_weight: np.ndarray
    Z_real: np.ndarray
    Z_syn: np.ndarray
    omega_syn: np.ndarray
    feature_names: tuple[str, ...]


class PlattCalibrator:
    """
    Binary Platt scaling on raw LightGBM scores.
    """

    def __init__(self, c_value: float = 10.0, max_iter: int = 1000):
        self.model = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=max_iter,
        )

    def fit(self, raw_score, source_label, sample_weight):
        raw_score = np.asarray(raw_score, dtype=np.float64).reshape(-1, 1)
        source_label = np.asarray(source_label, dtype=np.int8)
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        self.model.fit(raw_score, source_label, sample_weight=sample_weight)
        return self

    def predict_proba_real(self, raw_score):
        raw_score = np.asarray(raw_score, dtype=np.float64).reshape(-1, 1)
        return self.model.predict_proba(raw_score)[:, 1]


def _import_lightgbm():
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lightgbm is required for SHARPs density-ratio estimation."
        ) from exc

    return lgb


def _validate_feature_target_shapes(x, y, x_name, y_name):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"{x_name} must be a 2D array.")
    if y.ndim != 1:
        raise ValueError(f"{y_name} must be a 1D array.")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x_name} and {y_name} must have the same number of rows.")

    return x, y


def _validate_omega_syn(omega_syn, expected_length):
    omega_syn = np.asarray(omega_syn, dtype=np.float64)

    if omega_syn.ndim != 1:
        raise ValueError("omega_syn must be a 1D array.")
    if omega_syn.shape[0] != expected_length:
        raise ValueError("omega_syn length must match the number of synthetic rows.")
    if np.any(~np.isfinite(omega_syn)):
        raise ValueError("omega_syn contains non-finite values.")
    if np.any(omega_syn < 0.0):
        raise ValueError("omega_syn must be non-negative.")
    if float(np.sum(omega_syn)) <= 0.0:
        raise ValueError("omega_syn must have a strictly positive sum.")

    return omega_syn


def _validate_split_fractions(split_fractions: Mapping[str, float]):
    if not isinstance(split_fractions, Mapping):
        raise TypeError("split_fractions must be a mapping.")

    missing = [name for name in SPLIT_NAMES if name not in split_fractions]
    if missing:
        raise ValueError(f"split_fractions is missing keys: {missing}")

    fractions = np.asarray(
        [float(split_fractions[name]) for name in SPLIT_NAMES],
        dtype=np.float64,
    )

    if np.any(~np.isfinite(fractions)):
        raise ValueError("split_fractions must be finite.")
    if np.any(fractions <= 0.0):
        raise ValueError("split_fractions entries must be strictly positive.")
    if not np.isclose(np.sum(fractions), 1.0, atol=1e-8):
        raise ValueError("split_fractions must sum to 1.")

    return fractions


def _compute_split_counts(n_samples: int, split_fractions: Mapping[str, float]):
    fractions = _validate_split_fractions(split_fractions)

    if n_samples < len(SPLIT_NAMES):
        raise ValueError(
            "Need at least one sample per split for train/valid/calibration/test."
        )

    raw_counts = fractions * n_samples
    base_counts = np.floor(raw_counts).astype(int)
    remainder = int(n_samples - np.sum(base_counts))

    fractional_order = np.argsort(-(raw_counts - base_counts))
    for index in fractional_order[:remainder]:
        base_counts[index] += 1

    if np.any(base_counts <= 0):
        raise ValueError(
            "Not enough samples to allocate at least one row to each DRE split."
        )

    return {
        split_name: int(count)
        for split_name, count in zip(SPLIT_NAMES, base_counts.tolist())
    }


def split_source_arrays(x, y, split_fractions, rng, omega=None):
    """
    Split one source into train/valid/calibration/test partitions.
    """
    x, y = _validate_feature_target_shapes(x, y, "x", "y")

    if omega is not None:
        omega = _validate_omega_syn(omega, x.shape[0])

    counts = _compute_split_counts(x.shape[0], split_fractions)
    indices = rng.permutation(x.shape[0])

    split_dict = {}
    start = 0
    for split_name in SPLIT_NAMES:
        count = counts[split_name]
        split_indices = indices[start:start + count]
        split_value = {
            "X": x[split_indices],
            "Y": y[split_indices],
        }
        if omega is not None:
            split_value["omega"] = omega[split_indices]
        split_dict[split_name] = split_value
        start += count

    return split_dict


def build_feature_matrix(X, Y, add_log_y=True):
    """
    Build Z = (X, Y[, log1p(Y)]).
    """
    X, Y = _validate_feature_target_shapes(X, Y, "X", "Y")
    y_column = Y.reshape(-1, 1)

    features = [X, y_column]
    if add_log_y:
        if np.any(Y < 0.0):
            raise ValueError(
                "add_log_y=True requires non-negative Y values."
            )
        features.append(np.log1p(y_column))

    return np.hstack(features)


def build_feature_names(x_dim, add_log_y=True):
    """
    Build deterministic feature names for the DRE design matrix.
    """
    x_dim = int(x_dim)
    if x_dim <= 0:
        raise ValueError("x_dim must be positive.")

    feature_names = [f"x_{idx}" for idx in range(x_dim)]
    feature_names.append("y")
    if add_log_y:
        feature_names.append("log1p_y")

    return tuple(feature_names)


def build_feature_frame(X, Y, add_log_y=True):
    """
    Build a named feature frame for LightGBM fit/predict calls.
    """
    X, Y = _validate_feature_target_shapes(X, Y, "X", "Y")
    Z = build_feature_matrix(X, Y, add_log_y=add_log_y)
    feature_names = build_feature_names(X.shape[1], add_log_y=add_log_y)
    return pd.DataFrame(Z, columns=list(feature_names))


def matrix_to_feature_frame(Z, feature_names):
    """
    Convert an existing design matrix to a named pandas frame.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array.")

    feature_names = tuple(feature_names)
    if Z.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match Z columns.")

    return pd.DataFrame(Z, columns=list(feature_names))


def make_balanced_source_weights(n_real, omega_syn):
    """
    Construct classifier training weights with equal total source weight.
    """
    n_real = int(n_real)
    if n_real <= 0:
        raise ValueError("n_real must be positive.")

    omega_syn = np.asarray(omega_syn, dtype=np.float64)
    if omega_syn.ndim != 1:
        raise ValueError("omega_syn must be a 1D array.")
    omega_syn = _validate_omega_syn(omega_syn, omega_syn.shape[0])
    w_real = np.ones(n_real, dtype=np.float64)
    w_syn = omega_syn * (n_real / np.sum(omega_syn))
    return w_real, w_syn


def stack_real_and_synthetic(Z_real, Z_syn, w_real, w_syn):
    """
    Stack source arrays into one weighted classification dataset.
    """
    Z_real = np.asarray(Z_real, dtype=np.float64)
    Z_syn = np.asarray(Z_syn, dtype=np.float64)
    w_real = np.asarray(w_real, dtype=np.float64)
    w_syn = np.asarray(w_syn, dtype=np.float64)

    if Z_real.ndim != 2 or Z_syn.ndim != 2:
        raise ValueError("Z_real and Z_syn must be 2D arrays.")
    if Z_real.shape[1] != Z_syn.shape[1]:
        raise ValueError("Z_real and Z_syn must have the same number of columns.")
    if w_real.ndim != 1 or w_syn.ndim != 1:
        raise ValueError("w_real and w_syn must be 1D arrays.")
    if w_real.shape[0] != Z_real.shape[0]:
        raise ValueError("w_real length must match Z_real rows.")
    if w_syn.shape[0] != Z_syn.shape[0]:
        raise ValueError("w_syn length must match Z_syn rows.")

    Z = np.vstack([Z_real, Z_syn])
    source_label = np.concatenate([
        np.ones(Z_real.shape[0], dtype=np.int8),
        np.zeros(Z_syn.shape[0], dtype=np.int8),
    ])
    sample_weight = np.concatenate([w_real, w_syn])
    return Z, source_label, sample_weight


def prepare_split(split: SourceSplit, add_log_y=True):
    """
    Build a PreparedSplit from raw real/synthetic arrays.
    """
    X_real, Y_real = _validate_feature_target_shapes(
        split.X_real,
        split.Y_real,
        "split.X_real",
        "split.Y_real",
    )
    X_syn, Y_syn = _validate_feature_target_shapes(
        split.X_syn,
        split.Y_syn,
        "split.X_syn",
        "split.Y_syn",
    )

    if X_real.shape[1] != X_syn.shape[1]:
        raise ValueError("Real and synthetic feature dimensions must match.")

    omega_syn = _validate_omega_syn(split.omega_syn, X_syn.shape[0])
    Z_real = build_feature_matrix(X_real, Y_real, add_log_y=add_log_y)
    Z_syn = build_feature_matrix(X_syn, Y_syn, add_log_y=add_log_y)
    feature_names = build_feature_names(X_real.shape[1], add_log_y=add_log_y)
    w_real, w_syn = make_balanced_source_weights(Z_real.shape[0], omega_syn)
    Z, source_label, sample_weight = stack_real_and_synthetic(
        Z_real,
        Z_syn,
        w_real,
        w_syn,
    )

    return PreparedSplit(
        Z=Z,
        source_label=source_label,
        sample_weight=sample_weight,
        Z_real=Z_real,
        Z_syn=Z_syn,
        omega_syn=omega_syn,
        feature_names=feature_names,
    )


def train_lgbm_dre(train_split, valid_split, params=None):
    """
    Train a LightGBM source classifier whose raw score targets log(p / q).
    """
    lgb = _import_lightgbm()

    config = dict(DEFAULT_LGBM_PARAMS)
    if params is not None:
        config.update(params)

    early_stopping_rounds = int(config.pop("early_stopping_rounds", 200))
    log_evaluation_period = int(config.pop("log_evaluation_period", 0))

    model = lgb.LGBMClassifier(**config)
    callbacks = [
        lgb.early_stopping(
            stopping_rounds=early_stopping_rounds,
            first_metric_only=True,
            verbose=bool(log_evaluation_period),
        )
    ]
    if log_evaluation_period > 0:
        callbacks.append(lgb.log_evaluation(period=log_evaluation_period))

    train_Z = matrix_to_feature_frame(train_split.Z, train_split.feature_names)
    valid_Z = matrix_to_feature_frame(valid_split.Z, valid_split.feature_names)

    model.fit(
        train_Z,
        train_split.source_label,
        sample_weight=train_split.sample_weight,
        eval_set=[(valid_Z, valid_split.source_label)],
        eval_sample_weight=[valid_split.sample_weight],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )
    return model


def fit_platt_calibrator(model, cal_split):
    """
    Fit a held-out Platt calibrator on raw LightGBM scores.
    """
    raw_score = np.asarray(
        model.predict(
            matrix_to_feature_frame(cal_split.Z, cal_split.feature_names),
            raw_score=True,
        ),
        dtype=np.float64,
    ).ravel()
    calibrator = PlattCalibrator()
    calibrator.fit(raw_score, cal_split.source_label, cal_split.sample_weight)
    return calibrator


def predict_log_ratio(model, X, Y, add_log_y=True):
    """
    Predict log(p / q) using the LightGBM raw score.
    """
    feature_frame = build_feature_frame(X, Y, add_log_y=add_log_y)
    return np.asarray(model.predict(feature_frame, raw_score=True), dtype=np.float64).ravel()


def predict_ratio(model, calibrator, X, Y, add_log_y=True, clip_prob=1e-12):
    """
    Predict p / q, optionally using held-out probability calibration.
    """
    raw_score = predict_log_ratio(model, X, Y, add_log_y=add_log_y)

    return ratio_from_raw_score(
        raw_score=raw_score,
        calibrator=calibrator,
        clip_prob=clip_prob,
    )


def ratio_from_raw_score(raw_score, calibrator, clip_prob=1e-12):
    """
    Convert raw source-discrimination scores to p / q.
    """
    raw_score = np.asarray(raw_score, dtype=np.float64).ravel()

    if calibrator is None:
        return np.exp(raw_score)

    eta = np.asarray(
        calibrator.predict_proba_real(raw_score),
        dtype=np.float64,
    ).ravel()
    eta = np.clip(eta, float(clip_prob), 1.0 - float(clip_prob))
    return eta / (1.0 - eta)


def weighted_mean(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if x.ndim != 1 or w.ndim != 1:
        raise ValueError("x and w must be 1D arrays.")
    if x.shape[0] != w.shape[0]:
        raise ValueError("x and w must have the same length.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("w must have a strictly positive sum.")

    return float(np.sum(w * x) / np.sum(w))


def weighted_feature_means(Z, w):
    Z = np.asarray(Z, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array.")
    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")
    if Z.shape[0] != w.shape[0]:
        raise ValueError("w length must match Z rows.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("w must have a strictly positive sum.")

    return np.sum(Z * w.reshape(-1, 1), axis=0) / np.sum(w)


def effective_sample_size(w):
    w = np.asarray(w, dtype=np.float64)

    if w.ndim != 1:
        raise ValueError("w must be a 1D array.")
    if float(np.sum(w)) <= 0.0:
        raise ValueError("w must have a strictly positive sum.")

    return float((np.sum(w) ** 2) / np.sum(w ** 2))


def evaluate_dre_diagnostics(model, test_split, calibrator=None):
    """
    Compute denominator-side diagnostics on a held-out synthetic split.
    """
    raw_real = np.asarray(
        model.predict(
            matrix_to_feature_frame(test_split.Z_real, test_split.feature_names),
            raw_score=True,
        ),
        dtype=np.float64,
    ).ravel()
    raw_syn = np.asarray(
        model.predict(
            matrix_to_feature_frame(test_split.Z_syn, test_split.feature_names),
            raw_score=True,
        ),
        dtype=np.float64,
    ).ravel()

    if calibrator is None:
        ratio_real = np.exp(raw_real)
        ratio_syn = np.exp(raw_syn)
    else:
        ratio_real = ratio_from_raw_score(
            raw_score=raw_real,
            calibrator=calibrator,
        )
        ratio_syn = ratio_from_raw_score(
            raw_score=raw_syn,
            calibrator=calibrator,
        )

    omega = np.asarray(test_split.omega_syn, dtype=np.float64)
    q_weight = omega * ratio_syn

    return {
        "q_mean_r": weighted_mean(ratio_syn, omega),
        "real_mean_Z": np.mean(test_split.Z_real, axis=0),
        "q_reweighted_mean_Z": weighted_feature_means(test_split.Z_syn, q_weight),
        "mean_diff_Z": weighted_feature_means(test_split.Z_syn, q_weight)
        - np.mean(test_split.Z_real, axis=0),
        "ess_syn_proposal": effective_sample_size(omega),
        "ess_syn_reweighted": effective_sample_size(q_weight),
        "ratio_summary_syn": {
            "min": float(np.min(ratio_syn)),
            "p01": float(np.quantile(ratio_syn, 0.01)),
            "p50": float(np.quantile(ratio_syn, 0.50)),
            "p99": float(np.quantile(ratio_syn, 0.99)),
            "max": float(np.max(ratio_syn)),
        },
        "ratio_summary_real": {
            "min": float(np.min(ratio_real)),
            "p01": float(np.quantile(ratio_real, 0.01)),
            "p50": float(np.quantile(ratio_real, 0.50)),
            "p99": float(np.quantile(ratio_real, 0.99)),
            "max": float(np.max(ratio_real)),
        },
    }


__all__ = [
    "DEFAULT_LGBM_PARAMS",
    "PreparedSplit",
    "PlattCalibrator",
    "SPLIT_NAMES",
    "SourceSplit",
    "build_feature_frame",
    "build_feature_matrix",
    "build_feature_names",
    "effective_sample_size",
    "evaluate_dre_diagnostics",
    "fit_platt_calibrator",
    "matrix_to_feature_frame",
    "make_balanced_source_weights",
    "predict_log_ratio",
    "predict_ratio",
    "prepare_split",
    "ratio_from_raw_score",
    "split_source_arrays",
    "stack_real_and_synthetic",
    "train_lgbm_dre",
    "weighted_feature_means",
    "weighted_mean",
]
